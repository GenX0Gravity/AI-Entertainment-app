import io
import os
import re
import tempfile
import time
from threading import Thread

import numpy as np
import soundfile as sf
import streamlit as st
import torch

try:
    from huggingface_hub import login as _hf_login
except Exception:
    _hf_login = None

_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if _HF_TOKEN and _hf_login:
    try:
        _hf_login(token=_HF_TOKEN)
    except Exception:
        # Token is optional; app still works without it.
        pass

# Stereo / melody checkpoints are large; flaky networks often need longer timeouts + retries.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")


def _is_transient_download_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection",
            "timeout",
            "timed out",
            "reset",
            "broken pipe",
            "ssl",
            "eof",
            "network",
            "unreachable",
            "refused",
            "429",
            "502",
            "503",
            "504",
            "incomplete",
            "chunk",
            "contentlength",
            "remote end closed",
            "temporary failure",
        )
    )


def _hf_download_retry(fn, *, attempts: int = 5, base_delay: float = 2.0):
    """Retry Hugging Face downloads on typical connection / timeout errors."""
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if i == attempts - 1 or not _is_transient_download_error(e):
                raise
            time.sleep(base_delay * (2**i))
    raise last  # pragma: no cover

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

from transformers import (
    MusicgenForConditionalGeneration,
    MusicgenProcessor,
    pipeline,
)

try:
    from transformers import MusicgenStreamer
    _STREAMER_AVAILABLE = True
except ImportError:
    _STREAMER_AVAILABLE = False

try:
    from streamlit_mic_recorder import mic_recorder
    _MIC_RECORDER_AVAILABLE = True
except ImportError:
    _MIC_RECORDER_AVAILABLE = False

# ── CPU thread tuning ──────────────────────────────────────────────────────
_cpu_cores = os.cpu_count() or 2
try:
    torch.set_num_threads(_cpu_cores)
    torch.set_num_interop_threads(max(1, _cpu_cores // 2))
except RuntimeError:
    pass


# Tokens the audio codec produces per second of audio (MusicGen constant)
_TOKENS_PER_SEC = 50

# ── Model registry ─────────────────────────────────────────────────────────
MODEL_MAP = {
    "small  · fast, text-only":             "facebook/musicgen-small",
    "medium · richer, text-only":           "facebook/musicgen-medium",
    "melody · remix + mono":                "facebook/musicgen-melody",
    "stereo-melody · remix + stereo ★":     "facebook/musicgen-stereo-melody",
}

_TEXT_ONLY_MODELS = {
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
}

THEME_STYLES = {
    "Neon Night": {
        "bg": "#070A12",
        "panel": "#101726",
        "text": "#EAF6FF",
        "muted": "#8CA2C7",
        "accent": "#00E5FF",
    },
    "Aurora Pop": {
        "bg": "#071217",
        "panel": "#0F2A3A",
        "text": "#E7FBFF",
        "muted": "#88B7C7",
        "accent": "#28FF7A",
    },
    "Bollywood Gold": {
        "bg": "#1B1206",
        "panel": "#2B1A0B",
        "text": "#FFF1D6",
        "muted": "#D1B18A",
        "accent": "#FFC400",
    },
    "Cyber Sunset": {
        "bg": "#0E0715",
        "panel": "#1F0D2D",
        "text": "#F7ECFF",
        "muted": "#BCA4D6",
        "accent": "#FF4FD8",
    },
}

# Voice intent -> preset key (so "Avengers theme" directly selects the right BGM prompt)
VOICE_INTENT_PRESET_MAP = {
    # Avengers / superhero
    "avengers": "Avengers Theme BGM",
    "marvel": "Marvel Superhero BGM",
    "superhero": "Marvel Superhero BGM",
    "batman": "Batman Vigilante BGM",
    "interstellar": "Interstellar Space Drama BGM",
    "inception": "Inception Mind-Bending BGM",
    "star wars": "Star Wars Space Opera BGM",
    "harry potter": "Harry Potter Magical Fantasy BGM",
    "bond": "James Bond Spy Thriller BGM",
    "james bond": "James Bond Spy Thriller BGM",

    # Bollywood
    "bollywood": "Bollywood Drama BGM",
    "dance": "Bollywood Dance BGM",
    "sad": "Bollywood Sad Scene BGM",
    "monsoon": "Monsoon Bollywood BGM",
    "folk": "Bollywood Folk Fusion BGM",

    # Hollywood / movie
    "action": "Hollywood Action BGM",
    "thriller": "Hollywood Thriller BGM",
    "romance": "Hollywood Romance BGM",
    "sci-fi": "Hollywood Sci-Fi BGM",
    "sci fi": "Hollywood Sci-Fi BGM",
    "scifi": "Hollywood Sci-Fi BGM",
    "horror": "Hollywood Horror BGM",
    "adventure": "Hollywood Adventure BGM",

    # Weather
    "rain": "Rainy Weather BGM",
    "thunder": "Thunderstorm BGM",
    "sunny": "Sunny Morning BGM",
    "snow": "Winter Snow BGM",
    "fog": "Foggy Mystery BGM",

    # Structure hints
    "trailer": "Trailer Build-up BGM",
    "chase": "Chase Scene BGM",
    "credits": "End Credits BGM",
}

FRANCHISE_STYLE_MAP = {
    "avengers": "heroic cinematic action score, powerful brass, driving strings, hybrid percussion, triumphant energy",
    "marvel": "superhero cinematic score, bold brass motifs, tense strings, modern hybrid trailer percussion",
    "batman": "dark vigilante cinematic score, brooding low strings, deep drums, tense pulses",
    "interstellar": "space drama cinematic ambience, emotional synth-organ textures, wide atmospheric layers",
    "inception": "mind-bending cinematic suspense, pulsing low-end, ticking rhythms, dramatic rises",
    "star wars": "space opera orchestral adventure, heroic brass, sweeping strings, epic momentum",
    "harry potter": "magical orchestral fantasy, celesta sparkle, warm strings, wondrous atmosphere",
    "james bond": "spy thriller orchestral groove, confident brass stabs, tense rhythm, sleek cinematic style",
}


@st.cache_resource
def load_model(
    model_name: str,
    cpu_compile_text_only: bool = True,
    cpu_quantize_text_only: bool = True,
):
    """
    Loads and optimises the selected MusicGen variant.

    CPU optimisations for text-only models:
      • INT8 dynamic quantisation  (weights only, float32 activations)
      • torch.compile reduce-overhead

    Melody/stereo models use bfloat16 on CPU (safe + ~40% faster than float32).
    INT8 QLinear kernels are only applied to text-only models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = _hf_download_retry(
        lambda: MusicgenProcessor.from_pretrained(model_name)
    )

    if device.type == "cuda":
        model = _hf_download_retry(
            lambda: MusicgenForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        ).to(device)
    else:
        if model_name in _TEXT_ONLY_MODELS:
            model = _hf_download_retry(
                lambda: MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                )
            )
            if cpu_quantize_text_only:
                # INT8 quantisation — only safe for text-only path
                model = torch.quantization.quantize_dynamic(
                    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
                )
            if cpu_compile_text_only:
                try:
                    model = torch.compile(
                        model, mode="reduce-overhead", fullgraph=False
                    )
                except Exception:
                    pass
        else:
            # bfloat16 on CPU: safe for melody/stereo, ~40% less memory bandwidth
            model = _hf_download_retry(
                lambda: MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
            )

    model.eval()
    return processor, model, device


@st.cache_resource
def load_asr():
    return pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


# ── Helpers ────────────────────────────────────────────────────────────────

def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample a [channels, samples] numpy array.
    Uses torchaudio when available, otherwise falls back to numpy linear interpolation.
    """
    if orig_sr == target_sr:
        return waveform
    if _TORCHAUDIO_AVAILABLE:
        t = torch.from_numpy(waveform)
        t = torchaudio.functional.resample(t, orig_sr, target_sr)
        return t.numpy()
    # Numpy fallback: linear interpolation per channel
    n_orig = waveform.shape[-1]
    n_target = int(round(n_orig * target_sr / orig_sr))
    x_orig   = np.linspace(0, 1, n_orig)
    x_target = np.linspace(0, 1, n_target)
    if waveform.ndim == 1:
        return np.interp(x_target, x_orig, waveform).astype(waveform.dtype)
    return np.stack(
        [np.interp(x_target, x_orig, ch).astype(waveform.dtype) for ch in waveform]
    )


def postprocess(audio: np.ndarray, sample_rate: int) -> np.ndarray:

    """Peak-normalise to −1 dBFS then apply 0.5 s cosine fades."""
    # 1. Peak normalise
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.891  # 0.891 ≈ 10^(−1/20) → −1 dBFS

    # 2. Cosine fade-in / fade-out (operates on first axis = time for mono,
    #    or rows for stereo [samples × channels])
    fade_len = min(int(0.5 * sample_rate), audio.shape[0] // 4)
    t = np.linspace(0, np.pi, fade_len)
    fade_in  = ((1 - np.cos(t)) / 2).astype(audio.dtype)
    fade_out = ((1 + np.cos(t)) / 2).astype(audio.dtype)

    if audio.ndim == 1:
        audio[:fade_len]  *= fade_in
        audio[-fade_len:] *= fade_out
    else:  # stereo [samples, channels]
        audio[:fade_len]  *= fade_in[:, None]
        audio[-fade_len:] *= fade_out[:, None]

    return audio


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> io.BytesIO:
    """Encode audio as WAV into a fresh BytesIO buffer (always seeked to start)."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf


def wav_bytes_for_download(audio: np.ndarray, sample_rate: int):
    """Return (BytesIO for st.audio, raw bytes for st.download_button).

    st.download_button MUST receive plain `bytes`, not BytesIO.
    Streamlit serializes BytesIO via .read() which is position-dependent
    and unreliable — .getvalue() always returns the full buffer regardless
    of the internal read pointer.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    raw = buf.getvalue()          # full bytes — position-independent
    buf.seek(0)                   # rewind for st.audio
    return buf, raw


def _compute_generation_limits(model, duration_seconds: int) -> tuple[int, int]:
    """Return safe (max_new_tokens, model_max_pos) for MusicGen decoding."""
    model_max_pos = getattr(
        getattr(model.config, "decoder", model.config),
        "max_position_embeddings",
        2048,
    )
    max_new_tokens = min(
        int(duration_seconds * _TOKENS_PER_SEC),
        model_max_pos - 4 - 1,  # 4 = max codebook delay offset
    )
    return max_new_tokens, model_max_pos


def _to_output_tracks(audio_values, num_variations: int) -> list[np.ndarray]:
    """Convert model output tensor [batch, channels, samples] to playable numpy tracks."""
    tracks: list[np.ndarray] = []
    for i in range(num_variations):
        track = audio_values[i].cpu().float().numpy()  # [channels, samples]
        if track.ndim == 2 and track.shape[0] > 1:
            tracks.append(track.T)  # stereo -> [samples, channels]
        else:
            tracks.append(track[0] if track.ndim == 2 else track)  # mono -> [samples]
    return tracks


def _render_track_outputs(raw_tracks: list[np.ndarray], sample_rate: int, num_variations: int) -> None:
    """Post-process, preview, and expose download buttons for generated tracks."""
    for i, raw_track in enumerate(raw_tracks):
        audio = postprocess(raw_track, sample_rate)
        play_buf, dl_bytes = wav_bytes_for_download(audio, sample_rate)

        label = f"Variation {i + 1}" if num_variations > 1 else "Your track"
        st.markdown(f"**{label}**")
        st.audio(play_buf, format="audio/wav")
        st.download_button(
            label=f"📥 Download {'Variation ' + str(i+1) if num_variations > 1 else 'WAV'}",
            data=dl_bytes,
            file_name=f"ai_track{'_v' + str(i+1) if num_variations > 1 else ''}.wav",
            mime="audio/wav",
            key=f"dl_{i}",
            use_container_width=True,
        )


def _apply_theme(theme_name: str) -> None:
    """Apply a lightweight visual theme with CSS variables."""
    style = THEME_STYLES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: {style["bg"]};
            --panel-bg: {style["panel"]};
            --app-text: {style["text"]};
            --app-muted: {style["muted"]};
            --app-accent: {style["accent"]};
        }}
        .stApp {{
            background: radial-gradient(900px circle at 10% 0%, color-mix(in srgb, var(--app-accent) 35%, transparent), transparent 55%),
                        radial-gradient(900px circle at 90% 10%, color-mix(in srgb, var(--app-accent) 20%, transparent), transparent 60%),
                        var(--app-bg);
            color: var(--app-text);
        }}
        [data-testid="stSidebar"] {{
            background: var(--panel-bg);
        }}
        [data-testid="stMarkdownContainer"], .stCaption, .stAlert {{
            color: var(--app-text);
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid var(--app-accent);
            border-radius: 0.6rem;
            background: color-mix(in srgb, var(--panel-bg) 85%, transparent);
        }}
        .stButton > button, .stDownloadButton > button {{
            border: 1px solid var(--app-accent);
            border-radius: 0.55rem;
        }}
        .stSlider p, .stSelectbox label, .stTextInput label {{
            color: var(--app-muted) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_voice_prompt(
    transcript: str,
    style_preset: str,
    energy: str,
    tempo_hint: str,
    instruments: str,
) -> str:
    """Convert raw voice transcript into a richer generation prompt."""
    style_map = {
        "Natural": "",
        "Detailed": "high detail production, clean arrangement, evolving sections",
        "Cinematic": "cinematic dynamics, dramatic transitions, wide soundstage",
        "Club-ready": "tight low-end, punchy drums, DJ-friendly groove",
    }
    parts = [transcript.strip()]
    if style_map[style_preset]:
        parts.append(style_map[style_preset])
    if energy != "Any":
        parts.append(f"{energy.lower()} energy")
    if tempo_hint:
        parts.append(f"{tempo_hint} BPM feel")
    if instruments:
        parts.append(f"focus on {instruments.strip()}")
    return ", ".join([p for p in parts if p])


def _normalize_voice_text(text: str) -> str:
    text = text.strip().lower()
    # Replace non-alphanumerics with spaces; keep word boundaries usable.
    text = re.sub(r"[^a-z0-9\\s]+", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def _extract_bpm_hint(text: str) -> str | None:
    """Extract explicit BPM from speech, e.g. '120 bpm'."""
    lower = text.lower()
    match = re.search(r"(\\d{2,3})\\s*b\\s*p\\s*m", lower) or re.search(r"(\\d{2,3})\\s*bpm", lower)
    if not match:
        return None
    return match.group(1)


def _detect_energy_from_text(text: str) -> str | None:
    lower = _normalize_voice_text(text)
    if any(tok in lower for tok in ("chill", "calm", "soft", "slow", "relaxed")):
        return "Low"
    if any(tok in lower for tok in ("high", "hype", "energetic", "fast", "intense", "club")):
        return "High"
    return None


def _detect_intent_preset_key(text: str) -> str | None:
    """Detect a preset from the transcript (e.g., 'avengers theme')."""
    normalized = _normalize_voice_text(text)
    padded = f" {normalized} "

    # Prefer longer/multi-word matches first.
    for key in sorted(VOICE_INTENT_PRESET_MAP.keys(), key=len, reverse=True):
        if f" {key} " in padded:
            return VOICE_INTENT_PRESET_MAP[key]
    return None


def _extract_audio_bytes(source_obj) -> bytes | None:
    """Normalize audio payloads from uploader/mic recorder to raw bytes."""
    if source_obj is None:
        return None
    if isinstance(source_obj, (bytes, bytearray)):
        return bytes(source_obj)
    if hasattr(source_obj, "read"):
        return source_obj.read()
    if isinstance(source_obj, dict):
        # streamlit-mic-recorder can return dict payloads.
        for key in ("bytes", "audio", "data", "wav"):
            value = source_obj.get(key)
            if isinstance(value, (bytes, bytearray)):
                return bytes(value)
            if hasattr(value, "read"):
                return value.read()
    return None


def _rewrite_pop_culture_prompt(text: str) -> tuple[str, bool]:
    """Rewrite franchise/theme requests into style descriptors for better generation."""
    normalized = text.strip()
    lower_text = normalized.lower()
    matched = []
    for key, style_prompt in FRANCHISE_STYLE_MAP.items():
        if re.search(rf"\b{re.escape(key)}\b", lower_text):
            matched.append(style_prompt)
    if not matched:
        return normalized, False
    # Keep user intent while replacing direct soundtrack mimic requests.
    rewritten = f"{normalized}, inspired by cinematic superhero score aesthetics, " + ", ".join(matched)
    return rewritten, True


# ────────────────────────────────────────────────
# App UI
# ────────────────────────────────────────────────

if "ui_theme" not in st.session_state:
    st.session_state.ui_theme = "Neon Night"

with st.sidebar:
    st.subheader("UI")
    selected_theme = st.selectbox(
        "Theme",
        options=list(THEME_STYLES.keys()),
        index=list(THEME_STYLES.keys()).index(st.session_state.ui_theme),
    )
    st.session_state.ui_theme = selected_theme
    st.caption("Switch themes without restarting.")
    st.divider()
    st.subheader("Models")
    if st.button("Clear cached models", help="Use after a failed or partial download (e.g. connection reset)."):
        st.cache_resource.clear()
        st.success("Cache cleared. Reloading…")
        st.rerun()
    st.caption(
        "**Stereo-melody** is a large download from Hugging Face. "
        "Use stable Wi‑Fi, VPN off if blocked, and set `HF_TOKEN` for fewer rate limits."
    )

_apply_theme(st.session_state.ui_theme)

st.title("AI Music DJ 🎧")

# ── Model selector ─────────────────────────────────────────────────────────
st.subheader("1 · Choose your model")
default_model_index = 3 if torch.cuda.is_available() else 0
model_choice = st.radio(
    "Model",
    options=list(MODEL_MAP.keys()),
    index=default_model_index,  # GPU: stereo-melody, CPU: small
    horizontal=True,
    label_visibility="collapsed",
)
MODEL_NAME  = MODEL_MAP[model_choice]
is_melody   = "melody" in MODEL_NAME
is_stereo   = "stereo" in MODEL_NAME

if is_stereo:
    st.info(
        "**Stereo-melody** downloads extra weights from Hugging Face (can take several minutes). "
        "If you see a connection error: try again, use **Clear cached models** in the sidebar, "
        "set environment variable **`HF_TOKEN`**, or use **melody (mono)** until the download finishes."
    )

with st.expander("Model speed options", expanded=False):
    cpu_compile_text_only = st.toggle(
        "Enable torch.compile (CPU text-only)",
        value=True,
        help="Compiles the model on CPU. Faster at inference after loading, but increases initial load time.",
        disabled=torch.cuda.is_available(),
    )
    cpu_quantize_text_only = st.toggle(
        "Enable INT8 dynamic quantization (CPU text-only)",
        value=True,
        help="Reduces model size and speeds inference, with a small load-time cost.",
        disabled=torch.cuda.is_available(),
    )
    if st.button("Warm up all models", type="secondary"):
        with st.spinner("Warming up models (this may take a while)..."):
            for name in MODEL_MAP.values():
                load_model(
                    name,
                    cpu_compile_text_only=cpu_compile_text_only,
                    cpu_quantize_text_only=cpu_quantize_text_only,
                )
        st.success("Warm-up complete. Switching models should be much faster now.")

try:
    with st.spinner(f"Loading model: {model_choice} (downloading from Hugging Face if needed)…"):
        processor, model, device = load_model(
            MODEL_NAME,
            cpu_compile_text_only=cpu_compile_text_only,
            cpu_quantize_text_only=cpu_quantize_text_only,
        )
except Exception as e:
    st.error(
        f"**Could not load `{MODEL_NAME}`**\n\n"
        f"`{type(e).__name__}: {e}`\n\n"
        "**Common fixes:**\n"
        "- Check internet / firewall / VPN (try VPN off or different network).\n"
        "- Set **`HF_TOKEN`** (Hugging Face access token) — reduces rate limits.\n"
        "- Sidebar → **Clear cached models**, then select stereo-melody again.\n"
        "- Pre-download with: `huggingface-cli download facebook/musicgen-stereo-melody`\n"
    )
    st.stop()

st.caption(
    f"✅ Loaded: **{model_choice}** | "
    f"{'🎵 Melody-conditioned remix' if is_melody else '📝 Text-only'} | "
    f"{'🔊 Stereo output' if is_stereo else '🔈 Mono output'}"
)

# ── Device status banner ────────────────────────────────────────────────────
if device.type == "cpu":
    extra = " ⏳ Melody/stereo models skip quantisation — expect **60–120 s** per clip." if is_melody else ""
    st.warning(
        "⚠️ **Running on CPU** — "
        + ("INT8 quantisation + torch.compile active." if MODEL_NAME in _TEXT_ONLY_MODELS else "bfloat16 weights (melody/stereo models).")
        + f" Expect **~15–40 s** for 10 s (small).{extra}"
    )
else:
    st.success(f"✅ Running on GPU: `{torch.cuda.get_device_name(0)}`")

st.markdown(
    "Describe a mood/genre (e.g. **'Upbeat EDM for workout'**), "
    "optionally upload a melody seed, then generate."
)
st.caption("Powered by Meta MusicGen · Wav2Vec2 ASR")

# ── Prompt + Presets ────────────────────────────────────────────────────────
st.subheader("2 · Prompt")
if "prompt" not in st.session_state:
    st.session_state.prompt = "Upbeat EDM for workout"
if "pending_main_prompt" not in st.session_state:
    st.session_state.pending_main_prompt = None

# Apply any queued prompt update before creating the widget with key="main_prompt".
if st.session_state.pending_main_prompt is not None:
    st.session_state.main_prompt = st.session_state.pending_main_prompt
    st.session_state.pending_main_prompt = None

presets = {
    "Workout EDM":        "Upbeat EDM for workout, fast tempo, punchy drums",
    "Lo-fi study":        "Chill lo-fi hip hop for studying, soft drums, warm chords",
    "Chill house":        "Relaxed deep house, smooth bass, steady groove",
    "Epic":               "Epic orchestral, cinematic, powerful strings and brass",
    "Synthwave Retro":    "Retro synthwave, 100 BPM, analog bassline, dreamy pads, arpeggiated leads, neon night drive",
    "Drum & Bass":        "Fast liquid drum & bass, 174 BPM, rolling breakbeats, deep sub bass, atmospheric pads",
    "Cinematic Trailer":  "Epic cinematic trailer, huge percussion, risers, dramatic strings & brass, intense build-up",
    "Ambient Drone":      "Dark ambient drone, slow evolving pads, no drums, deep atmosphere, mysterious",
    "Modern Trap":        "Modern trap, 140 BPM half-time, heavy 808s, hi-hat rolls, dark minor synth melody",
    "Future Bass":        "Upbeat 128 BPM future bass, energetic saw synth leads, punchy sidechained kick, bright supersaw chords",
    "Hollywood Action BGM": "Hollywood action film background score, 130 BPM, hybrid orchestral, aggressive percussion, string ostinatos, brass stabs, rising tension",
    "Hollywood Thriller BGM": "Hollywood thriller background music, dark pulses, low strings, subtle synth drones, ticking rhythm, suspense build",
    "Hollywood Romance BGM": "Hollywood romantic movie background score, warm piano, soft strings, tender melody, emotional and hopeful",
    "Hollywood Sci-Fi BGM": "Hollywood sci-fi movie score, futuristic synth textures, deep sub pulses, wide ambient layers, cinematic transitions",
    "Hollywood Horror BGM": "Hollywood horror background score, eerie drones, dissonant strings, whisper textures, unsettling atmosphere",
    "Hollywood Adventure BGM": "Hollywood adventure film score, orchestral swells, heroic brass, rhythmic taikos, uplifting cinematic momentum",
    "Avengers Theme BGM": "Original cinematic superhero action theme with Avengers-like heroic brass, driving strings, hybrid percussion, and triumphant energy",
    "Marvel Superhero BGM": "Original cinematic superhero ensemble score with bold brass motifs, tense strings, modern hybrid trailer percussion, and heroic momentum",
    "Batman Vigilante BGM": "Original dark vigilante cinematic score with brooding low strings, deep punchy drums, tense pulses, and shadowy intensity",
    "Interstellar Space Drama BGM": "Original space drama cinematic ambience with emotional synth-organ textures, wide atmospheric layers, slow-bloom grandeur",
    "Inception Mind-Bending BGM": "Original mind-bending cinematic suspense with pulsing low-end, ticking rhythms, accelerating rises, and dramatic reveals",
    "Star Wars Space Opera BGM": "Original space opera orchestral adventure with heroic brass, sweeping strings, galaxy-wide momentum, and epic crescendos",
    "Harry Potter Magical Fantasy BGM": "Original magical orchestral fantasy with celesta sparkle, warm strings, evolving themes, and wondrous airy atmosphere",
    "James Bond Spy Thriller BGM": "Original spy thriller orchestral groove with confident brass stabs, tense rhythmic momentum, sleek cinematic color",
    "Bollywood Dance BGM": "Bollywood dance background track, energetic dhol rhythms, modern electronic groove, festive hooks, vibrant lead melody",
    "Bollywood Romance BGM": "Bollywood romantic background music, soulful flute, soft tabla, lush strings, emotional melodic phrasing",
    "Bollywood Drama BGM": "Bollywood drama film score, expressive strings, piano motifs, emotional progression, cinematic Indian orchestration",
    "Bollywood Action BGM": "Bollywood action background score, intense percussion, dhol and taiko fusion, dramatic brass, fast-paced tension",
    "Bollywood Folk Fusion BGM": "Bollywood folk fusion background music, sitar motifs, dholak groove, modern bass, cinematic folk energy",
    "Bollywood Sad Scene BGM": "Bollywood sad scene background score, slow piano, gentle strings, melancholic flute, heartfelt emotional tone",
    "Rainy Weather BGM": "Rainy day background music, soft piano, lo-fi texture, gentle ambience, calm reflective mood",
    "Thunderstorm BGM": "Thunderstorm cinematic ambience, deep rumble textures, dark drones, tension pads, dramatic weather mood",
    "Sunny Morning BGM": "Sunny morning background music, bright acoustic guitar, light percussion, warm piano, fresh uplifting vibe",
    "Winter Snow BGM": "Winter snowfall background score, delicate bells, soft strings, airy pads, peaceful cinematic calm",
    "Monsoon Bollywood BGM": "Bollywood monsoon romantic background music, expressive strings, bansuri flute, soft tabla, rain-soaked emotional mood",
    "Foggy Mystery BGM": "Foggy mystery movie background score, low drones, sparse piano notes, distant textures, slow suspense",
    "Emotional Movie BGM": "Emotional movie background score, cinematic piano and strings, gradual build, heartfelt dramatic climax",
    "Trailer Build-up BGM": "Movie trailer build-up background music, pulsing low end, risers, impacts, epic orchestral hybrid finish",
    "Chase Scene BGM": "Fast chase scene background score, driving percussion, tense strings, urgent brass accents, high energy pacing",
    "Detective Noir BGM": "Detective noir background music, moody jazz piano, brushed drums, upright bass, smoky late-night atmosphere",
    "Courtroom Drama BGM": "Courtroom drama background score, restrained piano motifs, low strings, emotional tension, serious cinematic tone",
    "End Credits BGM": "End credits cinematic music, reflective piano theme, warm strings, gentle closure, hopeful finale",
}

selected_preset = st.selectbox("Quick mood presets", list(presets.keys()), index=0)
if st.button("Load preset", use_container_width=False):
    st.session_state.prompt = presets[selected_preset]
    st.session_state.pending_main_prompt = presets[selected_preset]
    st.rerun()

movie_bgm_keys = [k for k in presets.keys() if "BGM" in k]
if movie_bgm_keys:
    st.caption("🎬 Movie BGM quick picks")
    selected_movie_bgm = st.selectbox("Movie BGM presets", movie_bgm_keys, index=0)
    if st.button("Load movie BGM preset", use_container_width=False):
        st.session_state.prompt = presets[selected_movie_bgm]
        st.session_state.pending_main_prompt = presets[selected_movie_bgm]
        st.rerun()

prompt = st.text_input("Your prompt", value=st.session_state.prompt, key="main_prompt")

with st.expander("💡 Pro Prompt Tips", expanded=False):
    st.markdown("""
    **Strongest results come from:**  
    Genre + BPM → Key instruments → Mood / Vibe / Structure

    **Great examples:**
    - upbeat 128 BPM future bass, energetic saw leads, punchy kick, festival drop energy
    - melancholic lo-fi jazz hop, dusty vinyl, 85 BPM, rhodes chords, nighttime vibe
    - epic cinematic trailer, massive percussion, risers, heroic brass & strings

    8–25 words works best. More detail = better coherence!
    """)

# ── Melody seed (remix mode only) ───────────────────────────────────────────
seed_waveform = None  # will be set below if a file is loaded

if is_melody:
    st.subheader("3 · Melody Seed (remix mode)")
    st.caption(
        "Upload 5–15 s of humming, guitar, piano, or any melody — "
        "this guides the generated track. Leave empty to fall back to text-only."
    )
    seed_file = st.file_uploader(
        "Melody seed (WAV / MP3)", type=["wav", "mp3"], key="melody_seed"
    )
    if seed_file is None:
        st.info("ℹ️ No seed uploaded — will generate from text prompt only.")
    else:
        try:
            audio_bytes = seed_file.read()
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                waveform, sr = sf.read(tmp_path, always_2d=True)  # [samples, channels]
            finally:
                # Always clean up the temp file
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            waveform = waveform.T.astype(np.float32)          # → [channels, samples]
            target_sr = 32000
            waveform = _resample(waveform, sr, target_sr)
            seed_waveform = torch.from_numpy(waveform[0:1])   # mono conditioning
            st.success(
                f"✅ Seed loaded — {seed_waveform.shape[-1] / target_sr:.1f} s, "
                f"resampled to {target_sr} Hz"
            )
        except Exception as e:
            st.error(f"Could not load seed: {e}. Try a clean WAV file.")

# ── Controls ────────────────────────────────────────────────────────────────
st.subheader("4 · Controls")

duration = st.slider("Duration (seconds)", min_value=5, max_value=30, value=10, step=5)

col_t, col_g = st.columns(2)
with col_t:
    temperature = st.slider(
        "Creativity (Temperature)", 0.70, 1.50, 1.00, 0.05,
        help="Higher = more unexpected & varied • Lower = safer & repetitive",
    )
with col_g:
    guidance_scale = st.slider(
        "Guidance Strength (CFG)", 1.0, 10.0, 3.5, 0.5,
        help="Higher = follows prompt more strictly. Great for remixes. (3–5 is a sweet spot)",
    )

with st.expander("⚙️ Advanced generation controls", expanded=False):
    st.caption("Fine-tune sampling. Defaults work well for most prompts.")
    col1, col2, col3 = st.columns(3)
    with col1:
        top_p = st.slider("Top-p", 0.70, 1.00, 0.95, 0.05,
                          help="Nucleus sampling probability mass.")
    with col2:
        top_k = st.slider("Top-k", 20, 250, 250, 5,
                          help="Token vocab cap per step.")
    with col3:
        num_variations = st.slider("Variations", 1, 4, 1, 1,
                                   help="Generate N tracks at once. Multiplies compute time.")
    col4, col5 = st.columns(2)
    with col4:
        lock_seed = st.toggle(
            "Lock random seed",
            value=False,
            help="Turn on for repeatable generations with same settings.",
        )
    with col5:
        seed_value = st.number_input(
            "Seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            disabled=not lock_seed,
            help="Used only when lock random seed is enabled.",
        )
    st.markdown("**Prompt Director (advanced)**")
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        auto_cinematic_boost = st.toggle(
            "Auto cinematic boost",
            value=True,
            help="Adds cinematic arrangement hints when prompt sounds like movie/BGM.",
        )
    with pcol2:
        arrangement = st.selectbox(
            "Arrangement arc",
            ["None", "Slow build", "Verse -> Drop", "Trailer rise", "Ambient evolving"],
            index=1,
        )
    with pcol3:
        stereo_width_hint = st.selectbox(
            "Stereo image hint",
            ["Default", "Wide cinematic", "Centered punch", "Mono compatible"],
            index=0,
        )
    qcol1, qcol2 = st.columns(2)
    with qcol1:
        performance_mode = st.selectbox(
            "Generation speed mode",
            ["Balanced", "Fast preview", "Ultra fast sketch"],
            index=0,
            help="Faster modes reduce generated tokens to return audio sooner.",
        )
    with qcol2:
        stream_status_frequency = st.selectbox(
            "Streaming UI refresh",
            ["Smooth", "Balanced", "Low overhead"],
            index=1,
            help="Lower refresh overhead improves speed on slower CPUs.",
        )

# ── Voice Input ─────────────────────────────────────────────────────────────
with st.expander("🎤 Voice command (beta)", expanded=False):
    st.caption("Capture voice from mic or upload audio, then turn speech into a polished prompt.")
    voice_input_mode = st.radio(
        "Voice source",
        ["Upload file", "Microphone capture", "Either"],
        horizontal=True,
    )
    voice_file = None
    mic_audio = None
    if voice_input_mode in ("Upload file", "Either"):
        voice_file = st.file_uploader("Upload WAV or MP3", type=["wav", "mp3"], key="voice_uploader")
    if voice_input_mode in ("Microphone capture", "Either"):
        if _MIC_RECORDER_AVAILABLE:
            mic_audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                format="wav",
                key="mic_recorder",
            )
        else:
            st.info("Microphone capture not available. Install `streamlit-mic-recorder` in your venv.")
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    with vcol1:
        voice_style = st.selectbox("Voice prompt style", ["Natural", "Detailed", "Cinematic", "Club-ready"])
    with vcol2:
        voice_energy = st.selectbox("Energy", ["Any", "Low", "Medium", "High"], index=2)
    with vcol3:
        tempo_hint = st.text_input("Tempo hint (BPM)", placeholder="e.g. 120")
    with vcol4:
        focus_instruments = st.text_input("Instrument focus", placeholder="e.g. warm pads, punchy kick")

    if st.button("Transcribe voice → use as prompt", type="secondary"):
        source_obj = mic_audio or voice_file
        if source_obj is None:
            st.warning("Provide voice input first (upload or microphone).")
        else:
            with st.spinner("Transcribing..."):
                try:
                    asr_pipeline = load_asr()
                    voice_bytes = _extract_audio_bytes(source_obj)
                    if not voice_bytes:
                        st.error("Could not read recorded audio. Please record again or upload a file.")
                        st.stop()
                    result = asr_pipeline(voice_bytes)
                    text = (
                        result[0]["text"].strip()
                        if isinstance(result, list) and result
                        else result.get("text", "").strip()
                    )
                    if text:
                        st.success(f"**Transcribed:** {text}")
                        st.session_state.transcribed_text = text

                        intent_key = _detect_intent_preset_key(text)
                        bpm_from_speech = _extract_bpm_hint(text)
                        energy_override = _detect_energy_from_text(text)
                        effective_energy = energy_override or voice_energy
                        effective_tempo = bpm_from_speech or tempo_hint.strip()

                        if intent_key and intent_key in presets:
                            base_prompt = presets[intent_key]
                            transcript_for_build = f"{base_prompt}, {text.strip()}"
                            st.info(f"Detected request -> mapped to preset: `{intent_key}`")
                        else:
                            rewritten_text, used_rewrite = _rewrite_pop_culture_prompt(text)
                            transcript_for_build = rewritten_text
                            if used_rewrite:
                                st.info("Detected theme words and converted them into cinematic style language.")

                        st.session_state.voice_built_prompt = _build_voice_prompt(
                            transcript=transcript_for_build,
                            style_preset=voice_style,
                            energy=effective_energy,
                            tempo_hint=effective_tempo,
                            instruments=focus_instruments.strip(),
                        )
                    else:
                        st.error("No clear speech detected — try a louder clip.")
                except Exception as e:
                    st.error(f"Transcription error: {e}")

    if "transcribed_text" in st.session_state:
        edited = st.text_input("Edit transcribed text:", value=st.session_state.transcribed_text)
        intent_key_edited = _detect_intent_preset_key(edited)
        bpm_from_speech_edited = _extract_bpm_hint(edited)
        energy_override_edited = _detect_energy_from_text(edited)
        effective_energy_edited = energy_override_edited or voice_energy
        effective_tempo_edited = bpm_from_speech_edited or tempo_hint.strip()

        if intent_key_edited and intent_key_edited in presets:
            base_prompt = presets[intent_key_edited]
            transcript_for_build = f"{base_prompt}, {edited.strip()}"
            built_prompt = _build_voice_prompt(
                transcript=transcript_for_build,
                style_preset=voice_style,
                energy=effective_energy_edited,
                tempo_hint=effective_tempo_edited,
                instruments=focus_instruments.strip(),
            )
            st.caption(f"Detected request -> mapped to preset: `{intent_key_edited}`")
        else:
            rewritten_edited, used_rewrite_edited = _rewrite_pop_culture_prompt(edited)
            built_prompt = _build_voice_prompt(
                transcript=rewritten_edited,
                style_preset=voice_style,
                energy=effective_energy_edited,
                tempo_hint=effective_tempo_edited,
                instruments=focus_instruments.strip(),
            )
            if used_rewrite_edited:
                st.caption("Theme words detected and converted into cinematic style language.")

        st.caption("Enhanced voice prompt preview:")
        st.code(built_prompt)
        if st.button("✅ Use enhanced prompt"):
            st.session_state.prompt = built_prompt
            st.session_state.pending_main_prompt = built_prompt
            del st.session_state.transcribed_text
            if "voice_built_prompt" in st.session_state:
                del st.session_state.voice_built_prompt
            st.success("Prompt updated!")
            st.rerun()

# ── Generate ─────────────────────────────────────────────────────────────────
st.divider()

if st.button("🎵 Generate music", type="primary", use_container_width=True):
    # session_state.main_prompt is kept in sync with the text_input widget
    current_prompt = (st.session_state.get("main_prompt") or prompt or "").strip()

    if not current_prompt:
        st.warning("Enter a prompt or load a preset first.")
        st.stop()

    est_time = "15–40 s" if device.type == "cpu" else "3–10 s"
    using_seed = seed_waveform is not None and is_melody
    using_stream = _STREAMER_AVAILABLE and num_variations == 1 and not is_stereo

    try:
        if lock_seed:
            torch.manual_seed(int(seed_value))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed_value))

        # Optional prompt shaping for movie/BGM style control.
        prompt_directives = []
        if auto_cinematic_boost and any(
            token in current_prompt.lower() for token in ("bgm", "movie", "cinematic", "theme", "trailer")
        ):
            prompt_directives.append("cinematic score mix, strong emotional progression, polished soundtrack style")
        arrangement_map = {
            "None": "",
            "Slow build": "slow intro, gradual layer build, controlled climax",
            "Verse -> Drop": "clear section change, pre-drop tension, impactful drop",
            "Trailer rise": "riser-driven structure, escalating tension, epic final hit",
            "Ambient evolving": "minimal rhythm, evolving ambient textures, long-form movement",
        }
        stereo_map = {
            "Default": "",
            "Wide cinematic": "wide stereo field, spacious reverb tails",
            "Centered punch": "tight center image, focused kick and bass",
            "Mono compatible": "mono-safe arrangement, avoid phase-heavy stereo tricks",
        }
        if arrangement_map.get(arrangement):
            prompt_directives.append(arrangement_map[arrangement])
        if stereo_map.get(stereo_width_hint):
            prompt_directives.append(stereo_map[stereo_width_hint])
        if prompt_directives:
            current_prompt = f"{current_prompt}, " + ", ".join(prompt_directives)

        if performance_mode != "Balanced":
            st.info(
                f"⚡ {performance_mode} active: returning output sooner by generating fewer tokens."
            )

        # ── Build processor inputs ───────────────────────────────────────────
        prompts = [current_prompt] * num_variations

        if using_seed:
            # Repeat seed for each variation
            seed_batch = seed_waveform.repeat(num_variations, 1)
            inputs = processor(
                text=prompts,
                audio=seed_batch,
                sampling_rate=32000,
                padding=True,
                return_tensors="pt",
            ).to(device)
            st.info("🎸 Melody conditioning active (remix mode)")
        else:
            inputs = processor(
                text=prompts,
                padding=True,
                return_tensors="pt",
            ).to(device)

        sample_rate = model.config.audio_encoder.sampling_rate

        # MusicGen's decoder has a hard position-embedding limit of 2048 tokens.
        # IMPORTANT: `repetition_penalty` is NOT supported by MusicGen — passing it
        # corrupts the delay-pattern mask and causes the OOB index crash.
        # We also cap tokens conservatively and pass max_length as a hard ceiling.
        max_new_tokens, model_max_pos = _compute_generation_limits(model, duration)
        speed_factor_map = {
            "Balanced": 1.0,
            "Fast preview": 0.7,
            "Ultra fast sketch": 0.5,
        }
        max_new_tokens = max(64, int(max_new_tokens * speed_factor_map[performance_mode]))
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=model_max_pos - 1,  # hard ceiling
            do_sample=True,
            temperature=temperature,
            guidance_scale=guidance_scale,
            top_p=top_p,
            top_k=top_k,
            # NOTE: repetition_penalty intentionally omitted — MusicGen does not
            # support it; passing it causes the "index 2048 out of bounds" crash.
        )

        st.subheader(
            f"Generated: **{current_prompt}** ({duration}s"
            + (f" × {num_variations} variations" if num_variations > 1 else "")
            + ")"
        )

        if using_stream:
            # ── Streaming path (single variation, mono, streamer available) ──
            audio_placeholder = st.empty()
            status_placeholder = st.empty()
            refresh_sec = {
                "Smooth": 0.1,
                "Balanced": 0.35,
                "Low overhead": 0.8,
            }[stream_status_frequency]

            streamer = MusicgenStreamer(model, device=device, play_steps=25)
            gen_kwargs["streamer"] = streamer

            thread = Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
            thread.start()

            audio_chunks = []
            total_samples = 0
            last_status_ts = 0.0
            for chunk in streamer:
                audio_chunks.append(chunk)
                total_samples += len(chunk)
                now = time.perf_counter()
                if now - last_status_ts >= refresh_sec:
                    status_placeholder.caption(
                        f"⏳ Streaming… {total_samples / sample_rate:.1f} s / {duration} s"
                    )
                    last_status_ts = now
            # Single concatenation at the end for the audio player
            combined = np.concatenate(audio_chunks)
            audio_placeholder.audio(audio_to_wav_bytes(combined, sample_rate), format="audio/wav")

            thread.join()
            status_placeholder.empty()
            raw_tracks = [combined]  # single mono track

        else:
            # ── Single-shot path (multi-variation or stereo) ─────────────────
            label = (
                f"Generating {num_variations} variation(s)"
                if num_variations > 1
                else f"Generating — ~{est_time} on {device.type.upper()}"
            )
            with st.spinner(label), torch.inference_mode():
                audio_values = model.generate(**gen_kwargs)

            raw_tracks = _to_output_tracks(audio_values, num_variations)

        # ── Post-processing & display ─────────────────────────────────────────
        _render_track_outputs(raw_tracks, sample_rate, num_variations)

        st.caption(
            "💡 Tip: Start with 5–10 s for fast previews. "
            "For remixes, keep seeds short (5–10 s) for the best blending."
        )

    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.caption(
            "Common fixes: reduce duration or variations, restart the app, or check available RAM."
        )

st.divider()
st.caption(
    "Built with Streamlit · Meta MusicGen (small / medium / melody / stereo-melody) · Wav2Vec2 ASR · Have fun! 🎶"
)