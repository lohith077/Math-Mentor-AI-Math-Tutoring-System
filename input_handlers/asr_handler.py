"""Whisper ASR using faster-whisper with math phrase normalization."""
import io, re, tempfile, os
from faster_whisper import WhisperModel
from config.settings import WHISPER_MODEL

_model = None

def get_whisper_model():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    return _model

MATH_PHRASE_MAP = {
    "square root of": "sqrt(",
    "raised to the power of": "**",
    "raised to": "**",
    "to the power": "**",
    "divided by": "/",
    "times": "*",
    "plus": "+",
    "minus": "-",
    "equals": "=",
    "pi": "π",
}

def normalize_math_phrases(text: str) -> str:
    result = text.lower()
    for phrase, symbol in MATH_PHRASE_MAP.items():
        result = result.replace(phrase, symbol)
    return result

def transcribe_audio(audio_bytes: bytes) -> dict:
    model = get_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        transcript = " ".join([seg.text for seg in segments]).strip()
        # Estimate confidence from avg log probability (convert to 0-100)
        # Use a safer check for all_language_probs to avoid NoneType subscription error
        prob = 0.85
        if hasattr(info, 'all_language_probs') and info.all_language_probs:
            prob = info.all_language_probs[0][1]
        
        confidence = min(100, max(0, round(prob * 100, 1)))
        normalized = normalize_math_phrases(transcript)
        return {"text": normalized, "confidence": confidence, "raw_transcript": transcript}
    finally:
        os.unlink(tmp_path)
