"""
GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

from transformers import WhisperForConditionalGeneration
import torchaudio

MODEL_NAME = "openai/whisper-small"
AUDIO_FILE = "/cs/student/projects3/COMP0158/grp_1/data/utterance/U_00003c3ae1c35c6f.flac"
base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
audio, sample_rate = torchaudio.load(AUDIO_FILE)

# Resample to 16kHz if needed
if sample_rate != 16000:
    audio = torchaudio.functional.resample(audio, sample_rate, 16000)
type(audio), audio.shape, audio.dtype
print("Audio shape:", audio.shape)
print("Audio dtype:", audio.dtype)
print("Audio sample rate:", 16000)
print("Base model loaded successfully.")
