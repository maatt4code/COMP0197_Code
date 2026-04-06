
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class Config:
    # ADMIN
    SEED = 42
    MAIN_DIR   = Path(__file__).resolve().parent

    # MODEL
    MODEL_NAME = "openai/whisper-small"
    MODEL_DIM = 768      # whisper-small d_model
    SAMPLE_RATE  = 16_000
    ADAPTER_WEIGHTS_DIR = MAIN_DIR / "weights"
    ADAPTER_WEIGHTS_FILE_PREFIX = "whisper_small"
    ADAPTER_NAMES = ["age_3_4", "age_5_7", "age_8_11", "unique_subjects"]

    # LoRA
    LORA_AGE_BUCKETS   = ["3-4", "5-7", "8-11"]
    LORA_AGE_BUCKET_MAP = {k: v for v, k in enumerate(LORA_AGE_BUCKETS)}
    LORA_ADAPTER_NAMES = [name for name in ADAPTER_NAMES if name.startswith("age_")]
    
    LORA_ADAPTER_DIRS = (lambda buckets, names, base, prefix: {
        bucket: base / f"{prefix}_{name}"
        for bucket, name in zip(buckets, names)
    })(LORA_AGE_BUCKETS, LORA_ADAPTER_NAMES, ADAPTER_WEIGHTS_DIR, ADAPTER_WEIGHTS_FILE_PREFIX)

    # Gating MLP
    GATING_MLP_DIR = ADAPTER_WEIGHTS_DIR / f"{ADAPTER_WEIGHTS_FILE_PREFIX}_gating_mlp"


    # RUNTIME
    TRAIN_ADAPTERS = True
    TRAIN_ENSEMBLE = True
    MOCK_TRAINING = True
    INFERENCE_ONLY = False

    @staticmethod
    def device(is_inference=True):
        if is_inference:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    @staticmethod
    def base_model():
        processor = WhisperProcessor.from_pretrained(Config.MODEL_NAME, language="English", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
        return processor, model
