
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class Config:
    # ADMIN
    __SEED = 42
    __MAIN_DIR   = Path(__file__).resolve().parent
    __FINAL_DIR = "final"
    __MOCK_DIR = "mock"

    # MODEL
    __MODEL_NAME = "openai/whisper-small"
    __MODEL_DIM = 768      # whisper-small d_model
    __SAMPLE_RATE  = 16_000
    __ADAPTER_WEIGHTS_DIR = __MAIN_DIR / "weights"
    __ADAPTER_NAMES = ["age_3_4", "age_5_7", "age_8_11", "gate_mlp", "unique_subjects"]

    # LoRA
    __LORA_AGE_BUCKETS   = ["3-4", "5-7", "8-11"]
    __LORA_AGE_BUCKET_MAP = {k: v for v, k in enumerate(__LORA_AGE_BUCKETS)}
    __LORA_ADAPTER_NAMES = [name for name in __ADAPTER_NAMES if name.startswith("age_")]

    @staticmethod
    def seed():
        return Config.__SEED

    @staticmethod
    def model_name():
        return Config.__MODEL_NAME

    @staticmethod
    def model_dim():
        return Config.__MODEL_DIM

    @staticmethod
    def sample_rate():
        return Config.__SAMPLE_RATE

    @staticmethod
    def base_model():
        processor = WhisperProcessor.from_pretrained(Config.MODEL_NAME, language="English", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
        return processor, model

    def __init__(self, is_inference=True):
        torch.manual_seed(self.__SEED)
        self.__TRAIN_ADAPTERS = False
        self.__TRAIN_ENSEMBLE = False
        self.__MOCK_TRAINING = False
        self.__INFERENCE_ONLY = is_inference
        self.__WEIGHTS_SUBDIR = self.__MOCK_DIR if self.__MOCK_TRAINING else self.__FINAL_DIR

        # Lora weights directories
        self.__LORA_ADAPTER_DIRS = (lambda buckets, names, base: {
            bucket: base / self.__WEIGHTS_SUBDIR / f"{name}"
            for bucket, name in zip(buckets, names)
        })(Config.__LORA_AGE_BUCKETS, Config.__LORA_ADAPTER_NAMES, Config.__ADAPTER_WEIGHTS_DIR)

        # Gating MLP directory
        self.__GATING_MLP_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__WEIGHTS_SUBDIR / "gate_mlp"

        # Unique ChildId weights directory
        self.__UNIQUE_SUBJECTS_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__WEIGHTS_SUBDIR / "unique_subjects"

    def device(self):
        if self.__INFERENCE_ONLY:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    def adapter_weights_path(self, adapter: str):
        assert adapter in Config.ADAPTER_NAMES, f"Adapter {adapter} not found in {Config.ADAPTER_NAMES}"
        if adapter in Config.LORA_ADAPTER_NAMES:
            path = self.__LORA_ADAPTER_DIRS[adapter]
        elif adapter == "gate_mlp":
            path = self.__GATING_MLP_DIR
        else:   # adapter == "unique_subjects"
            path = self.__UNIQUE_SUBJECTS_DIR
        return path
