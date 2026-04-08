
from pathlib import Path
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class Config:
    # ADMIN
    __SEED = 42
    __MAIN_DIR   = Path(__file__).resolve().parent
    __FINAL_DIR = "final"
    __MOCK_DIR = "mock"
    __BEST_DIR: str = "best"   # subdirectory under weights/ to *write* checkpoints; override via set_best_dir()
    __LOAD_DIR: str = "best"   # subdirectory under weights/ to *read* checkpoints; override via set_load_dir()

    # MODEL
    __MODEL_NAME = "openai/whisper-small"
    __MODEL_DIM = 768      # whisper-small d_model
    __SAMPLE_RATE  = 16_000
    __ADAPTER_WEIGHTS_DIR = __MAIN_DIR / "weights"
    __ADAPTER_NAMES = ["age_3_4", "age_5_7", "age_8_11", "gate_mlp", "unique_subjects"]

    # DATA
    # __DATA_DIR: where JSON manifest files live (local to this project)
    __DATA_DIR = __MAIN_DIR / "data"
    # External dataset layout:
    #   __BASE_DATA_DIR/
    #     audio/   ← audio_path values in JSON manifests are relative to this
    #     noise/   ← background-noise files for augmentation
    # Each can be overridden independently via Config.set_*() class methods.
    __BASE_DATA_DIR: Path = Path("/cs/student/projects3/COMP0158/grp_1/data")
    __AUDIO_DIR: Path | None = None   # None → derived from __BASE_DATA_DIR
    __NOISE_DIR: Path | None = None   # None → derived from __BASE_DATA_DIR

    # LoRA
    __LORA_AGE_BUCKETS   = ["3-4", "5-7", "8-11"]
    __LORA_AGE_BUCKET_MAP = {k: v for v, k in enumerate(__LORA_AGE_BUCKETS)}
    __LORA_ADAPTER_NAMES = [name for name in __ADAPTER_NAMES if name.startswith("age_")]
    __LORA_BUCKET_TO_ADAPTER = {b: n for b, n in zip(__LORA_AGE_BUCKETS, __LORA_ADAPTER_NAMES)}

    # ── class-level setters ──────────────────────────────────────────────────

    @classmethod
    def set_best_dir(cls, name: str) -> None:
        """Override the subdirectory under weights/ where trained checkpoints are *written*.

        E.g. Config.set_best_dir("run_01") → weights/run_01/<adapter_name>/
        """
        cls.__BEST_DIR = name

    @classmethod
    def set_load_dir(cls, name: str) -> None:
        """Override the subdirectory under weights/ from which checkpoints are *loaded*.

        Defaults to the same value as __BEST_DIR ("best").
        E.g. Config.set_load_dir("run_00") → load from weights/run_00/<adapter_name>/
        """
        cls.__LOAD_DIR = name

    @classmethod
    def set_base_data_dir(cls, path) -> None:
        """Override the base dataset directory (parent of audio/ and noise/)."""
        cls.__BASE_DATA_DIR = Path(path)

    @classmethod
    def set_audio_dir(cls, path) -> None:
        """Override the audio directory independently of base_data_dir."""
        cls.__AUDIO_DIR = Path(path)

    @classmethod
    def set_noise_dir(cls, path) -> None:
        """Override the noise directory independently of base_data_dir."""
        cls.__NOISE_DIR = Path(path)

    # ── static accessors ────────────────────────────────────────────────────

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
    def data_dir() -> Path:
        """Directory that holds the JSON manifest files."""
        return Config.__DATA_DIR

    @staticmethod
    def audio_dir() -> Path:
        """Root directory prepended to audio_path values found in JSON manifests."""
        if Config.__AUDIO_DIR is not None:
            return Config.__AUDIO_DIR
        return Config.__BASE_DATA_DIR / "audio"

    @staticmethod
    def noise_dir() -> Path:
        """Directory containing background-noise files for augmentation."""
        if Config.__NOISE_DIR is not None:
            return Config.__NOISE_DIR
        return Config.__BASE_DATA_DIR / "noise"

    @staticmethod
    def base_model():
        processor = WhisperProcessor.from_pretrained(Config.__MODEL_NAME, language="English", task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(Config.__MODEL_NAME)
        return processor, model

    @staticmethod
    def adapter_names():
        return Config.__ADAPTER_NAMES

    @staticmethod
    def lora_age_buckets():
        return Config.__LORA_AGE_BUCKETS

    @staticmethod
    def lora_bucket_to_adapter(bucket: str) -> str:
        assert bucket in Config.__LORA_BUCKET_TO_ADAPTER, \
            f"Unknown bucket {bucket!r}. Valid: {list(Config.__LORA_BUCKET_TO_ADAPTER)}"
        return Config.__LORA_BUCKET_TO_ADAPTER[bucket]

    # ── instance methods ────────────────────────────────────────────────────

    def __init__(self, is_inference=True):
        torch.manual_seed(self.__SEED)
        self.__MOCK_TRAINING = False
        self.__INFERENCE_ONLY = is_inference
        self.__WEIGHTS_SUBDIR = self.__MOCK_DIR if self.__MOCK_TRAINING else self.__FINAL_DIR

        # Final / mock weights directories
        self.__LORA_ADAPTER_DIRS = {
            bucket: Config.__ADAPTER_WEIGHTS_DIR / self.__WEIGHTS_SUBDIR / name
            for bucket, name in Config.__LORA_BUCKET_TO_ADAPTER.items()
        }
        self.__GATING_MLP_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__WEIGHTS_SUBDIR / "gate_mlp"
        self.__UNIQUE_SUBJECTS_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__WEIGHTS_SUBDIR / "unique_subjects"

        # Best weights directories (written by training runs)
        self.__BEST_LORA_DIRS = {
            bucket: Config.__ADAPTER_WEIGHTS_DIR / self.__BEST_DIR / name
            for bucket, name in Config.__LORA_BUCKET_TO_ADAPTER.items()
        }
        self.__BEST_GATING_MLP_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__BEST_DIR / "gate_mlp"
        self.__BEST_UNIQUE_SUBJECTS_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__BEST_DIR / "unique_subjects"

        # Load weights directories (read by mock mode and prerequisite loading)
        self.__LOAD_LORA_DIRS = {
            bucket: Config.__ADAPTER_WEIGHTS_DIR / self.__LOAD_DIR / name
            for bucket, name in Config.__LORA_BUCKET_TO_ADAPTER.items()
        }
        self.__LOAD_GATING_MLP_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__LOAD_DIR / "gate_mlp"
        self.__LOAD_UNIQUE_SUBJECTS_DIR = Config.__ADAPTER_WEIGHTS_DIR / self.__LOAD_DIR / "unique_subjects"

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

    def adapter_weights_path(self, adapter: str) -> Path:
        """Return the *final/mock* weights directory for the given adapter name."""
        assert adapter in Config.__ADAPTER_NAMES, \
            f"Adapter {adapter!r} not found. Valid: {Config.__ADAPTER_NAMES}"
        if adapter in Config.__LORA_ADAPTER_NAMES:
            bucket = next(b for b, n in Config.__LORA_BUCKET_TO_ADAPTER.items() if n == adapter)
            return self.__LORA_ADAPTER_DIRS[bucket]
        if adapter == "gate_mlp":
            return self.__GATING_MLP_DIR
        return self.__UNIQUE_SUBJECTS_DIR

    def adapter_best_weights_path(self, adapter: str) -> Path:
        """Return the *save* weights directory. Accepts adapter name or age bucket string."""
        if adapter in Config.__LORA_BUCKET_TO_ADAPTER:
            return self.__BEST_LORA_DIRS[adapter]
        if adapter in Config.__LORA_ADAPTER_NAMES:
            bucket = next(b for b, n in Config.__LORA_BUCKET_TO_ADAPTER.items() if n == adapter)
            return self.__BEST_LORA_DIRS[bucket]
        if adapter == "gate_mlp":
            return self.__BEST_GATING_MLP_DIR
        if adapter == "unique_subjects":
            return self.__BEST_UNIQUE_SUBJECTS_DIR
        raise ValueError(f"Unknown adapter {adapter!r}")

    def adapter_load_weights_path(self, adapter: str) -> Path:
        """Return the *load* weights directory. Accepts adapter name or age bucket string."""
        if adapter in Config.__LORA_BUCKET_TO_ADAPTER:
            return self.__LOAD_LORA_DIRS[adapter]
        if adapter in Config.__LORA_ADAPTER_NAMES:
            bucket = next(b for b, n in Config.__LORA_BUCKET_TO_ADAPTER.items() if n == adapter)
            return self.__LOAD_LORA_DIRS[bucket]
        if adapter == "gate_mlp":
            return self.__LOAD_GATING_MLP_DIR
        if adapter == "unique_subjects":
            return self.__LOAD_UNIQUE_SUBJECTS_DIR
        raise ValueError(f"Unknown adapter {adapter!r}")


class TrainingConfig:
    """Hyperparameters and data pointers for one adapter training run.

    ``train_json`` and ``val_json`` are **filenames** (e.g. ``train_by_age_bucket_3_4.json``).
    They are resolved at training time as ``Config.data_dir() / <filename>``.
    Audio paths inside the JSON are resolved as ``Config.audio_dir() / record["audio_path"]``.
    """

    # LoRA architecture
    LORA_R              = 16
    LORA_ALPHA          = 32
    LORA_DROPOUT        = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # Training loop
    BATCH_SIZE    = 16
    LEARNING_RATE = 1e-3
    NUM_EPOCHS    = 3
    WARMUP_STEPS  = 50

    # TA_TRAIN mode — tiny run to verify training code without killing a laptop
    TA_TRAIN_SAMPLES = 100
    TA_TRAIN_EPOCHS  = 5

    def __init__(self, adapter_name: str, train_json: str, val_json: str, age_group: str = None):
        self.adapter_name = adapter_name
        assert self.adapter_name in Config.adapter_names(), \
            f"Adapter {self.adapter_name!r} not found in {Config.adapter_names()}"
        self.train_json = train_json   # filename only, resolved via Config.data_dir()
        self.val_json   = val_json     # filename only, resolved via Config.data_dir()
        self.age_group  = age_group
