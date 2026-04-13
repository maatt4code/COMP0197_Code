"""
Whisper adapter training and shared inference utilities for COMP0197 Group 33.

Exposes:
- Training adapters (imported by train.py)
- Shared utilities (whisper_common, age_classifier, training_log)

GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

from .age_classifier import (
    AgeClassifierHead,
    TemperatureScaler,
    expected_calibration_error,
    load_gate_checkpoint,
    predict_gate_probs,
    run_gate_inference,
    save_gate_checkpoint,
)
from .train_by_age_groups_gatingmlp import GatingMLPAdapter
from .train_by_age_groups_lora import LoraAdapter
from .train_by_unique_subjects import UniqueSubjectsAdapter
from .training_log import StructuredTrainingLogger, TrainerMetricsCallback
from .whisper_common import (
    batch_records,
    encode_audios,
    extract_pooled_embeddings,
    get_whisper_encoder,
    load_audio,
    load_manifest_records,
    mean_pool_encoder_outputs,
    prepare_audio_features,
    transcribe_audio,
    transcribe_audio_with_details,
    transcribe_record,
)

__all__ = [
    # Training adapters
    "LoraAdapter",
    "GatingMLPAdapter",
    "UniqueSubjectsAdapter",
    # Gate / classifier
    "AgeClassifierHead",
    "TemperatureScaler",
    "expected_calibration_error",
    "load_gate_checkpoint",
    "predict_gate_probs",
    "run_gate_inference",
    "save_gate_checkpoint",
    # Logging
    "StructuredTrainingLogger",
    "TrainerMetricsCallback",
    # Shared Whisper utilities
    "batch_records",
    "encode_audios",
    "extract_pooled_embeddings",
    "get_whisper_encoder",
    "load_audio",
    "load_manifest_records",
    "mean_pool_encoder_outputs",
    "prepare_audio_features",
    "transcribe_audio",
    "transcribe_audio_with_details",
    "transcribe_record",
]
