"""
GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

import argparse
import sys
import torch
from config import Config, DEFAULT_BASE_DATA_DIR, TrainingConfig
from data_layout import print_validation_report, validate_data_layout
from models import train_by_age_groups_lora
from models import train_by_age_groups_gatingmlp
from models import train_by_unique_subjects
from pathlib import Path
from peft import PeftModel

HERE = Path(__file__).resolve().parent

# ── Adapter registry ──────────────────────────────────────────────────────────

ALL_ADAPTERS = [
    *[f"age_{b.replace('-', '_')}" for b in Config.lora_age_buckets()],
    "unique_subjects",
    "gate_mlp",
]

# Prerequisites that must be loaded (from best weights) before training a given adapter.
# Prereqs listed here are loaded but NOT trained unless also in --adapters.
ADAPTER_DEPS: dict[str, list[str]] = {
    "age_3_4":         [],
    "age_5_7":         [],
    "age_8_11":        [],
    "gate_mlp":        [],
    "unique_subjects": [],
}
ENSEMBLE_DEPS = ALL_ADAPTERS  # ensemble needs everything loaded from weights/final/

ADAPTER_TO_TRAINING_CONFIG: dict[str, TrainingConfig] = {
    **{
        f"age_{b.replace('-', '_')}": TrainingConfig(
            adapter_name=f"age_{b.replace('-', '_')}",
            train_json=f"train_by_age_bucket_{b.replace('-', '_')}.json",
            val_json=f"val_by_age_bucket_{b.replace('-', '_')}.json",
            age_group=b,
        )
        for b in Config.lora_age_buckets()
    },
    "unique_subjects": TrainingConfig(
        adapter_name="unique_subjects",
        train_json="train_by_child_id.json",
        val_json="val_by_child_id.json",
    ),
    "gate_mlp": TrainingConfig(
        adapter_name="gate_mlp",
        train_json="train_by_child_id.json",
        val_json="val_by_child_id.json",
    ),
}


# ── Training mode ─────────────────────────────────────────────────────────────

class TrainingMode:
    MOCK        = "mock"
    TA_TRAIN    = "ta_train"
    REALLY_TRAIN = "really_train"


# ── Prerequisite resolution ───────────────────────────────────────────────────

def _prereqs_for(adapters_to_train: list[str]) -> list[str]:
    """Return prerequisite adapters that need loading but are NOT being trained."""
    to_train = set(adapters_to_train)
    seen: set[str] = set()
    prereqs: list[str] = []
    for adapter in adapters_to_train:
        for dep in ADAPTER_DEPS.get(adapter, []):
            if dep not in to_train and dep not in seen:
                prereqs.append(dep)
                seen.add(dep)
    return prereqs


# ── Factory ───────────────────────────────────────────────────────────────────

class AdapterTrainerFactory:
    def __init__(self, config: Config, mode: str):
        self.config            = config
        self.mode              = mode
        self.device            = config.device()
        self.prereq_peft_model = None
        self.base_model_processor, self.base_model = config.base_model()

    @property
    def _mock(self) -> bool:
        return self.mode == TrainingMode.MOCK

    @property
    def _ta_train(self) -> bool:
        return self.mode == TrainingMode.TA_TRAIN

    def load_prereqs(self, prereq_adapters: list[str]) -> None:
        """Load prerequisite adapters from weights/best/ into self.prereq_peft_model."""
        if not prereq_adapters:
            return
        print(f"\n[prereqs] Loading {prereq_adapters} from load weights...")
        first      = prereq_adapters[0]
        load_dir   = self.config.adapter_load_weights_path(first)
        peft_model = PeftModel.from_pretrained(
            self.base_model, str(load_dir), adapter_name=first
        )
        for name in prereq_adapters[1:]:
            load_dir = self.config.adapter_load_weights_path(name)
            peft_model.load_adapter(str(load_dir), adapter_name=name)
            print(f"[prereqs]   loaded {name}")
        self.prereq_peft_model = peft_model.to(self.device).eval()

    def load_final_ensemble(self) -> tuple:
        """Load all adapter weights from weights/final/ for ensemble training.

        Returns
        -------
        peft_model : PeftModel
            Base Whisper with all LoRA adapters (age_3_4, age_5_7, age_8_11,
            unique_subjects) loaded from weights/final/.
        gate_ckpt : dict
            Raw gate checkpoint dict loaded from weights/final/gate_mlp/gate_mlp.pt.
        processor : WhisperProcessor
        """
        lora_adapters = [a for a in ALL_ADAPTERS if a != "gate_mlp"]

        print("\n[ensemble] Loading final adapter weights...")
        first = lora_adapters[0]
        final_dir = self.config.adapter_weights_path(first)
        peft_model = PeftModel.from_pretrained(
            self.base_model, str(final_dir), adapter_name=first
        )
        print(f"[ensemble]   loaded {first} from {final_dir}")
        for name in lora_adapters[1:]:
            final_dir = self.config.adapter_weights_path(name)
            peft_model.load_adapter(str(final_dir), adapter_name=name)
            print(f"[ensemble]   loaded {name} from {final_dir}")

        gate_path = self.config.adapter_weights_path("gate_mlp") / "gate_mlp.pt"
        print(f"[ensemble]   loading gate_mlp from {gate_path}")
        gate_ckpt = torch.load(str(gate_path), map_location=self.device)

        peft_model = peft_model.to(self.device).eval()
        return peft_model, gate_ckpt, self.base_model_processor

    def get_adapter_trainer(self, cfg: TrainingConfig):
        adapter_name = cfg.adapter_name
        if adapter_name.startswith("age_"):
            return train_by_age_groups_lora.LoraAdapter(
                self.config, cfg.age_group, self.base_model, self.base_model_processor,
                mock=self._mock, ta_train=self._ta_train,
            )
        elif adapter_name == "gate_mlp":
            return train_by_age_groups_gatingmlp.GatingMLPAdapter(
                self.config, self.base_model, self.base_model_processor,
                mock=self._mock, ta_train=self._ta_train,
            )
        elif adapter_name == "unique_subjects":
            return train_by_unique_subjects.UniqueSubjectsAdapter(
                self.config, self.base_model, self.base_model_processor,
                mock=self._mock, ta_train=self._ta_train,
            )
        else:
            raise ValueError(f"Unknown adapter name: {adapter_name!r}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one or more Whisper adapter modules.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--adapters",
        action="append",
        choices=ALL_ADAPTERS,
        dest="adapters",
        metavar="ADAPTER",
        default=None,
        help=(
            "Adapter to train (repeatable, default: all adapters).\n"
            f"Choices: {ALL_ADAPTERS}\n"
            "Prerequisite adapters are loaded automatically from weights/best/.\n"
            "Examples:\n"
            "  --adapters age_3_4\n"
            "  --adapters age_3_4 --adapters gate_mlp"
        ),
    )
    parser.add_argument(
        "--train-ensemble",
        action="store_true",
        help=(
            "Train the ensemble on top of all adapters.\n"
            "Loads all adapters from weights/final/ (release checkpoints).\n"
            "Cannot be used with --really-train."
        ),
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help=(
            "Validate JSON manifests under data/ and audio (and optionally noise) directories.\n"
            "Exits 0 if the expected layout is present; prints actionable errors otherwise.\n"
            "Does not download data — point --base-data-dir at your dataset root."
        ),
    )
    parser.add_argument(
        "--best-dir",
        type=str,
        default=None,
        help=(
            "Subdirectory under weights/ where trained checkpoints are *written*.\n"
            "Default: 'best'  →  weights/best/<adapter_name>/\n"
            "Example: --best-dir run_01  →  weights/run_01/<adapter_name>/"
        ),
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help=(
            "Subdirectory under weights/ from which checkpoints are *loaded*\n"
            "(mock mode and prerequisites). Defaults to the same value as --best-dir.\n"
            "Example: --best-dir run_01 --load-dir run_00\n"
            "  → loads prereqs from weights/run_00/, saves new run to weights/run_01/"
        ),
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=None,
        help=(
            "Base dataset directory (parent of audio/ and noise/).\n"
            f"Default: {DEFAULT_BASE_DATA_DIR}"
        ),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help=(
            "Audio root directory (audio_path values in JSONs are relative to this).\n"
            "Default: <base-data-dir>/audio/"
        ),
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help=(
            "Noise files directory for augmentation.\n"
            "Default: <base-data-dir>/noise/"
        ),
    )
    parser.add_argument(
        "--no-plot-training-logs",
        action="store_true",
        help=(
            "After --ta-train or --really-train, skip writing training_logs.pdf "
            "(multi-panel curves from weights/<best-dir>/*/training_log.json)."
        ),
    )

    # ── Mutually exclusive training modes (default: mock) ─────────────────────
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help=(
            "[DEFAULT] Mock mode: skip training loops.\n"
            "  Loads and verifies adapter_config.json from weights/best/.\n"
            "  Prerequisites are still loaded from weights/best/ as normal."
        ),
    )
    mode_group.add_argument(
        "--ta-train",
        action="store_true",
        help=(
            f"TA verification mode: train on {TrainingConfig.TA_TRAIN_SAMPLES} samples "
            f"for {TrainingConfig.TA_TRAIN_EPOCHS} epochs only.\n"
            "Lets TAs verify training code without a full dataset or GPU."
        ),
    )
    mode_group.add_argument(
        "--really-train",
        action="store_true",
        help=(
            "Full training mode: train on the complete dataset.\n"
            "Requires explicit confirmation at the prompt.\n"
            "Cannot be combined with --train-ensemble."
        ),
    )
    return parser.parse_args()


def _resolve_mode(args: argparse.Namespace) -> str:
    """Return the active TrainingMode, validate exclusivity rules, prompt if needed."""
    # argparse mutually_exclusive_group prevents >1 flag being set,
    # but --mock has default=True so we need to detect the *explicit* intent.
    if args.really_train:
        if args.train_ensemble:
            print(
                "ERROR: --really-train and --train-ensemble cannot be used together.\n"
                "       --train-ensemble loads adapter weights from weights/final/; "
                "using freshly trained (not yet saved) weights is not supported.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Prompt for confirmation
        print("=" * 60)
        print("  WARNING: REALLY_TRAIN mode selected.")
        print(f"  Adapters : {args.adapters or 'all'}")
        print(f"  Audio dir: {Config.audio_dir()}")
        print("  This will run a FULL training run. This may take a long time.")
        print("=" * 60)
        answer = input("  Type 'yes' to proceed: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)
        return TrainingMode.REALLY_TRAIN

    if args.ta_train:
        return TrainingMode.TA_TRAIN

    # Default (--mock or no flag given)
    return TrainingMode.MOCK


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.best_dir is not None:
        Config.set_best_dir(args.best_dir)
    if args.load_dir is not None:
        Config.set_load_dir(args.load_dir)
    if args.base_data_dir is not None:
        Config.set_base_data_dir(args.base_data_dir)
    if args.audio_dir is not None:
        Config.set_audio_dir(args.audio_dir)
    if args.noise_dir is not None:
        Config.set_noise_dir(args.noise_dir)

    if args.prepare_data:
        ok, msgs = validate_data_layout(require_noise=False)
        print_validation_report(ok, msgs)
        sys.exit(0 if ok else 1)

    mode = _resolve_mode(args)

    config = Config(is_inference=False)

    adapters_to_train: list[str] = args.adapters if args.adapters is not None else ALL_ADAPTERS.copy()
    prereqs = _prereqs_for(adapters_to_train)
    ensemble_prereqs = ENSEMBLE_DEPS if args.train_ensemble else []

    if mode in (TrainingMode.REALLY_TRAIN, TrainingMode.TA_TRAIN):
        ok, msgs = validate_data_layout(require_noise=True)
        if not ok:
            print_validation_report(ok, msgs)
            sys.exit(1)

    print(f"Device        : {config.device()}")
    print(f"Audio dir     : {Config.audio_dir()}")
    print(f"Noise dir     : {Config.noise_dir()}")
    print(f"Training mode : {mode}")
    print(f"Adapters      : {adapters_to_train}")
    print(f"Prerequisites : {prereqs or '(none)'}")
    print(f"Train ensemble: {args.train_ensemble}")

    factory = AdapterTrainerFactory(config, mode=mode)

    # Load prerequisites (adapters required by targets but not being trained)
    all_prereqs = list(dict.fromkeys(
        prereqs + [p for p in ensemble_prereqs if p not in adapters_to_train]
    ))
    factory.load_prereqs(all_prereqs)

    # Train each requested adapter
    for adapter_name in adapters_to_train:
        cfg: TrainingConfig = ADAPTER_TO_TRAINING_CONFIG[adapter_name]
        print(f"\nAdapter: {cfg.adapter_name} | train={cfg.train_json} | val={cfg.val_json}")
        adapter = factory.get_adapter_trainer(cfg)
        adapter.train(train_json=cfg.train_json, val_json=cfg.val_json)

    if args.train_ensemble:
        # Hook for ensemble trainer: loads weights/final/ and returns (peft_model, gate_ckpt, processor).
        factory.load_final_ensemble()
        print("\n[ensemble] Ensemble training is not yet implemented.")

    if (
        mode in (TrainingMode.REALLY_TRAIN, TrainingMode.TA_TRAIN)
        and not args.no_plot_training_logs
    ):
        weights_parent = config.adapter_best_weights_path(adapters_to_train[0]).parent
        try:
            from plot_training_logs import write_training_logs_pdf

            out_pdf = write_training_logs_pdf(weights_dir=weights_parent, out=HERE / "training_logs.pdf")
            print(f"\n[plots] Training log PDF → {out_pdf}")
        except FileNotFoundError:
            print("\n[plots] No training_log.json files found; skipping training_logs.pdf.")
        except ImportError as exc:
            print(f"\n[plots] Skipping training_logs.pdf (import failed: {exc}).")


if __name__ == "__main__":
    main()
