import argparse
from config import Config, TrainingConfig
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
    "age_3_4":        [],
    "age_5_7":        [],
    "age_8_11":       [],
    "gate_mlp":       ["age_3_4", "age_5_7", "age_8_11"],
    "unique_subjects": [],
}
ENSEMBLE_DEPS = ALL_ADAPTERS  # ensemble needs everything

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
        train_json="train_by_age_bucket_all.json",
        val_json="val_by_age_bucket_all.json",
    ),
}


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
    def __init__(self, config: Config, mock: bool = False):
        self.config            = config
        self.mock              = mock
        self.device            = config.device()
        self.prereq_peft_model = None
        self.base_model_processor, self.base_model = config.base_model()

    def load_prereqs(self, prereq_adapters: list[str]) -> None:
        """Load prerequisite adapters from weights/best/ into self.prereq_peft_model."""
        if not prereq_adapters:
            return
        print(f"\n[prereqs] Loading {prereq_adapters} from best weights...")
        first      = prereq_adapters[0]
        best_dir   = self.config.adapter_best_weights_path(first)
        peft_model = PeftModel.from_pretrained(
            self.base_model, str(best_dir), adapter_name=first
        )
        for name in prereq_adapters[1:]:
            best_dir = self.config.adapter_best_weights_path(name)
            peft_model.load_adapter(str(best_dir), adapter_name=name)
            print(f"[prereqs]   loaded {name}")
        self.prereq_peft_model = peft_model.to(self.device).eval()

    def get_adapter_trainer(self, cfg: TrainingConfig):
        adapter_name = cfg.adapter_name
        if adapter_name.startswith("age_"):
            return train_by_age_groups_lora.LoraAdapter(
                self.config, cfg.age_group, self.base_model, self.base_model_processor,
                mock=self.mock,
            )
        elif adapter_name == "gate_mlp":
            assert self.prereq_peft_model is not None, (
                "gate_mlp requires age adapter prerequisites — call load_prereqs() first."
            )
            return train_by_age_groups_gatingmlp.GatingMLPAdapter(
                self.config, self.prereq_peft_model, self.base_model_processor,
                mock=self.mock,
            )
        elif adapter_name == "unique_subjects":
            return train_by_unique_subjects.UniqueSubjectsAdapter(
                self.config, self.base_model, self.base_model_processor
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
            "All adapters are loaded from weights/best/ as prerequisites."
        ),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help=(
            "Root directory that audio_path values in JSON manifests are relative to.\n"
            "Overrides Config.audio_root()."
        ),
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=None,
        help="Directory of background-noise files for augmentation (optional).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Mock mode: skip training loops.\n"
            "  - For adapters being 'trained': load and verify their adapter_config.json.\n"
            "  - Prerequisites are still loaded from weights/best/ as normal."
        ),
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.audio_dir is not None:
        Config.set_audio_root(args.audio_dir)

    config = Config(is_inference=False)

    adapters_to_train: list[str] = args.adapters if args.adapters is not None else [
        a for a in ALL_ADAPTERS if a != "gate_mlp"  # gate_mlp excluded until implemented
    ]
    prereqs = _prereqs_for(adapters_to_train)
    ensemble_prereqs = ENSEMBLE_DEPS if args.train_ensemble else []

    print(f"Device        : {config.device()}")
    print(f"Audio root    : {Config.audio_root()}")
    print(f"Noise dir     : {args.noise_dir or '(none)'}")
    print(f"Mock mode     : {args.mock}")
    print(f"Adapters      : {adapters_to_train}")
    print(f"Prerequisites : {prereqs or '(none)'}")
    print(f"Train ensemble: {args.train_ensemble}")

    factory = AdapterTrainerFactory(config, mock=args.mock)

    # Load prerequisites (adapters required by the targets but not being trained)
    all_prereqs = list(dict.fromkeys(prereqs + [p for p in ensemble_prereqs if p not in adapters_to_train]))
    factory.load_prereqs(all_prereqs)

    # Train (or mock-train) each requested adapter
    for adapter_name in adapters_to_train:
        cfg: TrainingConfig = ADAPTER_TO_TRAINING_CONFIG[adapter_name]
        print(f"\nAdapter: {cfg.adapter_name} | train={cfg.train_json} | val={cfg.val_json}")
        adapter = factory.get_adapter_trainer(cfg)
        adapter.train(train_json=cfg.train_json, val_json=cfg.val_json)

    if args.train_ensemble:
        print("\n[ensemble] Ensemble training is not yet implemented.")


if __name__ == "__main__":
    main()
