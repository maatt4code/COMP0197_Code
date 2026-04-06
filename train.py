from config import Config, TrainingConfig
from models import train_by_age_groups_lora
# from models import train_by_age_groups_gatingmlp
from models import train_by_unique_subjects
from pathlib import Path
from peft import PeftModel, LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent


class AdapterTrainerfactory:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device()
        self.base_model_processor, self.base_model = config.base_model()

    def get_adapter_trainer(self, config: TrainingConfig):
        adapter_name = config.adapter_name
        if adapter_name.startswith("age_"):
            return train_by_age_groups_lora.LoraAdapter(self.config, config.age_group, self.base_model, self.base_model_processor)
        # elif adapter_name == "gate_mlp":
        #     return train_by_age_groups_gatingmlp.GatingMLPAdapter(self.config, self.base_model, self.base_model_processor)
        elif adapter_name == "unique_subjects":
            return train_by_age_groups_unique_subjects.UniqueSubjectsAdapter(self.config, self.base_model, self.base_model_processor)
        else:
            raise ValueError(f"Unknown adapter name: {adapter_name}")


def main():
    config = Config(is_inference=False)
    device = config.device()
    print(f"Using device: {device}")

    base_model_processor, base_model = config.base_model()

    train_adapter_configs = [        
        TrainingConfig(
            adapter_name=f"age_{age_group.replace('-', '_')}",
            train_json= HERE / "data" / f"train_age_{age_group.replace('-', '_')}.json",
            val_json=HERE / "data" / f"val_age_{age_group.replace('-', '_')}.json",
            age_group=age_group
        ) for age_group in Config.lora_age_buckets()
    ]
    train_adapter_configs.append(
        TrainingConfig(
            adapter_name="gate_mlp",
            train_json=HERE / "data" / "train_samples.jsonl",
            val_json=HERE / "data" / "val_samples.jsonl"
        )
    )
    train_adapter_configs.append(
        TrainingConfig(
            adapter_name="unique_subjects",
            train_json=HERE / "data" / "train_samples.jsonl",
            val_json=HERE / "data" / "val_samples.jsonl"
        )
    )

    # Initialize base_model with dummy adapter weights so no-one needs to call PeftModel.from_pretrained() in the adapter trainers, which can cause OOM issues on GPU if not done carefully
    peft_model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]), adapter_name="default")
    for cfg in train_adapter_configs:
        print(f"Adapter: {cfg.adapter_name}, Train JSON: {cfg.train_json}, Val JSON: {cfg.val_json}, Age Group: {cfg.age_group}")
        adapter = AdapterTrainerfactory(config).get_adapter_trainer(cfg)
        adapter.train(peft_model=peft_model, adapter_name=cfg.adapter_name, train_json=cfg.train_json, val_json=cfg.val_json)



if __name__ == "__main__":
    main()
