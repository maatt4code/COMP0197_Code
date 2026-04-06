from config import Config
import os
import torch

__all__ = ["LoraAdapter"]

class LoraAdapter:
    def __init__(self, config: Config, age_bucket: str, base_model, processor):
        self.config = config
        self.device = config.device()
        self.processor = processor
        self.age_bucket = age_bucket
        self.base_model = base_model.to(self.device).eval()
        assert self.age_bucket in Config.lora_age_buckets(), f"Age bucket {self.age_bucket} not found in {Config.lora_age_buckets()}"

    def train(self, peft_model, adapter_name, train_json, val_json):
        weights_path = self.config.adapter_weights_path(self.age_bucket)
        assert os.path.exists(weights_path), f"Adapter weights file not found: {weights_path}"
        assert os.path.exists(train_json), f"Train JSON file not found: {train_json}"
        assert os.path.exists(val_json), f"Val JSON file not found: {val_json}"
        try:
            adapter_weights = torch.load(weights_path, map_location=self.device)
            print(f"Loaded adapter weights for age bucket {self.age_bucket} from {weights_path}")
        except Exception as e:
            print(f"Error loading adapter weights for age bucket {self.age_bucket}: {e}")
            raise e
        peft_model.load_adapter(
            weights_path,
            adapter_name=adapter_name
        )
        peft_model.set_adapter(adapter_name)
        peft_model.delete_adapter(adapter_name)