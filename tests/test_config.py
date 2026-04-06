import sys
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

from config import Config

# Create an instance of the Config class (requires a config_file argument)
config_instance = Config()

# 1. Print instance attributes and their values
print("--- Instance Attributes ---")
for attr, value in vars(config_instance).items():
    print(f"{attr}: {value}")

# 2. Print class attributes and their values (filtering out Python's built-in dunder methods)
print("\n--- Class Attributes ---")
for attr, value in vars(Config).items():
    if not attr.startswith('__'):
        print(f"{attr}: {value}")

print(Config.device())
print(Config.device(is_inference=False))

AGE_BUCKETS   = ["3-4", "5-7", "8-11"]
ADAPTER_NAMES = [name for name in Config.ADAPTER_NAMES if name.startswith("age_")]
ADAPTER_DIRS = {k: v
                for k, v in zip(
                    AGE_BUCKETS,
                    [PARENT_DIR / Config.ADAPTER_WEIGHTS_DIR / f"{Config.ADAPTER_WEIGHTS_FILE_PREFIX}_{name}" for name in ADAPTER_NAMES]
                    )
                }
print(ADAPTER_DIRS)

