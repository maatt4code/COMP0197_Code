Canonical manifests for training and evaluation live in this directory (root of data/).
train.py and test.py resolve JSON paths as data/<filename> only (see config.Config.data_dir()).
FLAC audio and noise files are not stored here — they live under the base data directory
(config.DEFAULT_BASE_DATA_DIR unless you pass --base-data-dir / --audio-dir).

TA quick eval: use data/test_ta_200.json (200 utterances, seed 42) via:
  python test.py --test-json test_ta_200.json

Regenerate the subset with:
  python data/build_ta_test_subset.py
