"""
test.py — Evaluate base Whisper, individual adapters, weighted MoLE, and gate routing.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel

from config import Config
from metrics import wer as compute_wer
from models.age_classifier import (
    expected_calibration_error,
    load_gate_checkpoint,
    run_gate_inference,
)
from models.whisper_common import (
    get_whisper_encoder,
    load_audio,
    load_manifest_records,
    transcribe_audio,
)

HERE = Path(__file__).resolve().parent
TEMP_WEIGHTED_ADAPTER = "__mole_weighted_tmp__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Whisper adapters and gate-based routing policies.",
    )
    parser.add_argument(
        "--load-dir",
        type=str,
        default=None,
        help=(
            "Subdirectory under weights/ from which checkpoints are loaded. "
            "Defaults to 'best'."
        ),
    )
    parser.add_argument(
        "--base-data-dir",
        type=Path,
        default=None,
        help=(
            "Base dataset directory (parent of audio/ and noise/).\n"
            f"Default: {Path('/cs/student/projects3/COMP0158/grp_1/data')}"
        ),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Audio root directory (audio_path values in JSONs are relative to this).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HERE / "test_results",
        help="Directory under which timestamped evaluation outputs are written.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to the first N test records for a quick smoke test.",
    )
    return parser.parse_args()


def _resolve_adapter_method(model, method_name: str):
    if hasattr(model, method_name):
        return getattr(model, method_name)
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, method_name):
        return getattr(base_model, method_name)
    return None


def _delete_adapter_if_present(model, adapter_name: str) -> None:
    delete_adapter = _resolve_adapter_method(model, "delete_adapter")
    if delete_adapter is None:
        return
    try:
        delete_adapter(adapter_name)
    except Exception:
        pass


def transcribe_weighted_mole(
    peft_model,
    processor,
    audio,
    device: str,
    adapter_names: list[str],
    weights: list[float],
) -> str:
    add_weighted_adapter = _resolve_adapter_method(peft_model, "add_weighted_adapter")
    delete_adapter = _resolve_adapter_method(peft_model, "delete_adapter")
    if add_weighted_adapter is None or delete_adapter is None:
        raise AttributeError(
            "This PEFT installation does not expose add_weighted_adapter/delete_adapter, "
            "so weighted MoLE evaluation cannot run."
        )

    _delete_adapter_if_present(peft_model, TEMP_WEIGHTED_ADAPTER)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
    add_weighted_adapter(
        adapter_names,
        weight_tensor.tolist(),
        TEMP_WEIGHTED_ADAPTER,
        combination_type="linear",
    )

    try:
        return transcribe_audio(
            model=peft_model,
            processor=processor,
            audio=audio,
            device=device,
            adapter_name=TEMP_WEIGHTED_ADAPTER,
        )
    finally:
        set_adapter = _resolve_adapter_method(peft_model, "set_adapter")
        if set_adapter is not None:
            set_adapter(adapter_names[0])
        _delete_adapter_if_present(peft_model, TEMP_WEIGHTED_ADAPTER)


def load_peft_model_and_gate(config: Config):
    processor, base_model = config.base_model()
    device = config.device()
    adapter_names = ["age_3_4", "age_5_7", "age_8_11", "unique_subjects"]

    first_adapter = adapter_names[0]
    first_dir = config.adapter_load_weights_path(first_adapter)
    assert first_dir.exists(), f"Adapter weights not found: {first_dir}"
    peft_model = PeftModel.from_pretrained(
        base_model,
        str(first_dir),
        adapter_name=first_adapter,
    )

    for adapter_name in adapter_names[1:]:
        adapter_dir = config.adapter_load_weights_path(adapter_name)
        assert adapter_dir.exists(), f"Adapter weights not found: {adapter_dir}"
        peft_model.load_adapter(str(adapter_dir), adapter_name=adapter_name)

    peft_model = peft_model.to(device).eval()
    gate_path = config.adapter_load_weights_path("gate_mlp") / "gate_mlp.pt"
    assert gate_path.exists(), f"Gate checkpoint not found: {gate_path}"
    gate_bundle = load_gate_checkpoint(checkpoint_path=gate_path, device=device)
    return processor, peft_model, gate_bundle


def compute_summary_rows(records: list[dict], predictions_by_mode: dict[str, list[str]]) -> list[dict]:
    rows: list[dict] = []
    splits = ["overall", *Config.lora_age_buckets()]

    for mode, hypotheses in predictions_by_mode.items():
        for split in splits:
            if split == "overall":
                split_records = records
                split_hypotheses = hypotheses
            else:
                indices = [idx for idx, record in enumerate(records) if record["age_bucket"] == split]
                split_records = [records[idx] for idx in indices]
                split_hypotheses = [hypotheses[idx] for idx in indices]

            references = [record["orthographic_text"] for record in split_records]
            wer_value = compute_wer(references, split_hypotheses) if split_records else float("nan")
            rows.append(
                {
                    "mode": mode,
                    "split": split,
                    "n": len(split_records),
                    "wer": wer_value,
                }
            )

    return rows


def compute_classifier_rows(
    records: list[dict],
    calibrated_logits: torch.Tensor,
    labels: torch.Tensor,
    age_buckets: list[str],
) -> list[dict]:
    criterion = torch.nn.CrossEntropyLoss()
    probs = torch.softmax(calibrated_logits, dim=-1)
    predictions = probs.argmax(dim=-1)
    splits = ["overall", *age_buckets]
    rows: list[dict] = []

    for split in splits:
        if split == "overall":
            split_mask = torch.ones(len(records), dtype=torch.bool)
        else:
            split_mask = torch.tensor(
                [record["age_bucket"] == split for record in records],
                dtype=torch.bool,
            )

        split_logits = calibrated_logits[split_mask]
        split_probs = probs[split_mask]
        split_labels = labels[split_mask]
        split_predictions = predictions[split_mask]
        if split_labels.numel() == 0:
            continue

        accuracy = (split_predictions == split_labels).float().mean().item()
        nll = criterion(split_logits, split_labels).item()
        ece = expected_calibration_error(split_probs, split_labels)

        rows.append(
            {
                "split": split,
                "n": int(split_labels.numel()),
                "accuracy": float(accuracy),
                "nll": float(nll),
                "ece": float(ece),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def print_summary_table(rows: list[dict]) -> None:
    print("\nASR Results")
    header = f"{'mode':<16} {'split':<8} {'n':>6} {'wer':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<16} {row['split']:<8} {row['n']:>6} {row['wer']:>10.4f}"
        )


def main() -> None:
    args = parse_args()

    if args.load_dir is not None:
        Config.set_load_dir(args.load_dir)
    if args.base_data_dir is not None:
        Config.set_base_data_dir(args.base_data_dir)
    if args.audio_dir is not None:
        Config.set_audio_dir(args.audio_dir)

    config = Config(is_inference=False)
    device = config.device()
    audio_root = Config.audio_dir()
    test_manifest = Config.data_dir() / "test_by_child_id.json"
    records = load_manifest_records(test_manifest)
    if args.max_samples is not None:
        assert args.max_samples > 0, "--max-samples must be a positive integer."
        records = records[: args.max_samples]

    print(f"Device     : {device}")
    print(f"Audio dir  : {audio_root}")
    print(f"Load dir   : {args.load_dir or 'best'}")
    print(f"Test JSON  : {test_manifest}")
    print(f"Utterances : {len(records)}")
    if args.max_samples is not None:
        print(f"Max samples: {args.max_samples}")

    processor, peft_model, gate_bundle = load_peft_model_and_gate(config)
    encoder = get_whisper_encoder(peft_model)

    if gate_bundle["whisper_model_name"] != Config.model_name():
        print(
            f"Warning: gate checkpoint expects {gate_bundle['whisper_model_name']}, "
            f"but this project loads {Config.model_name()}."
        )

    mode_predictions: dict[str, list[str]] = {
        "base": [],
        "age_3_4": [],
        "age_5_7": [],
        "age_8_11": [],
        "unique_subjects": [],
        "mole_weighted": [],
        "gated_router": [],
    }
    prediction_rows: list[dict] = []
    all_gate_logits: list[torch.Tensor] = []
    all_gate_labels: list[int] = []

    for idx, record in enumerate(records, start=1):
        audio = load_audio(audio_root / record["audio_path"])
        gate_outputs = run_gate_inference(
            inputs=audio,
            encoder=encoder,
            processor=processor,
            classifier_head=gate_bundle["classifier_head"],
            temperature_scaler=gate_bundle["temperature_scaler"],
            device=device,
            age_buckets=gate_bundle["age_buckets"],
            adapter_model=peft_model,
        )

        probs = gate_outputs["probs"][0]
        calibrated_logits = gate_outputs["calibrated_logits"][0]
        predicted_idx = int(probs.argmax().item())
        routed_bucket = gate_bundle["age_buckets"][predicted_idx]
        routed_adapter = gate_bundle["adapter_names"][predicted_idx]
        gate_prob_dict = {
            bucket: float(probs[bucket_idx].item())
            for bucket_idx, bucket in enumerate(gate_bundle["age_buckets"])
        }

        utterance_predictions = {
            "base": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name=None,
            ),
            "age_3_4": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name="age_3_4",
            ),
            "age_5_7": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name="age_5_7",
            ),
            "age_8_11": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name="age_8_11",
            ),
            "unique_subjects": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name="unique_subjects",
            ),
            "mole_weighted": transcribe_weighted_mole(
                peft_model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_names=gate_bundle["adapter_names"],
                weights=[gate_prob_dict[bucket] for bucket in gate_bundle["age_buckets"]],
            ),
            "gated_router": transcribe_audio(
                model=peft_model,
                processor=processor,
                audio=audio,
                device=device,
                adapter_name=routed_adapter,
            ),
        }

        for mode_name, hypothesis in utterance_predictions.items():
            mode_predictions[mode_name].append(hypothesis)

        all_gate_logits.append(calibrated_logits)
        all_gate_labels.append(gate_bundle["age_dict"][record["age_bucket"]])
        prediction_rows.append(
            {
                "utterance_id": record.get("utterance_id", ""),
                "child_id": record.get("child_id", ""),
                "audio_path": record["audio_path"],
                "age_bucket": record["age_bucket"],
                "reference": record["orthographic_text"],
                "gate_prediction": routed_bucket,
                "routed_adapter": routed_adapter,
                "gate_probs": gate_prob_dict,
                "transcriptions": utterance_predictions,
            }
        )

        if idx % 25 == 0 or idx == len(records):
            print(f"Processed {idx:>5} / {len(records)} utterances")

    calibrated_logits = torch.stack(all_gate_logits, dim=0)
    gate_labels = torch.tensor(all_gate_labels, dtype=torch.long)

    summary_rows = compute_summary_rows(records, mode_predictions)
    classifier_rows = compute_classifier_rows(
        records=records,
        calibrated_logits=calibrated_logits,
        labels=gate_labels,
        age_buckets=gate_bundle["age_buckets"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        results_dir / "summary.csv",
        summary_rows,
        fieldnames=["mode", "split", "n", "wer"],
    )
    write_csv(
        results_dir / "classifier_metrics.csv",
        classifier_rows,
        fieldnames=["split", "n", "accuracy", "nll", "ece"],
    )
    write_jsonl(results_dir / "predictions.jsonl", prediction_rows)

    metrics_payload = {
        "generated_at": timestamp,
        "load_dir": args.load_dir or "best",
        "summary": summary_rows,
        "classifier_metrics": classifier_rows,
    }
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print_summary_table(summary_rows)
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
