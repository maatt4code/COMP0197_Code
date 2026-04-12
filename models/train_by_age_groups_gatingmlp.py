"""
train_by_age_groups_gatingmlp.py — Notebook-style age-routing classifier training
==================================================================================
Trains a frozen-encoder age classifier on top of Whisper-small pooled encoder
features, then fits a temperature scaler for calibrated routing probabilities.

The resulting checkpoint is saved to:
    weights/<best-dir>/gate_mlp/gate_mlp.pt

GenAI disclosure: Assistive tools (e.g. Cursor/LLM-based coding assistants) were used for
refactoring, documentation, and boilerplate. All changes were reviewed and tested locally.
"""

from __future__ import annotations

import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PARENT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from transformers import WhisperProcessor

from config import Config, TrainingConfig
from models.age_classifier import (
    AgeClassifierHead,
    TemperatureScaler,
    expected_calibration_error,
    load_gate_checkpoint,
    save_gate_checkpoint,
)
from models.whisper_common import (
    extract_pooled_embeddings,
    get_whisper_encoder,
    load_manifest_records,
)

__all__ = ["GatingMLPAdapter"]


GATE_HIDDEN = 512
GATE_DROPOUT = 0.3
GATE_BATCH_SIZE = 16
GATE_EPOCHS = 5
GATE_LEARNING_RATE = 1e-3
GATE_WEIGHT_DECAY = 1e-4
GATE_MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-3


def _build_train_loader(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(embeddings, labels)
    label_counts = Counter(labels.tolist())
    sample_weights = torch.tensor(
        [1.0 / label_counts[int(label)] for label in labels.tolist()],
        dtype=torch.double,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _build_eval_loader(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(embeddings, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _run_epoch(
    classifier_head: AgeClassifierHead,
    dataloader: DataLoader,
    criterion,
    device: str,
    optimizer=None,
) -> dict[str, float]:
    is_training = optimizer is not None
    classifier_head.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()
            logits = classifier_head(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                classifier_head.parameters(),
                GATE_MAX_GRAD_NORM,
            )
            optimizer.step()
        else:
            with torch.no_grad():
                logits = classifier_head(embeddings)
                loss = criterion(logits, labels)

        predictions = logits.argmax(dim=-1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
    }


def _collect_logits_and_labels(
    classifier_head: AgeClassifierHead,
    dataloader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    classifier_head.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            logits = classifier_head(embeddings)
            logits_list.append(logits.detach())
            labels_list.append(labels.detach())

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def _fit_temperature_scaler(
    classifier_head: AgeClassifierHead,
    val_loader: DataLoader,
    criterion,
    device: str,
) -> tuple[TemperatureScaler, dict[str, float]]:
    scaler = TemperatureScaler().to(device)
    val_logits, val_labels = _collect_logits_and_labels(
        classifier_head=classifier_head,
        dataloader=val_loader,
        device=device,
    )

    uncalibrated_probs = torch.softmax(val_logits, dim=-1)
    uncalibrated_nll = criterion(val_logits, val_labels).item()
    uncalibrated_ece = expected_calibration_error(uncalibrated_probs, val_labels)

    calibration_optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def calibration_closure():
        calibration_optimizer.zero_grad()
        loss = criterion(scaler(val_logits), val_labels)
        loss.backward()
        return loss

    calibration_optimizer.step(calibration_closure)

    calibrated_logits = scaler(val_logits)
    calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
    calibrated_nll = criterion(calibrated_logits, val_labels).item()
    calibrated_ece = expected_calibration_error(calibrated_probs, val_labels)

    metrics = {
        "temperature": float(scaler.temperature.detach().cpu().item()),
        "val_nll_uncalibrated": float(uncalibrated_nll),
        "val_nll_calibrated": float(calibrated_nll),
        "val_ece_uncalibrated": float(uncalibrated_ece),
        "val_ece_calibrated": float(calibrated_ece),
    }
    return scaler, metrics


class GatingMLPAdapter:
    """Train the notebook-style age classifier using the shared base Whisper encoder."""

    def __init__(
        self,
        config: Config,
        base_model,
        processor: WhisperProcessor,
        mock: bool = False,
        ta_train: bool = False,
    ):
        self.config = config
        self.device = config.device()
        self.base_model = base_model.to(self.device).eval()
        self.processor = processor
        self.mock = mock
        self.ta_train = ta_train

    def train(self, train_json: str, val_json: str) -> None:
        if self.mock:
            self._mock_load()
            return

        train_path = Config.data_dir() / train_json
        val_path = Config.data_dir() / val_json
        assert train_path.exists(), f"Train JSON not found: {train_path}"
        assert val_path.exists(), f"Val JSON not found: {val_path}"

        train_records = load_manifest_records(train_path)
        val_records = load_manifest_records(val_path)
        n_epochs = GATE_EPOCHS

        if self.ta_train:
            train_records = train_records[: TrainingConfig.TA_TRAIN_SAMPLES]
            val_records = val_records[: TrainingConfig.TA_TRAIN_SAMPLES]
            n_epochs = TrainingConfig.TA_TRAIN_EPOCHS

        audio_root = Config.audio_dir()
        encoder = get_whisper_encoder(self.base_model)
        for param in encoder.parameters():
            param.requires_grad = False

        print(f"\n{'=' * 60}")
        print(
            "  Gate classifier training"
            + (" [TA_TRAIN]" if self.ta_train else "")
        )
        print(f"  Train JSON : {train_path}  ({len(train_records)} records)")
        print(f"  Val JSON   : {val_path}  ({len(val_records)} records)")
        print(f"  Epochs     : {n_epochs}")
        print(f"  Device     : {self.device}")
        print(f"  Train dist : {Counter(r['age_bucket'] for r in train_records)}")
        print(f"  Val dist   : {Counter(r['age_bucket'] for r in val_records)}")
        print("=" * 60)

        print("\n[1/3] Extracting frozen encoder embeddings...")
        train_embeddings, train_labels = extract_pooled_embeddings(
            records=train_records,
            processor=self.processor,
            encoder=encoder,
            audio_root=audio_root,
            device=self.device,
            batch_size=GATE_BATCH_SIZE,
        )
        val_embeddings, val_labels = extract_pooled_embeddings(
            records=val_records,
            processor=self.processor,
            encoder=encoder,
            audio_root=audio_root,
            device=self.device,
            batch_size=GATE_BATCH_SIZE,
        )
        print(f"  Train embeddings: {tuple(train_embeddings.shape)}")
        print(f"  Val embeddings  : {tuple(val_embeddings.shape)}")

        train_loader = _build_train_loader(
            embeddings=train_embeddings,
            labels=train_labels,
            batch_size=GATE_BATCH_SIZE,
        )
        val_loader = _build_eval_loader(
            embeddings=val_embeddings,
            labels=val_labels,
            batch_size=GATE_BATCH_SIZE,
        )

        print("\n[2/3] Training classifier head...")
        classifier_head = AgeClassifierHead(
            d_model=Config.model_dim(),
            hidden=GATE_HIDDEN,
            n_classes=len(Config.lora_age_buckets()),
            dropout=GATE_DROPOUT,
        ).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            classifier_head.parameters(),
            lr=GATE_LEARNING_RATE,
            weight_decay=GATE_WEIGHT_DECAY,
        )

        history: list[dict] = []
        best_metric = None
        best_epoch = 0
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, n_epochs + 1):
            train_metrics = _run_epoch(
                classifier_head=classifier_head,
                dataloader=train_loader,
                criterion=criterion,
                device=self.device,
                optimizer=optimizer,
            )
            val_metrics = _run_epoch(
                classifier_head=classifier_head,
                dataloader=val_loader,
                criterion=criterion,
                device=self.device,
                optimizer=None,
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(epoch_metrics)

            monitor_value = epoch_metrics["val_loss"]
            improved = best_metric is None or monitor_value < (best_metric - EARLY_STOPPING_MIN_DELTA)
            if improved:
                best_metric = monitor_value
                best_epoch = epoch
                best_state_dict = deepcopy(classifier_head.state_dict())
                epochs_without_improvement = 0
                status = "*"
            else:
                epochs_without_improvement += 1
                status = ""

            print(
                f"  epoch {epoch:02d} | "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} {status}"
            )

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"  early stopping at epoch {epoch} "
                    f"(best val_loss={best_metric:.4f} from epoch {best_epoch})"
                )
                break

        assert best_state_dict is not None, "Classifier training did not produce a best checkpoint."
        classifier_head.load_state_dict(best_state_dict)

        print("\n[3/3] Fitting temperature scaler and saving checkpoint...")
        temperature_scaler, calibration_metrics = _fit_temperature_scaler(
            classifier_head=classifier_head,
            val_loader=val_loader,
            criterion=criterion,
            device=self.device,
        )
        print(
            "  temperature={temperature:.3f} | "
            "val NLL {val_nll_uncalibrated:.4f}->{val_nll_calibrated:.4f} | "
            "val ECE {val_ece_uncalibrated:.4f}->{val_ece_calibrated:.4f}".format(
                **calibration_metrics
            )
        )

        history.append({"calibration": calibration_metrics})

        best_dir = self.config.adapter_best_weights_path("gate_mlp")
        best_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = best_dir / "gate_mlp.pt"
        save_gate_checkpoint(
            checkpoint_path=checkpoint_path,
            classifier_head=classifier_head,
            temperature_scaler=temperature_scaler,
            age_dict={bucket: idx for idx, bucket in enumerate(Config.lora_age_buckets())},
            adapter_names=[
                Config.lora_bucket_to_adapter(bucket)
                for bucket in Config.lora_age_buckets()
            ],
            history=history,
            best_epoch=best_epoch,
            whisper_model_name=Config.model_name(),
        )
        print(f"  saved gate checkpoint -> {checkpoint_path}")

    def _mock_load(self) -> None:
        checkpoint_path = self.config.adapter_load_weights_path("gate_mlp") / "gate_mlp.pt"
        print(f"\n[mock] Gate classifier — loading checkpoint from {checkpoint_path}")
        assert checkpoint_path.exists(), (
            f"[mock] gate_mlp.pt not found at {checkpoint_path}. "
            "Run without --mock first to produce a checkpoint."
        )
        loaded = load_gate_checkpoint(checkpoint_path=checkpoint_path, device="cpu")
        classifier_head = loaded["classifier_head"]
        print(
            f"[mock] OK — d_model={classifier_head.d_model}, hidden={classifier_head.hidden}, "
            f"n_classes={classifier_head.n_classes}, temperature="
            f"{loaded['temperature_scaler'].temperature.detach().cpu().item():.3f}"
        )
