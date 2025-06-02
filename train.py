import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import DatasetDict, load_dataset
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_target_modules(model_name):
    if "deberta" in model_name:
        return ["query_proj", "value_proj"]
    elif "distilbert" in model_name:
        return ["q_lin", "v_lin"]
    else:
        return ["q_proj", "v_proj"]


class FocalLoss(nn.Module):
    """Multi‑class focal loss with optional label smoothing."""

    def __init__(self, gamma=2.0, weight=None, ignore_index=-100, smooth_eps=0.0):
        super().__init__()
        self.gamma = gamma
        self.smooth_eps = smooth_eps
        self.ce = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, targets):
        logp = self.ce(logits, targets)  # NLL per‑sample
        p_t = torch.exp(-logp)
        loss = ((1 - p_t) ** self.gamma) * logp
        if self.smooth_eps > 0:
            num_classes = logits.size(1)
            loss = (1 - self.smooth_eps) * loss + self.smooth_eps / num_classes
        return loss.mean()


class FGM:
    """Fast Gradient Method adversarial training."""

    def __init__(self, model, epsilon=1e-3):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and "embedding" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class FGTrainer(Trainer):
    """Trainer that applies one‑step FGM each training step when enabled."""

    def __init__(self, fgm=None, *args, **kwargs):
        self.fgm = fgm
        super().__init__(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        # 1) standard forward / backward  → loss already back‑prop’d by Trainer
        loss = super().training_step(*args, **kwargs)
        if self.fgm is None:
            return loss
        # 2) adversarial forward / backward
        self.fgm.attack()
        adv_loss = super().training_step(*args, **kwargs)
        self.fgm.restore()
        return 0.5 * (loss + adv_loss)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("Emotion detection training script")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="ce")
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--use_fgm", action="store_true")
    p.add_argument("--over_sampling", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def load_kaggle_dataset(path: str) -> DatasetDict:
    """Load the Kaggle "Emotion Detection from Text" dataset."""
    data = load_dataset("csv", data_files={"train": path})
    data = data.class_encode_column('sentiment')

    # Split into train/test
    data = data["train"].train_test_split(test_size=0.2, stratify_by_column='sentiment')

    # Standardise column names
    data = data.rename_column('content', "text")
    data = data.rename_column('sentiment', "label")

    # Remove unused columns
    data = data.remove_columns("tweet_id")

    return data


# -----------------------------------------------------------------------------
# Tokenization & metrics
# -----------------------------------------------------------------------------

def tokenize(ex, tok, max_len):
    return tok(ex["text"], truncation=True, max_length=max_len)

metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


# -----------------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------------

def save_confusion(preds, labels, id2label, path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=id2label, yticklabels=id2label, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_curves(log_history, path, eval_steps):
    tr_epochs, tr_loss = [], []
    val_epochs, val_loss, val_f1 = [], [], []
    for e in log_history:
        if "epoch" in e and "loss" in e and "eval_loss" not in e:
            tr_loss.append(e["loss"])
            tr_epochs.append(e["epoch"])
        if "eval_loss" in e:
            val_loss.append(e["eval_loss"])
            val_f1.append(e.get("eval_f1_macro"))
            val_epochs.append(e["epoch"])

    fig, ax1 = plt.subplots(figsize=(7,5))
    line1, = ax1.plot(tr_epochs, tr_loss, label="train_loss")
    lines = [line1]
    if val_loss:
        line2, = ax1.plot(val_epochs, val_loss, label="val_loss")
        lines.append(line2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2 = ax1.twinx()
    if val_f1 and None not in val_f1:
        line3, = ax2.plot(val_epochs, val_f1, linestyle="--", label="val_f1")
        lines.append(line3)
        ax2.set_ylabel("F1 (macro)")

    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{ts}_{Path(args.model_name).name.replace('/', '-')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    ds = load_kaggle_dataset(args.dataset_path)

    label_names = ds["train"].features["label"].names
    num_labels = len(label_names)
    counts_per_class = np.bincount(ds["train"]["label"])
    label2id = {l: i for i, l in enumerate(label_names)}
    id2label = {i: l for i, l in enumerate(label_names)}

    # ---------------------------------------------------------------- tokenise
    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = ds.map(lambda e: tokenize(e, tok, args.max_length), batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tok)

    # ---------------------------------------------------------------- model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, label2id=label2id, id2label=id2label,
    )

    if args.use_lora:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=get_target_modules(args.model_name),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",  # 序列分类
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    criterion = None
    if args.loss == "focal":
        weights_per_class = torch.tensor(counts_per_class.sum() / counts_per_class, dtype=torch.float)
        criterion = FocalLoss(args.gamma, weight=weights_per_class, smooth_eps=args.label_smoothing)
    elif args.loss == "weighted_ce":
        weights_per_class = torch.tensor(1 / np.sqrt(counts_per_class), dtype=torch.float)
        criterion = nn.CrossEntropyLoss(weight=weights_per_class)

    # ---------------------------------------------------------------- train
    train_args = TrainingArguments(
        output_dir=run_dir/"ckpt",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.max_epochs,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=["none"],
    )

    trainer_args = {
        "model": model,
        "args": train_args,
        "train_dataset": ds["train"],
        "eval_dataset": ds["test"],
        "processing_class": tok,
        "data_collator": collator,
        "compute_metrics": lambda e: compute_metrics(e),
        "callbacks": [EarlyStoppingCallback(3)],
    }
    if args.loss != 'ce':
        def compute_loss_func(out, labels, num_items_in_batch=None):
            nonlocal criterion
            criterion = criterion.to(out.logits)
            loss = criterion(out.logits, labels)
            return loss
        trainer_args["compute_loss_func"] = compute_loss_func
    if args.use_fgm:
        trainer_args["fgm"] = FGM(model)
    trainer_cls = FGTrainer if args.use_fgm else Trainer
    trainer = trainer_cls(**trainer_args)

    if args.over_sampling:
        weights_per_class = 1 / np.sqrt(counts_per_class)
        sample_weights = weights_per_class[ds["train"]["label"]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,  # 过采样
        )
        Trainer._get_train_sampler = lambda self: sampler

    trainer.train()

    # ---------------------------------------------------------------- evaluate
    metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    (run_dir/"metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir/"config.json").write_text(json.dumps(vars(args), indent=2))

    save_curves(trainer.state.log_history, run_dir/"curves.png", args.eval_steps)
    preds = np.argmax(trainer.predict(ds["test"]).predictions, axis=-1)
    save_confusion(preds, ds["test"]["label"], label_names, run_dir/"confusion_matrix.png")

    trainer.save_model(run_dir/"best_model")
    print("Final test metrics:\n", json.dumps(metrics, indent=2))
    print("Artifacts saved to", run_dir.resolve())


if __name__ == "__main__":
    main()
