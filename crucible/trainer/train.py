import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from crucible.trainer.dataset import InteractionDataset
from crucible.tools.mlflow_utils import log_mlflow_run
import json
import hashlib


def get_git_sha():
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="distilgpt2")
    parser.add_argument("--out_dir", default="models/latest")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--log_path", default="logs/interaction_logs.jsonl")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    dataset = InteractionDataset(args.log_path, args.checkpoint, args.max_len)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if args.mixed_precision and torch.cuda.is_available()
        else None
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            inputs = torch.tensor(batch).to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                if (i + 1) % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
            else:
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss
                loss.backward()
                if (i + 1) % args.accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.out_dir, f"epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            meta = {
                "git_sha": get_git_sha(),
                "dataset_hash": hash_file(args.log_path),
                "hyperparams": vars(args),
                "metrics": {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                },
            }
            with open(os.path.join(save_path, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            log_mlflow_run(meta, save_path)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            inputs = torch.tensor(batch).to(device)
            outputs = model(inputs, labels=inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


if __name__ == "__main__":
    main()
