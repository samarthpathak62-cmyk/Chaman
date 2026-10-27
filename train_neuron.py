"""
Fixed training script for ~2B Neuron model.
Streaming FineWeb-Edu + proper tensor handling.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from neuron_model import NeuronConfig, NeuronModel
from tqdm import tqdm
import os

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_streaming_dataset(tokenizer, max_length=512, buffer_size=10000):
    print("Loading FineWeb-Edu (streaming)...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    def tokenize(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    tokenized = tokenized.shuffle(buffer_size=buffer_size, seed=42)
    return tokenized

def collate_fn(batch):
    """Convert list of dicts into proper tensors."""
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = NeuronConfig()
    model = NeuronModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.2f}B")

    tokenizer = get_tokenizer()
    dataset = get_streaming_dataset(tokenizer, max_length=512, buffer_size=20000)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model.train()
    steps = 0
    max_steps = 50000
    pbar = tqdm(total=max_steps, desc="Training")

    for batch in dataloader:
        if steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            _, loss = model(input_ids, attention_mask=attention_mask, labels=input_ids)

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        steps += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if steps % 5000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/neuron_2b_step{steps}.pt")
            print(f"\nCheckpoint saved at step {steps}")

    pbar.close()
    os.makedirs("final_model", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, "final_model/neuron_2b.pt")
    print("Training complete. Model saved to final_model/neuron_2b.pt")

if __name__ == "__main__":
    train()
