"""
Training script for ~2B Neuron model.
Uses FineWeb-Edu with streaming + high buffer.
Kaggle pe 2B tight hai — batch_size=1 + checkpointing use karna.
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
    # Best current educational web dataset
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        split="train",
        streaming=True,
    )

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    # High buffer for better shuffling in streaming
    tokenized = tokenized.shuffle(buffer_size=buffer_size, seed=42)
    return tokenized

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = NeuronConfig()  # already ~2B
    model = NeuronModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.2f}B")

    tokenizer = get_tokenizer()
    dataset = get_streaming_dataset(tokenizer, max_length=512, buffer_size=2000000)

    # Streaming dataset needs iterable dataloader
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    model.train()
    steps = 0
    max_steps = 50000  # adjust based on how much data you want to see
    pbar = tqdm(total=max_steps, desc="Training")

    for batch in dataloader:
        if steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
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
        pbar.set_postfix(loss=f"{loss.item():.4f}", step=steps)

        # Optional: save checkpoint every 5000 steps
        if steps % 5000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/neuron_2b_step{steps}.pt")

    pbar.close()
    os.makedirs("final_model", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, "final_model/neuron_2b.pt")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
