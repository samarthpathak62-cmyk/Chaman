"""
Memory-optimized training script for Neuron model on Kaggle.
Uses bf16 + gradient checkpointing + smaller sequence length.
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from neuron_model import NeuronConfig, NeuronModel
from tqdm import tqdm
import os
import gc

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_streaming_dataset(tokenizer, max_length=256, buffer_size=5000):
    print("Loading FineWeb-Edu (streaming)...")
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
            return_tensors=None,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    tokenized = tokenized.shuffle(buffer_size=buffer_size, seed=42)
    return tokenized

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Memory friendly config (still ~1.7-1.8B range)
    config = NeuronConfig(
        hidden_size=2048,
        num_hidden_layers=28,          # reduced from 32
        max_position_embeddings=256,
    )
    model = NeuronModel(config).to(device)

    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.2f}B")

    tokenizer = get_tokenizer()
    dataset = get_streaming_dataset(tokenizer, max_length=256, buffer_size=5000)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Use bfloat16 if available, else float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    model.train()
    steps = 0
    max_steps = 20000
    pbar = tqdm(total=max_steps, desc="Training")

    torch.cuda.empty_cache()
    gc.collect()

    for batch in dataloader:
        if steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=True):
            _, loss = model(input_ids, attention_mask=attention_mask, labels=input_ids)

        optimizer.zero_grad(set_to_none=True)
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

        if steps % 2000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/neuron_step{steps}.pt")
            torch.cuda.empty_cache()
            print(f"\nCheckpoint saved at step {steps}")

    pbar.close()
    os.makedirs("final_model", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, "final_model/neuron.pt")
    print("Training complete.")

if __name__ == "__main__":
    train()
