"""
🚀 DEVMIND Fine-Tuning on Google Colab
Run this script in Google Colab for GPU acceleration
"""

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================
print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    "torch",
    "transformers>=4.39.0,<4.50",
    "peft>=0.7.0",
    "datasets>=2.10.0",
    "accelerate",
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ Dependencies installed!")

# ============================================================================
# CELL 2: Setup and imports
# ============================================================================
import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

print("=" * 70)
print("🚀 PHASE 5: Fine-Tuning TinyLlama with LoRA (Colab Edition)")
print("=" * 70)

# ============================================================================
# CELL 3: Upload dataset (or use from files)
# ============================================================================
print("\n📁 Setting up dataset...")

# Option A: If files already in Colab (first upload to Colab or Google Drive)
# For now, we'll create paths assuming you upload the .jsonl files

DATA_DIR = "/content/finetuning_dataset"

# Create directory if it doesn't exist
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

# You'll upload train.jsonl, val.jsonl to Colab first, or mount Google Drive

print(f"✅ Dataset directory ready: {DATA_DIR}")

# ============================================================================
# CELL 4: Load model and tokenizer
# ============================================================================
print("\n[1/4] Loading model...")

# Colab has GPU, so we can use cuda
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_OUTPUT_DIR = "/content/tinyllama_finetuned"

print(f"⚙️ Config:")
print(f"  Model: {MODEL_ID}")
print(f"  Device: {DEVICE}")
print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded ({total_params:,} parameters)")

# ============================================================================
# CELL 5: Setup LoRA
# ============================================================================
print(f"\n[2/4] Setting up LoRA...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# CELL 6: Load and preprocess dataset
# ============================================================================
print(f"\n[3/4] Loading training data...")

dataset = load_dataset("json", data_files={
    "train": f"{DATA_DIR}/train.jsonl",
    "validation": f"{DATA_DIR}/val.jsonl",
})

def formatting_func(example):
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"
    return {"text": text}

dataset = dataset.map(formatting_func)

def preprocess_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text", "instruction", "input", "output"])
print(f"✅ Loaded {len(dataset['train'])} training samples")

# ============================================================================
# CELL 7: Train
# ============================================================================
print(f"\n[4/4] Training...")

training_args = TrainingArguments(
    output_dir=ADAPTER_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=8,  # Colab can handle larger batches!
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    remove_unused_columns=False,
    seed=42,
    label_names=["labels"],
    fp16=True,  # Use mixed precision on Colab GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

# ============================================================================
# CELL 8: Save and download
# ============================================================================
print(f"\n✅ Training complete!")
model.save_pretrained(ADAPTER_OUTPUT_DIR)
tokenizer.save_pretrained(ADAPTER_OUTPUT_DIR)

print("\n" + "=" * 70)
print("✅ FINE-TUNING COMPLETE!")
print("=" * 70)
print(f"\n📁 Model saved to: {ADAPTER_OUTPUT_DIR}")
print(f"\n📥 Download the folder from Colab Files panel")
print("=" * 70 + "\n")
