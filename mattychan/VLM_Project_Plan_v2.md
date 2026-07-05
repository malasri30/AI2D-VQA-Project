# VLM Fine-Tuning on AI2D — Full Project Plan
**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop  
**Dataset:** AI2D (4,903 images, 15,501 questions, 17 categories)  
**Framework:** PyTorch + HuggingFace Transformers + PEFT (LoRA) + TRL

---

## Overview

The project runs in **two phases**:

**Phase 1** — Evaluate all 3 base VLMs zero-shot on AI2D. No training, just inference. Pick the best performing model.

**Phase 2** — Fine-tune only the winning model under 2 conditions (Tuning A and Tuning B), then compare all three results: base vs A vs B.

This design is scientifically cleaner — you don't waste GPU time fine-tuning a weak base, and the A vs B ablation is more meaningful on a model that's already demonstrated diagram understanding. We also visualize attention maps at inference and during training.

### Phase 1 — Base Model Comparison (all 3, zero-shot eval only)

| Model | Zero-shot Eval | Fine-tune? |
|---|---|---|
| Qwen2-VL-2B | ✓ | only if winner |
| LLaVA-1.5-7B | ✓ | only if winner |
| InternVL2-2B | ✓ | only if winner |

→ **Winner selected by accuracy on test set**

### Phase 2 — Fine-Tuning the Winner (1×3 ablation)

| Condition | Input to model | Purpose |
|---|---|---|
| Base | — | Baseline reference (from Phase 1) |
| Tuning A | Image + Question | Does fine-tuning on AI2D help? |
| Tuning B | Image + Question + Annotation | Does annotation context add further value? |

> **Why these models?**  
> - `Qwen2-VL-2B` — lightweight, fast, strong multimodal encoder, fits comfortably in 6GB with 4-bit  
> - `LLaVA-1.5-7B` — the reference architecture for diagram QA in literature, 4-bit fits in 6GB  
> - `InternVL2-2B` — strong on visual reasoning, very memory efficient, good baseline contrast  
>
> **Why not 7B models across the board?** At 4-bit, a 7B model uses ~4.5GB leaving ~1.5GB for activations/gradients — tight but doable with batch_size=1 + gradient checkpointing. 2B models give more headroom for larger effective batch sizes via gradient accumulation.



---

## Directory Structure

```
VLM_Final_Project/
├── ai2d/                          # Original dataset (read-only)
│   ├── images/
│   ├── annotations/
│   ├── questions/
│   └── categories.json
│
├── metadata_output/               # From your metadata script
│   ├── metadata_images.csv
│   ├── metadata_questions.csv
│   ├── metadata_integrity.csv
│   └── metadata_full.json
│
├── processed_data/                # ← Created in Step 2
│   ├── splits/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── annotation_summaries/      # Pre-processed annotation text per image
│
├── checkpoints/                   # ← Fine-tune weights (winner model only)
│   ├── <winner>_tuningA/          # e.g. qwen2vl_2b_tuningA
│   └── <winner>_tuningB/          # e.g. qwen2vl_2b_tuningB
│
├── logs/                          # ← Training logs
│   ├── qwen2vl_2b_tuningA/
│   ├── ...
│   └── training_summary.csv       # Aggregated loss curves all models
│
├── results/                       # ← Evaluation results
│   ├── predictions/               # Per-model per-condition predictions
│   └── comparison_report.csv      # Final 3×3 accuracy table
│
├── attention_maps/                # ← Saved attention visualizations
│
└── notebooks/
    ├── 00_environment_setup.ipynb
    ├── 01_eda_and_data_analysis.ipynb
    ├── 02_data_preprocessing.ipynb
    ├── 03_model1_qwen2vl.ipynb
    ├── 04_model2_llava.ipynb
    ├── 05_model3_internvl2.ipynb
    └── 06_comparison_and_analysis.ipynb
```

---

## Notebook 00 — Environment Setup

```python
# Cell 1 — Install all dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install transformers>=4.45.0
!pip install bitsandbytes peft trl
!pip install accelerate datasets
!pip install qwen-vl-utils          # for Qwen2-VL image processing
!pip install sentencepiece          # for LLaVA tokenizer
!pip install einops timm            # for InternVL2
!pip install matplotlib seaborn pandas numpy pillow
!pip install bertviz                # attention visualization

# Cell 2 — GPU check
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 3 — Create directory structure
import os
for d in [
    "processed_data/splits",
    "processed_data/annotation_summaries",
    "checkpoints",
    "logs",
    "results/predictions",
    "attention_maps"
]:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")
```

---

## Notebook 01 — EDA and Data Analysis

### What to cover
- Category distribution bar chart (imbalanced — partsOfA dominates)
- Questions per image histogram
- abcLabel vs descriptive split per category
- Correct answer index distribution (uniformity check)
- Annotation richness per category (avg blobs, relationships, text nodes)
- Sample image viewer: show image + its annotation elements + questions side by side

```python
# Key cell — category + question breakdown
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_img = pd.read_csv("metadata_output/metadata_images.csv")
df_q   = pd.read_csv("metadata_output/metadata_questions.csv")

# Images per category
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
df_img["category"].value_counts().plot(kind="barh", ax=axes[0], title="Images per Category")

# abcLabel breakdown per category
abc_by_cat = df_q.merge(df_img[["image_name","category"]], on="image_name")
abc_by_cat.groupby(["category","abc_label"]).size().unstack().plot(
    kind="bar", ax=axes[1], title="abcLabel vs Descriptive Questions per Category"
)
plt.tight_layout()
plt.savefig("results/eda_overview.png", dpi=150)
plt.show()
```

---

## Notebook 02 — Data Preprocessing

This is the most important notebook. Two dataset versions are produced:

### Version A — Image + Question only
Standard VQA format. Each sample is:
```
Image: <diagram.png>
Question: What is A in the diagram?
A) ears   B) nose   C) mouth   D) face
Answer: D) face
```

### Version B — Image + Question + Annotation context
Annotation text is injected into the prompt as structured context:
```
Image: <diagram.png>
Diagram elements: HAIR(E) → face blob | EYES(D) → face blob | NOSE(F) → face blob | ...
Relationships: intraObjectLinkage ×5, imageTitle ×1
Question: What is A in the diagram?
A) ears   B) nose   C) mouth   D) face
Answer: D) face
```

### Stratified Split Strategy
Given the severe category imbalance, split **within each category**:
- Train: 70% | Val: 15% | Test: 15%
- Filter out images with 0 questions before splitting
- Save image names per split so all 3 models use identical splits

```python
import json, os, random
import pandas as pd
from collections import defaultdict

random.seed(42)

df_img = pd.read_csv("metadata_output/metadata_images.csv")
df_q   = pd.read_csv("metadata_output/metadata_questions.csv")
with open("ai2d/categories.json") as f:
    categories = json.load(f)

# Only keep images that have at least 1 question
answerable = df_img[df_img["q_question_count"] > 0]["image_name"].tolist()
print(f"Answerable images: {len(answerable)} / {len(df_img)}")

# Stratified split
train_imgs, val_imgs, test_imgs = [], [], []
by_category = defaultdict(list)
for img in answerable:
    by_category[categories.get(img, "other")].append(img)

for cat, imgs in by_category.items():
    random.shuffle(imgs)
    n = len(imgs)
    n_val  = max(1, int(n * 0.15))
    n_test = max(1, int(n * 0.15))
    test_imgs  += imgs[:n_test]
    val_imgs   += imgs[n_test:n_test+n_val]
    train_imgs += imgs[n_test+n_val:]

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# ── Build samples ──
def build_samples(image_names, version="A"):
    """
    version A: image + question only
    version B: image + question + annotation context
    """
    samples = []
    ann_dir = "ai2d/annotations"
    
    for img_name in image_names:
        q_rows = df_q[df_q["image_name"] == img_name]
        
        ann_context = ""
        if version == "B":
            ann_path = os.path.join(ann_dir, img_name + ".json")
            if os.path.exists(ann_path):
                with open(ann_path) as f:
                    ann = json.load(f)
                
                # Build readable annotation summary
                labels = []
                for tid, t in ann.get("text", {}).items():
                    val  = t.get("value", "").strip()
                    repl = t.get("replacementText", "").strip()
                    if val:
                        labels.append(f"{val}({repl})" if repl else val)
                
                rel_cats = defaultdict(int)
                for rel in ann.get("relationships", {}).values():
                    rel_cats[rel.get("category","unknown")] += 1
                
                rel_str = " | ".join(f"{k}×{v}" for k,v in rel_cats.items())
                ann_context = f"Diagram labels: {', '.join(labels)}\nRelationships: {rel_str}\n"
        
        for _, q_row in q_rows.iterrows():
            answers = q_row["answer_texts"].split(" | ")
            options = ["A","B","C","D"]
            correct_idx = int(q_row["correct_answer_index"])
            
            # Format as multiple choice
            choices_str = "  ".join(
                f"{options[i]}) {ans}" for i, ans in enumerate(answers)
            )
            correct_letter = options[correct_idx]
            correct_text   = answers[correct_idx]
            
            question_str = q_row["question_text"]
            if q_row["abc_label"]:
                question_str = "[Label Question] " + question_str
            
            user_text = f"{ann_context}Question: {question_str}\n{choices_str}"
            
            samples.append({
                "image_name":    img_name,
                "image_path":    f"ai2d/images/{img_name}",
                "category":      categories.get(img_name, "other"),
                "question_id":   q_row["question_id"],
                "abc_label":     q_row["abc_label"],
                "user_text":     user_text,
                "answer":        f"{correct_letter}) {correct_text}",
                "correct_index": correct_idx,
                "version":       version,
            })
    
    return samples

# Build both versions
for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
    for ver in ["A", "B"]:
        samples = build_samples(split_imgs, version=ver)
        out_path = f"processed_data/splits/{split_name}_v{ver}.json"
        with open(out_path, "w") as f:
            json.dump(samples, f, indent=2)
        print(f"✓ {out_path}  ({len(samples)} samples)")
```

---

## Notebooks 03–05 — Per-Model Training

Each model notebook follows the **same structure** so results are comparable. Here is the full template (substitute model config per notebook):

### Model Configs

```python
# Notebook 03 — Qwen2-VL-2B
MODEL_CONFIG = {
    "name":     "qwen2vl_2b",
    "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "type":     "qwen2vl",
    "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

# Notebook 04 — LLaVA-1.5-7B
MODEL_CONFIG = {
    "name":     "llava_7b",
    "model_id": "llava-hf/llava-1.5-7b-hf",
    "type":     "llava",
    "lora_targets": ["q_proj", "v_proj"],
}

# Notebook 05 — InternVL2-2B
MODEL_CONFIG = {
    "name":     "internvl2_2b",
    "model_id": "OpenGVLab/InternVL2-2B",
    "type":     "internvl2",
    "lora_targets": ["q_proj", "v_proj", "k_proj"],
}
```

### RTX 3050 6GB Hyperparameters

```python
# ── GPU-optimised for 6GB VRAM ──
HW_CONFIG = {
    "batch_size":              1,       # VRAM constraint — never go higher
    "gradient_accumulation":   8,       # effective batch = 8, simulates larger batches
    "max_seq_len":             256,     # diagram questions are short
    "lora_r":                  8,       # LoRA rank — low keeps VRAM down
    "lora_alpha":              16,
    "lora_dropout":            0.05,
    "learning_rate":           2e-4,    # higher LR compensates for small effective batch
    "warmup_ratio":            0.03,
    "epochs":                  3,
    "optim":                   "paged_adamw_8bit",   # 8-bit saves ~500MB vs 32-bit
    "gradient_checkpointing":  True,    # trades compute for VRAM
    "fp16":                    False,
    "bf16":                    True,    # RTX 3050 supports bf16
    "dataloader_num_workers":  2,
    "logging_steps":           10,
    "eval_steps":              50,
    "save_steps":              50,
    "save_total_limit":        2,       # keep only last 2 checkpoints to save disk
}
```

### Full Training Cell Template

```python
import os, json, gc, time, torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor, AutoTokenizer,
    BitsAndBytesConfig, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# ── Paths ──
CKPT_DIR = f"checkpoints/{MODEL_CONFIG['name']}_tuning{TUNING_VERSION}"
LOG_DIR  = f"logs/{MODEL_CONFIG['name']}_tuning{TUNING_VERSION}"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

# ── Load data ──
with open(f"processed_data/splits/train_v{TUNING_VERSION}.json") as f:
    train_data = json.load(f)
with open(f"processed_data/splits/val_v{TUNING_VERSION}.json") as f:
    val_data = json.load(f)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# ── System prompt ──
SYSTEM_PROMPT = (
    "You are a VLM specialized in scientific diagram understanding. "
    "Answer multiple-choice questions about diagrams by selecting the correct option letter and text. "
    "Respond ONLY in this format: X) answer text"
)

def format_sample(sample):
    return [
        {"role": "system",    "content": [{"type": "text",  "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [
            {"type": "image", "image": Image.open(sample["image_path"]).convert("RGB")},
            {"type": "text",  "text": sample["user_text"]},
        ]},
        {"role": "assistant", "content": [{"type": "text",  "text": sample["answer"]}]},
    ]

train_dataset = [format_sample(s) for s in train_data]
val_dataset   = [format_sample(s) for s in val_data]

# ── Load model (4-bit quantised) ──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# NOTE: model loading call differs per architecture — see per-model section below
model     = load_model(MODEL_CONFIG, bnb_config)   # defined per notebook
processor = load_processor(MODEL_CONFIG)

# ── LoRA ──
peft_config = LoraConfig(
    r=HW_CONFIG["lora_r"],
    lora_alpha=HW_CONFIG["lora_alpha"],
    lora_dropout=HW_CONFIG["lora_dropout"],
    bias="none",
    target_modules=MODEL_CONFIG["lora_targets"],
    task_type="CAUSAL_LM",
)
print(f"Trainable params before LoRA: {model.num_parameters():,}")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── Custom logging callback ──
class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        self.records  = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            record = {"step": state.global_step, **logs}
            self.records.append(record)
            # Write incrementally so logs survive crashes
            pd.DataFrame(self.records).to_csv(self.log_path, index=False)

log_callback = LoggingCallback(f"{LOG_DIR}/training_log.csv")

# ── Collate function ──
def collate_fn(examples):
    texts = [processor.apply_chat_template(ex, tokenize=False) for ex in examples]
    images = [ex[1]["content"][0]["image"] for ex in examples]
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    return batch

# ── Training args ──
training_args = SFTConfig(
    output_dir=CKPT_DIR,
    num_train_epochs=HW_CONFIG["epochs"],
    per_device_train_batch_size=HW_CONFIG["batch_size"],
    per_device_eval_batch_size=HW_CONFIG["batch_size"],
    gradient_accumulation_steps=HW_CONFIG["gradient_accumulation"],
    gradient_checkpointing=HW_CONFIG["gradient_checkpointing"],
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=HW_CONFIG["learning_rate"],
    warmup_ratio=HW_CONFIG["warmup_ratio"],
    bf16=HW_CONFIG["bf16"],
    optim=HW_CONFIG["optim"],
    logging_steps=HW_CONFIG["logging_steps"],
    eval_steps=HW_CONFIG["eval_steps"],
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=HW_CONFIG["save_steps"],
    save_total_limit=HW_CONFIG["save_total_limit"],
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    max_grad_norm=1.0,
    dataset_kwargs={"skip_prepare_dataset": True},
    max_seq_length=HW_CONFIG["max_seq_len"],
    remove_unused_columns=False,
    dataloader_num_workers=HW_CONFIG["dataloader_num_workers"],
    report_to="none",                      # no wandb
    logging_dir=LOG_DIR,
)

# ── Trainer ──
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    processing_class=processor.tokenizer,
    callbacks=[log_callback],
)

# ── Baseline eval before training ──
print("=== Baseline evaluation (before fine-tuning) ===")
baseline_metrics = trainer.evaluate()
print(baseline_metrics)
pd.DataFrame([baseline_metrics]).to_csv(f"{LOG_DIR}/baseline_metrics.csv", index=False)

# ── Train ──
print(f"\n=== Training {MODEL_CONFIG['name']} — Tuning {TUNING_VERSION} ===")
trainer.train()

# ── Save final model ──
trainer.save_model(CKPT_DIR)
processor.save_pretrained(CKPT_DIR)
print(f"✓ Saved to {CKPT_DIR}")

# ── Plot loss curve ──
log_df = pd.read_csv(f"{LOG_DIR}/training_log.csv")
fig, ax = plt.subplots(figsize=(10, 4))
if "loss" in log_df.columns:
    ax.plot(log_df["step"], log_df["loss"], label="Train Loss")
if "eval_loss" in log_df.columns:
    eval_df = log_df.dropna(subset=["eval_loss"])
    ax.plot(eval_df["step"], eval_df["eval_loss"], label="Val Loss", linestyle="--")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title(f"{MODEL_CONFIG['name']} — Tuning {TUNING_VERSION} Loss Curve")
ax.legend()
plt.tight_layout()
plt.savefig(f"{LOG_DIR}/loss_curve.png", dpi=150)
plt.show()

# ── Clear VRAM between runs ──
def clear_memory():
    for var in ["model", "trainer", "peft_model", "bnb_config", "processor"]:
        if var in globals():
            del globals()[var]
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")

clear_memory()
```

---

## Evaluation

### Phase 1 — Base Model Comparison

```python
results = []

MODELS = [
    {"name": "qwen2vl_2b",   "model_id": "Qwen/Qwen2-VL-2B-Instruct",  "type": "qwen2vl"},
    {"name": "llava_7b",     "model_id": "llava-hf/llava-1.5-7b-hf",    "type": "llava"},
    {"name": "internvl2_2b", "model_id": "OpenGVLab/InternVL2-2B",       "type": "internvl2"},
]

# Use version A test split for base eval (no annotation context)
with open("processed_data/splits/test_vA.json") as f:
    test_data = json.load(f)

for model_cfg in MODELS:
    model, processor = load_base_model(model_cfg)   # 4-bit, no LoRA
    
    correct, preds = 0, []
    for sample in test_data:
        pred        = run_inference(model, processor, sample)
        gt          = sample["answer"][0]
        pred_letter = pred.strip()[0].upper() if pred.strip() else "?"
        is_correct  = pred_letter == gt
        correct    += is_correct
        preds.append({
            "question_id":  sample["question_id"],
            "category":     sample["category"],
            "abc_label":    sample["abc_label"],
            "prediction":   pred,
            "ground_truth": sample["answer"],
            "correct":      is_correct,
        })
    
    accuracy = correct / len(test_data) * 100
    print(f"{model_cfg['name']} | Base | Acc: {accuracy:.1f}%")
    pd.DataFrame(preds).to_csv(
        f"results/predictions/{model_cfg['name']}_base.csv", index=False
    )
    results.append({"model": model_cfg["name"], "condition": "base", "accuracy": accuracy})
    clear_memory()

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
print(results_df)
WINNER = results_df.iloc[0]["model"]
print(f"\n→ Winner: {WINNER}")
results_df.to_csv("results/phase1_base_comparison.csv", index=False)
```

### Phase 2 — Fine-Tuned Evaluation (winner only)

```python
winner_cfg = next(m for m in MODELS if m["name"] == WINNER)

for condition in ["A", "B"]:
    ckpt = f"checkpoints/{WINNER}_tuning{condition}"
    model, processor = load_finetuned_model(winner_cfg, ckpt)
    
    test_ver = condition   # A uses vA split, B uses vB split
    with open(f"processed_data/splits/test_v{test_ver}.json") as f:
        test_data = json.load(f)
    
    correct, preds = 0, []
    for sample in test_data:
        pred        = run_inference(model, processor, sample)
        gt          = sample["answer"][0]
        pred_letter = pred.strip()[0].upper() if pred.strip() else "?"
        is_correct  = pred_letter == gt
        correct    += is_correct
        preds.append({
            "question_id":  sample["question_id"],
            "category":     sample["category"],
            "abc_label":    sample["abc_label"],
            "prediction":   pred,
            "ground_truth": sample["answer"],
            "correct":      is_correct,
        })
    
    accuracy = correct / len(test_data) * 100
    print(f"{WINNER} | Tuning {condition} | Acc: {accuracy:.1f}%")
    pd.DataFrame(preds).to_csv(
        f"results/predictions/{WINNER}_tuning{condition}.csv", index=False
    )
    results.append({"model": WINNER, "condition": f"Tuning{condition}", "accuracy": accuracy})
    clear_memory()

pd.DataFrame(results).to_csv("results/comparison_report.csv", index=False)
```

---

## Attention Visualization

Two modes: **inference-time** (what the model looks at when answering) and **training-time** (how attention evolves).

### Inference Attention Maps

```python
# Works for Qwen2-VL and LLaVA (both use standard HF attention outputs)
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def visualize_attention(model, processor, sample, layer=-1, head=0, save_path=None):
    """
    Overlays attention from visual tokens onto the original image.
    layer: which transformer layer (-1 = last)
    head:  which attention head
    """
    image = Image.open(sample["image_path"]).convert("RGB")
    
    # Prepare inputs with output_attentions=True
    text = processor.apply_chat_template(
        format_sample(sample)[0:2], tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )
    
    # outputs.attentions: tuple of (batch, heads, seq_len, seq_len) per layer
    attn = outputs.attentions[layer]          # shape: (1, heads, seq, seq)
    attn = attn[0, head].cpu().float()        # shape: (seq, seq)
    
    # Find visual token positions
    # For Qwen2-VL, image tokens are between <|vision_start|> and <|vision_end|>
    input_ids = inputs["input_ids"][0]
    # Average attention from last generated token to all visual tokens
    # (last row of attention matrix = where the last token attends to)
    last_token_attn = attn[-1]               # shape: (seq_len,)
    
    # Get image token positions (model-specific — adjust per architecture)
    img_token_mask = get_image_token_mask(input_ids, processor)
    visual_attn    = last_token_attn[img_token_mask]
    
    # Reshape to 2D grid (approximate square for patch-based ViTs)
    n_patches  = visual_attn.shape[0]
    grid_size  = int(n_patches ** 0.5)
    attn_grid  = visual_attn[:grid_size*grid_size].reshape(grid_size, grid_size).numpy()
    
    # Normalize and resize to image size
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    attn_resized = np.array(Image.fromarray((attn_grid * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )) / 255.0
    
    # Plot overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image);                   axes[0].set_title("Original")
    axes[1].imshow(attn_resized, cmap="hot"); axes[1].set_title(f"Attention (L{layer} H{head})")
    axes[2].imshow(image)
    axes[2].imshow(attn_resized, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    
    q_text = sample["user_text"][:80]
    plt.suptitle(f"Q: {q_text}...\nAnswer: {sample['answer']}", fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return attn_grid

# Usage:
sample = test_data[42]
attn = visualize_attention(
    model, processor, sample,
    layer=-1, head=0,
    save_path=f"attention_maps/{MODEL_CONFIG['name']}_sample42.png"
)

# Multi-head comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, head in enumerate(range(8)):
    attn = outputs.attentions[-1][0, head].cpu().float()
    # ... reshape and plot per head
```

### Training-Time Attention (every N steps)

```python
class AttentionSnapshotCallback(TrainerCallback):
    """Saves attention map on a fixed sample every N steps."""
    
    def __init__(self, model, processor, sample, save_dir, every_n_steps=100):
        self.model      = model
        self.processor  = processor
        self.sample     = sample
        self.save_dir   = save_dir
        self.every_n    = every_n_steps
        os.makedirs(save_dir, exist_ok=True)
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n == 0:
            save_path = f"{self.save_dir}/step_{state.global_step:05d}.png"
            self.model.eval()
            with torch.no_grad():
                visualize_attention(
                    self.model, self.processor, self.sample,
                    save_path=save_path
                )
            self.model.train()

# Add to trainer:
attn_callback = AttentionSnapshotCallback(
    model, processor,
    sample=format_sample(val_data[0]),
    save_dir=f"{LOG_DIR}/attention_snapshots",
    every_n_steps=100
)
trainer = SFTTrainer(..., callbacks=[log_callback, attn_callback])
```

---

## Notebook 06 — Comparison and Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results = pd.read_csv("results/comparison_report.csv")
winner  = pd.read_csv("results/phase1_base_comparison.csv").iloc[0]["model"]

# ── Phase 1: base model bar chart ──
phase1 = results[results["condition"] == "base"].sort_values("accuracy", ascending=True)
plt.figure(figsize=(7, 4))
plt.barh(phase1["model"], phase1["accuracy"], color=["#aec6cf","#aec6cf","#4a90d9"])
plt.xlabel("Accuracy (%)")
plt.title("Phase 1 — Base Model Comparison (Zero-Shot)")
plt.tight_layout()
plt.savefig("results/phase1_comparison.png", dpi=150)

# ── Phase 2: base vs TuningA vs TuningB for winner ──
phase2 = results[results["model"] == winner]
plt.figure(figsize=(6, 4))
plt.bar(phase2["condition"], phase2["accuracy"], color=["#aec6cf","#4a90d9","#2e5fa3"])
plt.ylabel("Accuracy (%)")
plt.title(f"Phase 2 — {winner}: Base vs Tuning A vs Tuning B")
plt.tight_layout()
plt.savefig("results/phase2_ablation.png", dpi=150)

# ── Per-category breakdown (winner, all 3 conditions) ──
all_preds = []
for condition in ["base", "TuningA", "TuningB"]:
    fname = f"results/predictions/{winner}_{condition}.csv"
    df = pd.read_csv(fname)
    df["condition"] = condition
    all_preds.append(df)

all_preds = pd.concat(all_preds)
cat_acc = all_preds.groupby(["condition","category"])["correct"].mean().mul(100).reset_index()
cat_acc.columns = ["condition","category","accuracy"]

pivot = cat_acc.pivot(index="category", columns="condition", values="accuracy")
plt.figure(figsize=(12, 7))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn", linewidths=0.5)
plt.title(f"{winner} — Accuracy by Category and Condition")
plt.tight_layout()
plt.savefig("results/category_heatmap.png", dpi=150)

# ── abcLabel vs descriptive split ──
abc_acc = all_preds.groupby(["condition","abc_label"])["correct"].mean().mul(100).reset_index()
print("abcLabel vs Descriptive accuracy:")
print(abc_acc)

# ── Loss curves (Tuning A vs B) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for i, condition in enumerate(["A","B"]):
    log_path = f"logs/{winner}_tuning{condition}/training_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if "loss" in df.columns:
            axes[i].plot(df["step"], df["loss"], label="Train Loss")
        if "eval_loss" in df.columns:
            ev = df.dropna(subset=["eval_loss"])
            axes[i].plot(ev["step"], ev["eval_loss"], linestyle="--", label="Val Loss")
        axes[i].set_title(f"Tuning {condition} Loss Curve")
        axes[i].set_xlabel("Step")
        axes[i].legend()
plt.tight_layout()
plt.savefig("results/loss_curves.png", dpi=150)
plt.show()
```

---

## Execution Order

```
── PHASE 1 ──────────────────────────────────────────────
00_environment_setup.ipynb          ← run once
01_eda_and_data_analysis.ipynb      ← understand the data
02_data_preprocessing.ipynb         ← build dataset versions A and B
03_model1_qwen2vl.ipynb             ← zero-shot eval only
04_model2_llava.ipynb               ← zero-shot eval only
05_model3_internvl2.ipynb           ← zero-shot eval only
                                    ↓
                             compare results → pick winner

── PHASE 2 ──────────────────────────────────────────────
0X_winner_tuningA.ipynb             ← fine-tune winner on image+Q
0X_winner_tuningB.ipynb             ← fine-tune winner on image+Q+annotation
06_comparison_and_analysis.ipynb    ← base vs A vs B + attention map analysis
```

> **If a run crashes:** every checkpoint is saved every 50 steps. Restart from `trainer.train(resume_from_checkpoint=CKPT_DIR)`. The logging callback writes to CSV incrementally so no log data is lost.

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Experiment design | Phase 1: compare 3 bases → Phase 2: ablate winner | Efficient — don't fine-tune weak models |
| Quantization | 4-bit NF4 + double quant | Fits 6GB VRAM |
| Optimizer | paged_adamw_8bit | ~500MB less VRAM vs 32-bit |
| Effective batch | 1 × 8 grad accum = 8 | Stable gradients within VRAM budget |
| LoRA rank | r=8, alpha=16 | Low VRAM overhead, standard for VLMs |
| LoRA targets | q/k/v/o projections | Covers full attention, not just QV |

| Split strategy | Stratified by category | Prevents eval leakage from dominant categories |
| abcLabel handling | Prefix flag in prompt | Lets model know to use letter-label reasoning |
| Annotation injection | Tuning B only | Clean comparison — A vs B isolates annotation value |
| Attention viz | Last-layer, last-token | Most informative for answer generation |
