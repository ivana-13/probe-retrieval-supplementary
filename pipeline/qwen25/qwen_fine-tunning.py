import argparse
import json
from pathlib import Path
from typing import Dict, Any
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from unsloth import FastLanguageModel
from transformers import AutoProcessor, TrainingArguments, Trainer
from typing import List, Tuple

from pycocotools.coco import COCO
import random

import math
from bitsandbytes.optim import AdamW8bit

# ===============================
# Dataset
# ===============================
class ContrastiveImageDataset(Dataset):
    """Dataset from JSONL with fields: image_path, pos_caption, neg_caption."""

    def __init__(self, jsonl_path: str, processor, prompt_cfg: Dict[str, str]):
        self.items = [json.loads(l) for l in open(jsonl_path)]
        self.processor = processor
        self.prompt_cfg = prompt_cfg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ex = self.items[idx]
        image = Image.open(ex["image_path"]).convert("RGB")

        # Prompts
        img_prompt = self.prompt_cfg["image_prompt"]
        pos_cap_prompt = self.prompt_cfg["caption_prompt"].replace("{caption}", ex["pos_caption"])
        neg_cap_prompt = self.prompt_cfg["caption_prompt"].replace("{caption}", ex["neg_caption"])

        device = torch.device("cuda")

        # Encode image
        chat = self.processor.apply_chat_template(
            [{"role": "user",
            "content": [{"type": "image", "image": image},
                  {"type": "text", "text": img_prompt}]}],
            tokenize=False,
            add_generation_prompt=False
            )

        img_enc = self.processor(
        text=chat,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=False
        )

        # Encode positive caption
        pos_enc = self.processor(
        text=[pos_cap_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

        # Encode negative caption
        neg_enc = self.processor(
        text=[neg_cap_prompt],
        return_tensors="pt",
        padding=True,
        truncation=True
    )
        return img_enc, pos_enc, neg_enc
    
class COCODataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, processor, prompt_cfg: Dict[str, str], sample_size: int = None,
        seed: int = 42):
        self.coco = COCO(ann_file)
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.prompt_cfg = prompt_cfg

        # Get all image–caption pairs
        pairs = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            captions = [a["caption"] for a in anns]
            img_info = self.coco.loadImgs([img_id])[0]
            img_path = self.img_dir / img_info["file_name"]

            for cap in captions:
                pairs.append((str(img_path), cap))
        
        # Random subsample
        if sample_size is not None and sample_size < len(pairs):
            random.seed(seed)
            pairs = random.sample(pairs, sample_size)

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB").resize((512, 512))

        # Prompts
        img_prompt = self.prompt_cfg["image_prompt"]
        cap_prompt = self.prompt_cfg["caption_prompt"].replace("{caption}", caption)

        # Encode image
        chat = self.processor.apply_chat_template(
            [{"role": "user",
              "content": [{"type": "image", "image": image},
                          {"type": "text", "text": img_prompt}]}],
            tokenize=False,
            add_generation_prompt=False
        )

        img_enc = self.processor(
            text=chat,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=512,
            truncation=True
        )
        chat = self.processor.apply_chat_template(
        [{"role": "user",
          "content": [{"type": "text", "text": cap_prompt}]}],
        tokenize=False,
        add_generation_prompt=False
    )

        # Encode caption
        cap_enc = self.processor(
            text=chat,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )

        return img_enc, cap_enc

def coco_collate_fn(batch): 
    imgs, caps = zip(*batch) 

    def collate_dict(list_of_dicts): 
        return {k: torch.cat([d[k] for d in list_of_dicts], dim=0) for k in list_of_dicts[0]} 

    img_batch = collate_dict(imgs) 
    cap_batch = collate_dict(caps) 
    
    return img_batch, cap_batch

# ===============================
# Custom Trainer with contrastive loss
# ===============================
def contrastive_collate_fn(batch):
    imgs, pos, neg = zip(*batch)

    # Collate a list of dicts into a dict of tensors
    def collate_dict(list_of_dicts):
        return {k: torch.cat([d[k] for d in list_of_dicts], dim=0) for k in list_of_dicts[0]}

    img_batch = collate_dict(imgs)
    pos_batch = collate_dict(pos)
    neg_batch = collate_dict(neg)

    return img_batch, pos_batch, neg_batch


class ContrastiveTrainer(Trainer):
    def __init__(self, tau: float = 0.07, loss_type: str = "InfoNCE", bias: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.loss_type = loss_type
        self.bias = bias

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        img_enc, cap_enc = inputs

        def get_emb(enc):
            out = model(**enc, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]  # access last layer hidden states
            return last_hidden[:, -1, :]

        hi = get_emb(img_enc)   # [B, D]
        hc = get_emb(cap_enc)   # [B, D]

        hi, hc = map(lambda x: F.normalize(x, dim=1), (hi, hc))

        if self.loss_type == 'InfoNCE':

            sim = (hi @ hc.T) / self.tau


            # Positive sims are the diagonal
            # logsumexp trick
            log_denom = torch.logsumexp(sim, dim=1)
             # mask positives
            pos_mask = torch.eye(sim.size(0), device=sim.device)
            pos_sim = sim.masked_fill(pos_mask == 0, float("-inf"))
            log_num = torch.logsumexp(pos_sim, dim=1)

            loss = -(log_num - log_denom)

            return loss.mean()

        if self.loss_type == 'Sigmoid_loss':
            sim = hi @ hc.T  # [batch, batch]
            t = math.log(self.tau)   # scalar
            b = self.bias  # scalar

            z = torch.ones_like(sim) * -1
            z.fill_diagonal_(1)


            # Compute logits for sigmoid loss
            logits = - z * (t * sim - b)  # [batch, batch]

            # Apply log(1 + exp(logit)) = F.softplus(logit)
            loss = F.softplus(logits).mean()

            return (loss, {"loss": loss}) if return_outputs else loss


# ===============================
# Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with Unsloth + Contrastive Loss")

    # Model + LoRA
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Data
    parser.add_argument("--image_prompt", type=str, default='This image means in one word: ')
    parser.add_argument("--caption_prompt", type=str, default='This caption: \" {caption} \" means in one word: ')

    # Training
    parser.add_argument("--output_dir", type=str, default="./qwen_unsloth_contrastive")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--bias", type=float, default=0)
    parser.add_argument("--loss_type", type=str, default='InfoNCE')
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=1)

    args = parser.parse_args()

    # ===============================
    # Load model on GPU
    # ===============================
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=512,      # set max_seq_length to fit GPU memory
        dtype=None,
        load_in_4bit=True,
        device_map="auto"        # auto GPU placement
    )

    model._set_gradient_checkpointing(True)

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
         task_type="CAUSAL_LM"
    )

    # Move model to device (GPU)
    model.to(device)

    # Processor
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Dataset
    dataset = COCODataset(
        img_dir="./coco/train2014/train2014",
        ann_file="./coco/annotations/captions_train2014.json",
        processor=processor,
        prompt_cfg={"image_prompt": args.image_prompt, "caption_prompt": args.caption_prompt},
        sample_size=10000,
        seed=42
    )
    # Training args
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        bf16=True if device == "cuda" else False,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    # Trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        tau=args.tau,
        loss_type=args.loss_type,
        bias=args.bias,
        data_collator=coco_collate_fn
    )

    trainer.train()
    name = "fine_tunned_qwen" + args.loss_type + 'new_version'
    trainer.model.save_pretrained(name)
    processor.save_pretrained(name)

if __name__ == "__main__":
    main()
