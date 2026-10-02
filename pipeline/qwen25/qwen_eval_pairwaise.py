import os
import pandas as pd
import requests
from tqdm import tqdm

import torch
from unsloth import FastLanguageModel
from transformers import AutoProcessor
import concurrent.futures
import random

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CSV_PATH = "svo_probes.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=512,
    dtype=None,
    device_map="auto",
    load_in_4bit=True,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_disable()
model.eval()

def ask_model_pairwise(pos_url, neg_url, caption):
    """Ask Qwen-VL which of two images matches caption better."""

    if random.random() < 0.5:
        url_a = pos_url
        url_b = neg_url
        correct_answer = 'A'
    else:
        url_a = neg_url
        url_b = pos_url
        correct_answer = 'B'


    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": url_a},
                    {"type": "image", "image": url_b},
                    {"type": "text",
                     "text": (
                         f"You are an image caption expert.\n"
                         f"Two images are given above. The first image is IMAGE_A, "
                         f"the second image is IMAGE_B.\n\n"
                         f"Caption: '{caption}'\n\n"
                         f"Question: Which image matches the caption better? "
                         f"Answer with just 'A' if IMAGE_A matches better, or just 'B' if IMAGE_B matches better."
                     )},
                ],
            },
        ]

        # build inputs
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        # generate
        output_ids = model.generate(**inputs, max_new_tokens=10, use_cache=False)[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].lower().strip()

        if "a" in answer:
            if correct_answer == 'A':
                return True
            else:
                return False
        elif "b" in answer:
            if correct_answer == 'B':
                return True
            else:
                return False
        else:
            return None
    except Exception as e:
        print(f"Error in ask_model_pairwise: {e}")
        return None


def safe_ask_model_pairwise(pos_url, neg_url, caption, timeout=30):
    """Safe wrapper with timeout."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ask_model_pairwise, pos_url, neg_url, caption)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Timeout on {pos_url}, {neg_url}")
            return None
        except Exception as e:
            print(f"Error on {pos_url}, {neg_url}: {e}")
            return None


# load dataset
df = pd.read_csv(CSV_PATH)

# resume from partial results
if os.path.exists("partial_results_pairwise.csv"):
    results = pd.read_csv("partial_results_pairwise.csv").to_dict("records")
    start_idx = results[-1]["index"] + 1
    print(f"Resuming from index {start_idx} (already processed {len(results)} rows).")
else:
    results = []
    start_idx = 0
    print("Starting from scratch.")


for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):
    sentence = row["sentence"]
    pos_id, neg_id = row["pos_image_id"], row["neg_image_id"]

    try:
        pred = safe_ask_model_pairwise('images/' +str(pos_id)+'.jpg', 'images/' +str(neg_id)+'.jpg', sentence)

        results.append({
                "index": i,
                "correct": pred,
                "subj_neg": row["subj_neg"],
                "verb_neg": row["verb_neg"],
                "obj_neg": row["obj_neg"]
            })

    except Exception as e:
        print(f"Skipping row {i} due to error: {e}")
        continue

    if i % 100 == 0 and i > 0:
        pd.DataFrame(results).to_csv("partial_results_pairwise.csv", index=False)
        torch.cuda.empty_cache()

# final save
results_df = pd.DataFrame(results)
results_df.to_csv("svo_probes_results_pairwise.csv", index=False)

# compute accuracy
overall_acc = results_df["correct"].mean()

# per corruption type
group_acc = {}
for col in ["subj_neg", "verb_neg", "obj_neg"]:
    subset = results_df[results_df[col] == True]
    group_acc[col] = subset["correct"].mean() if not subset.empty else None

print("Overall accuracy:", overall_acc)
print("Accuracy by corruption type:")
print(group_acc)
