import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


import torch
from unsloth import FastLanguageModel
from transformers import AutoProcessor

import concurrent.futures

def safe_ask_model(url, caption, timeout=30):
    """Run ask_model with a timeout. Return None if it hangs too long."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ask_model, url, caption)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(f"Timeout on {url}")
            return None
        except Exception as e:
            print(f"Error on {url}: {e}")
            return None

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CSV_PATH = "svo_probes.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"


model, tokenizer = FastLanguageModel.from_pretrained(
MODEL_NAME,
max_seq_length=512,
dtype=None,
device_map="auto",
load_in_4bit=True # keep consistent with fine-tuning setup
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

def ask_model(image_url, caption):
    """Ask Qwen-VL if image matches caption."""

    # build chat input
    try: 
        messages = [
        {
        "role": "user",
        "content": [
        {"type": "image",
         "image": 'images/'+ str(image_url)+'.jpg'},
        {"type": "text", 
         "text": f"You are an image caption expert. Does this image match the caption: '{caption}'? Answer yes only if everything from the caption - subject, verb and object - matches the image, otherwise answer no."},
        ],
        },  
        ]

        # Build chat input
        inputs = processor.apply_chat_template(messages, tokenize=True,  return_dict=True, add_generation_prompt=True,  return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, max_new_tokens=10)[:, inputs.input_ids.shape[1]:]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].lower().strip()

        if "yes" in answer:
            return "yes"
        elif "no" in answer:
            return "no"
        else:
            return None
    except Exception as e:
        #print(f"Error processing {image_url}: {e}")
        return None
    
df = pd.read_csv(CSV_PATH)

# Load partial results if file exists
if os.path.exists("partial_results.csv"):
    results = pd.read_csv("partial_results.csv").to_dict("records")
    start_idx = results[-1]["index"] + 1   # resume from last processed index + 1
    print(f"Resuming from index {start_idx} (already processed {len(results)} rows).")
else:
    results = []
    start_idx = 0
    print("Starting from scratch.")



for i, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):
    sentence = row["sentence"]
    pos_id, neg_id = row["pos_image_id"], row["neg_image_id"]
        # run model
    try:
        pos_pred = safe_ask_model(pos_id, sentence)
        neg_pred = safe_ask_model(neg_id, sentence)

        # expected answers
        pos_correct = pos_pred == "yes"
        neg_correct = neg_pred == "no"


        overall_correct = pos_correct and neg_correct


        results.append({
            "index": i,
            "pos_pred": pos_pred,
            "neg_pred": neg_pred,
            "pos_correct": pos_correct,
            "neg_correct": neg_correct,
            "subj_neg": row["subj_neg"],
            "verb_neg": row["verb_neg"],
            "obj_neg": row["obj_neg"]
            })
    except:
        continue
    if i % 100 == 0 and i > 0:
        pd.DataFrame(results).to_csv("partial_results.csv", index=False)

    torch.cuda.empty_cache()

results_df = pd.DataFrame(results)


# overall accuracy
overall_acc = pd.concat([results_df['pos_correct'], results_df['neg_correct']], ignore_index=True).mean()


# group accuracies
group_acc = {}
for col in ["subj_neg", "verb_neg", "obj_neg"]:
    subset = results_df[results_df[col] == True]
    group_acc[col] = pd.concat([subset['pos_correct'], subset['neg_correct']], ignore_index=True).mean() if not subset.empty else None


print("Overall accuracy:", overall_acc)
print("Accuracy by corruption type:")
print(group_acc)


# save results
results_df.to_csv("svo_probes_results.csv", index=False)