# Implementation details

Checkpoints, prompts, training details and statistics behind the paper's experiments.

## Embedding models

| Model | Checkpoint and extraction |
|---|---|
| CLIP | ViT-L/14, OpenAI weights, through open_clip |
| BLIP-2 | LAVIS `blip2_feature_extractor` (`pretrain`): projected first query token for images, projected [CLS] for captions (256-d) |
| FLAVA | `facebook/flava-full`: projected [CLS] of the image and of the text encoder |
| SigLIP2 | `google/siglip2-base-patch16-224`, captions padded to 64 tokens |
| Qwen2.5-VL-FT | Qwen2.5-VL-7B-Instruct in 4-bit with our QLoRA adapter; embedding = last hidden state of the final token (3,584-d) after the prompt "This image means in one word: " (images) or "This caption: "[caption]" means in one word: " (captions) |
| Qwen3-VL-Embed | `Qwen3-VL-Embedding-2B` in fp16, images resized to 1024 x 1024, no instruction (2,048-d) |

**BLIP-2 variant.** The published BLIP-2 retrieval setting takes the maximum similarity over all 32 query tokens and
re-ranks the top candidates with the matching head. We use the first query token only and no re-ranking, so the BLIP-2
rows of the paper describe this reduced variant and are lower than published retrieval results.

**Scoring.** Retrieval and pairwise decisions use the dot product of the image and caption embeddings. The embeddings of
BLIP-2, Qwen2.5-VL-FT and Qwen3-VL-Embed are unit-normalised, so for them this is the cosine similarity; those of CLIP,
FLAVA and SigLIP2 are used as extracted on SVO-Probes (`svo_eval/embeddings.py`). `scripts/03c_metrics_cosine.py`
repeats the SVO-Probes metrics with cosine similarity for these three models, and `scripts/43_cosine_check.py` repeats
the comparison of handcrafted and other-mined negatives on the judged pool (`extra_results.md`, section 16). All models are scored with cosine
similarity on COCO (`svo_eval/coco_scoring.py`).

**Qwen2.5-VL-FT adapter.** QLoRA on the attention projections (q, k, v, o; rank 64, alpha 16, dropout 0.05), InfoNCE
over in-batch pairs with temperature 0.1, 10,000 MS-COCO train pairs, one epoch at batch 16 with 16 accumulation steps
(effective batch 256, 40 optimiser steps), learning rate 4e-4 (`qwen25/qwen_fine-tunning.py`).

## Binary matching decisions

The paper reports these decisions in one compact table; the full results (pairs the model itself or only other
models retrieved, areas under the ROC curve) are in `extra_results.md`, section 9.

- **BLIP-2**: the first-stage matching head of LAVIS `blip2_image_text_matching` (`pretrain`), ITM logits averaged over
  the query tokens; a match when the softmax match probability exceeds 0.5.
- **FLAVA**: the ITM head of `facebook/flava-full`; a match when the softmax match probability exceeds 0.5. Batched and
  single-pair decisions are identical (`tests/test_itm_heads.py`).
- **Qwen2.5-VL (zero-shot)**: Qwen2.5-VL-7B-Instruct without the adapter, 4-bit, greedy decoding. ITM prompt:
  "You are an image caption expert. Does this image match the caption: '[caption]'? Answer yes only if everything from
  the caption - subject, verb and object - matches the image, otherwise answer no." Pairwise prompt (both images shown,
  in random order): "You are an image caption expert. Two images are given above. The first image is IMAGE_A, the second
  image is IMAGE_B. Caption: '[caption]' Question: Which image matches the caption better? Answer with just 'A' if
  IMAGE_A matches better, or just 'B' if IMAGE_B matches better." On SugarCrepe, whose negatives are captions, the
  pairwise prompt shows one image and the two captions. Eight SVO-Probes triplets received no parseable pairwise answer
  and are excluded.
- **Qwen3-VL-Embed**: the joint image–caption embedding under the instruction "Represent the given image and caption for
  binary classification to determine whether they match or not" is compared with the embeddings of "Yes" and "No" under
  "Represent the given answer for binary classification"; a match when the similarity to "Yes" is higher.

The same code scores the 67,370 SVO-Probes benchmark pairs and the judged pools; 10 benchmark pairs and 2 pool pairs
with unreadable or very small images are skipped.

## Statistics

- Intervals: 95% percentile intervals from 2,000 bootstrap resamples over queries (over triplets or pairs for the probe
  accuracies).
- Tests: two-sided paired randomisation (sign-flip) test on the per-query differences, 100,000 permutations, with Holm
  correction over the model-and-direction cells of a comparison.
- Random negatives: ten items drawn uniformly from the dataset without the relevant items, averaged over 20 draws that
  are shared by all models (seeded per query).
- n-way curves: 50 draws of k random items; other-mined negatives are added in a random order, averaged over 50 orders;
  both on the queries that have at least ten other-mined negatives for the scored model.

`metric_definitions.md` gives the exact definition of every reported quantity.
