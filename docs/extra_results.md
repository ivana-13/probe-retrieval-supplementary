# Results that did not fit into the paper

All values are percentages unless stated otherwise. T→I: caption queries; I→T: image queries. Definitions are in `metric_definitions.md`; every table is generated from `pipeline/results/`.

## 1. Matched negatives on SVO-Probes: variants and the queries whose positive is outside the model's top 10

`Handcr. (own)` scores each handcrafted negative against its own triplet's positive; `Handcr.` against the best relevant item, as in the paper. `p` is the Holm-corrected p-value of handcrafted against other-mined.

| Dir. | Model | Random | Handcr. | Handcr. (own) | Other | Self | p | n outside top 10 | Handcr. (subset) | Other (subset) | p (subset) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T→I | CLIP | 95.6 | 86.7 | 85.9 | 77.4 | 23.3 | 0.0011 | 65 | 79.5 | 65.2 | 0.0005 |
| T→I | BLIP-2 | 98.7 | 90.8 | 90.2 | 82.4 | 33.4 | 0.0011 | 59 | 84.4 | 70.2 | 0.0005 |
| T→I | FLAVA | 98.8 | 90.1 | 89.4 | 79.8 | 25.8 | 0.0004 | 64 | 85.4 | 68.3 | 0.0001 |
| T→I | SigLIP2 | 98.3 | 89.5 | 88.7 | 79.5 | 37.3 | 0.0011 | 56 | 82.0 | 63.3 | 0.0004 |
| T→I | Qwen2.5-VL-FT | 99.1 | 91.1 | 89.8 | 83.6 | 31.7 | 0.0014 | 58 | 86.2 | 71.9 | 0.0002 |
| T→I | Qwen3-VL-Embed | 99.3 | 95.1 | 94.6 | 85.4 | 40.4 | 0.0005 | 46 | 89.7 | 68.3 | 0.0004 |
| I→T | CLIP | 98.7 | 87.8 | 87.2 | 76.8 | 19.4 | 0.0005 | 72 | 83.0 | 67.8 | 0.0004 |
| I→T | BLIP-2 | 99.1 | 93.1 | 93.1 | 85.2 | 41.0 | 0.0015 | 47 | 88.2 | 68.6 | 0.0001 |
| I→T | FLAVA | 98.2 | 85.2 | 84.7 | 79.0 | 23.8 | 0.0247 | 69 | 79.5 | 69.6 | 0.0108 |
| I→T | SigLIP2 | 99.6 | 94.6 | 94.4 | 87.8 | 46.0 | 0.0105 | 40 | 89.6 | 69.5 | 0.0005 |
| I→T | Qwen2.5-VL-FT | 99.4 | 93.0 | 92.6 | 84.2 | 39.9 | 0.0011 | 49 | 87.8 | 67.4 | 0.0001 |
| I→T | Qwen3-VL-Embed | 99.7 | 95.7 | 95.3 | 88.0 | 37.3 | 0.0011 | 47 | 91.3 | 74.4 | 0.0004 |

## 2. Pool composition and label noise (SVO-Probes)

Other-mined accuracy without the items that only the two Qwen models retrieved, and, for image queries, with only the items that all three annotators marked incorrect (the three-annotator lists are those of the four dual encoders). `gap` = handcrafted (own positive) minus the restricted other-mined accuracy; p-values uncorrected.

| Dir. | Model | Other | Qwen-only share | Other w/o Qwen-only | gap | p | Other, unanimous | gap | p |
|---|---|---|---|---|---|---|---|---|---|
| T→I | CLIP | 77.4 | 31 | 78.8 | 6.9 | 0.0077 | – | – | – |
| T→I | BLIP-2 | 82.4 | 29 | 83.3 | 6.8 | 0.0067 | – | – | – |
| T→I | FLAVA | 79.8 | 30 | 78.5 | 10.6 | 0.0001 | – | – | – |
| T→I | SigLIP2 | 79.5 | 28 | 79.9 | 8.6 | 0.0014 | – | – | – |
| T→I | Qwen2.5-VL-FT | 83.6 | 10 | 84.7 | 5.0 | 0.0173 | – | – | – |
| T→I | Qwen3-VL-Embed | 85.4 | 15 | 86.2 | 8.3 | 0.0005 | – | – | – |
| I→T | CLIP | 76.8 | 26 | 75.3 | 11.7 | 0.0001 | 78.9 | 7.7 | 0.0106 |
| I→T | BLIP-2 | 85.2 | 24 | 86.9 | 6.2 | 0.0038 | 88.6 | 4.3 | 0.0274 |
| I→T | FLAVA | 79.0 | 27 | 78.0 | 6.7 | 0.0242 | 79.2 | 5.1 | 0.1251 |
| I→T | SigLIP2 | 87.8 | 24 | 87.2 | 7.2 | 0.0051 | 87.7 | 6.6 | 0.0091 |
| I→T | Qwen2.5-VL-FT | 84.2 | 8 | 85.6 | 7.1 | 0.0023 | 86.3 | 6.4 | 0.0043 |
| I→T | Qwen3-VL-Embed | 88.0 | 13 | 88.6 | 6.7 | 0.0011 | 89.6 | 5.7 | 0.0044 |

## 3. Cross-model matrices (SVO-Probes)

Accuracy of the row model on the judged-incorrect items that the column model retrieved.

**T→I**

| Scored \ mined by | CLIP | BLIP-2 | FLAVA | SigLIP2 | Qwen2.5-VL-FT | Qwen3-VL-Embed |
|---|---|---|---|---|---|---|
| CLIP | 23.3 | 73.6 | 74.3 | 66.9 | 67.3 | 59.2 |
| BLIP-2 | 81.7 | 33.4 | 72.5 | 67.1 | 70.2 | 67.5 |
| FLAVA | 81.6 | 67.1 | 25.8 | 57.6 | 70.6 | 76.0 |
| SigLIP2 | 79.7 | 70.9 | 67.3 | 37.3 | 69.2 | 72.5 |
| Qwen2.5-VL-FT | 78.8 | 75.2 | 78.9 | 72.5 | 31.7 | 62.6 |
| Qwen3-VL-Embed | 80.4 | 77.8 | 81.6 | 80.0 | 68.6 | 40.4 |

**I→T**

| Scored \ mined by | CLIP | BLIP-2 | FLAVA | SigLIP2 | Qwen2.5-VL-FT | Qwen3-VL-Embed |
|---|---|---|---|---|---|---|
| CLIP | 19.4 | 73.1 | 74.1 | 66.9 | 72.5 | 73.7 |
| BLIP-2 | 87.2 | 41.0 | 81.7 | 74.6 | 66.9 | 60.2 |
| FLAVA | 73.4 | 67.9 | 23.8 | 66.8 | 71.2 | 75.0 |
| SigLIP2 | 85.8 | 82.6 | 83.6 | 46.0 | 82.6 | 75.1 |
| Qwen2.5-VL-FT | 88.0 | 71.7 | 82.6 | 73.2 | 39.9 | 55.0 |
| Qwen3-VL-Embed | 89.2 | 74.4 | 86.9 | 78.7 | 69.8 | 37.3 |

## 4. Probe outcome against retrieval outcome (SVO-Probes, 100 queries per direction)

P+ / P−: the positive scores above every handcrafted negative of the query, or not. R+ / R−: the top-ranked item is judged correct, or not.

| Dir. | Model | P+R+ | P+R− | P−R+ | P−R− | R+ given P+ | R+ given P− | Fisher p |
|---|---|---|---|---|---|---|---|---|
| T→I | CLIP | 45 | 22 | 19 | 14 | 67 | 58 | 0.38 |
| T→I | BLIP-2 | 55 | 17 | 21 | 7 | 76 | 75 | 1.00 |
| T→I | FLAVA | 54 | 18 | 18 | 10 | 75 | 64 | 0.33 |
| T→I | SigLIP2 | 55 | 16 | 20 | 9 | 77 | 69 | 0.45 |
| T→I | Qwen2.5-VL-FT | 54 | 16 | 16 | 14 | 77 | 53 | 0.03 |
| T→I | Qwen3-VL-Embed | 67 | 14 | 13 | 6 | 83 | 68 | 0.20 |
| I→T | CLIP | 36 | 33 | 18 | 13 | 52 | 58 | 0.67 |
| I→T | BLIP-2 | 64 | 19 | 11 | 6 | 77 | 65 | 0.36 |
| I→T | FLAVA | 38 | 27 | 24 | 11 | 58 | 69 | 0.39 |
| I→T | SigLIP2 | 67 | 19 | 11 | 3 | 78 | 79 | 1.00 |
| I→T | Qwen2.5-VL-FT | 62 | 17 | 17 | 4 | 78 | 81 | 1.00 |
| I→T | Qwen3-VL-Embed | 78 | 8 | 12 | 2 | 91 | 86 | 0.63 |

## 5. Error types (SVO-Probes)

Share of each error type among the judged-incorrect items of the model's top-10 lists, and the error type of the top-ranked item on the queries that pass the probe and fail retrieval (counts).

| Dir. | Model | Subject | Verb | Object | P+R−: subject | verb | object | no type |
|---|---|---|---|---|---|---|---|---|
| T→I | CLIP | 22.2 | 41.4 | 36.4 | 3 | 8 | 11 | 0 |
| T→I | BLIP-2 | 23.1 | 52.6 | 24.3 | 3 | 8 | 6 | 0 |
| T→I | FLAVA | 26.8 | 46.9 | 26.3 | 4 | 7 | 7 | 0 |
| T→I | SigLIP2 | 30.1 | 42.8 | 27.1 | 5 | 9 | 2 | 0 |
| T→I | Qwen2.5-VL-FT | 21.2 | 52.2 | 26.6 | 3 | 8 | 5 | 0 |
| T→I | Qwen3-VL-Embed | 10.8 | 48.5 | 40.7 | 1 | 12 | 1 | 0 |
| I→T | CLIP | 23.5 | 39.4 | 37.1 | 7 | 6 | 8 | 12 |
| I→T | BLIP-2 | 17.2 | 48.4 | 34.4 | 6 | 7 | 3 | 3 |
| I→T | FLAVA | 20.0 | 51.4 | 28.5 | 4 | 13 | 6 | 4 |
| I→T | SigLIP2 | 28.7 | 43.0 | 28.3 | 4 | 5 | 7 | 3 |
| I→T | Qwen2.5-VL-FT | 22.1 | 46.6 | 31.3 | 2 | 11 | 4 | 0 |
| I→T | Qwen3-VL-Embed | 14.9 | 48.0 | 37.2 | 0 | 3 | 5 | 0 |

Composition of the benchmark's triplets: verb 63.4, object 20.8, subject 15.8.

## 6. n-way comparison: ten other-mined negatives against random negatives

On the queries with at least ten other-mined negatives: share of queries on which the positive outranks ten random and ten other-mined negatives, and the number of random negatives that lower this share as much as the ten other-mined ones.

| Benchmark | Dir. | Model | Queries | 10 random | 10 other-mined | Equivalent random k |
|---|---|---|---|---|---|---|
| SVO-Probes | T→I | CLIP | 67 | 85.0 | 45.4 | 379 |
| SVO-Probes | T→I | BLIP-2 | 74 | 89.9 | 55.6 | 297 |
| SVO-Probes | T→I | FLAVA | 70 | 92.5 | 50.6 | 431 |
| SVO-Probes | T→I | SigLIP2 | 75 | 90.0 | 57.4 | 326 |
| SVO-Probes | T→I | Qwen2.5-VL-FT | 70 | 92.6 | 52.3 | 347 |
| SVO-Probes | T→I | Qwen3-VL-Embed | 75 | 93.3 | 59.9 | 347 |
| SVO-Probes | I→T | CLIP | 66 | 89.7 | 40.9 | 360 |
| SVO-Probes | I→T | BLIP-2 | 68 | 92.4 | 67.6 | 233 |
| SVO-Probes | I→T | FLAVA | 61 | 87.6 | 44.0 | 230 |
| SVO-Probes | I→T | SigLIP2 | 75 | 96.0 | 74.5 | 209 |
| SVO-Probes | I→T | Qwen2.5-VL-FT | 71 | 93.9 | 59.4 | 314 |
| SVO-Probes | I→T | Qwen3-VL-Embed | 76 | 95.6 | 68.0 | 275 |
| COCO | T→I | CLIP | 88 | 98.6 | 84.3 | 188 |
| COCO | T→I | BLIP-2 | 87 | 94.8 | 61.2 | 208 |
| COCO | T→I | FLAVA | 89 | 99.1 | 92.5 | 127 |
| COCO | T→I | SigLIP2 | 92 | 99.4 | 90.8 | 226 |
| COCO | T→I | Qwen2.5-VL-FT | 90 | 98.2 | 83.4 | 168 |
| COCO | T→I | Qwen3-VL-Embed | 92 | 99.5 | 92.7 | 178 |
| COCO | I→T | CLIP | 81 | 99.6 | 87.1 | 743 |
| COCO | I→T | BLIP-2 | 78 | 99.5 | 88.2 | 369 |
| COCO | I→T | FLAVA | 80 | 99.8 | 97.0 | 249 |
| COCO | I→T | SigLIP2 | 85 | 99.9 | 96.0 | 548 |
| COCO | I→T | Qwen2.5-VL-FT | 86 | 99.6 | 89.9 | 508 |
| COCO | I→T | Qwen3-VL-Embed | 91 | 100.0 | 98.9 | 153 |

## 7. COCO: mined negatives under the assessor's and the expert's labels

The 12 queries per direction that the expert labelled, with the same pooled candidates under both label sets.

| Dir. | Model | Self (o3) | Self (expert) | Other (o3) | Other (expert) | Other − self (o3) | Other − self (expert) |
|---|---|---|---|---|---|---|---|
| T→I | CLIP | 67.8 | 67.6 | 86.5 | 86.3 | 18.8 | 18.6 |
| T→I | BLIP-2 | 46.9 | 45.2 | 91.4 | 90.8 | 44.5 | 45.6 |
| T→I | FLAVA | 86.1 | 87.0 | 93.2 | 93.2 | 7.1 | 6.2 |
| T→I | SigLIP2 | 83.3 | 83.3 | 94.9 | 94.7 | 11.6 | 11.4 |
| T→I | Qwen2.5-VL-FT | 68.5 | 70.9 | 95.5 | 94.8 | 27.0 | 23.9 |
| T→I | Qwen3-VL-Embed | 88.9 | 88.0 | 98.8 | 98.8 | 9.9 | 10.8 |
| I→T | CLIP | 84.4 | 77.4 | 99.5 | 99.5 | 15.1 | 22.1 |
| I→T | BLIP-2 | 80.8 | 86.3 | 99.4 | 100.0 | 18.6 | 13.7 |
| I→T | FLAVA | 77.7 | 77.4 | 100.0 | 100.0 | 22.3 | 22.6 |
| I→T | SigLIP2 | 85.1 | 82.8 | 100.0 | 100.0 | 14.9 | 17.2 |
| I→T | Qwen2.5-VL-FT | 89.4 | 89.8 | 100.0 | 100.0 | 10.6 | 10.2 |
| I→T | Qwen3-VL-Embed | 90.0 | 86.4 | 100.0 | 100.0 | 10.0 | 13.6 |

Pairs labelled by both: T→I 346 (15 incorrect for o3 but correct for the expert, 8 the other way round); I→T 359 (43 incorrect for o3 but correct for the expert, 13 the other way round).

## 8. Assessor agreement in detail

o3 against the human label on each model's own SVO-Probes list (1,000 pairs per cell; majority vote for the three-annotator lists, the expert otherwise).

| Dir. | Model | Pairs with an o3 label | Accuracy | Cohen's κ | Correct (human) | Correct (o3) |
|---|---|---|---|---|---|---|
| T→I | CLIP | 980 | 84.4 | 0.69 | 51.6 | 46.8 |
| T→I | BLIP-2 | 936 | 82.8 | 0.65 | 61.4 | 54.9 |
| T→I | FLAVA | 930 | 81.8 | 0.64 | 58.5 | 48.9 |
| T→I | SigLIP2 | 990 | 83.0 | 0.65 | 62.4 | 57.0 |
| T→I | Qwen2.5-VL-FT | 1000 | 84.5 | 0.68 | 59.4 | 57.9 |
| T→I | Qwen3-VL-Embed | 1000 | 81.8 | 0.60 | 67.6 | 61.8 |
| I→T | CLIP | 990 | 82.8 | 0.66 | 51.2 | 47.8 |
| I→T | BLIP-2 | 990 | 83.9 | 0.67 | 61.8 | 54.8 |
| I→T | FLAVA | 990 | 81.8 | 0.63 | 46.8 | 43.5 |
| I→T | SigLIP2 | 990 | 81.6 | 0.61 | 64.5 | 58.9 |
| I→T | Qwen2.5-VL-FT | 1000 | 82.8 | 0.65 | 62.8 | 52.4 |
| I→T | Qwen3-VL-Embed | 1000 | 81.5 | 0.60 | 70.3 | 59.2 |

o3 against the expert on the COCO validation sample.

| Direction | Pairs | Accuracy | Precision (correct) | Recall (correct) | F1 | Cohen's κ |
|---|---|---|---|---|---|---|
| T→I | 346 | 93.4 | 0.67 | 0.52 | 0.58 | 0.55 |
| I→T | 359 | 84.4 | 0.87 | 0.68 | 0.76 | 0.65 |
| Both | 705 | 88.8 | 0.83 | 0.65 | 0.73 | 0.66 |

## 9. Matching heads without a threshold

Area under the ROC curve of the head's match probability, and the share of incorrect pairs in each pool (the accuracy of always answering "no match").

| Pairs | AUC BLIP-2 | AUC FLAVA | Incorrect pairs |
|---|---|---|---|
| SVO-Probes pool, T→I | 0.68 | 0.59 | 48.2 |
| SVO-Probes pool, I→T | 0.65 | 0.55 | 50.5 |
| SugarCrepe pairs | 0.77 | 0.59 | 50.0 |
| COCO pool, T→I | 0.89 | 0.68 | 82.9 |
| COCO pool, I→T | 0.84 | 0.54 | 59.3 |

## 10. Probe accuracy against retrieval success, embedding models only

Qwen2.5-VL-FT with its own similarity decision. Kendall's τ between pairwise accuracy and strict Success@1.

**SVO-Probes** (τ: T→I 0.47, I→T 0.47)

| Model | Pairwise accuracy | Success@1 T→I | Success@1 I→T |
|---|---|---|---|
| CLIP | 82.0 | 8.7 | 6.5 |
| BLIP-2 | 87.6 | 9.7 | 13.2 |
| FLAVA | 86.3 | 8.6 | 5.6 |
| SigLIP2 | 86.0 | 10.4 | 13.4 |
| Qwen2.5-VL-FT | 87.0 | 12.2 | 13.2 |
| Qwen3-VL-Embed | 90.8 | 14.4 | 16.8 |

Paired differences in pairwise accuracy (95% bootstrap interval): FLAVA − SigLIP2: +0.3 [-0.3, +0.8]; BLIP-2 − SigLIP2: +1.5 [+1.0, +2.0]; FLAVA − CLIP: +4.3 [+3.6, +4.9]; BLIP-2 − Qwen2.5-VL-FT: +0.6 [+0.1, +1.1].

**COCO with SugarCrepe** (τ: T→I 0.87, I→T 0.47)

| Model | Pairwise accuracy | Success@1 T→I | Success@1 I→T |
|---|---|---|---|
| CLIP | 79.2 | 36.0 | 57.0 |
| BLIP-2 | 77.2 | 30.3 | 40.4 |
| FLAVA | 83.8 | 41.5 | 48.8 |
| SigLIP2 | 82.7 | 48.5 | 64.2 |
| Qwen2.5-VL-FT | 80.9 | 37.9 | 51.8 |
| Qwen3-VL-Embed | 87.6 | 54.8 | 71.2 |

Paired differences in pairwise accuracy (95% bootstrap interval): FLAVA − SigLIP2: +1.1 [+0.1, +2.1]; FLAVA − CLIP: +4.6 [+3.5, +5.7]; Qwen2.5-VL-FT − CLIP: +1.6 [+0.4, +2.8].

## 11. SugarCrepe pairwise accuracy by subset

| Model | add_att | add_obj | replace_att | replace_obj | replace_rel | swap_att | swap_obj |
|---|---|---|---|---|---|---|---|
| CLIP | 74.3 | 80.2 | 82.6 | 94.2 | 71.2 | 62.9 | 63.7 |
| BLIP-2 | 74.6 | 85.8 | 74.5 | 94.1 | 64.2 | 51.1 | 53.5 |
| FLAVA | 69.2 | 89.9 | 85.8 | 94.1 | 74.7 | 76.0 | 71.8 |
| SigLIP2 | 79.0 | 90.0 | 81.6 | 94.2 | 69.9 | 70.9 | 62.9 |
| Qwen2.5-VL (zero-shot) | 89.9 | 97.5 | 94.9 | 98.6 | 92.8 | 94.9 | 89.0 |
| Qwen2.5-VL-FT (embedding) | 78.5 | 84.3 | 77.8 | 93.5 | 72.5 | 67.3 | 68.6 |
| Qwen3-VL-Embed | 85.5 | 90.6 | 90.5 | 97.6 | 79.2 | 77.5 | 68.2 |

## 12. Reusability of the SVO-Probes pool

Leave-one-out test: the items that only the given model contributed to the pool are treated as unjudged. `Unique` is the share of the model's top-10 items that no other model retrieved.

| Dir. | Model | Unique | Judged S@1 | S@1 leave-one-out | Judged P@10 | P@10 leave-one-out |
|---|---|---|---|---|---|---|
| T→I | CLIP | 55.7 | 64.0 | 48.0 | 51.4 | 32.9 |
| T→I | BLIP-2 | 40.7 | 77.0 | 58.0 | 60.3 | 41.6 |
| T→I | FLAVA | 46.8 | 72.0 | 61.0 | 55.6 | 36.9 |
| T→I | SigLIP2 | 32.5 | 75.0 | 69.0 | 62.7 | 49.2 |
| T→I | Qwen2.5-VL-FT | 40.1 | 71.0 | 56.0 | 60.0 | 42.1 |
| T→I | Qwen3-VL-Embed | 33.2 | 80.0 | 64.0 | 67.9 | 49.0 |
| I→T | CLIP | 61.9 | 54.0 | 31.0 | 51.3 | 28.0 |
| I→T | BLIP-2 | 32.5 | 75.0 | 69.0 | 62.2 | 48.6 |
| I→T | FLAVA | 57.2 | 61.0 | 48.0 | 46.3 | 29.9 |
| I→T | SigLIP2 | 35.2 | 78.0 | 69.0 | 64.8 | 48.4 |
| I→T | Qwen2.5-VL-FT | 33.1 | 78.0 | 70.0 | 61.5 | 48.8 |
| I→T | Qwen3-VL-Embed | 29.4 | 86.0 | 78.0 | 69.4 | 52.6 |

## 13. SugarCrepe negatives: best relevant caption against the probe's own decision

Image queries of the COCO pool. `Best relevant` scores each edited caption against the best of the image's relevant captions, as every condition of the matched-negative table does; `Own caption` is the probe's decision, each edited caption against the caption it was made from. p-values: paired randomisation test, Holm-corrected.

| Model | Best relevant | Own caption | All pooled | Self-mined | p (all vs own) | p (self vs own) |
|---|---|---|---|---|---|---|
| CLIP | 93.8 | 70.3 | 90.3 | 73.7 | 0.0001 | 1.00 |
| BLIP-2 | 91.2 | 69.1 | 88.8 | 64.2 | 0.0001 | 1.00 |
| FLAVA | 96.4 | 80.2 | 93.4 | 74.8 | 0.0001 | 1.00 |
| SigLIP2 | 94.6 | 77.2 | 95.5 | 84.3 | 0.0001 | 0.63 |
| Qwen2.5-VL-FT | 97.8 | 77.5 | 92.8 | 74.4 | 0.0001 | 1.00 |
| Qwen3-VL-Embed | 94.7 | 80.4 | 98.1 | 88.4 | 0.0001 | 0.22 |

## 14. Sensitivity to the pool depth

A pool of depth D keeps the judged candidates that at least one model ranked within its top D; self- and other-mined negatives are redefined for that depth. Depth 10 is the pool of the paper; depths 5 and 3 are shallower pools from the same labels. `gap` = handcrafted minus other-mined accuracy, with its Holm-corrected p-value. On COCO handcrafted negatives exist for image queries only.

| Benchmark | Dir. | Model | Other (D=3) | gap | p | Other (D=5) | gap | p | Other (D=10) | gap | p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SVO-Probes | T→I | CLIP | 69.6 | +16.2 | 0.0001 | 73.1 | +12.9 | 0.0002 | 77.4 | +9.3 | 0.0011 |
| SVO-Probes | T→I | BLIP-2 | 71.7 | +19.4 | 0.0001 | 75.1 | +15.4 | 0.0001 | 82.4 | +8.4 | 0.0011 |
| SVO-Probes | T→I | FLAVA | 74.7 | +16.2 | 0.0001 | 74.9 | +14.8 | 0.0001 | 79.8 | +10.2 | 0.0004 |
| SVO-Probes | T→I | SigLIP2 | 70.7 | +17.9 | 0.0001 | 74.0 | +15.2 | 0.0001 | 79.5 | +10.0 | 0.0011 |
| SVO-Probes | T→I | Qwen2.5-VL-FT | 75.1 | +15.8 | 0.0001 | 77.0 | +13.9 | 0.0001 | 83.6 | +7.5 | 0.0014 |
| SVO-Probes | T→I | Qwen3-VL-Embed | 79.5 | +15.1 | 0.0001 | 82.1 | +12.8 | 0.0001 | 85.4 | +9.7 | 0.0005 |
| SVO-Probes | I→T | CLIP | 69.5 | +17.7 | 0.0001 | 72.8 | +15.1 | 0.0001 | 76.8 | +10.9 | 0.0005 |
| SVO-Probes | I→T | BLIP-2 | 82.2 | +10.7 | 0.0003 | 83.1 | +9.9 | 0.0003 | 85.2 | +7.9 | 0.0015 |
| SVO-Probes | I→T | FLAVA | 73.7 | +10.6 | 0.0040 | 75.7 | +9.1 | 0.0044 | 79.0 | +6.2 | 0.0247 |
| SVO-Probes | I→T | SigLIP2 | 80.6 | +13.5 | 0.0002 | 86.0 | +8.5 | 0.0025 | 87.8 | +6.8 | 0.0105 |
| SVO-Probes | I→T | Qwen2.5-VL-FT | 78.5 | +14.3 | 0.0001 | 80.2 | +12.8 | 0.0001 | 84.2 | +9.0 | 0.0011 |
| SVO-Probes | I→T | Qwen3-VL-Embed | 81.5 | +13.8 | 0.0001 | 84.8 | +10.8 | 0.0001 | 88.0 | +7.7 | 0.0011 |
| COCO | I→T | CLIP | 95.6 | -2.3 | 1.0000 | 97.0 | -3.4 | 0.3665 | 97.4 | -3.6 | 0.1752 |
| COCO | I→T | BLIP-2 | 95.6 | -5.8 | 0.3162 | 97.3 | -6.5 | 0.0474 | 98.4 | -7.2 | 0.0069 |
| COCO | I→T | FLAVA | 96.8 | -0.8 | 1.0000 | 98.2 | -1.9 | 0.6082 | 99.7 | -3.4 | 0.0054 |
| COCO | I→T | SigLIP2 | 98.4 | -3.8 | 0.3162 | 98.6 | -4.1 | 0.1157 | 99.2 | -4.7 | 0.0069 |
| COCO | I→T | Qwen2.5-VL-FT | 96.7 | +1.1 | 1.0000 | 97.5 | +0.2 | 0.8499 | 98.0 | -0.2 | 0.8631 |
| COCO | I→T | Qwen3-VL-Embed | 99.4 | -5.1 | 0.0169 | 99.4 | -4.8 | 0.0451 | 99.8 | -5.1 | 0.0035 |

Self-mined negatives stay significantly harder than other-mined ones at every depth on both benchmarks.

## 15. Retrieval over the full datasets, including the mean reciprocal rank

Strict labels. MRR@20: mean reciprocal rank of the first relevant item, 0 beyond rank 20.

**SVO-Probes** (Kendall's τ between the model orderings by Success@1 and by MRR@20: T→I 0.87, I→T 1.00)

| Model | T→I S@1 | S@5 | S@10 | MRR@20 | I→T S@1 | S@5 | S@10 | MRR@20 |
|---|---|---|---|---|---|---|---|---|
| CLIP | 8.7 | 23.8 | 32.8 | 15.9 | 6.5 | 20.5 | 30.3 | 13.4 |
| BLIP-2 | 9.7 | 27.1 | 37.9 | 18.1 | 13.2 | 35.3 | 47.1 | 23.5 |
| FLAVA | 8.6 | 24.3 | 34.7 | 16.3 | 5.6 | 17.2 | 24.9 | 11.4 |
| SigLIP2 | 10.4 | 28.5 | 39.1 | 19.0 | 13.4 | 35.9 | 48.1 | 23.9 |
| Qwen2.5-VL-FT | 12.2 | 31.7 | 42.6 | 21.4 | 13.2 | 35.0 | 46.7 | 23.4 |
| Qwen3-VL-Embed | 14.4 | 36.9 | 49.0 | 24.8 | 16.8 | 41.0 | 53.3 | 28.0 |

**COCO val2017** (Kendall's τ between the model orderings by Success@1 and by MRR@20: T→I 1.00, I→T 1.00)

| Model | T→I S@1 | S@5 | S@10 | MRR@20 | I→T S@1 | S@5 | S@10 | MRR@20 |
|---|---|---|---|---|---|---|---|---|
| CLIP | 36.0 | 61.0 | 71.0 | 47.3 | 57.0 | 79.8 | 86.9 | 67.1 |
| BLIP-2 | 30.3 | 52.5 | 62.6 | 40.6 | 40.4 | 68.1 | 78.7 | 52.8 |
| FLAVA | 41.5 | 71.3 | 81.7 | 54.7 | 48.8 | 80.1 | 89.9 | 62.4 |
| SigLIP2 | 48.5 | 73.1 | 81.5 | 59.5 | 64.2 | 85.6 | 91.2 | 73.5 |
| Qwen2.5-VL-FT | 37.9 | 63.2 | 73.5 | 49.4 | 51.8 | 76.8 | 84.9 | 62.7 |
| Qwen3-VL-Embed | 54.8 | 79.4 | 86.9 | 65.7 | 71.2 | 89.9 | 94.6 | 79.4 |
