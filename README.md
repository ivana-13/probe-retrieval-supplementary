# Supplementary repository

Judged retrieval pools, annotation material and analysis code for an anonymous submission on image–text matching
probes and retrieval.

The paper treats two probing benchmarks, SVO-Probes and SugarCrepe over COCO, as retrieval test collections. For 100
caption queries and 100 image queries per benchmark, the top-10 results of six models were labelled, and the same
two-way decision is then evaluated against handcrafted, random and retrieval-mined negatives.

## What is here

```
svo_probes/     SVO-Probes side: the filtered benchmark file and the judged top-10 lists of the six models
pipeline/
  svo_eval/     analysis package (datasets, embeddings and scoring, pools, metrics, matching heads)
  scripts/      numbered scripts, from embedding extraction to tables and figures
  results/      every result file behind the paper's numbers, per-query values and model decisions
  paper/        the generated LaTeX table bodies and figures
  qwen25/       fine-tuning and prompting scripts of the Qwen2.5-VL models
  tests/        unit tests of the package
docs/
  annotation_instructions.md   who labelled what, and the instructions the annotators received
  assessor_prompts.md          the prompts of the LLM assessors, word for word
  implementation_details.md    checkpoints, prompts, hyperparameters, scoring, statistics
  metric_definitions.md        exact definition of every reported quantity
  extra_results.md             tables that did not fit into the paper
```

### Judged pools

- **SVO-Probes** (`svo_probes/*_selected_samples_{images,text}_retrieval_evaluation.json`): for each of the 100 queries
  of a direction, the model's top-10 list with the human labels (`human evaluation`: `correct`, `subject incorrect`,
  `verb incorrect`, `object incorrect`; `human evaluation 2` / `3`: the votes of the second and third annotator where
  three annotators labelled the list). The `..._o3_experiment.json` files add the label of the o3 assessor
  (`o3 evaluation`), and for the four dual encoders that of GPT-4o (`GPT evaluation`). `images` = text-to-image
  retrieval (caption queries), `text` = image-to-text retrieval (image queries). 7,428 distinct pairs in total.
- **The union pool** (`pipeline/results/fixed_pool.json`): per direction and query, every judged candidate with its
  label, error type, the models that retrieved it and their ranks.
- **COCO** (`pipeline/results/coco_pool_candidates.json`, `coco_pool_labels_o3.json`, `coco_pool_labels_human.json`):
  the pooled top-10 candidates of the six models for 100 queries per direction (6,562 pairs), the o3 labels of all of
  them and the expert's labels for the validation sample of 12 queries per direction.

Images are not redistributed. SVO-Probes image ids follow the benchmark's `pos_image_id` / `neg_image_id`; COCO ids are
those of val2017.

## Checking the paper's numbers without any model

`pipeline/results/` contains the per-query values of every condition and `svo_probes/` the labelled lists, so the
pool, the assessor validation and the tables can be regenerated directly (run from `pipeline/` with that folder on
`PYTHONPATH`):

```
cd pipeline
pip install -r requirements.txt
python scripts/02_build_pool.py        # the union pool from the labelled lists, inter-annotator agreement
python scripts/09_judge_validation.py  # LLM assessors against the human labels
python scripts/34_tables.py            # matched-negative, retrieval and assessor tables, n-way figure, text ranges
python scripts/12_make_tables.py       # probe, ITM-pool and assessor-validation tables of SVO-Probes
```

`scripts/31_coco_tables.py` does the same for COCO once the SugarCrepe files are in `pipeline/data/coco/`.

## Recomputing from the models

1. Obtain the data: the SVO-Probes images (the URLs are in the original benchmark; save them as
   `svo_probes/images/<image_id>.jpg`), COCO val2017 with its captions, and the SugarCrepe files (`pipeline/data/coco/`).
2. Extract embeddings: `scripts/00_extract_embeddings.py` (SVO-Probes) and `scripts/20_coco_embed.py`,
   `25_coco_embed_qwen3.py` (COCO). BLIP-2 uses LAVIS in a separate environment (`scripts/README_lavis.md`). The
   Qwen2.5-VL-FT adapter is trained with `qwen25/qwen_fine-tunning.py`.
3. Probe and retrieval metrics: `03_metrics_full.py`, `03b_itm_benchmark.py`, `21_coco_metrics.py`.
4. Pools and labels: `02_build_pool.py` assembles the SVO-Probes pool from the labelled lists; on COCO
   `21_coco_metrics.py` writes the pool candidates, `23_coco_judge_o3.py` labels them with the assessor and
   `28_coco_human_sample.py` draws and imports the human validation sample.
5. Analyses: `16_svo_analyses.py long`, `33_coco_analyses.py` (matched negatives, tests, n-way curves, probe against
   retrieval outcome), `06_cross_model.py`, `07_itm_pools.py`, `30_coco_itm_pool_eval.py`, `09_judge_validation.py`,
   `10_error_analysis.py`, `15_pooling_bias.py`, `36_coco_human_check.py`, `37_probe_vs_retrieval_order.py`,
   `38_itm_head_auc.py`.
6. Tables and figures: `12_make_tables.py`, `13_make_figures.py`, `31_coco_tables.py`, `34_tables.py`.

Embeddings, model checkpoints and the Qwen2.5-VL-FT adapter are not part of this repository because of their size; the
unit tests in `pipeline/tests/` that score pairs need the extracted embeddings.

## Data sources and terms

SVO-Probes (Hendricks and Nematzadeh, 2021), MS-COCO (Lin et al., 2014) and SugarCrepe (Hsieh et al., 2023) remain
under the terms of their providers; `svo_probes/filtered_svo.csv` is a row subset of the SVO-Probes benchmark file. The
licence of the labels and code in this repository will be stated with the camera-ready version.
