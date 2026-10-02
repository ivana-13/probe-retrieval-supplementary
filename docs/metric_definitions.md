# Metric definitions

## Datasets and directions

- **SVO-Probes**: the 10,978 captions and 13,285 images of `svo_probes/filtered_svo.csv` (33,685 triplets; the rows
  of the original benchmark whose images could still be downloaded). A caption is strictly relevant to each image
  listed as its `pos_image_id`; an image is strictly relevant to each caption for which it is the `pos_image_id`.
- **COCO with SugarCrepe**: the 5,000 images of COCO val2017 and their 24,789 distinct captions; SugarCrepe supplies
  7,511 probe items (image, caption, negative caption).
- **Text-to-image (T→I)**: query = caption, candidates = all images. **Image-to-text (I→T)**: query = image,
  candidates = all captions (on SVO-Probes the 11,455 images with at least one caption are queries).

## Probe and retrieval measures

- **Success@K**: share of queries whose first strictly relevant candidate is ranked within the top K.
- **Pairwise accuracy**: share of probe items in which the positive pair scores above the negative pair.
- **ITM accuracy**: share of positive and negative pairs that the binary matching decision classifies correctly
  (67,370 pairs on SVO-Probes, 15,022 on SugarCrepe). Unanswered items of the prompted model are excluded and counted,
  never scored as correct or as wrong.
- **Success@1 on the pool queries**: *strict* = the first strictly relevant item has rank 1; *human* / *o3* = the label
  of the model's top-ranked item. The o3 value is computed over the queries whose top-ranked item received an o3 label.

## Matched negatives

For a query q with relevant set R(q) and a set of negatives N(q), the accuracy is the mean over c in N(q) of
1[max over p in R(q) of s(q, p) > s(q, c)], averaged over the 100 pool queries of a direction. Sources of N(q):

- `benchmark`: the benchmark's handcrafted negatives of the query (SVO-Probes: the negative images of the caption's
  triplets; for an image query, the captions written for the negative images of its triplets. SugarCrepe: the negative
  captions of the image; there are none for caption queries). `benchmark_own` scores each handcrafted negative against
  its own triplet's positive instead of the best relevant item (at most 1.3 points apart on SVO-Probes).
- `random`: ten items drawn uniformly from the dataset without R(q), averaged over 20 draws; the generator is seeded by
  the md5 of (direction, query, "random"), so every model sees the same draws.
- `pool_other`: judged-incorrect pool items that the scored model did not retrieve in its own top 10.
- `pool_self`: judged-incorrect pool items from the scored model's own top 10.
- `pool_all`: all judged-incorrect pool items of the query.

**Positive outside own top 10**: the queries whose first strictly relevant item has rank > 10 for the scored model. On
all other queries the other-mined value is 1 by construction (checked: no violations on either benchmark).

**Tests**: two-sided paired randomisation (sign-flip) test on the per-query differences, 100,000 permutations, seed 0;
Holm correction over the model-and-direction cells of a comparison.

**Robustness variants on SVO-Probes**: `pool_other_dual` (without items retrieved only by the two Qwen models),
`pool_other_unanimous` (image queries; only items that all three annotators marked incorrect in every list that carries
three votes), `pool_other_consistent` (without pairs labelled differently in two lists).

## n-way curves

Share of queries on which the best relevant item outranks all of k negatives. Random: 50 draws of k dataset items.
Other-mined: the query's other-mined negatives are added in a random order (nested prefixes), averaged over 50 orders,
k in {1, 2, 5, 10}. Both curves use the queries with at least ten other-mined negatives for the scored model. The
number of random negatives equivalent to ten other-mined ones is read off the random curve by log-linear interpolation.

## Probe outcome and retrieval outcome

A query passes the probe when every handcrafted decision of the query is correct (each triplet's own positive above
its negative). The retrieval outcome is the label of the model's top-ranked item. Association: Fisher's exact test per
model and direction, Holm-corrected.

## Cross-model matrix

For scorer A and miner B: accuracy of A on the judged-incorrect items that B retrieved in its top 10.

## Assessor agreement

Accuracy, precision and recall of the `correct` class, its F1 and Cohen's kappa between the assessor's label and the
human label (majority vote where three annotators labelled a pair), on the pairs for which the assessor returned a
parseable label; four-way kappa on the labels correct / subject / verb / object incorrect.
