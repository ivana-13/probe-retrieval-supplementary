from svo_eval import pools

fp = pools.save_fixed_pool()
for d in fp:
    n = sum(len(v) for v in fp[d].values())
    pos = sum(e["correct"] for v in fp[d].values() for e in v.values())
    dis = sum(1 for v in fp[d].values() for e in v.values() if e["labels"].get("disagreement"))
    sizes = [len(v) for v in fp[d].values()]
    print(d, "pairs", n, "correct", pos, "disagreements across files", dis,
          "pool size per query: min %d median %d max %d" % (min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes)))
print("Fleiss kappa i2t (4 dual encoders, 3 annotators):", round(pools.fleiss_kappa_i2t(), 4))
