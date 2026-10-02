import json
from functools import lru_cache
from .paths import list_file
from .metrics import first_gold_rank


@lru_cache(maxsize=None)
def load_lists(model, direction):
    raw = json.load(open(list_file(model, direction), encoding="utf-8"))
    if direction == "t2i":
        return {cap: [int(x) for x in lst] for cap, lst in raw.items()}
    return {int(img): [str(x) for x in lst] for img, lst in raw.items()}


def ranks_from_lists(model, direction, coll):
    lists = load_lists(model, direction)
    if direction == "t2i":
        return {cap: first_gold_rank(lists[cap], coll.cap2imgs[cap]) for cap in coll.captions if cap in lists}
    return {img: first_gold_rank(lists[img], coll.img2caps[img])
            for img in coll.images if coll.img2caps.get(img) and img in lists}
