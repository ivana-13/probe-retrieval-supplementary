"""Embed the COCO val2017 images, the val2017 captions and the SugarCrepe negative captions with one encoder.

Usage: python scripts/20_coco_embed.py CLIP|FLAVA|SigLIP2|BLIP2   (BLIP2 runs in the LAVIS environment)
Outputs results/coco_emb/{MODEL}_{caption,image,negative}_embeddings.npy plus index arrays. Scores on this collection
are cosine similarities (all embeddings are normalised when loaded)."""
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from svo_eval.paths import RESULTS
from svo_eval.coco import load_coco

MODEL = sys.argv[1]
OUT = RESULTS / "coco_emb"
OUT.mkdir(parents=True, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
coll = load_coco()
captions = list(coll.captions)
# probe texts (positive or negative captions) that are not among the collection captions
negatives = sorted(({p["negative_caption"] for p in coll.probe} | {p["caption"] for p in coll.probe}) - set(captions))
image_ids = list(coll.images)

if MODEL == "CLIP":
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-L-14")
    model = model.to(device).eval()

    @torch.no_grad()
    def enc_text(batch):
        return model.encode_text(tok(batch).to(device)).float().cpu().numpy()

    @torch.no_grad()
    def enc_img(imgs):
        return model.encode_image(torch.stack([preprocess(i) for i in imgs]).to(device)).float().cpu().numpy()

elif MODEL == "FLAVA":
    from transformers import FlavaModel, FlavaImageProcessor, BertTokenizer
    model = FlavaModel.from_pretrained("facebook/flava-full").to(device).eval()
    fe = FlavaImageProcessor.from_pretrained("facebook/flava-full")
    tok = BertTokenizer.from_pretrained("facebook/flava-full")

    @torch.no_grad()
    def enc_text(batch):
        out = [model.get_text_features(**tok(text=[t], return_tensors="pt", truncation=True, max_length=77).to(device))[:, 0, :] for t in batch]
        return torch.cat(out).float().cpu().numpy()

    @torch.no_grad()
    def enc_img(imgs):
        return model.get_image_features(**fe(images=imgs, return_tensors="pt").to(device))[:, 0, :].float().cpu().numpy()

elif MODEL == "SigLIP2":
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip2-base-patch16-224"
    model = AutoModel.from_pretrained(name).to(device).eval()
    proc = AutoProcessor.from_pretrained(name)

    @torch.no_grad()
    def enc_text(batch):
        inp = proc(text=batch, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
        return model.get_text_features(**inp).float().cpu().numpy()

    @torch.no_grad()
    def enc_img(imgs):
        return model.get_image_features(**proc(images=imgs, return_tensors="pt").to(device)).float().cpu().numpy()

elif MODEL == "BLIP2":
    from lavis.models import load_model_and_preprocess
    model, vis_proc, txt_proc = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain",
                                                          is_eval=True, device=torch.device(device))

    @torch.no_grad()
    def enc_text(batch):
        feats = model.extract_features({"text_input": [txt_proc["eval"](t) for t in batch]}, mode="text")
        return feats.text_embeds_proj[:, 0, :].float().cpu().numpy()

    @torch.no_grad()
    def enc_img(imgs):
        img = torch.stack([vis_proc["eval"](i) for i in imgs]).to(device)
        return model.extract_features({"image": img}, mode="image").image_embeds_proj[:, 0, :].float().cpu().numpy()
else:
    raise SystemExit("unknown model")

BS = 32


def encode_texts(texts, desc):
    return np.concatenate([enc_text(texts[i:i + BS]) for i in tqdm(range(0, len(texts), BS), desc=desc)])


cap_emb = encode_texts(captions, f"{MODEL} captions")
neg_emb = encode_texts(negatives, f"{MODEL} negatives") if negatives else np.zeros((0, cap_emb.shape[1]), np.float32)
img_emb, failed = [], []
for i in tqdm(range(0, len(image_ids), BS), desc=f"{MODEL} images"):
    ids = image_ids[i:i + BS]
    imgs = [Image.open(coll.image_path(iid)).convert("RGB") for iid in ids]
    try:
        img_emb.append(enc_img(imgs))
    except Exception:
        rows = []
        for iid, im in zip(ids, imgs):
            try:
                rows.append(enc_img([im])[0])
            except Exception:
                failed.append(iid)
                rows.append(1e-5 * np.ones(cap_emb.shape[1], np.float32))
        img_emb.append(np.stack(rows))
img_emb = np.concatenate(img_emb)
np.save(OUT / f"{MODEL}_caption_embeddings.npy", cap_emb.astype(np.float32))
np.save(OUT / f"{MODEL}_negative_embeddings.npy", neg_emb.astype(np.float32))
np.save(OUT / f"{MODEL}_image_embeddings.npy", img_emb.astype(np.float32))
np.save(OUT / f"{MODEL}_captions.npy", np.array(captions, dtype=str))
np.save(OUT / f"{MODEL}_negatives.npy", np.array(negatives, dtype=str))
np.save(OUT / f"{MODEL}_image_id.npy", np.array(image_ids, dtype=int))
print(MODEL, cap_emb.shape, neg_emb.shape, img_emb.shape, "failed images:", failed)
