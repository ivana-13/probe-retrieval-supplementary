"""Re-extract caption and image embeddings for CLIP, FLAVA or SigLIP2 with the checkpoints used in the papers."""
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from svo_eval.paths import EMB, IMAGES_DIR
from svo_eval.collection import load_collection

MODEL = sys.argv[1]            # CLIP | FLAVA | SigLIP2
device = "cuda" if torch.cuda.is_available() else "cpu"
coll = load_collection()
captions = list(coll.captions)
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
        out = [model.get_text_features(**tok(text=[t], return_tensors="pt").to(device))[:, 0, :] for t in batch]
        return torch.cat(out).float().cpu().numpy()

    @torch.no_grad()
    def enc_img(imgs):
        out = model.get_image_features(**fe(images=imgs, return_tensors="pt").to(device))[:, 0, :]
        return out.float().cpu().numpy()

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
        inp = proc(images=imgs, return_tensors="pt").to(device)
        return model.get_image_features(**inp).float().cpu().numpy()
else:
    raise SystemExit("unknown model")

BS = 32
cap_emb = np.concatenate([enc_text(captions[i:i + BS]) for i in tqdm(range(0, len(captions), BS), desc=f"{MODEL} captions")])
dim = cap_emb.shape[1]
img_emb, failed = [], []
for i in tqdm(range(0, len(image_ids), BS), desc=f"{MODEL} images"):
    batch_ids = image_ids[i:i + BS]
    imgs = [Image.open(IMAGES_DIR / f"{iid}.jpg").convert("RGB") for iid in batch_ids]
    try:
        img_emb.append(enc_img(imgs))
    except Exception:
        rows = []
        for iid, im in zip(batch_ids, imgs):
            try:
                rows.append(enc_img([im])[0])
            except Exception:
                failed.append(iid)
                rows.append(1e-5 * np.ones(dim, dtype=np.float32))
        img_emb.append(np.stack(rows))
img_emb = np.concatenate(img_emb)
np.save(EMB / f"{MODEL}_caption_embeddings.npy", cap_emb.astype(np.float32))
np.save(EMB / f"{MODEL}_image_embeddings.npy", img_emb.astype(np.float32))
np.save(EMB / f"{MODEL}_captions.npy", np.array(captions, dtype=str))
np.save(EMB / f"{MODEL}_image_id.npy", np.array(image_ids, dtype=int))
print(MODEL, cap_emb.shape, img_emb.shape, "failed images:", failed)
