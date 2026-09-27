"""Quick inference demo using the native transformers SAM 3 pipeline.

The published Hub API is the same class from hub/pipeline.py:

    from transformers import pipeline
    pipe = pipeline("segearth-ov3-segmentation", model="YOUR_NAMESPACE/SegEarth-OV-3",
                    trust_remote_code=True, text_prompts=[...])
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from segearthov3 import pipeline

img_path = "resources/oem_koeln_50.tif"

# A list of text prompts — not a dataset-specific cls_*.txt file.
# Comma-separated strings (or nested lists) are synonym groups for one class.
text_prompts = [
    "background",
    "bareland,barren",
    "grass",
    "road",
    "car",
    "tree,forest",
    "water,river",
    "cropland",
    "building,roof,house",
]

pipe = pipeline(
    model="facebook/sam3",
    text_prompts=text_prompts,
    prob_thd=0.1,
    confidence_threshold=0.1,
    slide_stride=512,
    slide_crop=512,
)

result = pipe(img_path)
seg_pred = result["pred_sem_seg"]

img = Image.open(img_path).convert("RGB")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(img)
ax[0].axis("off")
ax[1].imshow(seg_pred, cmap="viridis")
ax[1].axis("off")
plt.tight_layout()
out_path = Path("seg_pred.png")
plt.savefig(out_path, bbox_inches="tight")
print(f"Saved {out_path} with classes: {result['text_prompts']}")
