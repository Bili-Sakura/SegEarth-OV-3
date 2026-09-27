import os

import numpy as np
import torch
from PIL import Image

from segearthov3 import pipeline
from segearthov3.vocabularies import STPLS3D

"""
[Split Step 1] 2D Inference Script
Function: Load images, run large model inference, and save the results as 8-bit single-channel PNG mask images.
"""


def init_ovss_pipeline():
    print("[*] Initializing SegEarth-OV3 transformers pipeline...")
    return pipeline(
        model="facebook/sam3",
        text_prompts=STPLS3D,
        prob_thd=0.1,
        confidence_threshold=0.1,
        bg_idx=5,
    )


def main():
    img_dir = "dataset/3D/WMSC"
    out_masks_dir = "./work_dirs/WMSC_2d_masks"

    os.makedirs(out_masks_dir, exist_ok=True)
    pipe = init_ovss_pipeline()

    image_files = [
        name
        for name in os.listdir(img_dir)
        if name.lower().endswith((".jpg", ".png", ".tif", ".tiff"))
    ]

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(img_dir, img_name)
        mask_filename = os.path.splitext(img_name)[0] + ".png"
        out_mask_path = os.path.join(out_masks_dir, mask_filename)

        if os.path.exists(out_mask_path):
            print(f"  [{idx + 1}/{len(image_files)}] {img_name} mask already exists, skipping.")
            continue

        print(f"  [{idx + 1}/{len(image_files)}] Inference: {img_name}")
        with torch.no_grad():
            result = pipe(Image.open(img_path).convert("RGB"))
        label_map_2d = result["pred_sem_seg"].astype(np.uint8)
        Image.fromarray(label_map_2d).save(out_mask_path)

    print("\nAll 2D inferences completed, masks saved to:", out_masks_dir)


if __name__ == "__main__":
    main()
