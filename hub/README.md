---
license: mit
base_model: facebook/sam3
pipeline_tag: image-segmentation
library_name: transformers
tags:
  - sam3
  - remote-sensing
  - open-vocabulary
  - semantic-segmentation
  - custom_code
---

# SegEarth-OV3

Training-free open-vocabulary semantic segmentation for remote sensing images. This Hub repo is a **standard Transformers custom pipeline** ([docs](https://huggingface.co/docs/transformers/en/add_new_pipeline)) on top of native [`facebook/sam3`](https://huggingface.co/facebook/sam3) (`Sam3Model` / `Sam3Processor`). There is no `facebookresearch/sam3` package.

Pipeline code is MIT. SAM 3 weights follow the [`facebook/sam3`](https://huggingface.co/facebook/sam3) license.

## Files

```
pipeline.py     # transformers.Pipeline subclass (self-contained, no package imports)
config.json     # Sam3 config + custom_pipelines + default text_prompts
```

`config.json` registers the task the official way:

```json
"custom_pipelines": {
  "segearth-ov3-segmentation": {
    "impl": "pipeline.SegEarthOV3Pipeline",
    "pt": ["AutoModel"],
    "type": "image"
  }
}
```

## Usage

After this repo is published (pipeline code + SAM 3 weights, or your local snapshot):

```python
from transformers import pipeline

pipe = pipeline(
    "segearth-ov3-segmentation",
    model="YOUR_NAMESPACE/SegEarth-OV-3",
    trust_remote_code=True,
    text_prompts=["background", "building,house", "road"],
)
result = pipe("image.tif")
# result["pred_sem_seg"]  # [H, W] class ids
# result["seg_logits"]    # [C, H, W]
# result["text_prompts"]  # class names after synonym grouping
```

If only the code files from this folder are on the Hub and weights stay on `facebook/sam3`:

```python
from transformers import pipeline, Sam3Model, Sam3Processor
from huggingface_hub import hf_hub_download
import importlib.util

model = Sam3Model.from_pretrained("facebook/sam3")
processor = Sam3Processor.from_pretrained("facebook/sam3")
src = hf_hub_download("YOUR_NAMESPACE/SegEarth-OV-3", "pipeline.py")
spec = importlib.util.spec_from_file_location("segearth_ov3_pipeline", src)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

pipe = mod.SegEarthOV3Pipeline(
    model=model,
    tokenizer=processor.tokenizer,
    image_processor=processor.image_processor,
    processor=processor,
    text_prompts=["building", "road", "water"],
)
```

`facebook/sam3` is gated: accept the license and run `hf auth login`.

## Prompts

Pass a **list** of class names. Comma-separated strings or nested lists are synonyms for one class and are max-pooled after inference.

```python
pipe(image, text_prompts=["background", ["building", "house"], "road"])
```

Defaults live in `config.json` as `text_prompts` (OpenEarthMap-style land cover) and are used when you omit the argument. Pin `revision=` to a commit you have reviewed when using `trust_remote_code=True`.

## Fusion

Per prompt the pipeline fuses SAM 3's instance masks, semantic head, and presence score (SegEarth-OV3). Vision embeddings are computed once and reused across the prompt list.

## Citation

```bibtex
@article{li2025segearthov3,
  title={SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images},
  author={Li, Kaiyu and Zhang, Shengqi and Wang, Yujie and Deng, Yupeng and Wang, Zhi and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.08730},
  year={2025}
}
```
