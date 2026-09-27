<div align="center">

<h1>SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images</h1>

<!-- <h3></h3> -->

<div>
    <strong>Adapting SAM 3 for remote sensing open-vocabulary semantic segmentation, change detection, and 3D semantic segmentation</strong>
</div>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://github.com/bavarianvilliager' target='_blank'>Shengqi Zhang</a><sup>1</sup>&emsp;
    <a href='https://github.com/acousmasyndrome' target='_blank'>Yujie Wang</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=Lmqy-D4AAAAJ&hl=zh-CN&oi=ao' target='_blank'>Yupeng Deng</a><sup>2</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Chinese Academy of Sciences&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://github.com/earth-insights/SegEarth-OV-3" target='_blank'>[Code]</a> • <a href="https://arxiv.org/abs/2512.08730" target='_blank'>[arXiv]</a> • <a href="https://github.com/earth-insights/SegEarth-OV-3/blob/main/demo.py" target='_blank'>[Demo]</a> •
    </h4>
</div>

</div>

<img src="resources/vis.png" width="100%"/>

> Inference results of SegEarth-OV3 on a remote sensing image exceeding 10k×10k resolution. The image originates from [OpenMapCD](https://zenodo.org/records/14028095).

<img src="https://github.com/user-attachments/assets/d17ce794-9cd8-47cc-8b2c-a9b3e4739a10" width="100%"/>

> The overall inference pipeline of SegEarth-OV3. Given an input image and a list of text prompts, we leverage SAM 3's decoupled outputs. The pipeline involves: (1) instance aggregation to consolidate sparse object predictions; (2) dual-head mask fusion to combine the fine-grained instance details with the global coverage of the semantic head; and (3) presence-guided filtering (using the presence score) to suppress false positives from absent categories. "MAX" denotes the element-wise maximum operation, and "×" denotes multiplication.

## Abstract
> *Most existing methods for training-free open-vocabulary semantic segmentation are based on CLIP. While these approaches have made progress, they often face challenges in precise localization or require complex pipelines to combine separate modules, especially in remote sensing scenarios where numerous dense and small targets are present. Recently, Segment Anything Model 3 (SAM 3) was proposed, unifying segmentation and recognition in a promptable framework. In this paper, we present a comprehensive exploration of applying SAM 3 to the remote sensing open-vocabulary tasks (\textit{i.e.}, 2D semantic segmentation, change detection, and 3D semantic segmentation) without any training. First, we implement a mask fusion strategy that combines the outputs from SAM 3's semantic segmentation head and the Transformer decoder (instance head). This allows us to leverage the strengths of both heads for better land coverage. Second, we utilize the presence score from the presence head to filter out categories that do not exist in the scene, reducing false positives caused by the vast vocabulary sizes and patch-level processing in geospatial scenes. Furthermore, we extend our method to open-vocabulary change detection by a joint instance- and pixel-level verification strategy built directly upon our fused logits. We evaluate our method on extensive remote sensing datasets and tasks, including 20 segmentation datasets, 3 change detection datasets, and a 3D segmentation dataset. Experiments show that our method achieves promising performance, demonstrating the potential of SAM 3 for remote sensing open-vocabulary tasks.*

## Dependencies and Installation

SAM 3 is loaded **natively through Hugging Face `transformers`** (`Sam3Model` / `Sam3Processor`). This repo no longer vendors or imports the `facebookresearch/sam3` package.

```bash
pip install -r requirements.txt
# Evaluation also needs a working mmcv + mmsegmentation install.
```

`facebook/sam3` is a gated Hub model. Accept the license on the [model card](https://huggingface.co/facebook/sam3) and authenticate:

```bash
hf auth login
```

## Datasets
We include the following dataset configurations in this repo: 
1) `Semantic Segmentation`: OpenEarthMap, LoveDA, iSAID, Potsdam, Vaihingen, UAVid<sup>img</sup>, UDD5, VDD
2) `Building Extraction`: WHU<sup>Aerial</sup>, WHU<sup>Sat.Ⅱ</sup>, Inria, xBD<sup>pre</sup>
4) `Road Extraction`: CHN6-CUG, DeepGlobe, Massachusetts, SpaceNet
5) `Water Extraction`: WBS-SI
6) `Gaofen Series Data`: GID, GF-7 Building Dataset, Low-Grade Road Dataset
7) `Change Detection`: LEVIR-CD, WHU-CD, S2Looking
8) `3D Segmentation`: STPLS3D (WMSC)

For 1) - 4), please refer to [SegEarth-OV/dataset_prepare.md](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md) for dataset preparation.  
For 6) - 8). please refer to [dataset_prepare.md](dataset_prepare.md) for dataset preparation.

## Load SAM 3

The custom inference pipeline downloads and caches `facebook/sam3` via `transformers` on first use. You can also point `model=` at a local snapshot of that Hub repo. A separate `sam3.pt` / BPE file is not required.

## Hugging Face Hub (standard custom pipeline)

The publishable Transformers snapshot is [`hub/`](hub/). That folder is laid out the official way ([Adding a new pipeline](https://huggingface.co/docs/transformers/en/add_new_pipeline)):

```
hub/pipeline.py   # self-contained transformers.Pipeline (no package imports)
hub/config.json   # Sam3 config + custom_pipelines + default text_prompts
hub/README.md     # model card
```

`config.json` registers the task as:

```json
"custom_pipelines": {
  "segearth-ov3-segmentation": {
    "impl": "pipeline.SegEarthOV3Pipeline",
    "pt": ["AutoModel"],
    "type": "image"
  }
}
```

After you upload that snapshot to a model repo (`python scripts/push_to_hub.py --repo-id YOUR_NAMESPACE/SegEarth-OV-3 --code-only` or `--with-weights`):

```python
from transformers import pipeline

pipe = pipeline(
    "segearth-ov3-segmentation",
    model="YOUR_NAMESPACE/SegEarth-OV-3",
    trust_remote_code=True,
    text_prompts=["background", "building", "road"],
)
result = pipe("image.tif")
```

`--with-weights` is the official `Pipeline.push_to_hub` path (copies `pipeline.py` next to the SAM 3 checkpoint). `--code-only` uploads just the three files above; pair them with `facebook/sam3` as shown in `hub/README.md`. Pin `revision=` to a commit you have reviewed when using `trust_remote_code=True`. SAM 3 weights stay under the [`facebook/sam3`](https://huggingface.co/facebook/sam3) license.

This GitHub checkout uses the same `hub/pipeline.py` through `from segearthov3 import pipeline`.

## Quick Inference

The inference API takes a **list of text prompts** (one entry per class). Comma-separated strings or nested lists are synonym groups for the same class — they are queried separately and max-pooled back to one label.

```python
from segearthov3 import pipeline

pipe = pipeline(
    model="facebook/sam3",
    text_prompts=[
        "background",
        "bareland,barren",
        "grass",
        "road",
        "car",
        ["tree", "forest"],
        "water,river",
        "cropland",
        "building,roof,house",
    ],
    slide_crop=512,
    slide_stride=512,
    prob_thd=0.1,
    confidence_threshold=0.1,
)

result = pipe("resources/oem_koeln_50.tif")
# result["pred_sem_seg"]: [H, W] class ids
# result["seg_logits"]:   [C, H, W]
# result["text_prompts"]: class names after synonym grouping
```

Or run the demo script:

```
python demo.py
```

Preset evaluation vocabularies live in `segearthov3/vocabularies.py` as Python lists (not `configs/cls_*.txt` files). Override them per call with any custom `text_prompts=[...]`.

## Model evaluation

```
python eval.py ./configs/cfg_DATASET.py
```

Each `configs/cfg_*.py` now sets `model.text_prompts` to a list (imported from `segearthov3.vocabularies`).

## Results
<div>
<img src="https://github.com/user-attachments/assets/ae2895a9-0225-4cda-8ea0-3a5ee4e06224" width="80%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/fe56af77-6d16-45da-b705-4c02478c0c5e" width="80%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/7cae3ccd-82b7-42d3-a840-95ffaf09a5d4" width="80%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/65fdd629-83ed-441e-8c4d-3e09f9839c31" width="80%"/>
</div>

## Citation

```
@article{li2025segearthov3,
  title={SegEarth-OV3: Exploring SAM 3 for Open-Vocabulary Semantic Segmentation in Remote Sensing Images},
  author={Li, Kaiyu and Zhang, Shengqi and Wang, Yujie and Deng, Yupeng and Wang, Zhi and Meng, Deyu and Cao, Xiangyong},
  journal={arXiv preprint arXiv:2512.08730},
  year={2025}
}
```

## Acknowledgement
This implementation uses [Hugging Face Transformers SAM 3](https://huggingface.co/docs/transformers/en/model_doc/sam3) and is based on the [SAM 3](https://github.com/facebookresearch/sam3) paper/model and [SCLIP](https://github.com/wangf3014/SCLIP). We would also like to thank Xu Zhang for providing the [OmniOVCD](https://github.com/Erxucomeon/OmniOVCD) code, which forms the basis of the OVCD part in this code.

