_base_ = './base_config.py'

from segearthov3.vocabularies import COCO_STUFF

# model settings
model = dict(
    text_prompts=COCO_STUFF,
    confidence_threshold=0.2,
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = './data/COCOStuff'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))