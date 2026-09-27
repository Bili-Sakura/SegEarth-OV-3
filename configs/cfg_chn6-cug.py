_base_ = './base_config.py'

from segearthov3.vocabularies import CHN6_CUG

# model settings
model = dict(
    text_prompts=CHN6_CUG,
    prob_thd=0.3,
    confidence_threshold=0.5,
)

# dataset settings
dataset_type = 'CHN6_CUGDataset'
data_root = 'data/CHN6-CUG'

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
            img_path='val/image_cvt',
            seg_map_path='val/label_cvt'),
        pipeline=test_pipeline))