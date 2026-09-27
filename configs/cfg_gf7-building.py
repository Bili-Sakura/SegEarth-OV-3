_base_ = './base_config.py'

from segearthov3.vocabularies import GF7_BUILDING

# model settings
model = dict(
    text_prompts=GF7_BUILDING,
    prob_thd=0.5,
    confidence_threshold=0.1,
)

# dataset settings
dataset_type = 'GF7BuildingDataset'
data_root = 'data/GF7-building'

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
            img_path='Test/image',
            seg_map_path='Test/label_cvt'),
        pipeline=test_pipeline))