_base_ = './base_config.py'

from segearthov3.vocabularies import GF_ROAD

# model settings
model = dict(
    text_prompts=GF_ROAD,
    prob_thd=0.3,
    confidence_threshold=0.1,
)

# dataset settings
dataset_type = 'GFRoadDataset'
data_root = 'data/GF_LowGradeRoadDataset'

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
            img_path='test/image',
            seg_map_path='test/label'),
        pipeline=test_pipeline))