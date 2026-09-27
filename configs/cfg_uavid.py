_base_ = './base_config.py'

from segearthov3.vocabularies import UAVID

# model settings
model = dict(
    text_prompts=UAVID,
    prob_thd=0.3,
    confidence_threshold=0.3,
)

# dataset settings
dataset_type = 'UAVidDataset'
data_root = 'data/UAVid'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/test',
            seg_map_path='ann_dir/test'),
        pipeline=test_pipeline))