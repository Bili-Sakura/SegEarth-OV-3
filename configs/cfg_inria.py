_base_ = './base_config.py'

from segearthov3.vocabularies import INRIA

# model settings
model = dict(
    text_prompts=INRIA,
    prob_thd=0.5,
    confidence_threshold=0.5,
)

# dataset settings
dataset_type = 'InriaDataset'
data_root = 'data/Inria/AerialImageDataset'

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
            img_path='train/img_dir/split_test',
            seg_map_path='train/ann_dir/split_test'),
        pipeline=test_pipeline))