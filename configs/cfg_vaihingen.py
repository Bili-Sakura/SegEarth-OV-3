_base_ = './base_config.py'

from segearthov3.vocabularies import VAIHINGEN

# model settings
model = dict(
    text_prompts=VAIHINGEN,
    prob_thd=0.1,
    bg_idx=5,
    confidence_threshold=0.4,
)

# dataset settings
dataset_type = 'ISPRSDataset'
data_root = 'data/Vaihingen'

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
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))