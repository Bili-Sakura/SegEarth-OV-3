_base_ = './base_config.py'

from segearthov3.vocabularies import XBD

# model settings
model = dict(
    text_prompts=XBD,
    prob_thd=0.1,
    confidence_threshold=0.1,
)

# dataset settings
dataset_type = 'xBDDataset'
data_root = 'data/xBD'

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
            img_path='test/images_pre',
            seg_map_path='test/targets_binary_pre'),
        pipeline=test_pipeline))