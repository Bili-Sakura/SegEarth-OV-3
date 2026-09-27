_base_ = './base_config.py'

from segearthov3.vocabularies import LOVEDA

# model settings
model = dict(
    text_prompts=LOVEDA,
    confidence_threshold=0.5,
    prob_thd=0.5,
)

# dataset settings
# Category labels: background – 1, building – 2, road – 3, water – 4, barren – 5,forest – 6, agriculture – 7. 
# And the no-data regions were assigned 0 which should be ignored. 
dataset_type = 'LoveDADataset'
data_root = 'data/LoveDA'

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
        reduce_zero_label=True,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'),
        pipeline=test_pipeline))