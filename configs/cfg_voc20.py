_base_ = './base_config.py'

from segearthov3.vocabularies import VOC20

# model settings
model = dict(
    text_prompts=VOC20,
    confidence_threshold=0.2,
)

# dataset settings
dataset_type = 'PascalVOC20Dataset'
data_root = './data/VOC2012'

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
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline))