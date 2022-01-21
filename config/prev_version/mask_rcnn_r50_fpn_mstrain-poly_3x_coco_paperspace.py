#  inherit from base config and change only the bits required
_base_ = [
    "/content/mmdetection/configs/common/mstrain-poly_3x_coco_instance.py",
    "/content/mmdetection/configs/_base_/models/mask_rcnn_r50_fpn.py",
]

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=3), mask_head=dict(num_classes=3))
)
# dataset settings
dataset_type = "CocoDataset"
classes = (
    "shsy5y",
    "astro",
    "cort",
)
data_root = "/content/Sartorius-cell-segmentation-kaggle/input/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True, poly2mask=True),
    dict(
        type="Resize",
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode="range",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# Use RepeatDataset to speed up training
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type="RepeatDataset",
        times=1,  # need to dig in what it does exactly
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train_annotations/train_fold_0_annotations.json",
            img_prefix=data_root,
            classes=classes,
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_annotations/val_fold_0_annotations.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "val_annotations/val_fold_0_annotations.json",
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline,
    ),
)
evaluation = dict(interval=1, metric=["segm"], save_best="segm_mAP")

# optimizer
optimizer = dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
# Experiments show that using step=[9, 11] has higher performance
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[9, 11]
)
runner = dict(type="EpochBasedRunner", max_epochs=5)


# Misc
workflow = [("train", 1), ("val", 1)]
# load checkpoint from
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
# fp16 = dict(loss_scale=512.0) (maybe in future, once the pipeline is setup)
work_dir = "/content/v2_test"

load_from = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_20210524_201154-21b550bb.pth"
