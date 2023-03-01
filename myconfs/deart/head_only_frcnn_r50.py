_base_ = [
    '../odor3/faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/deart_instance.py',
    '../odor3/schedule_50e.py', '../../configs/_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=1)

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/net/cluster/zinnen/models/cross-eval/odor_on_deart/epoch_50.pth')
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=70
        )
    )
)

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4
)
