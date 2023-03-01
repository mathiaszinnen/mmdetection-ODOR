_base_ = [
    './faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/odor3_instance.py',
    './schedule_50e.py', '../../configs/_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=1)

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained',
            checkpoint='/net/cluster/zinnen/models/cross-eval/deart_on_odor/epoch_10.pth')
    )
)
