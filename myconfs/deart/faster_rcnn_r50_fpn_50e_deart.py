_base_ = [
    '../odor3/faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/deart_instance.py',
    '../odor3/schedule_50e.py', '../../configs/_base_/default_runtime.py'
]

model = dict(
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
