_base_ = [
    './cascade_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/odor3_instance.py',
    './schedule_50e.py', '../../configs/_base_/default_runtime.py'
]
