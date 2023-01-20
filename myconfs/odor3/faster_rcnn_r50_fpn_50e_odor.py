_base_ = [
    './faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/odor3_instance.py',
    './schedule_50e.py', '../_base_/default_runtime.py'
]
