model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=139,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CocoDataset'
data_root = 'data/ODOR-v3/'
classes = [
            'ant', 'camel', 'jewellery', 'frog', 'physalis', 'celery',
            'cauliflower', 'pepper', 'ranunculus', 'chess flower', 'cigarette',
            'matthiola', 'cabbage', 'earring', 'dandelion', 'neroli',
            'dragonfly', 'hyacinth', 'reptile/amphibia', 'apricot', 'snake',
            'lizard', 'asparagus', 'spring onion', 'snowflake', 'moth',
            'poppy', 'columbine', 'rabbit', 'geranium', 'crab', 'radish',
            'big cat', 'jan steen jug', 'monkey', 'snail', 'bellflower',
            'lilac', 'pot', 'peony', 'coffeepot', 'hazelnut', 'censer',
            'artichoke', 'dahlia', 'sniffing', 'fly', 'deer', 'caterpillar',
            'garlic', 'blackberry', 'chalice', 'lobster', 'necklace', 'bug',
            'insect', 'prawn', 'bracelet', 'carrot', 'cornflower', 'pumpkin',
            'orange', 'walnut', 'cat', 'daisy', 'forget-me-not', 'carafe',
            'match', 'beer stein', 'tobacco-box', 'violet', 'pomander',
            'bottle', 'candle', 'heliotrope', 'wine bottle', 'strawberry',
            'pomegranate', 'whale', 'lily of the valley', 'iris', 'tobacco',
            'olive', 'tobacco-packaging', 'meat', 'daffodil', 'melon', 'fire',
            'petunia', 'mushroom', 'teapot', 'ring', 'pig', 'ashtray',
            'cheese', 'onion', 'cup', 'nut', 'fig', 'drinking vessel',
            'donkey', 'holding the nose', 'lily', 'smoke', 'bread', 'currant',
            'glass without stem', 'anemone', 'mammal', 'chimney',
            'smoking equipment', 'bivalve', 'butterfly', 'gloves', 'lemon',
            'horse', 'plum', 'jasmine', 'pear', 'glass with stem', 'vegetable',
            'carnation', 'jug', 'goat', 'fish', 'apple', 'tulip', 'cherry',
            'cow', 'animal corpse', 'dog', 'fruit', 'bird', 'rose', 'peach',
            'sheep', 'pipe', 'grapes', 'flower'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        classes=(
            'ant', 'camel', 'jewellery', 'frog', 'physalis', 'celery',
            'cauliflower', 'pepper', 'ranunculus', 'chess flower', 'cigarette',
            'matthiola', 'cabbage', 'earring', 'dandelion', 'neroli',
            'dragonfly', 'hyacinth', 'reptile/amphibia', 'apricot', 'snake',
            'lizard', 'asparagus', 'spring onion', 'snowflake', 'moth',
            'poppy', 'columbine', 'rabbit', 'geranium', 'crab', 'radish',
            'big cat', 'jan steen jug', 'monkey', 'snail', 'bellflower',
            'lilac', 'pot', 'peony', 'coffeepot', 'hazelnut', 'censer',
            'artichoke', 'dahlia', 'sniffing', 'fly', 'deer', 'caterpillar',
            'garlic', 'blackberry', 'chalice', 'lobster', 'necklace', 'bug',
            'insect', 'prawn', 'bracelet', 'carrot', 'cornflower', 'pumpkin',
            'orange', 'walnut', 'cat', 'daisy', 'forget-me-not', 'carafe',
            'match', 'beer stein', 'tobacco-box', 'violet', 'pomander',
            'bottle', 'candle', 'heliotrope', 'wine bottle', 'strawberry',
            'pomegranate', 'whale', 'lily of the valley', 'iris', 'tobacco',
            'olive', 'tobacco-packaging', 'meat', 'daffodil', 'melon', 'fire',
            'petunia', 'mushroom', 'teapot', 'ring', 'pig', 'ashtray',
            'cheese', 'onion', 'cup', 'nut', 'fig', 'drinking vessel',
            'donkey', 'holding the nose', 'lily', 'smoke', 'bread', 'currant',
            'glass without stem', 'anemone', 'mammal', 'chimney',
            'smoking equipment', 'bivalve', 'butterfly', 'gloves', 'lemon',
            'horse', 'plum', 'jasmine', 'pear', 'glass with stem', 'vegetable',
            'carnation', 'jug', 'goat', 'fish', 'apple', 'tulip', 'cherry',
            'cow', 'animal corpse', 'dog', 'fruit', 'bird', 'rose', 'peach',
            'sheep', 'pipe', 'grapes', 'flower'),
        ann_file='data/ODOR-v3/coco-style/annotations/instances_train2017.json',
        img_prefix='data/ODOR-v3/coco-style/train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=(
            'ant', 'camel', 'jewellery', 'frog', 'physalis', 'celery',
            'cauliflower', 'pepper', 'ranunculus', 'chess flower', 'cigarette',
            'matthiola', 'cabbage', 'earring', 'dandelion', 'neroli',
            'dragonfly', 'hyacinth', 'reptile/amphibia', 'apricot', 'snake',
            'lizard', 'asparagus', 'spring onion', 'snowflake', 'moth',
            'poppy', 'columbine', 'rabbit', 'geranium', 'crab', 'radish',
            'big cat', 'jan steen jug', 'monkey', 'snail', 'bellflower',
            'lilac', 'pot', 'peony', 'coffeepot', 'hazelnut', 'censer',
            'artichoke', 'dahlia', 'sniffing', 'fly', 'deer', 'caterpillar',
            'garlic', 'blackberry', 'chalice', 'lobster', 'necklace', 'bug',
            'insect', 'prawn', 'bracelet', 'carrot', 'cornflower', 'pumpkin',
            'orange', 'walnut', 'cat', 'daisy', 'forget-me-not', 'carafe',
            'match', 'beer stein', 'tobacco-box', 'violet', 'pomander',
            'bottle', 'candle', 'heliotrope', 'wine bottle', 'strawberry',
            'pomegranate', 'whale', 'lily of the valley', 'iris', 'tobacco',
            'olive', 'tobacco-packaging', 'meat', 'daffodil', 'melon', 'fire',
            'petunia', 'mushroom', 'teapot', 'ring', 'pig', 'ashtray',
            'cheese', 'onion', 'cup', 'nut', 'fig', 'drinking vessel',
            'donkey', 'holding the nose', 'lily', 'smoke', 'bread', 'currant',
            'glass without stem', 'anemone', 'mammal', 'chimney',
            'smoking equipment', 'bivalve', 'butterfly', 'gloves', 'lemon',
            'horse', 'plum', 'jasmine', 'pear', 'glass with stem', 'vegetable',
            'carnation', 'jug', 'goat', 'fish', 'apple', 'tulip', 'cherry',
            'cow', 'animal corpse', 'dog', 'fruit', 'bird', 'rose', 'peach',
            'sheep', 'pipe', 'grapes', 'flower'),
        ann_file='data/ODOR-v3/coco-style/annotations/instances_train2017.json',
        img_prefix='data/ODOR-v3/coco-style/train2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=(
            'ant', 'camel', 'jewellery', 'frog', 'physalis', 'celery',
            'cauliflower', 'pepper', 'ranunculus', 'chess flower', 'cigarette',
            'matthiola', 'cabbage', 'earring', 'dandelion', 'neroli',
            'dragonfly', 'hyacinth', 'reptile/amphibia', 'apricot', 'snake',
            'lizard', 'asparagus', 'spring onion', 'snowflake', 'moth',
            'poppy', 'columbine', 'rabbit', 'geranium', 'crab', 'radish',
            'big cat', 'jan steen jug', 'monkey', 'snail', 'bellflower',
            'lilac', 'pot', 'peony', 'coffeepot', 'hazelnut', 'censer',
            'artichoke', 'dahlia', 'sniffing', 'fly', 'deer', 'caterpillar',
            'garlic', 'blackberry', 'chalice', 'lobster', 'necklace', 'bug',
            'insect', 'prawn', 'bracelet', 'carrot', 'cornflower', 'pumpkin',
            'orange', 'walnut', 'cat', 'daisy', 'forget-me-not', 'carafe',
            'match', 'beer stein', 'tobacco-box', 'violet', 'pomander',
            'bottle', 'candle', 'heliotrope', 'wine bottle', 'strawberry',
            'pomegranate', 'whale', 'lily of the valley', 'iris', 'tobacco',
            'olive', 'tobacco-packaging', 'meat', 'daffodil', 'melon', 'fire',
            'petunia', 'mushroom', 'teapot', 'ring', 'pig', 'ashtray',
            'cheese', 'onion', 'cup', 'nut', 'fig', 'drinking vessel',
            'donkey', 'holding the nose', 'lily', 'smoke', 'bread', 'currant',
            'glass without stem', 'anemone', 'mammal', 'chimney',
            'smoking equipment', 'bivalve', 'butterfly', 'gloves', 'lemon',
            'horse', 'plum', 'jasmine', 'pear', 'glass with stem', 'vegetable',
            'carnation', 'jug', 'goat', 'fish', 'apple', 'tulip', 'cherry',
            'cow', 'animal corpse', 'dog', 'fruit', 'bird', 'rose', 'peach',
            'sheep', 'pipe', 'grapes', 'flower'),
        ann_file='data/ODOR-v3/coco-style/annotations/instances_val2017.json',
        img_prefix='data/ODOR-v3/coco-style/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox'])
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/hdd/mmdetection-workdirs/frcnn_deart/epoch_50.pth'
resume_from = None
workflow = [('train', 50)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
epochs = 50
work_dir = '/hdd/mmdetection-workdirs/deart_to_odor_finetune'
auto_resume = False
gpu_ids = [1]
