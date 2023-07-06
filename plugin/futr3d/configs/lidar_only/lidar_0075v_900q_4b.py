_base_ = [
    '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/schedules/cyclic_20e.py', 
    '../../../configs/_base_/default_runtime.py'
]

plugin = 'plugin/futr3d'

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]

center_head = dict(
    type='CenterHead',
    in_channels=sum([256, 256]),
    tasks=[
        dict(num_class=1, class_names=['car']),
        dict(num_class=2, class_names=['truck', 'construction_vehicle']),
        dict(num_class=2, class_names=['bus', 'trailer']),
        dict(num_class=1, class_names=['barrier']),
        dict(num_class=2, class_names=['motorcycle', 'bicycle']),
        dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    ],
    common_heads=dict(
        reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
    share_conv_channel=64,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range[:2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        code_size=9),
    separate_head=dict(
        type='SeparateHead', init_bias=-2.19, final_kernel=3),
    loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
    loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    norm_bbox=True)

model = dict(
    type='FUTR3d',
    aux_weight=0.5,
    pts_voxel_layer=dict(
        max_num_points=10, 
        voxel_size=voxel_size, 
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    pts_bbox_head=dict(
        type='FUTR3DHead',
        use_dab=True,
        anchor_size=3,
        use_aux=True,
        aux_head=center_head,
        mix_selection=False,
        num_query=900,
        num_classes=10,
        in_channels=256,
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='FUTR3DTransformer',
            use_dab=True,
            decoder=dict(
                type='FUTR3DTransformerDecoder',
                num_layers=6,
                use_dab=True,
                anchor_size=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='FUTR3DAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0)),
    # training and testing settings
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range,
        pc_range=point_cloud_range,
        grid_size=[1440, 1440, 40],
        voxel_size=voxel_size,
        out_size_factor=8,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0)))),
    test_cfg=dict(pts=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=8,
        voxel_size=voxel_size[:2],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
        max_num=300,
        score_threshold=0,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    )))

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadReducedPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadReducedPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args),
    dict(
        type='LoadReducedPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925*2, 0.3925*2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadReducedPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args),
    dict(
        type='LoadReducedPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(
        type='LoadReducedPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args),
    dict(
        type='LoadReducedPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        reduce_beams_to=4,
        chosen_beam_id=[7, 9, 11, 13],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
            # and box_type_3d='Depth' in sunrgbd and scannet dataset.
            box_type_3d='LiDAR')),
    val=dict(pipeline=test_pipeline, classes=class_names, ann_file=data_root + 'nuscenes_infos_val.pkl'),
    test=dict(pipeline=test_pipeline, classes=class_names, ann_file=data_root + 'nuscenes_infos_val.pkl'))
# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
evaluation = dict(interval=2)

find_unused_parameters=True

custom_hooks = [dict(type='FadeOjectSampleHook', num_last_epochs=5)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
