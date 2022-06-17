# FUTR3D: A Unified Sensor Fusion Framework for 3D Detection
This repo implements the paper [FUTR3D: A Unified Sensor Fusion Framework for 3D Detection](https://arxiv.org/abs/2203.10642) - [project page](https://tsinghua-mars-lab.github.io/futr3d/)

We built our implementation upon MMdetection3D. The major part of the code is in the directory plugin/futr3d. 

## Environment
### Prerequisite
<ol>
<li> mmcv </li>
<li> mmdetection</li>
<li> mmdetection3d==0.17.3</li>
<li> nuscenes-devkit</li>
</ol>

### Data

For cameras with Radar setting, you should generate a meta file or say .pkl file including Radar infos.

```python:
python3 tools/data_converter/nusc_radar.py
```

For others, please follow the mmdet3d to process the data.

## Train

For example, to train FUTR3D with LiDAR only on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py 8
```

## Results
### LiDAR & Cam
| models      | mAP         | NDS |
| ----------- | ----------- | ----|
| [Res101 + 32 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py)  | 62.8 | 68.0 |
| [Res101 + 4 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_4b_step_38e.py)   | 54.9 | 61.5 |
| [Res101 + 1 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_1b_step_38e.py)   | 41.3 | 50.0 |

### Cam & Radar
| models      | mAP         | NDS |
| ----------- | ----------- | ----|
| [Res101 + Radar](./plugin/futr3d/configs/cam_radar/res101_radar.py)  | 35.0  | 45.9 |

### LiDAR only

| models      | mAP         | NDS |
| ----------- | ----------- | ----|
| [32 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py)  | 59.3 | 65.5 |
| [4 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_4b_step_38e.py)   | 42.1 | 54.8 |
| [1 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_1b_step_38e.py)   | 16.4 | 37.9 |

### Cam only
| models      | mAP   | NDS |
| ----------- | ----- | ----|
| [Res101](./plugin/futr3d/configs/cam_only/cam_only.py)  | 34.6  | 42.5|
