# FUTR3D: A Unified Sensor Fusion Framework for 3D Detection
This repo implements the paper FUTR3D: A Unified Sensor Fusion Framework for 3D Detection. [Paper](https://arxiv.org/abs/2203.10642) - [project page](https://tsinghua-mars-lab.github.io/futr3d/)

We built our implementation upon MMdetection3D. The major part of the code is in the directory `plugin/futr3d`. 

## Environment
### Prerequisite
<ol>
<li> mmcv-full>=1.3.8, <=1.4.0 </li>
<li> mmdet>=2.14.0, <=3.0.0</li>
<li> mmseg>=0.14.1, <=1.0.0</li>
<li> nuscenes-devkit</li>
</ol>

### Data

For cameras with Radar setting, you should generate a meta file or say .pkl file including Radar infos.

```python:
python3 tools/data_converter/nuscenes_converter_radar.py
```

For others, please follow the mmdet3d to process the data. https://mmdetection3d.readthedocs.io/en/stable/datasets/nuscenes_det.html

## Train

For example, to train FUTR3D with LiDAR only on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py 8
```

For LiDAR-Cam and Cam-Radar version, we need pre-trained model. 
The Cam-Radar uses DETR3D model as pre-trained model, please check [DETR3D](https://github.com/WangYueFt/detr3d).
The LiDAR-Cam uses fused LiDAR-only and Cam-only model as pre-trained model. You can use

```
python tools/fuse_model.py --img <cam checkpoint path> --lidar <lidar checkpoint path> --out <out model path>
```
to fuse cam-only and lidar-only models.

## Evaluate

For example, to evalaute FUTR3D with LiDAR-cam on 8 GPUs, please use

```
bash tools/dist_train.sh plugin/futr3d/configs/lidar_cam/res101_01voxel_step_3e.py ../lidar_cam.pth 8 --eval bbox
```


## Results

### LiDAR & Cam
| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----| ---- |
| [Res101 + 32 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py)  | 64.2 | 68.0 | [model](https://drive.google.com/file/d/1SJbIHaOZFPNXDbtBn1yL1UZRMStL5N5P/view?usp=share_link)|
| [Res101 + 4 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_4b_step_38e.py)   | 54.9 | 61.5 |
| [Res101 + 1 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_1b_step_38e.py)   | 41.3 | 50.0 |

### Cam & Radar
| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----| ----- |
| [Res101 + Radar](./plugin/futr3d/configs/cam_radar/res101_radar.py)  | 35.0  | 45.9 | [model](https://drive.google.com/file/d/1TRNeHrN5mOLWrUGEE0NJ3NxdtcAR5p6Q/view?usp=share_link) |

### LiDAR only

| models      | mAP         | NDS | Link |
| ----------- | ----------- | ----|  ----|
| [32 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py)  | 59.3 | 65.5 | [model](https://drive.google.com/file/d/1HTe-Ys0Ybijw7ArFm89hnjVT0_kjy_TL/view?usp=sharing)|
| [4 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_4b_step_38e.py)   | 42.1 | 54.8 |
| [1 beam VoxelNet](./plugin/futr3d/configs/lidar_only/01voxel_q6_1b_step_38e.py)   | 16.4 | 37.9 |

### Cam only
The camera-only version of FUTR3D is the same as DETR3D. Please check [DETR3D](https://github.com/WangYueFt/detr3d) for detail implementation.

## Acknowledgment

For the implementation, we rely heavily on [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), and [DETR3D](https://github.com/WangYueFt/detr3d)


## Related projects 
1. [DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](https://tsinghua-mars-lab.github.io/detr3d/)
2. [MUTR3D: A Multi-camera Tracking Framework via 3D-to-2D Queries](https://tsinghua-mars-lab.github.io/mutr3d/)
3. For more projects on Autonomous Driving, check out our Visual-Centric Autonomous Driving (VCAD) project page [webpage](https://tsinghua-mars-lab.github.io/vcad/) 


## Reference

```
@article{chen2022futr3d,
  title={FUTR3D: A Unified Sensor Fusion Framework for 3D Detection},
  author={Chen, Xuanyao and Zhang, Tianyuan and Wang, Yue and Wang, Yilun and Zhao, Hang},
  journal={arXiv preprint arXiv:2203.10642},
  year={2022}
}
```

Contact: Xuanyao Chen at: `xuanyaochen19@fudan.edu.cn` or `ixyaochen@gmail.com`