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

'''

python3 tools/data_converter/nusc_track.py

'''

For others, please follow the mmdet3d to process the data.

## Train

For example, to train FUTR3D with LiDAR only on 8 GPUs, please use

'''
bash tools/dist_train.sh plugin/futr3d/configs/lidar_only/01voxel_q6_step_38e.py 8
'''

## Results
