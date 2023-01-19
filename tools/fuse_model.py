import torch

img_ckpt = torch.load('work_dirs/res101_900q_24e/epoch_24.pth')
state_dict1 = img_ckpt['state_dict']

pts_ckpt = torch.load('work_dirs/01voxel_q6_step_38e_resume/epoch_2.pth')
state_dict2 = pts_ckpt['state_dict']
# pts_head in camera checkpoint will be overwrite by lidar checkpoint
state_dict1.update(state_dict2)

merged_state_dict = state_dict1


save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, 'pretrained/lidar_5831_r101_3412.pth')
