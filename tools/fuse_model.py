import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='fuse two checkpoints')
    parser.add_argument('--img', help='the image model checkpoint path')
    parser.add_argument('--lidar', help='the lidar model checkpoint path')
    parser.add_argument('--out', help='the fused model path')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    img_path = args.img
    pts_path = args.lidar
    out = args.out
    
    img_ckpt = torch.load(img_path)
    state_dict1 = img_ckpt['state_dict']

    pts_ckpt = torch.load(pts_path)
    state_dict2 = pts_ckpt['state_dict']

    # pts_head in camera checkpoint will be overwrite by lidar checkpoint
    state_dict1.update(state_dict2)

    merged_state_dict = state_dict1
    save_checkpoint = {'state_dict':merged_state_dict }

    torch.save(save_checkpoint, out)

if __name__ == '__main__':
    main()