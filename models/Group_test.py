import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from utils import misc
from knn_cuda import KNN
import cv2

class Group(nn.Module):

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = misc.fps(xyz, self.num_group) # B G 3
        _, idx = self.knn(xyz, center)
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def new_align(calib_Rtilt, calib_K, scale, points):
    xyz = torch.matmul(calib_Rtilt.transpose(2, 1), (1 / scale ** 2).unsqueeze(-1).unsqueeze(-1) * points.transpose(2, 1))
    xyz = xyz.transpose(2, 1)
    xyz[:, :, [0, 1, 2]] = xyz[:, :, [0, 2, 1]]
    xyz[:, :, 1] *= -1
    uv = torch.matmul(xyz, calib_K.transpose(2, 1)).detach()
    uv[:, :, 0] /= uv[:, :, 2]
    uv[:, :, 1] /= uv[:, :, 2]
    u, v = (uv[:, :, 0] - 1).round(), (uv[:, :, 1] - 1).round()
    # normalize
    u, v = u / MAX_WIDTH, v / MAX_HEIGHT
    return u, v

PATCH_SIZE = 16
def get_center_masks(u, v, img_size):
    B, M = u.shape
    token_nums = int((img_size[0]/PATCH_SIZE)*(img_size[1]/PATCH_SIZE))
    masks = torch.zeros((B, token_nums))
    print('masks:', masks.shape)
    H, W = img_size
    u, v = u*(W-1), v*(H-1)
    u, v = torch.floor(u / PATCH_SIZE), torch.floor(v / PATCH_SIZE)
    proj_patch_idx = (v * (img_size[1] // PATCH_SIZE) + u).long()
    print('proj_patch_idx:', proj_patch_idx.shape, proj_patch_idx)
    for i in range(B):
        masks[i, proj_patch_idx[i]] = 1

    return masks

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scan_path = config.scan_path
    point_cloud = np.load(scan_path + "_pc.npz")["pc"]

    calib_lines = [line for line in open(os.path.join(scan_path + '.txt')).readlines()]
    calib_Rtilt = np.reshape(np.array([float(x) for x in calib_lines[0].rstrip().split(' ')]), (3, 3), 'F')
    calib_K = np.reshape(np.array([float(x) for x in calib_lines[1].rstrip().split(' ')]), (3, 3), 'F')
    scale_ratio = 1.


    def load_image(img_filename):
        return cv2.imread(img_filename)


    full_img = load_image(os.path.join(scan_path + '.jpg'))
    MAX_HEIGHT = full_img.shape[0]  # full_img.shape[0]
    MAX_WIDTH = full_img.shape[1]  # full_img.shape[1]

    fx, fy = 256 / full_img.shape[1], 352 / full_img.shape[0]
    full_img = cv2.resize(full_img, None, fx=fx, fy=fy)
    full_img_height, full_img_width = full_img.shape[0], full_img.shape[1]
    print('full_img_height, full_img_width',full_img_height, full_img_width)

    # if not self.use_color:
    point_cloud = point_cloud[:, 0:3]
    points = torch.from_numpy(point_cloud.reshape(1, -1, 3)).to(device)
    points =points.contiguous()
    npoints = 2048
    pts = misc.fps(points, npoints)

    group_divider= Group(num_group=128, group_size=32)  # neighborhood [B,G,M,3] CENTER [B,G,3]
    neighborhood, center = group_divider(pts)
    print(neighborhood.shape, center.shape)

    Rtilt = torch.from_numpy(calib_Rtilt.reshape(1, -1, 3)).to(device)
    K = torch.from_numpy(calib_K.reshape(1, -1, 3)).to(device)
    scale = np.array(scale_ratio)
    # scale = torch.from_numpy(scale.reshape(1, -1))
    scale = torch.from_numpy(scale).to(device)
    u, v = new_align(Rtilt, K, scale, center)
    img_size = torch.tensor((256, 352))
    masks = get_center_masks(u, v, img_size)
    print(masks)