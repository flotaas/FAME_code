import numpy as np
import torch.cuda
from knn_cuda import KNN
from utils import misc
import torch.nn as nn

# Generate 10 random 3D points
num_points = 4
points = [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]],
          [[0., 0., 0.], [-1., 0., 0.], [0., -1., 0.], [-1., -1., 0.]]]
# points = [[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]]

# Convert the list of points into a PyTorch tensor
points = torch.tensor(points).to('cuda:0') # Each row is a 3D point (x, y, z)

class Group(nn.Module):  # FPS + KNN
    '''
    init:
    input: num_group: Represents the number of groups to be created using FPS.
            group_size: Number of points in each group obtained using KNN.
    forward:
    input: xyz, a batch of point clouds with shape (B, N, 3), where: B is the batch size. N is the number of points in each point cloud.
            3 represents the x, y, z coordinates of each point.
    The centers for the groups are determined using the misc.fps method. The output center has shape (B, G, 3).
    Using the KNN method, for each of these centers, its group_size closest neighbors are determined.
    The resulting indices idx have shape (B, G, M), where M is the group size.

    The reason for computing this idx_base is that the KNN function returns indices that are local to each batch item. When we want to gather
    the corresponding points from the flattened xyz tensor, we need global indices, and this base index helps to compute those global indices.
    idx_base serves as an offset for the indices in idx so that they correctly point to the locations in the flattened tensor.

    Here, xyz is reshaped into a two-dimensional tensor where all points from all batches are laid out in a single dimension of size
    (batch_size * num_points). Using the flattened idx, we can directly index into this tensor to fetch the relevant neighborhood points.
    Each row in this gathered tensor neighborhood represents a point in some neighborhood.
    and then reshaped to get the neighborhood tensor with shape (B, G, M, 3). The .contiguous() call ensures that the resultant tensor is stored
    in a contiguous block of memory. This can be important for certain PyTorch operations that expect tensors to be contiguous.

    Each point in the neighborhood is normalized by subtracting its corresponding center.
    '''

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3 Input: xyz: pointcloud data, [B, N, 3]
            ---------------------------
            output: neighborhood: B G M 3
                    center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # print(batch_size, num_points)
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz,
                          center)  # B G M idx, a tensor that holds the indices of the k-nearest neighbors for each center.
        # print(idx,idx.shape)
        assert idx.size(
            1) == self.num_group  # sanity checks to ensure that the output from the KNN function has the expected dimensions.
        assert idx.size(2) == self.group_size
        # creates a tensor [0, 1, 2, ... batch_size-1] on the same device as xyz. reshapes this tensor to shape (B, 1, 1).
        # The multiplication by num_points scales each entry, so the tensor becomes [[[0]], [[1*num_points]],... [[(batch_size-1)*num_points]]].
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # print(torch.arange(0, batch_size, device=xyz.device))
        # print(torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1))
        # print(idx_base)
        idx = idx + idx_base
        # print(idx)
        idx = idx.view(
            -1)  # To use these indices for a gather operation on a flattened xyz, we reshape idx into a 1D tensor
        # print(idx)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        # print(neighborhood.view(batch_size, self.num_group, self.group_size, 3))
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # print(neighborhood)
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


if __name__ == '__main__':
    # print(points, points.shape)  # Display the generated points
    group = Group(num_group=2, group_size=2).to('cuda:0')
    # print(group)
    neighborhood, center = group(points)
    print(neighborhood, neighborhood.shape)
    print(center, center.shape)
