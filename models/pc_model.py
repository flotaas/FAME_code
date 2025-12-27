import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from models.utils import misc
from models.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.utils.logger import *
import random
from knn_cuda import KNN
from models.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

from models.transformer import TransformerEncoderLayer as out_encoderlayer
from models.transformer import TransformerEncoder as out_encoder


class Encoder(nn.Module):  ## Embedding module
    '''
    init:
    input: encoder_channel, represents the number of output channels the encoder will produce.
    first_conv: The first convolution transforms 3 channels to 128 channels, kernel_size=1,
                and the second convolution transforms 128 channels to 256 channels.
    second_conv: The first convolution here transforms 512 channels to 512 channels,
                and the second convolution transforms from 512 channels to encoder_channel channels.

    forward:
    Input: point_groups is a tensor of shape (B, G, N, 3), where:
            B is the batch size.
            G is the number of point groups within each item in the batch.
            N is the number of points in each point group.
            The last dimension 3 represents the x, y, z coordinates of each point.
    The tensor point_groups is reshaped to (B*G, N, 3) so that the point groups from different batch items can be processed together.
    The tensor then goes through first_conv, producing a feature tensor of shape (B*G, 256, N).
    A global feature is computed using max-pooling across the N dimension, resulting in a tensor of shape (B*G, 256, 1).
    This global feature captures the most prominent feature across all points in each point group.
    This tensor goes through second_conv, producing a feature tensor of shape (B*G, encoder_channel, N).
    Another global feature is computed using max-pooling, producing a tensor of shape (B*G, encoder_channel).
    Finally, the tensor is reshaped back to (B, G, encoder_channel) and returned.

    output: produce a high-level feature representation for each group of points

    This encoder is designed for point cloud processing, where each item in the batch contains multiple groups of points,
    and the aim is to produce a high-level feature representation for each group of points.
    The use of 1D convolutions and the reshaping operations suggest that the network is designed to work with unordered sets of points,
    which is a common characteristic of point cloud data.

    Local Information: The original feature, as obtained from the first_conv layer,
    captures local patterns and nuances from the individual points within each point group.
    These local patterns can be essential for tasks that require detailed information about the local structure or geometry of the point cloud.

    Global Information: The global feature, obtained by applying max-pooling across all points in each group,
    captures the most salient or prominent feature across the entire point group.
    This provides a sort of summary or holistic view of the entire point group,
    encapsulating what might be considered the most "important" characteristic or feature of the set of points.
    Such global information can be crucial for tasks that benefit from understanding the overall context or nature of the point cloud group.
    '''

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


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
            output: neighborhood: B G M 3  # G: groups  #M: members
                    center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M idx, a tensor that holds the indices of the k-nearest neighbors for each center.
        assert idx.size(1) == self.num_group  # sanity checks to ensure that the output from the KNN function has the expected dimensions.
        assert idx.size(2) == self.group_size
        # creates a tensor [0, 1, 2, ... batch_size-1] on the same device as xyz. reshapes this tensor to shape (B, 1, 1).
        # The multiplication by num_points scales each entry, so the tensor becomes [[[0]], [[1*num_points]],... [[(batch_size-1)*num_points]]].
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(
            -1)  # To use these indices for a gather operation on a flattened xyz, we reshape idx into a 1D tensor
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    init:
    dim: Dimensionality of input features.
    num_heads: Number of attention heads in the multi-head attention mechanism. Multi-head attention allows
                the model to jointly attend to information from different representation subspaces.
    qkv_bias: Whether to use bias terms for the Q, K, V linear projections.
    qk_scale: Optional scaling factor for the attention scores.
    attn_drop and proj_drop: Dropout rates for the attention weights and the output, respectively.
    self.qkv: A linear layer that is used to obtain the Q (Query), K (Key), and V (Value) representations from the input x.
    self.attn_drop: Dropout layer for the attention mechanism.
    self.proj and self.proj_drop: Linear transformation and its associated dropout for the output of the attention mechanism.

    forward:
    x is the input tensor with shape (B, N, C), where:
    B is the batch size.
    N is the sequence length (or number of tokens).
    C is the feature dimensions.
    Q, K, V Projections: The input x is passed through the self.qkv linear layer. The result is reshaped, permuted to get the Q, K, V.
    Attention Score Calculation: The attention scores are computed as the dot product of the Q and K tensors then scaled by self.scale
    Compute Output: The attention weights are used to take a weighted average of the V tensor, resulting in the output of
    the attention mechanism. This output is then passed through the self.proj linear layer and its associated dropout.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)  # (B, N, C)
        return x


class Block(nn.Module):
    '''
    mlp_ratio: Determines the hidden dimension of the MLP as a ratio of dim.
    drop_path: Rate for stochastic depth, a regularization method to improve generalization by randomly dropping out entire layers during training.
    act_layer: Activation function for the MLP, defaulting to GELU.
    norm_layer: Normalization layer, defaulting to Layer Normalization.
    self.norm1 and self.norm2: Normalization layers preceding the attention and MLP layers, respectively.
    self.mlp: An MLP module with input dimension dim and hidden dimension determined by mlp_ratio.

    The output of the block is the input passed through the attention and MLP layers and then combined
    using the residual connections. It retains the same shape as the input.
    '''

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    '''
    embed_dim: Dimensionality of the input features.
    depth: The number of transformer blocks in the encoder.

    self.blocks: A list (wrapped by nn.ModuleList) of Block modules. The number of blocks is determined by
    the depth parameter. Each block will process the input sequentially. If drop_path_rate is a list, the rate
    for each block may vary, otherwise, a constant rate is used for all blocks.
    '''

    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    '''
    self.head: An identity layer. It act as a placeholder and can potentially be replaced with other modules in the future for specific tasks.
    self.apply(self._init_weights): This applies the _init_weights method to each module in Decoder, initializing weights and biases.

    Weight Initialization (_init_weights method):
        This method initializes the weights and biases for linear layers and layer normalization.
        For nn.Linear: The weights are initialized using Xavier uniform initialization and biases are initialized to zero.
        For nn.LayerNorm: Biases are set to zero and weights to 1.0.

    After processing through all blocks, the feature corresponding to the last return_token_num tokens is normalized using self.norm.
    Finally, the normalized feature is passed through self.head (which is currently an identity mapping).
    This is done to retrieve features corresponding to certain specific tokens, the tokens that represent masked pixels.
    '''

    def __init__(self, embed_dim=192, depth=2, num_heads=3, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        #x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        x = self.head(self.norm(x))
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    '''
    designed to process data that has a spatial nature, point clouds.
    '''
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger='Transformer')
        # embedding
        self.encoder_dims = config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )
        '''encoder of pointmae own'''
        # linspace在前两个参数之间生成一个间距均匀的一维张量,该张量的数值个数（即长度）由第三个参数 self.depth 指定。
        # [x.item() for x in ...]:This is a list. It iterates over each element in the tensor generated
        # by torch.linspace. convert the tensor values into a list of numbers.
        # This dpr list is used to specify drop path rates for different layers/blocks of the model,
        # likely for regularization purposes, where each layer or block can have a different drop path rate.
        #dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        #self.blocks = TransformerEncoder(
        #    embed_dim=self.trans_dim,
        #    depth=self.depth,
        #    drop_path_rate=dpr,
        #    num_heads=self.num_heads,
        #)
        '''encoder from 3detr'''
        self.dim_feedforward = config.transformer_config.dim_feedforward
        self.dropout = config.transformer_config.dropout
        encoder_layer = out_encoderlayer(
             d_model=self.trans_dim,
             nhead=self.num_heads,
             dim_feedforward=self.dim_feedforward,
             dropout=self.dropout,
        )
        self.blocks = out_encoder(encoder_layer=encoder_layer, num_layers=self.depth)

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)

        The goal of the method is to produce a boolean mask of shape (B, G) indicating which points or groups should be masked.
        calculates the distance from the randomly chosen point to all other points.
        The indices of the points are sorted based on their distance to the chosen point. Points closer to the randomly chosen point will have smaller indices.
        A mask_ratio of the closest points to the chosen point will be masked. The number of points to be masked is determined by mask_num.
        A mask of size G (number of groups) is then created, where the first mask_num closest points are set to 1 (indicating they are masked), and the rest are set to 0.
        the created mask is appended to the mask_idx list.
        All individual masks for each set of points in the batch are stacked together to form a boolean tensor of shape (B, G),
        which indicates which points in each set of the batch should be masked.

        mask a continuous segment of data based on the Euclidean distance of a randomly picked point from all others
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()  # return a mask of zeros, no point is masked.
        # mask a continuous part
        mask_idx = []
        for points in center:  # For each set of points in the batch:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        The method aims to create a boolean mask of size (B, G) where the goal is to indicate which points should be masked.
        self.num_mask gives the total number of points that need to be masked for each batch.
        For each batch (for i in range(B)):
            a. A mask is created which consists of G-self.num_mask unmasked points and self.num_mask masked points.
            b. This mask is then shuffled randomly using np.random.shuffle().
            c. The shuffled mask is assigned to the i-th batch in overall_mask.
        After looping through all batches, the overall_mask is converted from a numpy array to a PyTorch tensor.
        _mask_center_rand method randomly selects a mask_ratio of points from each set in the batch to be masked.
        The selection is done without any preference for the position or attributes of the points; it's purely random.
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)  # B G

    def forward(self, neighborhood, center, noaug=False):
        '''
        extracts features from the unmasked points, adds positional embeddings to those features, and then processes them through the
        transformer blocks to produce the final output. The mask used and the transformed features are returned as outputs.

        1. Generating the Mask:
        Depending on the mask_type attribute, the _mask_center_rand method (which creates a random mask) or the _mask_center_block method
        (which masks a continuous segment of points).
        The resulting mask, bool_masked_pos (B, G). A True value means the corresponding point is masked, and False means it's not masked.
        2. Processing the Neighborhood:
        The neighborhood input gets passed through the encoder, resulting in group_input_tokens, feature embeddings for each point.
        3. Masking the Input Tokens:
        Using the bool_masked_pos mask, points that are masked (set to True) are removed from group_input_tokens resulting in x_vis.
        the center points that are masked are selected and stored in masked_center.
        4. Positional Embeddings:
        The masked_center points are passed through the pos_embed module to get their positional embeddings, denoted as pos.
        5. Passing through Transformer Blocks:
        The masked input tokens x_vis and their positional embeddings pos are passed through the transformer blocks (self.blocks).
        This can be thought of as the main processing step where the tokens are transformed using attention mechanisms and feed-forward networks.
        After the transformer blocks, a normalization step is applied to the results using self.norm.
        6. Return:
        The function returns the transformed tokens x_vis and the mask bool_masked_pos.
        '''
        if self.mask_type == 'rand':
            # 创建随机屏蔽还是连续屏蔽
            bool_masked_pos = self._mask_center_rand(center,
                                                     noaug=noaug)  # B G, G indicates the number of groups or points
        else:
            bool_masked_pos = self._mask_center_block(center, noaug=noaug)

        group_input_tokens = self.encoder(neighborhood)  # B G C #[4, 32, 384] [B,G,M,3]-->[B,G,C]
        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)  # vis means visitable point groups
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)  # center[B,G,3] masked_center[B,G,3]
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos


class Point_MAE(nn.Module):
    '''
    mask_token is used during the decoding phase to represent the masked-out tokens.
    '''

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim

        self.decoder_trans_dim = config.transformer_config.decoder_trans_dim
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads

        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_trans_dim))

        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.decoder_trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.decoder_trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)  # neighborhood [B,G,M,3] CENTER [B,G,3]
        # change encoder output's dim shape to be the same as decoder dim
        self.decoder_embed = nn.Linear(self.trans_dim, self.decoder_trans_dim, bias=True)
        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.decoder_trans_dim, 3 * self.group_size, 1)
        )  # upsamples the decoded features to reconstruct the 3D points.

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError

    def forward(self, pts, vis=False, **kwargs):
        '''
        self.group_divider: The point cloud is divided into groups and centers are calculated.
        Position Embedding: It calculates position embeddings for both visible and masked center points separately.
        '''
        neighborhood, center = self.group_divider(pts)
        x_vis, mask = self.MAE_encoder(neighborhood, center)
        B, _, C = x_vis.shape  # B VIS C

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)  # N为屏蔽的点数
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        x_rec = self.MAE_decoder(x_full, pos_full, N)

        B, M, C = x_rec.shape
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

        gt_points = neighborhood[mask].reshape(B * M, -1, 3)
        loss1 = self.loss_func(rebuild_points, gt_points)

        # print(gt_points.cpu().numpy()[0])
        # print(pts.shape)
        # print(center.shape)
        # print(neighborhood.shape)
        # print(x_vis.shape)
        # print(x_rec.shape)
        # print(rebuild_points.shape)
        # print(gt_points.shape)
        # print(mask.shape)
        # print(neighborhood[mask].shape)
        # print(loss1)
        #
        # sys.exit()

        if vis:  # visualization
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
            mask_points = neighborhood[mask].reshape(B * M, -1, 3)  # added, masked points
            full_mask = mask_points + center[mask].unsqueeze(1)  # added, masked points
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret3 = full_mask.reshape(-1, 3).unsqueeze(0)  # added, full masked points
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)
            # return ret1, ret2
            # print(full_rebuild.shape)
            # print(full.shape)
            # print(ret1.shape)
            # print(ret2.shape)
            # sys.exit()
            return ret1, ret2, ret3, full_center
        else:
            return loss1


# finetune model
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim)
        )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret