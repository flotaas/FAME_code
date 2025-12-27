from timm.models.vision_transformer import Block as TBlock
import torch.nn as nn
import torch
import random
import models.utils.matcher
import numpy as np
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Optional, Union
from .multimae_utils import Block, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models.transformer import TransformerEncoderLayer as out_encoderlayer
from models.transformer import TransformerEncoder as out_encoder
from easydict import EasyDict as edict

class FrequencyLoss(nn.Module):
    """Frequency loss.

    Modified from:
    `<https://github.com/EndlessSora/focal-frequency-loss/blob/master/focal_frequency_loss/focal_frequency_loss.py>`_.

    Args:
        loss_gamma (float): the exponent to control the sharpness of the frequency distance. Defaults to 1.
        matrix_gamma (float): the scaling factor of the spectrum weight matrix for flexibility. Defaults to 1.
        patch_factor (int): the factor to crop image patches for patch-based frequency loss. Defaults to 1.
        ave_spectrum (bool): whether to use minibatch average spectrum. Defaults to False.
        with_matrix (bool): whether to use the spectrum weight matrix. Defaults to False.
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Defaults to False.
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Defaults to False.
    """

    def __init__(self,
                 loss_gamma=1.,
                 matrix_gamma=1.,
                 patch_factor=1,
                 ave_spectrum=False,
                 with_matrix=False,
                 log_matrix=False,
                 batch_matrix=False):
        super(FrequencyLoss, self).__init__()
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.with_matrix = with_matrix
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x):
        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, (
            'Patch factor should be divisible by image height and width')
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor
        for i in range(patch_factor):
            for j in range(patch_factor):
                patch_list.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        # stack to patch tensor
        y = torch.stack(patch_list, 1).float()  # NxPxCxHxW
        # perform 2D FFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm='ortho')
        # shift low frequency to the center
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # stack the real and imaginary parts along the last dimension
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq, matrix=None):
        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12) ** self.loss_gamma
        if self.with_matrix:
            # spectrum weight matrix
            if matrix is not None:
                # if the matrix is predefined
                weight_matrix = matrix.detach()
            else:
                # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
                matrix_tmp = (recon_freq - real_freq) ** 2
                matrix_tmp = torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.matrix_gamma

                # whether to adjust the spectrum weight matrix by logarithm
                if self.log_matrix:
                    matrix_tmp = torch.log(matrix_tmp + 1.0)

                # whether to calculate the spectrum weight matrix using batch-based statistics
                if self.batch_matrix:
                    matrix_tmp = matrix_tmp / matrix_tmp.max()
                else:
                    matrix_tmp = matrix_tmp / matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]

                matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
                matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
                weight_matrix = matrix_tmp.clone().detach()

            assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, (
                'The values of spectrum weight matrix should be in the range [0, 1], '
                'but got Min: %.10f Max: %.10f' % (weight_matrix.min().item(), weight_matrix.max().item()))
            # dynamic spectrum weighting (Hadamard product)
            loss = weight_matrix * loss
        return loss

    def forward(self, pred, target, matrix=None, **kwargs):
        """Forward function to calculate frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
            target (torch.Tensor): Target tensor with shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Defaults to None.
        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate frequency loss
        return self.loss_formulation(pred_freq, target_freq, matrix)

class PiMAE(nn.Module):
    '''
    The constructor accepts three arguments: pc_branch, img_branch, and config, which configure the model for different
    modalities (point cloud and image) and various architectural settings.

    self.hp_overlap_ratio: an attribute that controls the overlap ratio between different modalities or parts of the
    model; it's set from the configuration if available.
    dim_tokens, depth, num_heads, mlp_ratio: These variables are pulled from the configuration and specify the
    dimensions and architecture of the encoder.
    norm_layer: A partial function of nn.LayerNorm is created with a specified epsilon value, which will be used
    as the normalization layer throughout the model.
    self.token_fusion: A configuration flag indicating whether the model should perform some form of token-level
    fusion between the point cloud and image modalities.
    self.pc2img: Another flag from the configuration that dictates whether the model includes point cloud to image
    transformation.
    self.distill_loss: An instance of mean squared error loss, which might be used for distillation or some kind of
    loss calculation between the modalities.

    self.pc_branch and self.img_branch are set up as branches to handle point cloud and image modalities, respectively.
    self.modality_img_embedding and self.modality_pc_embedding, are initialized as parameters with zero tensors, which
    will be learned during training to represent each modality.
    self.cls_token: A learnable class token, usually aggregates information from the entire sequence for classification tasks.
    If token fusion is enabled, self.fusion_proj is a linear layer that projects concatenated tokens down to dim_tokens dimension.
    If the pc2img is set, the model includes a sequential layer self.increase_dim_feat to transform point cloud features
    to a final dimension that potentially includes RGB values and additional features.
    A transformer encoder self.blocks is created using the out_encoderlayer and out_encoder with specified dimensions
    and configurations.
    '''
    def __init__(self,
                 pc_branch,
                 img_branch, 
                 config):
        super().__init__()

        if hasattr(config, "hp_overlap_ratio"):
            self.hp_overlap_ratio = config.hp_overlap_ratio
        else:
            self.hp_overlap_ratio = 0.0
        dim_tokens = config.encoder.trans_dim
        depth = config.encoder.depth
        num_heads = config.encoder.num_heads
        mlp_ratio = config.encoder.mlp_ratio
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.token_fusion = config.token_fusion
        self.distill_loss = nn.MSELoss()

        # image, point cloud branches
        self.pc_branch = pc_branch
        self.img_branch = img_branch

        self.mask_ratio = config.mask_ratio
        self.recover_target_type = config.recover_target_type
        self.modality_img_embedding = nn.Parameter(torch.zeros(1, 1, dim_tokens))
        self.modality_pc_embedding = nn.Parameter(torch.zeros(1, 1, dim_tokens))

        # CLS token init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_tokens))

        # encoder from 3detr
        if self.token_fusion:
            self.fusion_proj = nn.Linear(dim_tokens*2, dim_tokens)
   
        dim_feedforward = config.encoder.dim_feedforward
        dropout = config.encoder.dropout
        encoder_layer = out_encoderlayer(
            d_model=dim_tokens,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.blocks = out_encoder(encoder_layer=encoder_layer, num_layers=depth)
        self.norm = norm_layer(dim_tokens)

        # joint decoder
        if config.decoder.depth:
            self.is_joint_decoder = config.decoder.depth
            decoder_depth = config.decoder.depth
            decoder_dim = config.decoder.trans_dim
            decoder_num_heads = config.decoder.num_heads
            decoder_mlp_ratio = config.decoder.mlp_ratio

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_dim, decoder_num_heads, decoder_mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])
            
            self.decoder_norm = norm_layer(decoder_dim)
            self.decoder_modality_img_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.decoder_modality_pc_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.apply(self._init_weights)
        self.criterion = FrequencyLoss(
            loss_gamma=config.freq_loss.loss_gamma,
            matrix_gamma=config.freq_loss.matrix_gamma,
            patch_factor=config.freq_loss.patch_factor,
            ave_spectrum=config.freq_loss.ave_spectrum,
            with_matrix=config.freq_loss.with_matrix,
            log_matrix=config.freq_loss.log_matrix,
            batch_matrix=config.freq_loss.batch_matrix).cuda()
        self.normalize_img = torchvision.transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

        '''self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=192,
                out_channels=384, kernel_size=1),
            nn.Conv2d(
                in_channels=384,
                out_channels=config.encoder_stride**2*3, kernel_size=1),
            nn.PixelShuffle(config.encoder_stride),   # Pixelshuffle会将为(∗,r**2*C,H,W) reshape成(∗,C,rH,rW)
        )  ''' # 这样的decoder或许太简单了？
        self.decoder1 = nn.ModuleList([
            TBlock(192, 3, 4.0, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) for i in range(2)])
        self.decoder2 = nn.Sequential(
            norm_layer(192),
            nn.Linear(192, 16 ** 2 * 3, bias=True)
        )

        # BP4D
        self.freq_decoder_pos_embed = nn.Parameter(torch.zeros(1, 196, 192),
                                              requires_grad=False)  # fixed sin-cos embedding, ## changed
        # Affwild
        # self.freq_decoder_pos_embed = nn.Parameter(torch.zeros(1, 49, 192),
        #                                       requires_grad=False)  # fixed sin-cos embedding, ## changed
        self.get_freq_loss = FrequencyLoss()
        self.in_chans = 3
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

    def _init_weights(self, m):
        '''
        If the module m is an instance of nn.Linear (a fully connected layer), the weights are initialized with a
        truncated normal distribution using a utility function trunc_normal_ with a standard deviation of 0.02.

        If the module m is an instance of nn.LayerNorm, the bias is set to 0 and the weights are set to 1.0. In layer
        normalization, biases and weights are parameters that are learned to scale and shift the normalized values.
        Initializing the weights to 1.0 and biases to 0 means that, initially, the normalization layer is equivalent to
        an identity operation, not affecting the normalized values.
        '''
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

    def generate_input_info(self, input__token_img, input_token_pc):
        '''
        The generate_input_info method creates an ordered dictionary (OrderedDict) to keep track of input tokens from two
        different modalities: point cloud (pc) and image (img). This dictionary contains metadata
        about the number and index positions of the tokens for each modality in a batch of data.

        Prepare Point Cloud Information:
        The number of tokens for the point cloud (num_tokens_pc) is the second dimension of input_token_pc
        (the first dimension is the batch size).
        A dictionary d_pc is created to store the number of point cloud tokens, the starting index in a concatenated
        sequence (start_idx), and the ending index (end_idx).
        The starting index i is initialized to 0 and updated by adding the number of point cloud tokens to it.

        Prepare Image Information:
        the number of tokens for the image input (num_tokens_img) is determined by the second dimension of input__token_img.
        dictionary d_img is created to store the number of image tokens and their respective start and end indices
        in the concatenated sequence.
        The starting index i is updated again by adding the number of image tokens to it.

        The dictionaries d_pc and d_img are stored in input_info under the keys 'pc' and 'img', within a nested
        dictionary under the key 'tasks'.
        The total number of tokens from both modalities is stored in input_info under the key 'num_task_tokens'.

        input_info contains the metadata for the tokens of point cloud and image modalities, the count of tokens and
        their positions in a hypothetical concatenated sequence.
        '''
        input_info = OrderedDict()
        i = 0
        input_info['tasks'] = {}
        #PointCloud
        num_tokens_pc = input_token_pc.shape[1]
        d_pc = {
            'num_tokens': num_tokens_pc,
            'start_idx': i,
            'end_idx': i + num_tokens_pc,
        }
        i += num_tokens_pc
        input_info['tasks']['pc'] = d_pc
        #IMG
        num_tokens_img = input__token_img.shape[1]
        d_img = {
            'num_tokens': num_tokens_img,
            'start_idx': i,
            'end_idx': i + num_tokens_img,
        }
        i += num_tokens_img
        input_info['tasks']['img'] = d_img
        input_info['num_task_tokens'] = i
        return input_info

    def random_masking(self, x, mask_ratio):
        """
        返回：
        函数返回 x_masked，即只包含未屏蔽标记的张量。
        掩码张量，值为 1 表示标记已被掩码，0 表示标记未被掩码。
        ids_restore，可用于将序列恢复到原来的排序。

        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        The random_masking function is designed to apply random masking to a sequence of data, (Masked Autoencoders (MAE).
        1. Input Parameters:
        x: N is the batch size, L is the sequence length, and D is the dimensionality of each token or feature in the sequence.
        2. Determine Number of Tokens to Keep:
        len_keep is calculated as the ceiling of the product of the sequence length L and (1 - mask_ratio), which
        determines the number of tokens that will remain unmasked. 未被掩码
        3. Generate Random Noise and Shuffle IDs:
        noise is a tensor of random values with shape [N, L], generated for each sequence in the batch.
        ids_shuffle is obtained by argsorting the noise tensor along the sequence length dimension, which provides a
        random permutation of indices for each sequence.
        4. Restore IDs:
        ids_restore is the argsort of ids_shuffle, which will later be used to reorder the sequence back to its
        original order after processing.
        5. Select Tokens to Keep:
        ids_keep contains the first len_keep indices from ids_shuffle, indicating which tokens will not be masked.
        x_masked is created by gathering the tokens from x using ids_keep. Only the unmasked tokens are retained, and
        the masked tokens are omitted, creating a shorter sequence of unmasked tokens.
        dim=0 index矩阵中的数的值代表的是0维度（行），所处位置除0维代表的是其他维度。
        dim=1length矩阵中的数的值代表的是1维度(列)，所处位置代表的是行数。
        6. Create Mask Tensor:
        mask tensor is initialized with 1 and has the same batch and sequence dimensions as x.
        The first len_keep positions in the mask for each sequence are set to zero, indicating these tokens are not masked.
        The mask tensor is then reordered using ids_restore to match the original order of the sequences.
        7. Return Values:
        The function returns x_masked, which is the tensor containing only the unmasked tokens.
        The mask tensor, where a value of 1 indicates the token was masked and 0 indicates the token was not masked.
        ids_restore, which can be used to return the sequence to its original ordering.
        """

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(np.ceil(L * (1 - mask_ratio))) 
        
        noise = torch.rand(N, L, device=x.device)  
        
        ids_shuffle = torch.argsort(noise, dim=1) # 默认降序
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep] 
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) #每个batch里选

        mask = torch.ones([N, L], device=x.device) 
        mask[:, :len_keep] = 0  
        mask = torch.gather(mask, dim=1, index=ids_restore) # mask每行代表当前batch里的N哪些被屏蔽了，0为没有屏蔽，1为屏蔽

        return x_masked, mask, ids_restore

    def forward_tokenizer(self, pts, imgs):
        '''
        返回：
        有位置编码的图像patch特征，有位置编码的点group特征，点group中心，临点。
        The image tensor imgs is passed through the patch_embed layer of the img_branch. This layer converts the 2D image into
        a sequence of flattened 2D patches and then linearly embeds each patch into a higher-dimensional space (the x_img tensor).
        Positional embeddings are added to the embedded image patches (x_img). Transformers do not inherently understand the order
        of the input data, so positional embeddings are used to inject positional information into the sequence of image patches.
        The group_divider function from the pc_branch is applied to the point cloud data (pts). This function organizes the points
        into groups and calculates some form of centroid point for each group, resulting in neighborhood (grouped points) and center points.
        The MAE_encoder of the pc_branch is then used to encode the neighborhood point groups into a feature representation (x_pc).
        Positional embeddings for the point cloud data are generated by applying the pos_embed function to the center tensor.
        The encoded point cloud features (x_pc) are then combined with their positional embeddings (pos).
        '''
        x_img = self.img_branch.patch_embed(imgs)  # img 转化为patch
        # add pos embed
        x_img = x_img + self.img_branch.pos_embed  # no cls token

        neighborhood, center = self.pc_branch.group_divider(pts)   # 将点云数据转化为不同的group，输出为中心点，及其对应的邻居点
        x_pc = self.pc_branch.MAE_encoder.encoder(neighborhood)  # B, Group, C
        pos = self.pc_branch.MAE_encoder.pos_embed(center)
        x_pc = x_pc + pos
    
        return x_img, x_pc, center, neighborhood
    
    def forward_pc_decoder(self, x_vis, center, mask):
        '''
        返回：
        联合解码器：全部特征，位置编码，mask数量
        The forward_pc_decoder method is a model for processing point clouds. The method handles the decoding part of
        the masked autoencoder for the point cloud data.
        1. input:
        The visible point cloud features x_vis are passed through a linear layer decoder_embed to possibly change their
        dimension to match that of the decoder.
        2. Preparing Positional Embeddings:
        Positional embeddings are generated for the visible (pos_emd_vis) and masked (pos_emd_mask) points using the
        decoder_pos_embed layer.
        This is done by selecting the corresponding centers using the mask tensor. These embeddings are then reshaped
        to match the feature dimension C.
        3. Mask Tokens:
        A mask_token, which is learned during training, is replicated to match the number of masked points in the batch
        B and the number of masked points N.
        The visible features x_vis and the replicated mask_token are concatenated to form x_full, which represents the
        full sequence with masked tokens included.
        4. Concatenating Positional Embeddings:
        The positional embeddings for the visible and masked points are concatenated to form pos_full, which provides
        positional information for the entire sequence,
        including the masked tokens.
        5. Joint Decoder Processing (Conditional):
        If the model is configured with a joint decoder (self.is_joint_decoder is True), the method returns the
        concatenated features x_full,
        the full positional embeddings pos_full, and the number of masked tokens N without passing them through the decoder.
        6. Decoding:
        If there is no joint decoder, x_full and pos_full are passed to the MAE_decoder to reconstruct the full sequence.
        The decoder takes in the entire sequence (visible and masked) and outputs the reconstructed sequence x_rec_full.
        7. Extracting Reconstructed Masked Points:
        The method then extracts the reconstructed features for the masked points (x_rec_masked) by slicing the output
        tensor to only include masked points.
        8. Return Values:
        The method returns x_rec_masked, which contains the reconstructed features for the masked points, and x_rec_full,
        the entire reconstructed sequence.
        '''
        x_vis = self.pc_branch.decoder_embed(x_vis)  # change dim
        B, _, C = x_vis.shape  # B VIS C
        pos_emd_vis = self.pc_branch.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
        pos_emd_mask = self.pc_branch.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        # print(pos_emd_vis.shape[1], pos_emd_mask.shape[1]) 52 76
        _, N, _ = pos_emd_mask.shape   # 这里的N表示mask的数量，即为pos_emd_mask.shape[1]=7
        mask_token = self.pc_branch.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

        full_length = x_full.shape[1]
        if self.is_joint_decoder:
            return x_full, pos_full, N

        x_rec_full = self.pc_branch.MAE_decoder(x_full, pos_full, full_length)
        x_rec_masked = x_rec_full[:, -N:, :]  # only include masked points

        return x_rec_masked, x_rec_full  

    def forward_pc_loss(self, x_rec, neighborhood, center, mask=None, align_props=None, img_feat=None):
        """
        Calculates the point cloud reconstruction loss. If self.pc2img, project point cloud to image, interpolate features
        And calculate cross-modality reconstruction loss.
        1. inputs：
        The method takes reconstructed point cloud features x_rec, the original neighborhood (grouped points), center
        (representative points), and mask (indicating which points were masked) as inputs. Optionally, it can also receive
        alignment properties align_props and image features img_feat for cross-modality loss calculation.
        2. Distillation Loss Calculation (Conditional on self.pc2img):
        If the model is configured with the pc2img flag, the method computes a cross-modality reconstruction loss:
        extracted the masked points in the original point cloud (neighborhood) and denormalized using the center tensor.
        These points are then projected onto the image plane using the provided alignment properties (align_props),
        resulting in projected_u and projected_v coordinates.
        Features from the corresponding locations in the image (img_feat) are extracted.
        The model then reconstructs these image features from the reconstructed point cloud features (x_rec), and a distillation
        loss (mean squared error) is computed between the extracted image features and the reconstructed image features.
        3. Point Cloud Reconstruction Loss:
        The method uses the increase_dim layer of pc_branch to upsample the reconstructed point cloud features (x_rec)
        into 3D space (rebuild_points).
        The ground truth points corresponding to the masked regions are extracted from neighborhood.
        A point cloud reconstruction loss is computed between the rebuild_points and the ground truth points using
        a loss function defined in pc_branch.
        4. Return Values:
        The method returns the point cloud reconstruction loss (pc_loss), the distillation loss (distill_loss), and
        the reconstructed points (rebuild_points).
        The distillation loss is only computed and returned if self.pc2img is True. Otherwise, it will be None.
        This method is crucial for training the PiMAE model, especially in a self-supervised setting where the model
        learns to reconstruct masked portions of
        the input. The inclusion of cross-modality distillation loss (when self.pc2img is True) indicates that the
        model also learns to align and reconstruct
        across different data modalities (point cloud to image), which is useful for tasks involving multi-modal learning.
        """
        B, M, C = x_rec.shape 

        rebuild_points = self.pc_branch.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)
        gt_points = neighborhood.reshape(B * M, -1, 3)
        pc_loss = self.pc_branch.loss_func(rebuild_points, gt_points)    

        return pc_loss, gt_points, rebuild_points
    
    def forward_img_decoder(self, x):
        '''
        The forward_img_decoder method handles the decoding process for the image modality. It reconstructs the image from
        its encoded form and applies optional processing if the model is configured with a joint decoder.
        1. Embedding and Mask Token Concatenation:
        The input x (encoded image features) is first passed through a linear layer decoder_embed to potentially change its dimensionality.
        mask_tokens, which are learned during training, are replicated to match the batch size and the length of the sequence after
        including masked tokens. These mask tokens represent the parts of the input that were masked during the encoding phase.
        The visible tokens (x) and mask tokens are concatenated to form x_, recreating the full length of the original sequence.
        The gather operation rearranges x_ according to ids_restore to restore the original order of the tokens in the sequence.
        2. Adding Positional Embeddings:
        Positional embeddings (decoder_pos_embed) are added to x to provide spatial information. This step is essential as Transformers
        do not inherently understand the order or position of the input data.
        3. Joint Decoder Conditional:
        If the model is configured with a joint decoder (self.is_joint_decoder is True), the method returns x immediately after adding
        positional embeddings, potentially for further joint processing with another modality.
        4. Specific Decoder Processing:
        If there is no joint decoder, x is passed sequentially through each block in decoder_blocks, which likely consists of
        Transformer layers or similar constructs for processing the sequence.
        After passing through all the decoder blocks, x is normalized using decoder_norm.
        5. Reconstruction of Image:
        The final step involves passing the normalized features through decoder_pred to reconstruct the image.
        6. Return Values:
        If there is no joint decoder, the method returns x_feature, which represents the features after decoding and
        normalization, and x_rec, which is the reconstructed image.
        If there is a joint decoder, only x is returned after adding positional embeddings.
        '''
        x = self.img_branch.decoder_embed(x)  # 降维
        #mask_tokens = self.img_branch.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        #x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        #x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        #x = x_  # no cls token

        x = x + self.img_branch.decoder_pos_embed
        if self.is_joint_decoder:  
            return x

        # Specific decoders
        for blk in self.img_branch.decoder_blocks:
            x = blk(x)
        x_feature = self.img_branch.decoder_norm(x)
        x_rec = self.img_branch.decoder_pred(x_feature)
        x_rec = self.img_branch.unpatchify(x_rec)
        # 将shape为Bs x 49 x 192的特征x_feature，输入卷积层为基础的decoder前需要先转化shape为Bs x 192 x 7 x 7
        h, w = self.img_branch.patch_embed.grid_size
        x_freq_rec = self.decoder(x_feature.permute(0, 2, 1).reshape(x_feature.shape[0], x_feature.shape[2], h, w))
        return x_feature, x_rec, x_freq_rec

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)

        The _mask_center_block is designed to generate a mask for point cloud data based on their spatial distribution.
        This method is particularly useful in self-supervised learning scenarios where part of the data is masked and the
        model learns to predict or reconstruct these masked portions.
        1. Skipping Mask Creation (Optional):
        If noaug is True or self.mask_ratio is 0, the function immediately returns a mask of zeros, indicating that no
        points are masked. This is likely a condition to bypass masking during certain phases of training or inference.
        2. Mask Generation:
        The method iterates over each set of points (center) in the batch. Each points tensor has a shape of G x 3, where
        G is the number of groups or points.
        For each set of points, it selects a random point and calculates the Euclidean distance (distance_matrix) of all
        points in the set from this randomly selected point.
        The distances are sorted in ascending order, and indices (idx) corresponding to the sorted distances are obtained.
        Based on the mask_ratio, a certain number of points closest to the randomly selected point are chosen to be masked. mask value is 1.
        3. Constructing the Final Mask:
        For each set of points, a boolean mask (mask.bool()) is created and added to mask_idx.
        These individual masks are then stacked together to form a single bool_masked_pos tensor that represents the
        mask for the entire batch. This tensor indicates which points in each set are masked (True) and which are not (False).
        4. Return Value:
        The method returns bool_masked_pos, a boolean tensor with the same batch and group dimensions as center. Each
        True value in this tensor indicates that the corresponding point is masked.
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        mask_idx = []
        for points in center:
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1) 

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device) 

        return bool_masked_pos

    def forward_img_loss(self, imgs, pred):
        """
                imgs: [N, 3, H, W]
                pred: [N, L, p*p*3]
                mask: [N, L], 0 is keep, 1 is remove,
        The forward_img_loss calculates the reconstruction loss for the image modality.
        1. Input Parameters:
        imgs: The original images with shape [N, 3, H, W], where N is the batch size, 3 is the number of color channels,
        and H, W are the height and width of the images.
        pred: The reconstructed image patches with shape [N, L, p*p*3], where L is the number of patches, and each patch
        is flattened into a vector of length p*p*3.
        mask: A mask tensor with shape [N, L], where 0 indicates patches that are kept (unmasked), and 1 indicates patches that are removed (masked).
        2. Patchify Original Images:
        The patchify method of the img_branch is used to divide the original images into patches, resulting in a tensor similar in shape to pred.
        3. Normalization (Conditional):
        If norm_pix_loss is set in img_branch, pixel-wise normalization is applied to the target patches. This involves
        subtracting the mean and dividing by the standard deviation for each patch, which can help stabilize training and improve convergence.
        4. Loss Calculation:
        The loss is calculated as the mean squared error between the predicted (pred) and target (target) patches. This
        is a per-patch loss, computed element-wise and then averaged over each patch's elements.
        The loss is then scaled by the mask tensor. Since the mask value is 1 for removed (masked) patches, this step ensures
        that the loss is computed only for those patches that were masked and supposed to be reconstructed by the model.
        the method calculates the mean of these scaled losses, effectively giving the average loss over all masked patches.

        """
        target = self.img_branch.patchify(imgs)
        #if self.img_branch.norm_pix_loss:
        #    mean = target.mean(dim=-1, keepdim=True)
        #    var = target.var(dim=-1, keepdim=True)
        #    target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        return loss

    def countPatchIndex(self, uv, img_token_dimenson):
        '''
        The countPatchIndex method is designed to create a mask indicating which image patches correspond
        to certain UV coordinates. This function is especially useful in scenarios where you need to map or relate features
        from one modality (like point clouds) to another (like images), based on their spatial alignment.
        1. Input Parameters:
        uv: UV coordinates tensor. The shape seems to be [N, 2], where N is the number of points, and each point has a U and V coordinate.
        img_token_dimenson: The total number of patches in the image, calculated as the product of the number of patches
        along the height and width dimensions.
        2. Initialize Mask:
        A mask tensor of zeros is initialized with the size equal to img_token_dimenson. This tensor will be used to mark
        which patches are associated with the given UV coordinates.
        3. Mapping UV Coordinates to Patch Indices:
        The method iterates over each UV coordinate pair in uv.
        For each pair, it calculates the corresponding patch index based on the assumption that the image is divided into
        patches of size 16x16. This is done by dividing the U and V coordinates by 16 and using the result to calculate
        the linear index (tempIndex) in the flattened patch array.
        The calculated index is constrained to be within valid patch range (tempIndex < 16*22 and tempIndex > 0). img_size: 256*352
        4. Update Mask:
        If a valid index is found for a UV pair, the corresponding position in the mask is set to 1.
        5. Return Value:
        The method returns the mask tensor, 1 at positions corresponding to the patches that the UV coordinates map to.
        '''
        mask = torch.zeros(img_token_dimenson)
        for i in range(uv.shape[0]):
            tempIndex = int(uv[i][1]/16)*22+int(uv[i][0]/16)
            if(tempIndex<16*22 and tempIndex>0):
                mask[tempIndex] = 1
        return mask
    
    def shuffle_to_pc_mask(self, pc_bool_mask, x, mask_ratio):
        """
        Shuffles img patch indexes to pc's boolean mask.    0 visible, 1 masked
        The shuffle_to_pc_mask method in the PiMAE class appears to align the masking of image patches with a given point
        cloud boolean mask (pc_bool_mask). This method can be particularly useful in scenarios where alignment between point
        cloud and image data is necessary, such as in multi-modal learning.
        1. Input Parameters:
        pc_bool_mask: A boolean mask for the point cloud data, indicating which points are visible (0) and which are masked (1).
        x: The image data tensor in the shape [N, L, D], where N is the batch size, L is the length of the sequence
        (number of patches), and D is the dimensionality of each patch.
        mask_ratio: The ratio of the image data to be masked.
        2. Determine Number of Tokens to Keep:
        len_keep is calculated as the ceiling of the product of the sequence length L and (1 - mask_ratio), ceiling向上取整
        which determines the number of patches that will remain unmasked.
        3. Create a Randomized Mask Based on Point Cloud Mask:
        noise is created as a random tensor with values between [0, 0.5]. This noise is added to the pc_bool_mask to generate
        random_pc_bool_mask, which combines the point cloud mask with random noise.
        The sorting order (descending) of the indices is determined by the hp_overlap_ratio. If the ratio is 1.0, the
        indices are sorted in ascending order; if it's 0.0, they are sorted in descending order.
        4. Shuffle and Restore Order:
        ids_shuffle is obtained by argsorting the random_pc_bool_mask, creating a shuffled order of indices.
        ids_restore is computed as the argsort of ids_shuffle, which will be used to restore the sequence to its original order.
        5. Select Tokens to Keep and Create Mask:
        ids_keep contains the indices of the tokens to be kept (unmasked).
        x_masked is the subset of x that corresponds to the unmasked tokens.
        mask : 0 keep and 1 indicating tokens to remove. aligned according to the original order of the tokens (ids_restore).
        6. Return Values:
        x_masked, the tensor containing only the unmasked tokens, the binary mask, and ids_restore, which can be used to
        restore the sequence to its original order.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(np.ceil(L * (1 - mask_ratio))) 
        assert pc_bool_mask.shape[0] == N 
        pc_m_ratio = torch.sum(pc_bool_mask, dim=1) 

        noise = torch.rand(N, L, device=x.device)/2  #  [0, 0.5]
        random_pc_bool_mask = pc_bool_mask + noise
        
        if self.hp_overlap_ratio == 1.0:
            descending=False
        elif self.hp_overlap_ratio == 0.0:
            descending=True
        else: 
            raise NotImplementedError

        ids_shuffle = torch.argsort(random_pc_bool_mask, dim=1, descending=descending) 
        ids_restore = torch.argsort(ids_shuffle, dim=1) 

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # binary mask: 0 keep, 1 remove
        mask = torch.ones([N, L], device=x.device)  
        mask[:, :len_keep] = 0  
        mask = torch.gather(mask, dim=1, index=ids_restore) 

        return x_masked, mask, ids_restore

                       
    def forward_joint_decoder(self, x_img, x_pc, center, mask):
        # step 1: add positional and modality embedding
        # img
        x_img = self.forward_img_decoder(x_img)
        x_img = x_img + self.decoder_modality_img_embed

        # pc
        x_pc, x_pc_pos, N = self.forward_pc_decoder(x_pc, center, mask)   # 这里为什么没有恢复原本顺序的操作？？
        x_pc = x_pc + x_pc_pos + self.decoder_modality_pc_embed   # 这里加了一次位置编码

        # step 2: concate tokens along L dim
        img_sz = x_img.shape[1]
        pc_sz = x_pc.shape[1]
        decoder_inputs = [x_img, x_pc]
        decoder_inputs = torch.cat(decoder_inputs, dim=1)

        # step 3: pass through joint decoder
        for blk in self.decoder_blocks:
            decoder_inputs = blk(decoder_inputs)
        decoder_inputs = self.decoder_norm(decoder_inputs)

        # step 4: split tokens along L dim
        rec_img = decoder_inputs[:, :img_sz, :]   # shape: Bs x 49 x 192
        rec_pc = decoder_inputs[:, -pc_sz:, :]   # shape: Bs x 128 x 192

        # step 5 (ours): pass through specific decoder for frequency
        h, w = self.img_branch.patch_embed.grid_size
        #rec_freq_img = self.decoder(rec_img.permute(0, 2, 1).reshape(rec_img.shape[0], rec_img.shape[2], h, w))
        freq_rec_img = rec_img + self.freq_decoder_pos_embed
        for blk in self.decoder1:
            freq_rec_img = blk(freq_rec_img)
        rec_freq_img = self.decoder2(freq_rec_img)
        rec_freq_img = self.img_branch.unpatchify(rec_freq_img)

        # step 6: pass through specific decoders
        # add positional embedding before decoder
        rec_img = rec_img + self.img_branch.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.img_branch.decoder_blocks:
            rec_img = blk(rec_img)
        img_feat = self.img_branch.decoder_norm(rec_img)  # shape: Bs x 49 x 192
        # predictor projection
        rec_img = self.img_branch.decoder_pred(img_feat)
        #rec_img = self.img_branch.unpatchify(rec_img)

        # ready for return
        # transformer
        rec_pc = self.pc_branch.MAE_decoder(rec_pc, x_pc_pos, N)  # normed inside the function  # 这里又加了一次位置编码
        # ready for return
        return img_feat, rec_img, rec_freq_img, rec_pc

    def get_mask(self, input_size=224, mask_radius1=12, mask_radius2=999):#BP4D224, AFFWILD112
        mask = torch.ones((input_size, input_size)).int()
        for y in range(input_size):
            for x in range(input_size):
                if ((x - input_size // 2) ** 2 + (y - input_size // 2) ** 2) >= mask_radius1 ** 2 \
                        and ((x - input_size // 2) ** 2 + (y - input_size // 2) ** 2) < mask_radius2 ** 2:
                    mask[y, x] = 0
        return 1 - mask

    def frequency_transform(self, x, mask):
        # 2D FFT
        x_freq = torch.fft.fft2(x)
        # shift low frequency to the center
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
        # mask a portion of frequencies
        x_freq_masked = x_freq * mask
        # restore the original frequency order
        x_freq_masked = torch.fft.ifftshift(x_freq_masked, dim=(-2, -1))
        # 2D iFFT (only keep the real part)
        x_corrupted = torch.fft.ifft2(x_freq_masked).real
        x_corrupted = torch.clamp(x_corrupted, min=0., max=1.)
        return x_corrupted

    def forward(self, pts_org, imgs_org, pts_lag, imgs_lag):
        save_dict = {}
        save_dict['imgs_org_gt'] = imgs_org
        save_dict['imgs_lag_gt'] = imgs_lag

        # 先对第T帧的图片进行低频打码
        imgs_mask = self.get_mask().unsqueeze(0).unsqueeze(1)
        imgs_mask = imgs_mask.repeat(imgs_org.shape[0], 1, 1, 1).to(imgs_org.device)  # 这里是让batch中的每一帧打相同的码
        imgs_org_corrupted = self.frequency_transform(imgs_org, imgs_mask)
        save_dict['imgs_org_corrupted'] = imgs_org_corrupted

        imgs_org = self.normalize_img(imgs_org)
        imgs_org_corrupted = self.normalize_img(imgs_org_corrupted)
        imgs_lag = self.normalize_img(imgs_lag)

        # tokenizing, embedding，加上位置编码和模态编码
        # 模型接收的是第T帧的有损图片和第T+n帧的mesh
        img_token, pc_token, center, neighborhood = self.forward_tokenizer(pts_lag, imgs_org_corrupted)
        img_token = img_token + self.modality_img_embedding
        pc_token = pc_token + self.modality_pc_embedding
        
        if self.pc_branch.MAE_encoder.mask_type == 'rand':   # 这里使用rand还是block
            bool_masked_pos = self.pc_branch.MAE_encoder._mask_center_rand(center, noaug=False)  # 随机mask点云数据， mask为1，非mask为0
        else:
            bool_masked_pos = self.pc_branch.MAE_encoder._mask_center_block(center, noaug=False)

        B, L, C = pc_token.shape
        pc_vis = pc_token[~bool_masked_pos].reshape(B, -1, C)  # 得到可见的点云数据
        img_vis = img_token  # img_vis即为img_token本身

        img_input_size = img_vis.shape[1]
        pc_input_size = pc_vis.shape[1]
      
        # img specific encoder
        for blk in self.img_branch.blocks:
            img_vis = blk(img_vis)

        # pc specific encoder
        pc_vis = pc_vis.transpose(0, 1)  # B,M,C -> M,B,C
        ret_pc = self.pc_branch.MAE_encoder.blocks(pc_vis, return_attn_weights=False)
        pc_vis = ret_pc[1]      # returns xyz, output, xyz_inds, and we need only output
        pc_vis = pc_vis.transpose(0, 1)  # M,B,C -> B,M,C
        
        # concat tokens
        inputs_token = [img_vis, pc_vis]
        inputs_token = torch.cat(inputs_token, dim=1)

        # add cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(inputs_token.shape[0], -1, -1)
        inputs_token = torch.cat((cls_tokens, inputs_token), dim=1)

        # shared encoder
        inputs_token = inputs_token.transpose(0, 1)  # B,M,C -> M,B,C
        ret_joint = self.blocks(inputs_token, return_attn_weights=False)
        inputs_token = ret_joint[1]    # returns xyz, output, xyz_inds, and we need only output
        inputs_token = inputs_token.transpose(0, 1)
        
        img_vis = inputs_token[:, 1:1+img_input_size, :]  # remove cls token  B x L x D
        pc_vis = inputs_token[:, -pc_input_size:, :]

        if self.is_joint_decoder:
            img_feat, img_rec, freq_img_rec, pc_rec_masked = self.forward_joint_decoder(x_img=img_vis, x_pc=pc_vis, center=center, mask=bool_masked_pos)
        else:
            img_feat, img_rec, freq_img_rec = self.forward_img_decoder(x=img_vis)
            pc_rec_masked, pc_rec_full = self.forward_pc_decoder(x_vis=pc_vis, center=center, mask=bool_masked_pos)
            N = pc_rec_full.shape[1] - pc_rec_masked.shape[1]
            pc_rec_vis = pc_rec_full[:, :N, :]

        # get img loss for lag
        img_loss = self.forward_img_loss(imgs=imgs_lag, pred=img_rec)  # shape: B x C x H x W

        # get frequency loss for org
        # imgs_mask中被mask的为0，没有被mask为1
        if self.recover_target_type == 'masked':
            freq_loss = self.get_freq_loss(freq_img_rec, imgs_org)  # freq_loss的shape是[16, 1, 3, 112, 112]，imgs_mask的shape是[16, 1, 112, 112]
            freq_loss = (freq_loss * (1 - imgs_mask.unsqueeze(1))).sum() / (1 - imgs_mask).sum() / self.in_chans /freq_loss.shape[1]   # 这里的self.in_chans是3吗？
        elif self.recover_target_type == 'normal':
            freq_loss = self.get_freq_loss(freq_img_rec, imgs_org)
            freq_loss = freq_loss.mean()
        # get pc loss
        # 因为需要重建的第T帧的mesh数据，本身是不一样的，因此不需要mask数据
        # print(pc_rec_masked.shape)  Bs, 128, 192 如果只返回被mask的部分，shape是Bs, 76, 192
        neighborhood_org, center_org = self.pc_branch.group_divider(pts_org)
        pc_loss, gt_points, rebuild_points = self.forward_pc_loss(x_rec=pc_rec_masked, neighborhood=neighborhood_org, center=center_org)
        # 保存特征
        save_dict['imgs_org_pred'] = freq_img_rec
        save_dict['imgs_lag_pred'] = img_rec
        save_dict['rebuild_points'] = rebuild_points
        save_dict['gt_points'] = gt_points

        return img_loss, freq_loss, pc_loss, save_dict







