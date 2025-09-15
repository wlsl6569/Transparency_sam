import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple
from sod_metric import ShapeContext

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None, sc_reward_weight=0.00):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.image_encoder = ImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )


        # --- Start of Fix ---
        # Initialize flags to ensure they always exist
        self.use_iou_loss = False
        self.use_sc_reward = False
        # --- End of Fix ---

        # EVP Handling - check if self.encoder exists or if it should be self.image_encoder
        if 'evp' in encoder_mode.get('name', ''): # Use .get for safety
             logger.warning("EVP prompt freezing logic found. Ensure 'self.image_encoder' is intended.")
             # Assuming self.image_encoder is the target here based on SAM structure
             for k, p in self.image_encoder.named_parameters():
                 # Make sure the condition logic is correct for your specific EVP setup
                 if "prompt" not in k: # Example condition, adjust if needed
                     p.requires_grad = False
             # Also freeze mask decoder if needed? Check EVP paper/implementation details.
             # for k, p in self.mask_decoder.named_parameters():
             #    p.requires_grad = False


        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            # Flags remain False (already initialized)
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()
            # Flags remain False (already initialized)
        elif self.loss_mode == 'iou' or self.loss_mode == 'iou_sc': # Combine IOU and SC
            # Use BBCE if you preferred it before for the base loss
            self.criterionBCE = torch.nn.BCEWithLogitsLoss() # Or BBCEWithLogitLoss()
            self.criterionIOU = IOU()
            self.use_iou_loss = True # Set flag to True
            self.use_sc_reward = (self.loss_mode == 'iou_sc') # Enable SC only if 'iou_sc'
        else:
            # If loss_mode is None or invalid, raise error
            raise ValueError(f"Unknown or invalid loss mode specified: {self.loss_mode}")


        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])
        self.sc_reward_weight = sc_reward_weight # Store the weight

    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size, : input_size]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def backward_G(self):
        """
        Backpropagation with segmentation loss + optional IOU loss + optional ShapeContext reward.
        """
        bs = self.pred_mask.shape[0]
        self.loss_G = self.criterionBCE(self.pred_mask, self.gt_mask)
        iou_loss_val = 0.0
        reinforce_loss_val = 0.0

        if self.use_iou_loss:
            iou_loss = self.criterionIOU(self.pred_mask, self.gt_mask)
            self.loss_G += iou_loss
            iou_loss_val = iou_loss.item()


        if self.use_sc_reward:
            # REINFORCE-style SC reward - calculated per sample
            with torch.no_grad(): # Ensure reward calculation doesn't affect gradients
                prob = torch.sigmoid(self.pred_mask) # [B, 1, H, W]
                dist = torch.distributions.Bernoulli(probs=prob) # Use probs for stability
                sampled_mask = dist.sample() # [B, 1, H, W], binary 0.0 or 1.0

                batch_rewards = []
                sc_metric = ShapeContext() # Initialize once
                gt_masks_np = self.gt_mask.cpu().numpy() # [B, 1, H, W]

                # Iterate through batch samples
                for i in range(bs):
                    pred_np_i = sampled_mask[i, 0].cpu().numpy().astype(np.uint8) # [H, W]
                    gt_np_i = gt_masks_np[i, 0].astype(np.uint8) # [H, W]

                    # Ensure masks are not empty for SC calculation
                    if np.sum(pred_np_i) > 0 and np.sum(gt_np_i) > 0:
                         # Run SC calculation - ensure sod_metric handles numpy inputs
                        sc_metric.step(pred_np_i, gt_np_i)
                        sc_score = sc_metric.get_results()['shape_context']
                        # Reward: higher for lower SC score (better match)
                        # Add epsilon to avoid log(0) or division by zero issues if sc_score can be 0
                        reward_val = np.exp(-sc_score)
                    else:
                        # Handle cases with empty masks (e.g., assign zero or average reward)
                        reward_val = 0.0 # Assign low reward if prediction or GT is empty

                    batch_rewards.append(reward_val)

                # Convert list of rewards to a tensor [B]
                rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device) # Shape: [B]

            # Calculate log probability of the sampled actions (masks)
            # log_prob shape: [B, 1, H, W]
            log_prob = dist.log_prob(sampled_mask)

            # Average log_prob per sample: [B, 1, H, W] -> [B]
            # Mean over channel, height, width dimensions
            log_prob_per_sample = log_prob.mean(dim=[1, 2, 3]) # Shape: [B]

            # REINFORCE loss: E[-log_prob * R] = - mean(log_prob * R)
            # Note the negative sign
            reinforce_loss = - (log_prob_per_sample * rewards).mean() # Mean over batch dimension

            # Add scaled reinforce loss to total loss
            self.loss_G += self.sc_reward_weight * reinforce_loss
            reinforce_loss_val = reinforce_loss.item()

            # Optional: Log rewards and losses for debugging
            # logger.debug(f"Batch Rewards: {rewards.cpu().numpy()}")
            # logger.debug(f"Mean Reward: {rewards.mean().item():.4f}")
            # logger.debug(f"LogProb per sample mean: {log_prob_per_sample.mean().item():.4f}")
            # logger.debug(f"Reinforce Loss (unscaled): {reinforce_loss_val:.4f}")
            # logger.debug(f"Total Loss: {self.loss_G.item():.4f}")


        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

