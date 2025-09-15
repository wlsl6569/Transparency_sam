
# ./models/mmseg/models/sam
# with the meta prompt


import numpy as np
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # --- 원래 point/box ---
        self.num_point_embeddings: int = 4
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # --- 원래 mask/dense ---
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # --- meta-aware ---
        self.meta_point_embed = nn.Embedding(3, 16)   # [B,H,U] 3클래스
        self.point_fuse = nn.Linear(embed_dim + 16, embed_dim)
        self.point_meta_dropout = nn.Dropout(p=0.20)

        self.meta_downscaling = nn.Sequential(
            nn.Conv2d(3, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.meta_dropout2d = nn.Dropout2d(p=0.20)

    # --------------------------
    # GT mask -> meta map 생성
    # --------------------------
    @staticmethod
    def generate_meta_map(gt_mask: np.ndarray, band_width: int = 3) -> np.ndarray:
        """
        gt_mask: (H,W), 0=bg, 1=fg
        return: (3,H,W) meta_map [B,H,U]
        """
        gt = gt_mask.astype(np.uint8)
        kernel = np.ones((band_width, band_width), np.uint8)

        eroded = cv2.erode(gt, kernel, iterations=1)
        dilated = cv2.dilate(gt, kernel, iterations=1)

        boundary = gt - eroded          # 경계 내부 band
        uncertain = dilated - gt        # 경계 외부 band
        hard = eroded                   # 내부 안정 영역

        meta_map = np.stack([boundary, hard, uncertain], axis=0).astype(np.float32)
        return meta_map

    def get_dense_pe(self) -> torch.Tensor:
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
        point_meta_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        points = points + 0.5
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)

        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0]  += self.point_embeddings[0].weight
        point_embedding[labels == 1]  += self.point_embeddings[1].weight

        if point_meta_labels is not None:
            meta_tok = self.meta_point_embed(point_meta_labels.clamp(min=0))
            meta_tok = self.point_meta_dropout(meta_tok)
            point_embedding = torch.cat([point_embedding, meta_tok], dim=-1)
            point_embedding = self.point_fuse(point_embedding)

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks) -> int:
        if points is not None: return points[0].shape[0]
        elif boxes is not None: return boxes.shape[0]
        elif masks is not None: return masks.shape[0]
        else: return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    @torch.no_grad()
    def sample_point_meta_labels(
        self,
        gt_masks: torch.Tensor,       # (B,H,W)
        points_px: torch.Tensor,      # (B,N,2)
        band_width: int = 3,
    ) -> torch.Tensor:
        """
        GT mask -> meta map -> 샘플링
        """
        B, H, W = gt_masks.shape
        meta_labels = []
        for b in range(B):
            meta_map = self.generate_meta_map(gt_masks[b].cpu().numpy(), band_width)  # (3,H,W)
            meta_map = torch.from_numpy(meta_map).to(points_px.device)
            xy = points_px[b].long().clamp_min(0)
            xy[..., 0] = xy[..., 0].clamp_max(W - 1)
            xy[..., 1] = xy[..., 1].clamp_max(H - 1)
            vals = meta_map[:, xy[:,1], xy[:,0]].T   # (N,3)
            meta_labels.append(vals.argmax(dim=1))   # (N,)
        return torch.stack(meta_labels, dim=0)       # (B,N)

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        *,
        gt_masks: Optional[torch.Tensor] = None,     # (B,H,W) GT mask
        point_meta_labels: Optional[torch.Tensor] = None,
        use_dense_meta: bool = True,
    ):
        bs = self._get_batch_size(points, boxes, masks)
        device = self._get_device()

        # sparse path
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=device)
        if points is not None:
            coords, labels = points
            if (point_meta_labels is None) and (gt_masks is not None):
                point_meta_labels = self.sample_point_meta_labels(gt_masks, coords)
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None),
                                                  point_meta_labels=point_meta_labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        # dense path
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # dense meta injection
        if use_dense_meta and (gt_masks is not None):
            meta_maps = []
            for b in range(gt_masks.shape[0]):
                meta_map = self.generate_meta_map(gt_masks[b].cpu().numpy())  # (3,H,W)
                meta_maps.append(torch.from_numpy(meta_map))
            meta_maps = torch.stack(meta_maps).to(dense_embeddings.device)   # (B,3,H,W)

            meta_down = F.interpolate(meta_maps, size=self.mask_input_size, mode="nearest")
            meta_feat = self.meta_downscaling(meta_down)
            meta_feat = self.meta_dropout2d(meta_feat)
            dense_embeddings = dense_embeddings + meta_feat

        return sparse_embeddings, dense_embeddings


