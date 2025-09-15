# -*- coding: utf-8 -*-
import torch
from skimage import measure
from scipy.spatial.distance import cdist
import numpy as np
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

import cv2
_EPS = np.spacing(1)    # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64


class ShapeContext:
    def __init__(self, nbins_r=5, nbins_theta=12, rmin=0.125, rmax=2.0, N=200, thr=0.5):
        self.shape_context_similarities = []
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta
        self.rmin, self.rmax = rmin, rmax
        self.N = N
        self.thr = thr

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
        return x

    def _to_binary(self, m):
        m = self._to_numpy(m)
        m = m.squeeze()  # (1,H,W) -> (H,W) 등
        # 값 범위로 logit 여부 추정
        if m.min() < 0.0 or m.max() > 1.0:
            m = 1.0 / (1.0 + np.exp(-m))  # sigmoid
        return (m > self.thr).astype(np.float32)

    def _extract_contours(self, binary_image):
        binary_image = self._to_numpy(binary_image)
        if binary_image.ndim == 3 and binary_image.shape[0] == 1:
            binary_image = binary_image[0]
        if binary_image.shape[0] < 2 or binary_image.shape[1] < 2:
            return None
        contours = measure.find_contours(binary_image, 0.5)
        if contours:
            return max(contours, key=len)  # (L,2) with (row=y, col=x)
        return None

    def _uniform_sample(self, curve_xy, N):
        # curve_xy: (L,2) as (y,x). 보편적 (x,y)로 변환
        pts = np.stack([curve_xy[:,1], curve_xy[:,0]], axis=1)  # (L,2) -> (x,y)
        L = len(pts)
        if L == 0:
            return None
        if L < N:
            idx = np.linspace(0, L-1, num=N).astype(int)
            return pts[idx]
        # 누적호 길이 기반 등간격 샘플링
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        total = cum[-1] if cum[-1] > 0 else 1.0
        targets = np.linspace(0, total, num=N)
        sampled = []
        j = 0
        for t in targets:
            while j+1 < len(cum) and cum[j+1] < t:
                j += 1
            if j+1 >= len(pts):
                sampled.append(pts[-1])
            else:
                ratio = (t - cum[j]) / max(1e-8, (cum[j+1] - cum[j]))
                sampled.append(pts[j] * (1-ratio) + pts[j+1] * ratio)
        return np.asarray(sampled, dtype=np.float32)

    def _compute_shape_context(self, points):
        # points: (N,2) in (x,y)
        eps = np.finfo(float).eps
        # 스케일 정규화: 평균 최근접거리 등 대신 평균 점간거리로 간단히
        D = cdist(points, points) + eps
        mean_dist = np.mean(D)
        Dn = D / max(mean_dist, eps)  # scale invariance

        # 각 anchor i에 대해 이웃 j와의 (log r, theta)
        angles = np.arctan2(points[:,1][:,None] - points[:,1],
                            points[:,0][:,None] - points[:,0])
        log_r = np.log(np.clip(Dn, self.rmin, self.rmax))  # 고정 반경 범위 적용 후 log

        # bin edges (고정 범위 기반)
        r_edges = np.linspace(np.log(self.rmin), np.log(self.rmax), self.nbins_r + 1)
        t_edges = np.linspace(-np.pi, np.pi, self.nbins_theta + 1)

        H = np.zeros((len(points), self.nbins_r * self.nbins_theta), dtype=np.float32)
        for i in range(len(points)):
            r_idx = np.digitize(log_r[i], r_edges) - 1
            t_idx = np.digitize(angles[i], t_edges) - 1
            valid = (r_idx >= 0) & (r_idx < self.nbins_r) & (t_idx >= 0) & (t_idx < self.nbins_theta) & (np.arange(len(points)) != i)
            bins = r_idx[valid] * self.nbins_theta + t_idx[valid]
            np.add.at(H[i], bins, 1.0)
        # 히스토그램 정규화(합=1)
        H /= (H.sum(axis=1, keepdims=True) + eps)
        return H

    def _chi2_cost(self, H1, H2):
        eps = np.finfo(float).eps
        costs = np.zeros((len(H1), len(H2)), dtype=np.float32)
        for i, h1 in enumerate(H1):
            # 벡터화 가능하지만 명료하게 유지
            for j, h2 in enumerate(H2):
                diff = (h1 - h2) ** 2
                s = h1 + h2 + eps
                costs[i, j] = 0.5 * np.sum(diff / s)
        return costs

    def _bidirectional_min_cost(self, H1, H2):
        C = self._chi2_cost(H1, H2)
        a = np.mean(np.min(C, axis=1))  # H1→H2
        b = np.mean(np.min(C, axis=0))  # H2→H1
        return 0.5 * (a + b)

    def step(self, pred, gt):
        # logit도 들어올 수 있으니 내부에서 바이너리화
        pred = self._to_binary(pred)
        gt   = self._to_binary(gt)

        c_pred = self._extract_contours(pred)
        c_gt   = self._extract_contours(gt)
        if c_pred is None or c_gt is None:
            self.shape_context_similarities.append(0.0)
            return

        sp = self._uniform_sample(c_pred, self.N)
        sg = self._uniform_sample(c_gt, self.N)
        if sp is None or sg is None:
            self.shape_context_similarities.append(0.0)
            return

        Hp = self._compute_shape_context(sp)
        Hg = self._compute_shape_context(sg)

        cost = self._bidirectional_min_cost(Hp, Hg)
        sim = 1.0 / (1.0 + cost)  # 비용→유사도(보상)
        self.shape_context_similarities.append(float(sim))

    def get_results(self):
        return {"shape_context": float(np.mean(self.shape_context_similarities)) if self.shape_context_similarities else 0.0}

'''
class ShapeContext:
    def __init__(self, nbins_r=5, nbins_theta=12):
        self.shape_context_similarities = []
        self.nbins_r = nbins_r
        self.nbins_theta = nbins_theta

    def _extract_contours(self, binary_image):
        if isinstance(binary_image, torch.Tensor):
            binary_image = binary_image.detach().cpu().numpy()

    # Remove singleton channel dimension if exists (e.g., shape: (1, H, W) → (H, W))
        if binary_image.ndim == 3 and binary_image.shape[0] == 1:
            binary_image = binary_image[0]
  
    # Check if the image is too small for contour detection
        if binary_image.shape[0] < 2 or binary_image.shape[1] < 2:
            print(f"[ShapeContext] Skipping small input: shape={binary_image.shape}") 
            return None

        contours = measure.find_contours(binary_image, 0.5)
        if contours:
            max_contour = max(contours, key=len)
            return max_contour
        return None


    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)
        pred_contour = self._extract_contours(pred)
        gt_contour = self._extract_contours(gt)
        if pred_contour is None or gt_contour is None:
            self.shape_context_similarities.append(0.0)
            return
        sc_pred = self._compute_shape_context(pred_contour)
        sc_gt = self._compute_shape_context(gt_contour)
        similarity_score = self._shape_context_matching(sc_pred, sc_gt)
        self.shape_context_similarities.append(similarity_score)

    def _compute_shape_context(self, points):
        distances = cdist(points, points)
        angles = np.arctan2(points[:, 1][:, np.newaxis] - points[:, 1],
                            points[:, 0][:, np.newaxis] - points[:, 0])
        log_distances = np.log(distances + np.finfo(float).eps)
        shape_contexts = np.zeros((len(points), self.nbins_r * self.nbins_theta))
        r_bin_edges = np.linspace(np.min(log_distances), np.max(log_distances), self.nbins_r + 1)
        theta_bin_edges = np.linspace(-np.pi, np.pi, self.nbins_theta + 1)
        for i, (log_r, theta) in enumerate(zip(log_distances, angles)):
            r_bin_idx = np.digitize(log_r, r_bin_edges) - 1
            theta_bin_idx = np.digitize(theta, theta_bin_edges) - 1
            valid_idx = (r_bin_idx >= 0) & (r_bin_idx < self.nbins_r) & (theta_bin_idx >= 0) & (theta_bin_idx < self.nbins_theta)
            for r_idx, theta_idx in zip(r_bin_idx[valid_idx], theta_bin_idx[valid_idx]):
                shape_contexts[i, r_idx * self.nbins_theta + theta_idx] += 1
        return shape_contexts

    def _shape_context_matching(self, sc1, sc2):
        distances = np.zeros((len(sc1), len(sc2)))
        for i, sc1_hist in enumerate(sc1):
            for j, sc2_hist in enumerate(sc2):
                diff = (sc1_hist - sc2_hist) ** 2
                sum_sc = (sc1_hist + sc2_hist)
                distances[i, j] = 0.5 * np.sum(diff / (sum_sc + np.finfo(float).eps))
        return np.mean(np.min(distances, axis=1))

    def get_results(self):
        return dict(shape_context=np.mean(self.shape_context_similarities) if self.shape_context_similarities else 0.0)
        '''


