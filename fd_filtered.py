import cv2
import numpy as np

# ------------------------------
# 1) FD 계산 (box-counting)
# ------------------------------
def box_counting_fd(patch, use_edges=True, min_fd=0.5):
    g = cv2.normalize(patch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if use_edges:
        g = cv2.Canny(g, 100, 200)
    _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    box_sizes = [s for s in [2,4,8,16,32] if s <= min(binary.shape)]
    counts = []
    for s in box_sizes:
        rs = (binary.shape[1]//s, binary.shape[0]//s)
        reduced = cv2.resize(binary, rs, interpolation=cv2.INTER_NEAREST)
        counts.append(np.sum(reduced>0))

    if len(set(counts)) <= 1 or all(c==0 for c in counts):
        return min_fd

    log_sizes = -np.log(np.array(box_sizes, np.float32)+1e-6)
    log_counts = np.log(np.maximum(counts,1))
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return slope

# ------------------------------
# 2) FD 맵 생성
# ------------------------------
def generate_fd_map(img_bgr, patch_size=16, scale=4):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    down = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale),
                      interpolation=cv2.INTER_AREA)
    fd_small = np.zeros_like(down, np.float32)

    Hs, Ws = down.shape
    for r in range(0, Hs, patch_size):
        for c in range(0, Ws, patch_size):
            patch = down[r:r+patch_size, c:c+patch_size]
            if patch.size==0: continue
            val = box_counting_fd(patch)
            fd_small[r:r+patch_size, c:c+patch_size] = val

    return cv2.resize(fd_small, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

# ------------------------------
# 3) Soft-selective filtering (강화 버전)
# ------------------------------
def fd_soft_filter_stronger(img_bgr, fd_map,
                            low_th=1.0, high_th=1.4,
                            large_kernel=41, power=2.0):
    """
    FD 낮음: 원본 유지
    FD 높음: large median (강하게)
    중간 FD: (원본 ↔ large) 비선형 보간
    """
    if large_kernel % 2 == 0: 
        large_kernel += 1
    large_f = cv2.medianBlur(img_bgr, large_kernel)

    # 0~1 정규화 + 비선형 power
    alpha_lin = np.clip((fd_map - low_th) / (high_th - low_th + 1e-6), 0.0, 1.0)
    alpha = alpha_lin ** power   # power↑ → 고FD에서 더 강하게 large 쪽으로

    out = np.empty_like(img_bgr, dtype=np.float32)

    low_mask  = (fd_map < low_th)
    high_mask = (fd_map > high_th)
    mid_mask  = ~(low_mask | high_mask)

    out[low_mask]  = img_bgr[low_mask]
    out[high_mask] = large_f[high_mask]
    a = alpha[..., None].astype(np.float32)
    out[mid_mask]  = (1.0 - a[mid_mask]) * img_bgr[mid_mask] + a[mid_mask] * large_f[mid_mask]

    return np.clip(out, 0, 255).astype(np.uint8)

# ------------------------------
# 4) Iterative filtering with k-means check
# ------------------------------
def fd_iterative_filter(img_bgr, max_iter=5, tol=1e-3,
                        low_th=1.0, high_th=1.4,
                        large_kernel=41, power=2.0):
    fd_map = generate_fd_map(img_bgr)
    prev_centers = None
    out = img_bgr.copy()

    for t in range(max_iter):
        # k-means on FD values
        data = fd_map.reshape(-1,1).astype(np.float32)
        _, labels, centers = cv2.kmeans(data, K=3, bestLabels=None,
                                        criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,1e-3),
                                        attempts=3, flags=cv2.KMEANS_PP_CENTERS)
        centers = np.sort(centers.flatten())
        if prev_centers is not None:
            diff = np.max(np.abs(centers-prev_centers))
            if diff < tol:
                break
        prev_centers = centers

        # filtering (강화 버전)
        out = fd_soft_filter_stronger(out, fd_map,
                                      low_th=low_th, high_th=high_th,
                                      large_kernel=large_kernel, power=power)
        # update FD map
        fd_map = generate_fd_map(out)

    return out


if __name__ == "__main__":
    img = cv2.imread("491.jpg")  # 입력 이미지
    result = fd_iterative_filter(img,
                                 max_iter=6,   # 반복 횟수 ↑ → 더 강하게
                                 low_th=1.0, high_th=1.4,
                                 large_kernel=41,  # median 커널 키우면 더 부드럽게
                                 power=2.0)       # power↑ → 고FD 쪽 강화
    cv2.imwrite("Filtered_491.jpg", result)

