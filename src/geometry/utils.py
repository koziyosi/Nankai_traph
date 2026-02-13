"""
幾何学ユーティリティ - Geometry Utilities
"""

import numpy as np
from scipy.spatial import KDTree
from typing import List, Dict, Optional, Any

def estimate_normals_optimized(centers: np.ndarray, k: int = 5) -> np.ndarray:
    """
    点群から法線ベクトルを推定する (KDTree使用, O(N log N))
    最近傍点を用いて局所平面を当てはめる

    Parameters:
        centers: 点群座標 [n, 3]
        k: 近傍探索点数

    Returns:
        normals: 法線ベクトル [n, 3]
    """
    n = len(centers)
    # 法線配列を初期化
    normals = np.zeros((n, 3))

    print(f"法線ベクトル推定中 (KDTree, k={k})...")

    # KDTree構築
    tree = KDTree(centers)

    # 各点について、近傍点を探す
    # k=3 (自分 + 2点) 以上必要
    dists, indices = tree.query(centers, k=k)

    for i in range(n):
        # 自身(0)を除く近傍点
        # indices[i] は [self, neighbor1, neighbor2, ...]

        p0 = centers[i]

        # 近傍点を使って平面を推定
        # トライ＆エラーで非同一直線上の点を探す
        found = False
        for j1 in range(1, k):
            p1 = centers[indices[i, j1]]
            v1 = p1 - p0
            if np.linalg.norm(v1) < 1e-6: continue

            for j2 in range(j1 + 1, k):
                p2 = centers[indices[i, j2]]
                v2 = p2 - p0
                if np.linalg.norm(v2) < 1e-6: continue

                # 外積
                n_vec = np.cross(v1, v2)
                norm = np.linalg.norm(n_vec)

                if norm > 1e-10:
                    n_vec /= norm
                    found = True
                    break
            if found: break

        if not found:
            # フォールバック
            n_vec = np.array([0.0, 0.0, -1.0])

        # 向きの調整: Z成分が負（上向き）になるように統一
        # Z-down座標系なので、上向きはZが減少する方向
        if n_vec[2] > 0:
            n_vec = -n_vec

        normals[i] = n_vec

    return normals


def classify_segments(cells: List[Any], slip: np.ndarray, threshold: float = 1.0) -> List[str]:
    """
    すべり分布からセグメントを特定

    Parameters:
        cells: Cellオブジェクトのリスト（segment属性を持つこと）
        slip: すべり分布配列
        threshold: 判定閾値 [m]

    Returns:
        segments: 破壊されたセグメントIDのリスト
    """
    seg_slips = {}

    for i, cell in enumerate(cells):
        if slip[i] > threshold and hasattr(cell, 'segment') and cell.segment:
            if cell.segment not in seg_slips:
                seg_slips[cell.segment] = []
            seg_slips[cell.segment].append(slip[i])

    return list(seg_slips.keys())
