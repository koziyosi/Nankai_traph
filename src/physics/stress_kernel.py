"""
弾性応力カーネルモジュール - Stress Kernel Calculation
転位理論に基づく Green関数を使用
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional
from dataclasses import dataclass
import os


@dataclass
class StressKernel:
    """
    弾性応力伝達カーネル K_ij
    
    j番目セルの単位すべりによる i番目セルへの静的剪断応力変化
    
    参考文献:
        - Mura (1987) Micromechanics of Defects in Solids
        - Kuroki et al. (2002) JGR
    """
    
    G: float = 30.0e9          # 剛性率 [Pa]
    nu: float = 0.25           # ポアソン比
    beta: float = 3750.0       # S波速度 [m/s]
    eta: float = 1.0           # 準動的補正係数
    
    # カーネル行列
    K: np.ndarray = None       # [n_cells, n_cells]
    K_self: np.ndarray = None  # 対角成分 K_ii
    
    # 最適化用
    _is_computed: bool = False
    
    def compute(self, 
                centers: np.ndarray,
                areas: np.ndarray,
                normals: np.ndarray,
                slip_direction: np.ndarray = None) -> np.ndarray:
        """
        応力カーネル行列を計算
        
        Parameters:
            centers: セル中心座標 [n_cells, 3] [km]
            areas: セル面積 [n_cells] [km²]
            normals: セル法線ベクトル [n_cells, 3]
            slip_direction: すべり方向 [n_cells, 3] (None の場合は N60°W)
        
        Returns:
            K: 応力カーネル行列 [n_cells, n_cells]
        """
        n_cells = len(centers)
        print(f"応力カーネル計算開始: {n_cells} x {n_cells} 行列")
        
        # km → m に変換
        centers_m = centers * 1000.0
        areas_m2 = areas * 1.0e6
        
        # すべり方向（デフォルト: N60°W = (sin(330°), cos(330°), 0)）
        if slip_direction is None:
            azimuth = np.radians(330.0)  # N60°W
            slip_direction = np.zeros((n_cells, 3))
            slip_direction[:, 0] = np.sin(azimuth)  # x成分
            slip_direction[:, 1] = np.cos(azimuth)  # y成分
        
        # 並列計算
        self.K = _compute_kernel_matrix(
            centers_m, areas_m2, normals, slip_direction,
            self.G, self.nu
        )
        
        # 対角成分を保存
        self.K_self = np.diag(self.K).copy()
        
        self._is_computed = True
        print(f"応力カーネル計算完了")
        
        return self.K
    
    def apply(self, slip: np.ndarray, V_pl: np.ndarray, t: float) -> np.ndarray:
        """
        応力カーネルを適用して剪断応力を計算
        
        τ_s = Σ_j K_ij * (V_pl_j * t - u_j)
        
        Parameters:
            slip: 累積すべり量 [n_cells] [m]
            V_pl: プレート収束速度 [n_cells] [m/s]
            t: 時間 [s]
        
        Returns:
            tau_s: 剪断応力 [n_cells] [Pa]
        """
        if not self._is_computed:
            raise RuntimeError("Kernel not computed yet. Call compute() first.")
        
        # 相対すべり（プレート運動 - 断層すべり）
        relative_slip = V_pl * t - slip
        
        # 行列-ベクトル積
        tau_s = self.K @ relative_slip
        
        return tau_s
    
    def apply_velocity(self, V: np.ndarray) -> np.ndarray:
        """
        すべり速度に対する応力変化率を計算
        
        dτ/dt = -Σ_j K_ij * V_j  (プレート収束は一定とする)
        
        Parameters:
            V: すべり速度 [n_cells] [m/s]
        
        Returns:
            dtau_dt: 応力変化率 [n_cells] [Pa/s]
        """
        return -self.K @ V
    
    def get_radiation_damping(self, V: np.ndarray) -> np.ndarray:
        """
        準動的近似の放射減衰項を計算
        
        η * G / β * V
        
        Parameters:
            V: すべり速度 [n_cells] [m/s]
        
        Returns:
            damping: 放射減衰項 [n_cells] [Pa]
        """
        return self.eta * self.G / self.beta * V
    
    def save(self, filepath: str):
        """カーネル行列をファイルに保存"""
        np.savez_compressed(
            filepath,
            K=self.K,
            K_self=self.K_self,
            G=self.G,
            nu=self.nu,
            beta=self.beta
        )
        print(f"カーネルを保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'StressKernel':
        """ファイルからカーネル行列を読み込み"""
        data = np.load(filepath)
        
        kernel = cls(
            G=float(data['G']),
            nu=float(data['nu']),
            beta=float(data['beta'])
        )
        kernel.K = data['K']
        kernel.K_self = data['K_self']
        kernel._is_computed = True
        
        print(f"カーネルを読み込み: {filepath}")
        return kernel


# ==============================================================================
# Numba JIT 最適化された計算関数
# ==============================================================================

@njit(parallel=True, cache=True)
def _compute_kernel_matrix(centers: np.ndarray,
                            areas: np.ndarray,
                            normals: np.ndarray,
                            slip_direction: np.ndarray,
                            G: float,
                            nu: float) -> np.ndarray:
    """
    応力カーネル行列の計算（並列化・JIT最適化）
    
    三角形転位による応力場の近似計算
    
    参考: Okada (1992) の点転位近似を使用
    """
    n = len(centers)
    K = np.zeros((n, n))
    
    # ラメ定数
    lam = 2.0 * G * nu / (1.0 - 2.0 * nu)
    
    for i in prange(n):
        xi = centers[i, 0]
        yi = centers[i, 1]
        zi = centers[i, 2]
        ni = normals[i]
        si = slip_direction[i]
        
        for j in range(n):
            if i == j:
                # 自己応力（Eshelby の解を使用）
                # K_ii ∝ G / L where L is cell dimension
                cell_size = np.sqrt(areas[j])
                K[i, j] = -G / (2.0 * np.pi * cell_size)
            else:
                xj = centers[j, 0]
                yj = centers[j, 1]
                zj = centers[j, 2]
                
                # 距離
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r < 1.0:  # 距離が1m未満は無視
                    continue
                
                # 点転位による応力（簡易版）
                # σ_ij ∝ G * Δu * A / r³
                # 
                # より正確な計算は Okada (1992) を参照
                
                r3 = r * r * r
                r5 = r3 * r * r
                
                # すべり方向成分
                slip_j = slip_direction[j]
                
                # 応力テンソルの計算（簡略化）
                # 剪断応力を slip_direction と normal の内積で評価
                
                # Green関数の係数
                coef = G * areas[j] / (2.0 * np.pi * r3)
                
                # 方向係数
                # τ = n_i · σ_ij · s_i
                r_vec = np.array([dx, dy, dz]) / r
                
                # 応力テンソルの寄与（点転位近似）
                # 主要項のみ
                n_dot_r = ni[0]*r_vec[0] + ni[1]*r_vec[1] + ni[2]*r_vec[2]
                s_dot_r = si[0]*r_vec[0] + si[1]*r_vec[1] + si[2]*r_vec[2]
                sj_dot_r = slip_j[0]*r_vec[0] + slip_j[1]*r_vec[1] + slip_j[2]*r_vec[2]
                
                # 剪断応力の寄与
                K[i, j] = coef * (3.0 * n_dot_r * s_dot_r * sj_dot_r - 
                                  (ni[0]*slip_j[0] + ni[1]*slip_j[1] + ni[2]*slip_j[2]) *
                                  (si[0]*r_vec[0] + si[1]*r_vec[1] + si[2]*r_vec[2]))
    
    return K


@njit(cache=True, fastmath=True)
def okada_stress(x: float, y: float, z: float,
                 strike: float, dip: float,
                 L: float, W: float,
                 slip: float,
                 G: float, nu: float) -> Tuple[float, float, float, float, float, float]:
    """
    Okada (1992) による矩形転位の応力場（未実装）
    
    TODO: 完全な実装が必要
    """
    # プレースホルダー
    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


@njit(parallel=True, cache=True, fastmath=True)
def apply_kernel_parallel(K: np.ndarray, slip: np.ndarray) -> np.ndarray:
    """
    カーネル行列の適用（並列化）
    
    τ = K @ slip
    """
    n = K.shape[0]
    tau = np.zeros(n)
    
    for i in prange(n):
        s = 0.0
        for j in range(n):
            s += K[i, j] * slip[j]
        tau[i] = s
    
    return tau


def compute_kernel_fft(centers: np.ndarray,
                       cell_size: float,
                       G: float,
                       nu: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    FFTを使用した高速カーネル計算（将来の最適化用）
    
    Regular gridの場合に有効
    O(N²) → O(N log N)
    
    TODO: 実装
    """
    raise NotImplementedError("FFT kernel computation not yet implemented")
