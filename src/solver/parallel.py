"""
並列計算アクセラレータモジュール
Numba, マルチプロセス, GPU(CuPy) 対応
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import os
import multiprocessing as mp


# GPU対応チェック
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from numba import njit, prange, set_num_threads, get_num_threads


@dataclass
class ParallelAccelerator:
    """
    並列計算アクセラレータ
    
    利用可能な場合はGPU (CuPy) を使用、
    そうでなければ Numba による CPU並列化を使用
    """
    
    use_gpu: bool = False
    n_threads: int = -1  # -1 = 自動検出
    
    # 内部状態
    _initialized: bool = False
    _actual_threads: int = 0
    _device_name: str = "CPU"
    
    def __post_init__(self):
        self.initialize()
    
    def initialize(self):
        """初期化"""
        if self._initialized:
            return
        
        # CPU スレッド数設定
        if self.n_threads == -1:
            self._actual_threads = mp.cpu_count()
        else:
            self._actual_threads = max(1, min(self.n_threads, mp.cpu_count()))
        
        set_num_threads(self._actual_threads)
        
        # GPU チェック
        if self.use_gpu and HAS_CUPY:
            try:
                cp.cuda.Device(0).compute_capability
                self._device_name = cp.cuda.Device(0).name
                print(f"GPU 検出: {self._device_name}")
            except Exception as e:
                print(f"GPU 初期化失敗: {e}, CPU にフォールバック")
                self.use_gpu = False
                self._device_name = "CPU"
        else:
            self.use_gpu = False
            self._device_name = "CPU"
        
        print(f"計算デバイス: {self._device_name}, スレッド数: {self._actual_threads}")
        self._initialized = True
    
    def to_device(self, arr: np.ndarray) -> np.ndarray:
        """配列をデバイスに転送"""
        if self.use_gpu and HAS_CUPY:
            return cp.asarray(arr)
        return arr
    
    def to_host(self, arr) -> np.ndarray:
        """配列をホストに転送"""
        if self.use_gpu and HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return arr
    
    def matmul(self, A, x):
        """行列-ベクトル積"""
        if self.use_gpu and HAS_CUPY:
            return cp.dot(A, x)
        return np.dot(A, x)
    
    def parallel_apply(self, 
                       func: Callable,
                       data: np.ndarray,
                       *args,
                       chunk_size: int = None) -> np.ndarray:
        """
        関数を並列適用
        """
        if self.use_gpu and HAS_CUPY:
            # GPU版
            data_gpu = cp.asarray(data)
            args_gpu = [cp.asarray(a) if isinstance(a, np.ndarray) else a 
                        for a in args]
            result = func(data_gpu, *args_gpu)
            return cp.asnumpy(result)
        else:
            # CPU版（Numba使用を想定）
            return func(data, *args)


# ==============================================================================
# GPU (CuPy) 最適化された計算関数
# ==============================================================================

def gpu_matmul(K: np.ndarray, x: np.ndarray) -> np.ndarray:
    """GPU上での行列-ベクトル積"""
    if HAS_CUPY:
        K_gpu = cp.asarray(K)
        x_gpu = cp.asarray(x)
        result = cp.dot(K_gpu, x_gpu)
        return cp.asnumpy(result)
    return np.dot(K, x)


def gpu_solve_velocity(tau: np.ndarray,
                       theta: np.ndarray,
                       a: np.ndarray,
                       b: np.ndarray,
                       L: np.ndarray,
                       sigma_eff: np.ndarray,
                       params: dict) -> np.ndarray:
    """
    GPU上でのNewton法による速度計算
    """
    # プレースホルダー - CPU版にフォールバック
    from ..physics.equations import _solve_velocity_newton
    return _solve_velocity_newton(
        tau, theta, a, b, L, sigma_eff,
        params['G'], params['beta'], params['eta'],
        params['mu_0'], params['V_0'], params['V_c']
    )


# ==============================================================================
# マルチプロセス版（大規模計算用）
# ==============================================================================

def _worker_kernel_chunk(args):
    """カーネル計算のワーカー関数"""
    i_start, i_end, centers, areas, normals, slip_dir, G, nu = args
    
    n = len(centers)
    K_chunk = np.zeros((i_end - i_start, n))
    
    for i in range(i_start, i_end):
        for j in range(n):
            if i == j:
                cell_size = np.sqrt(areas[j])
                K_chunk[i - i_start, j] = -G / (2.0 * np.pi * cell_size)
            else:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dz = centers[i, 2] - centers[j, 2]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                if r < 1.0:
                    continue
                
                r3 = r * r * r
                coef = G * areas[j] / (2.0 * np.pi * r3)
                
                r_vec = np.array([dx, dy, dz]) / r
                ni = normals[i]
                si = slip_dir[i]
                slip_j = slip_dir[j]
                
                n_dot_r = ni[0]*r_vec[0] + ni[1]*r_vec[1] + ni[2]*r_vec[2]
                s_dot_r = si[0]*r_vec[0] + si[1]*r_vec[1] + si[2]*r_vec[2]
                sj_dot_r = slip_j[0]*r_vec[0] + slip_j[1]*r_vec[1] + slip_j[2]*r_vec[2]
                
                K_chunk[i - i_start, j] = coef * (
                    3.0 * n_dot_r * s_dot_r * sj_dot_r -
                    (ni[0]*slip_j[0] + ni[1]*slip_j[1] + ni[2]*slip_j[2]) *
                    (si[0]*r_vec[0] + si[1]*r_vec[1] + si[2]*r_vec[2])
                )
    
    return i_start, K_chunk


def compute_kernel_multiprocess(centers: np.ndarray,
                                 areas: np.ndarray,
                                 normals: np.ndarray,
                                 slip_direction: np.ndarray,
                                 G: float,
                                 nu: float,
                                 n_workers: int = None) -> np.ndarray:
    """
    マルチプロセスでカーネル行列を計算
    
    大規模なメッシュの場合に有効
    """
    n = len(centers)
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # チャンクに分割
    chunk_size = max(1, n // n_workers)
    chunks = []
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        chunks.append((i, i_end, centers, areas, normals, slip_direction, G, nu))
    
    print(f"カーネル計算: {n_workers} プロセス, {len(chunks)} チャンク")
    
    # 並列実行
    K = np.zeros((n, n))
    
    with mp.Pool(n_workers) as pool:
        results = pool.map(_worker_kernel_chunk, chunks)
    
    for i_start, K_chunk in results:
        K[i_start:i_start + K_chunk.shape[0]] = K_chunk
    
    return K


# ==============================================================================
# ベンチマーク・診断ユーティリティ
# ==============================================================================

def benchmark_matmul(n: int = 1000, iterations: int = 100) -> dict:
    """
    行列-ベクトル積のベンチマーク
    """
    import time
    
    A = np.random.randn(n, n)
    x = np.random.randn(n)
    
    # NumPy版
    start = time.time()
    for _ in range(iterations):
        _ = np.dot(A, x)
    numpy_time = (time.time() - start) / iterations
    
    result = {'numpy': numpy_time}
    
    # CuPy版
    if HAS_CUPY:
        try:
            A_gpu = cp.asarray(A)
            x_gpu = cp.asarray(x)
            
            # ウォームアップ
            _ = cp.dot(A_gpu, x_gpu)
            cp.cuda.Stream.null.synchronize()
            
            start = time.time()
            for _ in range(iterations):
                _ = cp.dot(A_gpu, x_gpu)
            cp.cuda.Stream.null.synchronize()
            cupy_time = (time.time() - start) / iterations
            
            result['cupy'] = cupy_time
            result['speedup'] = numpy_time / cupy_time
        except Exception as e:
            result['cupy_error'] = str(e)
    
    return result


def get_system_info() -> dict:
    """システム情報を取得"""
    info = {
        'cpu_count': mp.cpu_count(),
        'numba_threads': get_num_threads(),
        'has_cupy': HAS_CUPY,
    }
    
    if HAS_CUPY:
        try:
            device = cp.cuda.Device(0)
            info['gpu_name'] = device.name
            info['gpu_memory'] = device.mem_info[1] / 1e9  # GB
        except Exception as e:
            # エラーをキャッチしてクラッシュを防ぐ
            info['gpu_error'] = f"Failed to access Device(0): {e}"
            info['has_cupy'] = False
    
    return info
