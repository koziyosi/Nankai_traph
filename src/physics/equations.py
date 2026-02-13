"""
支配方程式モジュール - Governing Equations
準動的平衡と状態変数の発展を統合
"""

import numpy as np
from numba import njit, prange
from typing import Tuple, Optional, Callable
from dataclasses import dataclass, field

from .friction import (
    RateStateFriction,
    compute_friction_stress_parallel,
    compute_state_evolution_parallel
)
from .stress_kernel import StressKernel


@dataclass
class QuasiDynamicEquations:
    """
    準動的地震サイクル方程式
    
    τ_s = τ_f （剪断応力 = 摩擦応力）の平衡を準動的に解く
    
    参考: Hirose et al. (2022), Rice (1993)
    """
    
    friction: RateStateFriction = field(default_factory=RateStateFriction)
    kernel: StressKernel = field(default_factory=StressKernel)
    
    # 物理パラメータ
    G: float = 30.0e9       # 剛性率 [Pa]
    beta: float = 3750.0    # S波速度 [m/s]
    eta: float = 1.0        # 準動的補正係数
    
    # GPU設定
    use_gpu: bool = False
    _velocity_kernel = None
    _evolution_kernel = None
    
    def __post_init__(self):
        self.kernel.G = self.G
        self.kernel.beta = self.beta
        self.kernel.eta = self.eta
        
        if self.use_gpu:
            try:
                import cupy as cp
                from .cupy_kernels import get_velocity_solver_kernel, get_state_evolution_kernel
                self._velocity_kernel = get_velocity_solver_kernel()
                self._evolution_kernel = get_state_evolution_kernel()
                print("GPU Kernels loaded successfully.")
            except ImportError as e:
                print(f"CuPy not found or kernel init failed: {e}. Fallback to CPU.")
                self.use_gpu = False

    def compute_derivatives(self,
                            t: float,
                            state: np.ndarray,
                            n_cells: int,
                            a: np.ndarray,
                            b: np.ndarray,
                            L: np.ndarray,
                            sigma_eff: np.ndarray,
                            V_pl: np.ndarray
                            ) -> np.ndarray:
        """
        状態変数の時間微分を計算
        """
        if self.use_gpu and self._velocity_kernel is not None:
            return self._compute_derivatives_gpu(
                t, state, n_cells, a, b, L, sigma_eff, V_pl
            )
        
        # 状態変数を分解
        tau = state[:n_cells]      # 剪断応力
        theta = state[n_cells:]    # 状態変数
        
        # すべり速度を求める（準動的平衡から） (CPU)
        V = self._solve_velocity(tau, theta, a, b, L, sigma_eff)
        
        # 応力の時間微分 (CPU)
        dtau_dt = _compute_stress_rate(
            self.kernel.K, V, V_pl, self.G, self.beta, self.eta
        )
        
        # 状態変数の時間微分 (CPU)
        dtheta_dt = compute_state_evolution_parallel(
            V, theta, L, self.friction.V_c
        )
        
        return np.concatenate([dtau_dt, dtheta_dt])

    def _compute_derivatives_gpu(self, t, state, n_cells, a, b, L, sigma_eff, V_pl):
        """GPUを使用した一括計算"""
        import cupy as cp
        
        # 配列をGPUへ転送 (オーバーヘッドはあるが、行列演算で取り返す)
        tau_gpu = cp.asarray(state[:n_cells])
        theta_gpu = cp.asarray(state[n_cells:])
        
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        L_gpu = cp.asarray(L)
        sigma_eff_gpu = cp.asarray(sigma_eff)
        V_pl_gpu = cp.asarray(V_pl)
        
        # カーネル行列 (K) をGPUへ
        if not hasattr(self.kernel, '_K_gpu'):
             print("Transferring Kernel matrix to GPU...")
             self.kernel._K_gpu = cp.asarray(self.kernel.K)
             print("Kernel transfer complete.")
        K_gpu = self.kernel._K_gpu
        
        # 1. 速度計算 (Newton) - Elementwise Kernel
        V_gpu = self._velocity_kernel(
            tau_gpu, theta_gpu, a_gpu, b_gpu, L_gpu, sigma_eff_gpu,
            self.G, self.beta, self.eta,
            self.friction.mu_0, self.friction.V_0, self.friction.V_c
        )
        
        # 2. 応力変化率 dtau/dt = K @ (V - V_pl)
        dV_gpu = V_gpu - V_pl_gpu
        dtau_dt_gpu = cp.dot(K_gpu, dV_gpu)
        
        # 3. 状態変数変化率
        dtheta_dt_gpu = self._evolution_kernel(
            V_gpu, theta_gpu, L_gpu, self.friction.V_c
        )
        
        # 結果結合
        result_gpu = cp.concatenate([dtau_dt_gpu, dtheta_dt_gpu])
        
        return cp.asnumpy(result_gpu)
    
    def _solve_velocity(self,
                        tau: np.ndarray,
                        theta: np.ndarray,
                        a: np.ndarray,
                        b: np.ndarray,
                        L: np.ndarray,
                        sigma_eff: np.ndarray) -> np.ndarray:
        """準動的平衡からすべり速度を求める (CPU版)"""
        return _solve_velocity_newton(
            tau, theta, a, b, L, sigma_eff,
            self.G, self.beta, self.eta,
            self.friction.mu_0, self.friction.V_0, self.friction.V_c
        )
    
    def get_slip_velocity(self,
                          tau: np.ndarray,
                          theta: np.ndarray,
                          a: np.ndarray,
                          b: np.ndarray,
                          L: np.ndarray,
                          sigma_eff: np.ndarray) -> np.ndarray:
        """すべり速度を取得（外部アクセス用）"""
        if self.use_gpu and self._velocity_kernel is not None:
             import cupy as cp
             # GPU版
             return cp.asnumpy(self._velocity_kernel(
                cp.asarray(tau), cp.asarray(theta), cp.asarray(a), cp.asarray(b), cp.asarray(L), cp.asarray(sigma_eff),
                self.G, self.beta, self.eta,
                self.friction.mu_0, self.friction.V_0, self.friction.V_c
             ))
        return self._solve_velocity(tau, theta, a, b, L, sigma_eff)
    
    def initialize_state(self,
                         n_cells: int,
                         sigma_eff: np.ndarray,
                         a: np.ndarray,
                         b: np.ndarray,
                         L: np.ndarray,
                         V_init: float = 3.17e-10  # 0.01 m/year
                         ) -> np.ndarray:
        """初期状態を設定"""
        V = np.full(n_cells, V_init)
        
        # 定常状態の Θ
        theta = L / V
        
        # 初期応力（定常摩擦から）
        mu_init = (self.friction.mu_0 + 
                   a * np.log(V / self.friction.V_0 + 1) +
                   b * np.log(self.friction.V_0 * theta / L + 1))
        tau = sigma_eff * mu_init
        
        # 放射減衰項を追加
        tau += self.eta * self.G / self.beta * V
        
        return np.concatenate([tau, theta])


# ==============================================================================
# Numba JIT 最適化された計算関数
# ==============================================================================

@njit(cache=True, fastmath=True)
def _compute_stress_rate(K: np.ndarray,
                          V: np.ndarray,
                          V_pl: np.ndarray,
                          G: float,
                          beta: float,
                          eta: float) -> np.ndarray:
    """
    応力の時間微分を計算 (CPU)
    """
    n = len(V)
    dtau_dt = np.zeros(n)
    
    # 相対速度 (dV = V - V_pl)
    dV = V - V_pl
    
    # K @ dV
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += K[i, j] * dV[j]
        dtau_dt[i] = s
    
    return dtau_dt


@njit(cache=True, fastmath=True)
def _solve_velocity_newton(tau: np.ndarray,
                            theta: np.ndarray,
                            a: np.ndarray,
                            b: np.ndarray,
                            L: np.ndarray,
                            sigma_eff: np.ndarray,
                            G: float,
                            beta: float,
                            eta: float,
                            mu_0: float,
                            V_0: float,
                            V_c: float,
                            max_iter: int = 50,
                            tol: float = 1.0e-10) -> np.ndarray:
    """
    Newton法で すべり速度を求める (CPU)
    """
    n = len(tau)
    V = np.empty(n)
    
    rad_coef = eta * G / beta
    
    for i in range(n):
        V_guess = V_0
        theta_i = max(theta[i], 1.0e-20)
        
        for iteration in range(max_iter):
            V_safe = max(V_guess, 1.0e-20)
            
            term1 = a[i] * np.log(V_safe / V_0 + 1.0)
            term2 = b[i] * np.log(V_0 * theta_i / L[i] + 1.0)
            mu_v = mu_0 + term1 + term2
            
            F = tau[i] - sigma_eff[i] * mu_v - rad_coef * V_safe
            
            dmu_dV = a[i] / (V_safe + V_0)
            dF_dV = -sigma_eff[i] * dmu_dV - rad_coef
            
            if abs(dF_dV) < 1.0e-30:
                break
            
            dV = -F / dF_dV
            V_new = V_guess + dV
            if V_new < 1.0e-20:
                V_new = V_guess * 0.1
            elif V_new > 1.0e3:
                V_new = 1.0e3
            
            if abs(dV) < tol * abs(V_guess) + tol:
                V_guess = V_new
                break
            
            V_guess = V_new
        
        V[i] = max(V_guess, 1.0e-20)
    
    return V


def create_derivative_function(equations: QuasiDynamicEquations,
                               n_cells: int,
                               a: np.ndarray,
                               b: np.ndarray,
                               L: np.ndarray,
                               sigma_eff: np.ndarray,
                               V_pl: np.ndarray) -> Callable:
    """
    ODEソルバー用の微分関数を作成
    """
    
    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        return equations.compute_derivatives(
            t, state, n_cells, a, b, L, sigma_eff, V_pl
        )
    
    return derivative
