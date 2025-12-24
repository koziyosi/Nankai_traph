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
    
    def __post_init__(self):
        self.kernel.G = self.G
        self.kernel.beta = self.beta
        self.kernel.eta = self.eta
    
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
        
        state = [tau_1, ..., tau_n, theta_1, ..., theta_n]
        
        Parameters:
            t: 時間 [s]
            state: 状態ベクトル [2*n_cells]
            n_cells: セル数
            a, b, L: 摩擦パラメータ
            sigma_eff: 有効法線応力 [Pa]
            V_pl: プレート収束速度 [m/s]
        
        Returns:
            d_state_dt: 時間微分 [2*n_cells]
        """
        # 状態変数を分解
        tau = state[:n_cells]      # 剪断応力
        theta = state[n_cells:]    # 状態変数
        
        # すべり速度を求める（準動的平衡から）
        V = self._solve_velocity(tau, theta, a, b, L, sigma_eff)
        
        # 応力の時間微分
        # dτ/dt = Σ_j K_ij * (V_pl_j - V_j) - η*G/β * dV/dt
        # 準動的近似では dV/dt の項を簡略化
        dtau_dt = _compute_stress_rate(
            self.kernel.K, V, V_pl, self.G, self.beta, self.eta
        )
        
        # 状態変数の時間微分
        dtheta_dt = compute_state_evolution_parallel(
            V, theta, L, self.friction.V_c
        )
        
        # 結合
        d_state_dt = np.concatenate([dtau_dt, dtheta_dt])
        
        return d_state_dt
    
    def _solve_velocity(self,
                        tau: np.ndarray,
                        theta: np.ndarray,
                        a: np.ndarray,
                        b: np.ndarray,
                        L: np.ndarray,
                        sigma_eff: np.ndarray) -> np.ndarray:
        """
        準動的平衡から すべり速度を求める
        
        τ_s = τ_f + η*G/β * V
        
        τ_f = σ_eff * [μ_0 + a*ln(V/V_0+1) + b*ln(V_0*Θ/L+1)]
        
        これは V に関する非線形方程式なので Newton法で解く
        """
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
        return self._solve_velocity(tau, theta, a, b, L, sigma_eff)
    
    def initialize_state(self,
                         n_cells: int,
                         sigma_eff: np.ndarray,
                         a: np.ndarray,
                         b: np.ndarray,
                         L: np.ndarray,
                         V_init: float = 3.17e-10  # 0.01 m/year
                         ) -> np.ndarray:
        """
        初期状態を設定
        
        論文では V(0) = 0.1 cm/year = 3.17e-11 m/s
        ここでは V(0) = 0.01 m/year = 3.17e-10 m/s を使用
        
        Parameters:
            n_cells: セル数
            sigma_eff: 有効法線応力
            a, b: 摩擦パラメータ
            L: 特徴的すべり量
            V_init: 初期すべり速度 [m/s]
        
        Returns:
            state: 初期状態 [2*n_cells]
        """
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
    応力の時間微分を計算
    
    dτ/dt = Σ_j K_ij * (V_pl_j - V_j)
    
    準動的近似では放射減衰項は速度に含まれる
    """
    n = len(V)
    dtau_dt = np.zeros(n)
    
    # 相対速度
    dV = V_pl - V
    
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
    Newton法で すべり速度を求める
    
    F(V) = τ - σ_eff * μ(V, Θ) - η*G/β * V = 0
    """
    n = len(tau)
    V = np.empty(n)
    
    # 放射減衰係数
    rad_coef = eta * G / beta
    
    for i in range(n):
        # 初期推定（線形近似）
        V_guess = V_0  # 参照速度から開始
        
        theta_i = max(theta[i], 1.0e-20)
        
        for iteration in range(max_iter):
            V_safe = max(V_guess, 1.0e-20)
            
            # F(V) の計算
            term1 = a[i] * np.log(V_safe / V_0 + 1.0)
            term2 = b[i] * np.log(V_0 * theta_i / L[i] + 1.0)
            mu_v = mu_0 + term1 + term2
            
            F = tau[i] - sigma_eff[i] * mu_v - rad_coef * V_safe
            
            # F'(V) の計算
            dmu_dV = a[i] / (V_safe + V_0)
            dF_dV = -sigma_eff[i] * dmu_dV - rad_coef
            
            # Newton更新
            if abs(dF_dV) < 1.0e-30:
                break
            
            dV = -F / dF_dV
            
            # ステップ制限
            V_new = V_guess + dV
            if V_new < 1.0e-20:
                V_new = V_guess * 0.1
            elif V_new > 1.0e3:  # 1 km/s を上限
                V_new = 1.0e3
            
            # 収束判定
            if abs(dV) < tol * abs(V_guess) + tol:
                V_guess = V_new
                break
            
            V_guess = V_new
        
        V[i] = max(V_guess, 1.0e-20)
    
    return V


@njit(parallel=True, cache=True, fastmath=True)
def compute_derivatives_parallel(K: np.ndarray,
                                  tau: np.ndarray,
                                  theta: np.ndarray,
                                  a: np.ndarray,
                                  b: np.ndarray,
                                  L: np.ndarray,
                                  sigma_eff: np.ndarray,
                                  V_pl: np.ndarray,
                                  G: float,
                                  beta: float,
                                  eta: float,
                                  mu_0: float,
                                  V_0: float,
                                  V_c: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    状態変数の時間微分を並列計算
    
    Returns:
        V: すべり速度 [n_cells]
        dtau_dt: 応力変化率 [n_cells]
        dtheta_dt: 状態変数変化率 [n_cells]
    """
    n = len(tau)
    V = np.empty(n)
    dtau_dt = np.zeros(n)
    dtheta_dt = np.empty(n)
    
    rad_coef = eta * G / beta
    
    # まず速度を計算（Newton法、並列化可能）
    for i in prange(n):
        V_guess = V_0
        theta_i = max(theta[i], 1.0e-20)
        
        for _ in range(50):
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
            V_new = max(min(V_new, 1.0e3), 1.0e-20)
            
            if abs(dV) < 1.0e-10 * abs(V_guess) + 1.0e-10:
                V_guess = V_new
                break
            V_guess = V_new
        
        V[i] = max(V_guess, 1.0e-20)
    
    # 応力変化率を計算
    for i in prange(n):
        s = 0.0
        for j in range(n):
            s += K[i, j] * (V_pl[j] - V[j])
        dtau_dt[i] = s
    
    # 状態変数変化率を計算
    for i in prange(n):
        V_i = max(V[i], 1.0e-20)
        theta_i = max(theta[i], 1.0e-20)
        x = V_i * theta_i / L[i]
        
        if x < 700:
            term1 = np.exp(-x)
        else:
            term1 = 0.0
        
        if V_c / V_i < 700:
            term2 = x * np.exp(-V_c / V_i)
        else:
            term2 = 0.0
        
        dtheta_dt[i] = term1 - term2
    
    return V, dtau_dt, dtheta_dt


def create_derivative_function(equations: QuasiDynamicEquations,
                               n_cells: int,
                               a: np.ndarray,
                               b: np.ndarray,
                               L: np.ndarray,
                               sigma_eff: np.ndarray,
                               V_pl: np.ndarray) -> Callable:
    """
    ODEソルバー用の微分関数を作成
    
    Returns:
        f(t, state) -> d_state_dt
    """
    K = equations.kernel.K
    G = equations.G
    beta = equations.beta
    eta = equations.eta
    mu_0 = equations.friction.mu_0
    V_0 = equations.friction.V_0
    V_c = equations.friction.V_c
    
    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        tau = state[:n_cells]
        theta = state[n_cells:]
        
        V, dtau_dt, dtheta_dt = compute_derivatives_parallel(
            K, tau, theta, a, b, L, sigma_eff, V_pl,
            G, beta, eta, mu_0, V_0, V_c
        )
        
        return np.concatenate([dtau_dt, dtheta_dt])
    
    return derivative
