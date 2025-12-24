"""
Rate-State Friction法則モジュール
Composite law (Kato and Tullis 2001) を実装
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
from dataclasses import dataclass


@dataclass
class RateStateFriction:
    """
    速度・状態依存摩擦則 (Rate-State Friction Law)
    
    Composite law (Kato and Tullis 2001):
    μ = μ_0 + a * ln(V/V_0 + 1) + b * ln(V_0 * Θ / L + 1)
    
    状態変数の発展則:
    dΘ/dt = exp(-V*Θ/L) - V*Θ/L * exp(-V_c/V)
    """
    
    mu_0: float = 0.6       # 定常摩擦係数
    V_0: float = 1.0e-6     # 参照速度 [m/s]
    V_c: float = 1.0e-8     # カットオフ速度 [m/s]
    
    def compute_friction_coefficient(self, 
                                     V: np.ndarray, 
                                     theta: np.ndarray,
                                     a: np.ndarray,
                                     b: np.ndarray,
                                     L: np.ndarray) -> np.ndarray:
        """
        摩擦係数を計算
        
        Parameters:
            V: すべり速度 [m/s]
            theta: 状態変数 [s]
            a, b: 摩擦パラメータ
            L: 特徴的すべり量 [m]
        
        Returns:
            mu: 摩擦係数
        """
        return _compute_friction_coefficient(
            V, theta, a, b, L, self.mu_0, self.V_0
        )
    
    def compute_friction_stress(self,
                                V: np.ndarray,
                                theta: np.ndarray,
                                a: np.ndarray,
                                b: np.ndarray,
                                L: np.ndarray,
                                sigma_eff: np.ndarray) -> np.ndarray:
        """
        摩擦応力を計算
        
        τ_f = σ_eff * μ
        """
        mu = self.compute_friction_coefficient(V, theta, a, b, L)
        return sigma_eff * mu
    
    def compute_state_evolution(self,
                                V: np.ndarray,
                                theta: np.ndarray,
                                L: np.ndarray) -> np.ndarray:
        """
        状態変数の時間発展を計算
        
        dΘ/dt = exp(-V*Θ/L) - V*Θ/L * exp(-V_c/V)
        """
        return _compute_state_evolution(V, theta, L, self.V_c)
    
    def compute_derivatives(self,
                            V: np.ndarray,
                            theta: np.ndarray,
                            a: np.ndarray,
                            b: np.ndarray,
                            L: np.ndarray,
                            sigma_eff: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        摩擦則の偏微分を計算（Newton-Raphson法用）
        
        Returns:
            dmu_dV: ∂μ/∂V
            dmu_dtheta: ∂μ/∂Θ
            dthetadot_dV: ∂(dΘ/dt)/∂V
        """
        return _compute_friction_derivatives(
            V, theta, a, b, L, sigma_eff, self.mu_0, self.V_0, self.V_c
        )


# ==============================================================================
# Numba JIT 最適化された計算関数
# ==============================================================================

@njit(cache=True, fastmath=True)
def _compute_friction_coefficient(V: np.ndarray, 
                                   theta: np.ndarray,
                                   a: np.ndarray,
                                   b: np.ndarray,
                                   L: np.ndarray,
                                   mu_0: float,
                                   V_0: float) -> np.ndarray:
    """摩擦係数の計算（JIT最適化）"""
    n = len(V)
    mu = np.empty(n)
    
    for i in range(n):
        # 数値安定性のためのクリッピング
        V_safe = max(V[i], 1.0e-20)
        theta_safe = max(theta[i], 1.0e-20)
        
        # Composite law
        term1 = a[i] * np.log(V_safe / V_0 + 1.0)
        term2 = b[i] * np.log(V_0 * theta_safe / L[i] + 1.0)
        
        mu[i] = mu_0 + term1 + term2
    
    return mu


@njit(cache=True, fastmath=True)
def _compute_state_evolution(V: np.ndarray,
                              theta: np.ndarray,
                              L: np.ndarray,
                              V_c: float) -> np.ndarray:
    """状態変数の時間発展（JIT最適化）"""
    n = len(V)
    dtheta_dt = np.empty(n)
    
    for i in range(n):
        V_safe = max(V[i], 1.0e-20)
        theta_safe = max(theta[i], 1.0e-20)
        
        x = V_safe * theta_safe / L[i]
        
        # Composite law の発展則
        if x < 700:  # exp()のオーバーフロー防止
            term1 = np.exp(-x)
        else:
            term1 = 0.0
        
        if V_c / V_safe < 700:
            term2 = x * np.exp(-V_c / V_safe)
        else:
            term2 = 0.0
        
        dtheta_dt[i] = term1 - term2
    
    return dtheta_dt


@njit(cache=True, fastmath=True)
def _compute_friction_derivatives(V: np.ndarray,
                                   theta: np.ndarray,
                                   a: np.ndarray,
                                   b: np.ndarray,
                                   L: np.ndarray,
                                   sigma_eff: np.ndarray,
                                   mu_0: float,
                                   V_0: float,
                                   V_c: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """摩擦則の偏微分（JIT最適化）"""
    n = len(V)
    dmu_dV = np.empty(n)
    dmu_dtheta = np.empty(n)
    dthetadot_dV = np.empty(n)
    
    for i in range(n):
        V_safe = max(V[i], 1.0e-20)
        theta_safe = max(theta[i], 1.0e-20)
        
        # ∂μ/∂V = a / (V + V_0)
        dmu_dV[i] = a[i] / (V_safe + V_0)
        
        # ∂μ/∂Θ = b * V_0 / (V_0 * Θ + L)
        dmu_dtheta[i] = b[i] * V_0 / (V_0 * theta_safe + L[i])
        
        # ∂(dΘ/dt)/∂V の計算
        x = V_safe * theta_safe / L[i]
        
        if x < 700:
            exp_neg_x = np.exp(-x)
        else:
            exp_neg_x = 0.0
        
        if V_c / V_safe < 700:
            exp_neg_Vc_V = np.exp(-V_c / V_safe)
        else:
            exp_neg_Vc_V = 0.0
        
        # d/dV of exp(-VΘ/L) = -Θ/L * exp(-VΘ/L)
        dterm1_dV = -theta_safe / L[i] * exp_neg_x
        
        # d/dV of (VΘ/L) * exp(-V_c/V) = Θ/L * exp(-V_c/V) + VΘ/L * V_c/V² * exp(-V_c/V)
        dterm2_dV = (theta_safe / L[i] * exp_neg_Vc_V + 
                     x * V_c / (V_safe * V_safe) * exp_neg_Vc_V)
        
        dthetadot_dV[i] = dterm1_dV - dterm2_dV
    
    return dmu_dV, dmu_dtheta, dthetadot_dV


@njit(parallel=True, cache=True, fastmath=True)
def compute_friction_stress_parallel(V: np.ndarray,
                                      theta: np.ndarray,
                                      a: np.ndarray,
                                      b: np.ndarray,
                                      L: np.ndarray,
                                      sigma_eff: np.ndarray,
                                      mu_0: float,
                                      V_0: float) -> np.ndarray:
    """摩擦応力の並列計算"""
    n = len(V)
    tau_f = np.empty(n)
    
    for i in prange(n):
        V_safe = max(V[i], 1.0e-20)
        theta_safe = max(theta[i], 1.0e-20)
        
        term1 = a[i] * np.log(V_safe / V_0 + 1.0)
        term2 = b[i] * np.log(V_0 * theta_safe / L[i] + 1.0)
        
        mu = mu_0 + term1 + term2
        tau_f[i] = sigma_eff[i] * mu
    
    return tau_f


@njit(parallel=True, cache=True, fastmath=True)
def compute_state_evolution_parallel(V: np.ndarray,
                                      theta: np.ndarray,
                                      L: np.ndarray,
                                      V_c: float) -> np.ndarray:
    """状態変数発展の並列計算"""
    n = len(V)
    dtheta_dt = np.empty(n)
    
    for i in prange(n):
        V_safe = max(V[i], 1.0e-20)
        theta_safe = max(theta[i], 1.0e-20)
        
        x = V_safe * theta_safe / L[i]
        
        if x < 700:
            term1 = np.exp(-x)
        else:
            term1 = 0.0
        
        if V_c / V_safe < 700:
            term2 = x * np.exp(-V_c / V_safe)
        else:
            term2 = 0.0
        
        dtheta_dt[i] = term1 - term2
    
    return dtheta_dt


def get_steady_state_theta(V: np.ndarray, L: np.ndarray, V_c: float = 1.0e-8) -> np.ndarray:
    """
    定常状態の状態変数を計算
    
    dΘ/dt = 0 を解く
    """
    # 定常状態では exp(-VΘ/L) = VΘ/L * exp(-V_c/V)
    # 近似的に: Θ ≈ L/V (低速限界)
    return L / np.maximum(V, 1.0e-20)
