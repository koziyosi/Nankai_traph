"""
Runge-Kutta 5次適応ステップソルバー
Press et al. (1992) Numerical Recipes に準拠
"""

import numpy as np
from numba import njit
from typing import Tuple, Callable, Optional, List
from dataclasses import dataclass, field
import time


@dataclass
class RK45Solver:
    """
    5次 Runge-Kutta法 (Dormand-Prince) with 適応ステップサイズ制御
    
    特徴:
    - 5次精度の解と4次精度の誤差推定を組み合わせ
    - 適応的なステップサイズ制御
    - 地震時は自動的に小さなステップに
    """
    
    # トレランス
    rtol: float = 1.0e-6
    atol: float = 1.0e-9
    
    # ステップサイズ制限
    dt_min: float = 0.01        # [s]
    dt_max: float = 1.0e8       # [s] (~3年)
    dt_initial: float = 1.0e6   # [s] (~11日)
    
    # 安全係数
    safety: float = 0.9
    p_grow: float = -0.2        # dt増加時の指数
    p_shrink: float = -0.25     # dt減少時の指数
    max_scale: float = 5.0      # 1ステップでの最大増加率
    min_scale: float = 0.1      # 1ステップでの最大減少率
    
    # 統計
    n_steps: int = 0
    n_accept: int = 0
    n_reject: int = 0
    
    def solve(self,
              f: Callable,
              y0: np.ndarray,
              t_span: Tuple[float, float],
              t_eval: np.ndarray = None,
              callback: Callable = None,
              max_steps: int = 10_000_000,
              verbose: bool = True) -> dict:
        """
        常微分方程式を解く
        """
        t0, t_end = t_span
        t = t0
        y = y0.copy()
        dt = self.dt_initial
        
        # 結果格納
        t_history = [t]
        y_history = [y.copy()]
        
        self.n_steps = 0
        self.n_accept = 0
        self.n_reject = 0
        
        start_time = time.time()
        last_print = start_time
        
        # t_eval用のインデックス
        eval_idx = 0 if t_eval is not None else None
        
        try:
            while t < t_end and self.n_steps < max_steps:
                # ステップサイズを終端に合わせる
                if t + dt > t_end:
                    dt = t_end - t
                
                # RK45ステップ
                y_new, error, k = self._rk45_step(f, t, y, dt)
                
                # 誤差評価
                scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(y_new))
                err_ratio = np.max(np.abs(error) / scale)
                
                if err_ratio <= 1.0:
                    # ステップ受理
                    self.n_accept += 1
                    t_new = t + dt
                    
                    # 補間して t_eval の値を記録
                    if t_eval is not None:
                        while eval_idx < len(t_eval) and t_eval[eval_idx] <= t_new:
                            if t_eval[eval_idx] >= t:
                                # Hermite補間
                                theta = (t_eval[eval_idx] - t) / dt
                                y_interp = self._hermite_interpolate(y, y_new, k[0], k[-1], dt, theta)
                                t_history.append(t_eval[eval_idx])
                                y_history.append(y_interp)
                            eval_idx += 1
                    else:
                        t_history.append(t_new)
                        y_history.append(y_new.copy())
                    
                    t = t_new
                    y = y_new
                    
                    # コールバック
                    if callback is not None:
                        callback(t, y, dt)
                    
                    # dt増加
                    if err_ratio > 0:
                        scale_factor = self.safety * err_ratio ** self.p_grow
                        scale_factor = min(scale_factor, self.max_scale)
                        dt = dt * scale_factor
                    else:
                        dt = dt * self.max_scale
                else:
                    # ステップ棄却
                    self.n_reject += 1
                    scale_factor = max(self.safety * err_ratio ** self.p_shrink, self.min_scale)
                    dt = dt * scale_factor
                
                # ステップサイズ制限
                dt = max(self.dt_min, min(dt, self.dt_max))
                
                self.n_steps += 1
                
                # 進捗表示
                if verbose and time.time() - last_print > 5.0:
                    progress = (t - t0) / (t_end - t0) * 100
                    elapsed = time.time() - start_time
                    print(f"  t = {t/3.15576e7:.2f} years, "
                          f"dt = {dt:.2e} s, "
                          f"progress = {progress:.1f}%, "
                          f"steps = {self.n_steps}, "
                          f"elapsed = {elapsed:.1f} s")
                    last_print = time.time()

        except KeyboardInterrupt:
            print("\nシミュレーションが中断されました (KeyboardInterrupt)")
        except Exception as e:
            print(f"\nシミュレーションエラー: {e}")
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"シミュレーション完了: {self.n_steps} ステップ, "
                  f"accept = {self.n_accept}, reject = {self.n_reject}, "
                  f"time = {elapsed:.1f} s")
        
        return {
            't': np.array(t_history),
            'y': np.array(y_history),
            'n_steps': self.n_steps,
            'n_accept': self.n_accept,
            'n_reject': self.n_reject,
            'elapsed': elapsed
        }
    
    def _rk45_step(self, f: Callable, t: float, y: np.ndarray, dt: float
                   ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Dormand-Prince RK5(4)7M ステップ
        """
        # Butcher係数
        c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
        
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
        
        # 5次の係数（解）
        b1, b2, b3, b4, b5, b6 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
        
        # 4次の係数（誤差推定用）
        b1s, b2s, b3s, b4s, b5s, b6s, b7s = (
            5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
        )
        
        # k値の計算
        k1 = f(t, y)
        k2 = f(t + c2*dt, y + dt*(a21*k1))
        k3 = f(t + c3*dt, y + dt*(a31*k1 + a32*k2))
        k4 = f(t + c4*dt, y + dt*(a41*k1 + a42*k2 + a43*k3))
        k5 = f(t + c5*dt, y + dt*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        k6 = f(t + c6*dt, y + dt*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        
        # 5次の解
        y_new = y + dt * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        
        k7 = f(t + dt, y_new)
        
        # 誤差推定（5次と4次の差）
        error = dt * ((b1-b1s)*k1 + (b3-b3s)*k3 + (b4-b4s)*k4 + 
                      (b5-b5s)*k5 + (b6-b6s)*k6 - b7s*k7)
        
        return y_new, error, [k1, k2, k3, k4, k5, k6, k7]
    
    def _hermite_interpolate(self, y0, y1, f0, f1, dt, theta):
        """Hermite補間"""
        return (1 - theta) * y0 + theta * y1 + \
               theta * (theta - 1) * ((1 - 2*theta) * (y1 - y0) + 
                                       (theta - 1) * dt * f0 + 
                                       theta * dt * f1)


# ==============================================================================
# Numba最適化された RK45 ステップ（コア計算用）
# ==============================================================================
# (Note: rk45_step_numba removed to keep file simple and as it was unused in main class)

def create_fast_solver(rtol: float = 1.0e-6,
                       atol: float = 1.0e-9) -> RK45Solver:
    """高速設定のソルバーを作成"""
    return RK45Solver(
        rtol=rtol,
        atol=atol,
        dt_min=0.1,
        dt_max=1.0e9,
        dt_initial=1.0e7
    )


def create_accurate_solver() -> RK45Solver:
    """高精度設定のソルバーを作成"""
    return RK45Solver(
        rtol=1.0e-8,
        atol=1.0e-12,
        dt_min=0.001,
        dt_max=1.0e7,
        dt_initial=1.0e5
    )
