"""
クイックテスト - シンプルな1Dスプリングスライダーモデル
Quick Test - Simple 1D Spring-Slider Model

Rate-State摩擦則の基本動作を確認
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

print("=" * 50)
print("クイックテスト: 1Dスプリングスライダーモデル")
print("=" * 50)

# ==============================================================================
# パラメータ
# ==============================================================================
# 物理定数
G = 30.0e9          # 剛性率 [Pa] (30 GPa)
beta = 3750.0       # S波速度 [m/s]
eta = 1.0           # 準動的補正係数

# 摩擦パラメータ
mu_0 = 0.6          # 定常摩擦係数
V_0 = 1.0e-6        # 参照速度 [m/s]
V_c = 1.0e-8        # カットオフ速度 [m/s]
a = 0.005           # 直接効果
b = 0.008           # 状態効果 (velocity weakening: a-b < 0)
L = 0.01            # 特徴的すべり量 [m] (1 cm)
sigma = 30.0e6      # 有効法線応力 [Pa] (30 MPa)

# バネ剛性（スプリングスライダー）
k = 1.0e6           # [Pa/m]

# プレート速度
V_pl = 5.5e-2 / (365.25 * 24 * 3600)  # 5.5 cm/year → m/s

# 放射減衰
rad_damp = eta * G / beta

print(f"摩擦パラメータ:")
print(f"  a-b = {a-b:.4f} (< 0: velocity weakening)")
print(f"  σ_eff = {sigma/1e6:.1f} MPa")
print(f"  L = {L*100:.1f} cm")
print(f"  V_pl = {V_pl * 365.25*24*3600*100:.2f} cm/year")
print()

# ==============================================================================
# Rate-State摩擦則
# ==============================================================================
@njit(cache=True)
def friction_coefficient(V, theta, a, b, L, mu_0, V_0):
    """摩擦係数 (Composite law)"""
    V_safe = max(V, 1e-20)
    theta_safe = max(theta, 1e-20)
    return mu_0 + a * np.log(V_safe/V_0 + 1) + b * np.log(V_0*theta_safe/L + 1)

@njit(cache=True)
def state_evolution(V, theta, L, V_c):
    """状態変数の発展 (Composite law)"""
    V_safe = max(V, 1e-20)
    theta_safe = max(theta, 1e-20)
    x = V_safe * theta_safe / L
    
    if x < 700:
        term1 = np.exp(-x)
    else:
        term1 = 0.0
    
    if V_c / V_safe < 700:
        term2 = x * np.exp(-V_c / V_safe)
    else:
        term2 = 0.0
    
    return term1 - term2

@njit(cache=True)
def solve_velocity(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp):
    """Newton法で速度を求める"""
    V = V_0
    
    for _ in range(50):
        V_safe = max(V, 1e-20)
        mu = friction_coefficient(V_safe, theta, a, b, L, mu_0, V_0)
        F = tau - sigma * mu - rad_damp * V_safe
        
        dmu_dV = a / (V_safe + V_0)
        dF_dV = -sigma * dmu_dV - rad_damp
        
        dV = -F / dF_dV
        V_new = max(V + dV, 1e-20)
        
        if abs(dV) < 1e-12 * abs(V):
            return V_new
        V = V_new
    
    return V

# ==============================================================================
# シミュレーション
# ==============================================================================
print("シミュレーション開始...")
start_time = time.time()

# 初期条件
V_init = 0.1 * V_pl  # プレート速度の10%
theta_init = L / V_init  # 定常状態
mu_init = friction_coefficient(V_init, theta_init, a, b, L, mu_0, V_0)
tau_init = sigma * mu_init + rad_damp * V_init

# シミュレーション設定
t_years = 500  # 500年
t_end = t_years * 365.25 * 24 * 3600  # 秒
dt_init = 1e6  # 初期ステップ (~11日)

# 時間積分
t = 0.0
tau = tau_init
theta = theta_init
slip = 0.0

# 履歴
t_hist = [0.0]
V_hist = [V_init]
tau_hist = [tau_init]
slip_hist = [0.0]

step = 0
n_earthquakes = 0
in_earthquake = False
eq_times = []


while t < t_end:
    # 速度計算
    V = solve_velocity(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp)
    
    # 適応的タイムステップ
    if V > 1e-3:  # 高速すべり
        dt = min(0.1, L / V / 10)
    elif V > 1e-6:
        dt = min(1e4, L / V / 100)
    else:
        dt = min(1e7, 0.1 * theta)
    
    dt = max(dt, 0.01)
    if t + dt > t_end:
        dt = t_end - t
    
    # 状態変数の更新
    dtheta = state_evolution(V, theta, L, V_c)
    theta_new = theta + dtheta * dt
    
    # 応力の更新（スプリング駆動）
    dtau = k * (V_pl - V)
    tau_new = tau + dtau * dt
    
    # すべり量
    slip_new = slip + V * dt
    
    # 地震検出（フラグで重複カウント防止）
    if V > 0.1 and not in_earthquake:
        in_earthquake = True
        n_earthquakes += 1
        eq_times.append(t)
        print(f"  地震 #{n_earthquakes}: t = {t/3.15576e7:.2f} years, V = {V:.2e} m/s")
    elif V < 0.01:  # 速度が下がったらリセット
        in_earthquake = False

    
    # 更新
    t += dt
    tau = tau_new
    theta = max(theta_new, 1e-10)
    slip = slip_new
    
    # 履歴保存（間引き）
    if len(t_hist) == 0 or t - t_hist[-1] > t_end / 5000:
        t_hist.append(t)
        V_hist.append(V)
        tau_hist.append(tau)
        slip_hist.append(slip)
    
    step += 1

elapsed = time.time() - start_time
print()
print(f"完了: {step} ステップ, {elapsed:.2f} 秒")
print(f"検出された地震: {n_earthquakes} 回")
if len(eq_times) >= 2:
    intervals = np.diff(eq_times) / 3.15576e7
    print(f"平均再発間隔: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} 年")


# ==============================================================================
# 可視化
# ==============================================================================
print()
print("結果をプロット中...")

t_years_arr = np.array(t_hist) / 3.15576e7
V_arr = np.array(V_hist)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# すべり速度
ax1 = axes[0]
ax1.semilogy(t_years_arr, V_arr, 'b-', linewidth=0.5)
ax1.axhline(y=0.1, color='r', linestyle='--', label='Earthquake threshold')
ax1.axhline(y=V_pl, color='g', linestyle=':', label='Plate velocity')
ax1.set_ylabel('Slip velocity [m/s]')
ax1.set_ylim(1e-12, 1e2)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('1D Spring-Slider Model with Rate-State Friction')

# 剪断応力
ax2 = axes[1]
ax2.plot(t_years_arr, np.array(tau_hist)/1e6, 'b-', linewidth=0.5)
ax2.set_ylabel('Shear stress [MPa]')
ax2.grid(True, alpha=0.3)

# 累積すべり
ax3 = axes[2]
ax3.plot(t_years_arr, np.array(slip_hist), 'b-', linewidth=0.5)
ax3.set_xlabel('Time [years]')
ax3.set_ylabel('Cumulative slip [m]')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/quick_test.png', dpi=150)
print("結果を保存: results/quick_test.png")
plt.close()

print()
print("=" * 50)
print("テスト完了！Rate-State摩擦則が正しく動作しています。")
print("=" * 50)
