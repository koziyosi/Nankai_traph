"""
南海トラフ地震シミュレーション - 2Dバージョン
Nankai Trough Earthquake Simulation - 2D Version

各地震の破壊領域を可視化する機能付き
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 日本語表示用
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

# 出力ディレクトリ
os.makedirs('results/earthquakes', exist_ok=True)

print("=" * 60)
print("南海トラフ地震シミュレーション（2D版）")
print("Nankai Trough Earthquake Simulation")
print("=" * 60)
print()

# ==============================================================================
# 領域定義（セグメント）
# ==============================================================================
SEGMENTS = {
    'tokai': {
        'name': '東海',
        'lon_range': (137.5, 139.5),
        'lat_range': (34.0, 35.5),
        'color': 'red',
        'x_range': (0.75, 1.0)  # 正規化X座標
    },
    'tonankai': {
        'name': '東南海',
        'lon_range': (135.5, 137.5),
        'lat_range': (33.5, 35.0),
        'color': 'orange',
        'x_range': (0.5, 0.75)
    },
    'nankai': {
        'name': '南海',
        'lon_range': (132.5, 135.5),
        'lat_range': (32.5, 34.5),
        'color': 'yellow',
        'x_range': (0.2, 0.5)
    },
    'hyuganada': {
        'name': '日向灘',
        'lon_range': (131.0, 132.5),
        'lat_range': (31.0, 33.0),
        'color': 'cyan',
        'x_range': (0.0, 0.2)
    }
}

# ==============================================================================
# パラメータ
# ==============================================================================
# 物理定数
G = 30.0e9          # 剛性率 [Pa]
beta = 3750.0       # S波速度 [m/s]
eta = 1.0           # 準動的補正係数
rad_damp = eta * G / beta

# 摩擦パラメータ
mu_0 = 0.6
V_0 = 1.0e-6
V_c = 1.0e-8
a = 0.005

# プレート速度
V_pl_base = 5.5e-2 / (365.25 * 24 * 3600)  # 5.5 cm/year → m/s

# グリッドサイズ
N_X = 40  # 走向方向
N_Y = 8   # ディップ方向
N_CELLS = N_X * N_Y

print(f"グリッドサイズ: {N_X} x {N_Y} = {N_CELLS} セル")

# ==============================================================================
# セルごとのパラメータ設定
# ==============================================================================
def setup_parameters():
    """空間的に不均質なパラメータを設定"""
    # 配列初期化
    b = np.full(N_CELLS, 0.008)  # velocity weakening
    sigma = np.full(N_CELLS, 30.0e6)  # 30 MPa
    L = np.full(N_CELLS, 0.4)  # 0.4 m
    V_pl = np.full(N_CELLS, V_pl_base)
    
    # セル座標（正規化: 0-1）
    cell_x = np.zeros(N_CELLS)  # 走向方向（西→東）
    cell_y = np.zeros(N_CELLS)  # ディップ方向（浅→深）
    
    for i in range(N_CELLS):
        ix = i % N_X
        iy = i // N_X
        cell_x[i] = (ix + 0.5) / N_X
        cell_y[i] = (iy + 0.5) / N_Y
    
    # 領域ごとの設定
    for i in range(N_CELLS):
        x = cell_x[i]
        depth_factor = cell_y[i]  # 0=浅い, 1=深い
        
        # ディップ方向の設定
        if depth_factor > 0.75:  # 深部：安定すべり域
            b[i] = 0.002  # velocity strengthening
            L[i] = 0.1    # LSSE用の小さいL
        
        # 東海（東端）
        if x > 0.8:
            sigma[i] = 40.0e6
            L[i] = 1.0
            V_pl[i] = V_pl_base * 0.3  # 遅い収束
        
        # 東南海
        elif x > 0.55:
            sigma[i] = 30.0e6
            L[i] = 0.3  # 小さいL = 短い再発間隔
        
        # 潮岬バリア（東南海・南海の境界）
        elif 0.45 < x < 0.55:
            L[i] = 7.5  # 大きなL = バリア
        
        # 南海
        elif x > 0.2:
            sigma[i] = 30.0e6
            L[i] = 0.4
        
        # 日向灘（西端）
        else:
            sigma[i] = 60.0e6  # 高い法線応力 = 長い再発間隔
            L[i] = 2.0
            V_pl[i] = V_pl_base * 1.2  # やや速い収束
    
    return b, sigma, L, V_pl, cell_x, cell_y


# ==============================================================================
# 簡易的な相互作用カーネル
# ==============================================================================
def create_interaction_kernel():
    """セル間の応力相互作用カーネルを作成"""
    K = np.zeros((N_CELLS, N_CELLS))
    
    for i in range(N_CELLS):
        ix_i = i % N_X
        iy_i = i // N_X
        
        for j in range(N_CELLS):
            ix_j = j % N_X
            iy_j = j // N_X
            
            if i == j:
                # 自己スティフネス
                K[i, j] = -1.0e6
            else:
                # 距離に応じた相互作用
                dx = abs(ix_i - ix_j)
                dy = abs(iy_i - iy_j)
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 5:
                    K[i, j] = 0.3e6 / (dist + 0.5)**2
    
    return K


# ==============================================================================
# Rate-State摩擦則
# ==============================================================================
@njit(cache=True)
def friction_coefficient(V, theta, a, b, L, mu_0, V_0):
    n = len(V)
    mu = np.empty(n)
    for i in range(n):
        V_safe = max(V[i], 1e-20)
        theta_safe = max(theta[i], 1e-20)
        mu[i] = mu_0 + a * np.log(V_safe/V_0 + 1) + b[i] * np.log(V_0*theta_safe/L[i] + 1)
    return mu

@njit(cache=True)
def state_evolution(V, theta, L, V_c):
    n = len(V)
    dtheta = np.empty(n)
    for i in range(n):
        V_safe = max(V[i], 1e-20)
        theta_safe = max(theta[i], 1e-20)
        x = V_safe * theta_safe / L[i]
        if x < 700:
            term1 = np.exp(-x)
        else:
            term1 = 0.0
        if V_c / V_safe < 700:
            term2 = x * np.exp(-V_c / V_safe)
        else:
            term2 = 0.0
        dtheta[i] = term1 - term2
    return dtheta

@njit(cache=True)
def solve_velocity(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp):
    n = len(tau)
    V = np.empty(n)
    
    for i in range(n):
        V_guess = V_0
        theta_i = max(theta[i], 1e-20)
        
        for _ in range(50):
            V_safe = max(V_guess, 1e-20)
            mu = mu_0 + a * np.log(V_safe/V_0 + 1) + b[i] * np.log(V_0*theta_i/L[i] + 1)
            F = tau[i] - sigma[i] * mu - rad_damp * V_safe
            dmu_dV = a / (V_safe + V_0)
            dF_dV = -sigma[i] * dmu_dV - rad_damp
            dV = -F / dF_dV
            V_new = max(V_guess + dV, 1e-20)
            if abs(dV) < 1e-12 * abs(V_guess):
                V_guess = V_new
                break
            V_guess = V_new
        
        V[i] = max(V_guess, 1e-20)
    
    return V


# ==============================================================================
# 地震イベントクラス
# ==============================================================================
@dataclass
class Earthquake:
    id: int
    t_start: float
    t_end: float
    cells: List[int]
    max_slip: float
    Mw: float
    slip_distribution: np.ndarray
    regions: List[str]


# ==============================================================================
# 可視化
# ==============================================================================
def plot_rupture_area(eq: Earthquake, cell_x: np.ndarray, cell_y: np.ndarray, 
                      save_path: str):
    """地震の破壊領域を可視化"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # すべり分布をグリッドに変換
    slip_grid = eq.slip_distribution.reshape(N_Y, N_X)
    
    # プロット
    x = np.linspace(131, 140, N_X + 1)
    y = np.linspace(0, 40, N_Y + 1)  # 深度
    
    # すべり分布
    cmap = LinearSegmentedColormap.from_list('slip', ['white', 'yellow', 'orange', 'red', 'darkred'])
    im = ax.pcolormesh(x, y, slip_grid, cmap=cmap, shading='flat', 
                       vmin=0, vmax=max(10, eq.max_slip))
    
    # セグメント境界
    for seg_name, seg in SEGMENTS.items():
        x_seg = seg['lon_range'][0]
        ax.axvline(x=x_seg, color='gray', linestyle='--', alpha=0.5)
        # ラベル
        x_mid = (seg['lon_range'][0] + seg['lon_range'][1]) / 2
        ax.text(x_mid, -2, seg['name'], ha='center', fontsize=10, fontweight='bold')
    
    # 軸設定
    ax.set_xlim(131, 140)
    ax.set_ylim(40, 0)  # 深度は下向き
    ax.set_xlabel('経度 [°E]', fontsize=12)
    ax.set_ylabel('深度 [km]', fontsize=12)
    
    # タイトル
    t_years = eq.t_start / (365.25 * 24 * 3600)
    regions_str = ', '.join(eq.regions) if eq.regions else '(不明)'
    ax.set_title(f'地震 #{eq.id}: t = {t_years:.1f} 年, Mw = {eq.Mw:.1f}\n'
                 f'破壊領域: {regions_str}', fontsize=14)
    
    # カラーバー
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('すべり量 [m]', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_timeline(earthquakes: List[Earthquake], save_path: str):
    """地震年表を作成"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 領域ごとに縦軸を分ける
    region_y = {'hyuganada': 0, 'nankai': 1, 'tonankai': 2, 'tokai': 3}
    region_labels = {'hyuganada': '日向灘', 'nankai': '南海', 
                     'tonankai': '東南海', 'tokai': '東海'}
    
    for eq in earthquakes:
        t_years = eq.t_start / (365.25 * 24 * 3600)
        
        # 破壊した領域に線を描く
        for region in eq.regions:
            if region in region_y:
                y = region_y[region]
                color = SEGMENTS[region]['color']
                ax.barh(y, width=2, left=t_years, height=0.6, 
                        color=color, edgecolor='black', alpha=0.8)
        
        # Mwに応じたマーカー
        if eq.Mw >= 8.5:
            ax.plot(t_years, 1.5, 'r*', markersize=15)
    
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels([region_labels[r] for r in ['hyuganada', 'nankai', 'tonankai', 'tokai']])
    ax.set_xlabel('時間 [年]', fontsize=12)
    ax.set_title('地震年表 - 領域別発生履歴', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def classify_regions(slip: np.ndarray, cell_x: np.ndarray, threshold: float = 1.0) -> List[str]:
    """すべり分布から破壊領域を特定"""
    regions = []
    
    for seg_name, seg in SEGMENTS.items():
        x_min, x_max = seg['x_range']
        mask = (cell_x >= x_min) & (cell_x < x_max) & (slip > threshold)
        if np.any(mask):
            regions.append(seg_name)
    
    return regions


# ==============================================================================
# シミュレーション
# ==============================================================================
def run_simulation(t_years: float = 1000):
    """2Dシミュレーションを実行"""
    print(f"\nシミュレーション期間: {t_years} 年")
    print("パラメータを設定中...")
    
    b, sigma, L, V_pl, cell_x, cell_y = setup_parameters()
    
    print("相互作用カーネルを計算中...")
    K = create_interaction_kernel()
    
    # 初期条件
    V_init = np.full(N_CELLS, 0.1 * V_pl_base)
    theta = L / V_init
    mu_init = friction_coefficient(V_init, theta, a, b, L, mu_0, V_0)
    tau = sigma * mu_init + rad_damp * V_init
    slip = np.zeros(N_CELLS)
    
    # シミュレーション設定
    t = 0.0
    t_end = t_years * 365.25 * 24 * 3600
    
    # イベント追跡
    earthquakes = []
    n_earthquakes = 0
    in_earthquake = False
    eq_start_time = 0.0
    eq_cells = []
    eq_slip_start = None
    
    # 履歴
    t_hist = [0.0]
    max_V_hist = [np.max(V_init)]
    
    step = 0
    start_time = time.time()
    last_print = start_time
    
    print("\nシミュレーション開始...")
    
    while t < t_end:
        # 速度計算
        V = solve_velocity(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp)
        
        # 適応タイムステップ
        V_max = np.max(V)
        if V_max > 1e-3:
            dt = min(0.1, np.min(L) / V_max / 10)
        elif V_max > 1e-6:
            dt = min(1e4, np.min(L) / V_max / 100)
        else:
            dt = min(1e7, 0.1 * np.min(theta))
        
        dt = max(dt, 0.01)
        if t + dt > t_end:
            dt = t_end - t
        
        # 状態更新
        dtheta = state_evolution(V, theta, L, V_c)
        theta = np.maximum(theta + dtheta * dt, 1e-10)
        
        # 応力更新（相互作用含む）
        dtau = K @ (V_pl - V)
        tau = tau + dtau * dt
        
        # すべり更新
        slip = slip + V * dt
        
        # 地震検出
        eq_cells_now = np.where(V > 0.1)[0]
        
        if len(eq_cells_now) > 0 and not in_earthquake:
            # 地震開始
            in_earthquake = True
            eq_start_time = t
            eq_cells = list(eq_cells_now)
            eq_slip_start = slip.copy()
        
        elif len(eq_cells_now) > 0 and in_earthquake:
            # 地震継続
            eq_cells = list(set(eq_cells) | set(eq_cells_now))
        
        elif len(eq_cells_now) == 0 and in_earthquake:
            # 地震終了
            in_earthquake = False
            n_earthquakes += 1
            
            # コサイスミックすべり
            coseismic_slip = slip - eq_slip_start
            max_slip = np.max(coseismic_slip)
            
            # モーメント計算
            cell_area = 50e3 * 50e3 / (N_X * N_Y)  # 概算面積
            M0 = G * np.sum(np.abs(coseismic_slip)) * cell_area
            Mw = (np.log10(M0) - 9.1) / 1.5 if M0 > 0 else 0
            
            # 領域分類
            regions = classify_regions(coseismic_slip, cell_x, threshold=0.5)
            
            # イベント作成
            eq = Earthquake(
                id=n_earthquakes,
                t_start=eq_start_time,
                t_end=t,
                cells=eq_cells,
                max_slip=max_slip,
                Mw=Mw,
                slip_distribution=coseismic_slip.copy(),
                regions=regions
            )
            earthquakes.append(eq)
            
            # 表示
            t_years_now = eq_start_time / (365.25 * 24 * 3600)
            regions_str = ', '.join([SEGMENTS[r]['name'] for r in regions]) if regions else '(不明)'
            print(f"  地震 #{n_earthquakes}: t = {t_years_now:.1f} 年, "
                  f"Mw = {Mw:.1f}, 領域 = {regions_str}")
            
            # 破壊領域を可視化
            save_path = f'results/earthquakes/eq_{n_earthquakes:03d}.png'
            plot_rupture_area(eq, cell_x, cell_y, save_path)
        
        # 更新
        t += dt
        step += 1
        
        # 履歴（間引き）
        if len(t_hist) == 0 or t - t_hist[-1] > t_end / 2000:
            t_hist.append(t)
            max_V_hist.append(V_max)
        
        # 進捗表示
        if time.time() - last_print > 5.0:
            progress = t / t_end * 100
            print(f"  進捗: {progress:.1f}%, t = {t/(365.25*24*3600):.1f} 年, "
                  f"地震数 = {n_earthquakes}")
            last_print = time.time()
    
    elapsed = time.time() - start_time
    
    print()
    print(f"完了: {step} ステップ, {elapsed:.1f} 秒")
    print(f"検出された地震: {n_earthquakes} 個")
    
    # 統計
    if len(earthquakes) >= 2:
        intervals = []
        for i in range(1, len(earthquakes)):
            dt = earthquakes[i].t_start - earthquakes[i-1].t_start
            intervals.append(dt / (365.25 * 24 * 3600))
        print(f"平均再発間隔: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} 年")
    
    # 年表作成
    if earthquakes:
        plot_timeline(earthquakes, 'results/timeline.png')
        print("\n年表を保存: results/timeline.png")
    
    # 速度時系列
    plt.figure(figsize=(14, 5))
    plt.semilogy(np.array(t_hist) / (365.25 * 24 * 3600), max_V_hist, 'b-', linewidth=0.5)
    plt.axhline(y=0.1, color='r', linestyle='--', label='地震閾値')
    plt.xlabel('時間 [年]')
    plt.ylabel('最大すべり速度 [m/s]')
    plt.title('すべり速度の時間発展')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/velocity_history.png', dpi=150)
    plt.close()
    print("速度履歴を保存: results/velocity_history.png")
    
    return earthquakes


# ==============================================================================
# メイン
# ==============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=float, default=1000, 
                        help='シミュレーション年数 (default: 1000)')
    args = parser.parse_args()
    
    earthquakes = run_simulation(t_years=args.years)
    
    print()
    print("=" * 60)
    print("完了！")
    print(f"  地震イベント画像: results/earthquakes/")
    print(f"  年表: results/timeline.png")
    print(f"  速度履歴: results/velocity_history.png")
    print("=" * 60)
