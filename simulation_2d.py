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

# グリッドサイズ（コマンドラインで設定可能）
N_X = 40  # 走向方向（デフォルト）
N_Y = 8   # ディップ方向（デフォルト）
N_CELLS = N_X * N_Y

def set_grid_size(nx: int, ny: int):
    """グリッドサイズを設定"""
    global N_X, N_Y, N_CELLS
    N_X = nx
    N_Y = ny
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


def setup_parameters_from_polygon(polygon_json_path: str):
    """
    polygon_data.json を使用して空間的に不均質なパラメータを設定
    
    Args:
        polygon_json_path: polygon_data.json のパス
    
    Returns:
        b, sigma, L, V_pl, cell_x, cell_y (各N_CELLS配列)
    """
    import json
    from matplotlib.path import Path as MplPath
    
    # デフォルトパラメータで初期化
    b = np.full(N_CELLS, 0.008)
    sigma = np.full(N_CELLS, 30.0e6)
    L = np.full(N_CELLS, 0.4)
    V_pl = np.full(N_CELLS, V_pl_base)
    
    # セル座標（実際の経度・緯度に変換）
    cell_x = np.zeros(N_CELLS)
    cell_y = np.zeros(N_CELLS)
    cell_lon = np.zeros(N_CELLS)
    cell_lat = np.zeros(N_CELLS)
    
    # 経度・緯度範囲（南海トラフ全域）
    lon_min, lon_max = 131.0, 139.0
    lat_min, lat_max = 30.5, 35.5
    
    for i in range(N_CELLS):
        ix = i % N_X
        iy = i // N_X
        cell_x[i] = (ix + 0.5) / N_X
        cell_y[i] = (iy + 0.5) / N_Y
        cell_lon[i] = lon_min + cell_x[i] * (lon_max - lon_min)
        cell_lat[i] = lat_min + cell_y[i] * (lat_max - lat_min)
    
    # polygon_data.json を読み込み
    try:
        with open(polygon_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        locked_zone = [(p['lon'], p['lat']) for p in data.get('locked_zone', [])]
        unlocked_zone = [(p['lon'], p['lat']) for p in data.get('unlocked_zone', [])]
        
        print(f"polygon_data.json を読み込みました: {polygon_json_path}")
        print(f"  固着域: {len(locked_zone)} 点")
        print(f"  非固着域: {len(unlocked_zone)} 点")
        
        # ポリゴンパスを作成
        locked_coords = np.array(locked_zone)
        unlocked_coords = np.array(unlocked_zone)
        
        locked_path = MplPath(locked_coords) if len(locked_coords) > 2 else None
        unlocked_path = MplPath(unlocked_coords) if len(unlocked_coords) > 2 else None
        
        # 各セルがどのゾーンに属するかを判定
        for i in range(N_CELLS):
            point = np.array([[cell_lon[i], cell_lat[i]]])
            in_locked = locked_path.contains_points(point)[0] if locked_path else False
            in_unlocked = unlocked_path.contains_points(point)[0] if unlocked_path else False
            
            depth_factor = cell_y[i]  # 0=浅い, 1=深い
            
            if in_locked:
                # 固着域：地震発生しやすいパラメータ
                b[i] = 0.008  # velocity weakening
                sigma[i] = 30.0e6 + depth_factor * 10.0e6
                L[i] = 0.3 + depth_factor * 0.2
                V_pl[i] = V_pl_base * (1.0 - 0.3 * cell_x[i])
            elif in_unlocked:
                # 非固着域：安定すべり（LSSE発生可能）
                b[i] = 0.003  # velocity strengthening寄り
                sigma[i] = 20.0e6
                L[i] = 0.08
                V_pl[i] = V_pl_base * (1.0 - 0.3 * cell_x[i])
            else:
                # どちらにも属さない：デフォルト値を維持
                # 深さに応じた調整
                if depth_factor > 0.75:
                    b[i] = 0.002
                    L[i] = 0.1
        
        locked_count = sum(1 for i in range(N_CELLS) 
                          if locked_path and locked_path.contains_points([[cell_lon[i], cell_lat[i]]])[0])
        unlocked_count = sum(1 for i in range(N_CELLS)
                            if unlocked_path and unlocked_path.contains_points([[cell_lon[i], cell_lat[i]]])[0])
        print(f"  セル分類: 固着域={locked_count}, 非固着域={unlocked_count}")
        
    except FileNotFoundError:
        print(f"警告: {polygon_json_path} が見つかりません。デフォルトパラメータを使用します。")
    except Exception as e:
        print(f"警告: {polygon_json_path} の読み込みエラー: {e}")
    
    return b, sigma, L, V_pl, cell_x, cell_y


# ==============================================================================
# 簡易的な相互作用カーネル（Numba最適化版）
# ==============================================================================
@njit(cache=True)
def _compute_kernel_2d(n_cells: int, n_x: int, scale_factor: float, interaction_range: int) -> np.ndarray:
    """Numba JIT最適化された2Dグリッドカーネル計算"""
    K = np.zeros((n_cells, n_cells))
    
    for i in range(n_cells):
        ix_i = i % n_x
        iy_i = i // n_x
        
        for j in range(n_cells):
            ix_j = j % n_x
            iy_j = j // n_x
            
            if i == j:
                K[i, j] = -1.0e6 * scale_factor
            else:
                dx = abs(ix_i - ix_j)
                dy = abs(iy_i - iy_j)
                dist = np.sqrt(float(dx**2 + dy**2))
                
                if dist < interaction_range:
                    K[i, j] = 0.3e6 * scale_factor / (dist + 0.5)**2
    
    return K


def create_interaction_kernel():
    """セル間の応力相互作用カーネルを作成（高速版）"""
    scale_factor = (40 * 8) / N_CELLS
    interaction_range = max(3, int(5 * np.sqrt(scale_factor)))
    
    K = _compute_kernel_2d(N_CELLS, N_X, scale_factor, interaction_range)
    
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
def run_simulation(t_years: float = 1000, polygon_json_path: str = None):
    """2Dシミュレーションを実行"""
    print(f"\nシミュレーション期間: {t_years} 年")
    print("パラメータを設定中...")
    
    if polygon_json_path:
        b, sigma, L, V_pl, cell_x, cell_y = setup_parameters_from_polygon(polygon_json_path)
    else:
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
    
    # 高速化用の閾値
    V_SEISMIC = 1e-3      # 地震判定閾値 [m/s]
    V_PRESEISMIC = 1e-5   # 準備段階閾値 [m/s]
    
    while t < t_end:
        # 速度計算
        V = solve_velocity(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp)
        V_max = np.max(V)
        L_min = np.min(L)
        
        # ========================================
        # 適応タイムステップ（1000年スパン対応）
        # ========================================
        if V_max > V_SEISMIC:
            # 地震発生中：細かいステップ
            dt = min(1.0, L_min / V_max / 10)
            dt = max(dt, 0.01)  # 最低0.01秒
        elif V_max > V_PRESEISMIC:
            # 準備段階：中程度のステップ
            dt = min(3600.0, L_min / V_max / 20)
            dt = max(dt, 1.0)
        else:
            # 静穏期：大きなステップ（最大30日）
            # θの変化率から安定なdtを推定
            dtheta_est = state_evolution(V, theta, L, V_c)
            max_dtheta_ratio = np.max(np.abs(dtheta_est) / np.maximum(theta, 1e-10))
            if max_dtheta_ratio > 0:
                dt_theta = 0.1 / max_dtheta_ratio  # θが10%以上変化しないように
            else:
                dt_theta = 86400.0 * 30
            dt = min(86400.0 * 30, dt_theta)  # 最大30日
            dt = max(dt, 60.0)  # 最低1分
        
        if t + dt > t_end:
            dt = t_end - t
        
        # ========================================
        # 状態更新
        # ========================================
        dtheta = state_evolution(V, theta, L, V_c)
        theta = np.maximum(theta + dtheta * dt, 1e-10)
        
        # 応力更新（相互作用含む）
        dtau = K @ (V_pl - V)
        dtau_scaled = dtau * dt
        
        # 発散防止：静穏期は大きめ、地震時は厳しく
        if V_max > V_SEISMIC:
            max_dtau = 0.3 * np.min(sigma)
        else:
            max_dtau = 0.8 * np.min(sigma)
        dtau_scaled = np.clip(dtau_scaled, -max_dtau, max_dtau)
        tau = tau + dtau_scaled
        
        # 応力の下限
        tau = np.maximum(tau, 0.1 * sigma * mu_0)
        
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
        
        # 進捗表示（時間ベース、重複防止）
        current_time = time.time()
        current_t_years = t / (365.25 * 24 * 3600)
        if current_time - last_print > 5.0:
            progress = t / t_end * 100
            # 前回と同じ進捗なら表示しない
            if not hasattr(run_simulation, 'last_progress') or \
               abs(progress - run_simulation.last_progress) > 0.1:
                print(f"  進捗: {progress:.1f}%, t = {current_t_years:.1f} 年, "
                      f"地震数 = {n_earthquakes}, dt = {dt:.2e} s")
                run_simulation.last_progress = progress
            last_print = current_time
    
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
    
    parser = argparse.ArgumentParser(
        description='南海トラフ地震シミュレーション（2D版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
例:
  python simulation_2d.py --years 1000
  python simulation_2d.py --years 500 --nx 60 --ny 12
  python simulation_2d.py --nx 80 --ny 16  (高解像度)
  python simulation_2d.py --nx 20 --ny 4   (低解像度・高速)
  python simulation_2d.py --polygon-data polygon_data.json --years 500
        '''
    )
    parser.add_argument('--years', type=float, default=1000, 
                        help='シミュレーション年数 (default: 1000)')
    parser.add_argument('--nx', type=int, default=40, 
                        help='走向方向のセル数 (default: 40)')
    parser.add_argument('--ny', type=int, default=8, 
                        help='ディップ方向のセル数 (default: 8)')
    parser.add_argument('--polygon-data', type=str, default=None,
                        help='polygon_data.json のパス（指定するとポリゴンデータを使用）')
    args = parser.parse_args()
    
    # グリッドサイズを設定
    set_grid_size(args.nx, args.ny)
    
    print("=" * 60)
    print("南海トラフ地震シミュレーション（2D版）")
    print("Nankai Trough Earthquake Simulation")
    print("=" * 60)
    print()
    
    earthquakes = run_simulation(t_years=args.years, polygon_json_path=args.polygon_data)
    
    print()
    print("=" * 60)
    print("完了！")
    print(f"  地震イベント画像: results/earthquakes/")
    print(f"  年表: results/timeline.png")
    print(f"  速度履歴: results/velocity_history.png")
    print("=" * 60)

