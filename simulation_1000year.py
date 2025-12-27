"""
南海トラフ地震シミュレーション - 1000年版
Nankai Trough Earthquake Simulation - 1000 Year Version

Hirose et al. (2022) に基づく長期シミュレーション
- 静穏期は3か月ステップでスロースリップをチェック
- 兆候検出時に細かいステップに切り替え
- 地震発生時は0.01〜1秒ステップ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Optional

# プレート形状モジュールをインポート
from plate_geometry import (
    Cell, SEGMENTS, ASPERITIES, BARRIERS, LSSE_REGIONS, SEAMOUNTS,
    create_realistic_mesh, plot_plate_geometry, plot_rupture_map,
    load_polygon_data, create_mesh_from_polygon
)

# 日本語表示用
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

# 出力ディレクトリ
os.makedirs('results/earthquakes', exist_ok=True)
os.makedirs('results/lsse', exist_ok=True)

# ==============================================================================
# 物理定数
# ==============================================================================
G = 30.0e9          # 剛性率 [Pa]
beta = 3750.0       # S波速度 [m/s]
eta = 1.0           # 準動的補正係数
rad_damp = eta * G / beta

# 摩擦パラメータ
mu_0 = 0.6
V_0 = 1.0e-6
V_c = 1.0e-8

# 時間定数
SECONDS_PER_DAY = 86400.0
SECONDS_PER_MONTH = SECONDS_PER_DAY * 30.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

# 閾値
V_EARTHQUAKE = 0.1        # 地震判定速度 [m/s]
V_PRESEISMIC = 1e-5       # 準備段階速度 [m/s]
V_LSSE = 1e-8             # LSSE判定速度 [m/s] (> V_pl の数倍)

# ==============================================================================
# Rate-State摩擦則（Numba最適化）
# ==============================================================================
@njit(cache=True)
def solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n):
    """Newton法で速度を求める（安定版）"""
    V = np.empty(n)
    
    for i in range(n):
        V_guess = V_0
        theta_i = max(theta[i], 1e-20)
        sigma_i = max(sigma[i], 1e6)  # 最低1MPa
        
        for _ in range(50):
            V_safe = max(V_guess, 1e-20)
            V_safe = min(V_safe, 1e3)  # 上限 1 km/s
            
            mu = mu_0 + a[i] * np.log(V_safe/V_0 + 1) + b[i] * np.log(V_0*theta_i/L[i] + 1)
            F = tau[i] - sigma_i * mu - rad_damp * V_safe
            dmu_dV = a[i] / (V_safe + V_0)
            dF_dV = -sigma_i * dmu_dV - rad_damp
            
            if abs(dF_dV) < 1e-30:
                break
                
            dV = -F / dF_dV
            
            # ステップ制限
            dV = max(dV, -V_guess * 0.9)  # 90%以上の減少を防ぐ
            dV = min(dV, V_guess * 10)     # 10倍以上の増加を防ぐ
            
            V_new = V_guess + dV
            V_new = max(V_new, 1e-20)
            V_new = min(V_new, 1e3)
            
            if abs(dV) < 1e-12 * abs(V_guess):
                V_guess = V_new
                break
            V_guess = V_new
        
        V[i] = max(V_guess, 1e-20)
    
    return V


@njit(cache=True)
def state_evolution_array(V, theta, L, V_c, n):
    """状態変数の発展（安定版）"""
    dtheta = np.empty(n)
    for i in range(n):
        V_safe = max(V[i], 1e-20)
        theta_safe = max(theta[i], 1e-20)
        x = V_safe * theta_safe / L[i]
        
        # オーバーフロー防止
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


# ==============================================================================
# 相互作用カーネル（安定版）
# ==============================================================================
def create_interaction_kernel(cells: List[Cell], base_cells: int = 320) -> np.ndarray:
    """
    距離に基づく応力相互作用カーネル
    セル数に応じてスケーリングを調整
    """
    n = len(cells)
    K = np.zeros((n, n))
    
    # スケールファクター（基準セル数に対する比率）
    scale = base_cells / n
    
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])
    
    for i in range(n):
        for j in range(n):
            if i == j:
                # 自己スティフネス
                K[i, j] = -1.5e6 * scale
            else:
                # 距離計算（球面近似）
                dlat = abs(lats[i] - lats[j])
                dlon = abs(lons[i] - lons[j]) * np.cos(np.radians((lats[i] + lats[j])/2))
                dist_deg = np.sqrt(dlat**2 + dlon**2)
                dist_km = dist_deg * 111
                
                if dist_km < 150:  # 150km以内のみ相互作用
                    K[i, j] = 0.5e6 * scale / (dist_km / 10 + 1)**2
    
    return K


# ==============================================================================
# 地震・LSSEイベント
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
    segments: List[str]


@dataclass
class SlowSlipEvent:
    id: int
    t_start: float
    t_end: float
    cells: List[int]
    max_slip: float
    Mw: float
    region: str


def classify_segments(cells: List[Cell], slip: np.ndarray, 
                      threshold: float = 1.0) -> List[str]:
    """すべり分布からセグメントを特定"""
    seg_slips = {}
    
    for i, cell in enumerate(cells):
        if slip[i] > threshold and cell.segment:
            if cell.segment not in seg_slips:
                seg_slips[cell.segment] = []
            seg_slips[cell.segment].append(slip[i])
    
    return list(seg_slips.keys())


def detect_lsse_region(cells: List[Cell], V: np.ndarray, 
                       V_pl: np.ndarray, threshold_factor: float = 5.0) -> Optional[str]:
    """スロースリップの発生域を検出"""
    for i, cell in enumerate(cells):
        if V[i] > V_pl[i] * threshold_factor and V[i] < V_EARTHQUAKE:
            # LSSE域内かチェック
            for lsse_name, lsse in LSSE_REGIONS.items():
                dist = np.sqrt((cell.lon - lsse['center'][0])**2 + 
                              (cell.lat - lsse['center'][1])**2)
                if dist < 0.5 and lsse['depth_range'][0] <= cell.depth <= lsse['depth_range'][1]:
                    return lsse_name
    return None


# ==============================================================================
# シミュレーション（1000年対応版）
# ==============================================================================
def run_simulation(cells: List[Cell], t_years: float = 1000,
                   quiet_step_days: float = 90,
                   checkpoint_years: float = 100) -> Dict:
    """
    1000年スパンのシミュレーションを実行
    
    Parameters:
        cells: セルリスト
        t_years: シミュレーション年数
        quiet_step_days: 静穏期のタイムステップ（日）
        checkpoint_years: チェックポイント間隔（年）
    
    Returns:
        結果辞書
    """
    n = len(cells)
    
    print(f"\n{'='*60}")
    print(f"南海トラフ地震シミュレーション - {t_years}年版")
    print(f"{'='*60}")
    print(f"セル数: {n}")
    print(f"静穏期ステップ: {quiet_step_days} 日")
    
    # パラメータ配列を作成
    a = np.array([c.a for c in cells])
    b = np.array([c.b for c in cells])
    sigma = np.array([c.sigma_eff for c in cells])
    L = np.array([c.L for c in cells])
    V_pl = np.array([c.V_pl for c in cells])
    areas = np.array([c.area for c in cells]) * 1e6  # km² → m²
    
    print("相互作用カーネルを計算中...")
    K = create_interaction_kernel(cells)
    
    # 初期条件（静穏期からスタート）
    # プレート速度の1倍で初期化（静穏期の状態）
    V_init = np.mean(V_pl)  # プレート速度と同じ
    V = V_pl.copy()  # 各セルのプレート速度で初期化
    
    # 状態変数θは定常状態に近い値で初期化
    theta = L / V
    
    # 初期応力は定常状態
    mu_init = mu_0 + a * np.log(V/V_0 + 1) + b * np.log(V_0*theta/L + 1)
    tau = sigma * mu_init + rad_damp * V
    
    # 初期摂動（非常に小さく、徐々に蓄積させる）
    np.random.seed(42)
    # 摂動を0.1%に縮小（地震を誘発しないように）
    tau += sigma * 0.001 * np.random.randn(n)
    
    slip = np.zeros(n)
    
    # 時間設定
    t = 0.0
    t_end = t_years * SECONDS_PER_YEAR
    quiet_dt = quiet_step_days * SECONDS_PER_DAY
    
    # イベント追跡
    earthquakes = []
    slow_slips = []
    n_earthquakes = 0
    n_lsse = 0
    in_earthquake = False
    in_lsse = False
    eq_start_time = 0.0
    lsse_start_time = 0.0
    eq_cells = []
    lsse_cells = []
    eq_slip_start = None
    lsse_slip_start = None
    
    step = 0
    start_time = time.time()
    last_print = start_time
    last_checkpoint = 0.0
    
    # 速度履歴（サンプリング）
    V_max_hist = []
    t_hist = []
    sample_interval = t_end / 5000
    last_sample = -sample_interval
    
    # --- 初期状態の確認 ---
    V_test = solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n)
    V_max_init = np.max(V_test)
    print(f"\n初期最大速度: {V_max_init:.2e} m/s")
    if V_max_init > V_PRESEISMIC:
        print(f"  警告: 初期速度が準備段階閾値({V_PRESEISMIC:.2e})を超えています")
    if V_max_init > V_EARTHQUAKE:
        print(f"  警告: 初期速度が地震閾値({V_EARTHQUAKE:.2e})を超えています！")
        print(f"  → 初期摂動を調整中...")
        # 摂動を打ち消す
        tau = sigma * mu_init + rad_damp * V
        V_test = solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n)
        print(f"  調整後の最大速度: {np.max(V_test):.2e} m/s")
    
    sys.stdout.flush()
    print("\nシミュレーション開始...")
    sys.stdout.flush()
    
    while t < t_end:
        # 速度計算
        V = solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n)
        V_max = np.max(V)
        L_min = np.min(L)
        
        # ========================================
        # 適応タイムステップ（3フェーズ）
        # ========================================
        if V_max > V_EARTHQUAKE:
            # フェーズ1: 地震発生中（細かいステップ）
            dt = min(1.0, L_min / V_max / 10)
            dt = max(dt, 0.01)
        elif V_max > V_PRESEISMIC:
            # フェーズ2: 準備段階（中程度のステップ）
            dt = min(3600.0, L_min / V_max / 50)
            dt = max(dt, 1.0)
        else:
            # フェーズ3: 静穏期（大きなステップ）
            # θの変化率から安定なdtを推定
            dtheta_est = state_evolution_array(V, theta, L, V_c, n)
            max_dtheta_ratio = np.max(np.abs(dtheta_est) / np.maximum(theta, 1e-10))
            
            if max_dtheta_ratio > 0:
                dt_theta = 0.05 / max_dtheta_ratio  # θが5%以上変化しないように
            else:
                dt_theta = quiet_dt
            
            dt = min(quiet_dt, dt_theta)
            dt = max(dt, 60.0)  # 最低1分
        
        if t + dt > t_end:
            dt = t_end - t
        
        # ========================================
        # 状態更新（安定版）
        # ========================================
        dtheta = state_evolution_array(V, theta, L, V_c, n)
        theta = np.maximum(theta + dtheta * dt, 1e-10)
        
        # 応力更新（相互作用含む）
        dtau = K @ (V_pl - V)
        dtau_scaled = dtau * dt
        
        # 発散防止（フェーズに応じてクリップ）
        if V_max > V_EARTHQUAKE:
            max_dtau = 0.2 * np.min(sigma)
        elif V_max > V_PRESEISMIC:
            max_dtau = 0.5 * np.min(sigma)
        else:
            max_dtau = 0.8 * np.min(sigma)
        
        dtau_scaled = np.clip(dtau_scaled, -max_dtau, max_dtau)
        tau = tau + dtau_scaled
        
        # 応力の下限・上限
        tau = np.clip(tau, 0.1 * sigma * mu_0, 2.0 * sigma * mu_0)
        
        # すべり更新
        slip = slip + V * dt
        
        # ========================================
        # 地震検出
        # ========================================
        eq_cells_now = np.where(V > 0.1)[0]
        
        if len(eq_cells_now) > 0 and not in_earthquake:
            in_earthquake = True
            eq_start_time = t
            eq_cells = list(eq_cells_now)
            eq_slip_start = slip.copy()
        elif len(eq_cells_now) > 0 and in_earthquake:
            eq_cells = list(set(eq_cells) | set(eq_cells_now))
        elif len(eq_cells_now) == 0 and in_earthquake:
            in_earthquake = False
            n_earthquakes += 1
            
            coseismic_slip = slip - eq_slip_start
            max_slip = np.max(coseismic_slip)
            
            # モーメント計算
            M0 = G * np.sum(np.abs(coseismic_slip) * areas)
            Mw = (np.log10(M0) - 9.1) / 1.5 if M0 > 0 else 0
            
            # セグメント分類
            segments = classify_segments(cells, coseismic_slip, threshold=0.5)
            
            eq = Earthquake(
                id=n_earthquakes,
                t_start=eq_start_time,
                t_end=t,
                cells=eq_cells,
                max_slip=max_slip,
                Mw=Mw,
                slip_distribution=coseismic_slip.copy(),
                segments=segments
            )
            earthquakes.append(eq)
            
            t_years_now = eq_start_time / SECONDS_PER_YEAR
            segs_str = ', '.join([SEGMENTS.get(s, {}).get('name', s) for s in segments])
            print(f"  ★ 地震 #{n_earthquakes}: t = {t_years_now:.1f} 年, "
                  f"Mw = {Mw:.1f}, 領域 = {segs_str}")
            
            # 破壊マップ保存
            save_path = f'results/earthquakes/eq_{n_earthquakes:03d}.png'
            plot_rupture_map(cells, coseismic_slip, n_earthquakes, 
                           t_years_now, Mw, segments, save_path)
        
        # ========================================
        # LSSE検出（静穏期のみ）
        # ========================================
        if V_max < V_PRESEISMIC:
            lsse_region = detect_lsse_region(cells, V, V_pl)
            
            if lsse_region and not in_lsse:
                in_lsse = True
                lsse_start_time = t
                lsse_cells = [i for i in range(n) if V[i] > V_pl[i] * 5]
                lsse_slip_start = slip.copy()
            elif lsse_region and in_lsse:
                lsse_cells = list(set(lsse_cells) | 
                                  set([i for i in range(n) if V[i] > V_pl[i] * 5]))
            elif not lsse_region and in_lsse:
                # LSSE終了
                in_lsse = False
                duration = t - lsse_start_time
                
                if duration > SECONDS_PER_DAY * 30:  # 30日以上
                    n_lsse += 1
                    lsse_slip = slip - lsse_slip_start
                    max_slip = np.max(lsse_slip)
                    
                    M0 = G * np.sum(np.abs(lsse_slip) * areas)
                    Mw = (np.log10(M0) - 9.1) / 1.5 if M0 > 0 else 0
                    
                    lsse_event = SlowSlipEvent(
                        id=n_lsse,
                        t_start=lsse_start_time,
                        t_end=t,
                        cells=lsse_cells,
                        max_slip=max_slip,
                        Mw=Mw,
                        region=lsse_region or 'unknown'
                    )
                    slow_slips.append(lsse_event)
                    
                    t_years_now = lsse_start_time / SECONDS_PER_YEAR
                    duration_days = duration / SECONDS_PER_DAY
                    print(f"  ○ LSSE #{n_lsse}: t = {t_years_now:.1f} 年, "
                          f"Mw = {Mw:.1f}, 期間 = {duration_days:.0f} 日")
        
        # 更新
        t += dt
        step += 1
        
        # 履歴サンプリング
        if t - last_sample >= sample_interval:
            t_hist.append(t)
            V_max_hist.append(V_max)
            last_sample = t
        
        # デバッグ: 最初の10ステップを表示
        if step <= 10:
            print(f"  [DEBUG] step={step}, V_max={V_max:.2e}, dt={dt:.2e}s, t={t/SECONDS_PER_YEAR:.6f}年")
            sys.stdout.flush()
        
        # 進捗表示（時間ベース）
        current_time = time.time()
        if current_time - last_print > 10.0:
            progress = t / t_end * 100
            t_years_now = t / SECONDS_PER_YEAR
            elapsed = current_time - start_time
            eta = elapsed / (progress / 100) - elapsed if progress > 0 else 0
            
            print(f"  進捗: {progress:.1f}%, t = {t_years_now:.0f} 年, "
                  f"EQ = {n_earthquakes}, LSSE = {n_lsse}, dt = {dt:.2e} s, "
                  f"ETA = {eta/60:.0f} 分")
            sys.stdout.flush()
            last_print = current_time
        
        # チェックポイント
        if t - last_checkpoint > checkpoint_years * SECONDS_PER_YEAR:
            checkpoint_path = f'results/checkpoint_{int(t/SECONDS_PER_YEAR):04d}yr.npz'
            np.savez_compressed(checkpoint_path, 
                               tau=tau, theta=theta, slip=slip, t=t)
            last_checkpoint = t
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"完了: {step} ステップ, {elapsed:.1f} 秒 ({elapsed/60:.1f} 分)")
    print(f"検出された地震: {n_earthquakes} 個")
    print(f"検出されたLSSE: {n_lsse} 個")
    print(f"{'='*60}")
    
    # 統計
    if len(earthquakes) >= 2:
        intervals = []
        for i in range(1, len(earthquakes)):
            dt_eq = earthquakes[i].t_start - earthquakes[i-1].t_start
            intervals.append(dt_eq / SECONDS_PER_YEAR)
        print(f"平均再発間隔: {np.mean(intervals):.1f} ± {np.std(intervals):.1f} 年")
    
    return {
        'earthquakes': earthquakes,
        'slow_slips': slow_slips,
        'cells': cells,
        't_hist': np.array(t_hist),
        'V_max_hist': np.array(V_max_hist),
        'elapsed': elapsed,
        'n_steps': step
    }


# ==============================================================================
# 可視化
# ==============================================================================
def plot_timeline(earthquakes: List[Earthquake], slow_slips: List[SlowSlipEvent],
                  save_path: str):
    """地震・LSSE年表（論文図6スタイル）"""
    fig, ax = plt.subplots(figsize=(18, 8))
    
    # Y軸はセグメント
    seg_order = ['Z', 'A', 'B', 'C', 'D']
    seg_y = {s: i for i, s in enumerate(seg_order)}
    
    # 地震
    for eq in earthquakes:
        t_years = eq.t_start / SECONDS_PER_YEAR
        
        for seg in eq.segments:
            if seg in seg_y:
                y = seg_y[seg]
                color = SEGMENTS[seg]['color']
                width = max(2, (eq.Mw - 6) * 2)
                ax.barh(y, width=width, left=t_years, height=0.6, 
                       color=color, edgecolor='black', alpha=0.8)
        
        # 大規模地震にマーカー
        if eq.Mw >= 8.0:
            ax.plot(t_years, 2, 'r*', markersize=10 + (eq.Mw - 8) * 5)
    
    # LSSE（下部に表示）
    for lsse in slow_slips:
        t_years = lsse.t_start / SECONDS_PER_YEAR
        duration_years = (lsse.t_end - lsse.t_start) / SECONDS_PER_YEAR
        ax.barh(-1, width=duration_years, left=t_years, height=0.4,
               color='purple', alpha=0.5)
    
    ax.set_yticks([-1] + list(range(len(seg_order))))
    ax.set_yticklabels(['LSSE'] + [f"{s}: {SEGMENTS[s]['name']}" for s in seg_order])
    ax.set_xlabel('時間 [年]', fontsize=12)
    ax.set_title(f'地震・LSSE年表 (地震: {len(earthquakes)}, LSSE: {len(slow_slips)})', 
                fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"年表を保存: {save_path}")


def plot_velocity_history(t_hist: np.ndarray, V_max_hist: np.ndarray, 
                          save_path: str):
    """速度履歴プロット"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    t_years = t_hist / SECONDS_PER_YEAR
    
    ax.semilogy(t_years, V_max_hist, 'b-', linewidth=0.5)
    ax.axhline(y=V_EARTHQUAKE, color='r', linestyle='--', label='地震閾値')
    ax.axhline(y=V_PRESEISMIC, color='orange', linestyle=':', label='準備段階閾値')
    
    ax.set_xlabel('時間 [年]')
    ax.set_ylabel('最大すべり速度 [m/s]')
    ax.set_title('すべり速度の時間発展')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"速度履歴を保存: {save_path}")


# ==============================================================================
# メイン
# ==============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='南海トラフ地震シミュレーション - 1000年版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
例:
  python simulation_1000year.py --years 1000
  python simulation_1000year.py --years 500 --nx 60 --ny 8 --quiet-step 60
  python simulation_1000year.py --years 2000 --nx 80 --ny 10
  python simulation_1000year.py --polygon-data polygon_data.json --years 500
        '''
    )
    parser.add_argument('--years', type=float, default=1000, help='シミュレーション年数')
    parser.add_argument('--nx', type=int, default=60, help='走向方向セル数')
    parser.add_argument('--ny', type=int, default=8, help='ディップ方向セル数')
    parser.add_argument('--quiet-step', type=float, default=90, 
                        help='静穏期タイムステップ（日）')
    parser.add_argument('--polygon-data', type=str, default=None,
                        help='polygon_data.json のパス（指定するとポリゴンデータからメッシュ生成）')
    args = parser.parse_args()
    
    # メッシュ生成
    if args.polygon_data:
        print(f"\npolygon_data.json からメッシュ生成中... ({args.nx} x {args.ny})")
        polygon_data = load_polygon_data(args.polygon_data)
        if polygon_data:
            cells = create_mesh_from_polygon(polygon_data, n_along=args.nx, n_down=args.ny)
        else:
            print("警告: ポリゴンデータの読み込みに失敗。デフォルトメッシュを使用します。")
            cells = create_realistic_mesh(n_along=args.nx, n_down=args.ny)
    else:
        print(f"\nメッシュ生成中... ({args.nx} x {args.ny})")
        cells = create_realistic_mesh(n_along=args.nx, n_down=args.ny)
    print(f"生成されたセル数: {len(cells)}")
    
    # プレート形状を保存
    plot_plate_geometry(cells, 'results/plate_geometry.png')
    
    # シミュレーション実行
    result = run_simulation(
        cells, 
        t_years=args.years,
        quiet_step_days=args.quiet_step
    )
    
    # 可視化
    if result['earthquakes'] or result['slow_slips']:
        plot_timeline(result['earthquakes'], result['slow_slips'], 
                     'results/timeline_full.png')
    
    if len(result['t_hist']) > 0:
        plot_velocity_history(result['t_hist'], result['V_max_hist'],
                             'results/velocity_history.png')
    
    print("\n" + "="*60)
    print("完了！")
    print(f"  プレート形状: results/plate_geometry.png")
    print(f"  地震イベント: results/earthquakes/")
    print(f"  年表: results/timeline_full.png")
    print(f"  速度履歴: results/velocity_history.png")
    print("="*60)
