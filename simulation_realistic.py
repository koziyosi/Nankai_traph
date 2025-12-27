"""
南海トラフ地震シミュレーション - リアル形状版
Nankai Trough Earthquake Simulation - Realistic Geometry Version

Hirose et al. (2022) に基づく完全版シミュレーション
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
import os
import time
from dataclasses import dataclass
from typing import List, Dict

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

# ==============================================================================
# Rate-State摩擦則（Numba最適化）
# 標準的なDieterich-Ruina則（正則化版）
# ==============================================================================
@njit(cache=True)
def solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n):
    """Newton法で速度を求める（正則化Rate-State則）"""
    V = np.empty(n)
    
    for i in range(n):
        V_guess = V_0
        theta_i = max(theta[i], 1e-20)
        
        for _ in range(50):
            V_safe = max(V_guess, 1e-20)
            # 正則化Rate-State則: sinh^-1形式で V→0 の発散を回避
            # mu = mu_0 + a*asinh(V/(2*V_0)) + b*ln(V_0*theta/L)
            # 簡略化版: mu = mu_0 + a*ln(V/V_0) + b*ln(theta*V_0/L)
            log_V_ratio = np.log(V_safe / V_0)
            log_theta_term = np.log(theta_i * V_0 / L[i])
            mu = mu_0 + a[i] * log_V_ratio + b[i] * log_theta_term
            
            F = tau[i] - sigma[i] * mu - rad_damp * V_safe
            dmu_dV = a[i] / V_safe
            dF_dV = -sigma[i] * dmu_dV - rad_damp
            dV = -F / dF_dV
            V_new = max(V_guess + dV, 1e-20)
            if abs(dV) < 1e-12 * abs(V_guess):
                V_guess = V_new
                break
            V_guess = V_new
        
        V[i] = min(max(V_guess, 1e-20), 20.0)  # 上限20m/sでクリップ（数値爆発防止）
    
    return V


@njit(cache=True)
def state_evolution_array(V, theta, L, V_c, n):
    """
    状態変数の発展（Aging law / Slowness law）
    dθ/dt = 1 - V*θ/L
    
    これが標準的な状態発展則（Dieterich, 1979）
    """
    dtheta = np.empty(n)
    for i in range(n):
        V_safe = max(V[i], 1e-20)
        theta_safe = max(theta[i], 1e-20)
        # Aging law: dθ/dt = 1 - V*θ/L
        dtheta[i] = 1.0 - V_safe * theta_safe / L[i]
    return dtheta


# ==============================================================================
# 相互作用カーネル（Numba最適化版）
# ==============================================================================
@njit(cache=True, parallel=True)
def _compute_kernel_fast(lons: np.ndarray, lats: np.ndarray, n: int) -> np.ndarray:
    """Numba JIT最適化された相互作用カーネル計算"""
    K = np.zeros((n, n))
    # スケーリング則の修正: 剛性はセルサイズに反比例（=数nに比例）すべき
    # Original: scale = (40 * 8) / n  (誤り: nが増えると剛性が下がる)
    # Correct:  scale = n / 720.0     (修正: n=480(60x8)のとき元の約0.66になるよう調整しつつ、nに比例させる)
    scale = n / 720.0
    
    for i in range(n):
        lat_rad_i = np.radians(lats[i])
        for j in range(n):
            if i == j:
                # 自己スティフネス（正の値であるべき）
                K[i, j] = 2.0e6 * scale
            else:
                # 距離に応じた相互作用（球面距離の近似）
                dlat = abs(lats[i] - lats[j])
                avg_lat = (lats[i] + lats[j]) / 2.0
                dlon = abs(lons[i] - lons[j]) * np.cos(np.radians(avg_lat))
                dist_deg = np.sqrt(dlat**2 + dlon**2)
                dist_km = dist_deg * 111.0
                
                if dist_km < 100.0:  # 100km以内のみ相互作用
                    K[i, j] = 0.6e6 * scale / (dist_km / 10.0 + 1.0)**2
    
    return K


def create_interaction_kernel(cells: List[Cell]) -> np.ndarray:
    """距離に基づく応力相互作用カーネル（高速版）"""
    n = len(cells)
    
    # セル座標を抽出
    lons = np.array([c.lon for c in cells], dtype=np.float64)
    lats = np.array([c.lat for c in cells], dtype=np.float64)
    
    # Numba JIT関数を呼び出し
    K = _compute_kernel_fast(lons, lats, n)
    
    return K


# ==============================================================================
# 地震イベント
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


# ==============================================================================
# シミュレーション
# ==============================================================================
def run_simulation(cells: List[Cell], t_years: float = 1000) -> List[Earthquake]:
    """シミュレーション実行"""
    n = len(cells)
    
    print(f"\nシミュレーション期間: {t_years} 年")
    print(f"セル数: {n}")
    
    # パラメータ配列を作成
    a = np.array([c.a for c in cells])
    b = np.array([c.b for c in cells])
    sigma = np.array([c.sigma_eff for c in cells])
    L = np.array([c.L for c in cells])
    V_pl = np.array([c.V_pl for c in cells])
    areas = np.array([c.area for c in cells]) * 1e6  # km² → m²
    
    print("相互作用カーネルを計算中...")
    K = create_interaction_kernel(cells)
    
    # 初期条件（修正版Rate-State則に対応）
    # プレート速度付近から開始して定常状態に近づける
    V_init = np.mean(V_pl)  # プレート速度から開始
    V = np.full(n, V_init)
    theta = L / V  # 定常状態の状態変数
    
    # 初期摩擦と応力（修正版式に対応）
    mu_init = mu_0 + a * np.log(V/V_0) + b * np.log(theta*V_0/L)
    tau = sigma * mu_init + rad_damp * V
    
    # 初期応力に摂動を加えて地震を誘発（対称性を崩すのみの微小ノイズ）
    np.random.seed(42)
    # 以前は1-15%の摂動を与えていたが、いきなり大地震になるのを防ぐため0.01%にする
    tau_perturbation = sigma * 0.0001 * np.random.randn(n)  # 0.01%の微小摂動
    tau = tau + tau_perturbation
    
    # 強制的な地震誘発（critical_indices）は削除し、静穏スタートとする
    # critical_indices = np.random.choice(n, size=min(30, n//5), replace=False)
    # tau[critical_indices] += sigma[critical_indices] * 0.02
    
    slip = np.zeros(n)

    
    # 時間設定
    t = 0.0
    t_end = t_years * 365.25 * 24 * 3600
    
    # イベント追跡
    earthquakes = []
    n_earthquakes = 0
    in_earthquake = False
    eq_start_time = 0.0
    eq_cells = []
    eq_slip_start = None
    
    step = 0
    start_time = time.time()
    last_print = start_time
    
    print("\nシミュレーション開始...")
    
    while t < t_end:
        # 速度計算
        V = solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n)
        
        # 適応タイムステップ（長期シミュレーション最適化版）
        V_max = np.max(V)
        L_min = np.min(L)
        
        # 速度に基づく動的タイムステップ
        # 速度に基づく動的タイムステップ
        if V_max > 0.1:  # 激震時（V > 10 cm/s）
            dt = min(2.0, L_min / V_max / 5)
            dt = max(dt, 0.05)
        elif V_max > 1e-3:  # 地震/活動期（V > 1 mm/s）
            # ユーザー要望で高速化：精度より速度優先
            # dtが小さくなりすぎないように下限を大きく設定
            dt = min(20.0, L_min / V_max / 2)
            dt = max(dt, 2.0)  # 最低2.0秒
        elif V_max > 1e-5:  # 加速段階（V > 10 μm/s）
            dt = min(86400.0, L_min / V_max / 10)  # 最大1日
            dt = max(dt, 1.0)  # 最低1秒（修正：加速期はもう少し細かく）
        elif V_max > 1e-7:  # 準備段階（V > 100 nm/s）
            dt = min(86400.0 * 30, L_min / V_max / 20)  # 最大1ヶ月
            dt = max(dt, 100.0)  # 最低100秒
        else:  # 静穏期（V < 100 nm/s）
            # 長期シミュレーション用：最大1年のタイムステップ
            dt = 86400.0 * 365  # 1年
        
        # dtの急激な増加を抑制（前回比1.5倍まで）
        # これにより、静穏期(1年)から加速期(数秒)への激しい変化の前に
        # 中間的なステップを挟み、数値安定性を高める
        if step > 0:
            dt = min(dt, dt_prev * 1.5)
        dt_prev = dt
        
        if t + dt > t_end:
            dt = t_end - t
        
        # 状態更新
        dtheta = state_evolution_array(V, theta, L, V_c, n)
        theta = np.maximum(theta + dtheta * dt, 1e-10)
        
        # 応力更新（オーバーフロー防止付き）
        # V_pl - V: 固着時(V < V_pl)は正→応力蓄積、地震時(V > V_pl)は負→応力解放
        dtau = K @ (V_pl - V)
        
        # 応力変化のクリッピング（数値安定性のため）
        max_dtau = 0.1 * np.min(sigma)  # 応力の10%を上限
        dtau_clipped = np.clip(dtau * dt, -max_dtau, max_dtau)
        tau = tau + dtau_clipped
        
        # 応力の下限（負の応力を防止）
        tau = np.maximum(tau, 0.1 * sigma * mu_0)
        
        # すべり更新
        slip = slip + V * dt
        
        # 地震検出（V > 1e-3 m/s を地震的すべりと判定）
        eq_cells_now = np.where(V > 1e-3)[0]
        
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
            # Hanks & Kanamori (1979): Mw = (log10(M0) - 9.05) / 1.5
            Mw = (np.log10(M0) - 9.05) / 1.5 if M0 > 0 else 0
            
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
            
            t_years_now = eq_start_time / (365.25 * 24 * 3600)
            segs_str = ', '.join([SEGMENTS.get(s, {}).get('name', s) for s in segments])
            print(f"  地震 #{n_earthquakes}: t = {t_years_now:.1f} 年, "
                  f"Mw = {Mw:.1f}, 領域 = {segs_str}")
            
            # 破壊マップ保存
            save_path = f'results/earthquakes/eq_{n_earthquakes:03d}.png'
            plot_rupture_map(cells, coseismic_slip, n_earthquakes, 
                           t_years_now, Mw, segments, save_path)
        
        t += dt
        step += 1
        
        # 進捗表示（重複防止）
        current_time = time.time()
        if current_time - last_print > 10.0:
            progress = t / t_end * 100
            t_years_now = t / (365.25 * 24 * 3600)
            if not hasattr(run_simulation, 'last_progress') or \
               abs(progress - run_simulation.last_progress) > 0.1:
                print(f"  進捗: {progress:.1f}%, t = {t_years_now:.1f} 年, "
                      f"地震数 = {n_earthquakes}, dt = {dt:.2e} s, Vmax = {V_max:.2e} m/s")
                run_simulation.last_progress = progress
            last_print = current_time
    
    elapsed = time.time() - start_time
    print(f"\n完了: {step} ステップ, {elapsed:.1f} 秒")
    print(f"検出された地震: {n_earthquakes} 個")
    
    return earthquakes


def plot_timeline(earthquakes: List[Earthquake], save_path: str):
    """地震年表（論文図6スタイル）"""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Y軸はセグメント
    seg_order = ['Z', 'A', 'B', 'C', 'D']
    seg_y = {s: i for i, s in enumerate(seg_order)}
    
    for eq in earthquakes:
        t_years = eq.t_start / (365.25 * 24 * 3600)
        
        for seg in eq.segments:
            if seg in seg_y:
                y = seg_y[seg]
                color = SEGMENTS[seg]['color']
                width = max(1, eq.Mw - 6)  # Mwに応じた幅
                ax.barh(y, width=width, left=t_years, height=0.6, 
                       color=color, edgecolor='black', alpha=0.8)
    
    ax.set_yticks(range(len(seg_order)))
    ax.set_yticklabels([f"{s}: {SEGMENTS[s]['name']}" for s in seg_order])
    ax.set_xlabel('時間 [年]')
    ax.set_title('地震年表 - セグメント別発生履歴\n(Hirose et al. 2022 図6スタイル)', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"年表を保存: {save_path}")


# ==============================================================================
# メイン
# ==============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='南海トラフ地震シミュレーション（リアル形状版）',
        epilog='''
例:
  python simulation_realistic.py --years 500
  python simulation_realistic.py --years 1000 --nx 80 --ny 12
  python simulation_realistic.py --polygon-data polygon_data.json --years 500
        '''
    )
    parser.add_argument('--years', type=float, default=500, help='シミュレーション年数')
    parser.add_argument('--nx', type=int, default=60, help='走向方向セル数')
    parser.add_argument('--ny', type=int, default=8, help='ディップ方向セル数')
    parser.add_argument('--polygon-data', type=str, default=None,
                        help='polygon_data.json のパス（指定するとポリゴンデータからメッシュ生成）')
    args = parser.parse_args()
    
    print("=" * 60)
    print("南海トラフ地震シミュレーション（リアル形状版）")
    print("Hirose et al. (2022) に基づく")
    print("=" * 60)
    
    # メッシュ生成
    # メッシュ生成
    polygon_file = args.polygon_data
    if polygon_file is None and os.path.exists('polygon_data.json'):
        polygon_file = 'polygon_data.json'
        print("  情報: polygon_data.json を自動検出しました。これを使用します。")

    if polygon_file:
        print(f"\n{polygon_file} からメッシュ生成中... ({args.nx} x {args.ny})")
        polygon_data = load_polygon_data(polygon_file)
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
    earthquakes = run_simulation(cells, t_years=args.years)
    
    # 年表作成
    if earthquakes:
        plot_timeline(earthquakes, 'results/timeline.png')
    
    print("\n" + "=" * 60)
    print("完了！")
    print(f"  プレート形状: results/plate_geometry.png")
    print(f"  地震イベント: results/earthquakes/")
    print(f"  年表: results/timeline.png")
    print("=" * 60)
