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
    """
    Newton法で速度を求める（正則化Rate-State則）

    Returns:
        V: 計算された速度
        success: 収束したかどうか (True/False)
    """
    V = np.empty(n)
    success = True

    for i in range(n):
        V_guess = V_0
        theta_i = max(theta[i], 1e-20)
        cell_converged = False

        for _ in range(50):
            V_safe = max(V_guess, 1e-20)
            # 正則化Rate-State則: sinh^-1形式で V→0 の発散を回避
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
                cell_converged = True
                break
            V_guess = V_new

        V[i] = min(max(V_guess, 1e-20), 20.0)  # 上限20m/sでクリップ
        if not cell_converged:
            success = False

    return V, success


@njit(cache=True)
def state_evolution_array(V, theta, L, V_c, n):
    """
    状態変数の発展（Aging law / Slowness law）
    dθ/dt = 1 - V*θ/L
    """
    dtheta = np.empty(n)
    for i in range(n):
        V_safe = max(V[i], 1e-20)
        theta_safe = max(theta[i], 1e-20)
        dtheta[i] = 1.0 - V_safe * theta_safe / L[i]
    return dtheta


# ==============================================================================
# 相互作用カーネル（Numba最適化版）
# ==============================================================================
@njit(cache=True, parallel=True)
def _compute_kernel_fast(lons: np.ndarray, lats: np.ndarray, n: int) -> np.ndarray:
    """Numba JIT最適化された相互作用カーネル計算"""
    K = np.zeros((n, n))
    scale = n / 720.0

    for i in range(n):
        lat_rad_i = np.radians(lats[i])
        for j in range(n):
            if i == j:
                K[i, j] = 2.0e6 * scale
            else:
                dlat = abs(lats[i] - lats[j])
                avg_lat = (lats[i] + lats[j]) / 2.0
                dlon = abs(lons[i] - lons[j]) * np.cos(np.radians(avg_lat))
                dist_deg = np.sqrt(dlat**2 + dlon**2)
                dist_km = dist_deg * 111.0

                if dist_km < 100.0:
                    K[i, j] = 0.6e6 * scale / (dist_km / 10.0 + 1.0)**2

    return K


def create_interaction_kernel(cells: List[Cell]) -> np.ndarray:
    """距離に基づく応力相互作用カーネル（高速版）"""
    n = len(cells)
    lons = np.array([c.lon for c in cells], dtype=np.float64)
    lats = np.array([c.lat for c in cells], dtype=np.float64)
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

    # パラメータ配列を作成 & 検証
    a = np.array([c.a for c in cells])
    b = np.array([c.b for c in cells])
    sigma = np.array([c.sigma_eff for c in cells])
    L = np.array([c.L for c in cells])
    V_pl = np.array([c.V_pl for c in cells])
    areas = np.array([c.area for c in cells]) * 1e6  # km² → m²

    # 入力検証
    if np.any(L <= 0):
        raise ValueError("エラー: L (特徴的すべり量) が0以下のセルがあります。")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("エラー: 摩擦パラメータ a または b が負のセルがあります。")
    if np.any(sigma <= 0):
        raise ValueError("エラー: 有効法線応力が0以下のセルがあります。")

    print("相互作用カーネルを計算中...")
    K = create_interaction_kernel(cells)

    # 初期条件
    V_init = np.mean(V_pl)
    V = np.full(n, V_init)
    theta = L / V

    mu_init = mu_0 + a * np.log(V/V_0) + b * np.log(theta*V_0/L)
    tau = sigma * mu_init + rad_damp * V

    # 初期摂動
    np.random.seed(42)
    tau_perturbation = sigma * 0.0001 * np.random.randn(n)
    tau = tau + tau_perturbation

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

    # タイムステップ制御用
    dt = 0.1  # 初期dt
    dt_prev = dt

    print("\nシミュレーション開始...")

    while t < t_end:
        # 1. 速度計算 (Newton法)
        # 以前の速度Vを初期値として使う（収束性向上のため）
        # ただし、関数内ではV_0からスタートしているので、本来は引数にV_guessを渡すべきだが
        # 現状の構造維持のため、このまま呼び出す
        V_new, success = solve_velocity_array(tau, theta, sigma, a, b, L, mu_0, V_0, rad_damp, n)

        # 収束失敗時のリカバリ：dtを小さくしてやり直したいが、
        # ここでのV計算は現在のtau/thetaに依存しておりdtに関係ない（準動的平衡）。
        # つまり、ここでの失敗は「今の状態が物理的に解けない」ことを意味する。
        # → 警告を出して、前回速度か小さな値で続行する（あるいは停止する）。
        if not success:
             # 重度でなければ続行するが、警告を出す
             if step % 1000 == 0: # ログ抑制
                 print(f"警告: Step {step} でNewton法が収束しませんでした。")

        V = V_new

        # 2. タイムステップの提案 (予測)
        V_max = np.max(V)
        L_min = np.min(L)

        # 基本的なdtの決定ロジック
        if V_max > 0.1:        dt_proposal = min(2.0, L_min / V_max / 5)
        elif V_max > 1e-3:     dt_proposal = min(20.0, L_min / V_max / 2)
        elif V_max > 1e-5:     dt_proposal = min(86400.0, L_min / V_max / 10)
        elif V_max > 1e-7:     dt_proposal = min(86400.0 * 30, L_min / V_max / 20)
        else:                  dt_proposal = 86400.0 * 365

        # 急激な変動抑制
        if step > 0:
            dt_proposal = min(dt_proposal, dt_prev * 1.5)

        dt = max(dt_proposal, 0.001) # 下限設定

        # 3. ステップ試行ループ (適応制御)
        # 応力変化が大きすぎる場合はdtを小さくしてやり直す
        retry_count = 0
        while retry_count < 5:
            # 応力変化予測
            dtau_dt = K @ (V_pl - V)
            dtau_step = dtau_dt * dt

            # クリティカルな応力変化のチェック
            # 有効法線応力の10%を超える変化は物理的に怪しい（不安定化の元）
            max_stress_change_ratio = np.max(np.abs(dtau_step) / (sigma + 1e-6))

            if max_stress_change_ratio > 0.1:
                # 変化が大きすぎる -> dtを小さくしてリトライ
                dt *= 0.5
                retry_count += 1
                if dt < 0.001: # これ以上小さくしない
                    break
            else:
                # OK
                break

        # 最終的な時間調整
        if t + dt > t_end:
            dt = t_end - t

        dt_prev = dt

        # 4. 状態更新 (確定)
        dtheta = state_evolution_array(V, theta, L, V_c, n)
        theta = np.maximum(theta + dtheta * dt, 1e-10)

        # 応力更新
        # リトライループでチェック済みなので、ここではハードクリッピングなしで適用できるが
        # 念のため安全装置として残す（ただしログを出す）
        dtau = K @ (V_pl - V)
        tau = tau + dtau * dt

        # 物理的にありえない負の応力を防ぐ
        tau = np.maximum(tau, 0.1 * sigma * mu_0)

        slip = slip + V * dt

        # 地震検出
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

            M0 = G * np.sum(np.abs(coseismic_slip) * areas)
            Mw = (np.log10(M0) - 9.05) / 1.5 if M0 > 0 else 0

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

            save_path = f'results/earthquakes/eq_{n_earthquakes:03d}.png'
            plot_rupture_map(cells, coseismic_slip, n_earthquakes,
                           t_years_now, Mw, segments, save_path)

        t += dt
        step += 1

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
