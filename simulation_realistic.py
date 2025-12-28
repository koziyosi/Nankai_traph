"""
南海トラフ地震シミュレーション - リアル形状版
Nankai Trough Earthquake Simulation - Realistic Geometry Version

Hirose et al. (2022) に基づく完全版シミュレーション
Modified to use rigorous physics modules from src/
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os
import time
from dataclasses import dataclass

# Import from src
import sys
sys.path.append(os.getcwd()) # Ensure root is in path

from src.geometry.coordinates import CoordinateSystem
from src.physics.stress_kernel import StressKernel
from src.physics.friction import RateStateFriction
from src.physics.equations import QuasiDynamicEquations, create_derivative_function
from src.solver.runge_kutta import RK45Solver

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
# 地震イベント構造体
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


def estimate_normals(centers: np.ndarray) -> np.ndarray:
    """
    点群から法線ベクトルを推定する (Z-down座標系)
    最近傍点を用いて局所平面を当てはめる
    """
    n = len(centers)
    normals = np.zeros((n, 3))

    # 簡易的な実装: 各点について、最も近い2点を探して平面を作る
    for i in range(n):
        # 距離計算
        diff = centers - centers[i]
        dist_sq = np.sum(diff**2, axis=1)

        # 自分自身(0)を除く、最も近い2点を探す
        dist_sq[i] = np.inf
        indices = np.argsort(dist_sq)[:2]

        p0 = centers[i]
        p1 = centers[indices[0]]
        p2 = centers[indices[1]]

        v1 = p1 - p0
        v2 = p2 - p0

        # 外積
        n_vec = np.cross(v1, v2)
        norm = np.linalg.norm(n_vec)

        if norm > 1e-10:
            n_vec /= norm
        else:
            n_vec = np.array([0.0, 0.0, -1.0]) # fallback

        # 向きの調整: Z成分が負（上向き）になるように統一
        # Z-down座標系なので、上向きはZが減少する方向
        if n_vec[2] > 0:
            n_vec = -n_vec

        normals[i] = n_vec

    return normals


# ==============================================================================
# シミュレーション
# ==============================================================================
def run_simulation(cells: List[Cell], t_years: float = 1000) -> List[Earthquake]:
    """シミュレーション実行 (Using src modules)"""
    n = len(cells)

    print(f"\nシミュレーション期間: {t_years} 年")
    print(f"セル数: {n}")

    # 1. データ準備
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])
    depths = np.array([c.depth for c in cells])
    areas = np.array([c.area for c in cells]) # km^2

    a = np.array([c.a for c in cells])
    b = np.array([c.b for c in cells])
    sigma_eff = np.array([c.sigma_eff for c in cells])
    L = np.array([c.L for c in cells])
    V_pl = np.array([c.V_pl for c in cells])

    # 入力検証
    if np.any(L <= 0):
        raise ValueError("エラー: L (特徴的すべり量) が0以下のセルがあります。")
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("エラー: 摩擦パラメータ a または b が負のセルがあります。")

    # 2. 座標変換と法線推定
    coords = CoordinateSystem()
    x, y, z = coords.geo_to_local(lons, lats, depths)
    centers = np.column_stack((x, y, z)) # [n, 3] in km

    print("法線ベクトルを推定中...")
    normals = estimate_normals(centers)

    # 3. 物理モデル構築
    print("応力カーネルを計算中 (src.physics.StressKernel)...")
    kernel = StressKernel(G=30.0e9, nu=0.25)
    kernel.compute(centers, areas, normals) # centers in km, areas in km^2

    equations = QuasiDynamicEquations(
        friction=RateStateFriction(),
        kernel=kernel,
        G=30.0e9,
        beta=3750.0
    )

    # 4. 初期条件
    V_init = 1.0e-9 # m/s (approx plate rate)
    state0 = equations.initialize_state(n, sigma_eff, a, b, L, V_init)

    # 5. ソルバー設定
    derivative_func = create_derivative_function(
        equations, n, a, b, L, sigma_eff, V_pl
    )

    solver = RK45Solver(
        rtol=1.0e-5, # 少し緩めて高速化
        atol=1.0e-7,
        dt_min=0.001,
        dt_max=86400 * 30, # 1ヶ月
        dt_initial=1.0
    )

    t_end = t_years * 365.25 * 24 * 3600

    # 6. コールバックとイベント検出
    earthquakes = []

    # 状態追跡用クロージャ変数
    sim_state = {
        'n_earthquakes': 0,
        'in_earthquake': False,
        'eq_start_time': 0.0,
        'eq_cells': set(),
        'eq_slip_start': np.zeros(n),
        'slip_cumulative': np.zeros(n),
        'last_print': time.time(),
        'start_time': time.time()
    }

    def callback(t, state, dt):
        tau = state[:n]
        theta = state[n:]

        # 速度計算
        V = equations.get_slip_velocity(tau, theta, a, b, L, sigma_eff)

        # すべり更新
        sim_state['slip_cumulative'] += V * dt

        # 地震検出ロジック
        eq_cells_now = np.where(V > 1e-2)[0] # 1cm/s threshold

        if len(eq_cells_now) > 0:
            if not sim_state['in_earthquake']:
                # 地震開始
                sim_state['in_earthquake'] = True
                sim_state['eq_start_time'] = t
                sim_state['eq_cells'] = set(eq_cells_now)
                sim_state['eq_slip_start'] = sim_state['slip_cumulative'].copy()
            else:
                # 地震継続中
                sim_state['eq_cells'].update(eq_cells_now)
        elif sim_state['in_earthquake']:
            # 地震終了
            sim_state['in_earthquake'] = False
            sim_state['n_earthquakes'] += 1

            coseismic_slip = sim_state['slip_cumulative'] - sim_state['eq_slip_start']
            max_slip = np.max(coseismic_slip)

            # モーメント計算
            # Area in m^2 = areas (km^2) * 1e6
            M0 = 30.0e9 * np.sum(np.abs(coseismic_slip) * areas * 1e6)
            Mw = (np.log10(M0) - 9.05) / 1.5 if M0 > 0 else 0

            # セグメント分類
            segments = classify_segments(cells, coseismic_slip, threshold=0.1)

            eq = Earthquake(
                id=sim_state['n_earthquakes'],
                t_start=sim_state['eq_start_time'],
                t_end=t,
                cells=list(sim_state['eq_cells']),
                max_slip=max_slip,
                Mw=Mw,
                slip_distribution=coseismic_slip.copy(),
                segments=segments
            )
            earthquakes.append(eq)

            t_years_now = sim_state['eq_start_time'] / (365.25 * 24 * 3600)
            segs_str = ', '.join([SEGMENTS.get(s, {}).get('name', s) for s in segments])
            print(f"  地震 #{eq.id}: t = {t_years_now:.1f} 年, "
                  f"Mw = {Mw:.1f}, 領域 = {segs_str}")

            # 地震マップ保存
            save_path = f'results/earthquakes/eq_{eq.id:03d}.png'
            plot_rupture_map(cells, coseismic_slip, eq.id,
                           t_years_now, Mw, segments, save_path)

        # 進捗表示
        current_time = time.time()
        if current_time - sim_state['last_print'] > 5.0:
            progress = t / t_end * 100
            t_y = t / (365.25 * 24 * 3600)
            V_max = np.max(V)
            print(f"  進捗: {progress:.1f}%, t = {t_y:.1f} 年, "
                  f"地震数 = {sim_state['n_earthquakes']}, dt = {dt:.2e} s, Vmax = {V_max:.2e} m/s")
            sim_state['last_print'] = current_time

    print("\nシミュレーション開始 (RK45)...")
    result = solver.solve(
        f=derivative_func,
        y0=state0,
        t_span=(0, t_end),
        callback=callback,
        verbose=False
    )

    elapsed = time.time() - sim_state['start_time']
    print(f"\n完了: {result['n_steps']} ステップ, {elapsed:.1f} 秒")
    print(f"検出された地震: {sim_state['n_earthquakes']} 個")

    return earthquakes


def plot_timeline(earthquakes: List[Earthquake], save_path: str):
    """地震年表（論文図6スタイル）"""
    if not earthquakes:
        return

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
                width = max(1, eq.Mw - 6) * 5  # 可視化用に幅を調整
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
