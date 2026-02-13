"""
南海トラフ地震シミュレーション - リアル形状版
Nankai Trough Earthquake Simulation - Realistic Geometry Version

Hirose et al. (2022) に基づく完全版シミュレーション
Modified to use rigorous physics modules from src/ and optimized for performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any
import os
import sys

# Import from src
sys.path.append(os.getcwd()) # Ensure root is in path

from src.simulation.realistic import RealisticSimulation, Earthquake

# プレート形状モジュールをインポート
from plate_geometry import (
    Cell, SEGMENTS, ASPERITIES, BARRIERS, LSSE_REGIONS, SEAMOUNTS,
    create_realistic_mesh, plot_plate_geometry, plot_rupture_map,
    load_polygon_data, create_mesh_from_polygon
)

# 可視化モジュール
from src.output.interactive import create_interactive_plate_plot

# 日本語表示用
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

# 出力ディレクトリ
os.makedirs('results/earthquakes', exist_ok=True)

# ==============================================================================
# シミュレーション
# ==============================================================================
def run_simulation(cells: List[Cell], t_years: float = 1000,
                   animate: bool = False,
                   interactive: bool = False) -> List[Earthquake]:
    """シミュレーション実行 (Using RealisticSimulation class)"""

    # シミュレーションクラスのインスタンス化
    sim = RealisticSimulation(cells, segments_info=SEGMENTS)

    # コールバック設定（地震発生時にマップを保存）
    def on_earthquake(eq: Earthquake):
        t_years_now = eq.t_start / (365.25 * 24 * 3600)
        save_path = f'results/earthquakes/eq_{eq.id:03d}.png'

        # 破壊域マップを作成
        plot_rupture_map(cells, eq.slip_distribution, eq.id,
                       t_years_now, eq.Mw, eq.segments, save_path)

        # インタラクティブ3Dプロット (HTML)
        if interactive:
            html_path = f'results/earthquakes/eq_{eq.id:03d}.html'
            create_interactive_plate_plot(cells, eq.slip_distribution, f"Earthquake #{eq.id} (Mw={eq.Mw:.1f})", html_path)

    sim.on_earthquake = on_earthquake

    # 実行
    sim.animate_earthquakes = animate
    earthquakes = sim.run(years=t_years, animate=animate)

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
                # 幅はMwに応じて変えるが、最低でも1年分確保
                width = max(1.0, (eq.Mw - 6.0) * 5.0)

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
        description='南海トラフ地震シミュレーション（リアル形状版・最適化済み）',
        epilog='''
例:
  python simulation_realistic.py --years 500
  python simulation_realistic.py --years 1000 --nx 80 --ny 12
  python simulation_realistic.py --polygon-data polygon_data.json --years 500
  python simulation_realistic.py --years 200 --animate --interactive
        '''
    )
    parser.add_argument('--years', type=float, default=500, help='シミュレーション年数')
    parser.add_argument('--nx', type=int, default=60, help='走向方向セル数')
    parser.add_argument('--ny', type=int, default=8, help='ディップ方向セル数')
    parser.add_argument('--polygon-data', type=str, default=None,
                        help='polygon_data.json のパス（指定するとポリゴンデータからメッシュ生成）')
    parser.add_argument('--animate', action='store_true', help='地震発生時のアニメーション（GIF）を作成')
    parser.add_argument('--interactive', action='store_true', help='地震発生時のインタラクティブ3Dプロット（HTML）を作成')
    args = parser.parse_args()

    print("=" * 60)
    print("南海トラフ地震シミュレーション（リアル形状版・最適化済み）")
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

    # インタラクティブ3Dプレート形状 (初期状態)
    if args.interactive:
        create_interactive_plate_plot(cells, title="Nankai Trough Plate Geometry (Coupling)", save_path='results/plate_geometry_3d.html')

    # シミュレーション実行
    earthquakes = run_simulation(cells, t_years=args.years,
                               animate=args.animate,
                               interactive=args.interactive)

    # 年表作成
    if earthquakes:
        plot_timeline(earthquakes, 'results/timeline.png')

    print("\n" + "=" * 60)
    print("完了！")
    print(f"  プレート形状: results/plate_geometry.png")
    if args.interactive:
        print(f"  3Dプレート形状: results/plate_geometry_3d.html")
    print(f"  地震イベント: results/earthquakes/")
    print(f"  年表: results/timeline.png")
    print("=" * 60)
