"""
南海トラフ地震シミュレーション - Animation Module
地震発生時のすべり分布の時間変化をアニメーション化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional, Any
import os

# Import from parent directories if needed (for type hints)
from src.geometry.mesh import Cell

def create_slip_animation(
    cells: List[Any],
    slip_history: List[np.ndarray],
    time_history: List[float],
    event_id: int,
    save_path: str = None,
    fps: int = 10,
    duration_sec: float = 5.0
):
    """
    地震発生時のすべり分布アニメーションを作成 (GIF/MP4)

    Args:
        cells: セル情報のリスト (lon, latを持つオブジェクト)
        slip_history: 各タイムステップでのすべり分布 [steps, n_cells]
        time_history: 各ステップの時間 [steps]
        event_id: 地震ID
        save_path: 保存先パス (e.g., 'results/animations/eq_001.gif')
    """
    if not slip_history:
        print("警告: アニメーション用の履歴データがありません。")
        return

    n_steps = len(slip_history)
    print(f"アニメーション作成中 (Event #{event_id}): {n_steps} フレーム...")

    # データ準備
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])

    # 累積すべりから増分すべり（地震時のみ）を計算
    # slip_history[0] を基準とする
    base_slip = slip_history[0]
    slip_evolution = [s - base_slip for s in slip_history]
    max_slip = np.max(slip_evolution[-1])

    # プロット設定
    fig, ax = plt.subplots(figsize=(10, 6))

    # カラーマップ
    cmap = LinearSegmentedColormap.from_list(
        'slip', ['white', 'yellow', 'orange', 'red', 'darkred']
    )

    # 初期プロット
    sc = ax.scatter(lons, lats, c=slip_evolution[0], cmap=cmap, s=30,
                   vmin=0, vmax=max(0.1, max_slip), alpha=0.8, edgecolors='none')

    cbar = plt.colorbar(sc, ax=ax, label='すべり量 [m]')

    ax.set_xlim(131.0, 139.5)
    ax.set_ylim(30.5, 36.0)
    ax.set_xlabel('経度 [°E]')
    ax.set_ylabel('緯度 [°N]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    title = ax.set_title(f"地震 #{event_id}: t = {time_history[0]:.4f} 年")

    def update(frame):
        current_slip = slip_evolution[frame]
        sc.set_array(current_slip)
        t_years = time_history[frame]
        title.set_text(f"地震 #{event_id}: t = {t_years:.4f} 年\n最大すべり: {np.max(current_slip):.2f} m")
        return sc, title

    # アニメーション作成
    ani = animation.FuncAnimation(
        fig, update, frames=n_steps, interval=1000/fps, blit=True
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps)
        elif save_path.endswith('.mp4'):
            try:
                ani.save(save_path, writer='ffmpeg', fps=fps)
            except Exception as e:
                print(f"FFmpegが見つからないためGIFとして保存します: {e}")
                ani.save(save_path.replace('.mp4', '.gif'), writer='pillow', fps=fps)
        print(f"アニメーション保存完了: {save_path}")

    plt.close()
