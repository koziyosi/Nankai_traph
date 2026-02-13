"""
南海トラフ地震シミュレーション - Interactive Visualization Module
Plotly を使用したインタラクティブな3D可視化
"""

import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional, Any
import os

# Import from parent directories if needed (for type hints)
# from src.geometry.mesh import Cell

def create_interactive_plate_plot(
    cells: List[Any],
    slip_distribution: Optional[np.ndarray] = None,
    title: str = "Nankai Trough Plate Geometry",
    save_path: str = "results/interactive_plate.html"
):
    """
    プレート境界の3D形状とすべり分布をインタラクティブに可視化

    Args:
        cells: セル情報のリスト (lon, lat, depth, area, etc.)
        slip_distribution: 各セルのすべり量または物理量 (Optional)
        title: プロットのタイトル
        save_path: 保存先HTMLパス
    """
    if not cells:
        print("警告: 可視化するセルデータがありません。")
        return

    n_cells = len(cells)
    print(f"インタラクティブ3Dプロット作成中 ({n_cells} セル)...")

    # 座標データ
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])
    depths = np.array([c.depth for c in cells]) # km (positive down)

    # 色付けデータ
    if slip_distribution is not None:
        colors = slip_distribution
        cbar_title = "Slip [m]"
        colorscale = "Inferno"
        cmin = 0
        cmax = max(0.1, np.max(slip_distribution))
    else:
        # デフォルトは深度または固着度
        # L (特徴的すべり量) の逆数などを表示すると面白い
        L_vals = np.array([c.L for c in cells])
        colors = 1.0 / np.maximum(L_vals, 0.01) # Coupling strength proxy
        cbar_title = "Coupling (1/L)"
        colorscale = "Viridis"
        cmin = np.min(colors)
        cmax = np.max(colors)

    # ホバーテキスト
    hover_texts = []
    for i, c in enumerate(cells):
        txt = f"ID: {c.id}<br>"
        txt += f"Lon: {c.lon:.2f}, Lat: {c.lat:.2f}<br>"
        txt += f"Depth: {c.depth:.1f} km<br>"
        if hasattr(c, 'segment'):
            txt += f"Segment: {c.segment}<br>"
        if slip_distribution is not None:
            txt += f"Slip: {slip_distribution[i]:.2f} m"
        hover_texts.append(txt)

    # 3D散布図 (Mesh3dだと三角形接続情報が必要だが、ここではScatter3dで点群表示)
    # 点群だと重たいので、数千点ならOK。数万点ならMesh3d推奨。
    # ここでは簡易的にScatter3dを使用。

    trace = go.Scatter3d(
        x=lons,
        y=lats,
        z=-depths, # PlotlyはZ-upなので、深度を負にする
        mode='markers',
        marker=dict(
            size=4,
            color=colors,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=cbar_title),
            opacity=0.8
        ),
        text=hover_texts,
        hoverinfo='text'
    )

    # レイアウト設定
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Longitude [°E]'),
            yaxis=dict(title='Latitude [°N]'),
            zaxis=dict(title='Elevation [km] (Depth < 0)'),
            aspectratio=dict(x=1, y=1, z=0.3) # 深度方向を強調しない
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig = go.Figure(data=[trace], layout=layout)

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    print(f"インタラクティブプロット保存完了: {save_path}")
