"""
可視化モジュール
シミュレーション結果の可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os


# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'DejaVu Sans']


@dataclass
class Visualizer:
    """シミュレーション結果の可視化"""
    
    output_dir: str = "results"
    dpi: int = 150
    figsize: Tuple[int, int] = (12, 8)
    
    # 地図用の範囲
    lon_range: Tuple[float, float] = (131.0, 140.0)
    lat_range: Tuple[float, float] = (30.5, 35.5)
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_mesh(self, 
                  mesh,
                  values: np.ndarray = None,
                  title: str = "Plate Boundary Mesh",
                  cmap: str = 'viridis',
                  clabel: str = None,
                  show_coastline: bool = True,
                  save_name: str = None):
        """
        三角形メッシュを表示
        
        Parameters:
            mesh: TriangularMesh オブジェクト
            values: セルごとの値（色付け用）
            title: タイトル
            cmap: カラーマップ
            clabel: カラーバーラベル
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # メッシュ中心を地理座標に変換
        lons = np.zeros(mesh.n_cells)
        lats = np.zeros(mesh.n_cells)
        
        for i in range(mesh.n_cells):
            lon, lat, _ = mesh.coords.local_to_geo(
                mesh.centers[i, 0], mesh.centers[i, 1]
            )
            lons[i] = lon[0]
            lats[i] = lat[0]
        
        if values is not None:
            sc = ax.scatter(lons, lats, c=values, cmap=cmap, s=50, alpha=0.8)
            cbar = plt.colorbar(sc, ax=ax)
            if clabel:
                cbar.set_label(clabel)
        else:
            ax.scatter(lons, lats, c=mesh.depths, cmap='Blues_r', s=30, alpha=0.6)
        
        if show_coastline:
            self._add_coastline(ax)
        
        ax.set_xlim(self.lon_range)
        ax.set_ylim(self.lat_range)
        ax.set_xlabel('Longitude [°E]')
        ax.set_ylabel('Latitude [°N]')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存: {filepath}")
        
        plt.close()
        return fig
    
    def plot_slip_distribution(self,
                               mesh,
                               slip: np.ndarray,
                               title: str = "Slip Distribution",
                               save_name: str = None):
        """すべり分布を表示"""
        return self.plot_mesh(
            mesh, 
            values=slip,
            title=title,
            cmap='hot_r',
            clabel='Slip [m]',
            save_name=save_name
        )
    
    def plot_velocity_timeseries(self,
                                 t: np.ndarray,
                                 V: np.ndarray,
                                 cell_indices: List[int] = None,
                                 cell_names: List[str] = None,
                                 title: str = "Slip Velocity Time Series",
                                 save_name: str = None):
        """
        すべり速度の時系列を表示
        
        Parameters:
            t: 時刻 [s]
            V: すべり速度 [n_times, n_cells] [m/s]
            cell_indices: 表示するセルのインデックス
            cell_names: 凡例用の名前
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 年単位に変換
        t_years = t / 3.15576e7
        
        if cell_indices is None:
            cell_indices = range(min(5, V.shape[1]))
        
        for i, idx in enumerate(cell_indices):
            label = cell_names[i] if cell_names else f'Cell {idx}'
            ax.semilogy(t_years, V[:, idx], label=label, alpha=0.8)
        
        ax.axhline(y=0.1, color='r', linestyle='--', label='Earthquake threshold')
        
        ax.set_xlabel('Time [years]')
        ax.set_ylabel('Slip velocity [m/s]')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存: {filepath}")
        
        plt.close()
        return fig
    
    def plot_event_timeline(self,
                            earthquakes: List,
                            t_range: Tuple[float, float] = None,
                            title: str = "Earthquake Timeline",
                            save_name: str = None):
        """
        地震イベントの年表を表示
        
        論文の図6に相当
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 年単位に変換
        times = [eq.t_start / 3.15576e7 for eq in earthquakes]
        Mws = [eq.Mw for eq in earthquakes]
        
        # バーチャート形式
        colors = []
        for mw in Mws:
            if mw >= 8.5:
                colors.append('red')
            elif mw >= 8.0:
                colors.append('orange')
            elif mw >= 7.5:
                colors.append('yellow')
            else:
                colors.append('gray')
        
        ax.bar(times, Mws, width=5, color=colors, edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Time [years]')
        ax.set_ylabel('Magnitude (Mw)')
        ax.set_title(title)
        
        if t_range:
            ax.set_xlim(t_range)
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', edgecolor='black', label='Mw ≥ 8.5'),
            Patch(facecolor='orange', edgecolor='black', label='Mw ≥ 8.0'),
            Patch(facecolor='yellow', edgecolor='black', label='Mw ≥ 7.5'),
            Patch(facecolor='gray', edgecolor='black', label='Mw < 7.5'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, axis='y', alpha=0.3)
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存: {filepath}")
        
        plt.close()
        return fig
    
    def plot_slip_deficit_rate(self,
                               mesh,
                               V: np.ndarray,
                               V_pl: np.ndarray,
                               title: str = "Slip Deficit Rate",
                               save_name: str = None):
        """
        すべり欠損レートを表示
        
        Slip deficit rate = V_pl - V
        """
        # m/s → cm/year に変換
        deficit_rate = (V_pl - V) * 100 * 3.15576e7
        
        return self.plot_mesh(
            mesh,
            values=deficit_rate,
            title=title,
            cmap='RdYlBu_r',
            clabel='Slip deficit rate [cm/year]',
            save_name=save_name
        )
    
    def plot_3d_mesh(self,
                     mesh,
                     values: np.ndarray = None,
                     title: str = "3D Plate Boundary",
                     save_name: str = None):
        """
        3Dでプレート境界を表示
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # メッシュ中心を地理座標に変換
        lons = np.zeros(mesh.n_cells)
        lats = np.zeros(mesh.n_cells)
        
        for i in range(mesh.n_cells):
            lon, lat, _ = mesh.coords.local_to_geo(
                mesh.centers[i, 0], mesh.centers[i, 1]
            )
            lons[i] = lon[0]
            lats[i] = lat[0]
        
        # 深度を負に（下向き）
        depths = -mesh.depths
        
        if values is not None:
            sc = ax.scatter(lons, lats, depths, c=values, cmap='hot_r', s=30)
            plt.colorbar(sc, ax=ax, shrink=0.5)
        else:
            ax.scatter(lons, lats, depths, c=depths, cmap='Blues', s=30)
        
        ax.set_xlabel('Longitude [°E]')
        ax.set_ylabel('Latitude [°N]')
        ax.set_zlabel('Depth [km]')
        ax.set_title(title)
        
        # 視点設定
        ax.view_init(elev=20, azim=-60)
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"図を保存: {filepath}")
        
        plt.close()
        return fig
    
    def create_animation(self,
                         mesh,
                         slip_history: np.ndarray,
                         t: np.ndarray,
                         fps: int = 10,
                         save_name: str = None):
        """
        すべり分布のアニメーションを作成
        
        Parameters:
            mesh: TriangularMesh
            slip_history: [n_times, n_cells]
            t: 時刻 [n_times]
            fps: フレームレート
            save_name: 保存ファイル名
        """
        from matplotlib.animation import FuncAnimation
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # メッシュ中心の座標
        lons = np.zeros(mesh.n_cells)
        lats = np.zeros(mesh.n_cells)
        for i in range(mesh.n_cells):
            lon, lat, _ = mesh.coords.local_to_geo(
                mesh.centers[i, 0], mesh.centers[i, 1]
            )
            lons[i] = lon[0]
            lats[i] = lat[0]
        
        # 初期フレーム
        vmax = np.max(slip_history)
        sc = ax.scatter(lons, lats, c=slip_history[0], 
                        cmap='hot_r', vmin=0, vmax=vmax, s=50)
        plt.colorbar(sc, ax=ax, label='Cumulative slip [m]')
        
        title = ax.set_title(f't = {t[0]/3.15576e7:.1f} years')
        ax.set_xlim(self.lon_range)
        ax.set_ylim(self.lat_range)
        
        def update(frame):
            sc.set_array(slip_history[frame])
            title.set_text(f't = {t[frame]/3.15576e7:.1f} years')
            return sc, title
        
        anim = FuncAnimation(fig, update, frames=len(t), 
                             interval=1000/fps, blit=True)
        
        if save_name:
            filepath = os.path.join(self.output_dir, save_name)
            anim.save(filepath, fps=fps, dpi=self.dpi)
            print(f"アニメーションを保存: {filepath}")
        
        plt.close()
        return anim
    
    def _add_coastline(self, ax):
        """簡易海岸線を追加"""
        # 日本の本州・四国・九州の概略海岸線
        # （簡易版、詳細は cartopy や geopandas を使用）
        coastline = np.array([
            # 紀伊半島
            [135.0, 33.5], [136.0, 34.0], [136.5, 34.5], [137.0, 34.7],
            [138.0, 34.8], [139.0, 35.0], [139.5, 35.2],
            # 四国
            [132.5, 33.0], [133.0, 33.5], [133.5, 33.8], [134.0, 34.0],
            [134.5, 34.2], [135.0, 34.3],
            # 九州
            [131.0, 31.5], [131.5, 32.0], [132.0, 32.5], [132.5, 33.0],
        ])
        
        # 点として表示
        ax.plot(coastline[:, 0], coastline[:, 1], 'k.', markersize=2, alpha=0.3)
    
    def save_summary_plots(self, 
                           mesh,
                           earthquakes: List,
                           slip_final: np.ndarray,
                           V_final: np.ndarray,
                           V_pl: np.ndarray):
        """サマリー図を一括生成"""
        print("サマリー図を生成中...")
        
        # メッシュ図
        self.plot_mesh(mesh, title="Mesh Overview", save_name="mesh.png")
        
        # 最終すべり分布
        self.plot_slip_distribution(mesh, slip_final, 
                                    title="Final Slip Distribution",
                                    save_name="slip_final.png")
        
        # すべり欠損レート
        self.plot_slip_deficit_rate(mesh, V_final, V_pl,
                                    save_name="slip_deficit.png")
        
        # 3D表示
        self.plot_3d_mesh(mesh, values=slip_final, save_name="mesh_3d.png")
        
        # 地震年表
        if earthquakes:
            self.plot_event_timeline(earthquakes, save_name="timeline.png")
        
        print(f"サマリー図を {self.output_dir} に保存しました")
