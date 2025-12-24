"""
三角形メッシュ生成モジュール - Triangular Mesh Generator
プレート境界面を三角形要素で離散化
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from numba import njit, prange
import os

from .plate_boundary import PlateBoundary
from .coordinates import CoordinateSystem


@dataclass
class Cell:
    """三角形セルの情報"""
    id: int
    vertices: np.ndarray  # [3, 3] - 3頂点 x (x, y, z)
    center: np.ndarray    # [3] - セル中心座標
    area: float           # 面積 [km²]
    normal: np.ndarray    # [3] - 法線ベクトル
    strike: float         # 走向 [rad]
    dip: float            # 傾斜 [rad]
    depth: float          # 中心深度 [km]
    
    # パラメータ（後で設定）
    sigma_eff: float = 30.0e6   # 有効法線応力 [Pa]
    L: float = 0.4              # 特徴的すべり量 [m]
    a: float = 0.005            # 摩擦パラメータa
    b: float = 0.008            # 摩擦パラメータb
    V_pl: float = 0.05          # プレート収束速度 [m/year]


@dataclass
class TriangularMesh:
    """
    三角形メッシュクラス
    
    プレート境界面を走向方向にストリップ状に分割し、
    各ストリップを三角形で離散化
    """
    
    plate: PlateBoundary = field(default_factory=PlateBoundary)
    coords: CoordinateSystem = field(default_factory=CoordinateSystem)
    cell_size: float = 10.0  # km
    
    # メッシュデータ
    cells: List[Cell] = field(default_factory=list)
    n_cells: int = 0
    
    # 配列形式（高速計算用）
    vertices: np.ndarray = None      # [n_cells, 3, 3]
    centers: np.ndarray = None       # [n_cells, 3]
    areas: np.ndarray = None         # [n_cells]
    normals: np.ndarray = None       # [n_cells, 3]
    depths: np.ndarray = None        # [n_cells]
    
    # パラメータ配列
    sigma_eff: np.ndarray = None     # [n_cells]
    L: np.ndarray = None             # [n_cells]
    a: np.ndarray = None             # [n_cells]
    b: np.ndarray = None             # [n_cells]
    V_pl: np.ndarray = None          # [n_cells]
    
    def generate(self, 
                 lon_range: Tuple[float, float] = (131.0, 140.0),
                 lat_range: Tuple[float, float] = (30.5, 35.5),
                 depth_range: Tuple[float, float] = (0.0, 40.0)) -> None:
        """
        メッシュを生成
        
        Parameters:
            lon_range: 経度範囲
            lat_range: 緯度範囲
            depth_range: 深度範囲 [km]
        """
        print(f"メッシュ生成開始: セルサイズ={self.cell_size}km")
        
        # 4隅をローカル座標に変換して範囲を決定
        corners_lon = [lon_range[0], lon_range[1], lon_range[0], lon_range[1]]
        corners_lat = [lat_range[0], lat_range[0], lat_range[1], lat_range[1]]
        
        x_coords = []
        y_coords = []
        for lon, lat in zip(corners_lon, corners_lat):
            x, y, _ = self.coords.geo_to_local(lon, lat, 0)
            x_coords.append(x[0] if hasattr(x, '__len__') else x)
            y_coords.append(y[0] if hasattr(y, '__len__') else y)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        print(f"  ローカル座標範囲: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}] km")
        
        # Y方向（ディップ方向）のストリップ数
        n_strips = max(1, int((y_max - y_min) / self.cell_size) + 1)
        y_edges = np.linspace(y_min, y_max, n_strips + 1)
        
        # X方向（走向方向）の点数
        n_x = max(1, int((x_max - x_min) / self.cell_size) + 1)
        x_edges = np.linspace(x_min, x_max, n_x + 1)
        
        cells_list = []
        cell_id = 0

        
        for i in range(n_strips):
            y0, y1 = y_edges[i], y_edges[i + 1]
            
            for j in range(n_x):
                x0, x1 = x_edges[j], x_edges[j + 1]
                
                # 4つの頂点
                corners = [
                    (x0, y0), (x1, y0),
                    (x0, y1), (x1, y1)
                ]
                
                # 各頂点の深度を取得
                z_corners = []
                valid = True
                for cx, cy in corners:
                    lon, lat, _ = self.coords.local_to_geo(cx, cy)
                    lon_val = lon[0] if hasattr(lon, '__len__') else lon
                    lat_val = lat[0] if hasattr(lat, '__len__') else lat
                    z = self.plate.get_depth(lon_val, lat_val)
                    z_val = z[0] if hasattr(z, '__len__') else float(z)
                    
                    if z_val < depth_range[0] or z_val > depth_range[1]:
                        valid = False
                        break
                    z_corners.append(float(z_val))

                
                if not valid:
                    continue
                
                # 2つの三角形を作成
                # Triangle 1: (0, 1, 2)
                v1 = np.array([
                    [corners[0][0], corners[0][1], z_corners[0]],
                    [corners[1][0], corners[1][1], z_corners[1]],
                    [corners[2][0], corners[2][1], z_corners[2]],
                ])
                cell1 = self._create_cell(cell_id, v1)
                if cell1 is not None:
                    cells_list.append(cell1)
                    cell_id += 1
                
                # Triangle 2: (1, 3, 2)
                v2 = np.array([
                    [corners[1][0], corners[1][1], z_corners[1]],
                    [corners[3][0], corners[3][1], z_corners[3]],
                    [corners[2][0], corners[2][1], z_corners[2]],
                ])
                cell2 = self._create_cell(cell_id, v2)
                if cell2 is not None:
                    cells_list.append(cell2)
                    cell_id += 1
        
        self.cells = cells_list
        self.n_cells = len(cells_list)
        
        # 配列形式に変換
        self._convert_to_arrays()
        
        print(f"メッシュ生成完了: {self.n_cells} セル")
    
    def _create_cell(self, cell_id: int, vertices: np.ndarray) -> Optional[Cell]:
        """三角形セルを作成"""
        # 面積計算
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)
        
        if area < 1e-6:  # 退化した三角形
            return None
        
        # 法線（上向きに正規化）
        normal = cross / np.linalg.norm(cross)
        if normal[2] < 0:
            normal = -normal
        
        # 中心
        center = np.mean(vertices, axis=0)
        
        # 傾斜
        dip = np.arccos(np.clip(normal[2], -1, 1))
        
        # 走向
        if np.abs(normal[2]) < 0.9999:
            strike_vec = np.array([-normal[1], normal[0], 0])
            strike_vec = strike_vec / np.linalg.norm(strike_vec)
            strike = np.arctan2(strike_vec[0], strike_vec[1])
        else:
            strike = 0.0
        
        return Cell(
            id=cell_id,
            vertices=vertices,
            center=center,
            area=area,
            normal=normal,
            strike=strike,
            dip=dip,
            depth=center[2]
        )
    
    def _convert_to_arrays(self):
        """リスト形式を配列形式に変換（高速計算用）"""
        n = self.n_cells
        
        self.vertices = np.zeros((n, 3, 3))
        self.centers = np.zeros((n, 3))
        self.areas = np.zeros(n)
        self.normals = np.zeros((n, 3))
        self.depths = np.zeros(n)
        
        self.sigma_eff = np.zeros(n)
        self.L = np.zeros(n)
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.V_pl = np.zeros(n)
        
        for i, cell in enumerate(self.cells):
            self.vertices[i] = cell.vertices
            self.centers[i] = cell.center
            self.areas[i] = cell.area
            self.normals[i] = cell.normal
            self.depths[i] = cell.depth
            
            self.sigma_eff[i] = cell.sigma_eff
            self.L[i] = cell.L
            self.a[i] = cell.a
            self.b[i] = cell.b
            self.V_pl[i] = cell.V_pl
    
    def set_parameters(self, 
                       sigma_eff: np.ndarray = None,
                       L: np.ndarray = None,
                       a: np.ndarray = None,
                       b: np.ndarray = None,
                       V_pl: np.ndarray = None):
        """パラメータ配列を設定"""
        if sigma_eff is not None:
            self.sigma_eff = sigma_eff
        if L is not None:
            self.L = L
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if V_pl is not None:
            self.V_pl = V_pl
    
    def get_segment_mask(self, 
                         lon_range: Tuple[float, float],
                         lat_range: Tuple[float, float],
                         depth_range: Tuple[float, float] = None) -> np.ndarray:
        """
        指定領域内のセルのマスクを取得
        
        Returns:
            mask: bool array [n_cells]
        """
        # ローカル座標からジオ座標に変換
        centers_geo = np.zeros((self.n_cells, 3))
        for i in range(self.n_cells):
            lon, lat, dep = self.coords.local_to_geo(
                self.centers[i, 0], 
                self.centers[i, 1],
                self.centers[i, 2]
            )
            centers_geo[i] = [lon[0], lat[0], dep[0]]
        
        mask = (
            (centers_geo[:, 0] >= lon_range[0]) & 
            (centers_geo[:, 0] <= lon_range[1]) &
            (centers_geo[:, 1] >= lat_range[0]) & 
            (centers_geo[:, 1] <= lat_range[1])
        )
        
        if depth_range is not None:
            mask &= (
                (self.depths >= depth_range[0]) & 
                (self.depths <= depth_range[1])
            )
        
        return mask
    
    def save(self, filepath: str):
        """メッシュをファイルに保存"""
        np.savez_compressed(
            filepath,
            vertices=self.vertices,
            centers=self.centers,
            areas=self.areas,
            normals=self.normals,
            depths=self.depths,
            sigma_eff=self.sigma_eff,
            L=self.L,
            a=self.a,
            b=self.b,
            V_pl=self.V_pl,
            cell_size=self.cell_size,
            n_cells=self.n_cells
        )
        print(f"メッシュを保存: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TriangularMesh':
        """ファイルからメッシュを読み込み"""
        data = np.load(filepath)
        
        mesh = cls(cell_size=float(data['cell_size']))
        mesh.n_cells = int(data['n_cells'])
        mesh.vertices = data['vertices']
        mesh.centers = data['centers']
        mesh.areas = data['areas']
        mesh.normals = data['normals']
        mesh.depths = data['depths']
        mesh.sigma_eff = data['sigma_eff']
        mesh.L = data['L']
        mesh.a = data['a']
        mesh.b = data['b']
        mesh.V_pl = data['V_pl']
        
        print(f"メッシュを読み込み: {filepath} ({mesh.n_cells} セル)")
        return mesh


@njit(parallel=True, cache=True)
def compute_cell_areas(vertices: np.ndarray) -> np.ndarray:
    """
    全セルの面積を並列計算
    
    Parameters:
        vertices: [n_cells, 3, 3]
    
    Returns:
        areas: [n_cells]
    """
    n_cells = vertices.shape[0]
    areas = np.empty(n_cells)
    
    for i in prange(n_cells):
        v1 = vertices[i, 1] - vertices[i, 0]
        v2 = vertices[i, 2] - vertices[i, 0]
        
        # クロス積
        cx = v1[1] * v2[2] - v1[2] * v2[1]
        cy = v1[2] * v2[0] - v1[0] * v2[2]
        cz = v1[0] * v2[1] - v1[1] * v2[0]
        
        areas[i] = 0.5 * np.sqrt(cx*cx + cy*cy + cz*cz)
    
    return areas
