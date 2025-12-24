"""
座標変換モジュール - Coordinate System Transformations
地理座標（緯度経度）と局所座標（XYZ）の変換
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from numba import njit, prange


@dataclass
class CoordinateSystem:
    """
    座標系変換クラス
    
    論文では:
    - X軸: N150°W方向（走向方向に直交）
    - Y軸: N60°W方向（平均的なプレート収束方向）
    - Z軸: 深さ方向（下向き正）
    """
    
    # 参照点（モデル中心）
    origin_lon: float = 135.0  # 紀伊水道付近
    origin_lat: float = 33.0
    
    # 座標軸の方位角
    y_azimuth: float = 330.0  # N60°W = 330° (プレート収束方向)
    
    # 地球半径
    earth_radius: float = 6371.0  # km
    
    def __post_init__(self):
        # 事前計算
        self.cos_lat_origin = np.cos(np.radians(self.origin_lat))
        self.y_azimuth_rad = np.radians(self.y_azimuth)
        self.x_azimuth_rad = self.y_azimuth_rad + np.pi / 2  # X軸はY軸に直交
        
    def geo_to_local(self, lon: np.ndarray, lat: np.ndarray, depth: np.ndarray = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        地理座標 → 局所直交座標
        
        Parameters:
            lon: 経度 [degrees]
            lat: 緯度 [degrees]
            depth: 深度 [km] (optional)
        
        Returns:
            x, y, z: 局所座標 [km]
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)
        
        # 緯度経度差をkm換算
        dlon = lon - self.origin_lon
        dlat = lat - self.origin_lat
        
        # 球面近似で距離変換
        dx_geo = dlon * self.cos_lat_origin * np.pi / 180.0 * self.earth_radius
        dy_geo = dlat * np.pi / 180.0 * self.earth_radius
        
        # 座標回転（地理座標系 → モデル座標系）
        cos_az = np.cos(self.y_azimuth_rad - np.pi/2)  # 北からの回転
        sin_az = np.sin(self.y_azimuth_rad - np.pi/2)
        
        x = dx_geo * cos_az + dy_geo * sin_az
        y = -dx_geo * sin_az + dy_geo * cos_az
        
        if depth is not None:
            z = np.atleast_1d(depth)
        else:
            z = np.zeros_like(x)
            
        return x, y, z
    
    def local_to_geo(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        局所直交座標 → 地理座標
        
        Parameters:
            x, y: 局所座標 [km]
            z: 深度 [km] (optional)
        
        Returns:
            lon, lat, depth [degrees, degrees, km]
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        
        # 逆回転
        cos_az = np.cos(self.y_azimuth_rad - np.pi/2)
        sin_az = np.sin(self.y_azimuth_rad - np.pi/2)
        
        dx_geo = x * cos_az - y * sin_az
        dy_geo = x * sin_az + y * cos_az
        
        # km → 緯度経度
        dlon = dx_geo / (self.cos_lat_origin * np.pi / 180.0 * self.earth_radius)
        dlat = dy_geo / (np.pi / 180.0 * self.earth_radius)
        
        lon = self.origin_lon + dlon
        lat = self.origin_lat + dlat
        
        depth = z if z is not None else np.zeros_like(x)
        
        return lon, lat, depth


@njit(parallel=True, cache=True)
def compute_distances(x1: np.ndarray, y1: np.ndarray, z1: np.ndarray,
                      x2: np.ndarray, y2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    2点間の3D距離を計算（並列化・JIT最適化）
    
    Parameters:
        x1, y1, z1: 点群1の座標
        x2, y2, z2: 点群2の座標
    
    Returns:
        distances: 距離行列 [n1 x n2]
    """
    n1 = len(x1)
    n2 = len(x2)
    distances = np.empty((n1, n2), dtype=np.float64)
    
    for i in prange(n1):
        for j in range(n2):
            dx = x1[i] - x2[j]
            dy = y1[i] - y2[j]
            dz = z1[i] - z2[j]
            distances[i, j] = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    return distances


@njit(cache=True)
def rotate_vector(vx: float, vy: float, angle: float) -> Tuple[float, float]:
    """
    2Dベクトルを反時計回りに回転
    
    Parameters:
        vx, vy: 入力ベクトル
        angle: 回転角 [radians]
    
    Returns:
        rotated vx, vy
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return vx * cos_a - vy * sin_a, vx * sin_a + vy * cos_a


def compute_strike_dip(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                       vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    三角形要素の走向と傾斜を計算
    
    Parameters:
        vertices: 頂点座標 [n_cells, 3, 3] (cell, vertex, xyz)
    
    Returns:
        strike: 走向角 [radians]
        dip: 傾斜角 [radians]
    """
    n_cells = len(vertices)
    strike = np.zeros(n_cells)
    dip = np.zeros(n_cells)
    
    for i in range(n_cells):
        # 三角形の2辺ベクトル
        v1 = vertices[i, 1] - vertices[i, 0]
        v2 = vertices[i, 2] - vertices[i, 0]
        
        # 法線ベクトル（上向きに正規化）
        normal = np.cross(v1, v2)
        if normal[2] < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)
        
        # 傾斜角
        dip[i] = np.arccos(normal[2])
        
        # 走向（法線の水平成分に直交）
        if np.abs(normal[2]) < 0.9999:
            strike_vec = np.array([-normal[1], normal[0], 0])
            strike_vec = strike_vec / np.linalg.norm(strike_vec)
            strike[i] = np.arctan2(strike_vec[0], strike_vec[1])
        else:
            strike[i] = 0.0
    
    return strike, dip
