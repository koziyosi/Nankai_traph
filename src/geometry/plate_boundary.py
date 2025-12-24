"""
プレート境界形状モジュール - Plate Boundary Geometry
Hirose et al. (2008) のプレート形状データに基づく
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os


@dataclass
class PlateBoundary:
    """
    フィリピン海プレート上面の形状モデル
    
    深度 0-10km: トラフ軸から線形補間
    深度 10-40km: 実際のプレート形状 (Hirose et al. 2008)
    """
    
    # モデル範囲
    lon_min: float = 131.0
    lon_max: float = 140.0
    lat_min: float = 30.5
    lat_max: float = 35.5
    
    # 深度コンター（経験的モデル）
    depth_contours: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """プレート深度モデルを初期化"""
        if not self.depth_contours:
            self._create_default_model()
    
    def _create_default_model(self):
        """
        Hirose et al. (2008) に基づく簡易プレート深度モデル
        各緯度での経度-深度関係を定義
        """
        # 深度等値線のおおよその位置（経度, 緯度）
        # これは論文の図から読み取った近似値
        
        # トラフ軸（深度 ~0km）
        self.trough_axis = np.array([
            [139.5, 34.9],
            [138.5, 34.3],
            [137.5, 33.9],
            [136.5, 33.6],
            [135.5, 33.3],
            [134.5, 33.0],
            [133.5, 32.6],
            [132.5, 32.2],
            [131.5, 31.7],
        ])
        
        # 10km等深線
        self.depth_10km = np.array([
            [139.0, 35.0],
            [138.0, 34.5],
            [137.0, 34.1],
            [136.0, 33.8],
            [135.0, 33.5],
            [134.0, 33.2],
            [133.0, 32.9],
            [132.0, 32.5],
            [131.5, 32.1],
        ])
        
        # 20km等深線
        self.depth_20km = np.array([
            [138.5, 35.2],
            [137.5, 34.8],
            [136.5, 34.4],
            [135.5, 34.1],
            [134.5, 33.7],
            [133.5, 33.4],
            [132.5, 33.0],
            [131.8, 32.6],
        ])
        
        # 30km等深線
        self.depth_30km = np.array([
            [138.0, 35.4],
            [137.0, 35.0],
            [136.0, 34.7],
            [135.0, 34.4],
            [134.0, 34.0],
            [133.0, 33.7],
            [132.0, 33.3],
        ])
        
        # 40km等深線
        self.depth_40km = np.array([
            [137.5, 35.5],
            [136.5, 35.2],
            [135.5, 34.9],
            [134.5, 34.5],
            [133.5, 34.1],
            [132.5, 33.7],
        ])
    
    def get_depth(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        指定した経度・緯度でのプレート深度を取得
        
        Parameters:
            lon: 経度 [degrees]
            lat: 緯度 [degrees]
        
        Returns:
            depth: プレート深度 [km]
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)
        depth = np.zeros_like(lon)
        
        for i in range(len(lon)):
            depth[i] = self._interpolate_depth(lon[i], lat[i])
        
        return depth
    
    def _interpolate_depth(self, lon: float, lat: float) -> float:
        """単一点での深度補間"""
        # 簡易的な距離ベース補間
        # 各等深線からの距離を計算し、補間
        
        contours = [
            (0.0, self.trough_axis),
            (10.0, self.depth_10km),
            (20.0, self.depth_20km),
            (30.0, self.depth_30km),
            (40.0, self.depth_40km),
        ]
        
        distances = []
        for depth_val, contour in contours:
            # 等深線への最短距離
            d = np.sqrt((contour[:, 0] - lon)**2 + (contour[:, 1] - lat)**2)
            min_dist = np.min(d)
            distances.append((depth_val, min_dist))
        
        # 2つの最近接等深線を使って補間
        distances.sort(key=lambda x: x[1])
        d1, dist1 = distances[0]
        d2, dist2 = distances[1]
        
        if dist1 + dist2 < 1e-10:
            return d1
        
        # 逆距離重み付け補間
        w1 = 1.0 / (dist1 + 0.001)
        w2 = 1.0 / (dist2 + 0.001)
        depth = (d1 * w1 + d2 * w2) / (w1 + w2)
        
        return np.clip(depth, 0.0, 40.0)
    
    def get_plate_surface_grid(self, nx: int = 50, ny: int = 30
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        プレート表面のグリッドデータを生成
        
        Parameters:
            nx, ny: グリッド点数
        
        Returns:
            lon_grid, lat_grid, depth_grid
        """
        lon = np.linspace(self.lon_min, self.lon_max, nx)
        lat = np.linspace(self.lat_min, self.lat_max, ny)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        depth_grid = np.zeros_like(lon_grid)
        for i in range(ny):
            for j in range(nx):
                depth_grid[i, j] = self._interpolate_depth(lon_grid[i, j], lat_grid[i, j])
        
        return lon_grid, lat_grid, depth_grid
    
    def is_on_plate(self, lon: float, lat: float, depth: float, 
                    tolerance: float = 2.0) -> bool:
        """
        指定点がプレート境界面上にあるかチェック
        
        Parameters:
            lon, lat: 位置 [degrees]
            depth: 深度 [km]
            tolerance: 許容誤差 [km]
        
        Returns:
            True if on plate boundary
        """
        plate_depth = self._interpolate_depth(lon, lat)
        return abs(depth - plate_depth) < tolerance


def create_plate_boundary_from_data(filepath: str = None) -> PlateBoundary:
    """
    外部データファイルからプレート境界を作成
    
    Parameters:
        filepath: プレート深度データのパス
                  (Hirose et al. 2008 形式)
    
    Returns:
        PlateBoundary オブジェクト
    """
    if filepath and os.path.exists(filepath):
        # TODO: 実際のデータ形式に合わせて実装
        # データは https://www.mri-jma.go.jp/Dep/sei/fhirose/plate/en.PlateData.html から取得可能
        pass
    
    # デフォルトモデルを返す
    return PlateBoundary()
