"""
プレート境界形状モジュール - Plate Boundary Geometry
Hirose et al. (2008) のプレート形状データに基づく
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import os
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

@dataclass
class PlateBoundary:
    """
    フィリピン海プレート上面の形状モデル
    
    データファイルから読み込んだ点群データを用いて、
    任意の地点のプレート深度を補間計算する。
    """
    
    # モデル範囲
    lon_min: float = 131.0
    lon_max: float = 140.0
    lat_min: float = 30.5
    lat_max: float = 35.5
    
    # データポイント
    depth_data_points: Optional[np.ndarray] = None # [[lon, lat, depth], ...]
    _linear_interpolator: Optional[object] = None
    _nearest_interpolator: Optional[object] = None
    
    def __post_init__(self):
        """プレート深度モデルを初期化"""
        self._load_data()
    
    def _load_data(self):
        """
        CSVファイルからプレート深度データを読み込む
        """
        # データファイルのパス（デフォルト）
        csv_path = os.path.join(os.getcwd(), 'nankai_depth_points.csv')
        
        if os.path.exists(csv_path):
            print(f"プレート深度データを読み込み中: {csv_path}")
            try:
                df = pd.read_csv(csv_path)
                # カラム名の揺らぎに対応
                if 'longitude' in df.columns and 'latitude' in df.columns and 'depth_km' in df.columns:
                     self.depth_data_points = df[['longitude', 'latitude', 'depth_km']].values
                else:
                    print("警告: CSVファイルのカラム名が想定と異なります。読み込みをスキップします。")
                    self._create_default_model()
            except Exception as e:
                 print(f"警告: データ読み込みエラー: {e}")
                 self._create_default_model()
        else:
            print("警告: nankai_depth_points.csv が見つかりません。デフォルト（ハードコード）モデルを使用します。")
            self._create_default_model()
        
        # 補間器を作成（キャッシュ）
        if self.depth_data_points is not None:
             points = self.depth_data_points[:, :2]
             values = self.depth_data_points[:, 2]
             self._linear_interpolator = LinearNDInterpolator(points, values)
             self._nearest_interpolator = NearestNDInterpolator(points, values)

    def _create_default_model(self):
        """
        Hirose et al. (2008) に基づく簡易プレート深度モデル（フォールバック用）
        """
        # トラフ軸（深度 ~0km）
        trough_axis = np.array([
            [139.5, 34.9, 0.0], [138.5, 34.3, 0.0], [137.5, 33.9, 0.0],
            [136.5, 33.6, 0.0], [135.5, 33.3, 0.0], [134.5, 33.0, 0.0],
            [133.5, 32.6, 0.0], [132.5, 32.2, 0.0], [131.5, 31.7, 0.0],
        ])
        
        # 10km等深線
        depth_10km = np.array([
            [139.0, 35.0, 10.0], [138.0, 34.5, 10.0], [137.0, 34.1, 10.0],
            [136.0, 33.8, 10.0], [135.0, 33.5, 10.0], [134.0, 33.2, 10.0],
            [133.0, 32.9, 10.0], [132.0, 32.5, 10.0], [131.5, 32.1, 10.0],
        ])
        
        # 20km等深線
        depth_20km = np.array([
            [138.5, 35.2, 20.0], [137.5, 34.8, 20.0], [136.5, 34.4, 20.0],
            [135.5, 34.1, 20.0], [134.5, 33.7, 20.0], [133.5, 33.4, 20.0],
            [132.5, 33.0, 20.0], [131.8, 32.6, 20.0],
        ])

        # 30km等深線
        depth_30km = np.array([
            [138.0, 35.4, 30.0], [137.0, 35.0, 30.0], [136.0, 34.7, 30.0],
            [135.0, 34.4, 30.0], [134.0, 34.0, 30.0], [133.0, 33.7, 30.0],
            [132.0, 33.3, 30.0],
        ])
        
        # 40km等深線
        depth_40km = np.array([
            [137.5, 35.5, 40.0], [136.5, 35.2, 40.0], [135.5, 34.9, 40.0],
            [134.5, 34.5, 40.0], [133.5, 34.1, 40.0], [132.5, 33.7, 40.0],
        ])
        
        self.depth_data_points = np.vstack([
            trough_axis, depth_10km, depth_20km, depth_30km, depth_40km
        ])
        
        points = self.depth_data_points[:, :2]
        values = self.depth_data_points[:, 2]
        self._linear_interpolator = LinearNDInterpolator(points, values)
        self._nearest_interpolator = NearestNDInterpolator(points, values)
    
    def get_depth(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """
        指定した経度・緯度でのプレート深度を取得 (Cacheされたinterpolatorを使用)
        
        Parameters:
            lon: 経度 [degrees]
            lat: 緯度 [degrees]
        
        Returns:
            depth: プレート深度 [km]
        """
        lon = np.atleast_1d(lon)
        lat = np.atleast_1d(lat)
        
        if self._linear_interpolator is None:
             return np.zeros_like(lon) # Should not happen if init worked

        # クエリ点
        xi = np.column_stack([lon, lat])
        
        # 線形補間
        depth = self._linear_interpolator(xi)
        
        # 線形補間外（外挿領域）は nearest で埋める
        mask_nan = np.isnan(depth)
        if np.any(mask_nan):
            depth[mask_nan] = self._nearest_interpolator(xi[mask_nan])
            
        return np.clip(depth, 0.0, 60.0) # 深度制限
    
    def _interpolate_depth(self, lon: float, lat: float) -> float:
        """単一点での深度補間（互換性のため残存）"""
        return float(self.get_depth(np.array([lon]), np.array([lat]))[0])

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
        
        depth_grid = self.get_depth(lon_grid.flatten(), lat_grid.flatten()).reshape(ny, nx)
        
        return lon_grid, lat_grid, depth_grid
    
    def is_on_plate(self, lon: float, lat: float, depth: float, 
                    tolerance: float = 2.0) -> bool:
        """
        指定点がプレート境界面上にあるかチェック
        """
        plate_depth = self._interpolate_depth(lon, lat)
        return abs(depth - plate_depth) < tolerance

def create_plate_boundary_from_data(filepath: str = None) -> PlateBoundary:
    """
    データファイルからプレート境界を作成
    """
    pb = PlateBoundary()
    if filepath and os.path.exists(filepath):
         # ファイルパスが指定された場合はそれを読み込む処理を追加可能
         # 現状は __post_init__ で自動読み込み
         pass
    return pb
