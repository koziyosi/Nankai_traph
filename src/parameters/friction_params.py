"""
摩擦パラメータ設定モジュール
Hirose et al. (2022) の空間的不均質パラメータを設定
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import os


@dataclass
class RegionParam:
    """領域パラメータ定義"""
    name: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    depth_min: float = 0.0
    depth_max: float = 1000.0  # 十分深い値
    value: float = 0.0


@dataclass
class FrictionParameterSetter:
    """
    摩擦パラメータの空間分布を設定
    
    論文の図4に基づく:
    - a-b: 深度依存（0-30km: -0.003, >30km: +0.003）
    - σ_eff: 海山位置で高い（30-60 MPa）
    - L: アスペリティで小さく、バリアで大きい
    - V_pl: 西側が大きい（5.5 cm/y）、東側が小さい（1.0 cm/y）
    """
    
    # デフォルト値
    a_default: float = 0.005
    b_seismogenic: float = 0.008   # 0-30 km
    b_stable: float = 0.002        # >30 km
    
    sigma_eff_default: float = 30.0e6  # 30 MPa
    L_default: float = 0.4             # 0.4 m
    
    V_pl_west: float = 0.055 / 3.15576e7   # 5.5 cm/year → m/s
    V_pl_east: float = 0.010 / 3.15576e7   # 1.0 cm/year → m/s
    
    # 外部CSVファイル（オプション）
    parameter_csv: Optional[str] = None
    
    # 領域定義
    sigma_eff_regions: List[RegionParam] = field(default_factory=list)
    L_regions: List[RegionParam] = field(default_factory=list)

    def __post_init__(self):
        """領域パラメータを初期化"""
        self._init_sigma_eff_regions()
        self._init_L_regions()

    def _init_sigma_eff_regions(self):
        """有効法線応力の領域設定"""
        self.sigma_eff_regions = [
            # 東海沖の海山（Point 1, 2）: 高い σ_eff
            # 深度による条件分岐があるため、2つに分割
            RegionParam("Tokai_Seamount_Shallow", 137.5, 139.0, 34.0, 35.0, 0.0, 20.0, 40.0e6),
            RegionParam("Tokai_Seamount_Deep",    137.5, 139.0, 34.0, 35.0, 20.0, 1000.0, 35.0e6),
            
            # 日向灘（Point 6）: 高い σ_eff
            RegionParam("Hyuganada", 131.0, 132.5, 31.0, 33.0, 0.0, 1000.0, 60.0e6),
            
            # 四国沖の遠方（Point 5）: やや高い
            RegionParam("Off_Shikoku", 133.0, 134.5, 32.0, 33.0, 0.0, 1000.0, 40.0e6),
        ]

    def _init_L_regions(self):
        """特徴的すべり量Lの領域設定"""
        # 評価順序に注意：特殊な条件を先に記述し、後勝ちにするか、
        # あるいはここでreturnするロジックにするかで実装が変わる。
        # 今回は _get_value で「最初に見つかったもの」を返す方式を採用するため、
        # 優先度の高い（より限定的な）領域を先にリストする。

        self.L_regions = [
             # LSSE 域（深部 > 30km）
            RegionParam("Tokai_LSSE",           137.0, 138.5, 34.5, 35.5, 30.0, 1000.0, 0.05),
            RegionParam("Bungo_LSSE",           131.5, 132.5, 32.5, 33.5, 30.0, 1000.0, 0.05),
            RegionParam("Kii_LSSE",             134.5, 135.5, 33.5, 34.5, 30.0, 1000.0, 0.08),
            RegionParam("W_Shikoku_LSSE",       132.5, 133.5, 33.0, 34.0, 30.0, 1000.0, 0.06),

            # バリア領域（大きなL）
            RegionParam("Shionomisaki_Barrier", 135.3, 135.8, 33.0, 34.0, 0.0, 1000.0, 7.5),
            RegionParam("Shima_Barrier",        136.5, 137.0, 33.5, 34.2, 0.0, 1000.0, 5.0),
            RegionParam("Tosabae",              134.0, 134.8, 32.5, 33.3, 0.0, 1000.0, 3.0),
            RegionParam("Ashizuri_Barrier",     132.5, 133.2, 32.0, 32.8, 0.0, 1000.0, 5.0),

            # アスペリティ（小さなL、深度浅め）
            RegionParam("Tonankai_Asperity",    135.5, 137.5, 33.5, 34.5, 0.0, 30.0, 0.3),
            RegionParam("Nankai_Asperity",      133.5, 135.5, 33.0, 34.0, 0.0, 30.0, 0.4),
        ]

    def set_parameters(self, mesh, coords_system) -> None:
        """メッシュにパラメータを設定（CSVがあれば優先使用）"""
        if self.parameter_csv and os.path.exists(self.parameter_csv):
            self._set_parameters_from_csv(mesh, coords_system)
        else:
            self._set_parameters_default(mesh, coords_system)

    def _set_parameters_from_csv(self, mesh, coords_system) -> None:
        """CSVファイルから補間してパラメータを設定"""
        print(f"CSVからパラメータを読み込み中: {self.parameter_csv}")
        try:
            df = pd.read_csv(self.parameter_csv)
            n_cells = mesh.n_cells
            
            # 必要なカラムの存在確認
            points = df[['longitude', 'latitude']].values
            
            # 各パラメータの補間器を作成
            interpolators = {}
            params_to_load = ['a', 'b', 'sigma_eff', 'L', 'V_pl']
            for p in params_to_load:
                if p in df.columns:
                    values = df[p].values
                    # V_pl の単位変換 (cm/year -> m/s) if necessary? 
                    # ユーザー提供CSVが m/s であると仮定するか、規約を決める必要がある。
                    # ここでは一旦そのまま読み込む。
                    interpolators[p + '_linear'] = LinearNDInterpolator(points, values)
                    interpolators[p + '_nearest'] = NearestNDInterpolator(points, values)
            
            # メッシュの各セル中心で値を計算
            mesh_params = {p: np.zeros(n_cells) for p in params_to_load}
            
            for i in range(n_cells):
                lon, lat, _ = coords_system.local_to_geo(
                    mesh.centers[i, 0], mesh.centers[i, 1]
                )
                lon, lat = lon[0], lat[0]
                target_point = np.array([[lon, lat]])
                
                for p in params_to_load:
                    if p + '_linear' in interpolators:
                        val = interpolators[p + '_linear'](target_point)[0]
                        if np.isnan(val):
                            val = interpolators[p + '_nearest'](target_point)[0]
                        mesh_params[p][i] = val
                    else:
                        # CSVにない場合はデフォルトロジックを使用
                        mesh_params[p][i] = self._get_default_value_for_cell(p, lon, lat, mesh.depths[i])
            
            mesh.set_parameters(
                sigma_eff=mesh_params['sigma_eff'],
                L=mesh_params['L'],
                a=mesh_params['a'],
                b=mesh_params['b'],
                V_pl=mesh_params['V_pl']
            )
            print(f"CSVから {n_cells} セルのパラメータ設定完了")
            
        except Exception as e:
            print(f"CSV読み込みエラー: {e}。デフォルト設定を使用します。")
            self._set_parameters_default(mesh, coords_system)

    def _get_default_value_for_cell(self, param_name, lon, lat, depth):
        """特定のセルのデフォルト値を計算"""
        if param_name == 'a':
            return self.a_default
        elif param_name == 'b':
            return self.b_seismogenic if depth <= 30.0 else self.b_stable
        elif param_name == 'sigma_eff':
            return self._get_value_from_regions(lon, lat, depth, self.sigma_eff_regions, self.sigma_eff_default)
        elif param_name == 'L':
            return self._get_value_from_regions(lon, lat, depth, self.L_regions, self.L_default)
        elif param_name == 'V_pl':
            return self._get_V_pl(lon)
        return 0.0

    def _set_parameters_default(self, mesh, coords_system) -> None:
        """従来のデフォルトロジックでパラメータを設定"""
        n_cells = mesh.n_cells
        a = np.full(n_cells, self.a_default)
        b = np.zeros(n_cells)
        sigma_eff = np.full(n_cells, self.sigma_eff_default)
        L = np.full(n_cells, self.L_default)
        V_pl = np.zeros(n_cells)
        
        for i in range(n_cells):
            lon, lat, _ = coords_system.local_to_geo(
                mesh.centers[i, 0], mesh.centers[i, 1]
            )
            lon, lat = lon[0], lat[0]
            depth = mesh.depths[i]
            
            if depth <= 30.0:
                b[i] = self.b_seismogenic
            else:
                b[i] = self.b_stable
            
            sigma_eff[i] = self._get_value_from_regions(lon, lat, depth, self.sigma_eff_regions, self.sigma_eff_default)
            L[i] = self._get_value_from_regions(lon, lat, depth, self.L_regions, self.L_default)
            V_pl[i] = self._get_V_pl(lon)
        
        mesh.set_parameters(sigma_eff=sigma_eff, L=L, a=a, b=b, V_pl=V_pl)
        print(f"デフォルトロジックで {n_cells} セルのパラメータ設定完了")
    
    def _get_value_from_regions(self, lon: float, lat: float, depth: float, regions: List[RegionParam], default_value: float) -> float:
        """
        領域リストから該当するパラメータ値を取得
        条件に合う最初の領域の値を返す
        """
        for region in regions:
            if (region.lon_min <= lon <= region.lon_max and
                region.lat_min <= lat <= region.lat_max and
                region.depth_min <= depth <= region.depth_max):
                return region.value
        return default_value
    
    def _get_V_pl(self, lon: float) -> float:
        """プレート収束速度を取得（経度に依存）"""
        # 西から東に向かって線形に減少
        lon_west = 131.0
        lon_east = 140.0
        
        # 線形補間
        ratio = (lon - lon_west) / (lon_east - lon_west)
        ratio = np.clip(ratio, 0.0, 1.0)
        
        return self.V_pl_west * (1 - ratio) + self.V_pl_east * ratio


def create_parameter_summary(mesh) -> Dict:
    """パラメータのサマリーを作成"""
    return {
        'n_cells': mesh.n_cells,
        'a_b_mean': float(np.mean(mesh.a - mesh.b)),
        'a_b_seismogenic_count': int(np.sum((mesh.a - mesh.b) < 0)),
        'a_b_stable_count': int(np.sum((mesh.a - mesh.b) > 0)),
        'sigma_eff_mean_MPa': float(np.mean(mesh.sigma_eff) / 1e6),
        'L_mean': float(np.mean(mesh.L)),
        'V_pl_mean_cm_year': float(np.mean(mesh.V_pl) * 3.15576e7 * 100),
    }


def get_reference_points() -> Dict[str, Tuple[float, float, str]]:
    """
    論文の参照点を取得
    
    Returns:
        {point_name: (lon, lat, description)}
    """
    return {
        'point_1': (138.5, 34.5, 'Tokai offshore (seamount)'),
        'point_2': (137.5, 34.2, 'Shima Peninsula offshore'),
        'point_3': (136.5, 33.8, 'Tonankai asperity'),
        'point_4': (134.5, 33.5, 'Nankai asperity'),
        'point_5': (133.5, 32.5, 'Off Shikoku'),
        'point_6': (131.5, 32.0, 'Hyuganada asperity'),
        'point_7': (131.0, 31.5, 'Southern Hyuganada'),
        'point_8': (137.5, 35.0, 'Tokai LSSE'),
        'point_9': (135.0, 34.0, 'Kii Channel LSSE'),
        'point_10': (133.0, 33.5, 'Western Shikoku LSSE'),
        'point_11': (132.0, 33.0, 'Bungo Channel LSSE'),
    }
