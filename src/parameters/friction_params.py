"""
摩擦パラメータ設定モジュール
Hirose et al. (2022) の空間的不均質パラメータを設定
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


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
    
    def set_parameters(self, mesh, coords_system) -> None:
        """
        メッシュにパラメータを設定
        
        Parameters:
            mesh: TriangularMesh オブジェクト
            coords_system: CoordinateSystem オブジェクト
        """
        n_cells = mesh.n_cells
        
        # 配列を初期化
        a = np.full(n_cells, self.a_default)
        b = np.zeros(n_cells)
        sigma_eff = np.full(n_cells, self.sigma_eff_default)
        L = np.full(n_cells, self.L_default)
        V_pl = np.zeros(n_cells)
        
        for i in range(n_cells):
            # 地理座標を取得
            lon, lat, _ = coords_system.local_to_geo(
                mesh.centers[i, 0], mesh.centers[i, 1]
            )
            lon, lat = lon[0], lat[0]
            depth = mesh.depths[i]
            
            # 深度依存の b 設定
            if depth <= 30.0:
                b[i] = self.b_seismogenic
            else:
                b[i] = self.b_stable
            
            # 有効法線応力 σ_eff
            sigma_eff[i] = self._get_sigma_eff(lon, lat, depth)
            
            # 特徴的すべり量 L
            L[i] = self._get_L(lon, lat, depth)
            
            # プレート収束速度 V_pl
            V_pl[i] = self._get_V_pl(lon)
        
        # メッシュに設定
        mesh.set_parameters(
            sigma_eff=sigma_eff,
            L=L,
            a=a,
            b=b,
            V_pl=V_pl
        )
        
        print(f"パラメータを設定: {n_cells} セル")
        print(f"  a-b range: [{(a-b).min():.4f}, {(a-b).max():.4f}]")
        print(f"  σ_eff range: [{sigma_eff.min()/1e6:.1f}, {sigma_eff.max()/1e6:.1f}] MPa")
        print(f"  L range: [{L.min():.2f}, {L.max():.1f}] m")
        print(f"  V_pl range: [{V_pl.min()*3.15576e7*100:.1f}, {V_pl.max()*3.15576e7*100:.1f}] cm/year")
    
    def _get_sigma_eff(self, lon: float, lat: float, depth: float) -> float:
        """有効法線応力を取得"""
        # 東海沖の海山（Point 1, 2）: 高い σ_eff
        if 137.5 <= lon <= 139.0 and 34.0 <= lat <= 35.0:
            if depth < 20:
                return 40.0e6
            else:
                return 35.0e6
        
        # 日向灘（Point 6）: 高い σ_eff（長い再発間隔のため）
        if 131.0 <= lon <= 132.5 and 31.0 <= lat <= 33.0:
            return 60.0e6
        
        # 四国沖の遠方（Point 5）: やや高い
        if 133.0 <= lon <= 134.5 and 32.0 <= lat <= 33.0:
            return 40.0e6
        
        # デフォルト
        return self.sigma_eff_default
    
    def _get_L(self, lon: float, lat: float, depth: float) -> float:
        """特徴的すべり量を取得"""
        # 潮岬沖のバリア: 大きな L
        if 135.3 <= lon <= 135.8 and 33.0 <= lat <= 34.0:
            return 7.5
        
        # 志摩半島沖のバリア
        if 136.5 <= lon <= 137.0 and 33.5 <= lat <= 34.2:
            return 5.0
        
        # 室戸岬南東（土佐碆）
        if 134.0 <= lon <= 134.8 and 32.5 <= lat <= 33.3:
            return 3.0
        
        # 足摺岬沖のバリア
        if 132.5 <= lon <= 133.2 and 32.0 <= lat <= 32.8:
            return 5.0
        
        # 東南海アスペリティ（Point 3）: 小さな L
        if 135.5 <= lon <= 137.5 and 33.5 <= lat <= 34.5:
            if depth <= 30:
                return 0.3
        
        # 南海アスペリティ（Point 4）: 小さな L
        if 133.5 <= lon <= 135.5 and 33.0 <= lat <= 34.0:
            if depth <= 30:
                return 0.4
        
        # LSSE 域（深部）: 小さな L
        if depth > 30:
            # 東海LSSE
            if 137.0 <= lon <= 138.5 and 34.5 <= lat <= 35.5:
                return 0.05
            # 豊後水道LSSE
            if 131.5 <= lon <= 132.5 and 32.5 <= lat <= 33.5:
                return 0.05
            # 紀伊水道LSSE
            if 134.5 <= lon <= 135.5 and 33.5 <= lat <= 34.5:
                return 0.08
            # 四国西部LSSE
            if 132.5 <= lon <= 133.5 and 33.0 <= lat <= 34.0:
                return 0.06
        
        # 周囲（階層的アスペリティモデル）
        return self.L_default
    
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
