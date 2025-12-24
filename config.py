"""
設定パラメータ - Configuration Parameters
Based on Hirose et al. (2022) Earth, Planets and Space
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class PhysicalConstants:
    """物理定数"""
    G: float = 30.0e9          # 剛性率 [Pa] (30 GPa)
    nu: float = 0.25           # ポアソン比
    beta: float = 3750.0       # S波速度 [m/s] (3.75 km/s)
    eta: float = 1.0           # 準動的補正係数
    rho_rock: float = 2800.0   # 岩石密度 [kg/m³]
    g: float = 9.8             # 重力加速度 [m/s²]


@dataclass
class FrictionParameters:
    """摩擦パラメータ"""
    mu_0: float = 0.6              # 定常摩擦係数
    V_0: float = 1.0e-6            # 参照速度 [m/s]
    V_c: float = 1.0e-8            # カットオフ速度 (Composite law) [m/s]
    a: float = 0.005               # 直接効果パラメータ
    b_seismogenic: float = 0.008   # 地震発生層 (0-30km) での b
    b_stable: float = 0.002        # 安定すべり層 (>30km) での b
    
    # 空間分布用のデフォルト値
    sigma_eff_default: float = 30.0e6   # 有効法線応力 [Pa] (30 MPa)
    L_default: float = 0.4              # 特徴的すべり量 [m]
    
    # 地震判定閾値
    V_earthquake: float = 0.1      # 地震時のすべり速度閾値 [m/s]


@dataclass
class GeometryParameters:
    """ジオメトリパラメータ"""
    # モデル領域 (経度, 緯度)
    lon_min: float = 131.0
    lon_max: float = 140.0
    lat_min: float = 30.5
    lat_max: float = 35.5
    
    # 深度範囲
    depth_min: float = 0.0       # [km]
    depth_max: float = 40.0      # [km]
    depth_seismogenic: float = 30.0  # 地震発生層下限 [km]
    
    # メッシュ解像度
    cell_size: float = 10.0      # セルサイズ [km] (論文では~5km、高速化のため10km)
    
    # プレート収束方向 (N60°W = 330°)
    convergence_direction: float = 330.0  # [degrees from N]
    
    # プレート収束速度 [m/year]
    V_pl_west: float = 0.055     # 西部 (5.5 cm/y)
    V_pl_east: float = 0.010     # 東端 (1.0 cm/y)


@dataclass
class SolverParameters:
    """ソルバーパラメータ"""
    # 時間積分
    dt_initial: float = 1.0e6       # 初期時間ステップ [s] (~11.6日)
    dt_min: float = 0.01            # 最小時間ステップ [s]
    dt_max: float = 1.0e8           # 最大時間ステップ [s] (~3年)
    
    # RK45トレランス
    rtol: float = 1.0e-6
    atol: float = 1.0e-9
    
    # シミュレーション期間
    t_years: float = 2000.0         # シミュレーション年数
    t_spinup_years: float = 500.0   # スピンアップ期間（解析から除外）
    
    # 出力間隔
    output_interval_years: float = 1.0
    
    # 並列化設定
    use_parallel: bool = True
    use_gpu: bool = False           # GPU (CuPy) を使用するか
    n_threads: int = -1             # -1 = 自動検出


@dataclass
class RegionDefinition:
    """領域定義（セグメント）"""
    name: str
    lon_range: Tuple[float, float]
    lat_range: Tuple[float, float]
    depth_range: Tuple[float, float] = (0.0, 40.0)
    sigma_eff: Optional[float] = None
    L: Optional[float] = None


# 主要セグメントの定義
SEGMENTS = {
    'tokai': RegionDefinition(
        name='東海',
        lon_range=(137.5, 139.5),
        lat_range=(34.0, 35.5),
        sigma_eff=35.0e6,
        L=1.0
    ),
    'tonankai': RegionDefinition(
        name='東南海',
        lon_range=(135.5, 137.5),
        lat_range=(33.0, 34.5),
        sigma_eff=30.0e6,
        L=0.3
    ),
    'nankai': RegionDefinition(
        name='南海',
        lon_range=(132.5, 135.5),
        lat_range=(32.0, 34.0),
        sigma_eff=30.0e6,
        L=0.4
    ),
    'hyuganada': RegionDefinition(
        name='日向灘',
        lon_range=(131.0, 132.5),
        lat_range=(31.0, 33.0),
        sigma_eff=60.0e6,
        L=2.0
    ),
    'shiono_barrier': RegionDefinition(
        name='潮岬バリア',
        lon_range=(135.3, 135.8),
        lat_range=(33.0, 34.0),
        L=7.5  # 大きなLでバリア効果
    ),
}

# LSSE発生域
LSSE_REGIONS = {
    'tokai_lsse': RegionDefinition(
        name='東海LSSE',
        lon_range=(137.0, 138.5),
        lat_range=(34.5, 35.5),
        depth_range=(30.0, 40.0),
        L=0.05
    ),
    'bungo_lsse': RegionDefinition(
        name='豊後水道LSSE',
        lon_range=(131.5, 132.5),
        lat_range=(32.5, 33.5),
        depth_range=(30.0, 40.0),
        L=0.05
    ),
}


@dataclass 
class SimulationConfig:
    """シミュレーション全体の設定"""
    physical: PhysicalConstants = field(default_factory=PhysicalConstants)
    friction: FrictionParameters = field(default_factory=FrictionParameters)
    geometry: GeometryParameters = field(default_factory=GeometryParameters)
    solver: SolverParameters = field(default_factory=SolverParameters)
    
    # 出力設定
    output_dir: str = "results"
    save_full_history: bool = False  # メモリ節約のため False
    save_events_only: bool = True
    
    @classmethod
    def default(cls) -> 'SimulationConfig':
        """デフォルト設定を返す"""
        return cls()
    
    @classmethod
    def high_resolution(cls) -> 'SimulationConfig':
        """高解像度設定（計算量大）"""
        config = cls()
        config.geometry.cell_size = 5.0
        config.solver.t_years = 5500.0
        return config
    
    @classmethod
    def fast_test(cls) -> 'SimulationConfig':
        """高速テスト用設定"""
        config = cls()
        config.geometry.cell_size = 20.0
        config.solver.t_years = 500.0
        config.solver.rtol = 1.0e-4
        return config


# シミュレーション定数
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # 1年の秒数
M_TO_KM = 1.0e-3
KM_TO_M = 1.0e3
PA_TO_MPA = 1.0e-6
MPA_TO_PA = 1.0e6
