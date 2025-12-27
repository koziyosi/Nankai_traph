"""
南海トラフ地震シミュレーション - リアル形状版
Nankai Trough Earthquake Simulation - Realistic Geometry

Hirose et al. (2008, 2022) に基づく現実的なプレート境界形状
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from numba import njit
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# 日本語表示用
plt.rcParams['font.family'] = ['MS Gothic', 'DejaVu Sans']

# 出力ディレクトリ
os.makedirs('results/earthquakes', exist_ok=True)

# ==============================================================================
# プレート境界の形状定義（Hirose et al. 2008 に基づく深度コンター）
# ==============================================================================
# 南海トラフの深度コンターポイント（緯度, 経度, 深度km）
# トラフ軸から40km深度まで

# トラフ軸（深度0km付近）- 西から東へ
TROUGH_AXIS = [
    (131.0, 31.2),   # 九州南部沖
    (132.0, 32.2),   # 日向灘沖
    (133.0, 32.8),   # 足摺岬沖
    (134.0, 33.0),   # 室戸岬沖
    (134.5, 33.2),   # 土佐碆沖
    (135.0, 33.5),   # 紀伊水道南
    (135.5, 33.5),   # 潮岬沖
    (136.0, 33.8),   # 紀伊半島南
    (137.0, 34.0),   # 志摩半島沖
    (138.0, 34.3),   # 渥美半島沖
    (139.0, 34.8),   # 駿河湾入口
]

# 深度10kmライン
DEPTH_10KM = [
    (131.5, 31.8),
    (132.3, 32.5),
    (133.2, 33.0),
    (134.0, 33.3),
    (134.8, 33.6),
    (135.5, 33.9),
    (136.2, 34.2),
    (137.2, 34.4),
    (138.0, 34.6),
    (138.5, 34.9),
]

# 深度20kmライン
DEPTH_20KM = [
    (131.8, 32.2),
    (132.5, 32.8),
    (133.3, 33.3),
    (134.2, 33.6),
    (134.9, 33.9),
    (135.6, 34.2),
    (136.5, 34.5),
    (137.5, 34.7),
    (138.3, 35.0),
]

# 深度30kmライン（発震層下限）
DEPTH_30KM = [
    (132.0, 32.5),
    (132.7, 33.0),
    (133.5, 33.5),
    (134.3, 33.8),
    (135.0, 34.1),
    (135.8, 34.4),
    (136.7, 34.7),
    (137.7, 34.9),
    (138.5, 35.2),
]

# ==============================================================================
# 各セグメントの定義（歴史地震の破壊域に基づく）
# ==============================================================================
SEGMENTS = {
    'Z': {  # 日向灘
        'name': '日向灘',
        'name_en': 'Hyuganada',
        'lon_range': (131.0, 132.5),
        'lat_range': (31.0, 33.0),
        'color': '#00CED1',  # Dark Turquoise
        'recurrence': 260,  # 約260年周期
    },
    'A': {  # 南海東部（土佐）
        'name': '南海(土佐)',
        'name_en': 'Nankai-Tosa',
        'lon_range': (132.5, 134.5),
        'lat_range': (32.5, 34.0),
        'color': '#FFD700',  # Gold
        'recurrence': 100,
    },
    'B': {  # 南海西部（紀伊水道）
        'name': '南海(紀伊水道)',
        'name_en': 'Nankai-Kii',
        'lon_range': (134.5, 135.5),
        'lat_range': (33.0, 34.5),
        'color': '#FFA500',  # Orange
        'recurrence': 100,
    },
    'C': {  # 東南海
        'name': '東南海',
        'name_en': 'Tonankai',
        'lon_range': (135.5, 137.5),
        'lat_range': (33.5, 35.0),
        'color': '#FF6347',  # Tomato
        'recurrence': 100,
    },
    'D': {  # 東海
        'name': '東海',
        'name_en': 'Tokai',
        'lon_range': (137.5, 139.0),
        'lat_range': (34.0, 35.5),
        'color': '#DC143C',  # Crimson
        'recurrence': 150,
    },
}

# 歴史地震の破壊パターン（6タイプ、Hirose et al. 2022 図5）
EVENT_TYPES = {
    'type1': {'name': 'A単独', 'segments': ['A'], 'color': 'purple'},
    'type2': {'name': 'AB連動', 'segments': ['A', 'B'], 'color': 'blue'},
    'type3': {'name': 'ABC連動', 'segments': ['A', 'B', 'C'], 'color': 'green'},
    'type4': {'name': 'CD連動', 'segments': ['C', 'D'], 'color': 'orange'},
    'type5': {'name': 'ABCD連動(宝永型)', 'segments': ['A', 'B', 'C', 'D'], 'color': 'red'},
    'type6': {'name': 'Z単独(日向灘)', 'segments': ['Z'], 'color': 'cyan'},
}

# ==============================================================================
# アスペリティ・バリア・LSSE域の定義（Hirose et al. 2022に基づく）
# ==============================================================================
# 主要アスペリティ（高固着、小さいL）- 歴史地震の主要破壊域に対応
ASPERITIES = {
    # 1944年東南海地震の主要破壊域（熊野灘沖）
    'kumano': {
        'name': '熊野灘沖アスペリティ',
        'name_en': 'Kumano-nada',
        'center': (136.8, 33.6),
        'radius': 0.7,
        'sigma_eff': 35e6,
        'L': 0.25,
        'color': '#FF4444',
    },
    # 1946年南海地震の主要破壊域（土佐湾沖）
    'tosa_bay': {
        'name': '土佐湾沖アスペリティ',
        'name_en': 'Tosa Bay',
        'center': (133.5, 32.8),
        'radius': 0.8,
        'sigma_eff': 35e6,
        'L': 0.3,
        'color': '#FF6666',
    },
    # 足摺岬沖（1707年宝永地震で破壊）
    'ashizuri': {
        'name': '足摺岬沖アスペリティ',
        'name_en': 'Ashizuri Cape',
        'center': (132.7, 32.5),
        'radius': 0.5,
        'sigma_eff': 40e6,
        'L': 0.35,
        'color': '#FF8888',
    },
    # 駿河湾沖（東海セグメント）
    'suruga_bay': {
        'name': '駿河湾沖アスペリティ',
        'name_en': 'Suruga Bay',
        'center': (138.5, 34.6),
        'radius': 0.5,
        'sigma_eff': 45e6,
        'L': 0.5,
        'color': '#FFAAAA',
    },
    # 日向灘（独立した地震活動域）
    'hyuganada': {
        'name': '日向灘アスペリティ',
        'name_en': 'Hyuganada',
        'center': (131.8, 31.8),
        'radius': 0.6,
        'sigma_eff': 55e6,
        'L': 1.5,
        'color': '#FFCCCC',
    },
}

# バリア（大きいL）- 破壊伝播を抑制する領域（Hirose 2022の"belt-shaped area"等）
BARRIERS = {
    # 潮岬沖バリア帯（東南海・南海の連動/分離を制御する最重要構造）
    'shiono_belt': {
        'name': '潮岬沖バリア帯',
        'name_en': 'Shiono Cape Belt',
        'center': (135.5, 33.4),
        'radius': 0.5,  # 帯状に広がる
        'L': 8.0,
        'color': '#9932CC',
    },
    # 紀伊水道バリア
    'kii_channel': {
        'name': '紀伊水道バリア',
        'name_en': 'Kii Channel',
        'center': (134.8, 33.8),
        'radius': 0.4,
        'L': 5.0,
        'color': '#BA55D3',
    },
    # 志摩半島沖バリア（東南海・東海の境界）
    'shima': {
        'name': '志摩半島沖バリア',
        'name_en': 'Shima Peninsula',
        'center': (137.0, 34.3),
        'radius': 0.4,
        'L': 6.0,
        'color': '#DA70D6',
    },
    # 土佐碆（紀伊水道沖）
    'tosa_bae': {
        'name': '土佐碆バリア',
        'name_en': 'Tosa-bae',
        'center': (134.3, 33.1),
        'radius': 0.3,
        'L': 4.0,
        'color': '#EE82EE',
    },
}

# 長期的スロースリップ域（LSSE）
LSSE_REGIONS = {
    'tokai': {
        'name': '東海LSSE',
        'center': (137.5, 35.0),
        'depth_range': (30, 40),
        'L': 0.05,
        'interval_years': 8,
    },
    'kii': {
        'name': '紀伊水道LSSE',
        'center': (135.0, 34.0),
        'depth_range': (30, 40),
        'L': 0.08,
        'interval_years': 7,
    },
    'shikoku_west': {
        'name': '四国西部LSSE',
        'center': (133.0, 33.5),
        'depth_range': (30, 40),
        'L': 0.06,
        'interval_years': 6,
    },
    'bungo': {
        'name': '豊後水道LSSE',
        'center': (132.0, 33.0),
        'depth_range': (30, 40),
        'L': 0.05,
        'interval_years': 7,
    },
}

# ==============================================================================
# 海山・海嶺（高法線応力領域）
# ==============================================================================
SEAMOUNTS = {
    'paleo_zenisu': {
        'name': '深部古銭洲海嶺',
        'center': (138.5, 34.5),
        'radius': 0.4,
        'sigma_eff': 35e6,
    },
    'zenisu': {
        'name': '古銭洲海嶺',
        'center': (138.0, 34.3),
        'radius': 0.3,
        'sigma_eff': 40e6,
    },
    'kyushu_palau': {
        'name': '九州・パラオ海嶺',
        'center': (131.5, 31.5),
        'radius': 0.5,
        'sigma_eff': 50e6,
    },
}


# ==============================================================================
# セルジオメトリ
# ==============================================================================
@dataclass
class Cell:
    """プレート境界セル"""
    id: int
    lon: float
    lat: float
    depth: float  # km
    area: float   # km²
    
    # パラメータ
    a: float = 0.005
    b: float = 0.008
    sigma_eff: float = 30e6  # Pa
    L: float = 0.4  # m
    V_pl: float = 5.0e-2 / (365.25*24*3600)  # m/s
    
    # 所属セグメント
    segment: str = ''
    

def create_realistic_mesh(n_along: int = 80, n_down: int = 10) -> List[Cell]:
    """
    現実的なプレート境界メッシュを生成
    
    Parameters:
        n_along: 走向方向のセル数
        n_down: ディップ方向のセル数
    
    Returns:
        cells: Cellオブジェクトのリスト
    """
    cells = []
    cell_id = 0
    
    # 経度範囲
    lon_min, lon_max = 131.0, 139.0
    lon_step = (lon_max - lon_min) / n_along
    
    for i in range(n_along):
        lon = lon_min + (i + 0.5) * lon_step
        
        # この経度でのトラフ軸の緯度を補間
        lat_trough = interpolate_contour(lon, TROUGH_AXIS)
        lat_30km = interpolate_contour(lon, DEPTH_30KM)
        
        if lat_trough is None or lat_30km is None:
            continue
        
        lat_step = (lat_30km - lat_trough) / n_down
        
        for j in range(n_down):
            lat = lat_trough + (j + 0.5) * lat_step
            
            # 深度を計算（線形補間）
            depth = 5 + j * 35 / n_down  # 5-40 km
            
            # セル面積（概算）
            area = lon_step * 111 * lat_step * 111 * np.cos(np.radians(lat))
            
            cell = Cell(
                id=cell_id,
                lon=lon,
                lat=lat,
                depth=depth,
                area=area,
            )
            
            # セグメント判定
            for seg_id, seg in SEGMENTS.items():
                if (seg['lon_range'][0] <= lon <= seg['lon_range'][1] and
                    seg['lat_range'][0] <= lat <= seg['lat_range'][1]):
                    cell.segment = seg_id
                    break
            
            # パラメータ設定
            set_cell_parameters(cell)
            
            cells.append(cell)
            cell_id += 1
    
    return cells


def interpolate_contour(lon: float, contour: List[Tuple[float, float]]) -> Optional[float]:
    """深度コンター上の緯度を補間"""
    for i in range(len(contour) - 1):
        lon1, lat1 = contour[i]
        lon2, lat2 = contour[i + 1]
        
        if lon1 <= lon <= lon2:
            t = (lon - lon1) / (lon2 - lon1)
            return lat1 + t * (lat2 - lat1)
    
    return None


def set_cell_parameters(cell: Cell):
    """セルのパラメータを設定"""
    # デフォルト値
    cell.a = 0.005
    cell.b = 0.008 if cell.depth <= 30 else 0.002
    cell.sigma_eff = 30e6
    cell.L = 0.4
    
    # プレート速度（西側が速い）
    lon_factor = (cell.lon - 131.0) / (139.0 - 131.0)
    cell.V_pl = 5.5e-2 * (1 - 0.8 * lon_factor) / (365.25 * 24 * 3600)
    
    # アスペリティ内か判定
    for asp_id, asp in ASPERITIES.items():
        dist = np.sqrt((cell.lon - asp['center'][0])**2 + 
                      (cell.lat - asp['center'][1])**2)
        if dist < asp['radius']:
            cell.sigma_eff = asp.get('sigma_eff', cell.sigma_eff)
            cell.L = asp.get('L', cell.L)
    
    # バリア内か判定
    for bar_id, bar in BARRIERS.items():
        dist = np.sqrt((cell.lon - bar['center'][0])**2 + 
                      (cell.lat - bar['center'][1])**2)
        if dist < bar['radius']:
            cell.L = bar['L']
    
    # 海山上か判定
    for sm_id, sm in SEAMOUNTS.items():
        dist = np.sqrt((cell.lon - sm['center'][0])**2 + 
                      (cell.lat - sm['center'][1])**2)
        if dist < sm['radius']:
            cell.sigma_eff = sm['sigma_eff']
    
    # LSSE域か判定
    for lsse_id, lsse in LSSE_REGIONS.items():
        dist = np.sqrt((cell.lon - lsse['center'][0])**2 + 
                      (cell.lat - lsse['center'][1])**2)
        if dist < 0.5 and lsse['depth_range'][0] <= cell.depth <= lsse['depth_range'][1]:
            cell.L = lsse['L']


# ==============================================================================
# polygon_data.json からのデータ読み込み
# ==============================================================================
def load_polygon_data(json_path: str = 'polygon_data.json') -> Optional[Dict]:
    """
    polygon_data.json からポリゴンデータを読み込む
    
    Args:
        json_path: JSONファイルのパス
    
    Returns:
        {
            'locked_zone': [(lon, lat, depth, desc), ...],
            'unlocked_zone': [(lon, lat, depth, desc), ...],
            'interior_points': [(lon, lat, depth, desc), ...]
        }
        読み込みに失敗した場合はNone
    """
    import json
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        result = {
            'locked_zone': [(p['lon'], p['lat'], p['depth'], p['desc']) 
                           for p in data.get('locked_zone', [])],
            'unlocked_zone': [(p['lon'], p['lat'], p['depth'], p['desc']) 
                             for p in data.get('unlocked_zone', [])],
            'interior_points': [(p['lon'], p['lat'], p['depth'], p['desc']) 
                               for p in data.get('interior_points', [])]
        }
        
        print(f"polygon_data.json を読み込みました: {json_path}")
        print(f"  固着域: {len(result['locked_zone'])} 点")
        print(f"  非固着域: {len(result['unlocked_zone'])} 点")
        print(f"  内部ポイント: {len(result['interior_points'])} 点")
        
        return result
        
    except FileNotFoundError:
        print(f"警告: {json_path} が見つかりません")
        return None
    except Exception as e:
        print(f"警告: {json_path} の読み込みエラー: {e}")
        return None


def create_mesh_from_polygon(
    polygon_data: Dict,
    n_along: int = 60,
    n_down: int = 8
) -> List[Cell]:
    """
    polygon_data.json のポリゴンデータからメッシュを生成
    
    固着域内のセルは地震発生しやすいパラメータ、
    非固着域内のセルは安定すべりパラメータを設定
    
    Args:
        polygon_data: load_polygon_data() で読み込んだデータ
        n_along: 走向方向のセル数
        n_down: ディップ方向のセル数
    
    Returns:
        Cellオブジェクトのリスト
    """
    from matplotlib.path import Path as MplPath
    from scipy.interpolate import griddata
    
    locked_zone = polygon_data['locked_zone']
    unlocked_zone = polygon_data['unlocked_zone']
    interior_points = polygon_data['interior_points']
    
    # ポリゴンパスを作成
    locked_coords = np.array([(p[0], p[1]) for p in locked_zone])
    unlocked_coords = np.array([(p[0], p[1]) for p in unlocked_zone])
    
    locked_path = MplPath(locked_coords) if len(locked_coords) > 2 else None
    unlocked_path = MplPath(unlocked_coords) if len(unlocked_coords) > 2 else None
    
    # 深度補間用のデータを準備
    all_coords = []
    all_depths = []
    for zone in [locked_zone, unlocked_zone, interior_points]:
        for p in zone:
            all_coords.append([p[0], p[1]])
            all_depths.append(p[2])
    all_coords = np.array(all_coords)
    all_depths = np.array(all_depths)
    
    # 経度・緯度の範囲を取得
    lon_min = np.min(all_coords[:, 0]) - 0.2
    lon_max = np.max(all_coords[:, 0]) + 0.2
    lat_min = np.min(all_coords[:, 1]) - 0.2
    lat_max = np.max(all_coords[:, 1]) + 0.2
    
    lon_step = (lon_max - lon_min) / n_along
    lat_step = (lat_max - lat_min) / n_down
    
    cells = []
    cell_id = 0
    
    for i in range(n_along):
        lon = lon_min + (i + 0.5) * lon_step
        
        for j in range(n_down):
            lat = lat_min + (j + 0.5) * lat_step
            point = np.array([[lon, lat]])
            
            # ポリゴン内判定
            in_locked = locked_path.contains_points(point)[0] if locked_path else False
            in_unlocked = unlocked_path.contains_points(point)[0] if unlocked_path else False
            
            # 少なくともどちらかのゾーンに含まれるセルのみ追加
            if not (in_locked or in_unlocked):
                continue
            
            # 深度を補間
            depth = griddata(all_coords, all_depths, point, method='linear')[0]
            if np.isnan(depth):
                depth = griddata(all_coords, all_depths, point, method='nearest')[0]
            
            # セル面積（概算）
            area = lon_step * 111 * lat_step * 111 * np.cos(np.radians(lat))
            
            cell = Cell(
                id=cell_id,
                lon=lon,
                lat=lat,
                depth=depth,
                area=area,
            )
            
            # セグメント判定
            for seg_id, seg in SEGMENTS.items():
                if (seg['lon_range'][0] <= lon <= seg['lon_range'][1] and
                    seg['lat_range'][0] <= lat <= seg['lat_range'][1]):
                    cell.segment = seg_id
                    break
            
            # ゾーンに応じたパラメータ設定
            if in_locked:
                # 固着域：地震発生しやすいパラメータ
                cell.a = 0.005
                cell.b = 0.008  # velocity weakening
                cell.sigma_eff = 30e6 + (depth - 20) * 0.5e6  # 深さに応じて増加
                cell.L = 0.3 + (depth - 10) * 0.01  # 浅いほど小さいL
            else:
                # 非固着域：安定すべりパラメータ
                cell.a = 0.005
                cell.b = 0.003  # velocity strengthening寄り
                cell.sigma_eff = 20e6
                cell.L = 0.08  # LSSE用の小さいL
            
            # プレート速度（西側が速い）
            lon_factor = (lon - 131.0) / (139.0 - 131.0)
            cell.V_pl = 5.5e-2 * (1 - 0.3 * lon_factor) / (365.25 * 24 * 3600)
            
            # アスペリティ・バリアの影響を上書きしない（ポリゴンデータ優先）
            # set_cell_parameters(cell)
            
            cells.append(cell)
            cell_id += 1
    
    print(f"ポリゴンデータからメッシュを生成: {len(cells)} セル")
    
    return cells


# ==============================================================================
# 可視化
# ==============================================================================
def plot_plate_geometry(cells: List[Cell], save_path: str = None):
    """プレート境界の形状を可視化（改良版）"""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # セルをプロット（固着度で色分け）
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])
    L_vals = np.array([c.L for c in cells])
    
    # 固着度（1/L が大きいほど固着が強い）
    coupling = 1.0 / np.maximum(L_vals, 0.01)
    coupling_norm = (coupling - coupling.min()) / (coupling.max() - coupling.min() + 1e-10)
    
    # カスタムカラーマップ（青=弱固着、赤=強固着）
    from matplotlib.colors import LinearSegmentedColormap
    coupling_cmap = LinearSegmentedColormap.from_list(
        'coupling', ['#3366CC', '#66CCFF', '#FFFF66', '#FF9933', '#CC3333']
    )
    
    sc = ax.scatter(lons, lats, c=coupling_norm, cmap=coupling_cmap, s=30, alpha=0.8, 
                   edgecolors='none', zorder=2)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('固着度 (1/L)', fontsize=11)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['弱', '中', '強'])
    
    # セグメント境界（薄い背景として）
    for seg_id, seg in SEGMENTS.items():
        rect = patches.Rectangle(
            (seg['lon_range'][0], seg['lat_range'][0]),
            seg['lon_range'][1] - seg['lon_range'][0],
            seg['lat_range'][1] - seg['lat_range'][0],
            linewidth=2, edgecolor=seg['color'], facecolor=seg['color'], 
            alpha=0.1, linestyle='-', zorder=1
        )
        ax.add_patch(rect)
        # セグメントラベル（上部に配置）
        x_mid = (seg['lon_range'][0] + seg['lon_range'][1]) / 2
        ax.text(x_mid, seg['lat_range'][1] + 0.15, f"{seg_id}: {seg['name']}", 
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=seg['color'], bbox=dict(boxstyle='round,pad=0.2', 
                facecolor='white', edgecolor=seg['color'], alpha=0.8))
    
    # アスペリティ（塗りつぶし付き円）
    asp_patches = []
    for asp_id, asp in ASPERITIES.items():
        color = asp.get('color', '#FF4444')
        circle = patches.Circle(asp['center'], asp['radius'], 
                               edgecolor=color, facecolor=color,
                               linewidth=2, alpha=0.3, zorder=3)
        ax.add_patch(circle)
        # 名前ラベル
        ax.text(asp['center'][0], asp['center'][1], asp['name'].replace('アスペリティ', ''),
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='darkred', zorder=4)
    
    # バリア（破線円）
    for bar_id, bar in BARRIERS.items():
        color = bar.get('color', '#9932CC')
        circle = patches.Circle(bar['center'], bar['radius'], 
                               edgecolor=color, facecolor=color,
                               linewidth=2.5, alpha=0.25, linestyle='--', zorder=3)
        ax.add_patch(circle)
        # 名前ラベル
        ax.text(bar['center'][0], bar['center'][1], bar['name'].replace('バリア', ''),
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='purple', zorder=4)
    
    # 凡例
    legend_elements = [
        patches.Patch(facecolor='#FF4444', edgecolor='#FF4444', alpha=0.5, 
                     label='アスペリティ（強固着域）'),
        patches.Patch(facecolor='#9932CC', edgecolor='#9932CC', alpha=0.4, 
                     linestyle='--', label='バリア（破壊停止域）'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlim(130.5, 139.5)
    ax.set_ylim(30.5, 36)
    ax.set_xlabel('経度 [°E]', fontsize=12)
    ax.set_ylabel('緯度 [°N]', fontsize=12)
    ax.set_title('南海トラフ プレート境界モデル\n固着度分布・アスペリティ・バリア（Hirose et al. 2022）', 
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, zorder=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"保存: {save_path}")
    
    plt.close()
    return fig


def plot_rupture_map(cells: List[Cell], slip: np.ndarray, 
                     eq_id: int, t_years: float, Mw: float,
                     regions: List[str], save_path: str):
    """地震の破壊域マップを作成"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # すべり量でセルを色付け
    lons = np.array([c.lon for c in cells])
    lats = np.array([c.lat for c in cells])
    
    # カラーマップ
    cmap = LinearSegmentedColormap.from_list(
        'slip', ['white', 'yellow', 'orange', 'red', 'darkred']
    )
    
    sc = ax.scatter(lons, lats, c=slip, cmap=cmap, s=50, 
                   vmin=0, vmax=max(10, np.max(slip)), alpha=0.8)
    plt.colorbar(sc, ax=ax, label='すべり量 [m]')
    
    # セグメント名
    for seg_id, seg in SEGMENTS.items():
        x_mid = (seg['lon_range'][0] + seg['lon_range'][1]) / 2
        if seg_id in regions:
            ax.axvline(x=seg['lon_range'][0], color=seg['color'], 
                      linestyle='--', alpha=0.5)
        ax.text(x_mid, 35.5, seg['name'], ha='center', fontsize=10, 
               color=seg['color'], fontweight='bold')
    
    # タイトル
    regions_str = ', '.join([SEGMENTS[r]['name'] for r in regions if r in SEGMENTS])
    ax.set_title(f'地震 #{eq_id}: t = {t_years:.1f} 年, Mw = {Mw:.1f}\n'
                 f'破壊領域: {regions_str}', fontsize=14)
    
    ax.set_xlim(130.5, 139.5)
    ax.set_ylim(30.5, 36)
    ax.set_xlabel('経度 [°E]')
    ax.set_ylabel('緯度 [°N]')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================================================================
# メイン（テスト用）
# ==============================================================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='南海トラフ プレート境界モデル')
    parser.add_argument('--nx', type=int, default=80, help='走向方向セル数')
    parser.add_argument('--ny', type=int, default=10, help='ディップ方向セル数')
    parser.add_argument('--use-polygon', action='store_true',
                        help='polygon_data.json からメッシュを生成')
    parser.add_argument('--polygon-data', type=str, default='polygon_data.json',
                        help='ポリゴンデータのJSONファイルパス')
    args = parser.parse_args()
    
    print("=" * 60)
    print("南海トラフ プレート境界モデル（現実形状版）")
    print("=" * 60)
    
    if args.use_polygon:
        # polygon_data.json からメッシュ生成
        print(f"\npolygon_data.json からメッシュ生成中... ({args.nx} x {args.ny})")
        polygon_data = load_polygon_data(args.polygon_data)
        if polygon_data:
            cells = create_mesh_from_polygon(polygon_data, n_along=args.nx, n_down=args.ny)
        else:
            print("警告: ポリゴンデータの読み込みに失敗。デフォルトメッシュを使用します。")
            cells = create_realistic_mesh(n_along=args.nx, n_down=args.ny)
    else:
        print(f"\nメッシュ生成中... ({args.nx} x {args.ny})")
        cells = create_realistic_mesh(n_along=args.nx, n_down=args.ny)
    
    print(f"生成されたセル数: {len(cells)}")
    
    # セグメント別集計
    seg_counts = {}
    for c in cells:
        seg_counts[c.segment] = seg_counts.get(c.segment, 0) + 1
    print("\nセグメント別セル数:")
    for seg_id, count in sorted(seg_counts.items()):
        name = SEGMENTS.get(seg_id, {}).get('name', '(境界外)')
        print(f"  {seg_id}: {name} = {count} セル")
    
    # 可視化
    print("\nプレート形状を可視化中...")
    plot_plate_geometry(cells, 'results/plate_geometry.png')
    
    print("\n完了！")

