import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import griddata
import re
import json
from pathlib import Path as FilePath

def dms_to_deg(dms_str):
    """
    度分秒形式 (35°41'40"N) を10進数 (35.6944...) に変換する関数
    """
    # 数字部分を抽出
    parts = re.split(r'[°\'"]', dms_str)
    if len(parts) >= 3:
        d = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return d + m/60 + s/3600
    return 0.0

def load_polygon_from_json(json_path='polygon_data.json'):
    """
    JSONファイルからポリゴンデータを読み込む
    polygon_editor.pyで編集したデータを使用可能にする
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        locked_zone = [(p['lon'], p['lat'], p['depth'], p['desc']) 
                       for p in data.get('locked_zone', [])]
        unlocked_zone = [(p['lon'], p['lat'], p['depth'], p['desc']) 
                         for p in data.get('unlocked_zone', [])]
        interior_points = [(p['lon'], p['lat'], p['depth'], p['desc']) 
                           for p in data.get('interior_points', [])]
        
        print(f"JSONから読み込み: {json_path}")
        print(f"  固着域: {len(locked_zone)}点")
        print(f"  非固着域: {len(unlocked_zone)}点")
        print(f"  内部ポイント: {len(interior_points)}点")
        
        return locked_zone, unlocked_zone, interior_points
    
    except FileNotFoundError:
        print(f"警告: {json_path}が見つかりません。デフォルトデータを使用します。")
        return None, None, None
    except json.JSONDecodeError:
        print(f"警告: {json_path}の解析エラー。デフォルトデータを使用します。")
        return None, None, None


def generate_nankai_depth_csv(use_json=True, json_path='polygon_data.json'):
    """
    南海トラフの深さポイントCSVを生成する
    
    Args:
        use_json: Trueの場合、polygon_data.jsonからデータを読み込む
        json_path: JSONファイルのパス
    """
    # ---------------------------------------------------------
    # 1. ポリゴンデータの読み込み（JSONまたはデフォルト）
    # ---------------------------------------------------------
    
    if use_json:
        json_locked, json_unlocked, json_interior = load_polygon_from_json(json_path)
        
        if json_locked is not None:
            # JSONから読み込んだデータを使用（すでに10進数形式）
            locked_zone_points = json_locked
            unlocked_zone_points = json_unlocked
            interior_points = json_interior
            use_dms_format = False
        else:
            # JSONが読めない場合はデフォルトデータを使用
            use_dms_format = True
    else:
        use_dms_format = True
    
    if use_dms_format:
        # ---------------------------------------------------------
        # デフォルトデータ（度分秒形式）
        # ---------------------------------------------------------
        locked_zone_points = [
            # ========== 内側ライン（陸側・固着域の北限）：西から東へ ==========
            # 日向灘から四国→近畿→中部方向へ（深部ポイントより南側）
            
            ("131°31'20\"E", "31°09'06\"N", 30, "日向灘 - 最西部（陸側）"),
            ("131°42'32\"E", "31°55'18\"N", 28, "日向灘 - 西部内陸"),
            ("132°06'32\"E", "32°31'43\"N", 25, "日向灘 - 西端固着域境界1"),
            ("132°26'59\"E", "32°38'46\"N", 22, "日向灘 - 西端固着域境界2"),
            ("132°47'08\"E", "33°38'04\"N", 28, "愛媛沖 - 固着域北限"),
            ("133°27'36\"E", "33°49'50\"N", 30, "南海 - 固着域北限"),
            ("134°32'53\"E", "34°04'52\"N", 35, "香川/徳島境界 - 固着域北限"),
            ("135°08'52\"E", "33°44'35\"N", 30, "和歌山沖 - 固着域北限"),
            ("136°17'48\"E", "34°27'09\"N", 40, "三重県南部 - 固着域北限"),
            ("136°56'35\"E", "34°45'00\"N", 42, "愛知県南部 - 固着域北限"),
            ("137°19'47\"E", "35°11'36\"N", 45, "岐阜県南部 - 固着域北限"),
            ("137°45'13\"E", "34°45'47\"N", 35, "静岡県真下 - 固着域北限"),
            ("138°07'43\"E", "34°47'32\"N", 30, "東海 - 北部固着域北限"),
            ("138°36'05\"E", "35°05'16\"N", 25, "静岡県 - 海岸線境界"),
            ("138°32'26\"E", "34°30'09\"N", 15, "東海 - 最東部"),
            ("138°21'08\"E", "34°00'46\"N", 10, "東海 - 最浅部"),
            ("137°28'46\"E", "33°40'19\"N", 12, "東海/東南海境界（海側）"),
            ("136°52'15\"E", "33°04'11\"N", 10, "東南海 - 中間点"),
            ("136°10'28\"E", "32°46'41\"N", 10, "南海/東南海境界"),
            ("135°13'33\"E", "32°38'53\"N", 10, "南海 - 屈曲点"),
            ("134°20'41\"E", "32°07'05\"N", 10, "南海 - 中央最浅部"),
            ("133°21'10\"E", "31°31'26\"N", 10, "南海/日向灘接合"),
            ("133°10'18\"E", "31°14'23\"N", 10, "日向灘 - 東側浅部1"),
            ("132°43'47\"E", "30°57'42\"N", 10, "日向灘 - 東側浅部2"),
            ("132°25'50\"E", "30°30'00\"N", 10, "日向灘 - 最南部"),
        ]
        
        unlocked_zone_points = [
            ("138°07'43\"E", "34°47'32\"N", 30, "東海 - 北部固着域北限"),
            ("137°45'13\"E", "34°45'47\"N", 35, "静岡県真下 - 固着域北限"),
            ("137°19'47\"E", "35°11'36\"N", 45, "岐阜県南部 - 固着域北限"),
            ("136°56'35\"E", "34°45'00\"N", 42, "愛知県南部 - 固着域北限"),
            ("136°17'48\"E", "34°27'09\"N", 40, "三重県南部 - 固着域北限"),
            ("135°08'52\"E", "33°44'35\"N", 30, "和歌山沖 - 固着域北限"),
            ("134°32'53\"E", "34°04'52\"N", 35, "香川/徳島境界 - 固着域北限"),
            ("133°27'36\"E", "33°49'50\"N", 30, "南海 - 固着域北限"),
            ("132°47'08\"E", "33°38'04\"N", 28, "愛媛沖 - 固着域北限"),
            ("132°26'59\"E", "32°38'46\"N", 22, "日向灘 - 西端固着域境界2"),
            ("132°06'32\"E", "32°31'43\"N", 25, "日向灘 - 西端固着域境界1"),
            ("131°42'32\"E", "31°55'18\"N", 28, "日向灘 - 西部内陸"),
            ("131°31'20\"E", "31°09'06\"N", 30, "日向灘 - 最西部（陸側）"),
            ("131°42'32\"E", "31°55'18\"N", 45, "日向灘 - 西部深部"),
            ("132°06'32\"E", "32°31'43\"N", 50, "日向灘 - 西端深部"),
            ("131°59'17\"E", "33°21'40\"N", 55, "大分県沖/愛媛県沖"),
            ("132°11'58\"E", "33°45'09\"N", 55, "山口県沖"),
            ("132°17'19\"E", "34°05'58\"N", 55, "非固着域北部1"),
            ("132°31'31\"E", "34°17'50\"N", 55, "非固着域北部2"),
            ("132°46'45\"E", "34°24'26\"N", 55, "非固着域北部3"),
            ("132°38'21\"E", "33°41'22\"N", 50, "愛媛県北部"),
            ("133°07'18\"E", "34°17'12\"N", 60, "海溝深部1"),
            ("133°55'41\"E", "34°28'00\"N", 60, "海溝深部2"),
            ("134°32'53\"E", "34°04'52\"N", 55, "香川/徳島境界"),
            ("134°37'44\"E", "34°33'58\"N", 55, "海溝深部3"),
            ("135°04'16\"E", "34°16'52\"N", 50, "海溝深部4"),
            ("135°31'40\"E", "34°09'45\"N", 55, "奈良県直下 - 非固着域"),
            ("136°02'11\"E", "35°27'20\"N", 65, "非固着域北側1"),
            ("136°22'58\"E", "35°40'48\"N", 68, "非固着域北側2"),
            ("136°50'26\"E", "35°19'36\"N", 62, "非固着域北側3"),
            ("136°06'51\"E", "35°01'39\"N", 60, "滋賀県 - 非固着域"),
            ("136°06'43\"E", "35°41'40\"N", 70, "福井県直下 - 最深部"),
            ("137°19'47\"E", "35°11'36\"N", 65, "岐阜県南部 - 非固着域"),
            ("138°28'06\"E", "35°39'36\"N", 60, "山梨県南部 - 非固着域"),
            ("138°36'05\"E", "35°05'16\"N", 45, "静岡県北部 - 非固着域"),
        ]
        
        interior_points = [
            ("137°28'41\"E", "34°14'36\"N", 20, "東海/東南海境界（内部）"),
            ("136°09'57\"E", "34°05'18\"N", 35, "東南海 - 三重県直下"),
            ("136°53'26\"E", "34°27'15\"N", 25, "東海/東南海境界1"),
            ("137°25'11\"E", "34°44'16\"N", 20, "東海/東南海境界2"),
            ("135°58'47\"E", "33°10'47\"N", 15, "南海/東南海境界"),
            ("133°04'07\"E", "31°52'28\"N", 10, "日向灘/南海間1"),
            ("132°43'01\"E", "32°16'09\"N", 10, "日向灘/南海間2"),
            ("135°45'25\"E", "33°24'45\"N", 25, "東海/東南海間1"),
            ("136°04'20\"E", "32°58'06\"N", 20, "東海/東南海間2"),
        ]

    # データ変換処理
    # ---------------------------------------------------------
    # 固着域ポリゴン用座標
    locked_coords = []
    locked_depths = []
    
    if use_dms_format:
        # DMS形式（度分秒）からの変換
        for lon_str, lat_str, d, desc in locked_zone_points:
            lon = dms_to_deg(lon_str)
            lat = dms_to_deg(lat_str)
            locked_coords.append([lon, lat])
            locked_depths.append(d)
    else:
        # JSON形式（すでに10進数）
        for lon, lat, d, desc in locked_zone_points:
            locked_coords.append([lon, lat])
            locked_depths.append(d)
    
    locked_coords = np.array(locked_coords)
    locked_depths = np.array(locked_depths)
    
    # ---------------------------------------------------------
    # 非固着域ポリゴン用座標
    unlocked_coords = []
    unlocked_depths = []
    
    if use_dms_format:
        for lon_str, lat_str, d, desc in unlocked_zone_points:
            lon = dms_to_deg(lon_str)
            lat = dms_to_deg(lat_str)
            unlocked_coords.append([lon, lat])
            unlocked_depths.append(d)
    else:
        for lon, lat, d, desc in unlocked_zone_points:
            unlocked_coords.append([lon, lat])
            unlocked_depths.append(d)
    
    unlocked_coords = np.array(unlocked_coords)
    unlocked_depths = np.array(unlocked_depths)
    
    # ---------------------------------------------------------
    # 補間用座標（固着域 + 非固着域 + 内部ポイント）
    all_coords = list(locked_coords) + list(unlocked_coords)
    all_depths = list(locked_depths) + list(unlocked_depths)
    
    if use_dms_format:
        for lon_str, lat_str, d, desc in interior_points:
            lon = dms_to_deg(lon_str)
            lat = dms_to_deg(lat_str)
            all_coords.append([lon, lat])
            all_depths.append(d)
    else:
        for lon, lat, d, desc in interior_points:
            all_coords.append([lon, lat])
            all_depths.append(d)
    
    all_coords = np.array(all_coords)
    all_depths = np.array(all_depths)

    # ---------------------------------------------------------
    # 2. グリッド生成 (10km間隔)
    # ---------------------------------------------------------
    # 範囲設定
    lat_min, lat_max = 30.0, 36.5
    lon_min, lon_max = 131.0, 139.5
    
    # 緯度0.09度≒10km, 経度0.11度≒10km
    lat_step = 0.09
    lon_step = 0.11
    
    grid_lats = np.arange(lat_min, lat_max, lat_step)
    grid_lons = np.arange(lon_min, lon_max, lon_step)
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)
    
    grid_points = np.vstack((lon_mesh.flatten(), lat_mesh.flatten())).T

    # ---------------------------------------------------------
    # 3. ポリゴン内判定（固着域・非固着域）
    # ---------------------------------------------------------
    locked_path = Path(locked_coords)
    unlocked_path = Path(unlocked_coords)
    
    # 固着域内のポイント
    locked_mask = locked_path.contains_points(grid_points)
    locked_points = grid_points[locked_mask]
    
    # 非固着域内のポイント（固着域と重複する場合は固着域優先）
    unlocked_mask = unlocked_path.contains_points(grid_points) & ~locked_mask
    unlocked_points = grid_points[unlocked_mask]
    
    # 全ターゲットポイント
    target_points = np.vstack([locked_points, unlocked_points]) if len(unlocked_points) > 0 else locked_points
    
    # ゾーン情報（固着域=1, 非固着域=0）
    zone_info = np.array([1] * len(locked_points) + [0] * len(unlocked_points))

    # ---------------------------------------------------------
    # 4. 深さの補間 (Interpolation)
    # ---------------------------------------------------------
    # 全ポイント(固着域+非固着域+内部)を使って深さを計算
    target_depths = griddata(all_coords, all_depths, target_points, method='linear')
    
    # 線形補間だと外側がNaNになることがあるので、nearest(最近傍)で埋める
    nan_mask = np.isnan(target_depths)
    if np.any(nan_mask):
        target_depths[nan_mask] = griddata(all_coords, all_depths, target_points[nan_mask], method='nearest')

    # ---------------------------------------------------------
    # 5. CSV出力
    # ---------------------------------------------------------
    df = pd.DataFrame({
        'latitude': target_points[:, 1],
        'longitude': target_points[:, 0],
        'depth_km': np.round(target_depths, 2),
        'zone': zone_info  # 1=固着域, 0=非固着域
    })
    
    output_filename = 'nankai_depth_points.csv'
    df.to_csv(output_filename, index=False)
    
    locked_count = np.sum(zone_info == 1)
    unlocked_count = np.sum(zone_info == 0)
    print(f"完了: {len(df)}個のデータを '{output_filename}' に保存しました。")
    print(f"  固着域: {locked_count}点, 非固着域: {unlocked_count}点")

    # ---------------------------------------------------------
    # 6. プロット表示 (確認用)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(18, 12))
    
    # 固着域ポリゴン（赤枠）
    ax.plot(np.append(locked_coords[:, 0], locked_coords[0, 0]), 
            np.append(locked_coords[:, 1], locked_coords[0, 1]), 
            'r-', linewidth=2, label='Locked Zone (Fixed)', zorder=3)
    ax.fill(locked_coords[:, 0], locked_coords[:, 1], 
            alpha=0.1, color='red', zorder=1)
    
    # 非固着域ポリゴン（青枠）
    ax.plot(np.append(unlocked_coords[:, 0], unlocked_coords[0, 0]), 
            np.append(unlocked_coords[:, 1], unlocked_coords[0, 1]), 
            'b--', linewidth=2, label='Unlocked Zone (Non-Fixed)', zorder=3)
    ax.fill(unlocked_coords[:, 0], unlocked_coords[:, 1], 
            alpha=0.1, color='blue', zorder=1)
    
    # 固着域ドット（暖色系）
    df_locked = df[df['zone'] == 1]
    sc1 = ax.scatter(df_locked['longitude'], df_locked['latitude'], 
                     c=df_locked['depth_km'], cmap='YlOrRd_r', s=20, alpha=0.9, 
                     vmin=10, vmax=70, zorder=2)
    
    # 非固着域ドット（寒色系）
    df_unlocked = df[df['zone'] == 0]
    if len(df_unlocked) > 0:
        ax.scatter(df_unlocked['longitude'], df_unlocked['latitude'], 
                   c=df_unlocked['depth_km'], cmap='Blues_r', s=20, alpha=0.7,
                   vmin=40, vmax=70, zorder=2, marker='s')
    
    # 固着域境界ポイント（赤丸）
    ax.scatter(locked_coords[:, 0], locked_coords[:, 1], 
               c='red', s=60, marker='o', edgecolors='white', linewidths=1.5, 
               zorder=5, label='Locked Zone Points')
    
    # 非固着域境界ポイント（青四角）
    ax.scatter(unlocked_coords[:, 0], unlocked_coords[:, 1], 
               c='blue', s=60, marker='s', edgecolors='white', linewidths=1.5, 
               zorder=5, label='Unlocked Zone Points')
    
    # 内部ポイント（緑ダイヤ）
    # 内部ポイント（緑ダイヤ）
    for point in interior_points:
        if use_dms_format:
            lon = dms_to_deg(point[0])
            lat = dms_to_deg(point[1])
        else:
            lon = point[0]
            lat = point[1]
        ax.scatter(lon, lat, c='green', s=40, marker='D', edgecolors='white', 
                   linewidths=1, zorder=4)
    
    plt.colorbar(sc1, ax=ax, label='Depth (km) - Locked Zone', shrink=0.6)
    ax.set_title(f'Nankai Trough: Locked Zone (N={locked_count}) + Unlocked Zone (N={unlocked_count})\n'
                 f'Reference: MRI-JMA Hirose Model', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_nankai_depth_csv()