import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path

def generate_nankai_csv():
    """
    南海トラフの固着域および周辺領域を模したドットデータを生成し、
    CSVファイルとして保存する関数。
    目標データ数: 約1500個
    間隔: おおよそ10km
    """
    
    # 1. 領域の定義（ポリゴンの頂点座標: 経度, 緯度）
    # 気象庁気象研究所のモデル図などを参考に、日向灘〜四国沖〜東海沖の
    # プレート境界域（固着域＋周辺のスロースリップ域等）を近似
    polygon_points = [
        (131.4, 31.2), # 日向灘 南端
        (132.5, 32.0),
        (134.0, 32.6), # 四国沖 トラフ軸付近
        (136.0, 33.2), # 紀伊半島沖
        (137.5, 33.8), # 東海沖
        (138.6, 34.6), # 駿河湾入り口
        (138.8, 35.2), # 駿河湾奥（富士川河口付近）
        
        # ここから陸側（深部境界）へ折り返し
        (138.2, 35.8), # 長野・山梨県境付近
        (137.0, 35.3), # 愛知・岐阜付近
        (135.5, 34.5), # 大阪・和歌山北部
        (133.5, 33.8), # 高知北部
        (132.0, 33.0), # 大分・愛媛海峡付近
        (131.2, 32.2), # 九州内陸
        (131.4, 31.2)  # 始点に戻る
    ]
    
    # 2. グリッドの生成
    # 10km間隔の近似値: 緯度方向 約0.09度, 経度方向 約0.11度
    lat_min, lat_max = 30.0, 36.0
    lon_min, lon_max = 130.0, 140.0
    
    lat_step = 0.09
    lon_step = 0.11
    
    lats = np.arange(lat_min, lat_max, lat_step)
    lons = np.arange(lon_min, lon_max, lon_step)
    
    # メッシュグリッド作成
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    
    # 3. ポリゴン内の点を抽出
    path = Path(polygon_points)
    mask = path.contains_points(points)
    nankai_points = points[mask]
    
    # データフレーム化
    df = pd.DataFrame(nankai_points, columns=['longitude', 'latitude'])
    
    # 緯度・経度の並び順をユーザー指定（緯度, 経度 の順など）がある場合はここで変更可能
    # 今回は一般的な csv形式として latitude, longitude の順に入れ替えて保存
    df = df[['latitude', 'longitude']]
    
    # 4. CSV出力
    output_filename = 'nankai_trough_points.csv'
    df.to_csv(output_filename, index=False)
    
    print(f"生成完了: {len(df)}個のドットデータを '{output_filename}' に保存しました。")
    
    # 5. 結果のプロット表示（確認用）
    plt.figure(figsize=(10, 6))
    
    # ポリゴン枠線
    poly_arr = np.array(polygon_points)
    plt.plot(poly_arr[:, 0], poly_arr[:, 1], 'r--', label='Boundary Area')
    
    # 生成されたドット
    plt.scatter(df['longitude'], df['latitude'], s=5, c='blue', alpha=0.6, label='Simulation Dots')
    
    plt.title(f'Nankai Trough Simulation Dots (N={len(df)})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # アスペクト比を揃える
    plt.show()

if __name__ == "__main__":
    generate_nankai_csv()