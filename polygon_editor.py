"""
南海トラフ ポリゴンエディター
ポイントをドラッグで移動、右クリックで追加/削除できるインタラクティブなエディター
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import json
import re
from pathlib import Path

class PolygonEditor:
    """インタラクティブなポリゴンエディター"""
    
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        # ポリゴンデータ
        self.locked_points = []  # 固着域ポイント [(lon, lat, depth, desc), ...]
        self.unlocked_points = []  # 非固着域ポイント
        self.interior_points = []  # 内部ポイント
        
        # 選択状態
        self.selected_polygon = None  # 'locked', 'unlocked', 'interior'
        self.selected_idx = None
        self.dragging = False
        
        # プロット要素
        self.locked_line = None
        self.unlocked_line = None
        self.locked_scatter = None
        self.unlocked_scatter = None
        self.interior_scatter = None
        
        # 編集モード
        self.edit_mode = 'locked'  # 'locked', 'unlocked', 'interior'
        
        # Undo/Redo履歴
        self.history = []  # 過去の状態
        self.redo_stack = []  # やり直し用スタック
        self.max_history = 50  # 最大履歴数
        
        # イベント接続
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.setup_ui()
    
    def setup_ui(self):
        """UIセットアップ"""
        self.ax.set_xlim(131, 140)
        self.ax.set_ylim(30, 36.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.update_title()
        
        # 操作説明（英語表示で文字化け回避）
        instructions = """Controls:
  Left Click+Drag: Move point
  Right Click: Add point
  Delete/Backspace: Delete point
  Arrow Keys: Reorder point
  1: Locked Zone mode
  2: Unlocked Zone mode  
  3: Interior Points mode
  Ctrl+Z: Undo
  Ctrl+Y: Redo
  S: Save JSON
  L: Load JSON
  Q: Quit
"""
        self.ax.text(0.02, 0.98, instructions, transform=self.ax.transAxes,
                     fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontfamily='monospace')
    
    def update_title(self):
        """タイトル更新"""
        mode_names = {'locked': 'Locked Zone', 'unlocked': 'Unlocked Zone', 'interior': 'Interior Points'}
        self.ax.set_title(f'Nankai Trough Polygon Editor - Mode: {mode_names[self.edit_mode]}\n'
                          f'Locked: {len(self.locked_points)}, Unlocked: {len(self.unlocked_points)}, '
                          f'Interior: {len(self.interior_points)}')
    
    def dms_to_deg(self, dms_str):
        """度分秒形式を10進数に変換"""
        parts = re.split(r'[°\'\"]', dms_str)
        if len(parts) >= 3:
            d = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return d + m/60 + s/3600
        return 0.0
    
    def deg_to_dms(self, deg, is_lat=True):
        """10進数を度分秒形式に変換"""
        d = int(deg)
        m = int((deg - d) * 60)
        s = (deg - d - m/60) * 3600
        direction = 'N' if is_lat else 'E'
        return f"{d}°{m:02d}'{s:05.2f}\"{direction}"
    
    def load_from_dot2(self):
        """dot_2.pyからデータを読み込む"""
        # デフォルトデータ
        self.locked_points = [
            (131.522, 31.152, 30, "日向灘 - 最西部"),
            (131.709, 31.922, 28, "日向灘 - 西部内陸"),
            (132.109, 32.529, 25, "日向灘 - 西端1"),
            (132.450, 32.646, 22, "日向灘 - 西端2"),
            (132.785, 33.634, 28, "愛媛沖"),
            (133.460, 33.831, 30, "南海"),
            (134.548, 34.081, 35, "香川/徳島"),
            (135.148, 33.743, 30, "和歌山沖"),
            (136.296, 34.452, 40, "三重県南部"),
            (136.943, 34.750, 42, "愛知県南部"),
            (137.330, 35.193, 45, "岐阜県南部"),
            (137.753, 34.763, 35, "静岡県真下"),
            (138.129, 34.792, 30, "東海北部"),
            (138.601, 35.088, 25, "静岡県"),
            (138.540, 34.502, 15, "東海最東部"),
            (138.353, 34.013, 10, "東海最浅部"),
            (137.479, 33.672, 12, "東海/東南海"),
            (136.870, 33.070, 10, "東南海中間"),
            (136.174, 32.778, 10, "南海/東南海"),
            (135.226, 32.648, 10, "南海屈曲"),
            (134.345, 32.118, 10, "南海中央"),
            (133.353, 31.524, 10, "南海/日向灘"),
            (133.172, 31.240, 10, "日向灘東1"),
            (132.730, 30.962, 10, "日向灘東2"),
            (132.431, 30.500, 10, "日向灘最南部"),
        ]
        
        self.unlocked_points = [
            (138.129, 34.792, 30, "東海北部共有"),
            (137.753, 34.763, 35, "静岡共有"),
            (137.330, 35.193, 45, "岐阜共有"),
            (136.943, 34.750, 42, "愛知共有"),
            (136.296, 34.452, 40, "三重共有"),
            (135.148, 33.743, 30, "和歌山共有"),
            (134.548, 34.081, 35, "香川共有"),
            (133.460, 33.831, 30, "南海共有"),
            (132.785, 33.634, 28, "愛媛共有"),
            (132.450, 32.646, 22, "日向灘共有2"),
            (132.109, 32.529, 25, "日向灘共有1"),
            (131.709, 31.922, 28, "日向灘西部"),
            (131.522, 31.152, 30, "日向灘最西部"),
            (131.709, 31.922, 45, "日向灘深部"),
            (132.109, 32.529, 50, "日向灘西端深部"),
            (131.987, 33.361, 55, "大分県沖"),
            (132.199, 33.753, 55, "山口県沖"),
            (132.287, 34.099, 55, "非固着北部1"),
            (132.526, 34.297, 55, "非固着北部2"),
            (132.779, 34.407, 55, "非固着北部3"),
            (132.639, 33.689, 50, "愛媛県北部"),
            (133.122, 34.287, 60, "海溝深部1"),
            (133.928, 34.467, 60, "海溝深部2"),
            (134.548, 34.081, 55, "香川/徳島"),
            (134.629, 34.566, 55, "海溝深部3"),
            (135.071, 34.282, 50, "海溝深部4"),
            (135.528, 34.163, 55, "奈良県直下"),
            (136.037, 35.456, 65, "非固着北側1"),
            (136.383, 35.680, 68, "非固着北側2"),
            (136.841, 35.327, 62, "非固着北側3"),
            (136.114, 35.027, 60, "滋賀県"),
            (136.112, 35.694, 70, "福井県直下"),
            (137.330, 35.193, 65, "岐阜深部"),
            (138.468, 35.660, 60, "山梨県南部"),
            (138.601, 35.088, 45, "静岡県北部"),
        ]
        
        self.interior_points = [
            (137.473, 34.243, 20, "東海/東南海内部"),
            (136.166, 34.088, 35, "東南海三重"),
            (136.891, 34.454, 25, "東海/東南海1"),
            (137.420, 34.738, 20, "東海/東南海2"),
            (135.980, 33.180, 15, "南海/東南海"),
            (133.068, 31.875, 10, "日向灘/南海間1"),
            (132.717, 32.269, 10, "日向灘/南海間2"),
            (135.757, 33.413, 25, "東海/東南海間1"),
            (136.072, 32.968, 20, "東海/東南海間2"),
        ]
    
    def save_state(self):
        """現在の状態を履歴に保存"""
        import copy
        state = {
            'locked': copy.deepcopy(self.locked_points),
            'unlocked': copy.deepcopy(self.unlocked_points),
            'interior': copy.deepcopy(self.interior_points)
        }
        self.history.append(state)
        
        # 履歴が最大を超えたら古いものを削除
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 新しい操作をしたらRedo履歴をクリア
        self.redo_stack.clear()
    
    def undo(self):
        """元に戻す (Undo)"""
        if len(self.history) < 2:
            print("これ以上戻れません")
            return
        
        import copy
        # 現在の状態をRedo用に保存
        current_state = {
            'locked': copy.deepcopy(self.locked_points),
            'unlocked': copy.deepcopy(self.unlocked_points),
            'interior': copy.deepcopy(self.interior_points)
        }
        self.redo_stack.append(current_state)
        
        # 履歴から一つ前の状態を取り出す
        self.history.pop()  # 現在の状態を削除
        prev_state = self.history[-1]  # 一つ前の状態
        
        self.locked_points = copy.deepcopy(prev_state['locked'])
        self.unlocked_points = copy.deepcopy(prev_state['unlocked'])
        self.interior_points = copy.deepcopy(prev_state['interior'])
        
        self.selected_idx = None
        self.draw_polygons()
        print(f"Undo実行 (残り履歴: {len(self.history)-1}件)")
    
    def redo(self):
        """やり直し (Redo)"""
        if not self.redo_stack:
            print("やり直しできる操作がありません")
            return
        
        import copy
        # Redoスタックから状態を取り出す
        next_state = self.redo_stack.pop()
        
        # 現在の状態を履歴に追加
        current_state = {
            'locked': copy.deepcopy(self.locked_points),
            'unlocked': copy.deepcopy(self.unlocked_points),
            'interior': copy.deepcopy(self.interior_points)
        }
        self.history.append(current_state)
        
        self.locked_points = copy.deepcopy(next_state['locked'])
        self.unlocked_points = copy.deepcopy(next_state['unlocked'])
        self.interior_points = copy.deepcopy(next_state['interior'])
        
        self.selected_idx = None
        self.draw_polygons()
        print(f"Redo実行 (残りRedo: {len(self.redo_stack)}件)")
    
    def draw_polygons(self):
        """ポリゴンを描画"""
        self.ax.clear()
        self.setup_ui()
        
        # 固着域
        if len(self.locked_points) > 2:
            lons = [p[0] for p in self.locked_points]
            lats = [p[1] for p in self.locked_points]
            self.ax.fill(lons + [lons[0]], lats + [lats[0]], 
                        alpha=0.2, color='red', zorder=1)
            self.ax.plot(lons + [lons[0]], lats + [lats[0]], 
                        'r-', linewidth=2, label='Locked Zone', zorder=2)
            self.locked_scatter = self.ax.scatter(lons, lats, 
                        c='red', s=80 if self.edit_mode == 'locked' else 40, 
                        marker='o', edgecolors='white', linewidths=1.5, zorder=5)
        
        # 非固着域
        if len(self.unlocked_points) > 2:
            lons = [p[0] for p in self.unlocked_points]
            lats = [p[1] for p in self.unlocked_points]
            self.ax.fill(lons + [lons[0]], lats + [lats[0]], 
                        alpha=0.2, color='blue', zorder=1)
            self.ax.plot(lons + [lons[0]], lats + [lats[0]], 
                        'b--', linewidth=2, label='Unlocked Zone', zorder=2)
            self.unlocked_scatter = self.ax.scatter(lons, lats, 
                        c='blue', s=80 if self.edit_mode == 'unlocked' else 40, 
                        marker='s', edgecolors='white', linewidths=1.5, zorder=5)
        
        # 内部ポイント
        if len(self.interior_points) > 0:
            lons = [p[0] for p in self.interior_points]
            lats = [p[1] for p in self.interior_points]
            self.interior_scatter = self.ax.scatter(lons, lats, 
                        c='green', s=80 if self.edit_mode == 'interior' else 40, 
                        marker='D', edgecolors='white', linewidths=1.5, 
                        zorder=4, label='Interior Points')
        
        # 選択ポイントのハイライト
        if self.selected_idx is not None:
            points = self.get_current_points()
            if 0 <= self.selected_idx < len(points):
                p = points[self.selected_idx]
                self.ax.scatter([p[0]], [p[1]], c='yellow', s=200, 
                               marker='*', edgecolors='black', linewidths=2, zorder=10)
        
        self.ax.legend(loc='upper right')
        self.update_title()
        self.fig.canvas.draw()
    
    def get_current_points(self):
        """現在の編集モードのポイントリストを取得"""
        if self.edit_mode == 'locked':
            return self.locked_points
        elif self.edit_mode == 'unlocked':
            return self.unlocked_points
        else:
            return self.interior_points
    
    def set_current_points(self, points):
        """現在の編集モードのポイントリストを設定"""
        if self.edit_mode == 'locked':
            self.locked_points = points
        elif self.edit_mode == 'unlocked':
            self.unlocked_points = points
        else:
            self.interior_points = points
    
    def find_nearest_point(self, x, y):
        """最も近いポイントを検索"""
        points = self.get_current_points()
        if not points:
            return None
        
        min_dist = float('inf')
        min_idx = None
        
        for i, p in enumerate(points):
            dist = np.sqrt((p[0] - x)**2 + (p[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        
        # しきい値（画面座標で約10ピクセル相当）
        if min_dist < 0.2:
            return min_idx
        return None
    
    def on_click(self, event):
        """クリックイベント"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # 左クリック
            idx = self.find_nearest_point(event.xdata, event.ydata)
            if idx is not None:
                self.save_state()  # ドラッグ開始時に状態を保存
                self.selected_idx = idx
                self.dragging = True
                self.draw_polygons()
        
        elif event.button == 3:  # 右クリック - ポイント追加
            points = self.get_current_points()
            depth = 30  # デフォルト深度
            desc = f"新規ポイント_{len(points)+1}"
            
            # 最適な挿入位置を探す
            if len(points) >= 2:
                min_dist = float('inf')
                insert_idx = len(points)
                
                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i + 1) % len(points)]
                    
                    # 線分とクリック位置の距離
                    dist = self.point_to_line_distance(
                        event.xdata, event.ydata, p1[0], p1[1], p2[0], p2[1])
                    
                    if dist < min_dist:
                        min_dist = dist
                        insert_idx = i + 1
                
                self.save_state()  # ポイント追加前に状態を保存
                points.insert(insert_idx, (event.xdata, event.ydata, depth, desc))
            else:
                self.save_state()  # ポイント追加前に状態を保存
                points.append((event.xdata, event.ydata, depth, desc))
            
            self.set_current_points(points)
            self.draw_polygons()
            print(f"ポイント追加: ({event.xdata:.4f}, {event.ydata:.4f})")
    
    def point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """点と線分の距離"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def on_release(self, event):
        """リリースイベント"""
        self.dragging = False
    
    def on_motion(self, event):
        """マウス移動イベント"""
        if not self.dragging or event.inaxes != self.ax:
            return
        
        if self.selected_idx is not None:
            points = self.get_current_points()
            if 0 <= self.selected_idx < len(points):
                old = points[self.selected_idx]
                points[self.selected_idx] = (event.xdata, event.ydata, old[2], old[3])
                self.set_current_points(points)
                self.draw_polygons()
    
    def on_key(self, event):
        """キーイベント"""
        if event.key == '1':
            self.edit_mode = 'locked'
            self.selected_idx = None
            self.draw_polygons()
            print("編集モード: 固着域")
        
        elif event.key == '2':
            self.edit_mode = 'unlocked'
            self.selected_idx = None
            self.draw_polygons()
            print("編集モード: 非固着域")
        
        elif event.key == '3':
            self.edit_mode = 'interior'
            self.selected_idx = None
            self.draw_polygons()
            print("編集モード: 内部ポイント")
        
        elif event.key in ['delete', 'backspace']:
            if self.selected_idx is not None:
                points = self.get_current_points()
                if 0 <= self.selected_idx < len(points):
                    self.save_state()  # 削除前に状態を保存
                    removed = points.pop(self.selected_idx)
                    self.set_current_points(points)
                    self.selected_idx = None
                    self.draw_polygons()
                    print(f"ポイント削除: {removed[3]}")
        
        elif event.key == 'ctrl+z':
            self.undo()
        
        elif event.key == 'ctrl+y':
            self.redo()
        
        # 矢印キーでポイントの順序を変更
        elif event.key in ['up', 'left']:
            if self.selected_idx is not None:
                points = self.get_current_points()
                if len(points) > 1 and self.selected_idx > 0:
                    self.save_state()
                    # 一つ前と入れ替え
                    points[self.selected_idx], points[self.selected_idx - 1] = \
                        points[self.selected_idx - 1], points[self.selected_idx]
                    self.set_current_points(points)
                    self.selected_idx -= 1
                    self.draw_polygons()
                    print(f"Point moved up (index: {self.selected_idx})")
        
        elif event.key in ['down', 'right']:
            if self.selected_idx is not None:
                points = self.get_current_points()
                if len(points) > 1 and self.selected_idx < len(points) - 1:
                    self.save_state()
                    # 一つ後と入れ替え
                    points[self.selected_idx], points[self.selected_idx + 1] = \
                        points[self.selected_idx + 1], points[self.selected_idx]
                    self.set_current_points(points)
                    self.selected_idx += 1
                    self.draw_polygons()
                    print(f"Point moved down (index: {self.selected_idx})")
        
        elif event.key == 's':
            self.save_to_json()
        
        elif event.key == 'l':
            self.load_from_json()
        
        elif event.key == 'q':
            plt.close(self.fig)
    
    def save_to_json(self, filename='polygon_data.json'):
        """JSONファイルに保存"""
        data = {
            'locked_zone': [
                {'lon': p[0], 'lat': p[1], 'depth': p[2], 'desc': p[3]}
                for p in self.locked_points
            ],
            'unlocked_zone': [
                {'lon': p[0], 'lat': p[1], 'depth': p[2], 'desc': p[3]}
                for p in self.unlocked_points
            ],
            'interior_points': [
                {'lon': p[0], 'lat': p[1], 'depth': p[2], 'desc': p[3]}
                for p in self.interior_points
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"保存しました: {filename}")
        print(f"  固着域: {len(self.locked_points)}点")
        print(f"  非固着域: {len(self.unlocked_points)}点")
        print(f"  内部ポイント: {len(self.interior_points)}点")
    
    def load_from_json(self, filename='polygon_data.json'):
        """JSONファイルから読み込み"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.locked_points = [
                (p['lon'], p['lat'], p['depth'], p['desc'])
                for p in data.get('locked_zone', [])
            ]
            self.unlocked_points = [
                (p['lon'], p['lat'], p['depth'], p['desc'])
                for p in data.get('unlocked_zone', [])
            ]
            self.interior_points = [
                (p['lon'], p['lat'], p['depth'], p['desc'])
                for p in data.get('interior_points', [])
            ]
            
            self.draw_polygons()
            print(f"読み込みました: {filename}")
            print(f"  固着域: {len(self.locked_points)}点")
            print(f"  非固着域: {len(self.unlocked_points)}点")
            print(f"  内部ポイント: {len(self.interior_points)}点")
            
        except FileNotFoundError:
            print(f"ファイルが見つかりません: {filename}")
        except json.JSONDecodeError:
            print(f"JSONの解析エラー: {filename}")
    
    def run(self):
        """エディターを実行"""
        self.load_from_dot2()
        self.save_state()  # 初期状態を保存
        self.draw_polygons()
        plt.show()


def main():
    print("=" * 60)
    print("南海トラフ ポリゴンエディター")
    print("=" * 60)
    print()
    print("操作方法:")
    print("  左クリック+ドラッグ: ポイント移動")
    print("  右クリック: ポイント追加（最適位置に挿入）")
    print("  Delete/Backspace: 選択ポイント削除")
    print("  1/2/3: 編集モード切替（固着域/非固着域/内部）")
    print("  Ctrl+Z: 元に戻す (Undo)")
    print("  Ctrl+Y: やり直し (Redo)")
    print("  S: JSON保存")
    print("  L: JSON読込")
    print("  Q: 終了")
    print()
    
    editor = PolygonEditor()
    editor.run()


if __name__ == "__main__":
    main()
