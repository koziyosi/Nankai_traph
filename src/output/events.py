"""
イベント検出・記録モジュール
地震イベントとLSSEの検出・分類
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import json
from datetime import datetime


@dataclass
class EarthquakeEvent:
    """地震イベント"""
    id: int
    t_start: float          # 開始時刻 [s]
    t_end: float            # 終了時刻 [s]
    duration: float         # 継続時間 [s]
    
    # 位置・規模
    cells: List[int]        # 関与セルのリスト
    hypocenter_cell: int    # 震源セル
    max_slip: float         # 最大すべり量 [m]
    total_moment: float     # 総モーメント [Nm]
    Mw: float              # マグニチュード
    
    # 領域分類
    region: str = ""        # "tonankai", "nankai", etc.
    event_type: str = ""    # "A", "B", "C", etc.
    
    # すべり分布
    slip_distribution: np.ndarray = None
    
    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            'id': self.id,
            't_start': self.t_start,
            't_end': self.t_end,
            'duration': self.duration,
            't_start_years': self.t_start / 3.15576e7,
            'cells': self.cells,
            'hypocenter_cell': self.hypocenter_cell,
            'max_slip': self.max_slip,
            'total_moment': self.total_moment,
            'Mw': self.Mw,
            'region': self.region,
            'event_type': self.event_type
        }


@dataclass
class SlowSlipEvent:
    """スロースリップイベント（LSSE）"""
    id: int
    t_start: float
    t_end: float
    duration: float
    
    cells: List[int]
    max_slip: float
    total_moment: float
    Mw: float
    
    region: str = ""  # "tokai_lsse", "bungo_lsse", etc.
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            't_start': self.t_start,
            't_end': self.t_end,
            'duration': self.duration,
            't_start_years': self.t_start / 3.15576e7,
            'duration_days': self.duration / 86400,
            'cells': self.cells,
            'max_slip': self.max_slip,
            'total_moment': self.total_moment,
            'Mw': self.Mw,
            'region': self.region
        }


@dataclass
class EventDetector:
    """
    地震・スロースリップイベント検出器
    
    論文の定義:
    - 地震: V > 0.1 m/s のセルがある期間
    - LSSE: V > V_pl かつ V < 0.1 m/s が一定期間続く
    """
    
    # 検出閾値
    V_earthquake: float = 0.1           # 地震判定速度 [m/s]
    V_lsse_factor: float = 10.0         # LSSE判定: V > V_pl * factor
    lsse_min_duration: float = 86400    # LSSE最小継続時間 [s] (1日)
    
    # 物理定数（Mw計算用）
    G: float = 30.0e9                   # 剛性率 [Pa]
    
    # イベントリスト
    earthquakes: List[EarthquakeEvent] = field(default_factory=list)
    slow_slips: List[SlowSlipEvent] = field(default_factory=list)
    
    # 検出状態
    _in_earthquake: bool = False
    _eq_start_time: float = 0.0
    _eq_cells: List[int] = field(default_factory=list)
    _eq_slip: np.ndarray = None
    _earthquake_count: int = 0
    
    def detect(self,
               t: float,
               V: np.ndarray,
               slip: np.ndarray,
               areas: np.ndarray,
               V_pl: np.ndarray = None) -> Optional[EarthquakeEvent]:
        """
        現在の状態からイベントを検出
        
        Parameters:
            t: 時刻 [s]
            V: すべり速度 [n_cells] [m/s]
            slip: 累積すべり量 [n_cells] [m]
            areas: セル面積 [n_cells] [m²]
            V_pl: プレート収束速度 [m/s] (LSSE検出用)
        
        Returns:
            終了した地震イベント（なければ None）
        """
        # 地震判定
        eq_cells = np.where(V > self.V_earthquake)[0]
        is_earthquake = len(eq_cells) > 0
        
        if is_earthquake and not self._in_earthquake:
            # 地震開始
            self._in_earthquake = True
            self._eq_start_time = t
            self._eq_cells = list(eq_cells)
            self._eq_slip = slip.copy()
            return None
        
        elif is_earthquake and self._in_earthquake:
            # 地震継続中
            self._eq_cells = list(set(self._eq_cells) | set(eq_cells))
            return None
        
        elif not is_earthquake and self._in_earthquake:
            # 地震終了
            self._in_earthquake = False
            
            # イベント作成
            coseismic_slip = slip - self._eq_slip
            
            event = self._create_earthquake_event(
                t_start=self._eq_start_time,
                t_end=t,
                cells=self._eq_cells,
                coseismic_slip=coseismic_slip,
                areas=areas
            )
            
            self.earthquakes.append(event)
            return event
        
        return None
    
    def _create_earthquake_event(self,
                                  t_start: float,
                                  t_end: float,
                                  cells: List[int],
                                  coseismic_slip: np.ndarray,
                                  areas: np.ndarray) -> EarthquakeEvent:
        """地震イベントオブジェクトを作成"""
        self._earthquake_count += 1
        
        # 最大すべり
        max_slip = np.max(coseismic_slip)
        
        # 震源セル（最初に大きくすべったセル）
        hypocenter = cells[0] if cells else 0
        
        # モーメント計算: M_0 = G * Σ(slip * area)
        total_moment = self.G * np.sum(np.abs(coseismic_slip) * areas)
        
        # マグニチュード: log(M_0) = 1.5*Mw + 9.1 (Kanamori 1977)
        if total_moment > 0:
            Mw = (np.log10(total_moment) - 9.1) / 1.5
        else:
            Mw = 0.0
        
        return EarthquakeEvent(
            id=self._earthquake_count,
            t_start=t_start,
            t_end=t_end,
            duration=t_end - t_start,
            cells=cells,
            hypocenter_cell=hypocenter,
            max_slip=max_slip,
            total_moment=total_moment,
            Mw=Mw,
            slip_distribution=coseismic_slip.copy()
        )
    
    def classify_region(self, event: EarthquakeEvent, 
                        cell_regions: Dict[int, str]) -> str:
        """
        地震の領域分類
        
        Parameters:
            event: 地震イベント
            cell_regions: セルID -> 領域名のマッピング
        
        Returns:
            領域名（複数の場合はカンマ区切り）
        """
        regions = set()
        for cell_id in event.cells:
            if cell_id in cell_regions:
                regions.add(cell_regions[cell_id])
        
        return ','.join(sorted(regions))
    
    def get_event_statistics(self) -> dict:
        """イベント統計を取得"""
        if not self.earthquakes:
            return {'count': 0}
        
        Mws = [eq.Mw for eq in self.earthquakes]
        intervals = []
        for i in range(1, len(self.earthquakes)):
            dt = self.earthquakes[i].t_start - self.earthquakes[i-1].t_start
            intervals.append(dt / 3.15576e7)  # years
        
        return {
            'count': len(self.earthquakes),
            'Mw_min': min(Mws),
            'Mw_max': max(Mws),
            'Mw_mean': np.mean(Mws),
            'interval_mean_years': np.mean(intervals) if intervals else 0,
            'interval_std_years': np.std(intervals) if intervals else 0
        }
    
    def save_events(self, filepath: str):
        """イベントをJSONファイルに保存"""
        data = {
            'metadata': {
                'created': datetime.now().isoformat(),
                'n_earthquakes': len(self.earthquakes),
                'n_slow_slips': len(self.slow_slips),
            },
            'earthquakes': [eq.to_dict() for eq in self.earthquakes],
            'slow_slips': [ss.to_dict() for ss in self.slow_slips]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"イベントを保存: {filepath}")
    
    @classmethod
    def load_events(cls, filepath: str) -> 'EventDetector':
        """JSONファイルからイベントを読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        detector = cls()
        
        for eq_data in data.get('earthquakes', []):
            event = EarthquakeEvent(
                id=eq_data['id'],
                t_start=eq_data['t_start'],
                t_end=eq_data['t_end'],
                duration=eq_data['duration'],
                cells=eq_data['cells'],
                hypocenter_cell=eq_data['hypocenter_cell'],
                max_slip=eq_data['max_slip'],
                total_moment=eq_data['total_moment'],
                Mw=eq_data['Mw'],
                region=eq_data.get('region', ''),
                event_type=eq_data.get('event_type', '')
            )
            detector.earthquakes.append(event)
        
        return detector


def detect_event_pairs(earthquakes: List[EarthquakeEvent],
                       max_interval_years: float = 3.0
                       ) -> List[Tuple[EarthquakeEvent, EarthquakeEvent]]:
    """
    東南海・南海地震ペアを検出
    
    Parameters:
        earthquakes: 地震リスト
        max_interval_years: ペアとみなす最大間隔 [年]
    
    Returns:
        ペアのリスト
    """
    pairs = []
    max_interval_s = max_interval_years * 3.15576e7
    
    used = set()
    for i, eq1 in enumerate(earthquakes):
        if i in used:
            continue
        
        for j, eq2 in enumerate(earthquakes):
            if j <= i or j in used:
                continue
            
            dt = eq2.t_start - eq1.t_start
            if 0 < dt < max_interval_s:
                # 東西で異なる領域のチェック
                # TODO: 領域情報を使用
                pairs.append((eq1, eq2))
                used.add(i)
                used.add(j)
                break
    
    return pairs
