"""
南海トラフ地震シミュレーション - Realistic Simulation Logic
"""

import numpy as np
import time
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable

# Add root to path to allow importing plate_geometry if not installed as package
sys.path.append(os.getcwd())

from src.geometry.coordinates import CoordinateSystem
from src.geometry.utils import estimate_normals_optimized, classify_segments
from src.physics.stress_kernel import StressKernel
from src.physics.friction import RateStateFriction
from src.physics.equations import QuasiDynamicEquations, create_derivative_function
from src.solver.runge_kutta import RK45Solver

@dataclass
class Earthquake:
    id: int
    t_start: float
    t_end: float
    cells: List[int]
    max_slip: float
    Mw: float
    slip_distribution: np.ndarray
    segments: List[str]

class RealisticSimulation:
    def __init__(self, cells: List[Any], segments_info: Dict[str, Any] = None):
        """
        Initialize the simulation with a mesh (list of cells).

        Args:
            cells: List of Cell objects (must have lon, lat, depth, area, and physics params)
            segments_info: Dictionary of segment definitions (for classification)
        """
        self.cells = cells
        self.segments_info = segments_info if segments_info else {}
        self.n_cells = len(cells)

        # Physics components
        self.stress_kernel = None
        self.equations = None
        self.solver = None

        # Simulation state
        self.current_time = 0.0
        self.state = None
        self.earthquakes = []

        # Internal tracking
        self._sim_state = {
            'n_earthquakes': 0,
            'in_earthquake': False,
            'eq_start_time': 0.0,
            'eq_cells': set(),
            'eq_slip_start': None,
            'slip_cumulative': np.zeros(self.n_cells),
            'last_print': time.time(),
            'start_time': time.time()
        }

        # Callbacks
        self.on_earthquake: Optional[Callable[[Earthquake], None]] = None

    def setup(self):
        """Setup physics and initial conditions"""
        print(f"シミュレーション初期化: {self.n_cells} セル")

        # 1. Coordinate transformation
        lons = np.array([c.lon for c in self.cells])
        lats = np.array([c.lat for c in self.cells])
        depths = np.array([c.depth for c in self.cells])
        areas = np.array([c.area for c in self.cells])

        coords = CoordinateSystem()
        x, y, z = coords.geo_to_local(lons, lats, depths)
        centers = np.column_stack((x, y, z))

        # 2. Normal estimation (Optimized)
        print("法線ベクトルを推定中 (KDTree)...")
        normals = estimate_normals_optimized(centers)

        # 3. Stress Kernel (Parallelized)
        print("応力カーネルを計算中 (Parallel)...")
        self.stress_kernel = StressKernel(G=30.0e9, nu=0.25)
        self.stress_kernel.compute(centers, areas, normals)

        # 4. Equations
        self.equations = QuasiDynamicEquations(
            friction=RateStateFriction(),
            kernel=self.stress_kernel,
            G=30.0e9,
            beta=3750.0
        )

        # 5. Initial State
        a = np.array([c.a for c in self.cells])
        b = np.array([c.b for c in self.cells])
        sigma_eff = np.array([c.sigma_eff for c in self.cells])
        L = np.array([c.L for c in self.cells])
        V_pl = np.array([c.V_pl for c in self.cells])

        # Validate parameters
        if np.any(L <= 0):
             raise ValueError("エラー: L (特徴的すべり量) が0以下のセルがあります。")

        V_init = 1.0e-9 # Initial velocity
        self.state = self.equations.initialize_state(self.n_cells, sigma_eff, a, b, L, V_init)

        # Store parameters for derivative function
        self.params = {
            'a': a, 'b': b, 'L': L,
            'sigma_eff': sigma_eff, 'V_pl': V_pl
        }

    def run(self, years: float = 1000) -> List[Earthquake]:
        """Run the simulation"""
        if self.equations is None:
            self.setup()

        t_end = years * 365.25 * 24 * 3600

        # Create derivative function
        derivative_func = create_derivative_function(
            self.equations, self.n_cells,
            self.params['a'], self.params['b'], self.params['L'],
            self.params['sigma_eff'], self.params['V_pl']
        )

        # Setup solver
        self.solver = RK45Solver(
            rtol=1.0e-5,
            atol=1.0e-7,
            dt_min=0.001,
            dt_max=86400 * 30,
            dt_initial=1.0
        )

        print(f"\nシミュレーション開始 (RK45): {years} 年")
        self._sim_state['start_time'] = time.time()
        self._sim_state['last_print'] = time.time()

        try:
            result = self.solver.solve(
                f=derivative_func,
                y0=self.state,
                t_span=(0, t_end),
                callback=self._callback,
                verbose=False
            )

            elapsed = time.time() - self._sim_state['start_time']
            print(f"\n完了: {result['n_steps']} ステップ, {elapsed:.1f} 秒")
            print(f"検出された地震: {self._sim_state['n_earthquakes']} 個")

        except KeyboardInterrupt:
            print("\n中断されました。")

        return self.earthquakes

    def _callback(self, t, state, dt):
        tau = state[:self.n_cells]
        theta = state[self.n_cells:]

        # 速度計算
        V = self.equations.get_slip_velocity(
            tau, theta,
            self.params['a'], self.params['b'], self.params['L'],
            self.params['sigma_eff']
        )

        # すべり更新
        self._sim_state['slip_cumulative'] += V * dt

        # 地震検出ロジック
        eq_cells_now = np.where(V > 1e-2)[0] # 1cm/s threshold

        if len(eq_cells_now) > 0:
            if not self._sim_state['in_earthquake']:
                # 地震開始
                self._sim_state['in_earthquake'] = True
                self._sim_state['eq_start_time'] = t
                self._sim_state['eq_cells'] = set(eq_cells_now)
                self._sim_state['eq_slip_start'] = self._sim_state['slip_cumulative'].copy()
            else:
                # 地震継続中
                self._sim_state['eq_cells'].update(eq_cells_now)
        elif self._sim_state['in_earthquake']:
            # 地震終了
            self._sim_state['in_earthquake'] = False
            self._sim_state['n_earthquakes'] += 1

            coseismic_slip = self._sim_state['slip_cumulative'] - self._sim_state['eq_slip_start']
            max_slip = np.max(coseismic_slip)

            # モーメント計算
            areas = np.array([c.area for c in self.cells])
            M0 = 30.0e9 * np.sum(np.abs(coseismic_slip) * areas * 1e6)
            Mw = (np.log10(M0) - 9.05) / 1.5 if M0 > 0 else 0

            # セグメント分類
            segments = classify_segments(self.cells, coseismic_slip, threshold=0.1)

            eq = Earthquake(
                id=self._sim_state['n_earthquakes'],
                t_start=self._sim_state['eq_start_time'],
                t_end=t,
                cells=list(self._sim_state['eq_cells']),
                max_slip=max_slip,
                Mw=Mw,
                slip_distribution=coseismic_slip.copy(),
                segments=segments
            )
            self.earthquakes.append(eq)

            t_years_now = self._sim_state['eq_start_time'] / (365.25 * 24 * 3600)

            # セグメント名を取得（もしsegments_infoがあれば）
            segs_str = ', '.join([
                self.segments_info.get(s, {}).get('name', s)
                if isinstance(self.segments_info.get(s), dict) else str(s)
                for s in segments
            ])

            print(f"  地震 #{eq.id}: t = {t_years_now:.1f} 年, "
                  f"Mw = {Mw:.1f}, 領域 = {segs_str}")

            # 外部コールバック呼び出し
            if self.on_earthquake:
                self.on_earthquake(eq)

        # 進捗表示
        current_time = time.time()
        if current_time - self._sim_state['last_print'] > 5.0:
            t_end = self.solver.dt_max * 1000 # Dummy if not available, but should be available from context?
            # Actually run() has t_end.
            # Just print simple progress
            t_y = t / (365.25 * 24 * 3600)
            V_max = np.max(V)
            print(f"  t = {t_y:.1f} 年, 地震数 = {self._sim_state['n_earthquakes']}, "
                  f"dt = {dt:.2e} s, Vmax = {V_max:.2e} m/s")
            self._sim_state['last_print'] = current_time
