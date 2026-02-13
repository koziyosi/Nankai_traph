"""
南海トラフ巨大地震シミュレーション - メインスクリプト
Nankai Trough Megaquake Simulation

Based on: Hirose et al. (2022) 
"Simulation of great earthquakes along the Nankai Trough"
Earth, Planets and Space

Usage:
    python main.py                    # デフォルト設定で実行
    python main.py --fast             # 高速テストモード
    python main.py --years 1000       # シミュレーション年数指定
    python main.py --cell-size 10     # セルサイズ指定 [km]
"""

import argparse
import numpy as np
import os
import sys
import time
from datetime import datetime

# パスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimulationConfig, SECONDS_PER_YEAR
from src.geometry import PlateBoundary, TriangularMesh, CoordinateSystem
from src.physics import RateStateFriction, StressKernel, QuasiDynamicEquations
from src.physics.equations import create_derivative_function
from src.solver import RK45Solver
from src.solver.parallel import ParallelAccelerator, get_system_info
from src.parameters import FrictionParameterSetter
from src.output import EventDetector, Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='南海トラフ巨大地震シミュレーション'
    )
    parser.add_argument('--fast', action='store_true',
                        help='高速テストモード（低解像度）')
    parser.add_argument('--years', type=float, default=None,
                        help='シミュレーション年数')
    parser.add_argument('--cell-size', type=float, default=None,
                        help='メッシュセルサイズ [km]')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='出力ディレクトリ')
    parser.add_argument('--load-kernel', type=str, default=None,
                        help='事前計算カーネルを読み込む')
    parser.add_argument('--save-kernel', action='store_true',
                        help='カーネルを保存する')
    parser.add_argument('--no-viz', action='store_true',
                        help='可視化をスキップ')
    parser.add_argument('--gpu', action='store_true',
                        help='GPU (CuPy) を使用')
    parser.add_argument('--friction-csv', type=str, default=None,
                        help='摩擦パラメータCSVファイルのパス')
    return parser.parse_args()


def run_simulation(config: SimulationConfig):
    """シミュレーションを実行"""
    
    print("=" * 60)
    print("南海トラフ巨大地震シミュレーション")
    print("Nankai Trough Megaquake Simulation")
    print("Based on Hirose et al. (2022)")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    # システム情報
    sys_info = get_system_info()
    print(f"システム情報: {sys_info}")
    print()
    
    # 出力ディレクトリ作成
    os.makedirs(config.output_dir, exist_ok=True)
    
    # =========================================================================
    # Step 1: ジオメトリ（メッシュ生成）
    # =========================================================================
    print("-" * 50)
    print("Step 1: メッシュ生成")
    print("-" * 50)
    
    coords = CoordinateSystem()
    plate = PlateBoundary()
    mesh = TriangularMesh(
        plate=plate,
        coords=coords,
        cell_size=config.geometry.cell_size
    )
    
    mesh.generate(
        lon_range=(config.geometry.lon_min, config.geometry.lon_max),
        lat_range=(config.geometry.lat_min, config.geometry.lat_max),
        depth_range=(config.geometry.depth_min, config.geometry.depth_max)
    )
    
    n_cells = mesh.n_cells
    print(f"  生成されたセル数: {n_cells}")
    print()
    
    # =========================================================================
    # Step 2: パラメータ設定
    # =========================================================================
    print("-" * 50)
    print("Step 2: 摩擦パラメータ設定")
    print("-" * 50)
    # 3. 摩擦パラメータの設定
    # config からパラメータCSVパスを渡す
    param_setter = FrictionParameterSetter(
        parameter_csv=config.geometry.friction_csv_path
    )
    param_setter.set_parameters(mesh, coords) # Changed coords_system to coords to match existing variable name
    print()
    
    # =========================================================================
    # Step 3: 応力カーネル計算
    # =========================================================================
    print("-" * 50)
    print("Step 3: 応力カーネル計算")
    print("-" * 50)
    
    kernel = StressKernel(
        G=config.physical.G,
        nu=config.physical.nu,
        beta=config.physical.beta
    )
    
    kernel_path = os.path.join(config.output_dir, 'stress_kernel.npz')
    
    if config.solver.use_gpu and os.path.exists(kernel_path):
        # 既存のカーネルを読み込み
        print(f"  カーネルを読み込み: {kernel_path}")
        kernel = StressKernel.load(kernel_path)
    else:
        # カーネルを計算
        print(f"  カーネル行列を計算中... ({n_cells} x {n_cells})")
        
        # すべり方向（N60°W）
        azimuth = np.radians(330.0)
        slip_dir = np.zeros((n_cells, 3))
        slip_dir[:, 0] = np.sin(azimuth)
        slip_dir[:, 1] = np.cos(azimuth)
        
        kernel.compute(
            centers=mesh.centers,
            areas=mesh.areas,
            normals=mesh.normals,
            slip_direction=slip_dir
        )
        
        # 保存（オプション）
        if True:  # 常に保存
            kernel.save(kernel_path)
    
    print()
    
    # =========================================================================
    # Step 4: 方程式系の構築
    # =========================================================================
    print("-" * 50)
    print("Step 4: 方程式系の構築")
    print("-" * 50)
    
    equations = QuasiDynamicEquations(
        friction=RateStateFriction(
            mu_0=config.friction.mu_0,
            V_0=config.friction.V_0,
            V_c=config.friction.V_c
        ),
        kernel=kernel,
        G=config.physical.G,
        beta=config.physical.beta,
        use_gpu=config.solver.use_gpu
    )
    
    # 初期条件
    V_init = 0.1 / 3.15576e7 / 100  # 0.1 cm/year → m/s
    state0 = equations.initialize_state(
        n_cells=n_cells,
        sigma_eff=mesh.sigma_eff,
        a=mesh.a,
        b=mesh.b,
        L=mesh.L,
        V_init=V_init
    )
    
    print(f"  自由度: {len(state0)} (= 2 × {n_cells})")
    print()
    
    # =========================================================================
    # Step 5: 時間積分
    # =========================================================================
    print("-" * 50)
    print("Step 5: 時間積分")
    print("-" * 50)
    
    t_end = config.solver.t_years * SECONDS_PER_YEAR
    
    print(f"  シミュレーション期間: {config.solver.t_years} 年")
    print(f"  = {t_end:.2e} 秒")
    print()
    
    # 微分関数
    derivative_func = create_derivative_function(
        equations=equations,
        n_cells=n_cells,
        a=mesh.a,
        b=mesh.b,
        L=mesh.L,
        sigma_eff=mesh.sigma_eff,
        V_pl=mesh.V_pl
    )
    
    # ソルバー
    solver = RK45Solver(
        rtol=config.solver.rtol,
        atol=config.solver.atol,
        dt_min=config.solver.dt_min,
        dt_max=config.solver.dt_max,
        dt_initial=config.solver.dt_initial
    )
    
    # イベント検出器
    detector = EventDetector(G=config.physical.G)
    
    # 履歴保存用（サンプリング）
    output_interval = config.solver.output_interval_years * SECONDS_PER_YEAR
    t_output = np.arange(0, t_end + output_interval, output_interval)
    
    # 累積すべり追跡
    slip = np.zeros(n_cells)
    
    # コールバック関数
    last_output_t = [0.0]
    V_history = []
    slip_history = []
    step_count = [0]
    
    viz = Visualizer(output_dir=config.output_dir)
    
    def callback(t, state, dt):
        step_count[0] += 1
        nonlocal slip
        tau = state[:n_cells]
        theta = state[n_cells:]
        
        # 速度計算
        V = equations.get_slip_velocity(tau, theta, mesh.a, mesh.b, mesh.L, mesh.sigma_eff)
        
        # すべり更新（簡易版）
        slip += V * dt
        
        # イベント検出
        event = detector.detect(t, V, slip, mesh.areas * 1e6, mesh.V_pl)
        if event:
            print(f"    *** 地震検出: ID={event.id}, Mw={event.Mw:.1f}, "
                  f"t={t/SECONDS_PER_YEAR:.1f} years ***")
            # 破壊分布を即座にプロット
            viz.plot_event_rupture(mesh, event)

        # デバッグ: 状態表示（定期的に）
        if step_count[0] % 500 == 0:
             print(f"    t={t/SECONDS_PER_YEAR:.4f}y, dt={dt:.1e}s, V_max={V.max():.2e} m/s")
        
        # 履歴保存（間引き）
        if t - last_output_t[0] >= output_interval:
            V_history.append(V.copy())
            slip_history.append(slip.copy())
            last_output_t[0] = t
    
    print("シミュレーション開始...")
    result = solver.solve(
        f=derivative_func,
        y0=state0,
        t_span=(0, t_end),
        callback=callback,
        verbose=True
    )
    
    print()
    print(f"完了: {result['n_steps']} ステップ, {result['elapsed']:.1f} 秒")
    print()
    
    # =========================================================================
    # Step 6: 結果の整理
    # =========================================================================
    print("-" * 50)
    print("Step 6: 結果の整理")
    print("-" * 50)
    
    # 統計
    stats = detector.get_event_statistics()
    print(f"  検出された地震: {stats.get('count', 0)} 個")
    if stats.get('count', 0) > 0:
        print(f"    Mw範囲: {stats['Mw_min']:.1f} - {stats['Mw_max']:.1f}")
        print(f"    平均間隔: {stats['interval_mean_years']:.1f} ± {stats['interval_std_years']:.1f} 年")
    
    # イベント保存
    events_path = os.path.join(config.output_dir, 'events.json')
    detector.save_events(events_path)
    
    # 最終状態保存
    final_state_path = os.path.join(config.output_dir, 'final_state.npz')
    np.savez_compressed(
        final_state_path,
        tau=result['y'][-1, :n_cells],
        theta=result['y'][-1, n_cells:],
        slip=slip,
        t=result['t']
    )
    print(f"  最終状態を保存: {final_state_path}")
    print()
    
    # =========================================================================
    # Step 7: 可視化
    # =========================================================================
    if not config.save_full_history:
        print("-" * 50)
        print("Step 7: 可視化")
        print("-" * 50)
        
        viz = Visualizer(output_dir=config.output_dir)
        
        # 最終状態の速度を計算
        tau_final = result['y'][-1, :n_cells]
        theta_final = result['y'][-1, n_cells:]
        V_final = equations.get_slip_velocity(
            tau_final, theta_final, mesh.a, mesh.b, mesh.L, mesh.sigma_eff
        )
        
        viz.save_summary_plots(
            mesh=mesh,
            earthquakes=detector.earthquakes,
            slip_final=slip,
            V_final=V_final,
            V_pl=mesh.V_pl
        )
        print()
    
    # =========================================================================
    # 完了
    # =========================================================================
    elapsed_total = time.time() - start_time
    print("=" * 60)
    print(f"シミュレーション完了!")
    print(f"  総実行時間: {elapsed_total:.1f} 秒 ({elapsed_total/60:.1f} 分)")
    print(f"  出力ディレクトリ: {config.output_dir}")
    print("=" * 60)
    
    return {
        'mesh': mesh,
        'result': result,
        'detector': detector,
        'config': config
    }


def main():
    args = parse_args()
    
    # 設定を構築
    if args.fast:
        config = SimulationConfig.fast_test()
    else:
        config = SimulationConfig.default()
    
    # コマンドライン引数で上書き
    if args.years is not None:
        config.solver.t_years = args.years
    if args.cell_size is not None:
        config.geometry.cell_size = args.cell_size
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.gpu:
        config.solver.use_gpu = True
    if args.no_viz:
        config.save_full_history = True  # 可視化をスキップするためのフラグ
    if args.friction_csv:
        config.geometry.friction_csv_path = args.friction_csv
    
    # 実行
    run_simulation(config)


if __name__ == '__main__':
    main()
