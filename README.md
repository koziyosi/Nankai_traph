# 南海トラフ巨大地震シミュレーション

# Nankai Trough Megaquake Simulation

Hirose et al. (2022) の論文に基づく、南海トラフ沿いの巨大地震発生サイクルを再現する3次元数値シミュレーション。

## 論文
>
> Hirose, F., Nakajima, J., and Hasegawa, A. (2022).
> "Simulation of great earthquakes along the Nankai Trough: reproduction of event history, slip areas of the Showa Tonankai and Nankai earthquakes, heterogeneous slip-deficit rates, and long-term slow slip events"
> Earth, Planets and Space, 74:132
> <https://doi.org/10.1186/s40623-022-01689-0>

## 特徴

- **Rate-State Friction法則**: Composite law (Kato and Tullis 2001)
- **3次元プレート境界**: 三角形メッシュで離散化
- **準動的近似**: Rice (1993) の準動的地震サイクルモデル
- **不均質パラメータ**: 海山・海嶺の影響を反映

## インストール

```bash
pip install -r requirements.txt
```

GPU高速化を使用する場合:

```bash
pip install cupy-cuda12x  # CUDA 12.x の場合
```

## 使用方法

### 基本実行

```bash
python main.py
```

### 高速テストモード

```bash
python main.py --fast
```

### パラメータ指定

```bash
python main.py --years 1000 --cell-size 15 --output-dir my_results
```

### オプション

- `--fast`: 低解像度・短期間の高速テスト
- `--years N`: シミュレーション年数
- `--cell-size N`: メッシュセルサイズ [km]
- `--output-dir DIR`: 出力ディレクトリ
- `--gpu`: GPU (CuPy) を使用
- `--no-viz`: 可視化をスキップ

## プロジェクト構成

```
nankai_simulation/
├── main.py                 # メインエントリーポイント
├── config.py               # 設定パラメータ
├── requirements.txt        # 依存ライブラリ
├── src/
│   ├── geometry/           # ジオメトリモジュール
│   │   ├── coordinates.py  # 座標変換
│   │   ├── plate_boundary.py  # プレート境界形状
│   │   └── mesh.py         # 三角形メッシュ
│   ├── physics/            # 物理モジュール
│   │   ├── friction.py     # Rate-State摩擦則
│   │   ├── stress_kernel.py  # 弾性応力カーネル
│   │   └── equations.py    # 支配方程式
│   ├── solver/             # ソルバーモジュール
│   │   ├── runge_kutta.py  # RK45適応ステップ
│   │   └── parallel.py     # 並列計算
│   ├── parameters/         # パラメータ設定
│   │   └── friction_params.py
│   └── output/             # 出力モジュール
│       ├── events.py       # イベント検出
│       └── visualization.py  # 可視化
└── results/                # 出力結果
```

## 出力

シミュレーション完了後、以下のファイルが `results/` に生成されます:

- `events.json`: 検出された地震イベントのリスト
- `final_state.npz`: 最終状態（応力、状態変数、すべり量）
- `stress_kernel.npz`: 計算された応力カーネル（再利用可能）
- `mesh.png`: メッシュ図
- `slip_final.png`: 最終すべり分布
- `slip_deficit.png`: すべり欠損レート
- `timeline.png`: 地震年表

## 計算の高速化について

1. **Numba JIT**: すべてのコア計算はNumbaで最適化
2. **応力カーネルの保存**: 一度計算したカーネルは保存して再利用
3. **適応タイムステップ**: 地震時は細かく、平穏時は大きなステップ
4. **並列計算**: マルチコアCPU対応、オプションでGPU対応

## 参考文献

- Dieterich, J. H. (1979). Modeling of rock friction. JGR.
- Ruina, A. (1983). Slip instability and state variable friction laws. JGR.
- Rice, J. R. (1993). Spatio-temporal complexity of slip on a fault. JGR.
- Kato, N. & Tullis, T. E. (2001). A composite rate- and state-dependent law. GRL.

## ライセンス

MIT License
