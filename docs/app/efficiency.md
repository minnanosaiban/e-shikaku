# 軽量化・高速化

<p class="kw">キーワード：モデル圧縮、知識蒸留（ソフトターゲット・温度付きソフトマックス）、プルーニング（枝刈り・宝くじ仮説）、量子化（2 値化・ストレートスルー）、軽量なネットワーク（MobileNet・EfficientNet）、LoRA、分散処理（データ並列・モデル並列・同期型・非同期型・陳腐化した勾配）、混合精度、GPU・TPU・FPGA、Docker</p>

!!! abstract "要点"
    - 目的は、**推論の速さ・メモリ・電力**（端末への搭載）と、**学習時間**（大規模データ・大規模モデル）の両方。精度をなるべく落とさずに、計算量とメモリを減らす。
    - **圧縮**の 3 本柱：**蒸留**（大きな先生の出力を、小さな生徒に真似させる）、**枝刈り**（不要な重み・ノードを消す）、**量子化**（数値のビット数を減らす）。
    - **学習の高速化**は**分散処理**。データを分ける**データ並列**、モデルを分ける**モデル並列**。更新を待ち合わせる**同期型**と、待たない**非同期型**（古い勾配が混ざる）の違いを押さえる。

---

## 全体像 {#overview}

| 手法 | 何を減らすか | 要点 |
|:--|:--|:--|
| **軽量なアーキテクチャ** | 計算量・パラメータ | depth-wise separable 畳み込み、ボトルネック、探索による設計 |
| **蒸留** | モデルの大きさ | 先生の確率（ソフトターゲット）を生徒が学ぶ |
| **枝刈り** | パラメータ数 | 重要でない重み・フィルタを消して疎にする |
| **量子化** | ビット数・メモリ | 32 ビット浮動小数点 → 8 ビット整数など |
| **分散処理** | 学習時間 | 複数の GPU・計算機で並列に学習 |

---

## 軽量なアーキテクチャ {#architecture}

- **MobileNet**（depth-wise separable 畳み込み、逆残差、NAS）と **EfficientNet**（複合スケーリング）は [CNN](../dl/cnn.md#architectures) にまとめてある。要点の計算量の比：空間方向（depth-wise）とチャネル方向（point-wise）を**分ける**と、標準の畳み込みに対して次の比になる。

$$
\frac{F^2C_{in}HW+C_{in}C_{out}HW}{F^2C_{in}C_{out}HW}=\frac1{C_{out}}+\frac1{F^2}\qquad(F=3,\ C_{out}=64:\ \tfrac1{64}+\tfrac19\approx0.127\approx\tfrac1{8})
$$

- ほかに、チャネルを組に分けた**グループ畳み込み**とチャネルの入れ替え（ShuffleNet）、1×1 畳み込みで絞る **Fire モジュール**（SqueezeNet）、探索で設計する **NAS**（MnasNet、MobileNetV3、EfficientNet-B0）がある。
- **Transformer の軽量化**：Attention の計算量は系列長の 2 乗なので、疎な Attention、メモリアクセスを工夫した **FlashAttention**（計算結果は同じで高速）などがある。
### LoRA（低ランク適応） {#lora}

- 大きな言語モデルのファインチューニングで、**元の重み $\mathbf{W}$ は凍結**し、低ランクの差分だけを学習する。更新するパラメータが激減する。

$$
\mathbf{W}'=\mathbf{W}+\frac{\alpha}{r}\mathbf{B}\mathbf{A},\qquad \mathbf{B}\in\mathbb{R}^{d\times r},\ \ \mathbf{A}\in\mathbb{R}^{r\times k},\ \ r\ll\min(d,k)\qquad(\text{パラメータ数 }r(d+k)\text{ と }dk)
$$

- 例：$d=k=4096,\ r=8$ なら、$8\times8192=65{,}536$ で、$4096^2=16{,}777{,}216$ の約 0.4 %。

---

## 知識蒸留 {#distillation}

- **蒸留**（Knowledge Distillation）：大きく高精度な**先生**（教師）モデルの出力を、小さな**生徒**モデルが真似て学ぶ。正解の 1/0（ハードターゲット）に加えて、先生の出力した**確率の分布（ソフトターゲット）**を教師信号にする。
- ソフトターゲットには、「猫の画像は、犬の確率が車より高い」のように、**クラス間の似かた**の情報（暗黙の知識）が入っている。1/0 のラベルにはない。
- そのままだと先生の確率は 1 つのクラスに偏って情報が見えにくいので、**温度** $T$ を使って確率を**なだらか**にする（**温度付きソフトマックス**）。

$$
q_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)},\qquad
L=(1-\lambda)\,\mathrm{CE}\bigl(\mathbf{t},\ \mathbf{q}^{\mathrm{S}}_{T=1}\bigr)+\lambda\,T^2\,D_{\mathrm{KL}}\bigl(\mathbf{q}^{\mathrm{T}}_{T}\,\Vert\,\mathbf{q}^{\mathrm{S}}_{T}\bigr)
$$

- $\mathbf{z}$：ロジット、$\mathbf{t}$：正解（one-hot）、$\mathbf{q}^{\mathrm{T}},\mathbf{q}^{\mathrm{S}}$：先生と生徒の出力、$\lambda$：2 つの損失の重み。**同じ温度 $T$** を先生と生徒の両方に使い、推論時は $T=1$ に戻す。
- ソフトターゲットの項に $T^2$ を掛けるのは、**勾配の大きさが約 $1/T^2$ に縮む**ので、ハードターゲットの項との比を保つため。
- 派生：**DistilBERT**（BERT を蒸留した小型版）、生徒が先生と同じ大きさの**自己蒸留**、中間層の出力まで合わせる方法。

<div class="ew" data-widget="temperature"></div>

!!! tip "動かしてみる"
    - $T=1$ のとき、最大のロジットのクラスに確率が偏ります。$T$ を大きくすると、確率が**なだらか**になり、2 番目・3 番目のクラスの「らしさ」が見えてきます（蒸留で生徒に伝える情報）。
    - $T$ を小さくする（0.1 付近）と、最大のクラスがほぼ 1 になり、ハードターゲットに近づきます。

---

## 枝刈り（プルーニング） {#pruning}

- 学習済みのネットワークから、**精度に大きく効かない重みやノードを取り除き**、パラメータ数と計算量を減らす。重みの**絶対値が小さいもの**を 0 にする（**マグニチュードプルーニング**）のが基本。

| 種類 | 何を消すか | 速度への効果 |
|:--|:--|:--|
| **非構造化** | 個々の重み（疎な行列になる） | メモリは減る。**専用のライブラリやハードウェアがないと速くならない** |
| **構造化** | **フィルタ・チャネル・ノード**ごと | 行列が小さくなるので、**そのまま速くなる** |

- 流れ：**学習 → 枝刈り → 再学習（ファインチューニング）**を、少しずつ繰り返す。枝刈りは学習の途中か、学習後に行う（学習前には、どれが不要か分からない）。
- L1 正則化は、重みを 0 に近づけるので、枝刈りと相性がよい。
- **宝くじ仮説**（Lottery Ticket Hypothesis）：大きなネットワークの中には、**初期値のまま単独で学習し直しても**元と同程度の精度になる**小さな部分ネットワーク（当たりくじ）**が存在する。「枝刈りで得た疎なネットワークは、なぜ最初から学習できないのか」という疑問から生まれた。
- 組み合わせた例：**Deep Compression**（枝刈り → 量子化 → ハフマン符号化）で、モデルを数十分の 1 に圧縮。

---

## 量子化 {#quantization}

- 重み・活性化（中間層の出力）・勾配を、**少ないビット数**で表す。32 ビット浮動小数点を 8 ビット整数にすると、メモリは **1/4**、整数演算なので計算も速く省電力（例：Google TPU は 8 ビットの計算ユニット）。精度は少し落ちる。
- **一様量子化**：値の範囲を $2^b$ 段階に等分し、整数 $q$ に丸める。$s$ は刻み幅（スケール）。

$$
q=\mathrm{clip}\Bigl(\mathrm{round}\Bigl(\frac{x}{s}\Bigr),\ -2^{b-1},\ 2^{b-1}-1\Bigr),\qquad \hat{x}=s\,q,\qquad s=\frac{\max|x|}{2^{b-1}-1}\ \ (\text{対称量子化})
$$

- 例：$b=8$、値の範囲 $[-1,1]$ なら $s=1/127$。$x=0.5$ は $q=\mathrm{round}(63.5)=64$、$\hat{x}=64/127\approx0.504$。**丸め誤差は最大 $s/2\approx0.004$**。ビット数を 1 減らすと刻みが約 2 倍になる。
- 範囲がずれた分布には、**ゼロ点** $z$ を足す非対称量子化 $\hat{x}=s(q-z)$ を使う。範囲の決め方は、重み全体で 1 つ（テンソル単位）より、**出力チャネルごと**のほうが精度がよい。GNMT は行ごとに $s_i=\max|W[i,:]|$ で 8 ビット化した。
- **学習後量子化**（PTQ）：学習済みモデルを後から量子化。少量のデータで範囲を決める（キャリブレーション）。**量子化を考慮した学習**（QAT）：学習中に量子化の丸めを**模擬**して、誤差に強い重みにする。
- **2 値化**（binarization）：重みや活性化を $\pm1$（1 ビット）にする。$\mathrm{sign}$ は、ほとんどの点で**微分が 0** なので、そのままでは勾配が流れない。そこで**ストレートスルー推定量**：順伝播は $\mathrm{sign}$、**逆伝播だけ恒等写像（微分 1）とみなして**、勾配をそのまま通す（$|x|\le1$ だけ通すクリップ付きもよく使う）。BinaryNet・XNOR-Net。VQ-VAE の量子化にも同じ考え方が使われる。
- **混合精度学習**：順伝播・逆伝播は **FP16**（または **bfloat16**）、重みの更新用のコピーは FP32 で持つ。FP16 は範囲が狭く、勾配が 0 になる**アンダーフロー**が起きるので、損失を定数倍して（**ロススケーリング**）から逆伝播する。16 ビットへの精度落としは**オーバーフロー**にも注意。

---

## 分散処理 {#distributed}

### データ並列とモデル並列

| | データ並列 | モデル並列 |
|:--|:--|:--|
| 分けるもの | **ミニバッチ**（データ） | **モデル**（層や重み） |
| 各計算機の持つもの | **モデルの全体**のコピー | モデルの**一部** |
| 使う場面 | 学習の高速化 | **1 台に載らない**巨大なモデル（高解像度の画像を入れる CNN、大規模言語モデル） |
| 通信するもの | **勾配**（またはパラメータ） | 層の**途中の出力**（活性化）と、その勾配 |

- **データ並列**：$K$ 台のワーカーが、**別々のデータ**（各 $B$ 個）で勾配 $\mathbf{g}_k$ を求め、集約して更新する。**同期型**の更新は、$K$ 台の平均勾配を使うので、**バッチサイズ $KB$ の SGD と同じ**。

$$
\mathbf{w}\leftarrow\mathbf{w}-\eta\cdot\frac1K\sum_{k=1}^{K}\mathbf{g}_k
$$

| | 同期型 | 非同期型 |
|:--|:--|:--|
| 更新 | 全ワーカーの勾配をそろえて**平均**し、全員が同じパラメータで続ける | 各ワーカーが、**終わった順**に勾配を送って更新（**待たない**） |
| 長所 | 常に最新の勾配。**収束が安定で精度がよい** | 遅いワーカー（ストラグラー）や**故障**があっても止まらない。待ち時間なし |
| 短所 | 最も遅いワーカーを**待つ** | **陳腐化した勾配**（古いパラメータで計算した勾配）が混ざり、学習が不安定 |

- **陳腐化した勾配**（stale gradient）への対策：学習率を下げる、古すぎる勾配を捨てる、ミニバッチの大きさを調整する。
- **大きなバッチ**にすると速くなるが、そのままだと精度が落ちる。**学習率をバッチ数に比例して大きくする**（線形スケーリング則）、学習の最初に学習率を徐々に上げる（**ウォームアップ**）などで補う。
- 通信の方式：**パラメータサーバー**（中央に集めて配る。通信が集中する）と、**All-Reduce**（ワーカー間で直接、リング状に平均を取る）。

### モデル並列の発展

- **パイプライン並列**（GPipe）：層を各 GPU に割り当て、ミニバッチを**小さな単位（マイクロバッチ）に分けて流し込み**、待ち時間（バブル）を減らす。**テンソル並列**：1 つの層の行列を分割（Megatron-LM）。**ZeRO**：最適化の状態・勾配・パラメータを各 GPU に分割して持ち、メモリを節約しながらデータ並列にする。
- GNMT は、**層ごとに別の GPU**（モデル並列）と、複数のレプリカ（データ並列）を組み合わせた。双方向 LSTM を最下層だけにしたのも、並列化のため。

---

## ハードウェアと開発環境 {#hardware}

| | 特徴 | 深層学習での使い方 |
|:--|:--|:--|
| CPU | 少数の高性能なコア。逐次処理が得意 | 前処理・小さなモデル・推論 |
| **GPU** | 数千の演算コアで、**同じ命令を異なるデータ**に実行（**SIMD**）。単精度（32 ビット）が中心、半精度（16 ビット）に対応 | 行列演算の並列化。深層学習の学習の標準。**GPGPU**（グラフィックス以外の汎用計算）。複数枚では**通信**が律速になりやすい |
| **TPU** | Google の深層学習専用。**シストリックアレイ**（積和演算器を直列につないで、結果をメモリに戻さず次へ渡す）、**8 ビット・bfloat16** | メモリアクセスが減り、高スループットで低消費電力 |
| **FPGA** | 製造後に**回路の構成を書き換えられる** | **エッジ端末**での高速・低消費電力の推論 |

- **Docker**：アプリを**環境ごと**（OS のライブラリ・フレームワークの版）まとめて動かす仕組み。ゲスト OS を持つ**仮想マシン**（ホスト型）と違い、**ホスト OS のカーネルを共有**する**コンテナ型**なので、起動が速く軽い。環境が再現できる反面、OS は選べない。
    - **Dockerfile**（設計図）を **build** して **Docker イメージ**（動作環境のテンプレート）を作り、**run** して **コンテナ**（実行中の環境）にする。1 つのイメージから多数のコンテナを作れる。イメージは Docker Hub などのレジストリから取得できる（`pull`）、登録できる（`push`）。
    - 主なコマンド：`docker build -t 名前 .`、`docker run -d -p ホストのポート:コンテナのポート イメージ名`、`docker container start / stop / rm`。

---

## 試験の着眼点

- **蒸留**：先生の**ソフトターゲット**を生徒が学ぶ。**温度 $T$** で確率をなだらかにし、損失に $T^2$ を掛ける。**枝刈り**：重要でない重みを消す。**宝くじ仮説**：当たりくじの部分ネットワークがある。構造化は速度に効きやすい。
- **量子化**：ビット数を減らして**メモリと計算を削減**。対象は**重み・勾配・活性化**。**2 値化**は勾配が 0 になるので**ストレートスルー**（逆伝播は恒等写像）。
- **モデル並列**：1 台に載らないモデル。**データ並列**：データを分けて勾配を集約。**同期型**は安定（遅い計算機を待つ）、**非同期型**は待たないが**陳腐化した勾配**で不安定。
- **GPU**：SIMD・GPGPU・単精度中心、半精度では**オーバーフロー**に注意。**TPU**：シストリックアレイ・8 ビット。**FPGA**：書き換え可能・エッジ。
- **Docker**：コンテナ型は**ホスト OS のカーネルを共有**（起動が速い・再現性が高い）。Dockerfile → イメージ → コンテナ。
- **MobileNet**：depth-wise と point-wise の分離で、計算量は $1/C_{out}+1/F^2$ 倍。

---

## 参考

- [Distilling the Knowledge in a Neural Network（Hinton ほか, 2015）](https://arxiv.org/abs/1503.02531)
- [DistilBERT, a distilled version of BERT（Sanh ほか, 2019）](https://arxiv.org/abs/1910.01108)
- [Learning both Weights and Connections for Efficient Neural Networks（Han ほか, 2015）](https://arxiv.org/abs/1506.02626)
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks（Frankle, Carbin, 2018）](https://arxiv.org/abs/1803.03635)
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding（Han ほか, 2015）](https://arxiv.org/abs/1510.00149)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference（Jacob ほか, 2017）](https://arxiv.org/abs/1712.05877)
- [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation（ストレートスルー, Bengio ほか, 2013）](https://arxiv.org/abs/1308.3432)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks（Rastegari ほか, 2016）](https://arxiv.org/abs/1603.05279)
- [Mixed Precision Training（Micikevicius ほか, 2017）](https://arxiv.org/abs/1710.03740)
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications（Howard ほか, 2017）](https://arxiv.org/abs/1704.04861)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks（Tan, Le, 2019）](https://arxiv.org/abs/1905.11946)
- [LoRA: Low-Rank Adaptation of Large Language Models（Hu ほか, 2021）](https://arxiv.org/abs/2106.09685)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（Dao ほか, 2022）](https://arxiv.org/abs/2205.14135)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour（Goyal ほか, 2017）](https://arxiv.org/abs/1706.02677)
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism（Huang ほか, 2018）](https://arxiv.org/abs/1811.06965)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models（Rajbhandari ほか, 2019）](https://arxiv.org/abs/1910.02054)
- [Docker ドキュメント（公式）](https://docs.docker.com/)
