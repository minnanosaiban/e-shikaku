# グラフニューラルネットワーク

<p class="kw">キーワード：グラフ、ノード予測・リンク予測・グラフ予測、隣接行列、次数行列、グラフラプラシアン、正規化ラプラシアン、グラフフーリエ変換、Spectral Convolution、ChebNet（チェビシェフ多項式近似）、GCN、Spatial Convolution、メッセージパッシング、GraphSAGE、GAT、GIN、過平滑化</p>

!!! abstract "要点"
    - グラフ（ノードとエッジ）は、格子状でなく、ノードごとに**隣の数が違う**。CNN の畳み込みをそのまま使えないので、**グラフ上の畳み込み**を定義する。
    - **スペクトル法**：グラフラプラシアンの固有ベクトルで**グラフフーリエ変換**し、周波数領域で重みを掛ける。ChebNet は多項式で近似して固有値分解を避け、**GCN** はその 1 次の近似で「**隣のノードの特徴を正規化して平均**してから線形変換」という単純な形になった。
    - **空間法**：隣のノードから**メッセージ**を集めて（集約）、自分を更新する。GraphSAGE・GAT・GIN などはこの枠組みで、**新しいグラフにも使える**。

---

## グラフとタスク {#tasks}

- グラフ $G=(V,E)$：ノード（頂点）$V$（$N$ 個）と、エッジ（辺）$E$。各ノードに特徴ベクトルがあり、全体を行列 $\mathbf{X}\in\mathbb{R}^{N\times F}$ にする（ここでは、$N$ はバッチではなく**ノード数**）。例：SNS（人とつながり）、分子（原子と結合）、道路網、論文の引用。
- CNN の画像は、どの画素も**同じ形の近傍**（3×3 など）を持つ格子。グラフは、近傍の数がノードごとに違い、**並び順に意味がない**（ノードを並べ替えても同じグラフ）。

| タスク | 何を予測するか | 例 |
|:--|:--|:--|
| **ノード予測** | 各ノードのラベル・値 | 論文の分野の分類（一部のノードだけラベルがある**半教師あり**が多い） |
| **リンク予測** | 2 つのノードの間にエッジがあるか | 友人の推薦、商品の推薦 |
| **グラフ予測** | グラフ全体のラベル・値 | 分子の性質（毒性など）の予測 |

---

## グラフの行列表現 {#matrices}

$$
A_{ij}=\begin{cases}1&((i,j)\in E)\\0&(\text{それ以外})\end{cases},\qquad
D_{ii}=\sum_jA_{ij},\qquad
\mathbf{L}=\mathbf{D}-\mathbf{A},\qquad
\mathbf{L}_{\mathrm{sym}}=\mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}=\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}
$$

- $\mathbf{A}$：**隣接行列**（つながっているか）。$\mathbf{D}$：**次数行列**（各ノードのエッジの数を対角に並べる）。$\mathbf{L}$：**ラプラシアン行列**。$\mathbf{L}_{\mathrm{sym}}$：**正規化ラプラシアン**（次数で正規化）。
- 例：3 つのノードが一列に並んだグラフ（1—2—3）。

$$
\mathbf{A}=\begin{pmatrix}0&1&0\\1&0&1\\0&1&0\end{pmatrix},\quad
\mathbf{D}=\mathrm{diag}(1,2,1),\quad
\mathbf{L}=\begin{pmatrix}1&-1&0\\-1&2&-1\\0&-1&1\end{pmatrix}
$$

- $\mathbf{L}$ の固有値は $0,\ 1,\ 3$、$\mathbf{L}_{\mathrm{sym}}$ の固有値は $0,\ 1,\ 2$。
- **性質**：ノードの値 $\mathbf{x}$（各ノードに 1 つの値）について、$\mathbf{x}^{\top}\mathbf{L}\mathbf{x}=\sum_{(i,j)\in E}(x_i-x_j)^2\ge0$（**つながったノードの値の差の二乗の和**）。上の例で $\mathbf{x}=(1,2,3)^{\top}$ なら $1+1=2$。したがって $\mathbf{L}$ は**半正定値**で、固有値は 0 以上。最小の固有値は 0（固有ベクトルは全部同じ値）。$\mathbf{L}_{\mathrm{sym}}$ の固有値は $[0,2]$ に収まる。

---

## スペクトル法：グラフフーリエ変換と畳み込み {#spectral}

- 普通の信号の畳み込みは、**フーリエ変換して、要素ごとに掛けて、逆変換**と同じ（畳み込み定理）。これをグラフに持ち込む。
- **グラフフーリエ変換**：正規化ラプラシアンを固有値分解する。**固有ベクトルが「周波数ごとの基底」**、固有値 $\lambda$ が周波数。

$$
\mathbf{L}_{\mathrm{sym}}=\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^{\top},\qquad
\mathcal{F}_G[\mathbf{x}]=\mathbf{Q}^{\top}\mathbf{x},\qquad
\mathcal{F}_G^{-1}[\hat{\mathbf{x}}]=\mathbf{Q}\hat{\mathbf{x}},\qquad
\mathcal{F}_G^{-1}\mathcal{F}_G[\mathbf{x}]=\mathbf{Q}\mathbf{Q}^{\top}\mathbf{x}=\mathbf{x}
$$

- $\mathbf{Q}$ の列は正規直交（$\mathbf{Q}\mathbf{Q}^{\top}=\mathbf{I}$）。**小さい固有値**の固有ベクトルは、隣どうしで値が似た**ゆるやかに変化する成分（低周波）**、**大きい固有値**は隣と値が激しく変わる成分（高周波）。
- **スペクトル畳み込み**：フィルタ $\mathbf{g}$ との畳み込みは、周波数領域での要素ごとの積。

$$
\mathbf{x}\ast_G\mathbf{g}=\mathcal{F}_G^{-1}\Bigl[\mathcal{F}_G[\mathbf{x}]\odot\mathcal{F}_G[\mathbf{g}]\Bigr]=\mathbf{Q}\Bigl(\bigl(\mathbf{Q}^{\top}\mathbf{g}\bigr)\odot\bigl(\mathbf{Q}^{\top}\mathbf{x}\bigr)\Bigr)=\mathbf{Q}\,\mathrm{diag}\bigl(\mathbf{Q}^{\top}\mathbf{g}\bigr)\,\mathbf{Q}^{\top}\mathbf{x}=\mathbf{Q}\,\boldsymbol{\Theta}\,\mathbf{Q}^{\top}\mathbf{x}
$$

- **学習するパラメータ**は、周波数領域のフィルタ $\boldsymbol{\Theta}=\mathrm{diag}(\theta_1,\dots,\theta_N)$（要素ごとの積は、ベクトルを対角行列にして掛ける形と同じ）。「逆変換 × パラメータ × フーリエ変換」。
- **多層・多チャネル**にすると、入力チャネル $i$（$f_l$ 個）から出力チャネル $j$ へ：

$$
\mathbf{x}_j^{(l+1)}=\rho\Bigl(\sum_{i=1}^{f_l}\mathbf{Q}\,\boldsymbol{\Theta}^{(l)}_{i,j}\,\mathbf{Q}^{\top}\mathbf{x}^{(l)}_i\Bigr)\qquad(\rho:\text{活性化関数})
$$

- **課題**：① **固有値分解**が必要で、計算量が $O(N^3)$。巨大なグラフでは無理。② 密な行列 $\mathbf{Q}$ を掛けるので、1 回の計算も重い。③ $\boldsymbol{\Theta}$ は固有ベクトル $\mathbf{Q}$ に**依存**するので、**グラフが変わる（ノード数が変わる）と使い回せない**。④ フィルタが**局所的**とは限らない（遠くのノードまで影響する）。

### ChebNet {#chebnet}

- フィルタ $\boldsymbol\Theta$ を、固有値 $\lambda$ の**多項式**で近似すれば、固有値分解を避けられる。**チェビシェフ多項式**を使う。

$$
\boldsymbol\Theta\approx\sum_{k=0}^{K}\theta_k\,T_k(\tilde{\boldsymbol\Lambda}),\qquad \tilde{\boldsymbol\Lambda}=\frac{2}{\lambda_{\max}}\boldsymbol\Lambda-\mathbf{I},\qquad
T_k(x)=2x\,T_{k-1}(x)-T_{k-2}(x),\quad T_0(x)=1,\ T_1(x)=x
$$

$$
\mathbf{x}\ast_G\mathbf{g}\approx\sum_{k=0}^{K}\theta_k\,T_k(\tilde{\mathbf{L}})\,\mathbf{x},\qquad \tilde{\mathbf{L}}=\frac{2}{\lambda_{\max}}\mathbf{L}_{\mathrm{sym}}-\mathbf{I}\qquad(\mathbf{Q}\,T_k(\tilde{\boldsymbol\Lambda})\,\mathbf{Q}^{\top}=T_k(\tilde{\mathbf{L}}))
$$

- $\tilde{\boldsymbol\Lambda}$：固有値を $[-1,1]$ に収めるスケーリング（チェビシェフ多項式の定義域）。**$\mathbf{Q}$ が式から消える**ので、固有値分解も、密な行列も不要。
- $T_k(\tilde{\mathbf{L}})\mathbf{x}$ は、漸化式で**疎行列とベクトルの積だけ**で計算でき、1 回 $O(|E|)$、全体で $O(K|E|)$（エッジの数に比例）。$\mathbf{L}$ の $k$ 乗は **$k$ ホップ先**までしか届かないので、フィルタは**半径 $K$ に局所化**される（$K$ が CNN のカーネルの大きさに相当）。

### GCN {#gcn}

- ChebNet で $K=1$、$\lambda_{\max}\approx2$ とすると、$\tilde{\mathbf{L}}=\mathbf{L}_{\mathrm{sym}}-\mathbf{I}=-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ なので、$\mathbf{x}\ast_G\mathbf{g}\approx\theta_0\mathbf{x}-\theta_1\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\mathbf{x}$。さらにパラメータを 1 つにして $\theta_0=-\theta_1=\theta$ とおくと：

$$
\mathbf{x}\ast_G\mathbf{g}\approx\theta\Bigl(\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\Bigr)\mathbf{x}
$$

- この行列の固有値は $[0,2]$ で、層を重ねると**値が発散・消失**しやすい。そこで**再正規化のトリック**：自己ループを足した $\tilde{\mathbf{A}}=\mathbf{A}+\mathbf{I}$、$\tilde{D}_{ii}=\sum_j\tilde{A}_{ij}$ を使って、$\mathbf{I}+\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}\to\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}=\hat{\mathbf{A}}$ に置き換える。**GCN の 1 層**：

$$
\mathbf{H}^{(l+1)}=\sigma\Bigl(\hat{\mathbf{A}}\,\mathbf{H}^{(l)}\,\mathbf{W}^{(l)}\Bigr),\qquad \hat{\mathbf{A}}=\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2},\qquad \mathbf{H}^{(0)}=\mathbf{X}
$$

| 記号 | 形 | 意味 |
|:--|:--|:--|
| $\mathbf{H}^{(l)}$ | $N\times F_l$ | 第 $l$ 層のノードの特徴 |
| $\mathbf{W}^{(l)}$ | $F_l\times F_{l+1}$ | 学習する重み（**全ノードで共有**。CNN のフィルタに相当） |
| $\hat{\mathbf{A}}$ | $N\times N$ | 自己ループ込みの正規化された隣接行列（疎） |

- 意味：「**自分と隣のノードの特徴を、次数で正規化しながら足し合わせ**（$\hat{\mathbf{A}}\mathbf{H}$）、**線形変換**（$\mathbf{W}$）して、活性化（$\sigma$）」。ノード分類（半教師あり）では 2 層にして、最後にソフトマックスと**ラベルのあるノードだけの交差エントロピー**で学習する。
- **例**（上の 1—2—3 のグラフ、特徴 $\mathbf{x}=(1,2,3)^{\top}$、$\mathbf{W}=1$、活性化なし）：$\tilde{\mathbf{D}}=\mathrm{diag}(2,3,2)$ なので、$\hat{A}_{ij}=\tilde{A}_{ij}/\sqrt{\tilde{d}_i\tilde{d}_j}$、すなわち $\hat{\mathbf{A}}=\begin{pmatrix}1/2&1/\sqrt6&0\\1/\sqrt6&1/3&1/\sqrt6\\0&1/\sqrt6&1/2\end{pmatrix}$。$\hat{\mathbf{A}}\mathbf{x}\approx(1.32,\ 2.30,\ 2.32)^{\top}$。両端のノードの特徴 1 と 3 が、隣のノード 2 に引き寄せられて近づく。

---

## 空間法：メッセージパッシング {#spatial}

- 固有値分解を使わず、**グラフの上で近傍の情報を集める**見方。各ノード $i$ が、近傍 $\mathcal{N}(i)$ から**メッセージ**を受け取り（**集約**）、自分の状態を**更新**する。

$$
\mathbf{h}_i^{(l+1)}=\sigma\Bigl(\sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}\,\mathbf{W}^{(l)\top}\mathbf{h}_j^{(l)}\Bigr)\qquad(\text{GCN は }c_{ij}=\sqrt{\tilde{d}_i\tilde{d}_j}\text{、}\mathcal{N}(i)\text{ に自分を含む})
$$

- GCN の式を、ノードごとに書いたもの（1 層ぶん、**1 ホップ**）。層を重ねるほど、より遠いノードの情報が届く（$L$ 層で $L$ ホップ）。

| 手法 | 集約と更新 | 特徴 |
|:--|:--|:--|
| **GraphSAGE** | 近傍を**サンプリング**して、平均・LSTM・max プーリングなどで集約し、自分の特徴と**連結**して変換 | 学習時にいなかった**新しいノード・グラフ**にも使える（**帰納的**）。大きなグラフでも、ミニバッチで学習できる |
| **GAT** | 近傍への重みを **Attention** で学習：$\alpha_{ij}=\mathrm{softmax}_j\bigl(\mathrm{LeakyReLU}(\mathbf{a}^{\top}[\mathbf{W}\mathbf{h}_i\,\Vert\,\mathbf{W}\mathbf{h}_j])\bigr)$、$\mathbf{h}_i'=\sigma(\sum_j\alpha_{ij}\mathbf{W}\mathbf{h}_j)$ | 近傍の**重要度**がデータで決まる。GCN の重み $1/c_{ij}$ は次数で固定 |
| **MPNN** | メッセージ関数 $M$、更新関数 $U$、読み出し $R$：$\mathbf{m}_i=\sum_jM(\mathbf{h}_i,\mathbf{h}_j,\mathbf{e}_{ij})$、$\mathbf{h}_i'=U(\mathbf{h}_i,\mathbf{m}_i)$ | **枠組み**。エッジの特徴も使える（分子など）。上の多くは特別な場合 |
| **GIN** | $\mathbf{h}_i'=\mathrm{MLP}\bigl((1+\epsilon)\mathbf{h}_i+\sum_{j\in\mathcal{N}(i)}\mathbf{h}_j\bigr)$ | **和**で集約（平均や最大では区別できない構造がある）。**WL テスト**（グラフの同型の判定）と同等の識別力 |

- **グラフ全体の予測**には、全ノードの特徴を 1 本のベクトルにまとめる**読み出し**（和・平均・最大のプーリング）をして、全結合層に入れる。ノードの並び順に依らない（**順序不変**）関数にする。
- **スペクトル法**は固定のグラフ全体を使う（**トランスダクティブ**）。**空間法**は局所の構造だけを使うので、**別のグラフにも適用できる**（**インダクティブ**）。

---

## 過平滑化と限界 {#oversmoothing}

- GCN の層を重ねると、$\hat{\mathbf{A}}^{L}\mathbf{X}$ が、ノードによらない**同じような値**に収束する（**過平滑化**）。上の例で $\hat{\mathbf{A}}$ を繰り返し掛けると、$(1,2,3)\to(1.32,\ 2.30,\ 2.32)\to\dots\to(1.84,\ 2.26,\ 1.84)$ となり、両端のノードが**区別できなくなる**。
- 原因：$\hat{\mathbf{A}}$ は低周波の成分だけを残す**ローパスフィルタ**で、固有値 $1-\tilde\lambda\in(-1,1]$ のうち 1 の成分以外が、層を重ねるごとに 0 に近づくから。そのため GCN は**浅い（2〜3 層）**のが普通。対策は、スキップ接続、正規化、層ごとの出力を連結する方法など。
- ほかの課題：遠いノードの情報が少数のノードに圧縮される（**過圧縮**）、グラフの規模が大きいときの計算（サンプリングで対処）。

---

## 試験の着眼点

- **ラプラシアン** $\mathbf{L}=\mathbf{D}-\mathbf{A}$（次数行列 $-$ 隣接行列）。正規化は $\mathbf{I}-\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$。$\mathbf{x}^{\top}\mathbf{L}\mathbf{x}=\sum_{(i,j)\in E}(x_i-x_j)^2$。
- **グラフフーリエ変換**は $\mathbf{Q}^{\top}\mathbf{x}$、逆変換は $\mathbf{Q}\hat{\mathbf{x}}$（$\mathbf{Q}$ は正規化ラプラシアンの固有ベクトル）。スペクトル畳み込みは $\mathbf{Q}\boldsymbol\Theta\mathbf{Q}^{\top}\mathbf{x}$：「逆変換・フィルタ・変換」。
- Spectral Conv の課題：**固有値分解が重い**（$O(N^3)$）、**グラフごとに $\mathbf{Q}$ が違うので、パラメータを別のグラフに使えない**。
- **ChebNet**：チェビシェフ多項式で近似して**固有値分解を避ける**。フィルタは**$K$ ホップに局所化**。**GCN** は $K=1$ の近似＋**自己ループを足した再正規化**：$\mathbf{H}'=\sigma(\tilde{\mathbf{D}}^{-1/2}\tilde{\mathbf{A}}\tilde{\mathbf{D}}^{-1/2}\mathbf{H}\mathbf{W})$。
- **空間法**は集約と更新。**GraphSAGE**（サンプリング・帰納的）、**GAT**（Attention で近傍に重み）、**GIN**（和で集約・識別力が高い）。
- 層を重ねすぎると**過平滑化**。

---

## 参考

- [A Comprehensive Survey on Graph Neural Networks（Wu ほか, 2019）](https://arxiv.org/abs/1901.00596)
- [Spectral Networks and Locally Connected Networks on Graphs（Bruna ほか, 2013）](https://arxiv.org/abs/1312.6203)
- [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering（ChebNet, Defferrard ほか, 2016）](https://arxiv.org/abs/1606.09375)
- [Semi-Supervised Classification with Graph Convolutional Networks（GCN, Kipf, Welling, 2016）](https://arxiv.org/abs/1609.02907)
- [Inductive Representation Learning on Large Graphs（GraphSAGE, Hamilton ほか, 2017）](https://arxiv.org/abs/1706.02216)
- [Graph Attention Networks（Veličković ほか, 2017）](https://arxiv.org/abs/1710.10903)
- [Neural Message Passing for Quantum Chemistry（MPNN, Gilmer ほか, 2017）](https://arxiv.org/abs/1704.01212)
- [How Powerful are Graph Neural Networks?（GIN, Xu ほか, 2018）](https://arxiv.org/abs/1810.00826)
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges（Bronstein ほか, 2021）](https://arxiv.org/abs/2104.13478)
