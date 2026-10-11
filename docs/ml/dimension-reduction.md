# 次元圧縮

<p class="kw">キーワード：主成分分析（PCA）、寄与率、線形判別分析（LDA）、独立成分分析（ICA）、SNE、Crowding Problem、t-SNE、UMAP</p>

!!! abstract "要点"
    - **次元圧縮**は、データの情報をなるべく残したまま $D$ 次元を $D'$ 次元（$D'<D$）に落とす教師なし学習。目的は可視化・ノイズ除去・計算量の削減・[次元の呪い](evaluation.md#curse)の緩和。
    - **PCA** は射影後の**分散が最大**になる向きを探す。共分散行列の固有値問題（または特異値分解）に帰着し、解が解析的に出る。
    - 線形の PCA/LDA/ICA に対し、**t-SNE・UMAP** は近傍関係を保つ非線形の可視化手法。距離の大小の読み方に注意が要る。

---

## 主成分分析（PCA） {#pca}

データ行列を $\mathbf{X}\in\mathbb{R}^{N\times D}$、平均を $\boldsymbol\mu$、中心化したものを $\tilde{\mathbf{x}}=\mathbf{x}-\boldsymbol\mu$ とする。

$$
\begin{aligned}
&\text{共分散行列} && \boldsymbol{\Sigma}=\frac1N\sum_{i=1}^{N}\tilde{\mathbf{x}}^{(i)}\tilde{\mathbf{x}}^{(i)\top}=\frac1N\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}\quad(D\times D) \\[3mm]
&\text{最適化問題} && \mathbf{w}_1=\operatorname*{arg\,max}_{\|\mathbf{w}\|=1}\ \mathbf{w}^{\top}\boldsymbol{\Sigma}\,\mathbf{w}\qquad\bigl(\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}\ \text{は射影後の分散}\bigr) \\[3mm]
&\text{固有値問題} && \boldsymbol{\Sigma}\mathbf{w}=\lambda\mathbf{w},\qquad \boldsymbol{\Sigma}\mathbf{W}=\mathbf{W}\boldsymbol{\Lambda}\ \ (\lambda_1\ge\cdots\ge\lambda_D\ge0) \\[3mm]
&\text{射影} && \mathbf{z}=\mathbf{W}_{D'}^{\top}\tilde{\mathbf{x}},\qquad \mathbf{W}_{D'}\in\mathbb{R}^{D\times D'}\ \text{（上位 }D'\text{ 個の固有ベクトルを列に並べる）} \\[3mm]
&\text{寄与率} && r_j=\frac{\lambda_j}{\sum_{l=1}^{D}\lambda_l},\qquad \text{累積寄与率}\ \ \frac{\sum_{j=1}^{D'}\lambda_j}{\sum_{l=1}^{D}\lambda_l}
\end{aligned}
$$

??? note "導出：なぜ固有値問題になるか"
    ラグランジュ関数 $\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}-\lambda(\mathbf{w}^{\top}\mathbf{w}-1)$ を $\mathbf{w}$ で微分して 0 とおくと、$2\boldsymbol{\Sigma}\mathbf{w}-2\lambda\mathbf{w}=0$、すなわち $\boldsymbol{\Sigma}\mathbf{w}=\lambda\mathbf{w}$。
    このとき $\mathbf{w}^{\top}\boldsymbol{\Sigma}\mathbf{w}=\lambda\mathbf{w}^{\top}\mathbf{w}=\lambda$ なので、分散を最大にするのは**最大固有値**の固有ベクトル。第2主成分は $\mathbf{w}_1$ と直交する制約の下で同様に求め、固有値の大きい順に並ぶ。

- **固有値 $\lambda_j$ ＝ 第 $j$ 主成分の分散**。全分散は $\operatorname{tr}\boldsymbol\Sigma=\sum_j\lambda_j$ で、回転しても変わらない。
- 捨てた成分の分散の和 $\sum_{j>D'}\lambda_j$ が、平均二乗の**再構成誤差**（$\hat{\mathbf{x}}=\boldsymbol\mu+\mathbf{W}_{D'}\mathbf{z}$ との差）に等しい。分散最大化と再構成誤差最小化は同じ問題。
- 主成分は互いに直交し、主成分得点どうしは無相関。
- **主成分得点**：$\mathbf{z}_k=\tilde{\mathbf{X}}\mathbf{w}_k$（各データを第 $k$ 軸に射影した座標）。
- **主成分負荷量**：元の変数 $x_j$ と主成分得点 $z_k$ の相関係数 $=w_{jk}\sqrt{\lambda_k}/\sigma_j$（$\sigma_j$ は $x_j$ の標準偏差）。固有ベクトルの成分 $w_{jk}$ に比例し、単位に左右されないので、軸の意味を解釈しやすい。
- 累積寄与率が 70〜90% になる $D'$ を選ぶ、または寄与率の折れ線（スクリープロット）の肘で決める。

<div class="ew" data-widget="dimred-pca"></div>

!!! tip "動かしてみる"
    - $\theta$ を動かすと、右の曲線 $s^2(\theta)$ が上下します。山の頂上（$\lambda_1$）に重なる向きが第1主成分で、谷底は $\lambda_2$ に一致します。
    - 残差（点から軸へ下ろした線の長さの二乗平均）は、$s^2$ が最大のときに最小になります。$s^2+\text{残差}=\operatorname{tr}\boldsymbol\Sigma$ は常に一定です。
    - $r$ を 1 に近づけると丸いデータになり、$\lambda_1\approx\lambda_2$ で曲線が平らになります。どの向きでも分散が同じなので、主成分が定まりません。

### 実装上のポイント

- **標準化**：単位の異なる変数を混ぜるときは、各変数を平均 0・分散 1 にそろえてから行う（共分散行列ではなく**相関行列**の固有値分解になる）。そろえないと、値の大きい変数が主成分を支配する。
- **特異値分解（SVD）**：$\tilde{\mathbf{X}}=\mathbf{U}\mathbf{S}\mathbf{V}^{\top}$ とすると、$\mathbf{W}=\mathbf{V}$、$\lambda_j=s_j^2/N$、得点 $=\mathbf{U}\mathbf{S}$。$\boldsymbol\Sigma$ を明示的に作らずに済み、数値的に安定なので実装ではこちらが標準（[固有値分解・特異値分解](../math/eigen-svd.md)）。
- $D>N$ のときは $\operatorname{rank}\boldsymbol\Sigma\le N-1$ なので、$N\times N$ のグラム行列 $\tilde{\mathbf{X}}\tilde{\mathbf{X}}^{\top}$ を使うほうが小さく済む。
- **カーネル PCA**：内積をカーネルに置き換えて、特徴空間で PCA をする非線形版。カーネル行列の固有値分解になる（[SVM](svm.md) のカーネルトリックと同じ考え方）。

---

## 線形判別分析（LDA） {#lda}

PCA と違って**ラベルを使う**教師あり手法。クラスのまとまりが最も際立つ向きを探す。

$$
\begin{aligned}
&\text{クラス内散布} && \mathbf{S}_W=\sum_{k=1}^{K}\sum_{\mathbf{x}^{(i)}\in\mathcal{C}_k}\bigl(\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr)\bigl(\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr)^{\top} \\[3mm]
&\text{クラス間散布} && \mathbf{S}_B=\sum_{k=1}^{K}N_k\,(\boldsymbol\mu_k-\boldsymbol\mu)(\boldsymbol\mu_k-\boldsymbol\mu)^{\top} \\[3mm]
&\text{フィッシャーの基準} && \mathbf{w}=\operatorname*{arg\,max}_{\mathbf{w}}\ \frac{\mathbf{w}^{\top}\mathbf{S}_B\mathbf{w}}{\mathbf{w}^{\top}\mathbf{S}_W\mathbf{w}} \\[3mm]
&\text{解} && \mathbf{S}_W^{-1}\mathbf{S}_B\,\mathbf{w}=\lambda\mathbf{w}\quad\text{（一般化固有値問題）}
\end{aligned}
$$

- $\boldsymbol\mu_k$ はクラス $k$ の平均、$\boldsymbol\mu$ は全体の平均、$N_k$ はクラス $k$ のデータ数。
- 「クラス間で離れ、クラス内で縮む」向きを選ぶ。分子がクラス間の分散、分母がクラス内の分散。
- $\operatorname{rank}\mathbf{S}_B\le K-1$ なので、取り出せる軸は**最大 $K-1$ 本**。2クラスなら $\mathbf{w}\propto\mathbf{S}_W^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)$ の1本だけ。
- **分類器としての LDA**：各クラスが**共通の共分散行列** $\boldsymbol\Sigma$ をもつ正規分布に従うと仮定し、事後確率が最大のクラスを選ぶ。

$$
\hat{y}=\operatorname*{arg\,max}_{k}\Bigl(\log P(\mathcal{C}_k)-\tfrac12\boldsymbol\mu_k^{\top}\boldsymbol\Sigma^{-1}\boldsymbol\mu_k+\mathbf{x}^{\top}\boldsymbol\Sigma^{-1}\boldsymbol\mu_k\Bigr)
$$

- 括弧の中は $\mathbf{x}$ の**線形関数**なので、決定境界は直線（超平面）。共分散がクラスごとに違うと仮定すると二次関数の境界になる（QDA）。
- 共分散を共通にする仮定が、[ナイーブベイズ](bayes-gp-crf.md#naive-bayes)（対角の共分散）との違い。

---

## 独立成分分析（ICA） {#ica}

観測 $\mathbf{x}$ が、互いに**統計的に独立**な信号 $\mathbf{s}$ の線形混合だと考え、混合を元に戻す。代表例は、複数の話者の声が混ざった録音の分離（カクテルパーティ問題）。

$$
\begin{aligned}
&\text{モデル} && \mathbf{x}=\mathbf{A}\mathbf{s},\qquad \hat{\mathbf{s}}=\mathbf{W}\mathbf{x}\quad(\mathbf{W}=\mathbf{A}^{-1}) \\[2mm]
&\text{対数尤度} && \log p(\mathbf{x})=\sum_{i=1}^{N}\sum_{k=1}^{D}\log p_k\bigl(\mathbf{w}_k^{\top}\mathbf{x}^{(i)}\bigr)+N\log|\det\mathbf{W}| \\[2mm]
&\text{ネゲントロピー（近似）} && J(y)\propto\bigl[\mathbb{E}\{G(y)\}-\mathbb{E}\{G(\nu)\}\bigr]^2,\quad G(u)=\log\cosh u \\[2mm]
&\text{FastICA の更新} && \mathbf{w}\leftarrow\mathbb{E}\{\mathbf{x}\,g(\mathbf{w}^{\top}\mathbf{x})\}-\mathbb{E}\{g'(\mathbf{w}^{\top}\mathbf{x})\}\,\mathbf{w},\quad \mathbf{w}\leftarrow\mathbf{w}/\|\mathbf{w}\|\quad(g=G')
\end{aligned}
$$

- $\nu$ は標準正規分布に従う変数。ネゲントロピーは「正規分布からどれだけ外れているか」（正規分布で 0、それ以外で正）。
- **手順**：①データを**白色化**（無相関・分散 1。ここまでは PCA で済む）、②白色化したデータの中で、**非ガウス性**（尖度やネゲントロピー）が最大になる向きを探す、③ FastICA の固定点反復で $\mathbf{W}$ を更新する。
- 独立な信号の和は正規分布に近づく（中心極限定理）ので、**混ぜたものより元の信号のほうが非ガウス的**。これが非ガウス性を最大化する根拠。
- 限界：成分の**順序と大きさ（符号）は決まらない**。ガウス分布の信号は高々1つしか分離できない。

| | PCA | ICA |
|:--|:--|:--|
| 基準 | 分散最大（2次の統計量） | 独立性・非ガウス性（高次の統計量） |
| 成分の関係 | 直交・無相関 | 独立（無相関より強い） |
| 順序 | 固有値の順に決まる | 決まらない |

---

## t-SNE {#tsne}

高次元の**近傍関係**を2〜3次元に写す可視化手法。SNE → 対称化 → t 分布の順に改良された。

$$
\begin{aligned}
&\text{高次元の類似度} && p_{j\mid i}=\frac{\exp\!\bigl(-\|\mathbf{x}_i-\mathbf{x}_j\|^2/2\sigma_i^2\bigr)}{\sum_{k\ne i}\exp\!\bigl(-\|\mathbf{x}_i-\mathbf{x}_k\|^2/2\sigma_i^2\bigr)},\qquad p_{ij}=\frac{p_{j\mid i}+p_{i\mid j}}{2N} \\[3mm]
&\text{低次元の類似度} && q_{ij}=\frac{\bigl(1+\|\mathbf{y}_i-\mathbf{y}_j\|^2\bigr)^{-1}}{\sum_{k\ne l}\bigl(1+\|\mathbf{y}_k-\mathbf{y}_l\|^2\bigr)^{-1}} \\[3mm]
&\text{目的関数} && L=D_{\mathrm{KL}}(P\,\|\,Q)=\sum_{i\ne j}p_{ij}\log\frac{p_{ij}}{q_{ij}} \\[3mm]
&\text{勾配} && \frac{\partial L}{\partial\mathbf{y}_i}=4\sum_{j}(p_{ij}-q_{ij})\,(\mathbf{y}_i-\mathbf{y}_j)\,\bigl(1+\|\mathbf{y}_i-\mathbf{y}_j\|^2\bigr)^{-1} \\[3mm]
&\text{更新} && \mathbf{y}_i\leftarrow\mathbf{y}_i-\eta\,\frac{\partial L}{\partial\mathbf{y}_i}
\end{aligned}
$$

- $\mathbf{x}_i$ は高次元の点、$\mathbf{y}_i$ は低次元に写した点（学習する対象）。
- **SNE**：高次元では点 $i$ を中心とするガウス分布の確率 $p_{j\mid i}$、低次元でも同様のガウス分布 $q_{j\mid i}$ を作り、KL ダイバージェンスの和 $\sum_i D_{\mathrm{KL}}(P_i\|Q_i)$ を最小化する。
- $\sigma_i$ は点ごとに、**パープレキシティ** $\mathrm{Perp}(P_i)=2^{H(P_i)}$（$H$ は $p_{j\mid i}$ のエントロピー、底は 2）が指定値（5〜50 程度）になるよう二分探索で決める。有効な近傍点の数に相当し、密度の違いを吸収する。
- **Crowding Problem（混み合い問題）**：高次元の球面の近くには多数の点が等距離に並べるが、2次元には同じ数の点を置く余地が少ない。距離が中くらいの点が中心に押し込まれ、クラスタがつぶれる。
- **t-SNE** は低次元側にだけ**自由度1の t 分布**（裾が重い）を使う。遠い点どうしの $q_{ij}$ が減りにくいので、中距離の点を遠くへ置いても罰が小さく、混み合いが解消される。
- $p_{ij}$ を対称化して全体で正規化すると、勾配が上の簡潔な式になり、学習が安定する（外れ値にも少なくとも 1 点分の重みが残る）。
- KL は非対称。**近い点を遠くに置く**（$p$ 大・$q$ 小）誤りは強く罰せられ、**遠い点を近くに置く**誤りは弱く罰せられる。そのため局所構造は保たれるが、大域構造は保証されない。

!!! warning "読み方の注意"
    - クラスタ間の距離や、クラスタの大きさ・密度に意味はない。パープレキシティや初期値で図が変わる（再現には乱数の固定が必要）。
    - 新しい点を後から写す関数を持たない。計算量は素朴には $O(N^2)$（Barnes-Hut 近似で $O(N\log N)$）。

---

## UMAP {#umap}

局所的な近傍グラフ（ファジィ集合）を高次元・低次元の両方に作り、その**クロスエントロピーを最小化**する。t-SNE より速く、大域構造も比較的保つとされる。

$$
\begin{aligned}
&\text{高次元の重み} && w_{ij}=\exp\!\Bigl(-\frac{\max\bigl(0,\ d(\mathbf{x}_i,\mathbf{x}_j)-\rho_i\bigr)}{\sigma_i}\Bigr) \\[3mm]
&\text{対称化} && \bar{w}_{ij}=w_{ij}+w_{ji}-w_{ij}\,w_{ji} \\[3mm]
&\text{低次元の重み} && v_{ij}=\bigl(1+a\,\|\mathbf{y}_i-\mathbf{y}_j\|^{2b}\bigr)^{-1} \\[3mm]
&\text{目的関数} && L=\sum_{i\ne j}\Bigl[\bar{w}_{ij}\log\frac{\bar{w}_{ij}}{v_{ij}}+(1-\bar{w}_{ij})\log\frac{1-\bar{w}_{ij}}{1-v_{ij}}\Bigr]
\end{aligned}
$$

- $\rho_i$：点 $i$ から最近傍までの距離。最近傍の重みが必ず 1 になり、**局所の密度の違いを正規化**する。
- $\sigma_i$：$\sum_j w_{ij}=\log_2 k$（$k$ は近傍数）になるよう点ごとに決める。t-SNE のパープレキシティに相当する。
- 対称化は確率的 t-コノルム（和 $-$ 積）で、辺 $i$–$j$ が「どちらか一方から見て存在する」確率。
- $a,b$：低次元の重みの形を決める定数。`min_dist`（低次元で点を詰めてよい最小距離）から曲線あてはめで決まる。
- 第1項は**引力**（高次元で近い対を近づける）、第2項は**斥力**（高次元で遠い対を遠ざける）。
- 手順：①近傍グラフを作る、②スペクトル埋め込みなどで低次元を初期化する、③確率的勾配降下法（負例サンプリング）で最小化する。
- 新しい点を既存の埋め込みへ写せる。距離は任意の尺度を使える。

| 手法 | 教師 | 線形 | 解 | 主な用途 |
|:--|:--|:--|:--|:--|
| PCA | なし | 線形 | 解析的（固有値分解） | 圧縮・前処理・可視化 |
| LDA | あり | 線形 | 解析的（一般化固有値問題） | 分類のための次元圧縮 |
| ICA | なし | 線形 | 反復（FastICA） | 信号の分離 |
| t-SNE | なし | 非線形 | 反復（勾配降下） | 可視化（局所構造） |
| UMAP | なし | 非線形 | 反復（SGD） | 可視化・前処理 |

---

## 試験の着眼点

- PCA は「分散最大」＝「再構成誤差最小」。固有値は分散、寄与率は $\lambda_j/\sum\lambda$。共分散行列は $\frac1N\tilde{\mathbf{X}}^{\top}\tilde{\mathbf{X}}$（$D\times D$）。
- 射影行列 $\mathbf{W}_{D'}$ は $D\times D'$、$\mathbf{z}=\mathbf{W}_{D'}^{\top}\tilde{\mathbf{x}}$。形で検算できる。
- PCA は教師なし、LDA は教師あり。LDA の軸は最大 $K-1$ 本。
- ICA は無相関ではなく**独立**を使い、非ガウス性を最大化する。順序と大きさは不定。
- t-SNE は SNE の混み合い問題を、低次元側の **t 分布**で解く。KL ダイバージェンスを最小化する。クラスタ間の距離は読まない。
- UMAP の目的関数は、ファジィ集合間のクロスエントロピー。

---

## 参考

- [Decomposing signals in components（scikit-learn 公式ドキュメント）](https://scikit-learn.org/stable/modules/decomposition.html)
- [Visualizing Data using t-SNE（van der Maaten, Hinton, 2008）](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction（McInnes, Healy, Melville, 2018）](https://arxiv.org/abs/1802.03426)
- Independent component analysis: algorithms and applications（Hyvärinen, Oja, 2000, Neural Networks）
