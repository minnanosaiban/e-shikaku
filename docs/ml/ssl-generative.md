# 自己教師あり学習・生成モデルの基礎

<p class="kw">キーワード：自己教師あり学習、オートエンコーダ、変分オートエンコーダ（VAE）、ELBO、再パラメータ化トリック、VQ-VAE、対照学習、生成モデル、敵対的生成ネットワーク（GAN）、隠れマルコフモデル（HMM）</p>

!!! abstract "要点"
    - **自己教師あり学習**は、データ自身から作った課題（復元・マスク予測・同じ画像の別の見え方を近づける）で表現を学ぶ。ラベルは要らない。
    - **VAE** は潜在変数 $\mathbf{z}$ を確率変数として扱い、対数尤度の下界 **ELBO**（再構成項 $-$ KL 項）を最大化する。サンプリングは再パラメータ化トリックで微分可能にする。
    - **GAN** は、生成器と識別器が min-max ゲームをして、生成分布を本物のデータ分布に近づける。尤度を直接は使わない。

---

## 自己教師あり学習とは {#ssl}

データの一部から別の一部を当てる**疑似課題**（プレテキストタスク）を自分で作り、解く過程で有用な特徴表現を得る。得た表現は、少量のラベルによる微調整（ファインチューニング）や、表現を固定した線形分類（リニアプロービング）で下流のタスクに使う。

| 方式 | 課題 | 例 |
|:--|:--|:--|
| 復元 | 入力そのものを再構成する | オートエンコーダ、ノイズ除去 AE |
| 欠損の予測 | 隠した部分を当てる | BERT（マスクされた語）、MAE（隠した画像パッチ） |
| 次の予測 | 過去から次の要素を当てる | GPT（次の単語）、CPC |
| 対照学習 | 同じ元データ由来の2つのビューを近づけ、他を遠ざける | SimCLR、MoCo |
| 自己蒸留 | 同じ入力の2つのビューで、一方のネットワークの出力を他方に合わせる | BYOL、DINO |

- 利点：ラベル付けが不要で、大量の未ラベルデータを使える。データ拡張（[データ拡張](../dl/augmentation.md)）が「何を同じとみなすか」を決める。

---

## オートエンコーダ（AE） {#ae}

入力を**ボトルネック**（低次元の潜在表現）に通して、入力自身を復元するよう学習する。

$$
\begin{aligned}
&\text{エンコード} && \mathbf{z}=f_\phi(\mathbf{x}) \\[2mm]
&\text{デコード} && \hat{\mathbf{x}}=g_\theta(\mathbf{z}) \\[2mm]
&\text{損失} && L(\theta,\phi)=\frac1N\sum_{i=1}^{N}\bigl\|\mathbf{x}^{(i)}-g_\theta\bigl(f_\phi(\mathbf{x}^{(i)})\bigr)\bigr\|^2 \\[2mm]
&\text{更新} && \theta\leftarrow\theta-\eta\nabla_\theta L,\qquad \phi\leftarrow\phi-\eta\nabla_\phi L
\end{aligned}
$$

- $\mathbf{z}$ は**潜在変数**（このページでは $\mathbf{z}$ を潜在変数に使う。層の出力ではない）。$\mathbf{z}$ の次元が入力より小さい（**不完全**）ことで、恒等写像を覚えるのを防ぎ、重要な特徴だけを残させる。
- 活性化が線形で損失が二乗誤差のとき、AE は [PCA](dimension-reduction.md#pca) と同じ部分空間を学ぶ。非線形にすると PCA より曲がった構造を扱える。
- **ノイズ除去 AE（DAE）**：入力にノイズを加えた $\tilde{\mathbf{x}}\sim q(\tilde{\mathbf{x}}\mid\mathbf{x})$ を入力し、**元の** $\mathbf{x}$ を復元させる。$L=\frac1N\sum\|\mathbf{x}^{(i)}-g_\theta(f_\phi(\tilde{\mathbf{x}}^{(i)}))\|^2$。少ないデータでもよい特徴が得られ、恒等写像を学べなくなる。
- **スパース AE**：潜在表現に L1 などの罰則を課し、少数のユニットだけが活性化するようにする。**縮小 AE（CAE）**：エンコーダのヤコビ行列のノルム $\|\partial\mathbf{z}/\partial\mathbf{x}\|_F^2$ を罰則にして、入力の小さな変化に鈍感な表現にする。
- 通常の AE は潜在空間の構造が不明で、ランダムな $\mathbf{z}$ からうまく生成できるとは限らない。そこで潜在変数に確率分布を与えたのが VAE。

---

## 変分オートエンコーダ（VAE） {#vae}

潜在変数付きの生成モデル。事前分布 $p(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$ から $\mathbf{z}$ をとり、デコーダ $p_\theta(\mathbf{x}\mid\mathbf{z})$ で $\mathbf{x}$ を生成する。目標は周辺尤度 $p_\theta(\mathbf{x})=\int p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ の最大化だが、積分が解けず事後分布 $p_\theta(\mathbf{z}\mid\mathbf{x})$ も求まらない。そこで事後分布を**エンコーダ** $q_\phi(\mathbf{z}\mid\mathbf{x})$ で近似する（変分推論）。

$$
\begin{aligned}
&\text{推論モデル} && q_\phi(\mathbf{z}\mid\mathbf{x})=\mathcal{N}\bigl(\mathbf{z}\mid\boldsymbol\mu_\phi(\mathbf{x}),\ \operatorname{diag}\,\boldsymbol\sigma_\phi^2(\mathbf{x})\bigr) \\[3mm]
&\text{対数尤度の分解} && \log p_\theta(\mathbf{x})=\mathrm{ELBO}(\theta,\phi;\mathbf{x})+D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p_\theta(\mathbf{z}\mid\mathbf{x})\bigr)\ \ge\ \mathrm{ELBO} \\[3mm]
&\text{ELBO} && \mathrm{ELBO}=\underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\bigl[\log p_\theta(\mathbf{x}\mid\mathbf{z})\bigr]}_{\text{再構成項（大きくする）}}-\underbrace{D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\bigr)}_{\text{正則化項（小さくする）}} \\[3mm]
&\text{再パラメータ化} && \mathbf{z}=\boldsymbol\mu_\phi(\mathbf{x})+\boldsymbol\sigma_\phi(\mathbf{x})\odot\boldsymbol\varepsilon,\qquad \boldsymbol\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I}) \\[3mm]
&\text{KL（閉じた式）} && D_{\mathrm{KL}}=-\frac12\sum_{j=1}^{J}\bigl(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\bigr) \\[3mm]
&\text{損失} && L=-\mathrm{ELBO}=\underbrace{\mathbb{E}\bigl[-\log p_\theta(\mathbf{x}\mid\mathbf{z})\bigr]}_{\text{再構成誤差}}+\underbrace{D_{\mathrm{KL}}\bigl(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\bigr)}_{\text{正則化項}}
\end{aligned}
$$

??? note "導出：ELBO"
    $\log p_\theta(\mathbf{x})=\mathbb{E}_{q}[\log p_\theta(\mathbf{x})]$ に $p_\theta(\mathbf{x})=p_\theta(\mathbf{x},\mathbf{z})/p_\theta(\mathbf{z}\mid\mathbf{x})$ を代入し、$q_\phi(\mathbf{z}\mid\mathbf{x})$ を分子分母に掛けると
    $\log p_\theta(\mathbf{x})=\mathbb{E}_q\bigl[\log\frac{p_\theta(\mathbf{x},\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})}\bigr]+D_{\mathrm{KL}}(q_\phi\|p_\theta(\mathbf{z}\mid\mathbf{x}))$。
    第1項が ELBO。さらに $p_\theta(\mathbf{x},\mathbf{z})=p_\theta(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})$ を使うと「再構成項 $-$ KL」の形になる。KL は 0 以上なので ELBO は $\log p_\theta(\mathbf{x})$ の下界で、$q$ が真の事後分布に一致したとき等号となる。

- **再構成項**：サンプル $\mathbf{z}\sim q_\phi$ をデコーダに通し、$\mathbf{x}$ の対数尤度を評価する。$\mathbf{x}$ が 0/1 ならベルヌーイで二値交差エントロピー $-\sum_d\bigl[x_d\log\hat{x}_d+(1-x_d)\log(1-\hat{x}_d)\bigr]$、実数なら分散固定の正規分布で二乗誤差（定数倍）になる。
- **KL 項**：近似事後分布を事前分布 $\mathcal{N}(\mathbf{0},\mathbf{I})$ に近づける正則化。検算：$\boldsymbol\mu=\mathbf{0},\ \boldsymbol\sigma=\mathbf{1}$ なら各項 $1+0-0-1=0$ で KL $=0$。
- **再パラメータ化トリック**：$\mathbf{z}$ を直接サンプリングすると確率的な操作を通して勾配が流れない。乱数 $\boldsymbol\varepsilon$ を外に出し、$\mathbf{z}$ を $\boldsymbol\mu,\boldsymbol\sigma$ の決定的な関数にすれば、[誤差逆伝播法](../dl/backprop.md)で学習できる。
- 生成時はエンコーダを捨て、$\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ をデコーダに通す。KL 項が潜在空間を事前分布の周りに整えているので、サンプルが意味のあるデータになる。

**課題**

- 生成画像が**ぼやける**（二乗誤差・単峰の正規分布が平均的な像を好むため）→ GAN が改善する。
- **事後崩壊**（posterior collapse）：デコーダが強いと、潜在変数を無視して $q_\phi(\mathbf{z}\mid\mathbf{x})\approx p(\mathbf{z})$ となり、KL 項が 0 に張り付く。学習初期は KL の重みを小さくして徐々に増やす（**KL アニーリング**）、潜在変数を離散化する（VQ-VAE）などで対処する。
- **$\beta$-VAE**：KL 項に係数 $\beta>1$ を掛けて、潜在変数の各次元が独立な要因を表すよう促す（もつれの解消）。

---

## VQ-VAE {#vqvae}

潜在変数を**離散**にする。エンコーダ出力 $\mathbf{z}_e$ を、学習する**コードブック** $\{\mathbf{e}_1,\dots,\mathbf{e}_K\}$ の最も近いベクトルに置き換える（ベクトル量子化）。高品質な画像・音声の生成や、離散トークン化に使われる。

$$
\begin{aligned}
&\text{量子化} && \mathbf{z}_q=\mathbf{e}_{k^*},\qquad k^*=\operatorname*{arg\,min}_{k}\|\mathbf{z}_e-\mathbf{e}_k\|^2 \\[3mm]
&\text{損失} && L=\underbrace{\|\mathbf{x}-\hat{\mathbf{x}}\|^2}_{\text{再構成誤差}}+\underbrace{\bigl\|\mathrm{sg}[\mathbf{z}_e]-\mathbf{e}_{k^*}\bigr\|^2}_{\text{コードブック損失}}+\underbrace{\beta\,\bigl\|\mathbf{z}_e-\mathrm{sg}[\mathbf{e}_{k^*}]\bigr\|^2}_{\text{コミットメント損失}} \\[3mm]
&\text{勾配の通し方} && \frac{\partial L}{\partial\mathbf{z}_e}\approx\frac{\partial L}{\partial\mathbf{z}_q}\qquad\bigl(\partial\mathbf{z}_q/\partial\mathbf{z}_e\approx1\bigr)
\end{aligned}
$$

- $\hat{\mathbf{x}}=g_\theta(\mathbf{z}_q)$。$\mathrm{sg}[\cdot]$ は**ストップグラジェント**：順伝播では恒等写像、逆伝播では勾配を 0 にする。
- 事後分布が one-hot（決定的）で事前分布が一様 $1/K$ なら、KL 項は $\log K$ の**定数**になり、損失に入れなくてよい。
- 最近傍の選択（$\arg\min$）は微分できないので、逆伝播では量子化を飛ばして $\mathbf{z}_q$ の勾配をそのまま $\mathbf{z}_e$ に渡す（**ストレートスルー推定**）。
- コードブック損失はコードを $\mathbf{z}_e$ に近づけ（更新されるのはコードのみ）、コミットメント損失はエンコーダの出力がコードから離れすぎないようにする（$\beta$ はその強さ）。

---

## 対照学習 {#contrastive}

同じデータから作った2つのビュー（例：同じ画像の別々の切り抜き）の表現を近づけ、他のデータの表現を遠ざける。バッチ内の他サンプルを**負例**として使う。

$$
L_i=-\log\frac{\exp\bigl(\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_j)/\tau\bigr)}{\sum_{k\ne i}\exp\bigl(\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_k)/\tau\bigr)},\qquad \mathrm{sim}(\mathbf{u},\mathbf{v})=\frac{\mathbf{u}^{\top}\mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}
$$

- $(i,j)$ が正例のペア、$k$ は $i$ 以外のすべて。**InfoNCE（NT-Xent）損失**と呼ばれ、形はソフトマックス交差エントロピーで、「$N$ 個の候補から正例を当てる」分類問題。
- 温度 $\tau$ が小さいほど、難しい負例を重く扱う。負例が多いほど学習が安定するので、大きなバッチやメモリバンク（MoCo）を使う。
- 自己蒸留（BYOL、DINO）は負例を使わず、教師役のネットワークを指数移動平均で更新するなどして、全表現が同じ値に潰れる（崩壊）のを防ぐ。

---

## 生成モデルの分類 {#taxonomy}

データ分布 $p(\mathbf{x})$ を学習し、新しいサンプルを作るモデル。

| 分類 | 密度の扱い | 代表例 |
|:--|:--|:--|
| 明示的・厳密 | 尤度を厳密に計算できる | 自己回帰モデル、正規化フロー |
| 明示的・近似 | 尤度の下界などを使う | VAE、拡散モデル |
| 暗黙的 | 尤度を持たず、サンプリングだけできる | GAN |

- 分類・回帰のように $P(y\mid\mathbf{x})$ だけを学ぶ**識別モデル**に対し、$P(\mathbf{x},y)$ や $P(\mathbf{x})$ を学ぶのが生成モデル（[ナイーブベイズ](bayes-gp-crf.md#naive-bayes)、GMM、HMM も生成モデル）。
- 拡散モデルなどの応用は[生成モデル](../app/generative.md)で扱う。

---

## GAN（敵対的生成ネットワーク） {#gan}

**生成器** $G_\theta$（ノイズ $\mathbf{z}\sim p(\mathbf{z})$ から偽データ $G_\theta(\mathbf{z})$ を作る）と、**識別器** $D_\phi$（入力が本物である確率 $D_\phi(\mathbf{x})\in[0,1]$ を出す）を競わせる。

$$
\begin{aligned}
&\text{目的関数} && \min_{\theta}\max_{\phi}\ V(\theta,\phi)=\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\bigl[\log D_\phi(\mathbf{x})\bigr]+\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\bigl[\log\bigl(1-D_\phi(G_\theta(\mathbf{z}))\bigr)\bigr] \\[3mm]
&\text{識別器の損失} && L_D=-\mathbb{E}_{\mathbf{x}\sim p_{\mathrm{data}}}\bigl[\log D_\phi(\mathbf{x})\bigr]-\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\bigl[\log\bigl(1-D_\phi(G_\theta(\mathbf{z}))\bigr)\bigr] \\[3mm]
&\text{生成器の損失} && L_G=-\mathbb{E}_{\mathbf{z}\sim p(\mathbf{z})}\bigl[\log D_\phi(G_\theta(\mathbf{z}))\bigr] \\[3mm]
&\text{更新} && \phi\leftarrow\phi-\eta\nabla_\phi L_D,\qquad \theta\leftarrow\theta-\eta\nabla_\theta L_G
\end{aligned}
$$

- **学習の手順**：①識別器を更新（$G$ を固定。本物をラベル 1、偽物をラベル 0 にした二値交差エントロピー）、②生成器を更新（$D$ を固定。偽物を「本物」と判定させる）、を交互に繰り返す。
- $L_G=-\log D(G(\mathbf{z}))$ は**非飽和**の損失。もとの $\log(1-D(G(\mathbf{z})))$ を最小化すると、学習初期に $D$ が偽物を簡単に見破るとき（$D\approx0$）勾配がほぼ 0 になって学べないので、こちらを使う。

??? note "最適な識別器と JS ダイバージェンス"
    $G$ を固定すると $V=\int\bigl[p_{\mathrm{data}}\log D+p_g\log(1-D)\bigr]d\mathbf{x}$（$p_g$ は生成分布）。点ごとに微分して 0 とおくと
    $D^*(\mathbf{x})=\dfrac{p_{\mathrm{data}}(\mathbf{x})}{p_{\mathrm{data}}(\mathbf{x})+p_g(\mathbf{x})}$。
    これを戻すと $V=-\log4+2\,D_{\mathrm{JS}}(p_{\mathrm{data}}\|p_g)$。JS ダイバージェンスは 0 以上で、$p_g=p_{\mathrm{data}}$ のとき 0 なので、大域的な最適解は $p_g=p_{\mathrm{data}}$（このとき $D^*=1/2$ で、識別器は本物と偽物を見分けられない。ナッシュ均衡）。

**課題**

- **モード崩壊**：生成器が一部のパターンしか作らなくなる。
- **勾配消失**：識別器が強すぎると、生成器に有益な勾配が伝わらない。モード崩壊と勾配消失の間で、識別器の学習の度合いにジレンマがある。
- **収束しない**：非凸の 2 人ゼロサムゲームなので、勾配法が均衡（鞍点）に収束する保証がない。損失の値からは学習をいつ止めるべきか、画質がどうかも判断しにくい。
- 高解像度では識別が簡単になり、メモリの都合でバッチも小さくなって不安定になる。

| 派生 | 内容 |
|:--|:--|
| DCGAN | 畳み込みで構成。生成器は転置畳み込みでアップサンプリング、識別器はストライド付き畳み込み。プーリングと全結合を使わない。バッチ正規化（生成器の出力層と識別器の入力層は除く）、生成器の出力は tanh、識別器は Leaky ReLU |
| 条件付き GAN（CGAN） | 条件 $\mathbf{c}$（クラスラベルなど）を $G$ と $D$ に与える。$D(\mathbf{x}\mid\mathbf{c})$、$G(\mathbf{z}\mid\mathbf{c})$ |
| pix2pix | 画像から画像への変換。条件画像 $\mathbf{x}$ と本物 $\mathbf{y}$ のペアを使う。$G^*=\arg\min_G\max_D L_{\mathrm{cGAN}}(G,D)+\lambda\,\mathbb{E}\Vert \mathbf{y}-G(\mathbf{x},\mathbf{z})\Vert _1$ |
| WGAN | JS の代わりにワッサースタイン距離を使う。識別器（批評家）に 1-リプシッツ制約を課し、モード崩壊と勾配消失を緩和 |
| PGGAN など | 低解像度から段階的に大きくして、高解像度を安定して学習 |

- pix2pix の**U-Net** 生成器は、エンコーダの浅い層の情報をスキップ接続でデコーダへ渡し、輪郭などの細部を保つ。**PatchGAN** 識別器は画像を小領域（パッチ）に分け、それぞれが本物か判定して平均する（パラメータが少なく、局所的な質感を見る）。**L1 損失**は全体の構造を合わせ、L2 よりぼやけにくい。ノイズ $\mathbf{z}$ はドロップアウトで代用する。

---

## 隠れマルコフモデル（HMM） {#hmm}

観測できない**状態**の系列 $q_1,\dots,q_T$ がマルコフ連鎖をなし、各時刻の状態から観測 $o_t$ が出力されるとする生成モデル。パラメータは $\lambda=(\mathbf{A},\mathbf{B},\boldsymbol\pi)$：遷移確率 $a_{ij}=P(q_{t+1}=j\mid q_t=i)$、出力確率 $b_i(o)=P(o_t=o\mid q_t=i)$、初期確率 $\pi_i=P(q_1=i)$。

$$
P(o_{1:T},q_{1:T})=\pi_{q_1}\,b_{q_1}(o_1)\prod_{t=2}^{T}a_{q_{t-1}q_t}\,b_{q_t}(o_t)
$$

| 問題 | アルゴリズム | 内容 |
|:--|:--|:--|
| 評価：$P(o_{1:T}\mid\lambda)$ | 前向きアルゴリズム | $\alpha_t(i)=P(o_{1:t},q_t=i\mid\lambda)$。$\alpha_1(i)=\pi_ib_i(o_1)$、$\alpha_{t+1}(j)=\bigl[\sum_i\alpha_t(i)a_{ij}\bigr]b_j(o_{t+1})$、$P=\sum_i\alpha_T(i)$ |
| 復号：最尤状態系列 | ビタビアルゴリズム | $\delta_t(j)=\max_i\delta_{t-1}(i)\,a_{ij}\,b_j(o_t)$ と argmax を記録し、最後から逆にたどる |
| 学習：$\lambda$ の推定 | Baum-Welch（EM） | E ステップで $\gamma,\xi$ を求め、M ステップで $\lambda$ を更新 |

$$
\gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{\sum_j\alpha_t(j)\beta_t(j)},\qquad
\xi_t(i,j)=\frac{\alpha_t(i)\,a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)}{P(o_{1:T}\mid\lambda)},\qquad
\hat{a}_{ij}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\quad\hat{\pi}_i=\gamma_1(i)
$$

- $\beta_t(i)=P(o_{t+1:T}\mid q_t=i,\lambda)$ は**後ろ向き確率**（$\beta_T(i)=1$、$\beta_t(i)=\sum_ja_{ij}b_j(o_{t+1})\beta_{t+1}(j)$）。$\gamma_t(i)$ は時刻 $t$ に状態 $i$ にいる確率、$\xi_t(i,j)$ は $t$ に $i$、$t+1$ に $j$ にいる確率。
- 出力確率は、離散なら $\hat{b}_j(v)=\sum_{t:o_t=v}\gamma_t(j)\big/\sum_t\gamma_t(j)$。連続なら各状態をガウス（混合）にする。
- 素朴に全状態系列を足すと $O(S^T)$ だが、前向き・ビタビは動的計画法で $O(TS^2)$（$S$ は状態数）。
- Baum-Welch は [GMM の EM](clustering.md#gmm) と同じ構造で、各反復で尤度は減らず、局所解に収束する。
- 音声認識や品詞タグ付けに使われた。同じ系列ラベリングを条件付き確率で扱う識別モデルが [CRF](bayes-gp-crf.md#crf)。

---

## 試験の着眼点

- AE の損失は再構成誤差。ボトルネックで恒等写像を防ぐ。DAE は**ノイズを加えた入力から元の入力**を復元する。
- VAE の損失 $=$ 再構成誤差 $+$ KL（$=-\mathrm{ELBO}$）。KL は閉じた式 $-\frac12\sum(1+\log\sigma^2-\mu^2-\sigma^2)$。再パラメータ化トリックは $\mathbf{z}=\boldsymbol\mu+\boldsymbol\sigma\odot\boldsymbol\varepsilon$ で、逆伝播を可能にするため。
- VAE の課題：生成のぼやけ、事後崩壊。VQ-VAE は離散潜在変数で、ストレートスルーとストップグラジェントを使い、KL 項は定数になる。
- GAN は min-max。最適識別器は $p_{\mathrm{data}}/(p_{\mathrm{data}}+p_g)$ で、最適解では $D=1/2$。目的関数は JS ダイバージェンス。課題はモード崩壊・勾配消失・収束性。
- DCGAN、CGAN、pix2pix（U-Net、PatchGAN、L1）の特徴を区別する。
- HMM は生成モデル（遷移・出力・初期確率）。前向きアルゴリズムで尤度、ビタビで最尤系列、Baum-Welch（EM）で学習。

---

## 参考

- [Auto-Encoding Variational Bayes（Kingma, Welling, 2013）](https://arxiv.org/abs/1312.6114)
- [Stochastic Backpropagation and Approximate Inference in Deep Generative Models（Rezende ら, 2014）](https://arxiv.org/abs/1401.4082)
- [Neural Discrete Representation Learning（van den Oord ら, 2017）](https://arxiv.org/abs/1711.00937)
- [Generative Adversarial Networks（Goodfellow ら, 2014）](https://arxiv.org/abs/1406.2661)
- [Conditional Generative Adversarial Nets（Mirza, Osindero, 2014）](https://arxiv.org/abs/1411.1784)
- [Unsupervised Representation Learning with Deep Convolutional GANs（Radford ら, 2015）](https://arxiv.org/abs/1511.06434)
- [Image-to-Image Translation with Conditional Adversarial Networks（Isola ら, 2016）](https://arxiv.org/abs/1611.07004)
- [Wasserstein GAN（Arjovsky ら, 2017）](https://arxiv.org/abs/1701.07875)
- [A Simple Framework for Contrastive Learning of Visual Representations（Chen ら, 2020）](https://arxiv.org/abs/2002.05709)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding（Devlin ら, 2018）](https://arxiv.org/abs/1810.04805)
- A tutorial on hidden Markov models and selected applications in speech recognition（Rabiner, 1989, Proceedings of the IEEE）
