# 確率過程

<p class="kw">キーワード：確率過程、マルコフ過程、マルコフ連鎖、定常分布、マルコフ決定過程、階差過程（ランダムウォーク）、ウィーナー過程、ポアソン過程、ガウス過程、隠れマルコフモデル、MCMC</p>

!!! abstract "要点"
    - **確率過程**は、時間（や空間）$t$ で並べた確率変数の族 $\{X_t\}_{t\in T}$。1回の実現を**標本路**という。
    - **マルコフ性**（次の状態の分布が現在の状態だけで決まる）があると、分布の変化は**遷移行列のべき乗** $\mathbf{p}_t=\mathbf{p}_0P^{t}$ で書ける。既約で非周期なら**定常分布**に収束する。
    - ランダムウォーク・ウィーナー過程・ポアソン過程・ガウス過程・隠れマルコフモデル・マルコフ決定過程は、どれも「何が確率的に動くか」の違い。

---

## 確率過程とは {#sp}

- 添字 $t$ が整数なら**離散時間**、実数なら**連続時間**。$X_t$ のとる値の集合が**状態空間**。

| 過程 | 説明 | 式 |
|:--|:--|:--|
| マルコフ過程 | 直前の状態だけで次の状態の確率が決まる | $P(X_{t+1}\mid X_t,\dots,X_0)=P(X_{t+1}\mid X_t)$ |
| マルコフ決定過程（MDP） | 行動を選ぶと、状態遷移と報酬が確率的に決まる | $P(S_{t+1}\mid S_t,A_t),\ \ R_{t+1}=r(S_t,A_t,S_{t+1})$ |
| 階差過程（ランダムウォーク） | 前の値に独立なノイズを足して進む | $X_{t+1}=X_t+\varepsilon_{t+1},\ \ \varepsilon_{t+1}\sim\mathcal{N}(0,\sigma^2)$ |
| ウィーナー過程（ブラウン運動） | 連続時間で、増分が独立な正規分布 | $W_0=0,\ \ W_t-W_s\sim\mathcal{N}(0,\,t-s)\ (s<t)$ |
| ポアソン過程 | 一定の割合で事象が起きた回数 | $N_t\sim\mathrm{Poisson}(\lambda t)$ |
| ガウス過程 | どの有限個の時点の値も多変量正規分布に従う | $(f(t_1),\dots,f(t_n))\sim\mathcal{N}(\mathbf{m},K)$ |
| 隠れマルコフモデル（HMM） | 観測できない状態がマルコフ連鎖で推移し、観測はその状態から出る | $P(Z_1)\prod_tP(Z_{t+1}\mid Z_t)\prod_tP(X_t\mid Z_t)$ |

---

## マルコフ連鎖 {#markov-chain}

状態が離散で時間も離散のマルコフ過程。$P_{ij}=P(X_{t+1}=j\mid X_t=i)$ を並べた**遷移行列** $P$ は、要素が非負で**各行の和が 1**。

$$
\mathbf{p}_{t+1}=\mathbf{p}_tP,\qquad \mathbf{p}_t=\mathbf{p}_0P^{t},\qquad P^{m+n}=P^{m}P^{n}\ \ (\text{チャップマン–コルモゴロフ}),\qquad
\mathbf{p}^{*}=\mathbf{p}^{*}P
$$

- $\mathbf{p}_t$ は時刻 $t$ の状態分布（**行ベクトル**）。$\mathbf{p}^{*}$ が**定常分布**で、$P$ の固有値 1 に対する**左**固有ベクトルを、和が 1 になるよう規格化したもの。
- **既約**（どの状態からどの状態へも行ける）かつ**非周期**（決まった周期で戻るだけ、ということがない）なら、定常分布は一意で、初期分布によらず $\mathbf{p}_t\to\mathbf{p}^{*}$ となる。
- 収束の速さは、1 以外で最大の固有値の絶対値 $|\lambda_2|$ で決まる（誤差が $|\lambda_2|^{t}$ で減る。[固有値分解](eigen-svd.md)）。
- **詳細釣り合い** $p^{*}_iP_{ij}=p^{*}_jP_{ji}$ が成り立てば、$\mathbf{p}^{*}$ は定常分布になる（両辺を $i$ について足すと $\sum_ip^{*}_iP_{ij}=p^{*}_j$）。

2状態の例（A から B へ移る確率 $\alpha$、B から A へ移る確率 $\beta$）。

$$
P=\begin{pmatrix}1-\alpha&\alpha\\\beta&1-\beta\end{pmatrix},\qquad
\mathbf{p}^{*}=\Bigl(\frac{\beta}{\alpha+\beta},\ \frac{\alpha}{\alpha+\beta}\Bigr),\qquad
P(X_t=\mathrm{A})=p^{*}_{\mathrm{A}}+\bigl(p_{0,\mathrm{A}}-p^{*}_{\mathrm{A}}\bigr)(1-\alpha-\beta)^{t}
$$

<div class="ew" data-widget="stoch-markov"></div>

!!! tip "動かしてみる"
    - $\alpha=0.3,\ \beta=0.1$ のとき定常分布は $(0.25,\ 0.75)$ です。$p_0=0.25$ に合わせると、$t$ を動かしても**確率が変わらなく**なります（定常分布は、かけても変わらない分布）。
    - $\alpha+\beta$ が 1 に近いほど $\lambda_2=1-\alpha-\beta$ が 0 に近づき、**ほぼ 1 ステップで**定常分布に着きます。$\alpha,\beta$ を小さくすると、収束がゆっくりになります。
    - $\alpha=\beta=1$ にすると $\lambda_2=-1$ で、A と B を**交互に行き来する**だけになり、収束しません（周期 2）。

### マルコフ連鎖モンテカルロ法（MCMC） {#mcmc}

- 正規化定数が計算できない分布 $p(\mathbf{x})\propto\tilde{p}(\mathbf{x})$（例：ベイズ推定の事後分布）から標本を得るために、**$p$ を定常分布とするマルコフ連鎖**を作る方法。
- **メトロポリス–ヘイスティングス法**：提案分布 $q(\mathbf{x}'\mid\mathbf{x})$ で候補を出し、次の確率で採択する。採択しなければ元の状態に留まる。

$$
a=\min\!\left(1,\ \frac{\tilde{p}(\mathbf{x}')\,q(\mathbf{x}\mid\mathbf{x}')}{\tilde{p}(\mathbf{x})\,q(\mathbf{x}'\mid\mathbf{x})}\right)
$$

- $\tilde{p}$ の**比**しか使わないので、正規化定数が打ち消し合う。この採択ルールは詳細釣り合いを満たす。
- ほかに、1変数ずつ条件付き分布から抽出する**ギブスサンプリング**がある。初期の標本は捨て（バーンイン）、隣り合う標本には相関がある。

---

## マルコフ決定過程（MDP） {#mdp}

マルコフ連鎖に**行動**と**報酬**を加えたもの。強化学習の問題設定（[強化学習](../app/rl.md)）。

$$
\begin{aligned}
&\text{状態遷移確率} && p(s'\mid s,a)=\Pr(S_{t+1}=s'\mid S_t=s,\ A_t=a) \\[2mm]
&\text{方策} && \pi(a\mid s)=\Pr(A_t=a\mid S_t=s) \\[2mm]
&\text{報酬} && R_{t+1}=r(S_t,A_t,S_{t+1}) \\[2mm]
&\text{リターン（累積割引報酬）} && G_t=\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}=R_{t+1}+\gamma G_{t+1} \\[2mm]
&\text{状態価値関数} && v_\pi(s)=\mathbb{E}_\pi\bigl[G_t\mid S_t=s\bigr] \\[2mm]
&\text{行動価値関数} && q_\pi(s,a)=\mathbb{E}_\pi\bigl[G_t\mid S_t=s,\ A_t=a\bigr] \\[2mm]
&\text{ベルマン方程式} && v_\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s'}p(s'\mid s,a)\Bigl[r(s,a,s')+\gamma\,v_\pi(s')\Bigr]
\end{aligned}
$$

- $\mathcal{S}$：状態集合、$\mathcal{A}$：行動集合、$\gamma\in[0,1)$：割引率（遠い未来の報酬を割り引く。$\gamma=1$ は有限の長さの問題でのみ使う）。
- 方策 $\pi$ を決めると、状態の列は遷移行列 $P_{ij}=\sum_a\pi(a\mid i)\,p(j\mid i,a)$ のマルコフ連鎖になる。ベルマン方程式は、「今の価値 ＝ 次の1歩の報酬 ＋ 割り引いた次の状態の価値」という $G_t=R_{t+1}+\gamma G_{t+1}$ の期待値版。

---

## ランダムウォークとウィーナー過程 {#wiener}

- **ランダムウォーク（階差過程）**：$X_{t+1}=X_t+\varepsilon_{t+1}$（$\varepsilon$ は独立で同じ分布）。$\varepsilon\sim\mathcal{N}(0,\sigma^2)$ なら $X_t=X_0+\sum_{s\le t}\varepsilon_s$ の分散は $t\sigma^2$。**分散は時間に比例し、標準偏差は $\sqrt{t}$ に比例**する。
- **ウィーナー過程** $W_t$：$W_0=0$、増分が独立で $W_t-W_s\sim\mathcal{N}(0,t-s)$、経路は連続（ただしどこでも微分できない）。ランダムウォークの刻みを細かくした極限。
- **ドリフトつきブラウン運動**（拡散過程）は確率微分方程式で書く。

$$
dX_t=\mu\,dt+\sigma\,dW_t
\quad\Longrightarrow\quad
X_t=X_0+\mu t+\sigma W_t\sim\mathcal{N}\bigl(X_0+\mu t,\ \sigma^2t\bigr)
$$

- 数値計算では $\Delta t$ ずつ進める（オイラー–丸山法）：$X_{t+\Delta t}=X_t+\mu\Delta t+\sigma\sqrt{\Delta t}\,\varepsilon$、$\varepsilon\sim\mathcal{N}(0,1)$。$\sqrt{\Delta t}$ なのは、増分の**分散**が $\Delta t$ に比例するため。
- 拡散モデルは、データに少しずつノイズを足す過程をこの形の確率微分方程式で表し、逆向きにたどってデータを生成する（[生成モデル](../app/generative.md)）。

<div class="ew" data-widget="stoch-walk"></div>

!!! tip "動かしてみる"
    - 刻み数 $n$ を 5 から 100 に増やすと、折れ線の経路がなめらかになり、**ウィーナー過程の経路に近づきます**。それでも帯（$\mu t\pm\sigma\sqrt{t}$）と終点 $X_T$ の分布は変わりません。
    - $\sigma$ を大きくすると、帯が $\sqrt{t}$ の形に広がります（直線的ではなく、最初に速く広がる）。$\mu$ は帯全体を傾けます。
    - 下の「標本路 24 本の平均・標準偏差」は、標本が少ないため理論値（$\mu T$、$\sigma\sqrt{T}$）から揺れます。

---

## ポアソン過程 {#poisson-process}

- 区間 $(0,t]$ に事象が起きる回数 $N_t$：$N_0=0$、増分は独立で定常、$N_t\sim\mathrm{Poisson}(\lambda t)$。$\mathbb{E}[N_t]=\mathbb{V}[N_t]=\lambda t$。
- 事象と事象の**間隔**は、独立に**指数分布** $\mathrm{Exp}(\lambda)$（平均 $1/\lambda$）に従う。
- 例：毎時 $\lambda=2$ 件の割合で起きる事象が、30 分間に 1 回も起きない確率は、$N\sim\mathrm{Poisson}(1)$ より $e^{-1}\approx0.368$。

---

## ガウス過程 {#gp}

- 関数 $f(t)$ そのものの確率分布。どんな有限個の入力 $t_1,\dots,t_n$ をとっても、$(f(t_1),\dots,f(t_n))$ が多変量正規分布 $\mathcal{N}(\mathbf{m},K)$ に従う。
- 平均は**平均関数** $m(t)$、共分散は**カーネル（共分散関数）** $K_{ij}=k(t_i,t_j)$ で決まる。$K$ は半正定値でなければならない。近い入力ほど値が似るように $k$ を選ぶ。
- 観測が得られたら、多変量正規分布の**条件付き分布**（[確率分布・ベイズの定理](probability.md#mvn)）を計算するだけで予測の平均と不確かさが得られる（[ガウス過程回帰](../ml/bayes-gp-crf.md)）。

---

## 隠れマルコフモデル（HMM） {#hmm}

観測できない状態 $Z_t$ がマルコフ連鎖で推移し、観測 $X_t$ は**その時刻の $Z_t$ だけ**から決まる。

$$
P(X_{1:T},Z_{1:T})=P(Z_1)\prod_{t=1}^{T-1}P(Z_{t+1}\mid Z_t)\prod_{t=1}^{T}P(X_t\mid Z_t)
$$

- パラメータ：初期分布 $P(Z_1)$、遷移確率 $a_{ij}=P(Z_{t+1}=j\mid Z_t=i)$、出力確率 $b_j(x)=P(X_t=x\mid Z_t=j)$。

| 問題 | 内容 | 解く方法 |
|:--|:--|:--|
| 評価 | 観測列の確率 $P(X_{1:T})$ を求める | **前向きアルゴリズム** |
| 復号 | 最も確からしい状態列を求める | **ビタビアルゴリズム** |
| 学習 | $a_{ij}$、$b_j$ を観測から推定する | **Baum–Welch**（EM アルゴリズム） |

- 前向きアルゴリズムは $\alpha_t(j)=P(X_{1:t},Z_t=j)$ を漸化式 $\alpha_t(j)=\Bigl[\sum_{i}\alpha_{t-1}(i)\,a_{ij}\Bigr]b_j(X_t)$ で求める。全状態列（$K^{T}$ 通り）を足し上げる代わりに、計算量が $O(TK^2)$ で済む。
- 音声認識の古典的な枠組み（[音声処理](../app/speech.md)）。ニューラルネットワークによる系列モデルの前身にあたる。

---

## 試験の着眼点

- マルコフ性：次の状態の分布は**現在の状態だけ**で決まる。遷移行列は**行の和が 1**。
- 定常分布は $\mathbf{p}^{*}=\mathbf{p}^{*}P$。既約かつ非周期なら一意で、そこへ収束する。収束の速さは第2固有値。
- ランダムウォークの分散は $t$ に比例。ウィーナー過程の増分は $\mathcal{N}(0,t-s)$。$dX=\mu\,dt+\sigma\,dW$ はウィーナー過程そのものではなく、ドリフトつきの拡散過程。
- ポアソン過程：回数は $\mathrm{Poisson}(\lambda t)$、間隔は $\mathrm{Exp}(\lambda)$。
- MDP ＝ マルコフ連鎖 ＋ 行動 ＋ 報酬。ベルマン方程式は $v_\pi$ の再帰式。
- ガウス過程 ＝ 任意の有限個の点が多変量正規分布。HMM の3つの問題は、評価・復号・学習。
- MCMC は正規化定数が不要。採択確率は $\tilde{p}$ の比で決まる。

## 参考

- [Reinforcement Learning: An Introduction, 2nd edition（Sutton, Barto, 2018）](http://incompleteideas.net/book/the-book-2nd.html)
- [Pattern Recognition and Machine Learning, Chapter 11 Sampling Methods, Chapter 13 Sequential Data（Bishop, 2006）](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/)
- [Gaussian Processes for Machine Learning（Rasmussen, Williams, 2006）](https://gaussianprocess.org/gpml/)
