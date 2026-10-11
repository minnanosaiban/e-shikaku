# 最適化理論

<p class="kw">キーワード：最適化問題、制約付き最適化、線形計画法、非線形計画法、凸最適化、二次計画法、整数計画法、動的計画法、ラグランジュ法、双対問題、KKT条件、数値的最適化、勾配降下法、ニュートン法、確率的勾配降下法、ミニバッチ、ベルマン方程式</p>

!!! abstract "要点"
    - 最適化は「**目的関数**を、**制約**のもとで最小にする点を探す」問題。**凸**なら局所解が大域解で、双対問題と **KKT 条件**で解を特徴づけられる。
    - 制約を $\lambda\ge0$ 倍して目的に足した**ラグランジュ関数**から、**双対問題**と **KKT 条件**（定常性・実行可能性・$\lambda\ge0$・**相補スラック性**）が導かれる。SVM の双対問題はその例。
    - 深層学習は**制約なし・非凸**なので、勾配法（GD・SGD・ミニバッチ）で解く。**動的計画法**は、価値を再帰的に書いて、小さい問題の解から全体の最適解を作る。

---

## 最適化問題の分類 {#problem-types}

$$
\begin{aligned}
&\min_{\mathbf{x}\in\mathbb{R}^n}\ f(\mathbf{x}) \\
&\text{s.t.}\quad g_i(\mathbf{x})\le0\ \ (i=1,\dots,m),\qquad h_j(\mathbf{x})=0\ \ (j=1,\dots,p)
\end{aligned}
$$

- $f$：**目的関数**、$g_i$：**不等式制約**、$h_j$：**等式制約**。制約を満たす点の集合が**実行可能領域**。最大化は $-f$ の最小化に直せる。

| 分類 | 内容 | 主な手法・キーワード |
|:--|:--|:--|
| 線形計画法（LP） | 目的も制約も線形 | 単体法、**内点法**、双対法 |
| 二次計画法（QP） | 目的が二次、制約が線形（凸なら凸最適化の特殊形） | SVM の双対問題 |
| **凸最適化** | 目的が凸関数、実行可能領域が凸集合 | 双対法、内点法、**KKT 条件** |
| 非線形計画法 | 目的や制約が非線形 | **ラグランジュ法**、KKT 条件、**数値的最適化**（勾配法・ニュートン法・座標降下法・遺伝的アルゴリズムなど） |
| 整数計画法 | 変数が整数（離散） | **分枝限定法**、カット平面法 |
| **動的計画法** | 段階的な意思決定問題 | **ベルマン方程式**、再帰 |

- 整数計画は一般に難しい（NP 困難）。緩和（整数条件を外した LP）の解で下界を作り、分枝で絞り込むのが分枝限定法。
- 機械学習の学習は多くが**制約なしの最小化**（正則化は制約でなく罰則項として加えることが多い）。

---

## 数値的最適化 {#numerical}

解析的に解けないとき、点を少しずつ更新して最適解に近づける。

$$
\begin{aligned}
&\text{勾配降下法} && \mathbf{w}^{(j+1)}=\mathbf{w}^{(j)}-\eta\,\nabla L\bigl(\mathbf{w}^{(j)}\bigr) \\[2mm]
&\text{ニュートン法} && \mathbf{w}^{(j+1)}=\mathbf{w}^{(j)}-\mathbf{H}^{-1}\nabla L\bigl(\mathbf{w}^{(j)}\bigr),\qquad \mathbf{H}=\nabla^2L\bigl(\mathbf{w}^{(j)}\bigr)\ \text{（ヘッセ行列）} \\[2mm]
&\text{1変数のニュートン法} && w^{(j+1)}=w^{(j)}-\dfrac{L'(w^{(j)})}{L''(w^{(j)})} \\[2mm]
&\text{確率的勾配降下法（SGD）} && \mathbf{w}^{(j+1)}=\mathbf{w}^{(j)}-\eta\,\nabla L_i\bigl(\mathbf{w}^{(j)}\bigr)\qquad(i\text{ は 1 サンプル}) \\[2mm]
&\text{ミニバッチ勾配降下法} && \mathbf{w}^{(j+1)}=\mathbf{w}^{(j)}-\eta\,\frac{1}{|B|}\sum_{i\in B}\nabla L_i\bigl(\mathbf{w}^{(j)}\bigr)
\end{aligned}
$$

| 手法 | 1 回の更新に使うデータ | 特徴 |
|:--|:--|:--|
| 勾配降下法（バッチ） | 全データ | 勾配が正確。1 回が重い |
| SGD | ランダムな 1 個 | 軽いが勾配のばらつき（ノイズ）が大きい。ノイズが浅い局所解から抜け出す助けにもなる |
| ミニバッチ | $\lvert B\rvert$ 個 | ばらつきと計算量の中間。GPU で並列化しやすく、実際はほぼこれ |
| ニュートン法 | 全データ（2階微分も） | 凸な2次関数なら**1回で最適解**。ただし $\mathbf{H}$ の計算と逆行列に $O(D^2)$ のメモリ・$O(D^3)$ の計算がかかり、パラメータの多い深層学習では使えない |

- 目的関数を点 $\mathbf{w}^{(j)}$ のまわりで **2 次近似**して、その最小点へ飛ぶのがニュートン法。勾配降下法は 1 次近似（直線）なので、歩幅 $\eta$ を人が決める必要がある。
- ニュートン法の近似として、$\mathbf{H}^{-1}$ を勾配の履歴から推定する**準ニュートン法**（BFGS、メモリを節約した L-BFGS）がある。
- **座標降下法**は 1 変数ずつ順番に最小化する（L1 正則化つき回帰など）。**遺伝的アルゴリズム・進化戦略**は勾配を使わず、候補の集団を選択・交叉・突然変異で更新する（微分できない目的にも使える）。**ベイズ最適化**は評価が高価な関数（ハイパーパラメータ探索）で、代理モデルを使って次に試す点を決める（[アンサンブル・ハイパーパラメータ](../dl/ensemble-hyperparam.md)）。
- Momentum・AdaGrad・RMSProp・Adam など、勾配法を改良した手法は [最適化手法](../dl/optimizer.md) で扱う。

### 勾配降下法は、歩幅が大きすぎると発散する

1 変数の 2 次関数 $f(w)=\tfrac12aw^2$（$a>0$）では $f'(w)=aw$ なので

$$
w_{k+1}=w_k-\eta\,a\,w_k=(1-\eta a)\,w_k\quad\Longrightarrow\quad w_k=(1-\eta a)^k\,w_0
$$

- $|1-\eta a|<1$、つまり $0<\eta<2/a$ のとき収束する。$\eta=1/a$ なら 1 回で最小点に着く。$\eta>2/a$ では符号を変えながら**振動して発散**する。
- 曲率 $a$（2 階微分の大きさ）が大きいほど、使える歩幅の上限は小さい。一般の関数ではヘッセ行列の最大固有値 $\lambda_{\max}$ が $a$ にあたり、$\eta<2/\lambda_{\max}$ が目安になる。
- ニュートン法は $w-f'/f''=w-aw/a=0$ で、**$a$ によらず 1 回で最小点**に着く（歩幅を曲率で自動調整している）。

<div class="ew" data-widget="opt-gd-newton"></div>

!!! tip "動かしてみる"
    - 勾配降下法で $\eta$ を $2/a$ の手前（$a=1$ なら 1.9 付近）まで大きくすると、左右に振動しながら収束します。2 を超えると発散します。
    - $a$ を大きくすると、同じ $\eta$ でも収束しにくくなります（$2/a$ が小さくなる）。
    - 「ニュートン法」に切り替えると、$\eta$ と $a$ に関係なく 1 回で 0 に着きます。

---

## 凸最適化 {#convex}

- **凸集合**：集合内の任意の 2 点を結ぶ線分が、集合に含まれる。
- **凸関数**：$f\bigl(\theta\mathbf{x}+(1-\theta)\mathbf{y}\bigr)\le\theta f(\mathbf{x})+(1-\theta)f(\mathbf{y})\ \ (0\le\theta\le1)$。2 回微分できるなら、**ヘッセ行列が半正定値**（どの方向にも曲がりが 0 以上）と同値。
- 凸最適化（凸関数を、凸集合の上で最小化）では、**局所最適解がそのまま大域最適解**。
- 凸なもの：線形回帰の二乗誤差、ロジスティック回帰の損失、SVM の目的。**非凸**なもの：ニューラルネットワークの損失（局所解・鞍点が多数ある）。

### 双対性の流れ

$$
\begin{aligned}
&\text{ラグランジュ関数} && \Lambda(\mathbf{x},\boldsymbol\lambda,\boldsymbol\nu)=f(\mathbf{x})+\sum_{i=1}^m\lambda_ig_i(\mathbf{x})+\sum_{j=1}^p\nu_jh_j(\mathbf{x}),\qquad\lambda_i\ge0 \\[2mm]
&\text{双対関数} && d(\boldsymbol\lambda,\boldsymbol\nu)=\inf_{\mathbf{x}}\Lambda(\mathbf{x},\boldsymbol\lambda,\boldsymbol\nu) \\[2mm]
&\text{双対問題} && \max_{\boldsymbol\lambda\ge\mathbf{0},\,\boldsymbol\nu}\ d(\boldsymbol\lambda,\boldsymbol\nu)
\end{aligned}
$$

- 主問題の最適値を $p^*$、双対問題の最適値を $d^*$ とすると、常に $d^*\le p^*$（**弱双対性**）。実行可能な $\mathbf{x}$ では $g_i\le0$ なので、$\lambda_i\ge0$ なら制約の項は $\le0$ で、$d(\boldsymbol\lambda,\boldsymbol\nu)\le f(\mathbf{x})$ となるため。
- 凸問題で、制約をすべて**厳密に**満たす点が存在する（**スレイターの条件**）なら $d^*=p^*$（**強双対性**）。このとき主問題と双対問題のどちらを解いてもよい。
- 双対関数は、主問題が凸でなくても**常に凹関数**。双対問題はいつも凸最適化になる。
- 流れ：①主問題を立てる → ②ラグランジュ関数 → ③$\mathbf{x}$ について $\inf$（双対関数）→ ④双対問題を $\boldsymbol\lambda,\boldsymbol\nu$ で最大化 → ⑤KKT 条件で $\mathbf{x}^*$ を戻す。

---

## ラグランジュ法（等式制約） {#lagrange}

- 等式制約 $h(\mathbf{x})=0$ のもとで $f$ を最小にする点では、**$f$ の等高線と制約の曲線が接する**。接する＝勾配が平行：$\nabla f(\mathbf{x}^*)+\nu\nabla h(\mathbf{x}^*)=\mathbf{0}$。
- ラグランジュ関数 $\Lambda=f+\nu h$ を $\mathbf{x}$ と $\nu$ で微分して 0 とおくと、この式と制約 $h=0$ が同時に得られる（制約を目的に組み込んだ）。
- 例：$\min\ x^2+y^2$ s.t. $x+y-1=0$。$\Lambda=x^2+y^2+\nu(x+y-1)$ より $2x+\nu=0,\ 2y+\nu=0$ から $x=y$、制約から $x=y=\tfrac12$、$\nu=-1$、最小値 $\tfrac12$。原点から直線 $x+y=1$ までの距離の 2 乗 $(1/\sqrt2)^2=\tfrac12$ と一致する。

---

## KKT 条件 {#kkt}

最適解 $\mathbf{x}^*$（と乗数 $\boldsymbol\lambda^*,\boldsymbol\nu^*$）が満たす条件：

$$
\begin{aligned}
&\text{定常性} && \nabla f(\mathbf{x}^*)+\sum_i\lambda_i^*\nabla g_i(\mathbf{x}^*)+\sum_j\nu_j^*\nabla h_j(\mathbf{x}^*)=\mathbf{0} \\[2mm]
&\text{主実行可能性} && g_i(\mathbf{x}^*)\le0,\qquad h_j(\mathbf{x}^*)=0 \\[2mm]
&\text{双対実行可能性} && \lambda_i^*\ge0 \\[2mm]
&\text{相補スラック性} && \lambda_i^*\,g_i(\mathbf{x}^*)=0
\end{aligned}
$$

- **相補スラック性**：各不等式制約について、「制約が**効いている**（$g_i=0$、境界上）」か「$\lambda_i=0$（制約が最適解に影響しない）」のどちらか。制約が緩い（$g_i<0$）なら必ず $\lambda_i=0$。
- $\lambda_i^*$ は、制約を少し緩めたときに最適値がどれだけ改善するか（**感度**）を表す。
- **必要性**：最適解がKKT条件を満たすには、制約に**正則性条件**（制約の勾配が一次独立、あるいはスレイターの条件など）が要る。**十分性**：凸問題なら、KKT 条件を満たす点は大域最適解。非凸では必要条件にとどまる。
- 「必要十分」と言えるのは、**凸問題（かつ正則性条件が成り立つ）**のとき。

### 動かして確かめる：1 つの不等式制約

目的 $f(\mathbf{x})=\tfrac12\|\mathbf{x}-\mathbf{x}_0\|^2$（$\mathbf{x}_0$ に近い点ほどよい）に、制約 $g(\mathbf{x})=x_1+x_2-c\le0$ を付けた問題です。KKT 条件を解くと、$s=x_{0,1}+x_{0,2}$ として次のようになります。

$$
\begin{aligned}
s\le c\ \text{（制約が緩い）}&:\ \lambda=0,\ \ \mathbf{x}^*=\mathbf{x}_0 \\
s>c\ \text{（制約が効く）}&:\ \lambda=\frac{s-c}{2}>0,\ \ \mathbf{x}^*=\mathbf{x}_0-\lambda\mathbf{1},\ \ g(\mathbf{x}^*)=0
\end{aligned}
$$

<div class="ew" data-widget="opt-kkt"></div>

!!! tip "動かしてみる"
    - $c$ を大きくして、$\mathbf{x}_0$ が灰色の領域に入ると、$\lambda=0$ になり最適解は $\mathbf{x}_0$ そのものです（制約は効かない）。
    - $c$ を小さくして境界が $\mathbf{x}_0$ に近づくと、最適解は境界上（円と直線が接する点）に移り、$\lambda$ は境界から $\mathbf{x}_0$ までの距離に比例して増えます。
    - どちらの場合も、$\lambda$ と $g$ の**積は 0** です（相補スラック性）。

---

## 例：SVM の双対問題 {#svm-example}

学習データ $(\mathbf{x}_n,t_n)$（$t_n\in\{-1,+1\}$、$n=1,\dots,N$）のソフトマージン SVM（[SVM](../ml/svm.md)）は、凸な 2 次計画（QP）。

$$
\min_{\mathbf{w},b,\boldsymbol\xi}\ \frac12\|\mathbf{w}\|^2+C\sum_{n=1}^N\xi_n\quad\text{s.t.}\quad t_n(\mathbf{w}^\top\mathbf{x}_n+b)\ge1-\xi_n,\ \ \xi_n\ge0
$$

$$
\Lambda=\frac12\|\mathbf{w}\|^2+C\sum_n\xi_n-\sum_n\alpha_n\bigl[t_n(\mathbf{w}^\top\mathbf{x}_n+b)-1+\xi_n\bigr]-\sum_n\mu_n\xi_n\qquad(\alpha_n,\mu_n\ge0)
$$

定常性（$\Lambda$ を主変数で微分して 0）：

$$
\frac{\partial\Lambda}{\partial\mathbf{w}}=\mathbf{0}\Rightarrow\mathbf{w}=\sum_n\alpha_nt_n\mathbf{x}_n,\qquad
\frac{\partial\Lambda}{\partial b}=0\Rightarrow\sum_n\alpha_nt_n=0,\qquad
\frac{\partial\Lambda}{\partial\xi_n}=0\Rightarrow\alpha_n+\mu_n=C
$$

これらを $\Lambda$ に代入すると、$\mathbf{w}$ と $b$ と $\xi_n$ が消えて双対問題になる。

$$
\max_{\boldsymbol\alpha}\ \sum_n\alpha_n-\frac12\sum_{n,m}\alpha_n\alpha_mt_nt_m\,\mathbf{x}_n^\top\mathbf{x}_m\quad\text{s.t.}\quad0\le\alpha_n\le C,\ \ \sum_n\alpha_nt_n=0
$$

- 内積 $\mathbf{x}_n^\top\mathbf{x}_m$ を**カーネル** $K(\mathbf{x}_n,\mathbf{x}_m)$ に置き換えるのが**カーネルトリック**。
- 相補スラック性：$\alpha_n\bigl[t_n(\mathbf{w}^\top\mathbf{x}_n+b)-1+\xi_n\bigr]=0$、$\mu_n\xi_n=0$。これより $\alpha_n=0$ ならマージン外の点、$0<\alpha_n<C$ ならマージン上の点（$\xi_n=0$）、$\alpha_n=C$ ならマージン内か誤分類（$\xi_n>0$）。$\alpha_n>0$ のデータが**サポートベクトル**。
- 双対問題は「最大化」。符号を反転した「最小化」の形で書くこともあるが、その場合は目的関数全体の符号が逆になる。

---

## 動的計画法 {#dp}

- **最適性の原理**：最適な行動列の「後半」は、後半だけで見ても最適。これにより、全体の問題を小さな問題の再帰で解ける。
- 状態 $x$、行動 $u$、遷移 $x'=T(x,u)$、コスト $c$ とする。**有限ホライズン**（時刻 $t=0,\dots,\tau$）では、終端から時間をさかのぼって計算する（**後ろ向き帰納**。時刻ごとに価値を伝播させる**再帰的解析**）。

$$
\begin{aligned}
&\text{終端} && V_\tau(x)=\phi(x) \\[2mm]
&\text{再帰（ベルマン方程式）} && V_t(x)=\min_{u\in\mathcal{U}}\Bigl\{c_t(x,u)+V_{t+1}\bigl(T(x,u)\bigr)\Bigr\} \\[2mm]
&\text{定常（無限ホライズン）} && V(x)=\min_{u\in\mathcal{U}}\Bigl\{c(x,u)+\gamma\,V\bigl(T(x,u)\bigr)\Bigr\},\qquad 0\le\gamma<1
\end{aligned}
$$

- $V_t(x)$：時刻 $t$ に状態 $x$ にいるときの、**これから先の最小の総コスト**（価値）。$\phi$：終端コスト。
- 定常の式には**割引率 $\gamma$**（または、コスト 0 の終端状態に必ず着くという条件）が要る。ないと、$V$ が無限大になったり定まらなかったりする。
- 価値を最小でなく**最大**にする（報酬 $r$ を最大化する）と、$\min\to\max$、$c\to r$ に置き換わる。強化学習のベルマン最適方程式 $V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)\bigl[r(s,a,s')+\gamma V^*(s')\bigr]$ はこの確率的な場合（[強化学習](../app/rl.md)）。
- **価値反復**は、右辺で左辺を繰り返し更新して固定点（$V^*$）に近づける方法。状態遷移の確率がわかっているときに使え、わからないときはサンプルから学ぶ（モンテカルロ法・TD 学習）。

### 例：最短経路

S から G へ、辺のコストの和を最小にする。G から順に $V$ を決める。

| 状態 | 行動（次の状態＋辺のコスト） | $V$ | 最適な次 |
|:--|:--|:--:|:--|
| G | — | 0 | — |
| C | G（5） | $5+0=5$ | G |
| D | G（1） | $1+0=1$ | G |
| A | C（2）、D（6） | $\min(2+5,\ 6+1)=7$ | C または D |
| B | C（1）、D（4） | $\min(1+5,\ 4+1)=5$ | D |
| S | A（2）、B（3） | $\min(2+7,\ 3+5)=8$ | B |

- 最短経路は S → B → D → G で、コストは $3+4+1=8$。
- 経路の総数は段数に対して指数的に増えるが、DP は各状態を**1 回ずつ**計算するだけで済む（同じ部分問題の再利用）。

---

## 試験の着眼点

- **ラグランジュ関数 → 双対関数（$\inf$）→ 双対問題（$\max$）→ KKT 条件**の順番。双対問題の変数は $\boldsymbol\lambda,\boldsymbol\nu$。
- KKT の 4 条件：**定常性・主実行可能性・双対実行可能性（$\lambda\ge0$）・相補スラック性（$\lambda g=0$）**。制約が効いていれば $g=0$、効いていなければ $\lambda=0$。
- KKT 条件は、**凸問題なら十分条件**。必要条件にするには正則性条件が要る。
- **弱双対性** $d^*\le p^*$ は常に成り立つ。**強双対性**は凸＋スレイターの条件で成り立つ。
- SVM の双対問題：$\sum\alpha_n-\frac12\sum\sum\alpha_n\alpha_mt_nt_m\mathbf{x}_n^\top\mathbf{x}_m$ を最大化、制約は $0\le\alpha_n\le C$ と $\sum\alpha_nt_n=0$。$\mathbf{w}=\sum\alpha_nt_n\mathbf{x}_n$。
- ニュートン法は **2 階微分（ヘッセ行列）**を使う。勾配降下法は 1 階微分のみ。2 次関数なら 1 回で収束。
- SGD は 1 サンプル、ミニバッチは $|B|$ サンプル、バッチ勾配降下は全サンプルで勾配を計算する。
- 勾配降下法の歩幅の条件：2 次関数で $0<\eta<2/a$。
- 動的計画法の**ベルマン方程式**は、価値を「今のコスト＋次の状態の価値」で書く再帰式。**最適性の原理**が基礎。

---

## 参考

- [Convex Optimization（Boyd, Vandenberghe, 2004）](https://web.stanford.edu/~boyd/cvxbook/)：凸集合・凸関数、双対性、KKT 条件
- [Optimization Methods for Large-Scale Machine Learning（Bottou, Curtis, Nocedal, 2018）](https://arxiv.org/abs/1606.04838)：SGD・ミニバッチ・2 次法
- [Reinforcement Learning: An Introduction（Sutton, Barto, 2018）](http://incompleteideas.net/book/the-book-2nd.html)：ベルマン方程式、動的計画法
