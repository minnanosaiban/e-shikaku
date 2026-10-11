# 決定木・アンサンブル

<p class="kw">キーワード：Random Forest、勾配ブースティング、分類木・回帰木、CART、Gini係数、アンサンブル、バギング、ブースティング</p>

!!! abstract "要点"
    - **決定木**は、特徴量のしきい値による二分岐を再帰的に繰り返して空間を区切る。各分岐は**不純度の減少量 $\Delta I$ が最大**になるように貪欲に選ぶ。解釈しやすいが、単独では過剰適合しやすく不安定。
    - **アンサンブル**は弱い学習器を束ねる。**バギング／ランダムフォレスト**は独立な木の平均で**分散**を減らし、**ブースティング**は前の誤差を次の木で直して**バイアス**を減らす。
    - **勾配ブースティング**は、損失の負の勾配（疑似残差）に木を当てはめて足していく（XGBoost・LightGBM・CatBoost）。

---

## 決定木 {#tree}

**CART**（Classification And Regression Tree）は、各ノードで 1 つの特徴量 $j$ としきい値 $\tau$ を選び、データを $x_j\le\tau$ と $x_j>\tau$ の 2 つに分ける**二分木**。葉に到達したサンプルの代表値を予測とする。

$$
\begin{aligned}
&\text{不純度減少量} && \Delta I=I(t)-\frac{|t_L|}{|t|}\,I(t_L)-\frac{|t_R|}{|t|}\,I(t_R) \\[2mm]
&\text{分岐の選択} && (j^{*},\tau^{*})=\mathop{\rm argmax}_{j,\,\tau}\ \Delta I(j,\tau) \\[2mm]
&\text{分類木の葉} && \hat{y}=\mathop{\rm argmax}_{c}\ p_c\quad\text{（葉の中で最も多いクラス）} \\[2mm]
&\text{回帰木の葉} && \hat{y}=\frac{1}{|t|}\sum_{\mathbf{x}_n\in t}y_n\quad\text{（葉の中の平均）}
\end{aligned}
$$

- $t$：親ノード、$t_L,t_R$：分岐後の左右の子ノード、$|t|$：ノード内のサンプル数、$p_c$：ノード内のクラス $c$ の割合。
- 学習は「全特徴量 × 全しきい値の候補（データ値の中点）」を調べて最良の分岐を選ぶ操作を、**停止条件**（最大の深さ、葉の最小サンプル数、$\Delta I$ が小さい、など）まで子ノードに再帰する。**貪欲法**なので全体最適は保証されない。

### 不純度

$$
\begin{aligned}
&\text{ジニ係数} && I_G(t)=1-\sum_{c}p_c^{\,2} \\[2mm]
&\text{エントロピー} && I_H(t)=-\sum_{c}p_c\log_2p_c \qquad (0\log0=0) \\[2mm]
&\text{誤分類率} && I_E(t)=1-\max_c p_c \\[2mm]
&\text{回帰木（分散）} && I_{\mathrm{var}}(t)=\frac{1}{|t|}\sum_{\mathbf{x}_n\in t}\bigl(y_n-\bar{y}_t\bigr)^2
\end{aligned}
$$

- 純粋なノード（1 クラスだけ）で 0。$C$ クラスで均等なとき最大：ジニ $1-1/C$、エントロピー $\log_2C$（2クラスなら 0.5 と 1）。
- **情報利得**は、不純度にエントロピーを使った $\Delta I$（$I=I_H$）のこと。ジニ係数を使っても同じ形の式（$\Delta I$）で分岐を選ぶ。ジニとエントロピーは似た結果になることが多く、誤分類率は分岐の選択には鈍感で使われない。
- 回帰木では、分岐後の二乗誤差の和 $\sum_{L}(y_n-\bar{y}_L)^2+\sum_{R}(y_n-\bar{y}_R)^2$ を最小にする分岐を選ぶ（$\Delta I$ を最大にするのと同じ）。
- 他のアルゴリズム：ID3・C4.5 はエントロピーを使い、C4.5 は**情報利得比**で多値の特徴量への偏りを補正する。CART は二分岐のみ。

<div class="ew" data-widget="tree-split"></div>

!!! tip "動かしてみる"
    - しきい値 $\tau$ を動かすと、左右のクラスの比率と不純度が変わり、$\Delta I$ の曲線上の点が動きます。左右とも純粋に近づく $\tau\approx4.45$ で $\Delta I$ が最大になります。
    - $\tau\approx2.7$ 付近にも小さな山があります。貪欲法は各ノードで最大の山だけを選びます。端（$\tau\approx0.2$ や $9.8$）は片側がほぼ空になり、$\Delta I\approx0$ です（分けても情報が増えない）。
    - ジニ係数・エントロピー・誤分類率を切り替えても、最大の位置は同じ $\tau\approx4.45$ です。変わるのは $\Delta I$ の大きさ（目盛）です。

### 過剰適合と剪定

- 制限なしに分割すると、訓練データを完全に覚えて**過剰適合**する。木が深いほど分散が大きい。
- 対策：**事前剪定**（最大の深さ・最小サンプル数を制限）、**事後剪定**（大きな木を作ってから、**コスト複雑度**で枝を刈る）：

$$
R_\alpha(T)=\sum_{\text{葉 }\ell}|t_\ell|\,I(t_\ell)+\alpha\,|T|\qquad(|T|：葉の数)
$$

- $\alpha$ が大きいほど小さな木が選ばれる。$\alpha$ は交差検証で決める。
- 長所：解釈しやすい、特徴量の**スケール変換に不変**（標準化不要）、非線形・交互作用を扱える、欠損や混在した型にも対応しやすい。短所：境界が軸に平行な階段状、データの小さな変化で木が変わる（**不安定**＝高分散）、単独では精度が出にくい。
- **特徴量の重要度**：その特徴量による分岐で減った不純度の合計（木全体で集計、ランダムフォレストでは平均）。

---

## アンサンブル学習 {#ensemble}

| 方式 | 作り方 | 主な効果 | 例 |
|:--|:--|:--|:--|
| **バギング** | 学習器を**独立・並列**に学習し、多数決・平均 | **分散**を減らす | ランダムフォレスト |
| **ブースティング** | 学習器を**逐次**に学習し、前の誤りを重点的に直す | **バイアス**を減らす | AdaBoost、勾配ブースティング |
| スタッキング | 複数の学習器の出力を、別のモデル（メタ学習器）の入力にする | 異種モデルを組み合わせる | — |

- 分散 $\sigma^2$ の予測器 $B$ 個を、相関 $\rho$ で平均したときの分散：

$$
\mathrm{Var}\Bigl(\frac1B\sum_{b=1}^{B}h_b\Bigr)=\rho\,\sigma^2+\frac{1-\rho}{B}\,\sigma^2
$$

- $B$ を増やすと第 2 項が消えるが、第 1 項（学習器どうしの相関）は残る。**学習器を互いに似せない**工夫が効く（ブートストラップ、特徴量のサンプリング）。
- 弱い学習器（性能が低いが偶然よりは良い）を使う。バギングには深い木（低バイアス・高分散）、ブースティングには浅い木（高バイアス・低分散）が向く。

---

## バギングとランダムフォレスト {#rf}

$$
\begin{aligned}
&\text{ブートストラップ} && \mathcal{D}_b=\text{訓練データから重複を許して }N\text{ 個抽出},\quad b=1,\dots,B \\[2mm]
&\text{特徴量の抽出} && \text{各分岐で }D\text{ 個の特徴量から }m\text{ 個をランダムに選び、その中で最良の分岐を探す} \\[2mm]
&\text{目安} && m\approx\sqrt{D}\ \text{（分類）},\qquad m\approx D/3\ \text{（回帰）} \\[2mm]
&\text{分類} && \hat{y}=\mathop{\rm argmax}_{c}\sum_{b=1}^{B}\mathbb{1}\bigl[h_b(\mathbf{x})=c\bigr] \\[2mm]
&\text{回帰} && \hat{y}=\frac1B\sum_{b=1}^{B}h_b(\mathbf{x})
\end{aligned}
$$

- **バギング**（Bootstrap Aggregating）は、ブートストラップ標本ごとに学習器を作って平均する。**ランダムフォレスト**はこれに「分岐ごとの特徴量のランダム抽出」を加え、木どうしの相関 $\rho$ を下げる。
- 1 つの標本に入らないデータの割合は $(1-1/N)^N\to e^{-1}\approx36.8\%$（入る割合は約 63.2%）。この **OOB（out-of-bag）サンプル**を検証用に使うと、別に検証データを用意せずに汎化誤差を推定できる。
- 木は互いに独立なので**並列化できる**。ハイパーパラメータ（$B$・$m$・木の深さ）に比較的鈍感で、過剰適合しにくく扱いやすい。$B$ を増やしても過剰適合は悪化しない。
- 重要度の求め方：不純度の減少量の合計（MDI）、特徴量をシャッフルしたときの精度低下（置換重要度）。

---

## ブースティング {#boosting}

### AdaBoost

ラベル $y_n\in\{-1,+1\}$、サンプルの重み $w_n$（初期値 $1/N$）。各回 $t$ で、重み付きの誤り率が小さい弱学習器 $h_t$ を作り、誤分類された点の重みを増やす。

$$
\begin{aligned}
&\text{誤り率} && \varepsilon_t=\frac{\sum_nw_n\,\mathbb{1}\bigl[h_t(\mathbf{x}_n)\ne y_n\bigr]}{\sum_nw_n} \\[2mm]
&\text{学習器の重み} && \alpha_t=\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t} \\[2mm]
&\text{サンプルの重み} && w_n\leftarrow w_n\exp\bigl(-\alpha_t\,y_nh_t(\mathbf{x}_n)\bigr)\quad(\text{和が 1 になるよう正規化}) \\[2mm]
&\text{最終予測} && H(\mathbf{x})=\mathrm{sign}\Bigl(\sum_{t=1}^{T}\alpha_t\,h_t(\mathbf{x})\Bigr)
\end{aligned}
$$

- $\varepsilon_t<0.5$ なら $\alpha_t>0$。誤分類された点は $e^{\alpha_t}$ 倍、正しい点は $e^{-\alpha_t}$ 倍になり、次の学習器は難しい点に集中する。指数損失 $e^{-yF}$ を段階的に最小化する手法と見なせる。

### 勾配ブースティング

関数空間での勾配降下法。現在のモデル $F_{t-1}$ の**損失の負の勾配（疑似残差）**に木を当てはめ、学習率 $\eta$ を掛けて足す。

$$
\begin{aligned}
&\text{初期化} && F_0(\mathbf{x})=\mathop{\rm argmin}_{c}\sum_{n=1}^{N}L(y_n,\,c) \\[2mm]
&\text{疑似残差} && r_{tn}=-\frac{\partial L\bigl(y_n,\,F(\mathbf{x}_n)\bigr)}{\partial F(\mathbf{x}_n)}\Bigg|_{F=F_{t-1}} \\[2mm]
&\text{木の学習} && h_t=\mathop{\rm argmin}_{h}\sum_{n=1}^{N}\bigl(r_{tn}-h(\mathbf{x}_n)\bigr)^2 \\[2mm]
&\text{更新} && F_t(\mathbf{x})=F_{t-1}(\mathbf{x})+\eta\,h_t(\mathbf{x}) \\[2mm]
&\text{予測} && \hat{y}=F_T(\mathbf{x})=F_0(\mathbf{x})+\eta\sum_{t=1}^{T}h_t(\mathbf{x})
\end{aligned}
$$

| 損失 $L(y,F)$ | 疑似残差 $r=-\partial L/\partial F$ | 備考 |
|:--|:--|:--|
| 二乗誤差 $\tfrac12(y-F)^2$ | $y-F$（普通の残差） | 回帰の標準 |
| 絶対誤差 $\lvert y-F\rvert$ | $\mathrm{sign}(y-F)$ | 外れ値に頑健 |
| 対数損失（$y\in\{0,1\}$、$F$ はロジット） | $y-\sigma(F)$ | 分類。ロジスティック回帰の勾配と同じ形 |

- **学習率** $\eta$（縮小、shrinkage）：小さいほど過剰適合しにくいが、必要な木の本数 $T$ が増える。木は**浅く**（深さ 3〜8 程度）、**早期終了**（検証誤差が悪化したら止める）、行・列の**サブサンプリング**を併用する。
- 逐次に学習するので並列化しにくく、ハイパーパラメータが多い。ただし表形式データでは最高精度が出やすい。

### XGBoost・LightGBM・CatBoost

- **XGBoost**：損失を 2 次まで展開（勾配 $g_n=\partial L/\partial\hat{y}_n$、ヘッセ $h_n=\partial^2L/\partial\hat{y}_n^2$）し、木の複雑さへの罰則 $\Omega=\gamma T+\tfrac12\lambda\sum_jw_j^2$（$T$：葉の数、$w_j$：葉の値）を目的に含める。葉 $j$ に入るサンプルの $G_j=\sum g_n$、$H_j=\sum h_n$ から

$$
w_j^{*}=-\frac{G_j}{H_j+\lambda},\qquad
\text{Gain}=\frac12\Bigl[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Bigr]-\gamma
$$

- **LightGBM**：ヒストグラムで高速化し、損失の減少が大きい葉から伸ばす（leaf-wise）。勾配の大きいデータを優先的に使う GOSS、排他的な特徴量をまとめる EFB。
- **CatBoost**：カテゴリ変数を目的変数の統計量で扱い（ordered target statistics）、予測のずれを避ける ordered boosting。

| | ランダムフォレスト | 勾配ブースティング |
|:--|:--|:--|
| 学習 | 並列（独立な木） | 逐次（前の誤差を修正） |
| 木 | 深い | 浅い |
| 主に減らすもの | 分散 | バイアス（と分散） |
| 過剰適合 | しにくい | $T$・$\eta$・深さの調整が必要 |
| 調整 | ほぼ不要 | 多い |

<div class="ew" data-widget="tree-boost"></div>

!!! tip "動かしてみる"
    - 勾配ブースティングで $T=1$ にすると、平均 $F_0$ に $\eta$ 倍した木 1 本を足しただけで、ほぼ平らな階段です。$T$ を増やすと、残差を少しずつ削って曲線に近づきます。
    - $\eta=1$、深さ 4、$T=50$ にすると訓練 RMSE がほぼ 0 になる一方、テスト RMSE が増えます（過剰適合）。$\eta=0.3$、深さ 2、$T=10$ 付近が良い点です。
    - バギングに切り替え（深さ 4）、$B=1$ と $B=50$ を比べると、1 本の木はデータのノイズに振り回されて階段が不安定ですが、50 本の平均はなめらかで、テスト RMSE が約 0.45 から 0.35 に下がります（分散の減少）。

---

## 試験の着眼点

- **ジニ係数** $1-\sum p_c^2$、**エントロピー** $-\sum p_c\log p_c$。不純度減少量 $\Delta I$ が最大の分岐を選ぶ。回帰木は分散（二乗誤差）。
- 木が深いと過剰適合 ⇒ 剪定・最大の深さの制限。決定木は標準化不要。
- バギング＝並列・**分散を減らす**、ブースティング＝逐次・**バイアスを減らす**。
- ランダムフォレスト＝バギング＋**特徴量のランダム抽出**（分類 $\sqrt{D}$、回帰 $D/3$ が目安）。OOB で汎化誤差を推定できる。
- 勾配ブースティング：初期値 → 疑似残差（負の勾配）→ 木で近似 → $\eta$ 倍して加算。二乗誤差なら疑似残差は普通の残差 $y-F$。
- AdaBoost は誤分類された点の重みを増やす。XGBoost は 2 次の勾配と葉への正則化、LightGBM は高速化、CatBoost はカテゴリ変数。
- バイアスとバリアンスの関係は [評価](evaluation.md)、損失とその勾配は [損失関数](../math/loss.md)。

---

## 参考

- [Random Forests（Breiman, 2001）](https://doi.org/10.1023/A:1010933404324)
- [Greedy function approximation: A gradient boosting machine（Friedman, 2001）](https://doi.org/10.1214/aos/1013203451)
- [XGBoost: A Scalable Tree Boosting System（Chen, Guestrin, 2016）](https://arxiv.org/abs/1603.02754)
- [scikit-learn User Guide: Decision Trees](https://scikit-learn.org/stable/modules/tree.html)、[Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
- [The Elements of Statistical Learning（Hastie, Tibshirani, Friedman）](https://hastie.su.domains/ElemStatLearn/)：第9・10・15章
