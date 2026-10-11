# 説明性（解釈可能性）

<p class="kw">キーワード：XAI（説明可能な AI）、大域的な説明・局所的な説明、特徴量重要度、PDP、サロゲートモデル、勾配ベースの可視化、CAM、Grad-CAM、Integrated Gradients、加法的特徴帰属法、LIME、SHAP、シャープレイ値、局所的正確性・欠落性・一貫性</p>

!!! abstract "要点"
    - 深層学習は**なぜその予測をしたのか**が見えにくい。説明性は、判断の根拠を示して、**誤り・偏り・近道（ショートカット）の検出**や、利用者の信頼につなげる。
    - 説明は、モデル全体の振る舞いを示す**大域的**な説明と、**1 つの予測**の根拠を示す**局所的**な説明に分かれる。深層学習の画像では **Grad-CAM**（特徴マップ × 勾配）、表形式などでは **LIME**（局所的な線形近似）と **SHAP**（シャープレイ値）が代表。
    - **SHAP** は、特徴量を協力ゲームの「プレイヤー」とみなして貢献度を公平に配分する。局所的正確性・欠落性・一貫性を**すべて満たす解が 1 つだけ**存在する。

---

## 説明の分類 {#taxonomy}

| 観点 | 種類 | 内容・例 |
|:--|:--|:--|
| 範囲 | **大域的**（global、モデルの説明） | モデルの全体の傾向・中身。線形モデルの係数、決定木、特徴量重要度、PDP、複雑なモデルを解釈しやすいモデルで近似（**サロゲート**） |
| | **局所的**（local、予測の説明） | **1 つのデータ**の予測に効いた特徴。LIME、SHAP、Grad-CAM |
| 適用先 | モデル非依存（model-agnostic） | 中身を見ず、入出力だけを使う。LIME、SHAP（Kernel SHAP）、PDP、順列重要度 |
| | モデル依存 | 内部（勾配・特徴マップ）を使う。Grad-CAM、Integrated Gradients |
| 時期 | **本質的に解釈可能** | 最初から解釈しやすい（線形回帰・決定木）。精度とのトレードオフがある |
| | **事後的**（post-hoc） | 学習済みのブラックボックスを後から説明する |

---

## 大域的な説明 {#global}

- **線形モデル・決定木**：係数や分岐がそのまま説明になる。
- **特徴量重要度**：木のモデルでは分割による不純度の減少の合計。**順列重要度**（permutation importance）は、ある特徴の値をシャッフルしたときの**精度の低下量**。モデルに依らない。
- **PDP**（部分依存プロット）：注目する特徴 $\mathbf{x}_S$ の値を変えたときの、予測の**平均的な変化**。ほかの特徴 $\mathbf{x}_C$ は、データの値で平均して消す。

$$
\hat{f}_S(\mathbf{x}_S)=\frac1n\sum_{i=1}^{n}f\bigl(\mathbf{x}_S,\ \mathbf{x}_C^{(i)}\bigr)
$$

- **サロゲートモデル**：ブラックボックスの**出力**を、決定木や線形モデルで近似し、その近似モデルを説明に使う。**感応度分析**は、入力を少し変えたときの出力の変化を調べる。

---

## 勾配と特徴マップによる可視化（画像） {#gradient}

- **サリエンシーマップ**：クラス $c$ のスコア $y^c$ の、入力画素についての**勾配の大きさ** $\bigl|\partial y^c/\partial x_i\bigr|$。「画素を少し変えると、スコアが大きく変わる場所」。ノイズが多い。**Grad×Input**、**SmoothGrad**（ノイズを加えたものの平均）、**Guided Backprop** がある。
- **Integrated Gradients**：基準の入力 $\mathbf{x}'$（黒い画像など）から $\mathbf{x}$ まで**直線で結んだ経路で、勾配を積分**する。勾配が**飽和**していても寄与を拾える。

$$
\mathrm{IG}_i(\mathbf{x})=(x_i-x_i')\int_0^1\frac{\partial F\bigl(\mathbf{x}'+\alpha(\mathbf{x}-\mathbf{x}')\bigr)}{\partial x_i}\,d\alpha,\qquad \sum_i\mathrm{IG}_i(\mathbf{x})=F(\mathbf{x})-F(\mathbf{x}')\quad(\text{完全性})
$$

### CAM {#cam}

- **CAM**（Class Activation Mapping）：最後の畳み込み層の特徴マップ $f_k(x,y)$（$k$ はチャネル）を、**クラス $c$ の重み** $w_k^c$ で足し合わせ、**判断の根拠の領域**を示す。ネットワークの最後が「**GAP（全域平均プーリング）→ 全結合層 1 つ**」であることが条件。

$$
S_c=\sum_kw_k^c\,F_k,\quad F_k=\frac1Z\sum_{x,y}f_k(x,y)\ \ \Longrightarrow\ \ S_c=\frac1Z\sum_{x,y}M_c(x,y),\qquad M_c(x,y)=\sum_kw_k^c\,f_k(x,y)
$$

- クラスのスコア $S_c$ は、$M_c$ を全位置で平均したもの。つまり $M_c(x,y)$ は、**各位置が $S_c$ にどれだけ寄与したか**。入力の大きさに拡大して、画像に重ねる。
- **課題**：GAP と全結合層にネットワークを**作り替えて再学習**する必要がある。分類モデルにしか使えない。

### Grad-CAM {#gradcam}

- **Grad-CAM**：CAM の重み $w_k^c$ を、**勾配**から作る。**学習済みの任意の CNN**（そのまま）に使える。特徴マップ $A^k$（最後の畳み込み層）について：

$$
\alpha_k^c=\frac1Z\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A^k_{ij}},\qquad
L^c_{\text{Grad-CAM}}=\mathrm{ReLU}\Bigl(\sum_k\alpha_k^c\,A^k\Bigr)
$$

- $y^c$：クラス $c$ の**ソフトマックスの前**のスコア、$Z$：特徴マップの画素数。
    1. 順伝播で特徴マップ $A^k$ を得る。
    2. 逆伝播で $\partial y^c/\partial A^k_{ij}$ を求め、**空間方向に平均**して重み $\alpha_k^c$ にする（勾配の GAP）。
    3. 重み付きの和を取り、**ReLU**（クラスに**正に**寄与する場所だけを残す）をかける。入力の大きさに拡大して重ねる。
- **CAM との関係**：GAP + 全結合のネットワークなら $y^c=\sum_kw_k^c\frac1Z\sum_{ij}A^k_{ij}$ なので $\partial y^c/\partial A^k_{ij}=w_k^c/Z$、すなわち $\alpha_k^c=w_k^c/Z$。**CAM と（定数倍を除いて）一致する**。Grad-CAM は CAM の**一般化**。
- **使い方**：誤分類の原因（背景だけを見ている、など）や、学習データの**偏り・ショートカット**（職業の分類で、顔や髪型を見ている）の発見。
- **課題**：勾配は**入力の小さな変化に敏感**で、ノイズが乗る。勾配が**飽和**すると消える。勾配を使わず、特徴マップを入力のマスクとして使って**出力の変化**で重みを決める **Score-CAM** などがある。

---

## LIME と SHAP {#lime-shap}

### 加法的特徴帰属法 {#additive}

- 説明したい 1 つの入力 $\mathbf{x}$ の周りで、モデル $f$ を**特徴の線形和**で近似する枠組み。$M$ 個の特徴のそれぞれが「ある（1）／ない（0）」を表すベクトルを $\mathbf{z}'\in\{0,1\}^M$ として：

$$
g(\mathbf{z}')=\phi_0+\sum_{i=1}^{M}\phi_i\,z_i'
$$

- $\phi_i$：特徴 $i$ の**貢献度**（重要度）、$\phi_0$：何も情報がないときの予測（基準値）。LIME と SHAP は、どちらもこの形。違いは $\phi_i$ の決め方。

### LIME {#lime}

- **LIME**（Local Interpretable Model-agnostic Explanations）：説明したい入力のまわりで、データを**少しずつ壊して**（画像なら領域を隠す、文章なら語を消す）ブラックボックス $f$ に通し、その出力を**近い点を重視した解釈しやすいモデル $g$**（疎な線形モデル）で近似する。$g$ の係数が重要度。

$$
\xi(\mathbf{x})=\arg\min_{g\in G}\ L(f,g,\pi_{\mathbf{x}})+\Omega(g),\qquad
L(f,g,\pi_{\mathbf{x}})=\sum_{\mathbf{z}}\pi_{\mathbf{x}}(\mathbf{z})\bigl(f(\mathbf{z})-g(\mathbf{z}')\bigr)^2,\qquad
\pi_{\mathbf{x}}(\mathbf{z})=\exp\Bigl(-\frac{D(\mathbf{x},\mathbf{z})^2}{\sigma^2}\Bigr)
$$

- $G$：解釈しやすいモデルの集合、$\Omega(g)$：$g$ の**複雑さ**への罰（使う特徴の数など）、$\pi_{\mathbf{x}}$：$\mathbf{x}$ に**近いサンプルほど大きい重み**（$D$ は距離、$\sigma$ は近さの幅）。近い範囲だけで**局所的に**当てはまればよいので、全体は複雑でも、局所は線形で近似できる。
- **モデルを選ばない**（深層学習・SVM・ランダムフォレスト）。弱点：サンプリングに依るので結果が**不安定**、近さの定め方に依る、理論的な保証が弱い。

### SHAP {#shap}

- **シャープレイ値**は、協力ゲーム理論で、チームの成果を各プレイヤーに**公平に配分**する方法。**特徴量＝プレイヤー**、**報酬＝予測値と平均的な予測値の差**として使う。特徴 $i$ の貢献度は、**ほかの特徴の部分集合 $S$ に $i$ を加えたときの予測の増分（限界貢献）**を、**あらゆる加える順序で平均**したもの。

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(n-|S|-1)!}{n!}\Bigl[f_{\mathbf{x}}(S\cup\{i\})-f_{\mathbf{x}}(S)\Bigr],\qquad f_{\mathbf{x}}(S)=\mathbb{E}\bigl[f(\mathbf{X})\mid\mathbf{X}_S=\mathbf{x}_S\bigr]
$$

- $N$：全特徴の集合（$n$ 個）、$f_{\mathbf{x}}(S)$：特徴 $S$ だけが分かっているときの予測の期待値。重み $\frac{|S|!(n-|S|-1)!}{n!}$ は、「$S$ が先に並び、次に $i$ が来る」順序の割合で、総和は 1。
- **例**：特徴が 2 つ。平均的な予測 $f_{\mathbf{x}}(\emptyset)=10$、特徴 1 だけ分かると 14、特徴 2 だけで 11、両方で 17（実際の予測）。

$$
\phi_1=\tfrac12(14-10)+\tfrac12(17-11)=5,\qquad \phi_2=\tfrac12(11-10)+\tfrac12(17-14)=2,\qquad 10+5+2=17
$$

- 貢献度の合計が「**予測 $-$ 平均**」になる（効率性）。特徴の順序で増分が変わるので、全順序の平均を取る。
- **SHAP の主定理**：加法的特徴帰属法 $g$ が次の **3 つの性質**をすべて満たすのは、シャープレイ値だけ。

| 性質 | 内容 |
|:--|:--|
| **局所的正確性**（local accuracy） | $f(\mathbf{x})=g(\mathbf{x}')=\phi_0+\sum_i\phi_i x_i'$。説明モデルが、その入力での予測と**一致** |
| **欠落性**（missingness） | 欠けている特徴（$x_i'=0$）は貢献度 0：$x_i'=0\Rightarrow\phi_i=0$ |
| **一貫性**（consistency） | モデルを変えて、ある特徴の限界貢献が**どの部分集合でも減らない**なら、その貢献度も**減らない** |

- 計算量は特徴の部分集合の数 $2^n$ で、そのままでは重い。**Kernel SHAP**（任意のモデル。シャープレイ用の重みつきの線形回帰で近似）、**Tree SHAP**（木のモデル。高速）、**Deep SHAP**（深層学習）、**Linear SHAP** など。
- 局所の説明を全データで集計すれば、大域的な重要度にもなる。

| | LIME | SHAP |
|:--|:--|:--|
| 考え方 | 近傍のサンプルへの**局所線形近似** | **シャープレイ値**（公理を満たす一意の配分） |
| 結果の安定性 | サンプリングで変動しやすい | 理論的な裏づけがあり安定（計算は重い） |
| 加法的特徴帰属 | ○ | ○（局所的正確性を保証） |

---

## 注意点 {#caveats}

- **Attention の重み**の可視化は、直感的だが、**説明そのものとは限らない**（重みを変えても出力が変わらない場合がある）。
- 説明手法の結果は、**正解がない**ので評価が難しい。重要と言われた部分を**消すと予測が落ちるか**（忠実性）、モデルの重みを**ランダムにすると説明が変わるか**（サニティチェック。画像の説明手法には、モデルの重みに依らず「それらしい」図を出してしまうものがある）を確認する。
- 相関の強い特徴では、貢献度の配分が曖昧になる。

---

## 試験の着眼点

- **局所的な説明**（1 つの予測）：LIME・SHAP・Grad-CAM。**大域的な説明**（モデル全体）：線形回帰・決定木・特徴量重要度・PDP・サロゲート。
- **CAM**：特徴マップ $\sum_kw_k^c f_k$。**GAP + 全結合**が条件で、再学習が必要。**Grad-CAM**：重み $\alpha_k^c$ を**勾配の空間平均**で求める。**ReLU** をかける。任意の CNN に使える。
- Grad-CAM の重み $=\frac1Z\sum\partial y^c/\partial A^k_{ij}$。$y^c$ は**ソフトマックス前**のスコア。
- **LIME**：近傍で線形近似、**モデル非依存**、近いサンプルほど大きな重み $\pi_{\mathbf{x}}$。**SHAP**：シャープレイ値、**局所的正確性・欠落性・一貫性**を満たす唯一の解。
- シャープレイ値の式：**部分集合の大きさに依る重み**で、限界貢献 $f(S\cup\{i\})-f(S)$ を平均。貢献度の和 $=$ 予測 $-$ 平均予測。
- **Integrated Gradients**：基準入力からの経路で勾配を積分、**完全性**。

---

## 参考

- [Learning Deep Features for Discriminative Localization（CAM, Zhou ほか, 2015）](https://arxiv.org/abs/1512.04150)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization（Selvaraju ほか, 2016）](https://arxiv.org/abs/1610.02391)
- [Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks（Wang ほか, 2019）](https://arxiv.org/abs/1910.01279)
- [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps（Simonyan ほか, 2013）](https://arxiv.org/abs/1312.6034)
- [Axiomatic Attribution for Deep Networks（Integrated Gradients, Sundararajan ほか, 2017）](https://arxiv.org/abs/1703.01365)
- [SmoothGrad: removing noise by adding noise（Smilkov ほか, 2017）](https://arxiv.org/abs/1706.03825)
- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier（LIME, Ribeiro ほか, 2016）](https://arxiv.org/abs/1602.04938)
- [A Unified Approach to Interpreting Model Predictions（SHAP, Lundberg, Lee, 2017）](https://arxiv.org/abs/1705.07874)
- [Sanity Checks for Saliency Maps（Adebayo ほか, 2018）](https://arxiv.org/abs/1810.03292)
- [Attention is not Explanation（Jain, Wallace, 2019）](https://arxiv.org/abs/1902.10186)
- [Interpretable Machine Learning（Molnar）](https://christophm.github.io/interpretable-ml-book/)
