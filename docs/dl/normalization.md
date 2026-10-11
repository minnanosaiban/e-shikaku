# 正規化

<p class="kw">キーワード：Batch Normalization、Layer Normalization、Instance Normalization、Group Normalization、Local Response Normalization、勾配クリッピング</p>

!!! abstract "要点"
    - 中間層の出力を、**平均 0・分散 1 に標準化**してから、学習できるスケール $\gamma$ とシフト $\beta$ で再調整する。学習が安定して速くなり、大きな学習率が使える。
    - 手法の違いは、**平均と分散をどの範囲で計算するか（正規化の軸）**だけ。BN は「バッチ・空間」、LN は「チャネル・空間（1サンプル）」、IN は「空間のみ」、GN は「チャネルのグループ・空間」。
    - BN は**バッチ統計**に頼るので、バッチが小さい・系列モデルでは不利。LN（Transformer・RNN）、GN（小バッチの検出・セグメンテーション）、IN（スタイル変換）はバッチに依存しない。
    - 勾配爆発は別の対策で、**勾配クリッピング**（ノルムの上限を決めて縮める）を使う。

---

## 正規化の軸 {#axes}

入力を $(N, C, H, W)$ の 4 次元テンソル（サンプル、チャネル、高さ、幅。全結合層なら $(N, D)$）とみて、平均と分散を取る**範囲 $S$** だけが違う。

$$
\mu=\frac{1}{|S|}\sum_{i\in S}x_i,\qquad
\sigma^2=\frac{1}{|S|}\sum_{i\in S}(x_i-\mu)^2,\qquad
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y_i=\gamma\hat{x}_i+\beta
$$

| 手法 | 統計を取る範囲 $S$ | 統計量の組の数 | $\lvert S\rvert$ |
|:--|:--|:--|:--|
| **Batch Norm** | $(N,H,W)$：チャネル $c$ ごとに、全サンプル・全位置 | $C$ | $NHW$ |
| **Layer Norm** | $(C,H,W)$：サンプル $n$ ごとに、全チャネル・全位置 | $N$ | $CHW$ |
| **Instance Norm** | $(H,W)$：サンプル $n$・チャネル $c$ ごとに、位置だけ | $NC$ | $HW$ |
| **Group Norm** | $(C/G,H,W)$：サンプル $n$・グループ $g$ ごとに | $NG$ | $(C/G)HW$ |

- $\epsilon$（$10^{-5}$ など）は 0 除算の防止。$\gamma,\beta$ は**学習するパラメータ**で、正規化で失われた表現力を戻す（$\gamma=\sqrt{\sigma^2+\epsilon},\ \beta=\mu$ なら元に戻せる）。
- **GN で $G=1$** にすると全チャネルが 1 グループで、**LN** と同じ。**$G=C$** にすると 1 チャネル 1 グループで、**IN** と同じ。GN はこの中間にある（例：$G=32$）。

<div class="ew" data-widget="norm-axes"></div>

!!! tip "動かしてみる"
    - BN に切り替えて $n$ を動かしても、紺の範囲（同じチャネルの全サンプル）は変わりません。$c$ を動かすと列ごと動きます。統計量の組は $C$ 個です。
    - LN は逆に、$n$ を動かすと行ごと動きます。1 サンプルの中で閉じているので、**他のサンプルに影響されません**。
    - IN は 1 セルだけ（空間 16 要素）。GN（$G=2$, $3$）は、隣り合うチャネルのかたまりを使い、LN と IN の中間になります。

---

## Batch Normalization {#batchnorm}

ミニバッチの統計で各チャネルを標準化する（Ioffe, Szegedy, 2015）。

$$
\begin{aligned}
&\text{学習時}\quad
\mu_{\mathcal{B},c}=\frac{1}{m}\sum_{i=1}^{m}x_{i,c},\qquad
\sigma_{\mathcal{B},c}^2=\frac{1}{m}\sum_{i=1}^{m}(x_{i,c}-\mu_{\mathcal{B},c})^2 \\[2mm]
&\hspace{3.2em}\hat{x}_{i,c}=\frac{x_{i,c}-\mu_{\mathcal{B},c}}{\sqrt{\sigma_{\mathcal{B},c}^2+\epsilon}},\qquad
y_{i,c}=\gamma_c\hat{x}_{i,c}+\beta_c \\[4mm]
&\text{推論時}\quad
\hat{x}_c=\frac{x_c-\mathbb{E}[x]_c}{\sqrt{\mathrm{Var}[x]_c+\epsilon}},\qquad
y_c=\gamma_c\hat{x}_c+\beta_c
\end{aligned}
$$

- $m$：チャネル $c$ の統計に使う要素数。全結合層なら $m=N$、畳み込み層なら**空間位置も含めて** $m=N\cdot H\cdot W$。
- **推論時**は 1 サンプルずつ処理するので、学習中に**移動平均**で蓄えた平均・分散（$\mathbb{E}[x]_c,\ \mathrm{Var}[x]_c$）を使う。

$$
\mathbb{E}[x]_c\leftarrow\alpha\,\mathbb{E}[x]_c+(1-\alpha)\,\mu_{\mathcal{B},c},\qquad
\mathrm{Var}[x]_c\leftarrow\alpha\,\mathrm{Var}[x]_c+(1-\alpha)\,\sigma_{\mathcal{B},c}^2
$$

- $\alpha$：移動平均の係数で、学習率 $\eta$ とは別のもの。PyTorch の `momentum` は**新しい統計に掛ける重み**（$=1-\alpha$、既定 0.1）なので、向きが逆。
- 推論時の BN は $y=\dfrac{\gamma}{\sqrt{\mathrm{Var}+\epsilon}}x+\Bigl(\beta-\dfrac{\gamma\,\mathbb{E}[x]}{\sqrt{\mathrm{Var}+\epsilon}}\Bigr)$ という**固定のアフィン変換**なので、直前の畳み込み・全結合層に**吸収できる**（推論の高速化）。
- 学習時と推論時で動作が違う（PyTorch の `train()` / `eval()`）。

??? note "逆伝播（勾配）"
    $g_i=\partial L/\partial\hat{x}_i=\gamma\,\partial L/\partial y_i$、$\tilde\sigma=\sqrt{\sigma^2+\epsilon}$ とする（チャネルを 1 つ取り出して書く）。

    $$
    \frac{\partial L}{\partial\gamma}=\sum_i\frac{\partial L}{\partial y_i}\hat{x}_i,\qquad
    \frac{\partial L}{\partial\beta}=\sum_i\frac{\partial L}{\partial y_i},\qquad
    \frac{\partial L}{\partial x_i}=\frac{1}{m\tilde\sigma}\Bigl(m\,g_i-\sum_j g_j-\hat{x}_i\sum_j g_j\hat{x}_j\Bigr)
    $$

    - $\mu,\sigma^2$ も $x_i$ の関数なので、バッチ内の**全サンプルに勾配が回り込む**（サンプルどうしが依存する）。
    - 検算：$\sum_i\partial L/\partial x_i=0$（$\sum_i\hat{x}_i=0$ のため）。入力全体を同じだけずらしても出力が変わらないことと整合する。

- **効果**：学習が安定して速くなる（大きな学習率が使える）、初期値への依存が減る、ミニバッチ統計のノイズによる**正則化**の効果（ドロップアウトを減らせることが多い）。当初は**内部共変量シフト**（層の入力分布が学習中に変わること）の軽減が理由とされたが、後の研究では**損失面がなめらかになる**効果が主だとされている。
- **弱点**：バッチが小さいと統計が不安定になる。系列の長さが可変な RNN や、サンプルごとに分布が違う生成タスクには向かない。分散学習ではデバイス間で統計を同期する必要がある。
- 畳み込み → BN → 活性化 の順が一般的。BN が平均を引くので、直前の層の**バイアスは不要**になる。

---

## Layer Normalization {#layernorm}

$$
\mu_n=\frac{1}{K}\sum_{k=1}^{K}x_n^{(k)},\qquad
\sigma_n^2=\frac{1}{K}\sum_{k=1}^{K}\bigl(x_n^{(k)}-\mu_n\bigr)^2,\qquad
y_n^{(k)}=\gamma^{(k)}\frac{x_n^{(k)}-\mu_n}{\sqrt{\sigma_n^2+\epsilon}}+\beta^{(k)}
$$

- **1 サンプルの特徴 $K$ 個全体**で正規化する（$K$ は層の特徴数。畳み込みでは $C\cdot H\cdot W$）。**バッチサイズに依存せず、学習時と推論時の処理が同じ**。
- RNN（時刻ごとに適用）と **Transformer**（トークンごとに $d_{\mathrm{model}}$ 次元で適用）の標準。系列長が可変でも使える。→ [Transformer](transformer.md)
- $\gamma,\beta$ は特徴ごとに持つ（BN のようにチャネル単位ではない）。
- **RMSNorm**：平均を引かず、二乗平均平方根だけで割る簡略版。$y^{(k)}=\gamma^{(k)}x^{(k)}/\sqrt{\tfrac1K\sum_k (x^{(k)})^2+\epsilon}$。計算が軽く、大規模言語モデルで使われる。

---

## Instance Normalization {#instancenorm}

$$
\mu_{n,c}=\frac{1}{S}\sum_{s=1}^{S}x_{n,c}^{(s)},\qquad
\sigma_{n,c}^2=\frac{1}{S}\sum_{s=1}^{S}\bigl(x_{n,c}^{(s)}-\mu_{n,c}\bigr)^2
$$

- **サンプルごと・チャネルごと**に、空間位置 $S=HW$ だけで正規化する。画像の**コントラストや明るさ（スタイル）**がチャネル平均・分散に現れるため、IN はそれらを取り除く。
- **スタイル変換**や画像生成で使う。**AdaIN** は、内容画像 $\mathbf{x}$ の統計を、スタイル画像 $\mathbf{y}$ の統計に差し替える：$\mathrm{AdaIN}(\mathbf{x},\mathbf{y})=\sigma(\mathbf{y})\dfrac{\mathbf{x}-\mu(\mathbf{x})}{\sigma(\mathbf{x})}+\mu(\mathbf{y})$。
- 学習時と推論時の処理は同じ（PyTorch の `InstanceNorm2d` は既定で $\gamma,\beta$ なし）。

---

## Group Normalization {#groupnorm}

$$
\mu_{n,g}=\frac{1}{|\mathcal{G}_{n,g}|}\sum_{j\in\mathcal{G}_{n,g}}x_j,\qquad
\sigma_{n,g}^2=\frac{1}{|\mathcal{G}_{n,g}|}\sum_{j\in\mathcal{G}_{n,g}}(x_j-\mu_{n,g})^2,\qquad |\mathcal{G}_{n,g}|=\frac{C}{G}\,HW
$$

- $C$ 個のチャネルを $G$ 個のグループに分け、**サンプルごと・グループごと**に、グループ内のチャネルと空間位置で正規化する。$\gamma,\beta$ はチャネルごと。
- **バッチサイズに依存しない**ので、メモリの都合で小バッチしか使えない**物体検出・セグメンテーション**で BN の代わりになる。バッチが大きいときの精度は BN にやや劣る。
- 隣り合うチャネルは似た特徴（同じ向きのエッジなど）を表しやすい、という考えから、グループ化が自然になる。

---

## 手法の比較 {#compare}

| | BN | LN | IN | GN |
|:--|:--|:--|:--|:--|
| バッチサイズへの依存 | あり（小さいと不安定） | なし | なし | なし |
| 学習時と推論時 | **違う**（移動平均を使う） | 同じ | 同じ | 同じ |
| 主な用途 | CNN 全般（画像分類） | RNN・Transformer | スタイル変換・画像生成 | 小バッチの検出・セグメンテーション |
| $\gamma,\beta$ | チャネルごと | 特徴ごと | チャネルごと（省略も多い） | チャネルごと |

### Local Response Normalization（LRN） {#lrn}

$$
b_{x,y}^{i}=\frac{a_{x,y}^{i}}{\Bigl(k+\alpha\sum_{j=\max(0,\,i-n/2)}^{\min(C-1,\,i+n/2)}\bigl(a_{x,y}^{j}\bigr)^2\Bigr)^{\beta}}
$$

- 同じ位置の**隣り合うチャネル**どうしで、強く反応したチャネルが近くのチャネルを抑える（側抑制）。AlexNet が採用（$k=2,\ n=5,\ \alpha=10^{-4},\ \beta=0.75$）。学習するパラメータはなく、今は BN に置き換えられている。

---

## 勾配クリッピング {#gradient-clipping}

勾配爆発を防ぐため、勾配ベクトルの**ノルム**が閾値 $v$ を超えたら縮める。RNN や深いネットワークで使う。

$$
\lVert\mathbf{g}\rVert=\sqrt{\sum_k\left\lVert\frac{\partial L}{\partial\mathbf{W}_k}\right\rVert_F^2},\qquad
\mathbf{g}\leftarrow\begin{cases}\dfrac{v}{\lVert\mathbf{g}\rVert}\,\mathbf{g}&(\lVert\mathbf{g}\rVert>v)\\[2mm]\mathbf{g}&(\text{それ以外})\end{cases}
$$

- $\mathbf{g}$：**全パラメータの勾配を 1 本につないだベクトル**。縮めるのは大きさだけで、**向きは変わらない**。
- 更新量の上限が決まるので、急な崖のある損失面でも一度に飛び出さない。→ [RNN](rnn.md)（勾配消失・勾配爆発）
- 各要素を $[-v,v]$ に切る「値のクリッピング」もあるが、こちらは向きが変わる。

---

## 試験の着眼点

- BN・LN・IN・GN の違いは**平均・分散を取る範囲**。BN＝チャネルごとに $(N,H,W)$、LN＝サンプルごとに $(C,H,W)$、IN＝サンプル×チャネルごとに $(H,W)$、GN＝サンプル×グループごとに $(C/G,H,W)$。
- GN は $G=1$ で LN、$G=C$ で IN。
- BN は**学習時にバッチ統計、推論時に移動平均**。$\gamma,\beta$ は学習するパラメータ。畳み込みでは $m=NHW$（空間も含める）。
- BN の利点：学習の高速化・大きな学習率・初期値への依存の低減・正則化効果。弱点：小バッチ、系列モデル。
- Transformer は LN、小バッチの検出は GN、スタイル変換は IN。
- 勾配クリッピングは**ノルム**を閾値以下にする（向きは保つ）。勾配爆発への対策で、勾配消失は防げない。
- 入力の標準化（学習データの平均・分散で、テストデータにも同じ値を使う）と、層の中の正規化は別の話。

---

## 参考

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift（Ioffe, Szegedy, 2015）](https://arxiv.org/abs/1502.03167)
- [Layer Normalization（Ba ほか, 2016）](https://arxiv.org/abs/1607.06450)
- [Instance Normalization: The Missing Ingredient for Fast Stylization（Ulyanov ほか, 2016）](https://arxiv.org/abs/1607.08022)
- [Group Normalization（Wu, He, 2018）](https://arxiv.org/abs/1803.08494)
- [How Does Batch Normalization Help Optimization?（Santurkar ほか, 2018）](https://arxiv.org/abs/1805.11604)
- [Root Mean Square Layer Normalization（Zhang, Sennrich, 2019）](https://arxiv.org/abs/1910.07467)
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization（Huang, Belongie, 2017）](https://arxiv.org/abs/1703.06868)
- [On the difficulty of training Recurrent Neural Networks（Pascanu ほか, 2012）](https://arxiv.org/abs/1211.5063)
