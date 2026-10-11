# パラメータの初期化

<p class="kw">キーワード：Xavier法/Glorot法、Kaiming法/He法、LeCunの初期化、対称性の破れ</p>

!!! abstract "要点"
    - 重みを**同じ値（0 を含む）**で初期化すると、同じ層のノードが同じ勾配を受けて同じままになる。**乱数で対称性を破る**。
    - 乱数のばらつき（分散）の大きさが、層を重ねたときの**出力と勾配の大きさ**を決める。大きすぎると飽和・爆発、小さすぎると消失する。
    - 各層の出力の分散を保つように、ノード数と活性化関数から分散を決める。**Xavier（sigmoid/tanh）：$2/(n_{\mathrm{in}}+n_{\mathrm{out}})$、He（ReLU系）：$2/n_{\mathrm{in}}$**。

---

## なぜ初期化が問題になるか {#why}

- **対称性の破れ**：全重みを同じ値にすると、同じ層のノードは入力も勾配も同一になり、更新後も同一のまま。何個ノードがあっても1個ぶんの働きしかしない。小さな**乱数（ガウス分布）**で初期化すると、「同じようで少しずつ違う」状態から始められる。
- **スケール**：初期値が大きすぎると、層を通るたびに出力が膨らんで tanh やシグモイドが**飽和**し（微分が 0）、ReLU では出力が発散する。小さすぎると出力と勾配が層ごとに縮んで**消失**する（[勾配消失](backprop.md#gradient-vanishing)）。
- **バイアス**は 0 で初期化してよい。対称性は重みの乱数で破れている。
- 初期化の目標は、**層を通っても出力の大きさが変わらず、逆伝播でも勾配の大きさが変わらない**こと。

---

## 分散の伝わり方 {#variance}

第 $l$ 層の1つのノードの総入力を $u=\sum_{i=1}^{n_{\mathrm{in}}}w_i z_i$ とする。重み $w_i$ は平均 0・分散 $\mathrm{Var}(w)$ で互いに独立、入力 $z_i$ とも独立と仮定すると

$$
\mathrm{Var}(u)=n_{\mathrm{in}}\,\mathrm{Var}(w)\,\mathbb{E}[z^2]
$$

- $\mathrm{Var}(w)$ が $1/n_{\mathrm{in}}$ より大きいと、層を通るたびに出力の分散は**指数的に増え**、小さいと**指数的に減る**。$L$ 層なら $\bigl(n_{\mathrm{in}}\mathrm{Var}(w)\bigr)^L$ 倍。
- 順伝播で分散を保つ条件：$n_{\mathrm{in}}\,\mathrm{Var}(w)=1$。
- 逆伝播のデルタ $\boldsymbol\delta^{(l-1)}=\bigl(\mathbf{W}^{\top}\boldsymbol\delta^{(l)}\bigr)\odot f'$ も同じ形で、保つ条件は $n_{\mathrm{out}}\,\mathrm{Var}(w)=1$。
- 2つの条件は一般に両立しない（$n_{\mathrm{in}}\ne n_{\mathrm{out}}$）。その**妥協**が Xavier。

??? note "導出：Xavier と He の分散"
    **Xavier（Glorot）**：活性化関数が原点付近で線形（$f(u)\approx u$、tanh など）とみなすと $\mathbb{E}[z^2]=\mathrm{Var}(u_{\mathrm{prev}})$。順伝播の条件 $\mathrm{Var}(w)=1/n_{\mathrm{in}}$ と逆伝播の条件 $\mathrm{Var}(w)=1/n_{\mathrm{out}}$ の調和平均をとると

    $$
    \mathrm{Var}(w)=\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}
    $$

    **He（Kaiming）**：ReLU は入力の負の半分を 0 にする。$u$ が 0 を中心に対称なら $\mathbb{E}[z^2]=\tfrac12\mathrm{Var}(u)$。分散を保つには $n_{\mathrm{in}}\,\mathrm{Var}(w)\cdot\tfrac12=1$ より

    $$
    \mathrm{Var}(w)=\frac{2}{n_{\mathrm{in}}}
    $$

    ReLU に対する Xavier は、毎層で分散が $1/2$ 倍になる。Leaky ReLU（負側の傾き $a$）なら $\mathbb{E}[z^2]=\tfrac{1+a^2}{2}\mathrm{Var}(u)$ なので $\mathrm{Var}(w)=\dfrac{2}{(1+a^2)\,n_{\mathrm{in}}}$。

---

## 手法 {#methods}

| 手法 | 正規分布 $\mathcal{N}(0,\sigma^2)$ の $\sigma^2$ | 一様分布 $U(-a,a)$ の $a$ | 向く活性化関数 |
|:--|:--|:--|:--|
| LeCun | $\dfrac{1}{n_{\mathrm{in}}}$ | $\sqrt{\dfrac{3}{n_{\mathrm{in}}}}$ | tanh、SELU |
| **Xavier / Glorot** | $\dfrac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$ | $\sqrt{\dfrac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}$ | シグモイド、tanh（原点付近が線形） |
| **He / Kaiming** | $\dfrac{2}{n_{\mathrm{in}}}$ | $\sqrt{\dfrac{6}{n_{\mathrm{in}}}}$ | ReLU 系 |

- $n_{\mathrm{in}}$：その層への入力ノード数（fan-in）、$n_{\mathrm{out}}$：出力ノード数（fan-out）。一様分布の分散は $a^2/3$ なので、正規分布と同じ分散にするには $a=\sqrt{3\sigma^2}$。
- 畳み込み層では $n_{\mathrm{in}}=C_{\mathrm{in}}\times k_h\times k_w$（入力チャネル数 × カーネルの面積）、$n_{\mathrm{out}}=C_{\mathrm{out}}\times k_h\times k_w$。
- 教科書などでは Xavier を簡略化して $\sigma=\sqrt{1/n_{\mathrm{in}}}$ と書く（実質 LeCun）。元の論文の式は $2/(n_{\mathrm{in}}+n_{\mathrm{out}})$。
- 同じノード数の層が続くなら $n_{\mathrm{in}}=n_{\mathrm{out}}$ で、Xavier は $1/n$、He は $2/n$。**He は Xavier の分散の 2 倍**（標準偏差で $\sqrt2$ 倍）。

---

## 動かしてみる {#try}

10 層の全結合ネットワークに乱数の入力を通し、**層ごとの出力の大きさ**（左：二乗平均平方根 RMS、縦軸は対数）と、**層 $l$ の出力の分布**（右）を見る。

<div class="ew" data-widget="init-variance"></div>

!!! tip "動かしてみる"
    - tanh で $\sigma=0.01$ にすると、RMS が層ごとに約 $0.01\sqrt{n}$ 倍に縮み、10 層目で約 $10^{-10}$ になります。分布は 0 に潰れます（勾配消失の状態）。
    - tanh で $\sigma=1$ にすると、出力は ±1 に張り付き（分布が両端に集まる）、微分が 0 に近い**飽和**になります。RMS は 1 近くで動かず、見かけでは大丈夫そうでも勾配が流れません。
    - tanh で Xavier にすると分布が釣鐘形に保たれます（tanh は線形ではないので、少しずつ縮みます）。
    - ReLU に切り替えて He にすると RMS が約 1 で保たれます。同じ ReLU で Xavier にすると毎層縮み、$\sigma=1$ では毎層 $\sqrt{n/2}$ 倍に膨らみます。
    - $n$ を大きくすると、固定の $\sigma$ では縮み方・膨らみ方が速くなります。Xavier や He は $n$ に合わせて $\sigma$ が変わるので影響されません。

---

## 実装での使い方

- PyTorch：`torch.nn.init.xavier_normal_` / `xavier_uniform_`、`kaiming_normal_` / `kaiming_uniform_`（`mode` で fan-in か fan-out、`nonlinearity` で活性化関数を指定）。全結合・畳み込みの既定値は Kaiming 一様分布の系統。
- `mode="fan_in"` は順伝播の分散を保ち、`mode="fan_out"` は逆伝播の勾配の分散を保つ。
- [バッチ正規化](normalization.md)などの正規化層を入れると、初期化への依存は小さくなる。事前学習済みの重みから始める場合（転移学習）は、層ごとに初期化し直さない。
- 全く違う方針として、固定のスケールを決めずに最初のミニバッチで出力が分散 1 になるよう調整する方法（LSUV）や、直交行列で初期化する方法がある。

---

## 試験の着眼点

- Xavier は**シグモイド・tanh**、He は **ReLU**（Leaky ReLU など）。取り違えない。
- 分散の式を言えるようにする：Xavier $\dfrac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}$、He $\dfrac{2}{n_{\mathrm{in}}}$、LeCun $\dfrac{1}{n_{\mathrm{in}}}$。標準偏差はその平方根。
- He の係数 2 の理由：**ReLU は入力の半分を 0 にする**ので、分散が半分になるのを補う。
- 全て同じ値で初期化すると**対称性が破れない**。バイアスは 0 でよい。
- 初期化の目的は、順伝播・逆伝播で各層の出力と勾配の**分散を保つ**こと（勾配消失・爆発の防止）。

---

## 参考

- [Understanding the difficulty of training deep feedforward neural networks（Glorot, Bengio, 2010）](https://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification（He ほか, 2015）](https://arxiv.org/abs/1502.01852)
- [torch.nn.init（PyTorch ドキュメント）](https://pytorch.org/docs/stable/nn.init.html)
- [Efficient BackProp（LeCun ほか, 1998）](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)：LeCun の初期化
- [All you need is a good init（Mishkin, Matas, 2016）](https://arxiv.org/abs/1511.06422)：LSUV
