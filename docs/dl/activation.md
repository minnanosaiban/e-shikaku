# 活性化関数

<p class="kw">キーワード：シグモイド関数、温度パラメータ、勾配消失、ReLU、Leaky ReLU、GELU、tanh、双曲線関数</p>

!!! abstract "要点"
    - 層の間に非線形な関数を挟まないと、何層重ねても**1つの線形変換**にしかならない。活性化関数が**非線形性**を入れる。
    - 逆伝播では層ごとに**導関数 $f'$** が掛かる。$f'$ の形が、勾配の流れやすさ（学習のしやすさ）を決める。
    - 流れは、シグモイド・tanh（飽和して勾配が消える）→ **ReLU**（勾配が保たれる）→ **GELU・Swish**（なめらかで dying ReLU が起きにくい）。

---

## 非線形性が必要な理由

$$
\mathbf{y}=\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2=(\mathbf{W}_2\mathbf{W}_1)\,\mathbf{x}+(\mathbf{W}_2\mathbf{b}_1+\mathbf{b}_2)
$$

- 右辺は $\mathbf{W}'\mathbf{x}+\mathbf{b}'$ の形で、**1層の線形変換と同じ**。層を増やしても表現力は増えない。
- 間に非線形な $f$ を挟んだ $\mathbf{W}_2 f(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2$ はこのようにまとめられない。

---

## 主な活性化関数

$$
\begin{aligned}
&\text{Step} && f(x)=\begin{cases}1&(x>0)\\0&(x\le0)\end{cases} && f'(x)=0\ \ (x\ne0) \\[4mm]
&\text{Identity} && f(x)=x && f'(x)=1 \\[4mm]
&\text{Sigmoid} && f(x)=\sigma(x)=\dfrac{1}{1+e^{-x}} && f'(x)=f(x)\bigl(1-f(x)\bigr) \\[4mm]
&\text{tanh} && f(x)=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}} && f'(x)=1-f(x)^2 \\[4mm]
&\text{ReLU} && f(x)=\max(0,\,x) && f'(x)=\begin{cases}1&(x>0)\\0&(x\le0)\end{cases} \\[4mm]
&\text{Leaky ReLU} && f(x)=\max(\alpha x,\,x) && f'(x)=\begin{cases}1&(x>0)\\\alpha&(x\le0)\end{cases} \\[4mm]
&\text{ELU} && f(x)=\begin{cases}x&(x\ge0)\\\alpha(e^{x}-1)&(x<0)\end{cases} && f'(x)=\begin{cases}1&(x\ge0)\\f(x)+\alpha&(x<0)\end{cases} \\[4mm]
&\text{GELU} && f(x)=x\,\Phi(x) && f'(x)=\Phi(x)+x\,\phi(x) \\[4mm]
&\text{Swish (SiLU)} && f(x)=x\,\sigma(x) && f'(x)=\sigma(x)\bigl(1+x(1-\sigma(x))\bigr) \\[4mm]
&\text{Mish} && f(x)=x\tanh\bigl(\ln(1+e^{x})\bigr) &&
\end{aligned}
$$

- $\Phi(x)$：標準正規分布の累積分布関数、$\phi(x)$：その確率密度関数。$\alpha$ は小さな正の定数。
- **Softmax**（出力層で使う）は、$f_i(\mathbf{x})=\dfrac{e^{x_i}}{\sum_j e^{x_j}}$。他の関数と違い、**入力ベクトル全体**から出力ベクトルを作る。そのため導関数はヤコビ行列になる。

$$
\frac{\partial f_i}{\partial x_j}=f_i(\mathbf{x})\bigl(\delta_{ij}-f_j(\mathbf{x})\bigr)\qquad(\delta_{ij}\text{：クロネッカーのデルタ})
$$

### 双曲線関数

$$
\sinh x=\frac{e^{x}-e^{-x}}{2},\qquad \cosh x=\frac{e^{x}+e^{-x}}{2},\qquad \tanh x=\frac{\sinh x}{\cosh x}=2\sigma(2x)-1
$$

- tanh はシグモイドを拡大・平行移動したもの。値域が $(-1,1)$ で、**原点を通る（ゼロ中心）**。

### 性質のまとめ

| 関数 | 値域 | $f'$ の最大値 | 特徴 |
|:--|:--|:--|:--|
| Sigmoid | $(0,1)$ | $0.25$（$x=0$） | 出力が確率として扱える。ゼロ中心でない。飽和して勾配が消える |
| tanh | $(-1,1)$ | $1$（$x=0$） | ゼロ中心。飽和して勾配が消える（シグモイドよりまし） |
| ReLU | $[0,\infty)$ | $1$ | 計算が軽く、$x>0$ で勾配が減衰しない。$x\le0$ で勾配が 0（dying ReLU） |
| Leaky ReLU | $(-\infty,\infty)$ | $1$ | $x\le0$ にも傾き $\alpha$ が残り、dying ReLU が起きにくい |
| ELU | $(-\alpha,\infty)$ | $1$ | 負側が $-\alpha$ へなめらかに飽和する |
| GELU | 約 $(-0.17,\infty)$ | 1 をわずかに超える | なめらか。**Transformer**（BERT・GPT など）で標準的 |
| Swish (SiLU) | 約 $(-0.28,\infty)$ | 1 をわずかに超える | $x\,\sigma(x)$ の自己ゲート。**EfficientNet** などで使われる |

---

## 曲線を重ねて見る

関数を切り替え、$x$ を動かして、値と傾きを同時に確かめます。実線が $f(x)$、破線が $f'(x)$ です。

<div class="ew" data-widget="activation"></div>

!!! tip "動かしてみる"
    - シグモイドは、破線（$f'$）の山の高さが 0.25 しかありません。$x$ を $\pm4$ に近づけると、$f'$ は 0 に近づきます（飽和）。
    - tanh は山の高さが 1 で、原点を通ります。シグモイドと違って出力が負にもなります。
    - ReLU の $f'$ は 0 か 1 の2値です。Leaky ReLU に切り替えて $\alpha$ を動かすと、$x<0$ 側に傾き $\alpha$ が残ります（ELU でも $\alpha$ が負側の形を決めます）。
    - GELU と Swish は、$x<0$ で $f$ がいったん負に下がってから 0 へ戻ります（単調ではない）。

---

## 温度パラメータ

シグモイドとソフトマックスは、入力を**温度 $T$** で割ると、出力の鋭さを変えられる。

$$
\sigma_T(x)=\frac{1}{1+e^{-x/T}},\qquad
\mathrm{softmax}_T(\mathbf{z})_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}
$$

- $T>1$：出力がなだらかになり、確率が均される。$T<1$：最大の要素に確率が集中する。
- $T\to0$：最大の要素だけが 1 の **one-hot**（argmax）に近づく。$T\to\infty$：すべて $1/K$ の**一様分布**に近づく。
- $T=1$ が通常のソフトマックス。蒸留（教師モデルの出力をなだらかにする）や、言語モデルの生成（多様さの調整）で使う。

<div class="ew" data-widget="temperature"></div>

!!! tip "動かしてみる"
    - $T$ を 0.1 まで下げると、$z_1$ のクラスがほぼ 1 になります。5 まで上げると、3つとも $1/3$ に近づきます。
    - $z_1,z_2,z_3$ をすべて同じ値にすると、$T$ に関係なく $1/3$ ずつになります。
    - 3つの $z$ を同じだけずらしても、確率は変わりません（ソフトマックスは定数の加算に対して不変）。実装では最大値を引いてから指数をとり、オーバーフローを防ぎます。

---

## 勾配消失と dying ReLU

- 逆伝播では $f'$ が層の数だけ掛かる。シグモイドは $f'\le0.25$ なので、$L$ 層で最大でも $0.25^{L}$ になり、勾配が急速に消える（[誤差逆伝播法](backprop.md#gradient-vanishing) の動く図で確認できる）。
- tanh は $x=0$ 付近では $f'\approx1$ だが、$|x|$ が大きいと**飽和**して $f'\to0$ になる。
- ReLU は $x>0$ で $f'=1$ のため減衰しない。一方で $x\le0$ に入り続けたユニットは勾配が 0 のまま更新されず、出力が 0 に固定される（**dying ReLU**）。
- Leaky ReLU・ELU・GELU・Swish は $x\le0$ 側にも小さな勾配が残るため、dying ReLU が起きにくい。

---

## 試験の着眼点

- シグモイドの微分は $f(1-f)$ で**最大 0.25**（$x=0$）。tanh の微分は $1-f^2$ で**最大 1**（$x=0$）。どちらも**出力だけから**計算できる。
- $\tanh x=2\sigma(2x)-1$。tanh はゼロ中心で、シグモイドはそうではない。
- ReLU は $x=0$ での微分を 0 とする実装が多い。dying ReLU は Leaky ReLU などで緩和する。
- GELU は $x\,\Phi(x)$（Transformer）、Swish は $x\,\sigma(x)$（EfficientNet）。
- ソフトマックスのヤコビ行列は $f_i(\delta_{ij}-f_j)$。交差エントロピーと組み合わせると、出力層のデルタは $\mathbf{y}-\mathbf{t}$ になる。
- 温度が**高い**ほど確率がなだらか（一様に近い）、**低い**ほど鋭い（one-hot に近い）。
