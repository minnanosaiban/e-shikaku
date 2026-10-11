# 誤差逆伝播法

<p class="kw">キーワード：誤差逆伝播法、連鎖律、偏微分によるデルタ、勾配消失、自動微分、計算グラフ</p>

!!! abstract "要点"
    - 損失 $L$ のすべてのパラメータについての勾配を、**出力側から入力側へ、連鎖律で一度に**求める方法。
    - 計算グラフの各ノードがすることは、**「上流から来た勾配 × 自分の局所勾配」を下流へ渡す**ことだけ。
    - 層を重ねると局所勾配の**積**になるため、1 より小さい値が重なると勾配が消える（勾配消失）。

---

## 連鎖律

$$
\begin{aligned}
&\text{1変数} && L=g(y),\ y=f(x) && \frac{dL}{dx}=\frac{dL}{dy}\cdot\frac{dy}{dx} \\[2mm]
&\text{多変数} && L=g(z),\ z=f(x,y) && \frac{\partial L}{\partial x}=\frac{dL}{dz}\cdot\frac{\partial z}{\partial x},\quad \frac{\partial L}{\partial y}=\frac{dL}{dz}\cdot\frac{\partial z}{\partial y} \\[2mm]
&\text{多層の合成} && L=h(w),\ w=g(z),\ z=f(x) && \frac{dL}{dx}=\frac{dL}{dw}\cdot\frac{dw}{dz}\cdot\frac{dz}{dx} \\[2mm]
&\text{ベクトル} && L=g(\mathbf{z}),\ \mathbf{z}=f(\mathbf{x}) && \frac{\partial L}{\partial \mathbf{x}}=\left(\frac{\partial \mathbf{z}}{\partial \mathbf{x}}\right)^{\!\top}\frac{\partial L}{\partial \mathbf{z}}
\end{aligned}
$$

---

## 計算グラフ

上流から来た勾配を $\dfrac{\partial L}{\partial z}$ とすると、各ノードは次の値を下流へ渡します。

| ノード | 順伝播 | 逆伝播で渡す勾配 | 覚え方 |
|:--|:--|:--|:--|
| 加算 | $z=x+y$ | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial z},\ \dfrac{\partial L}{\partial y}=\dfrac{\partial L}{\partial z}$ | そのまま分配 |
| 乗算 | $z=xy$ | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial z}\,y,\ \dfrac{\partial L}{\partial y}=\dfrac{\partial L}{\partial z}\,x$ | 相手の値を掛ける |
| シグモイド | $z=\sigma(x)$ | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial z}\,z(1-z)$ | 出力だけで計算できる |
| ReLU | $z=\max(0,x)$ | $\dfrac{\partial L}{\partial x}=\dfrac{\partial L}{\partial z}\,\mathbb{1}[x>0]$ | 通すか止めるか |

<div class="ew" data-widget="gate"></div>

!!! tip "動かしてみる"
    - 数式の行にマウスを乗せると、図の対応する辺が光ります。
    - 乗算で $x$ を動かすと、変わるのは **$\partial L/\partial y$ のほう**です（相手の値を掛けるため）。
    - シグモイドで $x$ を大きくすると、$z(1-z)$ が 0 に近づき、勾配がほとんど流れなくなります。

### Affine（全結合層）

$$
\mathbf{Y}=\mathbf{X}\mathbf{W}+\mathbf{b},\qquad
\frac{\partial L}{\partial \mathbf{X}}=\frac{\partial L}{\partial \mathbf{Y}}\,\mathbf{W}^{\top},\qquad
\frac{\partial L}{\partial \mathbf{W}}=\mathbf{X}^{\top}\frac{\partial L}{\partial \mathbf{Y}},\qquad
\frac{\partial L}{\partial \mathbf{b}}=\sum_{n=1}^{N}\left(\frac{\partial L}{\partial \mathbf{Y}}\right)_{n,:}
$$

| 記号 | 形 | 勾配の形の検算 |
|:--|:--|:--|
| $\mathbf{X}$ | $N\times D$ | $\dfrac{\partial L}{\partial \mathbf{Y}}\mathbf{W}^{\top}$：$(N\times M)(M\times D)=N\times D$ |
| $\mathbf{W}$ | $D\times M$ | $\mathbf{X}^{\top}\dfrac{\partial L}{\partial \mathbf{Y}}$：$(D\times N)(N\times M)=D\times M$ |
| $\mathbf{b}$ | $M$ | バッチ方向（$N$）に和をとって $M$ |
| $\mathbf{Y}$ | $N\times M$ | — |

- 勾配は**元の変数と同じ形**になる。転置をどちらに付けるか迷ったら、形が合うほうを選べばよい。
- バイアスは全サンプルに同じ値が足されている（ブロードキャスト）ので、逆伝播では**サンプル方向に和**をとる。

---

## 1つのニューロンで追う

$z=w_1x_1+w_2x_2+b,\ a=\sigma(z),\ L=\frac12(a-t)^2$ の順伝播と逆伝播です。

<div class="ew" data-widget="neuron"></div>

!!! tip "動かしてみる"
    - 「逆伝播」に切り替えて、上から順に行にマウスを乗せると、勾配が出力側から入力側へ流れる順番が分かります。
    - $t$ を $a$ に近づけると、$\partial L/\partial a=a-t$ が 0 になり、**すべての勾配が 0** になります（誤差がなければ学習は進まない）。
    - $x_1=0$ にすると、$\partial L/\partial w_1$ だけが 0 になります（入力が 0 の重みは更新されない）。

---

## 偏微分によるデルタ

第 $l$ 層を $\mathbf{u}^{(l)}=\mathbf{W}^{(l)}\mathbf{z}^{(l-1)}+\mathbf{b}^{(l)},\ \mathbf{z}^{(l)}=f(\mathbf{u}^{(l)})$ とし、**デルタ**を $\boldsymbol{\delta}^{(l)}\equiv\dfrac{\partial L}{\partial \mathbf{u}^{(l)}}$ と定義します。

$$
\begin{aligned}
&\text{出力層} && \boldsymbol{\delta}^{(\mathrm{out})}=\frac{\partial L}{\partial \mathbf{z}^{(\mathrm{out})}}\odot f'\bigl(\mathbf{u}^{(\mathrm{out})}\bigr)
\qquad \text{（ソフトマックス＋交差エントロピーなら } \boldsymbol{\delta}=\mathbf{y}-\mathbf{t}\text{）} \\[2mm]
&\text{隠れ層} && \boldsymbol{\delta}^{(l)}=\Bigl(\mathbf{W}^{(l+1)\top}\boldsymbol{\delta}^{(l+1)}\Bigr)\odot f'\bigl(\mathbf{u}^{(l)}\bigr) \\[2mm]
&\text{パラメータ} && \frac{\partial L}{\partial \mathbf{W}^{(l)}}=\boldsymbol{\delta}^{(l)}\,\mathbf{z}^{(l-1)\top},\qquad \frac{\partial L}{\partial \mathbf{b}^{(l)}}=\boldsymbol{\delta}^{(l)}
\end{aligned}
$$

- デルタさえ出力側から順に求めれば、各層の重みの勾配は「デルタ × 1つ前の層の出力」で決まる。
- 隠れ層の式に $f'$ が層ごとに1回ずつ掛かる。これが次の勾配消失の原因になる。

---

## 勾配消失

デルタを入力側へ伝えるたびに $f'(u)$ が掛かるため、$n$ 層ぶん伝えると $f'$ の $n$ 個の積になります。

| 活性化関数 | 微分 | 微分の最大値 |
|:--|:--|:--|
| シグモイド | $\sigma(u)\bigl(1-\sigma(u)\bigr)$ | $0.25$（$u=0$） |
| tanh | $1-\tanh^2(u)$ | $1$（$u=0$） |
| ReLU | $\mathbb{1}[u>0]$ | $1$（$u>0$ なら常に） |

<div class="ew" data-widget="vanishing"></div>

!!! tip "動かしてみる"
    - シグモイドは $u=0$ でも最大 $0.25$ なので、層を増やすと**どんな $u$ でも**急速に小さくなります（10層で約 $10^{-6}$）。
    - tanh は $u=0$ 付近なら保たれますが、$|u|$ が大きくなると（飽和すると）急に消えます。
    - ReLU は $u>0$ なら何層でも 1 のまま。ただし $u\le 0$ では 0 になります（dying ReLU）。

---

## 自動微分

- 計算をノードの組み合わせ（計算グラフ）として記録し、各ノードの局所勾配を連鎖律でつないで微分を求める方法。
- **リバースモード**（出力側から逆向き）：出力1つ・入力が多数のとき、1回の逆向きの計算ですべての入力についての勾配が求まる。**誤差逆伝播法はこれにあたる**（損失はスカラー、パラメータは多数）。
- **フォワードモード**（入力側から順向き）：入力1つあたり1回の計算が必要。入力が少なく出力が多いときに向く。
- 数値微分 $\dfrac{f(x+h)-f(x-h)}{2h}$ は、パラメータの数だけ計算が必要で遅いため、**勾配の検算（勾配チェック）**に使う。

---

## 試験の着眼点

- 乗算ノードは、**入力を入れ替えて**上流の勾配に掛ける。
- Affine の勾配は**形で検算**できる。$\partial L/\partial \mathbf{W}$ は $\mathbf{W}$ と同じ形。
- バイアスの勾配は、**バッチ方向に和**をとる。
- ソフトマックス＋交差エントロピーの出力層のデルタは $\mathbf{y}-\mathbf{t}$。
- シグモイドの微分の最大値は $0.25$。これが深いネットワークで勾配消失を起こす理由。
