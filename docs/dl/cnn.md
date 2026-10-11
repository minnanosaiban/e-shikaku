# 畳み込みニューラルネットワーク（CNN）

<p class="kw">キーワード：基本的な畳み込み演算、単純型細胞と複雑型細胞、受容野、特徴マップ、フィルタ、カーネル、パディング、ストライド、im2col、チャネル、特別な畳み込み、point-wise 畳み込み（1×1 畳み込み）、depth-wise 畳み込み、グループ化畳み込み、拡張（dilated）畳み込み、変形可能畳み込み、アップサンプリングと逆畳み込み（転置畳み込み）、プーリング、Max pooling、Lp pooling、Global Average Pooling、LeNet、AlexNet、VGG、GoogLeNet、ResNet、DenseNet、MobileNet、EfficientNet</p>

!!! abstract "要点"
    - 小さなフィルタを**画面全体で使い回す**（重み共有）ことで、全結合に比べてパラメータを桁違いに減らし、**平行移動に対して等変**な特徴抽出を実現する。
    - 出力サイズは $\lfloor (H+2P-F)/S\rfloor+1$、パラメータ数は $F^2C_{in}C_{out}+C_{out}$ が基本。**特別な畳み込みは、この2つの式のどこを削るか**で整理できる。
    - 代表的アーキテクチャの歴史は、**深くする（VGG・ResNet）→ 効率化する（Inception・MobileNet）→ 釣り合いよく大きくする（EfficientNet）** という流れ。

---

## CNN の全体像 {#overview}

| 順 | 層 | 役割 |
|:--|:--|:--|
| 1 | 畳み込み | フィルタを当てて局所パターン（エッジ・模様）を検出し、特徴マップを作る |
| 2 | 活性化（ReLU） | 非線形性を入れる（[活性化関数](activation.md)） |
| 3 | プーリング | 局所領域をまとめて縮小し、位置ずれに鈍感にする |
| 4 | 1〜3 を繰り返す | 浅い層は線やエッジ、深い層は部品や物体と、**階層的に**特徴を作る |
| 5 | Flatten → 全結合 → 出力層 | ベクトルに並べ直して分類・回帰する（[出力層と損失関数](mlp-loss.md)） |

- テンソルの形は $N\times C\times H\times W$（バッチ・チャネル・高さ・幅）で扱う。
- **単純型細胞と複雑型細胞**（Hubel と Wiesel の視覚野の研究）が着想の元。畳み込み層が「特定の向きの線に反応する」単純型細胞、プーリング層が「位置が少しずれても同じ反応を返す」複雑型細胞にあたる。ネオコグニトロンがこれを工学的に実現し、CNN につながった。
- 学習は通常のネットワークと同じ（[誤差逆伝播法](backprop.md)）。ただし重みが共有されるので、勾配は**使われたすべての位置の合計**になる。

---

## 畳み込み演算 {#convolution}

$$
\begin{aligned}
&\text{1チャネル} && Y_{i,j}=\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}K_{m,n}\,X_{Si+m,\;Sj+n}+b \\[2mm]
&\text{複数チャネル} && Y_{c',i,j}=\sum_{c=0}^{C_{in}-1}\sum_{m=0}^{F-1}\sum_{n=0}^{F-1}W_{c',c,m,n}\,X_{c,\;Si+m,\;Sj+n}+b_{c'} \\[2mm]
&\text{出力サイズ} && O=\left\lfloor\dfrac{H+2P-F}{S}\right\rfloor+1 \qquad (\text{幅も同様})
\end{aligned}
$$

| 用語 | 意味 |
|:--|:--|
| フィルタ（カーネル）$K$ | 窓の中身になる重み。大きさ $F\times F$ |
| 特徴マップ | フィルタ1枚を当てて得た出力。出力チャネル数 $C_{out}$ ＝フィルタの枚数 |
| ストライド $S$ | 窓をずらす幅。大きくすると出力が縮む |
| パディング $P$ | 入力の周囲に足す幅（ふつう 0 を足す）。端の情報を残し、サイズを保つ |
| チャネル | 入力の奥行き（RGB なら 3）。フィルタは**全チャネルをまたいで** $F\times F\times C_{in}$ の大きさを持つ |

- 深層学習の「畳み込み」は、カーネルを裏返さない**相互相関**である（数学の畳み込みとは反転の有無だけ違う。学習で決まる重みなので結果に差はない）。
- 出力サイズの $\lfloor\cdot\rfloor$（切り捨て）は、割り切れないときに端の余りを捨てるためのもの。
- **same パディング**：$S=1$ で $O=H$ にするには $P=(F-1)/2$（$F$ は奇数）。**valid**：$P=0$。
- 1つの出力に使われる入力は $F\times F\times C_{in}$ 個だけ（**疎な結合**）。同じ重みを全位置で使う（**重み共有**）。入力を平行移動すると出力も同じだけ平行移動する（**平行移動等変性**）。拡大・回転に対しては等変でない。

<div class="ew" data-widget="cnn-conv"></div>

!!! tip "動かしてみる"
    - $F$ や $S$ を変えると、出力の大きさ $O$ が式どおりに変わります。$S=2$ にすると、窓が 2 マスずつ飛んで出力がほぼ半分になります。
    - $H=7,\ F=3$ で $P=0$ にすると出力は 5×5 に縮み、$P=1$ にすると 7×7 に戻ります（same パディング）。
    - $H=7,\ F=3,\ P=0$ で $S=3$ にすると、$(7-3)/3$ が割り切れず、最後の1列が使われずに捨てられます（切り捨ての意味）。
    - 全結合との比較の行を見てください。同じ入出力の大きさを全結合でつなぐと、畳み込みの数百倍のパラメータが必要です。

### パラメータ数・計算量・受容野

$$
\begin{aligned}
&\text{パラメータ数} && F^2C_{in}C_{out}+C_{out} \qquad (\text{入力サイズに依存しない}) \\[2mm]
&\text{積和の回数} && F^2C_{in}C_{out}\,O_HO_W \\[2mm]
&\text{受容野} && r_l=r_{l-1}+(F_l-1)\,j_{l-1},\qquad j_l=j_{l-1}S_l\qquad (r_0=1,\ j_0=1)
\end{aligned}
$$

- 例：$32\times32\times3$ の画像に 3×3 のフィルタ 32 枚（same）→ パラメータ $3^2\cdot3\cdot32+32=896$。同じ入出力を全結合でつなぐと $3072\times32768\approx1.0\times10^8$。
- **受容野**：出力の1マスが見ている入力の範囲。$S=1$ の 3×3 を重ねると 3→5→7 と広がる。
- 7×7 を1層使う代わりに 3×3 を3層重ねても受容野は同じ 7×7 で、パラメータは $49C^2\to27C^2$ に減り、非線形性も増える（**VGG の発想**）。

### im2col による実装 {#im2col}

- 入力の各窓を1行に並べ直し（**im2col**）、畳み込みを**1回の行列積**にする。GPU に向く。

| 配列 | 形 |
|:--|:--|
| 入力 $\mathbf{X}$ | $(N,\,C_{in},\,H,\,W)$ |
| im2col 後 $\mathbf{X}_{col}$ | $(N\,O_HO_W,\ C_{in}F^2)$ |
| 重み（整形後）$\mathbf{W}_{mat}$ | $(C_{in}F^2,\ C_{out})$ |
| 出力 $\mathbf{Y}_{mat}=\mathbf{X}_{col}\mathbf{W}_{mat}+\mathbf{b}$ | $(N\,O_HO_W,\ C_{out})$ → reshape して $(N,\,C_{out},\,O_H,\,O_W)$ |

$$
\frac{\partial L}{\partial \mathbf{W}_{mat}}=\mathbf{X}_{col}^{\top}\frac{\partial L}{\partial \mathbf{Y}_{mat}},\qquad
\frac{\partial L}{\partial \mathbf{b}}=\sum_{\text{行}}\frac{\partial L}{\partial \mathbf{Y}_{mat}},\qquad
\frac{\partial L}{\partial \mathbf{X}_{col}}=\frac{\partial L}{\partial \mathbf{Y}_{mat}}\mathbf{W}_{mat}^{\top}\xrightarrow{\ \text{col2im}\ }\frac{\partial L}{\partial \mathbf{X}}
$$

- 全結合（Affine）の逆伝播と同じ形（[誤差逆伝播法](backprop.md)）。違いは最後の **col2im** で、窓が重なった入力のマスには**勾配を足し合わせる**点。
- im2col は入力を複数回コピーするためメモリを多く使う。

---

## 特別な畳み込み {#special-conv}

標準の畳み込みは「空間方向（$F\times F$）」と「チャネル方向（$C_{in}\to C_{out}$）」を**一度に**混ぜる。特別な畳み込みは、これを分けたり、間引いたり、広げたりする。

| 種類 | 何を変えるか | パラメータ数（バイアス除く） | 代表例 |
|:--|:--|:--|:--|
| 標準 | 空間とチャネルを同時に混ぜる | $F^2C_{in}C_{out}$ | LeNet、VGG、ResNet |
| point-wise（1×1） | チャネル方向だけ混ぜる | $C_{in}C_{out}$ | Inception、ResNet のボトルネック、MobileNet |
| depth-wise | 空間方向だけ。チャネルごとに独立 | $F^2C_{in}$ | MobileNet、Xception |
| グループ化 | チャネルを $G$ 組に分け、組の中だけで混ぜる | $F^2C_{in}C_{out}/G$ | AlexNet、ResNeXt |
| 拡張（dilated） | 窓の要素の間を空ける | 標準と同じ | DeepLab、WaveNet |
| 変形可能 | 窓の位置を学習で動かす | 標準＋オフセット用の畳み込み | DCN |
| 転置（逆畳み込み） | 解像度を**拡大**する | 標準と同じ | U-Net、FCN、GAN の生成器 |

### point-wise と depth-wise

$$
\begin{aligned}
&\text{point-wise (1×1)} && Y_{c',i,j}=\sum_{c=0}^{C_{in}-1}W_{c',c}\,X_{c,i,j} \\[2mm]
&\text{depth-wise} && Y_{c,i,j}=\sum_{m,n}K_{c,m,n}\,X_{c,\;i+m,\;j+n}
\end{aligned}
$$

- 1×1 畳み込みは、各画素のチャネルベクトルに**同じ全結合**をかける演算。チャネル数の増減（次元削減）、チャネル間の情報の統合、直後の活性化による非線形性の追加ができる（Network in Network）。
- depth-wise 畳み込みは、チャネルを混ぜずに各チャネルを別々のフィルタで空間畳み込みする。

### depth-wise separable 畳み込み {#dw-separable}

depth-wise のあとに point-wise をつなぐ。空間とチャネルの混ぜ方を**分解**する。

$$
\begin{aligned}
\text{標準} &\quad F^2C_{in}C_{out}\,HW \\[1mm]
\text{depth-wise}+\text{point-wise} &\quad F^2C_{in}\,HW+C_{in}C_{out}\,HW \\[1mm]
\text{比（分離可能／標準）} &\quad \frac{F^2C_{in}+C_{in}C_{out}}{F^2C_{in}C_{out}}=\frac{1}{C_{out}}+\frac{1}{F^2}
\end{aligned}
$$

- 例（$F=3,\ C_{in}=C_{out}=64$）：標準 $36{,}864$、分離可能 $576+4096=4{,}672$。比は $1/64+1/9\approx0.127$ で、約 **1/8** になる。
- $C_{out}$ が大きいので、比はほぼ $1/F^2$ で決まる。3×3 なら約 1/9 に近づく。

### グループ化畳み込み

- 入力チャネルを $G$ 組、出力チャネルも $G$ 組に分け、同じ番号の組同士だけをつなぐ。$G$ は $C_{in}$ と $C_{out}$ の約数。
- $G=1$ が標準、$G=C_{in}=C_{out}$ が depth-wise。パラメータと計算は $1/G$。
- AlexNet は2枚の GPU にチャネルを分けたことが起源。ResNeXt は組の数（cardinality）を増やして精度を上げた。

### 拡張（dilated）畳み込み {#dilated}

$$
Y_{i,j}=\sum_{m,n}K_{m,n}\,X_{i+rm,\;j+rn},\qquad F_{\mathrm{eff}}=r(F-1)+1,\qquad O=\left\lfloor\frac{H+2P-F_{\mathrm{eff}}}{S}\right\rfloor+1
$$

- $r$ は**拡張率**（$r=1$ が標準）。パラメータを増やさずに受容野を広げる。プーリングで解像度を落とさずに広い文脈を見られるので、セグメンテーションで使われる。
- **WaveNet**：因果畳み込み（未来を見ない）を、$r=1,2,4,\dots,512$ と倍々にして重ねる。カーネル幅 2 で受容野は $1+\sum(F-1)r=1+(1+2+\cdots+512)=1024$ サンプルに達する。RNN と違って時間方向に並列に学習できる。
- 格子状に抜けるため、$r$ を同じ値で重ねすぎると（gridding）見ていない画素ができる。

### 変形可能（deformable）畳み込み

$$
Y_{i,j}=\sum_{m,n}K_{m,n}\;X\bigl(i+m+\Delta y_{m,n},\;j+n+\Delta x_{m,n}\bigr)
$$

- 窓の各点の**オフセット** $(\Delta y,\Delta x)$ を、別の畳み込みで入力から予測する。整数でないので**双線形補間**で値を読む。
- 物体の形やスケールの変化に合わせて、サンプリング位置が動く。

### 転置畳み込み（逆畳み込み） {#transposed}

- 畳み込みを行列 $\mathbf{y}=\mathbf{C}\mathbf{x}$ と書いたとき、転置畳み込みは $\mathbf{x}'=\mathbf{C}^{\top}\mathbf{y}$。**畳み込みの逆伝播（入力側の勾配）と同じ演算**で、サイズを拡大する。
- 実装上は、入力の要素の間に $S-1$ 個の 0 を挟んでから、普通の畳み込みをかけるのと同じ。

$$
H'=S\,(H-1)+F-2P\ (+\ \text{output padding})
$$

- 例：$H=4,\ F=3,\ S=2,\ P=1$ → $H'=2\cdot3+3-2=7$。標準の畳み込みで 7 と 8 はどちらも出力 4 になるので、output padding でどちらかを選ぶ。
- 元の値を**復元するわけではない**（逆行列ではなく転置）。「デコンボリューション」は誤解を招く呼び名。
- $F$ が $S$ で割り切れないと重なりが不均一になり、**チェッカーボード状の模様**が出る。最近傍や双線形の**アップサンプリング＋通常の畳み込み**で避けることも多い。
- 用途：セグメンテーションのデコーダ、オートエンコーダの再構成、GAN の生成器。

<div class="ew" data-widget="cnn-params"></div>

!!! tip "動かしてみる"
    - 「DW＋PW」の棒は、$C_{out}$ を大きくすると $1/F^2$ に近づきます。$K=1$ にすると、分離しても得がなく、むしろ増えます（空間方向の混ぜがないため）。
    - $G$ を大きくすると「グループ化」の棒が $1/G$ に縮みます。$C_{in}$ と $C_{out}$ の公約数にならない $G$ は、自動で調整されます。
    - $r$ を変えても、標準のパラメータ数は変わりません。変わるのは実効カーネル $F_{\mathrm{eff}}$（受容野）だけです。

---

## プーリング {#pooling}

窓ごとに1つの値にまとめて**縮小**する。学習するパラメータはない。出力サイズは畳み込みと同じ式。チャネルごとに独立に行う。

| 種類 | 式（窓 $h\times w$、ストライド $s$） | 特徴 |
|:--|:--|:--|
| Max | $\max_{m,n}X_{si+m,\,sj+n}$ | 強い反応を残す。位置ずれに強い。最も一般的 |
| Average | $\dfrac{1}{hw}\sum_{m,n}X_{si+m,\,sj+n}$ | 滑らか。背景を含む全体の傾向 |
| Lp | $\left(\dfrac{1}{hw}\sum_{m,n}\lvert X_{si+m,\,sj+n}\rvert^{p}\right)^{1/p}$ | $p=1$ で平均（入力が非負のとき）、$p\to\infty$ で最大 |
| Global Average（GAP） | $\dfrac{1}{HW}\sum_{x,y}X_{c,x,y}$ | チャネルごとに1つの値へ。全結合層の代わり |
| Global Max | $\max_{x,y}X_{c,x,y}$ | 最も強い反応だけ。位置情報を捨てる |
| Spatial Pyramid（SPP） | $n_k\times n_k$ 分割のプーリングを連結 | 入力サイズが違っても**固定長**のベクトルになる |

$$
\begin{bmatrix}
1 & 3 & 2 & 1 \\
2 & 9 & 1 & 1 \\
1 & 3 & 2 & 3 \\
5 & 6 & 1 & 2
\end{bmatrix}
\xrightarrow{\ \max\ }
\begin{bmatrix}
9 & 2 \\
6 & 3
\end{bmatrix}
\qquad
\xrightarrow{\ \text{average}\ }
\begin{bmatrix}
3.75 & 1.25 \\
3.75 & 2
\end{bmatrix}
$$

- 2×2・ストライド 2。左上の窓は $\{1,3,2,9\}$ なので、最大は 9、平均は $15/4=3.75$。
- **逆伝播**：Max は**最大だった位置にだけ**勾配を返し、他は 0。Average は勾配を窓の全マスに $1/(hw)$ ずつ配る。
- 効果：計算量とメモリの削減、**小さな位置ずれへの不変性**、過学習の抑制。
- **GAP**：最終の特徴マップ（$C\times H\times W$）をチャネルごとに平均して $C$ 次元にし、直後の全結合・softmax につなぐ。全結合層のパラメータ（巨大になりがち）が不要になり、入力サイズにも依存しない。各チャネルが1クラスに対応する設計にすると、クラス活性化マップ（CAM）で判断根拠を可視化できる。
- **SPP**：例えば $1\times1,\ 2\times2,\ 4\times4$ の3段なら $C\,(1+4+16)=21C$ 次元に固定される。物体検出で任意サイズの領域を扱うのに使われた。
- 近年はプーリングの代わりに**ストライド 2 の畳み込み**で縮小することも多い。

---

## 代表的なアーキテクチャ {#architectures}

| モデル（年） | 要点 |
|:--|:--|
| LeNet-5（1998） | 畳み込み・サブサンプリング・全結合の原型。32×32 の手書き数字を入力とする |
| AlexNet（2012） | ILSVRC 2012 で優勝。畳み込み5層＋全結合3層、約 6,000 万パラメータ。**ReLU**、局所応答正規化（LRN）、重なりありのプーリング、**ドロップアウト**、データ拡張、2GPU |
| ZFNet（2013） | 特徴の可視化から、第1層を 11×11・ストライド 4 から **7×7・ストライド 2** に変更（エイリアシングの解消） |
| VGG（2014） | **3×3 の畳み込みだけ**を重ねて 16〜19 層に。プーリングで半分にするたびにチャネルを2倍。約 1.4 億パラメータ（多くは全結合層） |
| GoogLeNet（2014） | **Inception モジュール**。補助分類器、GAP で全結合を削減 |
| ResNet（2015） | **残差接続**で100層以上を学習可能にした |
| Wide ResNet（2016） | 深さより**幅**を増やす |
| DenseNet（2016） | 全層を**連結**で結ぶ |
| MobileNet v1〜v3（2017〜19） | depth-wise separable、逆残差、NAS による軽量・高速化 |
| EfficientNet（2019） | 深さ・幅・解像度を**同時に**一定比で拡大（複合スケーリング） |

### Inception（GoogLeNet）

- 1×1、3×3、5×5 の畳み込みと 3×3 の max pooling を**並列**に行い、チャネル方向に連結する。どの大きさの模様が有効かを、ネットワークに選ばせる。
- 3×3・5×5 の前に **1×1 でチャネルを減らす**（ボトルネック）ことで、計算量を抑える。
- **補助分類器**：中間層から出す分類器。学習時だけ損失に重み付きで加え、中間層へ直接誤差を流して勾配消失を防ぎ、正則化にもなる。推論では使わない。
- **Inception v3**：5×5 を 3×3 の2層に分解（25→18 でパラメータ約 28% 減）、$n\times n$ を $1\times n$ と $n\times1$ に分解（7×7 なら 49→14）。さらに Batch Normalization と**ラベルスムージング**（正解ラベルを $(1-\varepsilon)\mathbf{t}+\varepsilon/K$ に和らげる）を導入した。

### ResNet（残差ネットワーク） {#resnet}

$$
\mathbf{y}=F(\mathbf{x};\{W_i\})+\mathbf{x},\qquad \frac{\partial \mathbf{y}}{\partial \mathbf{x}}=\frac{\partial F}{\partial \mathbf{x}}+\mathbf{I}
$$

- **劣化問題**：素朴に層を深くすると、**訓練誤差まで**悪化する（過学習ではない）。
- ブロックは「入力そのもの」を足す**スキップ接続**を持ち、$F$ は残差 $H(\mathbf{x})-\mathbf{x}$ を学ぶ。何も変換したくなければ $F\to0$ にすればよく、恒等写像を学ぶより簡単。
- 逆伝播でも $+\mathbf{I}$ の項が必ず残るので、勾配が消えにくい（[勾配消失](backprop.md#gradient-vanishing)）。多数の短い経路の和とも見え、アンサンブルのような効果があるとされる。
- 入出力の次元が違うときは、1×1 畳み込みで合わせる（射影ショートカット）。
- **ボトルネックブロック**：1×1（256→64）→ 3×3（64）→ 1×1（64→256）。256 チャネルの入出力で約 $16{,}384+36{,}864+16{,}384=69{,}632$ で、通常ブロック（3×3 を2つ、64 チャネル）の $2\cdot9\cdot64^2=73{,}728$ とほぼ同じ計算量で、より深くできる。ResNet-50/101/152 で使う。

### Wide ResNet・DenseNet

- **Wide ResNet**（WRN-$n$-$k$）：$n$ は畳み込みの層数、$k$ は幅の倍率（チャネルを $k$ 倍）。浅くて幅が広いほうが、GPU で並列化しやすく速い。ブロックは $B(3,3)$ が良好。ブロック内の2つの畳み込みの間に**ドロップアウト**を入れる。
- **DenseNet**：ResNet が足し算なのに対し、**連結**する。

$$
\text{ResNet:}\ \ \mathbf{x}_{l}=H_l(\mathbf{x}_{l-1})+\mathbf{x}_{l-1},\qquad
\text{DenseNet:}\ \ \mathbf{x}_{l}=H_l\bigl([\mathbf{x}_0,\mathbf{x}_1,\dots,\mathbf{x}_{l-1}]\bigr)
$$

- 各層は**成長率** $k$ 枚の特徴マップだけを出す。$l$ 層目の入力チャネル数は $k_0+k(l-1)$。特徴を再利用するのでパラメータが少なくて済む。
- ブロックの間の**遷移層**（1×1 畳み込みで圧縮率 $\theta$ 倍に圧縮＋2×2 平均プーリング）で、チャネル数と大きさを減らす。

### MobileNet・EfficientNet

- **MobileNet v1**：[depth-wise separable](#dw-separable) を全面採用。**幅の倍率 $\alpha$**（チャネル数）と**解像度の倍率 $\rho$** で、精度と計算量のトレードオフを調整する。

$$
\text{計算量}=F^2\,\alpha C_{in}\,\rho^2H W+\alpha C_{in}\,\alpha C_{out}\,\rho^2H W
$$

- 2項目（point-wise）が支配的なので、おおよそ $\alpha^2\rho^2$ に比例して減る。
- **MobileNet v2**：**逆残差ブロック**（1×1 で**拡大**→ depth-wise 3×3 → 1×1 で**圧縮**、細い両端どうしをスキップ接続）と**線形ボトルネック**（圧縮後に ReLU を入れない。低次元で ReLU を使うと情報が失われるため）。活性化は $\mathrm{ReLU6}(x)=\min(\max(x,0),6)$。
- **MobileNet v3**：ブロック単位の探索（platform-aware NAS）と層単位の調整（NetAdapt）、**SE ブロック**、活性化 $\text{h-swish}(x)=x\,\mathrm{ReLU6}(x+3)/6$（swish の安価な近似）。
- **SE（Squeeze-and-Excitation）**：GAP でチャネルごとの値を作り（squeeze）、小さな全結合2層と sigmoid でチャネルごとの重みを出して（excitation）元の特徴に掛ける。
- **EfficientNet**：深さ $d$・幅 $w$・解像度 $r$ のどれか1つだけを大きくしても効果は頭打ちになる。そこで**複合係数** $\phi$ で同時に拡大する。

$$
d=\alpha^{\phi},\quad w=\beta^{\phi},\quad r=\gamma^{\phi},\qquad \alpha\beta^{2}\gamma^{2}\approx2\quad(\alpha,\beta,\gamma\ge1)
$$

- FLOPs は $d\,w^2r^2$ に比例する（幅と解像度は2乗で効く）ので、$\alpha\beta^2\gamma^2\approx2$ なら、$\phi$ を1増やすごとに FLOPs が約 2 倍になる。B0 で小さな格子探索により $\alpha=1.2,\ \beta=1.1,\ \gamma=1.15$（$1.2\times1.21\times1.3225\approx1.92$）を決め、$\phi$ を増やして B1〜B7 を作る。

物体検出（R-CNN 系・YOLO・SSD）やセグメンテーションなど、CNN の応用は [画像認識](../app/vision.md) を参照。

---

## 試験の着眼点

- 出力サイズ $\lfloor (H+2P-F)/S\rfloor+1$ を**暗算できる**ようにする。same パディングは $P=(F-1)/2$。
- パラメータ数は**入力の大きさに依存しない**（全結合との最大の違い）。チャネルまで含めて $F^2C_{in}C_{out}$ と数える。
- 畳み込みは**平行移動等変**、プーリングが**小さな位置ずれへの不変性**を与える。回転・拡大には等変でも不変でもない。
- depth-wise separable の計算量比は $1/C_{out}+1/F^2$。depth-wise は空間のみ、point-wise はチャネルのみ。
- グループ化畳み込みは $1/G$。$G=C_{in}=C_{out}$ が depth-wise。
- 拡張畳み込みは**パラメータを増やさず**受容野を広げる。転置畳み込みは**逆演算ではなく**拡大のための演算で、$H'=S(H-1)+F-2P$。
- Max pooling の逆伝播は最大位置だけ、Average は均等配分。GAP は全結合層を置き換えてパラメータを減らす。
- ResNet は劣化問題（訓練誤差の悪化）を残差接続で解決。勾配に $+1$ の項が残る。DenseNet は足し算ではなく連結。
- 年代と特徴の対応（AlexNet：ReLU・ドロップアウト、VGG：3×3 のみ、GoogLeNet：Inception、MobileNet：depth-wise separable、EfficientNet：複合スケーリング）。

---

## 参考

- [Convolutional Networks for Large-Scale Image Recognition（Simonyan, Zisserman, 2014）](https://arxiv.org/abs/1409.1556)（VGG）
- [Going Deeper with Convolutions（Szegedy ら, 2014）](https://arxiv.org/abs/1409.4842)（GoogLeNet）
- [Rethinking the Inception Architecture for Computer Vision（Szegedy ら, 2015）](https://arxiv.org/abs/1512.00567)（Inception v3）
- [Deep Residual Learning for Image Recognition（He ら, 2015）](https://arxiv.org/abs/1512.03385)（ResNet）
- [Wide Residual Networks（Zagoruyko, Komodakis, 2016）](https://arxiv.org/abs/1605.07146)
- [Densely Connected Convolutional Networks（Huang ら, 2016）](https://arxiv.org/abs/1608.06993)
- [MobileNets（Howard ら, 2017）](https://arxiv.org/abs/1704.04861)、[MobileNetV2（Sandler ら, 2018）](https://arxiv.org/abs/1801.04381)、[Searching for MobileNetV3（Howard ら, 2019）](https://arxiv.org/abs/1905.02244)
- [EfficientNet（Tan, Le, 2019）](https://arxiv.org/abs/1905.11946)
- [Multi-Scale Context Aggregation by Dilated Convolutions（Yu, Koltun, 2015）](https://arxiv.org/abs/1511.07122)
- [Deformable Convolutional Networks（Dai ら, 2017）](https://arxiv.org/abs/1703.06211)
- [A guide to convolution arithmetic for deep learning（Dumoulin, Visin, 2016）](https://arxiv.org/abs/1603.07285)
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition（He ら, 2014）](https://arxiv.org/abs/1406.4729)
- [Network In Network（Lin ら, 2013）](https://arxiv.org/abs/1312.4400)
- [Squeeze-and-Excitation Networks（Hu ら, 2017）](https://arxiv.org/abs/1709.01507)
- [WaveNet: A Generative Model for Raw Audio（van den Oord ら, 2016）](https://arxiv.org/abs/1609.03499)
- [Visualizing and Understanding Convolutional Networks（Zeiler, Fergus, 2013）](https://arxiv.org/abs/1311.2901)（ZFNet）
