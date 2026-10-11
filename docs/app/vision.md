# 画像認識・物体検出・セグメンテーション

<p class="kw">キーワード：画像認識、バウンディングボックス、IoU、NMS、2段階検出器、1段階検出器、R-CNN、SPPnet、Fast R-CNN、Faster R-CNN（RPN）、RoI プーリング、RoIAlign、Mask R-CNN、FPN、YOLO、SSD、FCOS（アンカーフリー）、セマンティックセグメンテーション、インスタンスセグメンテーション、パノプティックセグメンテーション、FCN、SegNet、U-Net、Vision Transformer、距離学習、SiameseNet、TripletNet、MAML</p>

!!! abstract "要点"
    - 画像分類（何が写っているか）の CNN を**バックボーン**にして、物体検出（**どこに・何が**）と、セグメンテーション（**画素ごとに何か**）へ広げる。分類モデルは [CNN](../dl/cnn.md) を参照。
    - 検出は**重なりの度合い（IoU）**で正解と照合し、重複した検出を**NMS**で整理する。**2段階**（R-CNN 系）は精度、**1段階**（YOLO・SSD・FCOS）は速度に強い。
    - セグメンテーションは**エンコーダ・デコーダ**で解像度を戻し、**スキップ接続**で細部を補う（FCN・U-Net）。

---

## 検出の基本：IoU と NMS {#iou-nms}

- 検出は、物体ごとに**バウンディングボックス**（矩形）と**クラス**を出す。矩形は $(x,y,w,h)$（中心と幅・高さ）か、左上と右下の座標で表す。
- 予測と正解の矩形 $A,B$ の一致度が **IoU**（Intersection over Union）。

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

- 完全に重なれば $1$、まったく重ならなければ $0$。$|A\cap B|=\max(0,\Delta x)\cdot\max(0,\Delta y)$（重なりの幅 × 高さ）。
- 検出が正解かの判定は「**IoU が基準（例：0.5）以上**」。mAP などの評価指標は [機械学習の課題・検証・性能指標](../ml/evaluation.md#detection) にまとめてある。
- 1つの物体に近い位置の検出が何個も出るので、**NMS**（Non-Maximum Suppression、非極大値抑制）で整理する。

$$
\begin{aligned}
&1.\ \text{信頼度（スコア）の高い順に並べる} \\
&2.\ \text{先頭の箱を残し、それと } \mathrm{IoU}>\theta \text{ の箱を捨てる} \\
&3.\ \text{残りの先頭について 2 を繰り返す}
\end{aligned}
$$

<div class="ew" data-widget="vision-iou-nms"></div>

!!! tip "動かしてみる"
    - 「IoU」で $d_x,d_y$ を動かして B を A から離すと、共通部分が減って IoU が下がります。B を A の中に収めると、$|A\cap B|=|B|$ になります。
    - $s$ を変えると、$|B|$ と $|A\cap B|$ が同時に変わります。重なりが同じでも B が大きいほど $|A\cup B|$ が増えて、IoU は下がります。
    - 「NMS」で $\theta$ を下げると、残る箱が減ります。**物体の間隔を 30 程度にして $\theta=0.5$** にすると、隣の別の物体まで捨てられます（密集した物体での弱点。$\theta$ を上げる、Soft-NMS を使う、などで対処します）。

---

## 2段階と1段階 {#two-one-stage}

| | 2段階検出器 | 1段階検出器 |
|:--|:--|:--|
| 流れ | ①物体らしい領域の候補（**領域提案**）を出す → ②候補ごとに分類と矩形の補正 | 画像から直接、位置とクラスを出す（領域提案なし） |
| 代表 | R-CNN、Fast R-CNN、Faster R-CNN、Mask R-CNN | YOLO、SSD、RetinaNet、FCOS |
| 長所・短所 | 精度が高い。遅い | 速い（実時間）。小さい物体・密集した物体が苦手になりやすい |

- 1段階は、画像全体に敷き詰めた候補のうち**背景がほとんど**で、正例と負例の数が極端に偏る。これを和らげる工夫が、**Focal Loss**（簡単な例の損失を下げる。[損失関数まとめ](../math/loss.md)）や、ハード・ネガティブ・マイニング（SSD）。

---

## R-CNN 系列 {#rcnn}

| 手法 | 領域提案 | 特徴抽出 | 分類・位置 | 改良点 |
|:--|:--|:--|:--|:--|
| R-CNN（2014） | Selective Search（約 2000 個） | 候補ごとに**切り出して同じ大きさに変形**し、CNN に通す | SVM で分類、線形回帰で矩形を補正 | 検出に CNN 特徴が有効と示した。**候補ごとに CNN を回すので遅い** |
| SPPnet（2014） | Selective Search | **画像全体を1回** CNN に通し、候補に対応する領域を **SPP** で固定長に | SVM・回帰 | 計算の共有。入力の大きさを固定しなくてよい |
| Fast R-CNN（2015） | Selective Search | 全体を1回 CNN → 候補を **RoI プーリング**で固定長に → 全結合2層 | **softmax** と**矩形回帰**の2つの頭（マルチタスク損失） | 全体を一括で学習。領域提案が新たな遅延の原因 |
| Faster R-CNN（2015） | **RPN**（CNN で提案） | 特徴マップを RPN と共有 | Fast R-CNN と同じ | 提案までを CNN に含め、**ほぼ end-to-end** |
| Mask R-CNN（2017） | RPN | **RoIAlign** | 分類・矩形に加え、**マスク**（画素ごとの2値）も | インスタンスセグメンテーションに拡張 |

- **SPP**（空間ピラミッドプーリング）：領域を $1\times1,\ 2\times2,\ 4\times4$ などの格子に分けてプーリングし、連結する。入力の大きさによらず**固定長**のベクトルになる。
- **RoI プーリング**：RoI の座標を特徴マップの整数の格子に**丸め**、$7\times7$ などの小領域に分けて最大値を取る。
- **RoIAlign**：丸めずに、サンプル点の値を周囲 4 点の**双線形補間**で求め、最大か平均でまとめる。丸めによる**位置ずれが消える**ので、画素単位の精度が必要なマスクに効く。

$$
f(x,y)=\sum_{i,j\in\{0,1\}}\bigl(1-|x-x_i|\bigr)\bigl(1-|y-y_j|\bigr)\,f(x_i,y_j)\qquad(x_i,y_j\text{：}(x,y)\text{ を囲む4つの格子点})
$$

### RPN とアンカー {#rpn}

- 特徴マップの各位置に、**アンカー**（あらかじめ決めた大きさ・縦横比の箱。3 スケール × 3 縦横比 = $k=9$）を置く。各アンカーについて「物体か背景か」の確率（$2k$ 出力）と、矩形の補正量（$4k$ 出力）を出す。
- 矩形の補正量は、アンカー $(x_a,y_a,w_a,h_a)$ に対する相対値で表す。大きさは**対数**で扱うので、補正後の幅・高さが負にならない。

$$
t_x=\frac{x-x_a}{w_a},\quad t_y=\frac{y-y_a}{h_a},\quad t_w=\log\frac{w}{w_a},\quad t_h=\log\frac{h}{h_a}
$$

### マルチタスク損失

$$
\begin{aligned}
&L_{\mathrm{RPN}}=\frac{1}{N_{\mathrm{cls}}}\sum_i L_{\mathrm{cls}}(p_i,p_i^*)+\lambda\frac{1}{N_{\mathrm{reg}}}\sum_i p_i^*\,L_{\mathrm{reg}}(\mathbf{t}_i,\mathbf{t}_i^*) \\[2mm]
&L_{\text{Fast}}=L_{\mathrm{cls}}(\mathbf{p},u)+\lambda\,[u\ge1]\,L_{\mathrm{loc}}(\mathbf{t}^u,\mathbf{v}),\qquad L_{\mathrm{cls}}=-\log p_u,\qquad L_{\mathrm{loc}}=\sum_{i\in\{x,y,w,h\}}\mathrm{smooth}_{L_1}(t_i^u-v_i) \\[2mm]
&\mathrm{smooth}_{L_1}(x)=\begin{cases}0.5\,x^2&(|x|<1)\\|x|-0.5&(\text{それ以外})\end{cases}
\end{aligned}
$$

- $p_i^*$：アンカーが正例なら 1、負例なら 0。**正例のときだけ**矩形の損失を足す。$u$：正解のクラス（0 は背景）、$[u\ge1]$ は背景では矩形損失を無視する印。$\mathbf{t}^u$：クラス $u$ 用の補正量の予測、$\mathbf{v}$：正解の補正量。
- **Smooth L1** は、0 付近では二乗（なめらか）、離れると絶対値（外れ値に頑健）。
- Mask R-CNN は $L=L_{\mathrm{cls}}+L_{\mathrm{box}}+L_{\mathrm{mask}}$。$L_{\mathrm{mask}}$ は、**正解クラスのマスク**だけについて、画素ごとの2値交差エントロピーの平均（クラス間で競合させない）。

### FPN（特徴ピラミッド） {#fpn}

- CNN の深い層は意味が強いが解像度が低く、浅い層は解像度が高いが意味が弱い。**FPN**（Feature Pyramid Network）は、深い層を**上から下へ**拡大して浅い層の特徴に足し（横方向は 1×1 畳み込みでチャネルを合わせる）、**どの解像度でも意味の強い特徴マップ**を作る。大きさの違う物体を、別々の階層で検出する。
- FPN は Faster R-CNN・Mask R-CNN・RetinaNet・FCOS などで標準的に使われる。

---

## 1段階検出器 {#one-stage}

### YOLO {#yolo}

- 画像を $S\times S$ の格子に分け、**各セルが** $B$ 個の箱と、クラス確率 $C$ 個を直接回帰する（**検出を回帰問題として扱う**）。出力は $S\times S\times(5B+C)$。論文は $S=7,\ B=2,\ C=20$ で $7\times7\times30$。入力は $448\times448$。
- 1つの箱は $(x,y,w,h)$ と**信頼度**の5つ。$x,y$ は**セルの境界に対する**中心、$w,h$ は**画像全体に対する**比。

$$
\text{信頼度}=\Pr(\mathrm{Object})\cdot\mathrm{IoU}^{\mathrm{truth}}_{\mathrm{pred}},\qquad
\text{クラス別の信頼度}=\Pr(\mathrm{Class}_i\mid\mathrm{Object})\cdot\Pr(\mathrm{Object})\cdot\mathrm{IoU}
$$

- 推論では、クラス別の信頼度の低い箱を捨て、**NMS** で整理する。
- 損失は、位置・信頼度・クラスの**二乗誤差の和**（予測を $\hat{\ \ }$、正解を記号のままで表す）。物体の有無で重みを変える。

$$
\begin{aligned}
L=\ &\lambda_{\mathrm{coord}}\sum_{i=1}^{S^2}\sum_{j=1}^{B}\mathbb{1}^{\mathrm{obj}}_{ij}\Bigl[(x_i-\hat{x}_i)^2+(y_i-\hat{y}_i)^2+(\sqrt{w_i}-\sqrt{\hat{w}_i})^2+(\sqrt{h_i}-\sqrt{\hat{h}_i})^2\Bigr] \\
&+\sum_{i,j}\mathbb{1}^{\mathrm{obj}}_{ij}(C_i-\hat{C}_i)^2+\lambda_{\mathrm{noobj}}\sum_{i,j}\mathbb{1}^{\mathrm{noobj}}_{ij}(C_i-\hat{C}_i)^2+\sum_{i}\mathbb{1}^{\mathrm{obj}}_{i}\sum_{c}\bigl(p_i(c)-\hat{p}_i(c)\bigr)^2
\end{aligned}
$$

- $\lambda_{\mathrm{coord}}=5,\ \lambda_{\mathrm{noobj}}=0.5$。物体のない箱が大多数なので、その信頼度の損失を軽くし、位置の損失を重くする。
- 幅・高さは**平方根**を取って比べる。大きな箱の数ピクセルのずれより、小さな箱のずれのほうが問題だから。
- 弱点：セルごとに箱は $B$ 個・クラスは 1 つなので、**小さい物体が群れている**ときや、1つのセルに複数の物体があるときが苦手。位置の精度も低め。活性化は Leaky ReLU（$x>0$ なら $x$、それ以外は $0.1x$）。後の版で、アンカーボックス・バッチ正規化（v2）、多スケール予測（v3）などが入った。

### SSD {#ssd}

- ベースの CNN（VGG16）の後ろに、**縮小していく特徴マップ**を何段も足し、**各段から**検出する。大きな特徴マップは小さな物体、小さな特徴マップは大きな物体を担当する（**マルチスケール**）。
- 各特徴マップの各位置に**デフォルトボックス**（アンカーと同じ）を置き、クラスの確信度と矩形の補正量を出す。SSD300 は全部で 8732 個の箱。
- 段 $k$（全 $m$ 段）の箱の基準の大きさ $s_k$（画像に対する比）と、縦横比 $a_r$ での幅・高さ：

$$
s_k=s_{\min}+\frac{s_{\max}-s_{\min}}{m-1}(k-1),\qquad w_k^{a}=s_k\sqrt{a_r},\quad h_k^{a}=\frac{s_k}{\sqrt{a_r}}\qquad(s_{\min}=0.2,\ s_{\max}=0.9)
$$

- 学習：正解の箱と IoU が 0.5 以上のデフォルトボックスを**正例**にする。負例は損失の大きいものから正例の 3 倍までに絞る（**ハード・ネガティブ・マイニング**）。

$$
\begin{aligned}
&L=\frac1N\bigl(L_{\mathrm{conf}}+\alpha L_{\mathrm{loc}}\bigr),\qquad N=\text{正例の数} \\[2mm]
&L_{\mathrm{loc}}=\sum_{i\in\mathrm{Pos}}\sum_{m\in\{cx,cy,w,h\}}\mathrm{smooth}_{L_1}\bigl(l_i^m-\hat{g}_i^m\bigr),\qquad
\hat{g}^{cx}=\frac{g^{cx}-d^{cx}}{d^{w}},\ \ \hat{g}^{cy}=\frac{g^{cy}-d^{cy}}{d^{h}},\ \ \hat{g}^{w}=\log\frac{g^{w}}{d^{w}},\ \ \hat{g}^{h}=\log\frac{g^{h}}{d^{h}} \\[2mm]
&L_{\mathrm{conf}}=-\sum_{i\in\mathrm{Pos}}\log\hat{c}_i^{\,p}-\sum_{i\in\mathrm{Neg}}\log\hat{c}_i^{\,0}\qquad(\hat{c}=\mathrm{softmax}(c),\ 0\text{ は背景})
\end{aligned}
$$

- $d$：デフォルトボックス、$g$：正解の箱、$l$：予測の補正量。RPN の $t_x\dots t_h$ と**同じ形**の符号化（中心は箱の大きさで割り、幅・高さは対数）。
- 小さな物体は、データ拡張（縮小して貼る・切り取り）で精度が上がる。

### アンカーフリー：FCOS {#fcos}

- **アンカーボックスの欠点**：縦横比・個数が精度を左右するハイパーパラメータになる／ほとんどが背景で正負が偏る／全アンカーの IoU 計算が重い。
- **FCOS**（Fully Convolutional One-Stage）は、特徴マップの各位置 $(x,y)$ から、**正解の箱の4辺までの距離** $(l^*,t^*,r^*,b^*)$ を直接回帰する。各位置が 1 つの箱を予測する。**FPN** の各階層（$P_3$〜$P_7$）が、大きさの範囲を分担する。

$$
l^*=x-x_0,\quad t^*=y-y_0,\quad r^*=x_1-x,\quad b^*=y_1-y\qquad(\text{正解の箱の左上 }(x_0,y_0)\text{、右下 }(x_1,y_1))
$$

- 正解の箱の内側の位置を**すべて正例**にするので、箱の中心から遠い位置からも質の低い箱が出る。そこで**センターネス**（中心らしさ）を同時に学習し、推論時に「クラスの確率 × センターネス」で順位を付けて質の低い箱を落とす（NMS の前に効く）。

$$
\mathrm{centerness}=\sqrt{\frac{\min(l^*,r^*)}{\max(l^*,r^*)}\cdot\frac{\min(t^*,b^*)}{\max(t^*,b^*)}}\in[0,1]
$$

- 箱の中心では $l^*=r^*,\ t^*=b^*$ で $1$、縁に近づくと $0$ に近づく。平方根は減衰をゆるやかにするため。損失は2値交差エントロピー。
- 損失全体：$L=\dfrac1{N_{\mathrm{pos}}}\sum_{x,y}L_{\mathrm{cls}}+\dfrac{\lambda}{N_{\mathrm{pos}}}\sum_{x,y}\mathbb{1}_{c^*_{x,y}>0}L_{\mathrm{reg}}$。$N_{\mathrm{pos}}$ は**正例の位置の数**、$L_{\mathrm{cls}}$ は Focal Loss、$L_{\mathrm{reg}}$ は IoU 損失。

---

## セグメンテーション {#segmentation}

| 種類 | 何を求めるか | 同じクラスの別の物体 | 背景（空・道路など） |
|:--|:--|:--|:--|
| **セマンティック** | 全画素のクラス | **区別しない** | 扱う |
| **インスタンス** | 物体ごとの領域 | **区別する** | 扱わない |
| **パノプティック** | 上の2つを統合（数えられる物体 = thing はインスタンス、数えられない背景 = stuff はセマンティック） | 区別する | 扱う |

- 損失は、**画素ごとのクロスエントロピー**の平均が基本。指標は画素精度、**mIoU**（クラスごとの IoU の平均）、**Dice 係数** $\dfrac{2|A\cap B|}{|A|+|B|}$（IoU が $J$ なら $2J/(1+J)$）。Dice 損失 $=1-\mathrm{Dice}$ は、領域が小さくクラスが偏るとき（医用画像）に使う。

### FCN {#fcn}

- **全結合層を 1×1 畳み込みに置き換え**て、画像の大きさを問わない**全層畳み込み**にした。出力は分類のクラスではなく、**クラス数チャネルの粗いマップ**。それを**転置畳み込み**（逆畳み込み）で入力の大きさまで拡大する。
- 拡大だけでは粗いので、**スキップ接続**を使う。より浅い層（プーリング層の出力）の特徴マップを、拡大した深い層の出力に**足し算**する（FCN-32s → 16s → 8s と細かくなる）。深い層は「何か」、浅い層は「どこか」の情報を持つ。

### SegNet {#segnet}

- **エンコーダ・デコーダ**。エンコーダの max pooling で、**選ばれた位置（インデックス）**を記憶し、デコーダの拡大でその位置に値を戻す。特徴マップそのものを持ち越す FCN より**メモリが小さく**、位置情報も保てる。バッチ正規化を使う。

### U-Net {#unet}

- 左右対称の **U 字**のエンコーダ・デコーダ。デコーダの各段に、同じ解像度のエンコーダの特徴マップを**切り出して連結（copy and crop → concatenate）**する。FCN が足し算なのに対し、U-Net は**チャネル方向に連結**する。
- **浅い層**の特徴は解像度が高く**局所的・細かい**（輪郭など）、**深い層**の特徴は解像度が低く**大域的・意味的**。スキップ接続が前者を、拡大の経路が後者をデコーダへ運び、両方を保ったまま元の解像度の出力を作る。
- 少ない教師データ（医用画像）向けに、**弾性変形のデータ拡張**や、接する細胞の境界に大きな重みを付ける損失が提案された。pix2pix の生成器にも使われる（→ [自己教師あり学習・生成モデル](../ml/ssl-generative.md)）。

### そのほか

- **拡張（dilated）畳み込み**（DeepLab）：プーリングで解像度を落とさずに受容野を広げる。実効カーネルは $k+(k-1)(r-1)$（$k=3,\ r=2$ で 5）。異なる拡張率を並べた **ASPP**、全体の文脈を集める **PSPNet** の**ピラミッドプーリング**がある。
- **Mask R-CNN** はインスタンスセグメンテーション（検出してから領域ごとにマスクを出す）、**パノプティック**は両者の統合。

---

## Vision Transformer {#vit}

- 画像を $16\times16$ などの**パッチ**に切り、各パッチを平坦化して線形変換した**トークン**の列として、Transformer のエンコーダに入れる（[Transformer](../dl/transformer.md)）。先頭に分類用トークン `[CLS]` を足し、**位置埋め込み**を足す。
- 畳み込みのような「近くを見る」仮定（**帰納バイアス**）がないので、**大量のデータで事前学習**すると CNN を超える。少ないデータでは CNN が有利。
- 検出の **DETR**（箱の集合を直接予測し、NMS が不要）や、**Swin**（窓の中で Attention）など、検出・分割にも広がっている。

---

## 少数ショット学習：距離学習とメタ学習 {#few-shot}

- **距離学習**：同じクラスは近く、違うクラスは遠くなる**埋め込み**を学ぶ。新しいクラスは、**埋め込みの近さ**で追加学習なしに識別できる（顔認証など、クラス数が多くて各クラスの画像が少ない場面）。

| 手法 | 入力 | 損失 |
|:--|:--|:--|
| SiameseNet | 画像の**ペア**（重みを共有する2つのネットワーク） | **対照損失（contrastive loss）** |
| TripletNet（FaceNet） | 基準・同クラス・別クラスの**3つ組** | **Triplet 損失** |

$$
\begin{aligned}
&L_{\mathrm{contrastive}}=\tfrac12\Bigl(Y\,d^2+(1-Y)\max(m-d,\,0)^2\Bigr),\qquad d=\|f(\mathbf{x}_i)-f(\mathbf{x}_j)\|_2\quad(Y=1\text{：同クラス}) \\[2mm]
&L_{\mathrm{triplet}}=\max\bigl(0,\ d_{ap}^2-d_{an}^2+m\bigr),\qquad d_{ap}=\|f(\mathbf{x}_a)-f(\mathbf{x}_p)\|_2,\ \ d_{an}=\|f(\mathbf{x}_a)-f(\mathbf{x}_n)\|_2
\end{aligned}
$$

- 対照損失：同クラス（$Y=1$）は距離 $d$ を縮め、別クラスはマージン $m$ より近いときだけ押し広げる。Triplet 損失：**基準から正例までの距離が、負例までの距離よりマージン $m$ 以上小さく**なれば損失は 0。
- Triplet は組み合わせが膨大で、学習が進むとほとんどの組が損失 0 になる。**FaceNet** は、ミニバッチ内で「**セミハード**」な負例 $d_{ap}^2<d_{an}^2<d_{ap}^2+m$ を選ぶ（難しすぎる負例は学習を不安定にする）。
- **MAML**（Model-Agnostic Meta-Learning）：**少数のデータと数回の勾配更新で新しいタスクに適応できる初期値 $\theta$** を学ぶ。タスク $\mathcal{T}_i$ ごとの内側の更新と、その後の損失で外側を更新する（**勾配の勾配**）。

$$
\theta_i'=\theta-\alpha\nabla_\theta L_{\mathcal{T}_i}(f_\theta),\qquad \theta\leftarrow\theta-\beta\nabla_\theta\sum_{\mathcal{T}_i}L_{\mathcal{T}_i}(f_{\theta_i'})
$$

- **Prototypical Networks**：少数のサンプル（サポート集合）をそれぞれ埋め込み、**クラスごとの平均（プロトタイプ）**を作る。問い合わせの画像は、**最も近いプロトタイプのクラス**に分類する（距離にマイナスをつけたものをソフトマックスにかける）。**Matching Networks** は、サポート集合の各サンプルへの Attention の重みで、ラベルを重み付き多数決する。
- ファインチューニングは1つのタスクの損失だけで初期値を決めるが、MAML は「**更新した後の**損失」が小さくなる初期値を選ぶ。微分可能な任意のモデルに使える一方、二階微分でメモリ・計算が重い。

---

## 試験の着眼点

- IoU $=\dfrac{|A\cap B|}{|A\cup B|}$。分母は**和集合**。NMS は「スコア順に残し、IoU が閾値を超える箱を捨てる」。**閾値を下げると**重複は減るが、近接した別の物体も消える。
- 2段階（R-CNN 系：領域提案 → 分類）は精度、1段階（YOLO・SSD・FCOS）は速度。1段階は**正負の偏り**に弱く、Focal Loss などで補う。
- R-CNN → SPPnet（特徴計算の共有）→ Fast R-CNN（RoI プーリング・マルチタスク損失）→ Faster R-CNN（**RPN** で領域提案も CNN 化）→ Mask R-CNN（**RoIAlign**・マスク）。
- **RoIAlign** は丸め（量子化）をやめて**双線形補間**を使う。位置ずれが減る。
- **アンカー**：特徴マップの各位置に置く基準の箱。補正量は $(\Delta x/w_a,\ \Delta y/h_a,\ \log w/w_a,\ \log h/h_a)$。**Smooth L1** は外れ値に頑健。
- YOLO：出力 $S\times S\times(5B+C)$、信頼度 $=\Pr(\mathrm{Object})\cdot\mathrm{IoU}$、幅・高さは平方根で比較。SSD：マルチスケール特徴マップ・デフォルトボックス・ハードネガティブマイニング。FCOS：**アンカーフリー**・4辺までの距離・**センターネス**・FPN。
- セグメンテーション：**セマンティック**（クラスのみ）、**インスタンス**（個体を区別）、**パノプティック**（両方）。FCN は**足し算**、U-Net は**連結**のスキップ接続。SegNet は**プーリングのインデックス**を再利用する。
- U-Net のスキップ接続は、浅い層の**細かい局所情報**をデコーダへ渡す（浅い層が大域的、ではない）。
- 距離学習：SiameseNet は**ペア**・対照損失、TripletNet は**3つ組**・Triplet 損失。MAML は**初期値**を学ぶメタ学習。

---

## 参考

- [Rich feature hierarchies for accurate object detection and semantic segmentation（R-CNN, Girshick ほか, 2013）](https://arxiv.org/abs/1311.2524)
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition（SPPnet, He ほか, 2014）](https://arxiv.org/abs/1406.4729)
- [Fast R-CNN（Girshick, 2015）](https://arxiv.org/abs/1504.08083)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（Ren ほか, 2015）](https://arxiv.org/abs/1506.01497)
- [Mask R-CNN（He ほか, 2017）](https://arxiv.org/abs/1703.06870)
- [Feature Pyramid Networks for Object Detection（Lin ほか, 2016）](https://arxiv.org/abs/1612.03144)
- [You Only Look Once: Unified, Real-Time Object Detection（YOLO, Redmon ほか, 2015）](https://arxiv.org/abs/1506.02640)
- [SSD: Single Shot MultiBox Detector（Liu ほか, 2015）](https://arxiv.org/abs/1512.02325)
- [Focal Loss for Dense Object Detection（RetinaNet, Lin ほか, 2017）](https://arxiv.org/abs/1708.02002)
- [FCOS: Fully Convolutional One-Stage Object Detection（Tian ほか, 2019）](https://arxiv.org/abs/1904.01355)
- [Fully Convolutional Networks for Semantic Segmentation（FCN, Long ほか, 2014）](https://arxiv.org/abs/1411.4038)
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation（Badrinarayanan ほか, 2015）](https://arxiv.org/abs/1511.00561)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation（Ronneberger ほか, 2015）](https://arxiv.org/abs/1505.04597)
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs（Chen ほか, 2016）](https://arxiv.org/abs/1606.00915)
- [Panoptic Segmentation（Kirillov ほか, 2018）](https://arxiv.org/abs/1801.00868)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（ViT, Dosovitskiy ほか, 2020）](https://arxiv.org/abs/2010.11929)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering（Schroff ほか, 2015）](https://arxiv.org/abs/1503.03832)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks（MAML, Finn ほか, 2017）](https://arxiv.org/abs/1703.03400)
- [Prototypical Networks for Few-shot Learning（Snell ほか, 2017）](https://arxiv.org/abs/1703.05175)
- [Matching Networks for One Shot Learning（Vinyals ほか, 2016）](https://arxiv.org/abs/1606.04080)
