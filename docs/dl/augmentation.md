# データ拡張

<p class="kw">キーワード：画像のデータ拡張、ノイズ付与、Random Flip・Erase・Crop・Contrast・Brightness・Rotate、RandAugment、MixUp、自然言語のデータ拡張、EDA、音声のデータ拡張、ボリューム変更、ピッチシフト、SpecAugment、CutMix、Cutout</p>

!!! abstract "要点"
    - 訓練データに**ラベルを保つ変換**（反転・切り抜き・ノイズなど）を加えて、データを人工的に増やす。**過学習を抑え**、変換に対する**頑健性**を上げる、データ側の正則化。
    - **どの変換が許されるか**は、タスクの「不変性」で決まる（文字認識で上下反転は不可、など）。
    - 発展形が、2つのデータを混ぜる **Mixup / CutMix**（**ラベルも混ぜる**）と、一部を隠す **Cutout / Random Erasing**。
    - 自然言語（EDA・逆翻訳）、音声（ノイズ・ピッチ変更・SpecAugment）にも同じ考え方がある。

---

## 画像の基本的な変換 {#image-basic}

- 変換の式は「出力の画素 $I'(x,y)$ を、入力 $I$ のどこから持ってくるか」で書く（変換後の座標から元の座標を引く）。$W,H$：画像の幅と高さ、座標は $0,\dots,W-1$ と $0,\dots,H-1$。

$$
\begin{aligned}
&\text{水平反転} && I'(x,y)=I(W-1-x,\ y) \\[2mm]
&\text{垂直反転} && I'(x,y)=I(x,\ H-1-y) \\[2mm]
&\text{回転} && I'(\mathbf{p})=I\bigl(\mathbf{R}(\theta)^{-1}(\mathbf{p}-\mathbf{c})+\mathbf{c}\bigr),\quad \theta\in[-\theta_{\max},\theta_{\max}] \\[2mm]
&\text{拡大・縮小} && I'(\mathbf{p})=I\bigl((\mathbf{p}-\mathbf{c})/s+\mathbf{c}\bigr),\quad s\in[s_{\min},s_{\max}] \\[2mm]
&\text{平行移動} && I'(x,y)=I(x-\Delta x,\ y-\Delta y) \\[2mm]
&\text{切り抜き（Crop）} && I'=I[y_1{:}y_2,\ x_1{:}x_2]\ \ (\text{のち元の大きさにリサイズ}) \\[2mm]
&\text{明るさ・コントラスト} && I'=\alpha I+\beta\quad(\alpha:\text{コントラスト},\ \beta:\text{明るさ}) \\[2mm]
&\text{ガウスノイズ} && I'=I+\boldsymbol{\varepsilon},\quad \boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\sigma^2\mathbf{I})
\end{aligned}
$$

- $\mathbf{p}=(x,y)$：画素の座標、$\mathbf{c}$：回転・拡大の中心（画像の中心）、$\mathbf{R}(\theta)$：回転行列。$\alpha I+\beta$ は 0〜255 などの範囲に**クリップ**する。
- 毎エポック、**ランダムに**パラメータ（角度・倍率・切り抜き位置など）を決めて適用する。各エポックで別の画像が見えるため、実質的にデータが増える。
- **テスト時には通常、拡張しない**。例外が Test-Time Augmentation（TTA）で、変換した複数の入力の予測を平均して精度を上げる（[アンサンブル](ensemble-hyperparam.md)の一種）。
- ラベルを変える変換は避ける：数字の 6 と 9（180° 回転）、文字の左右反転、医用画像の左右（臓器の位置が意味を持つ）など。

---

## 隠す・混ぜる変換 {#erase-mix}

### Cutout と Random Erasing {#cutout}

$$
\tilde{\mathbf{x}}=\mathbf{M}\odot\mathbf{x}+(\mathbf{1}-\mathbf{M})\odot\mathbf{v}
$$

- $\mathbf{M}\in\{0,1\}^{H\times W}$：隠す領域が 0 のマスク。$\mathbf{v}$：埋める値。ラベルは**変えない**。
- **Cutout**：固定の大きさの正方形を、ランダムな位置で**定数（0、正規化後は平均）**で埋める。**Random Erasing**：面積と縦横比をランダムにした長方形を、**ランダムな値**（ノイズや画素の平均）で埋める。
- 一部が隠れても認識できるようにして、特定の局所特徴に頼り切るのを防ぐ。遮蔽（オクルージョン）への頑健性も上がる。

### Mixup {#mixup}

$$
\tilde{\mathbf{x}}=\lambda\,\mathbf{x}_i+(1-\lambda)\,\mathbf{x}_j,\qquad
\tilde{\mathbf{y}}=\lambda\,\mathbf{y}_i+(1-\lambda)\,\mathbf{y}_j,\qquad
\lambda\sim\mathrm{Beta}(\alpha,\alpha)
$$

- 2 つのサンプルを**画素ごとに線形補間**し、**ラベル（one-hot）も同じ比率で混ぜる**。クラス間では出力も線形に変化する、という仮定を入れる効果があり、決定境界がなめらかになる。
- $\alpha=1$ なら $\lambda$ は一様分布。$\alpha<1$ では 0 か 1 に寄る（元のサンプルに近い）。$\alpha$ が大きいと $\lambda\approx0.5$ の強い混合が増える。
- 交差エントロピーは $\tilde{\mathbf{y}}$ について線形なので、損失は $\lambda\,\mathrm{CE}(f(\tilde{\mathbf{x}}),\mathbf{y}_i)+(1-\lambda)\,\mathrm{CE}(f(\tilde{\mathbf{x}}),\mathbf{y}_j)$ と同じ。
- 画像に限らず、音声や表形式、言語の埋め込み空間でも使える。

### CutMix {#cutmix}

$$
\tilde{\mathbf{x}}=\mathbf{M}\odot\mathbf{x}_i+(\mathbf{1}-\mathbf{M})\odot\mathbf{x}_j,\qquad
\tilde{\mathbf{y}}=\lambda\,\mathbf{y}_i+(1-\lambda)\,\mathbf{y}_j,\qquad
\lambda=1-\frac{\text{貼り付けた面積}}{WH}
$$

- 画像 $j$ の**長方形領域を切り取って**画像 $i$ に貼る。ラベルは**面積の比**で混ぜる。貼る長方形の幅と高さは $W\sqrt{1-\lambda},\ H\sqrt{1-\lambda}$（面積が $(1-\lambda)WH$）で、位置は一様乱数。画像からはみ出した分は切り落とし、$\lambda$ は実際の面積から計算し直す。
- Mixup と違って画素が不自然に重ならず、**局所的な特徴が保たれる**。Cutout と違って隠した部分に別の情報が入るので、学習に使える画素が減らない。
- **Mosaic**：4 枚の画像を 1 枚に並べる（YOLOv4 など物体検出で使用）。

<div class="ew" data-widget="aug-mix"></div>

!!! tip "動かしてみる"
    - Mixup で $\lambda$ を動かすと、画像のバツ印と枠の濃さが逆向きに変わり、ラベルも同じ比率で混ざります。
    - CutMix では、貼る正方形の一辺が $\lambda$ から決まり、ラベルの比は貼った**面積**に一致します（一辺 6 なら貼った面積は 36 で、$\tilde\lambda=0.64$）。
    - Cutout は同じ形で隠すだけで、ラベルは $(1,0)$ のままです。

---

## 自動で選ぶ方法 {#auto}

- **AutoAugment**：変換の組合せ（方策）を**強化学習で探索**。探索コストが大きい。
- **RandAugment**：変換の候補から**ランダムに $K$ 個**選び、**共通の強さ**で適用する。探索すべきパラメータは「個数」と「強さ」の 2 つだけで、探索コストが小さい。
- 手作業で変換を決めずにすむ反面、データセットごとに強さの調整は必要。

---

## 自然言語のデータ拡張 {#nlp}

- **EDA**（Easy Data Augmentation）：次の 4 操作を文にランダムに適用する。
    - 同義語置換（Synonym Replacement）：語を同義語に置き換える。
    - ランダム挿入（Random Insertion）：同義語を文中のどこかに挿入する。
    - ランダム入れ替え（Random Swap）：2 語の位置を入れ替える。
    - ランダム削除（Random Deletion）：語を確率的に削除する。
- **逆翻訳**（Back Translation）：別の言語に翻訳してから元の言語に戻し、**言い換え文**を得る。自然な文になりやすいが、翻訳モデルを動かすので計算コストが高い。
- **Mixup**：単語や文の**埋め込みベクトル**の段階で混ぜる（離散的な語そのものは混ぜられない）。
- 文の意味が変わらないことが前提。語の削除や入れ替えをやりすぎると、ラベルが変わってしまう。

---

## 音声のデータ拡張 {#audio}

| 手法 | 操作 | 狙い |
|:--|:--|:--|
| ノイズ付与 | 背景雑音やガウスノイズを加える | 雑音環境への頑健性 |
| ボリューム変更 | 振幅を定数倍する（ゲイン） | 音量・マイク距離の差 |
| 時間シフト・ストレッチ | 波形を前後にずらす・速度を変える | 発話位置・話速の差 |
| ピッチシフト | 音の高さを変える（速さは保つ） | 話者の声の高さの差 |
| Mixup | 2 つの音声（波形）を混ぜる | 画像と同じく境界をなめらかに |
| **SpecAugment** | スペクトログラム上で、時間方向・周波数方向の帯を隠す | 音声認識の標準的な拡張 |

- **SpecAugment**：対数メルスペクトログラム（周波数 × 時間）に対し、**時間ワーピング**（時間軸を局所的に歪める）、**周波数マスク**（連続する周波数チャネルを 0 にする）、**時間マスク**（連続する時間フレームを 0 にする）を行う。Cutout の音声版と見ればよい。波形でなく**スペクトログラム上**の操作なので軽い。→ [音声処理](../app/speech.md)

---

## 手法の比較 {#compare}

| 手法 | データ種類 | 計算コスト | ラベルを変えるか | 主な用途 |
|:--|:--|:--|:--|:--|
| 幾何変換（反転・回転・Crop） | 画像 | 低 | しない | 画像分類・物体検出 |
| 色の変換（明るさ・コントラスト） | 画像 | 低 | しない | 画像分類 |
| Cutout / Random Erasing | 画像 | 低 | しない | 画像分類、人物再識別 |
| Mixup | 画像・音声・埋め込み | 低 | する（比率で混合） | 分類全般 |
| CutMix | 画像 | 低 | する（面積比で混合） | 画像分類 |
| AutoAugment | 画像 | 高（探索時） | しない | 画像分類 |
| RandAugment | 画像 | 低 | しない | 画像分類 |
| 同義語置換・EDA | テキスト | 低 | しない | テキスト分類 |
| 逆翻訳 | テキスト | 高 | しない | 自然言語処理全般 |
| SpecAugment | 音声 | 低 | しない | 音声認識 |

---

## 試験の着眼点

- データ拡張の目的は、**データ量の水増し → 過学習の抑制・頑健性の向上**。ラベルを保つ変換が基本で、保てない変換は使わない。
- 学習時だけ適用し、推論時は適用しない（TTA は例外）。
- Mixup は**画素とラベルの両方**を $\lambda$ で線形補間（$\lambda\sim\mathrm{Beta}(\alpha,\alpha)$）。CutMix は**領域を貼り替え**、ラベルは**面積比**。Cutout は**ラベルを変えず**隠すだけ。
- Cutout＝定数で埋める、Random Erasing＝ランダム値で埋める。
- AutoAugment（強化学習で方策探索）と RandAugment（個数と強さの 2 パラメータ）の違い。
- EDA の 4 操作（同義語置換・挿入・入れ替え・削除）。逆翻訳は計算コストが高い。
- SpecAugment は**スペクトログラム**に対して時間・周波数のマスクと時間ワーピング。
- 関連：正則化として働く（[正則化](regularization.md)）。ノイズ付与は入力に対する L2 正則化と似た効果を持つ。

---

## 参考

- [mixup: Beyond Empirical Risk Minimization（Zhang ほか, 2017）](https://arxiv.org/abs/1710.09412)
- [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features（Yun ほか, 2019）](https://arxiv.org/abs/1905.04899)
- [Improved Regularization of Convolutional Neural Networks with Cutout（DeVries, Taylor, 2017）](https://arxiv.org/abs/1708.04552)
- [Random Erasing Data Augmentation（Zhong ほか, 2017）](https://arxiv.org/abs/1708.04896)
- [AutoAugment: Learning Augmentation Policies from Data（Cubuk ほか, 2018）](https://arxiv.org/abs/1805.09501)
- [RandAugment: Practical automated data augmentation with a reduced search space（Cubuk ほか, 2019）](https://arxiv.org/abs/1909.13719)
- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks（Wei, Zou, 2019）](https://arxiv.org/abs/1901.11196)
- [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition（Park ほか, 2019）](https://arxiv.org/abs/1904.08779)
