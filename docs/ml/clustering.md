# クラスタリング

<p class="kw">キーワード：k-means、k-means++、X-means、階層的クラスタリング、デンドログラム、ウォード法、群平均法、混合ガウスモデル（GMM）、EMアルゴリズム、DBSCAN</p>

!!! abstract "要点"
    - **クラスタリング**は、ラベルなしデータを「似たもの」のまとまりに分ける教師なし学習。何をもって似ているとするか（距離・密度・確率モデル）で手法が分かれる。
    - **k-means** は、割り当て（最も近い重心へ）と更新（クラスタの平均へ）を交互に繰り返し、クラスタ内の二乗距離の和 $J$ を**単調に減らす**。初期値に依存する局所解を返す。
    - **GMM** は k-means を確率モデルにして「どのクラスタに属するか」を確率（負担率）で表す。**EM アルゴリズム**で推定する。**DBSCAN** は密度でつなぎ、ノイズも検出する。

---

## k-means {#kmeans}

$N$ 個のデータ $\mathbf{x}^{(i)}$ を $K$ 個のクラスタ $C_1,\dots,C_K$ に分ける。各クラスタの代表を**重心**（セントロイド）$\boldsymbol\mu_k$ とする。

$$
\begin{aligned}
&\text{目的関数} && J=\sum_{k=1}^{K}\sum_{\mathbf{x}^{(i)}\in C_k}\bigl\|\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr\|^2=\sum_{i=1}^{N}\sum_{k=1}^{K}q_{ik}\bigl\|\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr\|^2 \\[3mm]
&\text{割り当て} && c^{(i)}=\operatorname*{arg\,min}_{k}\ \bigl\|\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr\|^2\qquad(q_{ik}=1\ \text{if}\ k=c^{(i)},\ \text{else}\ 0) \\[3mm]
&\text{更新} && \boldsymbol\mu_k=\frac{1}{|C_k|}\sum_{\mathbf{x}^{(i)}\in C_k}\mathbf{x}^{(i)}
\end{aligned}
$$

- $q_{ik}\in\{0,1\}$ はデータ $i$ がクラスタ $k$ に属するかを表す**帰属変数**。1つのデータは1つのクラスタにしか属さない（ハード割り当て）。
- 全体の最小化は解析的に解けない（NP 困難）ので、**Lloyd のアルゴリズム**で局所解を求める。

| 手順 | 内容 |
|:--|:--|
| ① 初期化 | $K$ 個の重心を決める（ランダムに $K$ 点を選ぶ、またはランダムに振り分けて平均をとる） |
| ② 割り当て | $\boldsymbol\mu_k$ を固定して、各点を最も近い重心のクラスタへ |
| ③ 更新 | $q_{ik}$ を固定して、各クラスタの平均を新しい重心に |
| ④ 収束判定 | 割り当てが変化しなくなるまで②③を繰り返す |

??? note "なぜ J が減るか（③の導出）"
    ②は $\boldsymbol\mu_k$ を固定したときに $J$ を最小にする $q_{ik}$ を選ぶ操作なので、$J$ は増えない。
    ③は $q_{ik}$ を固定して $\partial J/\partial\boldsymbol\mu_k=-2\sum_i q_{ik}(\mathbf{x}^{(i)}-\boldsymbol\mu_k)=0$ を解く操作で、$\boldsymbol\mu_k=\sum_i q_{ik}\mathbf{x}^{(i)}/\sum_i q_{ik}$（平均）となる。これも $J$ を増やさない。
    $J\ge0$ で割り当ての組み合わせは有限だから、必ず有限回で止まる。ただし**大域最適とは限らない**。

<div class="ew" data-widget="clustering-kmeans"></div>

!!! tip "動かしてみる"
    - 「1手進める」を押すと、割り当てと更新が交互に進み、右の棒（$J$）が**必ず下がる**（増えない）のが分かります。
    - 「ランダム初期化」で初期値の乱数を動かすと、同じデータでも**行き着く分け方が変わります**（$J$ の最終値が違う）。「k-means++」に切り替えると、悪い局所解になりにくくなります。
    - $K$ を実際の塊の数（3）より小さくすると塊が混ざり、大きくすると塊が分割されます。$K$ を増やすと $J$ は必ず小さくなるので、$J$ だけでは $K$ を決められません。

### 特徴と課題

- 計算量は 1 反復あたり $O(NKD)$。大規模データにはミニバッチ版もある。
- **初期値に依存する**（→ k-means++）。**外れ値に弱い**（平均を使うため）。**球状で大きさの近いクラスタ**を前提とする。**特徴のスケールに敏感**（標準化する）。
- **クラスタ数 $K$ を事前に決める**必要がある（→ X-means、下記の指標）。

### $K$ の選び方

- **エルボー法**：$K$ を変えて $J$ をプロットし、減り方が急に緩やかになる「肘」を選ぶ。
- **シルエット係数**：点 $i$ について、$a(i)$ を同じクラスタ内の他の点との平均距離、$b(i)$ を最も近い別クラスタの点との平均距離として

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}\in[-1,1]
$$

  1 に近いほどよく分離され、負なら別のクラスタのほうが近い。全点の平均が最大になる $K$ を選ぶ。
- 確率モデルなら AIC・BIC（GMM、X-means）。ほかにギャップ統計量など。

---

## k-means++ {#kmeanspp}

k-means の**初期値依存**を改善する。重心どうしが離れやすいように、確率的に初期重心を選ぶ。初期化以降は通常の k-means と同じ。

$$
D(\mathbf{x})^2=\min_{k\in\text{選択済み}}\|\mathbf{x}-\boldsymbol\mu_k\|^2,\qquad
P(\mathbf{x}^{(i)}\text{ を次の重心に選ぶ})=\frac{D(\mathbf{x}^{(i)})^2}{\sum_{j}D(\mathbf{x}^{(j)})^2}
$$

1. データ点から一様ランダムに1点を選び、第1重心とする。
2. 各点について、選択済みの重心までの最短距離の二乗 $D(\mathbf{x})^2$ を計算する。
3. $D(\mathbf{x})^2$ に比例する確率で、次の重心を1点選ぶ（既存の重心から遠い点ほど選ばれやすい）。
4. $K$ 個そろうまで 2・3 を繰り返し、そのあと k-means を実行する。

- 確率が距離の**二乗**に比例する点に注意（距離そのものではない）。
- 期待値の意味で、最適解の $O(\log K)$ 倍以内の $J$ が保証される。

---

## X-means {#xmeans}

**BIC**（ベイズ情報量基準）でクラスタを分割するか判断し、$K$ を自動で決める。

$$
\mathrm{BIC}=\log \hat{\mathcal{L}}-\frac{p}{2}\log N\qquad\Rightarrow\qquad \mathrm{BIC}(C_k\to C_{k1},C_{k2})>\mathrm{BIC}(C_k)\ \text{なら分割する}
$$

- $\hat{\mathcal{L}}$：クラスタ内のデータの最大尤度（各クラスタを正規分布とみなす）、$p$：モデルのパラメータ数、$N$：データ数。第2項が**パラメータが多いモデルへの罰**。
- この符号の取り方では BIC は大きいほどよい。「$-2\log\hat{\mathcal{L}}+p\log N$」の流儀では小さいほどよい（教科書で符号が逆なので注意）。

| 手順 | 内容 |
|:--|:--|
| ① 初期化 | $K=K_{\min}$ で k-means を実行する |
| ② 分割候補 | 各クラスタを2つに分ける（k-means++ で2重心を選び、そのクラスタ内で k-means） |
| ③ BIC 比較 | 分割前と後の BIC を比べ、改善するなら分割を採用する |
| ④ 繰り返し | $K_{\max}$ に達するか、分割するクラスタがなくなるまで②③を繰り返す |

---

## 階層的クラスタリング {#hierarchical}

最初は各データを1つのクラスタとし、**最も近い2つのクラスタを併合**していく（凝集型）。併合の過程を木にしたものが**デンドログラム**で、縦軸が併合したときのクラスタ間距離。好きな高さで横に切れば、その高さでのクラスタが得られる。$K$ を事前に決めなくてよい。

クラスタ $A,B$ の距離 $d(A,B)$ の測り方（**連結法**）で結果が変わる。

| 連結法 | クラスタ間距離 $d(A,B)$ | 特徴 |
|:--|:--|:--|
| 最短距離法（単連結） | $\min_{a\in A,b\in B}d(a,b)$ | 細長くつながる（**鎖効果**） |
| 最長距離法（完全連結） | $\max_{a\in A,b\in B}d(a,b)$ | 径の小さいまとまりになる |
| 群平均法 | $\dfrac{1}{\lvert A\rvert \lvert B\rvert }\sum_{a\in A}\sum_{b\in B}d(a,b)$ | 最短・最長の中間 |
| ウォード法 | $\dfrac{\lvert A\rvert \lvert B\rvert }{\lvert A\rvert +\lvert B\rvert }\Vert \boldsymbol\mu_A-\boldsymbol\mu_B\Vert ^2$ | 併合による**クラスタ内平方和の増加量**が最小のものを併合。k-means の $J$ に近い基準で、まとまりが揃う |
| 重心法 | $\Vert \boldsymbol\mu_A-\boldsymbol\mu_B\Vert ^2$ | 併合後に距離が小さくなる（逆転）ことがある |

- ウォード法の検算：2点（距離 $d$）の併合は $\frac{1\cdot1}{2}d^2=d^2/2$。2点を中点まわりに見た平方和 $2(d/2)^2=d^2/2$ に一致する。
- 距離行列が必要なので、メモリが $O(N^2)$、素朴な計算量は $O(N^3)$ で、大規模データには向かない。
- 逆向きに全体を分割していく**分割型**もある。

---

## 混合ガウスモデル（GMM）と EM アルゴリズム {#gmm}

データがどのクラスタから来たかは観測できない**潜在変数**だと考え、複数のガウス分布の混合でデータ全体をモデル化する。

$$
\begin{aligned}
&\text{モデル} && p(\mathbf{x})=\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k),\qquad \sum_k\pi_k=1 \\[3mm]
&\text{対数尤度} && \log\mathcal{L}=\sum_{i=1}^{N}\log\sum_{k=1}^{K}\pi_k\,\mathcal{N}(\mathbf{x}^{(i)}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k) \\[3mm]
&\text{E ステップ} && r_{ik}=\frac{\pi_k\,\mathcal{N}(\mathbf{x}^{(i)}\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)}{\sum_{j=1}^{K}\pi_j\,\mathcal{N}(\mathbf{x}^{(i)}\mid\boldsymbol\mu_j,\boldsymbol\Sigma_j)} \\[3mm]
&\text{M ステップ} && N_k=\sum_{i}r_{ik},\quad \boldsymbol\mu_k=\frac{1}{N_k}\sum_{i}r_{ik}\mathbf{x}^{(i)},\quad \boldsymbol\Sigma_k=\frac{1}{N_k}\sum_{i}r_{ik}\bigl(\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr)\bigl(\mathbf{x}^{(i)}-\boldsymbol\mu_k\bigr)^{\top},\quad \pi_k=\frac{N_k}{N}
\end{aligned}
$$

- $\pi_k$：**混合係数**（クラスタ $k$ から来る確率）、$r_{ik}$：データ $i$ がクラスタ $k$ に属する**負担率**（事後確率）。$\sum_k r_{ik}=1$。
- 対数の中に和があり、最尤解は閉じた形で求まらない。そこで**EM アルゴリズム**を使う。

| 手順 | 内容 |
|:--|:--|
| ① 初期化 | $\pi_k,\boldsymbol\mu_k,\boldsymbol\Sigma_k$ を決める（k-means の結果を使うことが多い） |
| ② E ステップ | 現在のパラメータで負担率 $r_{ik}$ を計算する（潜在変数の事後分布） |
| ③ M ステップ | 負担率を重みにした最尤推定でパラメータを更新する（M ステップは「重み付きの平均と共分散」） |
| ④ 収束判定 | 対数尤度の変化が閾値以下になるまで②③を繰り返す |

- 各反復で対数尤度は**減らない**（下界を E ステップで押し上げ、M ステップで最大化する、ということの帰結）。ただし**局所解**に陥る。初期値を変えて何度か実行する。
- ある成分がデータ1点に張り付いて $\boldsymbol\Sigma_k\to0$ になると尤度が発散する（**特異性**）。$\boldsymbol\Sigma_k$ に小さな $\varepsilon\mathbf{I}$ を足すなどして防ぐ。
- 成分数 $K$ は BIC・AIC で選ぶ。共分散の形（全共分散・対角・球状）も選べる。
- **k-means との関係**：共分散を $\sigma^2\mathbf{I}$（共通）、混合係数を等しくして $\sigma^2\to0$ とすると、$r_{ik}$ は 0/1 に近づき、E ステップは「最も近い重心への割り当て」、M ステップは「平均の計算」になる。つまり **k-means は GMM の極限（ハード割り当て）**。
- k-means との違い：**ソフト割り当て**（確率で表す）、**楕円形**のクラスタも表せる、生成モデルなのでデータをサンプリングできる。

---

## DBSCAN {#dbscan}

**密度**が高い領域をつなげてクラスタにする。$K$ を決める必要がなく、任意の形のクラスタとノイズを扱える。パラメータは近傍の半径 $\varepsilon$ と、コアになる最小点数 $\mathrm{minPts}$。

$$
\begin{aligned}
&\varepsilon\text{ 近傍} && \mathcal{N}_\varepsilon(\mathbf{x}^{(i)})=\{\mathbf{x}^{(j)}\mid d(\mathbf{x}^{(i)},\mathbf{x}^{(j)})\le\varepsilon\} \\[2mm]
&\text{コア点} && \bigl|\mathcal{N}_\varepsilon(\mathbf{x}^{(i)})\bigr|\ge\mathrm{minPts} \\[2mm]
&\text{密度直接到達可能} && \mathbf{x}^{(j)}\in\mathcal{N}_\varepsilon(\mathbf{x}^{(i)})\ \text{かつ}\ \mathbf{x}^{(i)}\ \text{がコア点}
\end{aligned}
$$

- $\varepsilon$ 近傍には点自身も含める。
- **密度到達可能**：$\mathbf{x}^{(1)}\to\mathbf{x}^{(2)}\to\cdots\to\mathbf{x}^{(n)}$ と、各段が密度直接到達可能な連鎖がある。
- 点は3種類：**コア点**（近傍が混んでいる）、**境界点**（コア点の近傍にあるが自身はコアでない）、**ノイズ**（どこにも到達できない）。

| 手順 | 内容 |
|:--|:--|
| ① コア点の判定 | 各点の $\varepsilon$ 近傍の点数を数え、$\mathrm{minPts}$ 以上ならコア点 |
| ② クラスタの拡張 | コア点から密度到達可能な点を、同じクラスタに統合していく |
| ③ ノイズ | どのクラスタにも属さない点をノイズとする |

- 利点：任意の形を扱える、ノイズを検出できる、$K$ 不要。
- 欠点：全体で**1つの $\varepsilon$** なので、密度が異なるクラスタが混在すると難しい（改良版が HDBSCAN）。高次元では距離が差を失い、$\varepsilon$ の設定が難しい。
- $\varepsilon$ は、各点の $k$ 番目の近傍までの距離を大きい順に並べた**k-距離グラフ**の肘から決める。

| 手法 | クラスタ数 | 割り当て | 形 | ノイズ |
|:--|:--|:--|:--|:--|
| k-means | 指定 | ハード | 球状 | 弱い |
| 階層的 | 木を切って決める | ハード | 連結法による | 弱い |
| GMM | 指定（BIC で選択） | ソフト | 楕円 | 弱い |
| DBSCAN | 自動 | ハード | 任意 | 検出する |

---

## 試験の着眼点

- k-means の目的関数は**クラスタ内の二乗距離和**。割り当て→更新を繰り返して $J$ は単調減少し、局所解に収束する。初期値依存・外れ値・$K$ の指定が弱点。
- k-means++ は距離の**二乗**に比例する確率で初期重心を選ぶ。X-means は BIC で分割を判断する。
- 階層的クラスタリングは $K$ 不要、デンドログラムを切って決める。ウォード法は平方和の増加が最小のものを併合する。
- GMM は EM アルゴリズム。E ステップで負担率、M ステップで加重平均・加重共分散・$\pi_k=N_k/N$。k-means はその極限。
- DBSCAN のパラメータは $\varepsilon$ と minPts。コア点・境界点・ノイズ。
- [次元圧縮](dimension-reduction.md)で先に次元を減らしてからクラスタリングすることが多い。評価は[性能指標](evaluation.md)の考え方（ラベルがあれば調整ランド指数など、なければシルエット係数）。

---

## 参考

- [Clustering（scikit-learn 公式ドキュメント）](https://scikit-learn.org/stable/modules/clustering.html)
- k-means++: The Advantages of Careful Seeding（Arthur, Vassilvitskii, 2007, SODA）
- X-means: Extending K-means with Efficient Estimation of the Number of Clusters（Pelleg, Moore, 2000, ICML）
- [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise（Ester ら, 1996）](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)
