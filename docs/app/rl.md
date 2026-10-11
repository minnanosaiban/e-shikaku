# 強化学習

<p class="kw">キーワード：マルコフ決定過程（MDP）、方策、収益、割引率、状態価値関数、行動価値関数、ベルマン方程式、動的計画法（方策反復・価値反復）、モンテカルロ法、TD 学習、SARSA、Q 学習、ε-greedy、DQN（経験再生・ターゲットネットワーク）、方策勾配法、REINFORCE、ベースライン、アドバンテージ、Actor-Critic、A3C、PPO、AlphaGo、AlphaGo Zero、モンテカルロ木探索</p>

!!! abstract "要点"
    - エージェントが環境と**やりとり**し、得た**報酬の合計（収益）**を最大にする行動のルール（**方策**）を、試行錯誤で学ぶ。教師データ（正解の行動）はない。
    - 道具は**価値関数**。「ある状態（と行動）から先で、どれだけ収益が得られるか」を、**ベルマン方程式**（今の報酬 ＋ 割り引いた次の価値）の形で更新する。更新する価値の元で、**Q 学習**（価値ベース）と**方策勾配法**（方策ベース）に分かれる。
    - 深層学習と組み合わせると、価値をネットワークで近似する **DQN**（経験再生・ターゲットネットワークで安定化）、方策もネットワークにする **Actor-Critic**、探索と組み合わせた **AlphaGo** になる。

---

## 問題設定：マルコフ決定過程 {#mdp}

$$
\cdots\ \to\ s_t\ \xrightarrow{\ a_t\sim\pi(\cdot\mid s_t)\ }\ (r_{t+1},\ s_{t+1})\ \to\ \cdots
$$

| 要素 | 記号 | 意味 |
|:--|:--|:--|
| 状態・行動の集合 | $\mathcal{S},\ \mathcal{A}$ | |
| 状態遷移確率 | $p(s'\mid s,a)$ | 状態 $s$ で行動 $a$ をとると、次の状態が $s'$ になる確率 |
| 報酬 | $r(s,a,s')$ | 1 ステップの報酬（即時報酬） |
| 割引率 | $\gamma\in[0,1]$ | 将来の報酬を割り引く |
| 方策 | $\pi(a\mid s)$ | 状態 $s$ で行動 $a$ をとる確率 |

- **マルコフ性**：次の状態は、**現在の状態と行動だけ**で決まり、それ以前の履歴には依らない。
- **収益**（リターン）$G_t$：時刻 $t$ 以降に得る報酬の割引和。再帰の形が重要。

$$
G_t=\sum_{k=0}^{\infty}\gamma^{k}\,r_{t+k+1}=r_{t+1}+\gamma\,G_{t+1}
$$

- $\gamma=0$ なら目先の報酬だけ、$1$ に近いほど先の報酬も重視。終わりのない課題でも、$\gamma<1$ なら和が発散しない（毎ステップ報酬 1 なら $G=1/(1-\gamma)$、$\gamma=0.9$ で 10）。
- **価値関数**：方策 $\pi$ のもとでの収益の期待値。

$$
v_\pi(s)=\mathbb{E}_\pi\bigl[G_t\mid s_t=s\bigr],\qquad q_\pi(s,a)=\mathbb{E}_\pi\bigl[G_t\mid s_t=s,\ a_t=a\bigr],\qquad v_\pi(s)=\sum_a\pi(a\mid s)\,q_\pi(s,a)
$$

---

## ベルマン方程式 {#bellman}

収益の再帰 $G_t=r_{t+1}+\gamma G_{t+1}$ の期待値をとると、**今の価値を、次の価値で書く**式になる。

$$
\begin{aligned}
&v_\pi(s)=\sum_{a}\pi(a\mid s)\sum_{s'}p(s'\mid s,a)\Bigl[r(s,a,s')+\gamma\,v_\pi(s')\Bigr] \\[2mm]
&q_\pi(s,a)=\sum_{s'}p(s'\mid s,a)\Bigl[r(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')\,q_\pi(s',a')\Bigr] \\[4mm]
&\text{（最適）}\quad v_*(s)=\max_a\sum_{s'}p(s'\mid s,a)\Bigl[r+\gamma\,v_*(s')\Bigr],\qquad
q_*(s,a)=\sum_{s'}p(s'\mid s,a)\Bigl[r+\gamma\max_{a'}q_*(s',a')\Bigr]
\end{aligned}
$$

- 最適な価値 $v_*,\ q_*$ がわかれば、**最適方策**は $\pi_*(s)=\arg\max_a q_*(s,a)$（貪欲に選ぶだけ）。
- 期待方程式は**方策を固定**した評価、最適方程式は **max** で方策も最適化する。

---

## 方策（行動の選び方） {#policy}

| 方策 | 行動の確率 | 特徴 |
|:--|:--|:--|
| **貪欲**（greedy） | 最大の $Q$ の行動を確率 1 で選ぶ | **探索しない**。今の推定が間違っていると、そのまま固定される |
| **$\varepsilon$-greedy** | $1-\varepsilon+\varepsilon/\lvert\mathcal{A}\rvert$（最大の行動）、$\varepsilon/\lvert\mathcal{A}\rvert$（それ以外） | 確率 $\varepsilon$ でランダム。**利用と探索**のバランス。$\varepsilon$ を徐々に下げることが多い |
| **ソフトマックス** | $\pi(a\mid s)=\dfrac{\exp(Q(s,a)/\tau)}{\sum_{a'}\exp(Q(s,a')/\tau)}$ | $Q$ が高いほど選ばれやすい。**温度 $\tau$** が高いと一様、低いと貪欲に近づく |

---

## 価値の推定（表形式） {#value-estimation}

| 手法 | 環境のモデル $p,r$ | 更新のタイミング | 更新の内容 |
|:--|:--|:--|:--|
| **動的計画法** | **既知が必要** | 全状態を掃引 | ベルマン方程式を使って、次の価値の**期待値**を計算 |
| **モンテカルロ法** | 不要 | **エピソードの終了後** | 実際の**収益 $G_t$** の平均 |
| **TD 学習** | 不要 | **1 ステップごと** | $r_{t+1}+\gamma V(s_{t+1})$（**ブートストラップ**） |

### 動的計画法 {#dp}

- **方策反復**：**方策評価**（今の方策の $v_\pi$ を、ベルマン期待方程式で収束するまで更新）と**方策改善**（$\pi\leftarrow\arg\max_a q_\pi(s,a)$ で貪欲に）を交互に繰り返す。
- **価値反復**：評価と改善を 1 つにして、**最適方程式の max**で価値だけを更新する。繰り返しの数は少なくて済む。最後に最適方策を取り出す。

$$
V_{k+1}(s)=\max_a\sum_{s'}p(s'\mid s,a)\Bigl[r(s,a,s')+\gamma\,V_k(s')\Bigr]
$$

- $\gamma<1$ なら、更新を繰り返すと**必ず収束**する（縮小写像）。弱点は、**遷移確率と報酬が分かっている**ときしか使えないこと（迷路は使える。運転は使えない）。分からないときは、経験（サンプル）から学ぶ**モデルフリー**の手法を使う。

### モンテカルロ法と TD 学習 {#mc-td}

$$
\begin{aligned}
&\text{MC}\quad &&V(s_t)\leftarrow V(s_t)+\alpha\bigl(G_t-V(s_t)\bigr) \\[2mm]
&\text{TD(0)}\quad &&V(s_t)\leftarrow V(s_t)+\alpha\bigl(\underbrace{r_{t+1}+\gamma V(s_{t+1})}_{\text{TD 目標}}-V(s_t)\bigr),\qquad \delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)\ \ (\text{TD 誤差})
\end{aligned}
$$

| MC | TD |
|:--|:--|
| 終了する課題のみ。**終わるまで更新できない** | 1 ステップごとに更新。**終わりのない課題**でも使える |
| 目標が**実際の収益**：**バイアスなし**、**分散が大きい** | 目標が**推定値を含む**：バイアスあり、**分散が小さい** |
| マルコフ性を使わない | マルコフ性を使う |

- $\alpha$ は**学習率（ステップ幅）**。更新は「現在の推定値を、目標へ $\alpha$ の割合だけ近づける」。

### SARSA と Q 学習 {#sarsa-q}

行動価値 $Q(s,a)$ を TD で更新する。違いは、**次の状態で何の価値を使うか**。

$$
\begin{aligned}
&\text{SARSA}\ (\text{オン方策}) && Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\Bigl[r_{t+1}+\gamma\,Q(s_{t+1},\,a_{t+1})-Q(s_t,a_t)\Bigr] \\[2mm]
&\text{Q 学習}\ (\text{オフ方策}) && Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\Bigl[r_{t+1}+\gamma\,\max_{a'}Q(s_{t+1},\,a')-Q(s_t,a_t)\Bigr]
\end{aligned}
$$

- **SARSA**：次に**実際に選んだ**行動 $a_{t+1}$（$\varepsilon$-greedy）の価値を使う。**行動をとる方策と、評価する方策が同じ**（**オン方策**）。探索で失敗する可能性も価値に織り込まれるので、慎重な（安全な）経路を学ぶ。名前は $(s,a,r,s',a')$ から。
- **Q 学習**：次の状態で**最大**の価値を使う。評価するのは貪欲方策で、行動は別の方策（$\varepsilon$-greedy など）で選んでよい（**オフ方策**）。探索の影響を受けず、**最適な $Q^*$ に収束**する（全ての $(s,a)$ を無限回訪れ、学習率が適切に小さくなる条件）。
- どちらも、**ベルマン方程式の右辺**をサンプルで近似し、そこへ近づけている。

<div class="ew" data-widget="rl-qlearn"></div>

!!! tip "動かしてみる"
    - 「Q学習」のまま $\alpha$ を 0 にすると、更新後の $Q(s,a)$ は更新前と同じです。$\alpha=1$ にすると、目標 $y$ そのものになります（$\alpha$ は目標へ近づく割合）。
    - $\gamma=0$ にすると、目標は $r$ だけになり、$Q(s',\cdot)$ のスライダーを動かしても結果が変わりません（先の価値を見ない）。
    - 「SARSA」に切り替えて「次に選ぶ」を最大でない行動にすると、目標が**小さく**なります（探索で選んだ行動の価値を使うため）。Q 学習は常に最大を使うので、選ぶ行動によらず同じです。

---

## DQN：深層強化学習 {#dqn}

- 状態が多い（画像など）と、表で $Q$ を持てない。$Q(s,a;\theta)$ を**ネットワークで近似**する。入力は状態、出力は**各行動の価値**（Atari では直近 4 フレームを重ねた画像 → CNN → 行動の数）。Q 学習の更新を、**二乗誤差の最小化**に置き換える。

$$
L(\theta)=\mathbb{E}_{(s,a,r,s')\sim D}\Bigl[\Bigl(\underbrace{r+\gamma\max_{a'}Q(s',a';\theta^{-})}_{\text{目標（固定）}}-Q(s,a;\theta)\Bigr)^2\Bigr]
$$

- ただし、そのまま使うと学習が不安定（「データの相関」と「目標が動く」ため）。DQN は 3 つの工夫を入れた。

| 工夫 | 解決すること |
|:--|:--|
| **経験再生**（experience replay） | 経験 $(s,a,r,s')$ を**メモリ $D$ に貯め**、ランダムに取り出して学習する。連続するデータは似ていて**相関が強い**（直近に引きずられる）のを断ち切り、同じ経験を何度も使える |
| **ターゲットネットワーク** | 目標の計算に使うネットワーク $\theta^{-}$ を、**一定期間固定**し、周期的に $\theta$ へコピーする。更新のたびに目標も動いて**追いかけっこ**になるのを防ぐ |
| **報酬のクリッピング** | 報酬を $\{-1,0,1\}$ にそろえる。ゲームごとの得点のスケールの違いを吸収でき、同じハイパーパラメータで学習できる（大小の区別は失う） |

- 誤差が大きいときの勾配爆発を避けるため、二乗誤差の代わりに **Huber 損失**（$|a|\le1$ で二乗、それ以外は絶対値）を使う。

$$
L_\delta(a)=\begin{cases}\tfrac12a^2&(|a|\le\delta)\\ \delta\bigl(|a|-\tfrac12\delta\bigr)&(\text{それ以外})\end{cases}
$$

- 改良：**Double DQN**（目標の行動の選択と評価を別のネットワークにして、**$\max$ による過大評価**を抑える）、**Dueling**（$Q=V(s)+A(s,a)-\overline{A}$ に分けて出力）、**優先度付き経験再生**（TD 誤差の大きい経験を多く使う）、これらを集めた **Rainbow**。
- DQN は**離散の行動**向き。連続の行動（ロボットの関節の角度）は、方策もネットワークにして扱う。

---

## 方策勾配法 {#policy-gradient}

- 価値ではなく**方策 $\pi_\theta(a\mid s)$ そのもの**をパラメータ化し、**期待収益 $J(\theta)$ を勾配上昇**で直接最大化する。行動が連続でも、確率的な方策でも扱える。

| 行動と状態 | 方策の形 |
|:--|:--|
| 離散の行動 | ネットワークの出力にソフトマックス：$\pi_\theta(a\mid s)=\dfrac{\exp(h_\theta(s,a))}{\sum_b\exp(h_\theta(s,b))}$ |
| 連続の行動 | **ガウス分布** $\mathcal{N}(\mu_\theta(s),\sigma^2)$ の平均（と分散）を出力 |

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\Bigl[\nabla_\theta\log\pi_\theta(a\mid s)\;q_{\pi_\theta}(s,a)\Bigr]\qquad(\text{方策勾配定理})
$$

??? note "導出の筋（対数微分のトリック）"
    $\nabla_\theta\pi_\theta=\pi_\theta\,\nabla_\theta\log\pi_\theta$ を使う。$\sum_s d(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\,q(s,a)=\sum_s d(s)\sum_a\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\,q(s,a)$ は、$\pi_\theta$ に従って集めたサンプルでの**期待値**になる（$d(s)$：方策に従ったときの状態の訪問頻度）。**環境の遷移確率 $p$ の微分が不要**なのが重要で、モデルが未知でも勾配が推定できる。

- 意味：「**収益が大きかった行動の確率を上げ、小さかった行動の確率を下げる**」。$q$ の大きさで重みが決まる。
- 例：離散のソフトマックス方策なら、$\nabla\log\pi_\theta(a\mid s)$ は logit $h_b$ について $\mathbb{1}[b=a]-\pi_\theta(b\mid s)$。選んだ行動の確率を上げ、他を下げる向き。

### REINFORCE・ベースライン・アドバンテージ {#reinforce}

- **REINFORCE**：$q_\pi$ を、エピソードで実際に得た収益 $G_t$ で置き換える（モンテカルロ近似）。$N$ エピソードの平均で勾配を近似する。

$$
\nabla_\theta J\approx\frac1N\sum_{n=1}^{N}\sum_{t}\nabla_\theta\log\pi_\theta\bigl(a_t^n\mid s_t^n\bigr)\,G_t^n
$$

- $G_t$ は**ばらつきが大きい**（分散が大きい）ので、学習が遅い。そこで状態だけに依る**ベースライン** $b(s)$ を引く。**勾配の期待値は変わらず（偏りなし）、分散だけが減る**。

$$
\nabla_\theta J=\mathbb{E}\Bigl[\nabla_\theta\log\pi_\theta(a\mid s)\,\bigl(q_\pi(s,a)-b(s)\bigr)\Bigr],\qquad
\mathbb{E}_a\bigl[\nabla_\theta\log\pi_\theta(a\mid s)\,b(s)\bigr]=b(s)\,\nabla_\theta\sum_a\pi_\theta(a\mid s)=b(s)\,\nabla_\theta1=0
$$

- ベースラインを**状態価値** $v_\pi(s)$ にすると、$q_\pi-v_\pi$ は「その行動が、平均よりどれだけ良いか」を表す**アドバンテージ** $A_\pi(s,a)$ になる。

$$
A_\pi(s,a)=q_\pi(s,a)-v_\pi(s),\qquad \nabla_\theta J=\mathbb{E}\Bigl[\nabla_\theta\log\pi_\theta(a\mid s)\,A_\pi(s,a)\Bigr]
$$

---

## Actor-Critic {#actor-critic}

- **Actor**（方策 $\pi_\theta$）が行動を選び、**Critic**（価値関数 $V_\phi$）がその良し悪しを評価する。REINFORCE の「エピソード終了まで待つ $G_t$」を、**Critic の推定**で置き換え、**ステップごと**に更新できるようにしたもの。
- **$k$ ステップ先読みのアドバンテージ**（$k=1$ なら TD 誤差 $\delta_t$）：

$$
\hat{A}_t=\sum_{i=0}^{k-1}\gamma^{i}\,r_{t+i+1}+\gamma^{k}\,V_\phi(s_{t+k})-V_\phi(s_t)
$$

| | 損失（最小化） |
|:--|:--|
| Actor | $-\log\pi_\theta(a_t\mid s_t)\,\hat{A}_t$（$\hat{A}_t$ は定数として扱う。エントロピー項を足すことが多い） |
| Critic | $\bigl(\sum_{i=0}^{k-1}\gamma^{i}r_{t+i+1}+\gamma^{k}V_\phi(s_{t+k})-V_\phi(s_t)\bigr)^2$（**$k$ ステップ先読みの収益**と $V_\phi(s_t)$ の差の 2 乗） |

- **A3C**（Asynchronous Advantage Actor-Critic）：**複数のワーカー**が、別々の環境で**並列**に経験を積み、**非同期**に共有のパラメータ（パラメータサーバー）の勾配を更新する。並列のサンプルが相関を弱める（経験再生の代わり）。現在の方策で集めた経験を使うので**オン方策**。同期型は **A2C**。
- **TRPO・PPO**：方策を一度に大きく変えると、学習が崩れる。**新旧の方策の比** $\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)$ を $[1-\epsilon,1+\epsilon]$ に**クリップ**して、更新を制限する（PPO）。

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb{E}_t\Bigl[\min\bigl(\rho_t\hat{A}_t,\ \mathrm{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\bigr)\Bigr]\quad(\text{最大化})
$$

- **DDPG・SAC**（連続行動）：DDPG は決定的な方策 $\mu_\theta(s)$ と $Q_\phi(s,a)$ を使い、経験再生とターゲットネットワークを使うオフ方策。**SAC** は「**報酬 $+$ 方策のエントロピー**」を最大化する（**最大エントロピー強化学習**）。エントロピー $H(\pi(\cdot\mid s))=-\sum_a\pi\log\pi$ は、行動のばらつき（探索）を促す。

$$
J=\mathbb{E}\Bigl[\sum_t\gamma^{t}\bigl(r_{t+1}+\beta\,H(\pi(\cdot\mid s_t))\bigr)\Bigr]\qquad(\beta:\text{エントロピーの重み})
$$

| 分類 | 手法 | 説明 |
|:--|:--|:--|
| 価値ベース | Q 学習・DQN・SARSA | 価値から貪欲に作る |
| 方策ベース | REINFORCE・PPO | 方策を直接最適化 |
| Actor-Critic | A3C・DDPG・SAC | 方策と価値の両方を学習 |
| オン方策 | SARSA・A3C・PPO | 今の方策で集めたデータだけを使う |
| オフ方策 | Q 学習・DQN・DDPG・SAC | 古い方策のデータも使える（経験再生が使える） |

---

## 模倣学習・逆強化学習 {#imitation}

- 報酬を設計しにくい課題では、**専門家のデモンストレーション**（状態と行動の組）から学ぶ。
- **行動クローニング**：専門家の行動を正解として、$\pi_\theta(a\mid s)$ を**教師あり学習**する。簡単だが、学習した方策が少しずれると、専門家のデータにない状態へ入り、**誤差が積み重なる**（分布のずれ）。**DAgger** は、学習した方策で動かして出会った状態に、専門家が正しい行動を付け足して再学習する。
- **逆強化学習**：専門家の行動から、**報酬関数そのもの**を推定し、その報酬で強化学習する。**GAIL** は、専門家と方策の行動を識別器で区別させる（GAN の考え方）。

---

## AlphaGo と AlphaGo Zero {#alphago}

### AlphaGo（2016）

| 部品 | 内容 |
|:--|:--|
| **SL 方策ネットワーク** $p_\sigma$ | **人間の棋譜**から、各局面の次の一手を当てる**教師あり学習**（CNN、盤面 → 361 手の確率）。先読みはしない |
| **RL 方策ネットワーク** $p_\rho$ | SL の重みで初期化し、**自己対戦**で**方策勾配法**により強くする |
| **価値ネットワーク** $v_\theta$ | 局面から**勝つ確率**（$-1\sim1$）を予測。RL 方策の自己対戦の結果 $z$ を教師にした**回帰** |
| **ロールアウト方策** | 軽い（線形の）モデルで、終局まで高速に打つ |
| **モンテカルロ木探索** | 上の部品を組み合わせて先読みする |

$$
\Delta\sigma\propto\frac{\partial\log p_\sigma(a\mid s)}{\partial\sigma},\qquad
\Delta\rho\propto\frac{\partial\log p_\rho(a_t\mid s_t)}{\partial\rho}\,z_t,\qquad
\Delta\theta\propto\bigl(z-v_\theta(s)\bigr)\frac{\partial v_\theta(s)}{\partial\theta}
$$

- RL 方策の更新は、**勝敗の報酬 $z_t=\pm1$** を掛けた方策勾配（勝った手は確率を上げ、負けた手は下げる）。
- **モンテカルロ木探索（MCTS）**：探索木に対し、①**選択**（根から葉まで、有望な手をたどる）→ ②**展開**（葉に新しい局面を追加）→ ③**評価**（価値ネットワークとロールアウトの結果を混ぜる）→ ④**更新**（通った辺の統計を更新）を繰り返す。最後に**訪問回数が最も多い**手を打つ。
- 選択の基準は、**行動価値 $Q$（活用）と、探索ボーナス $u$（まだ試していない手）**の和。

$$
a_t=\arg\max_a\Bigl(Q(s_t,a)+u(s_t,a)\Bigr),\qquad u(s,a)=c_{\mathrm{puct}}\,P(s,a)\,\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}
$$

- $P(s,a)$：SL 方策の事前確率（有望な手ほど大きい）、$N(s,a)$：訪問回数。**訪問が多い手ほど $u$ が小さく**なり、探索が広がる。葉の評価は $(1-\lambda)\,v_\theta(s_L)+\lambda z_L$（$z_L$ はロールアウトの結果）。

### AlphaGo Zero（2017）

- **人間の棋譜を使わず**、**自己対戦だけ**で学ぶ。4 つのネットワーク（ロールアウト・SL・RL・価値）を、**方策と価値を同時に出す 1 つのネットワーク**（**デュアルネットワーク** $(\mathbf{p},v)=f_\theta(s)$、残差ネットワーク）にまとめた。
- **ロールアウトを使わない**。MCTS の評価は、ネットワークの価値だけ。MCTS で得た手の確率 $\boldsymbol\pi$（訪問回数に比例）を**教師**にして方策を学び、**勝敗** $z$ で価値を学ぶ。

$$
L=(z-v)^2-\boldsymbol\pi^{\top}\log\mathbf{p}+c\,\|\theta\|^2
$$

- 同じ方法を、チェス・将棋にも適用したものが **AlphaZero**。

---

## 試験の着眼点

- **収益** $G_t=r_{t+1}+\gamma G_{t+1}$、**ベルマン方程式**＝「今の価値＝報酬＋$\gamma\times$次の価値」。最適方程式は $\max$ を含む。
- **方策反復**（評価と改善の繰り返し）と**価値反復**（max で 1 本化）。どちらも**環境のモデルが既知**のときの動的計画法。
- **MC**：エピソード終了後・バイアスなし・分散大。**TD**：1 ステップごと・ブートストラップ・分散小。
- **SARSA**＝オン方策、次の**実際の行動**の $Q$。**Q 学習**＝オフ方策、次の状態の **$\max$** の $Q$。
- **$\varepsilon$-greedy**：確率 $\varepsilon$ でランダム。探索と利用。**ソフトマックス方策**の温度。
- **DQN**：**経験再生**（相関の除去）・**ターゲットネットワーク**（目標を固定）・**報酬のクリッピング**・**Huber 損失**。
- **方策勾配定理**：$\mathbb{E}[\nabla\log\pi\cdot q]$。**ベースライン**は偏りを入れず分散を減らす。**アドバンテージ** $=q-v$。**Actor-Critic**：Actor が方策、Critic が価値。**A3C** は非同期・オン方策。
- **AlphaGo**：SL 方策 → RL 方策（方策勾配）→ 価値ネットワーク → MCTS。選択は $Q+u$、最終は**訪問回数最大**。**AlphaGo Zero**：自己対戦のみ・デュアルネットワーク・ロールアウトなし。

---

## 参考

- Sutton, Barto『Reinforcement Learning: An Introduction（第 2 版）』（[公開されている原稿](http://incompleteideas.net/book/the-book-2nd.html)）
- [Playing Atari with Deep Reinforcement Learning（DQN, Mnih ほか, 2013）](https://arxiv.org/abs/1312.5602)
- [Deep Reinforcement Learning with Double Q-learning（Van Hasselt ほか, 2015）](https://arxiv.org/abs/1509.06461)
- [Dueling Network Architectures for Deep Reinforcement Learning（Wang ほか, 2015）](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay（Schaul ほか, 2015）](https://arxiv.org/abs/1511.05952)
- [Rainbow: Combining Improvements in Deep Reinforcement Learning（Hessel ほか, 2017）](https://arxiv.org/abs/1710.02298)
- [Asynchronous Methods for Deep Reinforcement Learning（A3C, Mnih ほか, 2016）](https://arxiv.org/abs/1602.01783)
- [Trust Region Policy Optimization（Schulman ほか, 2015）](https://arxiv.org/abs/1502.05477)
- [Proximal Policy Optimization Algorithms（Schulman ほか, 2017）](https://arxiv.org/abs/1707.06347)
- [Continuous control with deep reinforcement learning（DDPG, Lillicrap ほか, 2015）](https://arxiv.org/abs/1509.02971)
- [Soft Actor-Critic（Haarnoja ほか, 2018）](https://arxiv.org/abs/1801.01290)
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning（DAgger, Ross ほか, 2010）](https://arxiv.org/abs/1011.0686)
- [Generative Adversarial Imitation Learning（Ho, Ermon, 2016）](https://arxiv.org/abs/1606.03476)
- [Mastering the Game of Go without Human Knowledge（AlphaGo Zero, Silver ほか, 2017, Nature）](https://www.nature.com/articles/nature24270)
- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm（AlphaZero, Silver ほか, 2017）](https://arxiv.org/abs/1712.01815)
- AlphaGo の原論文：Silver ほか「Mastering the game of Go with deep neural networks and tree search」（Nature, 2016）
