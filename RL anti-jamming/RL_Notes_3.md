## Chapter 3 Finite Markov Decision Processes

### 3.1 The Agent-Environment Interface

马尔可夫决策过程（Markov Decision Processes，MDP）是智能体与环境的交互中学习并最后达到目标的一个数学框架，学习和决策的人称为agent，agent之外的便为environment。

关于agent和environment的判定，即边界的确定十分灵活且主观，应该根据实际问题灵活确定。

简单来说，MDP为某一状态$S_t$时，根据此状态做出动作$A_t$，进而分别引发奖励$R_{t+1}$和状态的转移至$S_{t+1}$，所以同一时期的奖励和状态永远是相同下标的。由此会形成类似如下的轨迹（trajectory）：
$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,S_3,\cdots
$$
MDP的数学框架可将问题简化至以下三个方面：

- agent做出的决策（actions）；
- 决策做出的基础或者说是条件（states）；
- 决策做出的目标（rewards）。

根据以上参数，将$R_t$和$S_t$视为随机变量，则可定义以下函数：
$$
p(s',r\mid s,a)\doteq\Pr\{S_t=s',R_t=r\mid S_{t-1}=s,A_{t-1}=a\}
$$
其类似于马尔可夫过程的状态转移概率，只不过新增了一个随机变量$R_t$来作为状态转移时所获奖励的表征；同时，新增的$A_t$作为条件将普通的状态转移概率以动作为节点进行分支（下面的状态转移节点图更能清晰的体现 ），所以其本质就是在某一状态$S_{t - 1}$下，选择动作$A_{t - 1}$时，描述得到的奖励$R_t$和状态的转移$S_{t}$发生怎样的变化，有着怎样的分布的二维概率密度函数。以该函数为基础，又可派生定义出其他函数。

下式更加注重的是某状态做出**某决策后状态的转移**，不关注得到多少奖励，更加接近于马尔可夫过程的状态转移概率（仅仅只有增加了选择动作这个条件）。故其在二维概率分布的基础上，求其随机变量$S_t$的一维边缘函数；
$$
p(s'\mid s,a)\doteq\Pr\{S_t=s'\mid S_{t-1}=s,A_{t-1}=a\}\:=\:\sum_{r\in\mathcal{R}}p(s',r\mid s,a)
$$
下式更加在意**某状态选择某动作后得到的奖励期望**。同样在二维概率分布的基础上，先求其随机变量$R_t$的一维边缘函数，再对该边缘函数求一阶原点矩，由此得到二参数的奖励期望函数；
$$
r(s,a)\doteq\mathbb{E}[R_t\mid S_{t-1}=s,A_{t-1}=a]\:=\:\sum_{r\in\mathcal{R}}r\sum_{s'\in\mathcal{S}}p(s',r\mid s,a)
$$
利用贝叶斯公式得到后验概率后再对$R_t$求期望，可得到下式。相比于$r(s,a)$，其增加的$S_{t+1}$使得其更加关心状态转移后而得到的奖励期望，本质和后验概率一样。
$$
r(s,a,s')\doteq\mathbb{E}[R_t\mid S_{t-1}=s,A_{t-1}=a,S_t=s']=\sum_{r\in\mathcal{R}}r\frac{p(s',r\mid s,a)}{p(s'\mid s,a)}
$$
在$r(s, a)$之上，对动作$a$按照策略进行加权求和，得到当前状态$s$的平均奖励
$$
r(s) = \mathbb{E}[R_t \mid S_t = s] = \sum\limits_{a \in \mathcal{A}} \pi(a \mid s) r(s, a)
$$
如下为一个MDP的状态转移节点图

![image-20230924172133065](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230924172133065.png)

状态转移节点图有两个类型的节点，一是状态节点，二是动作节点，动作节点是马尔科夫链的状态转移图上新增的。

为了熟悉bellman最优方程，写出上述MDP的关于$v_*$和关于$q_*$的bellman最优方程
$$
\left.v_*(\mathrm{h})=\max\left\{\begin{array}{l}p(\mathrm{h}|\mathrm{h},\mathrm{s})[r(\mathrm{h},\mathrm{s},\mathrm{h})+\gamma v_*(\mathrm{h})]+p(1|\mathrm{h},\mathrm{s})[r(\mathrm{h},\mathrm{s},1)+\gamma v_*(1)],\\p(\mathrm{h}|\mathrm{h},\mathrm{w})[r(\mathrm{h},\mathrm{w},\mathrm{h})+\gamma v_*(\mathrm{h})]+p(1|\mathrm{h},\mathrm{w})[r(\mathrm{h},\mathrm{w},1)+\gamma v_*(1)]\end{array}\right.\right\}
$$
其他的自己算，或者直接看答案。关键是奖励一般只是简单的分布，所以将奖励分布函数简化很重要。

### 3.2 Goals and Rewards & 3.3 Returns and Episodes

目标（Goals）即我们到底想让agent达到一个怎样的目的，解决怎样的问题，但是我们只能通过设置agent的奖励（Rewards）来反映这件事，agent的任务就是最大化积累的奖励。通过设置奖励来使得agent达到我们想要的目标是强化学习一个重要的特征。我们所选择的奖励应该告诉agent什么是我们的目标，而不是告诉agent怎样达到目标。

刚才我们只泛泛的谈论agent的目的是最大化积累的奖励，那么该如何具体的量化这件事呢？假设agent当前处于状态$S_t$，那么我们希望agent能最大化的使当前状态的期望收益（expected return），使用$G_t$来表示收益（return），最简单形式的$G_t$为
$$
G_t\doteq R_{t+1}+R_{t+2}+R_{t+3}+\cdots+R_T
$$
其中$T$为最终时刻，记住**奖励$R_{T}$和终端状态$S_{T}$是同时产生的**，故此时系统共有从$S_0$到$S_T$共$T+1$个状态，我们共做了$T$次动作。我们将某次试验从最初始的状态到终端状态（terminal state）称为一个幕（episode）或一次试验（trial）。在一个任务中，可能包含很多个episodes，而每个episode都是相互之间独立的，这样的任务称之为episodic tasks。在一次episodic task中，我们将所有的nonterminal即非终端状态的集合用$\mathcal{S}$来表示，将所有状态（包括终端状态）用$\mathcal{S^+}$表示。

有终端状态的tasks我们称之为episodic tasks，同样的，没有终端状态的tasks我们称之为continuing tasks，continuing tasks也同样很常见，因为通常任务会不断地进行下去而不结束。此时其不再包含终端状态，所以其所有的状态集合为$\mathcal{S}$。我们同样关注其收益return，如果还是以episodic tasks一样定义$G_t$，最后得到的$G_t$是发散的。由此我们引入一个折扣因子（discount rate）$\gamma$，由此新定义的回报（discounted return）$G_t$如下
$$
G_t~\doteq~R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots~=~\sum_{k=0}^\infty\gamma^kR_{t+k+1}
$$
其中，折扣因子的范围为$0\leq\gamma\leq1$，当$\gamma$趋向于0时，代表着当前的agent更加的短期主义（myopic），而当$\gamma$趋向于1时，代表着当前的agent更加长期主义（farsighted）。折扣因子还可以表示我们对未来的不确定度，为了尽可能减少对于当前状态回报的方差，我们需要让不确定度大的更加未来的奖励占有更少的权重。容易证明，回报还满足以下递归关系
$$
\begin{aligned}
G_{t}& \doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\cdots   \\
&=R_{t+1}+\gamma\big(R_{t+2}+\gamma R_{t+3}+\gamma^2R_{t+4}+\cdots\big) \\
&=R_{t+1}+\gamma G_{t+1}
\end{aligned}
$$
一定注意，$G_t$是一个预想的概念，并没有实际发生，所以他与$S_t$和选择$A_t$是处于同一时期的，发生在返回奖励$R_{t + 1}$之前。

episodic tasks和continuing tasks不是绝对的，在某一个问题中，以不同的角度看待可能会得到不同的关于episode还是continuing的结果，例如

![image-20230924201732128](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230924201732128.png)

这个例子要移动小车使得小车上面的杆子永远不会掉下来，如果你视杆子掉下来的那一刻为final time，则这个任务是episodic task，你可以在杆子没掉下来的时候不断地基于奖励+1；如果你认为杆子永远不会掉下来，那么此时这个任务视为continuing task，你可以设计杆子掉下来的那一刻奖励为-1，并且设置折扣因子$\gamma$，那么初始状态得到的回报为$-\gamma^{K}$（这个$K$先不管了吧。。。）。

 continuing tasks和episodic tasks有相似但又不同的回报表达式，我们可以将这俩统一起来，如下
$$
G_t \doteq \sum\limits_{k = t + 1}^T \gamma^{k - t - 1} R_k
$$
对于episodic tasks，$\gamma = 1$；对于continuing tasks，$T = \infin$。

### 3.5 Policies and Value Functions

**Policy**
策略（policy）就是从状态空间中的任一状态映射到动作空间的概率分布的函数，也即在某个状态（包含所有的状态空间）下，选择某个动作的概率。此时，我们视选择的动作为随机变量$A_t$，我们在某个状态$S_t = s$的条件下，选择的动作服从条件概率分布$\pi(a\mid s)$。我们将对于所有状态下选择某个动作（随机变量）的概率分布统一标记为$\pi$，而强化学习的任务就是在不断的试验中得到最好的policy。

**Exercise 3.11**
首先补充全期望公式
$$
P(X\mid Y) = \sum\limits_{z}P(X, Z \mid Y)
$$
即把某个一维概率分布函数转化为某个二维概率分布函数的边缘函数，由此可引入一个新的随机变量$Z$，从而于已知条件扯上关系。

如果当前状态为$S_{t}$，求$R_{t+1}$的期望，用$\pi$和四变量的$p$来表示。

对于$p(s', r \mid s, a)$来说，$r$和$a$在本题中都是随机变量，$s$永远是条件，而$s'$不会出现，故
$$
p(r \mid s, a) = \sum\limits_{s'}p(s', r \mid s, a) \\
p(r, a \mid s) = p(a \mid s) p(r \mid s, a) = \pi(a\mid s)\sum\limits_{s'}p(s', r \mid s, a) \\
\mathbb{E}\big[ R_{t + 1} \mid S_{t} = s \big] = \sum\limits_{s', ~a, ~r} r \cdot\pi(a\mid s) \cdot p(s', r \mid s, a)
$$
**State-value function**
其更关心在指定状态下，所有动作在一起共同的回报期望
$$
v_{\pi}(s)\doteq\mathbb{E}_{\pi}[G_{t}\mid S_{t}=s]=\mathbb{E}_{\pi}\bigg[\sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}\bigg|S_{t}=s\bigg]
$$
**Action-value Function**
在state-value function的基础上，其更关心在某个动作和状态都给定的情况下回报的期望
$$
q_\pi(s,a)\doteq\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]=\mathbb{E}_\pi\left[\sum_{k=0}^\infty\gamma^kR_{t+k+1}~\Bigg|S_t=s,A_t=a\Bigg]\right.
$$
**Other Forms**
后面都在推导bellman equation，在此之前，先将某个策略的平均回报扩展至其他不同形式
$$
\eta(\pi) = \sum\limits_{s_0} p(s_0) v_{\pi}(s_0)
$$
这个很好理解。不失一般性，假设初始状态必为$s_0$，则有$\eta(\pi) = v_{\pi}(s_0)$
$$
\begin{aligned}
v_{\pi}(s_0) &= \mathbb{E}_{\pi}\bigg[\sum_{k=0}^{\infty}\gamma^{k}R_{k+1}\mid S_{0}=s_0\bigg] \\
&= \sum_{k=0}^{\infty}\gamma^{k}\mathbb{E}\big[R_{k+1} \mid S_0 = s_0\big]
\end{aligned}
$$
由此，单独内部奖励的均值
$$
\begin{aligned}
\mathbb{E}\big[R_{k+1} \mid S_0 = s_0\big] &= \sum\limits_s\Pr\{s_0 \to s, k, \pi\}\sum\limits_a \pi(a \mid s) \sum\limits_{s'} p(s' \mid s, a) r(s, a, s') \\
&= \sum\limits_s\Pr\{s_0 \to s, k, \pi\}\sum\limits_a \pi(a \mid s) r(s, a) \\
&= \sum\limits_s\Pr\{s_0 \to s, k, \pi\} r(s)
\end{aligned}
$$
故
$$
\begin{aligned}
v_{\pi}(s_0) &= \sum_{k=0}^{\infty}\gamma^{k} \sum\limits_s\Pr\{s_0 \to s, k, \pi\} r(s) \\
&= \mathbb{E}\big[\sum\limits_{k = 0}^{\infty} \gamma^k r(s_k)\big]
\end{aligned}
$$
也即**初始状态$s_0$的价值函数可视为每步所获得的平均奖励之和**。

进一步对其拓展，若令$\tau$为某一特定特定轨迹，而$p(\tau)$就是该轨迹出现的概率，则初始状态$s_0$的价值函数还可视为
$$
v_{\pi}(s_0) = \sum\limits_{\tau} p(\tau) G(\tau)
$$
具体做法就是将$\Pr\{s_0 \to s, k, \pi\}$转化为具体轨迹的形式，这里不详细展开了。

综上所述，这里以三种视角看待了某个策略$\pi$下的平均回报。首先，定义了策略$\pi$下的回报为$\eta(\pi)$，显然其和初始状态的价值函数具有简单而密切的关系；其次，不失一般性，固定将初始状态表示为$s_0$，由此通过$v_{\pi}(s_0)$的定义式，可获得其第一种视角，也即沿着某个特定的状态奖励路线，累积其所获得的奖励，可以将其视为最细节的视角；将上一步继续拓展，可推导得到平均回报就是每个特定步数可到达的所有状态的平均奖励之和，得到第二种视角，该视角相比第一种要从更加宏观的角度观察，关注的是某个状态的平均奖励；最后第三种视角，可轻易推广至不同轨迹的回报以及其对应的出现概率的加权和，这种视角最为宏观，其关注的是轨迹整体。

**Formula 1**
$v_{\pi}(s)$和$q(s, a)$的关系依靠策略分布函数联系
$$
v_{\pi}(s) = \sum\limits_{a}q_{\pi}(s, a) \cdot \pi(a|s)
$$
$v_{\pi}(s)$是基于当前状态对未来回报的期望，其包含了动作的平均；而$q_{\pi}(s,a)$不仅包含状态，还包含了某个具体的动作，显然$v_{\pi}(s)$是$q_{\pi}(s, a)$关于$a$的加权平均。

**Formula 2**
$q_{\pi}(s, a)$和后继状态$v_{\pi}(s')$的关系依靠转移函数联系
$$
\begin{array}{rcl}
q_{\pi}(s, a) &=& \mathbb{E}_\pi[G_t\mid S_t=s,A_t=a] \\
&=&\mathbb{E}_{\pi}[R_{t + 1} + \gamma G_{t + 1} \mid S_t = s, A_t = a] \\
&=&\mathbb{E}_{\pi}[R_{t + 1} \mid S_t = s, A_t = a] + \gamma ~ \mathbb{E}[G_{t + 1} \mid S_t = s, A_t = a] \\ 
\end{array}
$$
易知
$$
\mathbb{E}_{\pi}[R_{t + 1} \mid S_t = s, A_t = a] = \sum\limits_{s', ~r}r~p(s', r \mid s, a) \\
\mathbb{E}_{\pi}[G_{t + 1} \mid S_t = s, A_t = a] = \sum\limits_{g_{t + 1}}g_{t+1}~p(g_{t + 1} \mid s, a) \\
\begin{array}{rcl}
p(g_{t + 1} \mid s, a) &=& \sum\limits_{s'}p(g_{t + 1}, s' \mid s, a ) \\
&=& \sum\limits_{s'} p(s' \mid s, a) p(g_{t + 1} \mid s', s, a)
\end{array}
$$
由于Markov property，$A_t = a$和$S_t = s$同时被忽略
$$
\begin{array}{rcl}
p(g_{t + 1} \mid s, a) &=& \sum\limits_{s'} p(s' \mid s, a) p(g_{t + 1} \mid s', s, a) \\
&=& \sum\limits_{s'} p(s' \mid s, a) p(g_{t + 1} \mid s') \\
&=& \sum\limits_{s', ~r} p(s', r \mid s, a) p(g_{t + 1} \mid s') \\
\end{array}
$$
故
$$
\begin{array}{rcl}
\mathbb{E}_{\pi}[G_{t + 1} \mid S_t = s, A_t = a] &=& \sum\limits_{s', ~r} p(s', r \mid s, a) \mathbb{E}[G_{t + 1} \mid S_{t + 1} = s'] \\
&=& \sum\limits_{s', ~r} p(s', r \mid s, a) v_{\pi}(s')
\end{array}
$$
最终得到
$$
\begin{array}{rcl}
q_{\pi}(s, a) &=&\mathbb{E}_{\pi}[R_{t + 1} \mid S_t = s, A_t = a] + \gamma ~ \mathbb{E}[G_{t + 1} \mid S_t = s, A_t = a] \\ 
&=& \sum\limits_{s', ~r}r~p(s', r \mid s, a) + \gamma\sum\limits_{s', ~r} p(s', r \mid s, a) v_{\pi}(s') \\
&=& \sum\limits_{s', ~r} p(s', r \mid s, a)(r + \gamma~ v_{\pi}(s'))
\end{array}
$$
由formula 1和formula 2我们最终可以得到以下关于$v_{\pi}(s)$的递归关系，这个递归关系就是关于$v_{\pi}$的bellman方程，他描述了当前的状态可以由后继状态来表示。
$$
v_{\pi}(s) = \sum\limits_{a} \pi(a \mid s) \sum\limits_{s', r} p(s', r \mid s, a)(r + \gamma~ v_{\pi}(s'))\\
$$
同理，关于$q_{\pi}(s, a)$的bellman方程如下
$$
q_{\pi}(s, a) = \sum_{s', r}p(s', r \mid s, a)(r + \gamma \sum_{a'}\pi(a' \mid s')q_{\pi}(s', a'))
$$

常用回溯图（backup diagrams）来描述MDP，其中，空心节点表示状态-动作选择分支，实心节点表示动作-状态转移分支。

![image-20230925162429349](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230925162429349.png)

注意看从上往下的顺序依次为$v_{\pi}(s)$、$q_{\pi}(s, a)$和$v_{\pi}(s')$，他们之间相邻的关系由formula 1和formula 2来表示，formula 1中以策略分布来联系；formula 2以状态转移来维系。由此可引出关于$v_{\pi}(s)$和$q_{\pi}(s, a)$的递归表达式，也即bellman方程。由于bellman方程必须同时使用formula 1和formula 2，所以必然会同时牵扯到策略分布函数和状态转移函数。而bellman optimal方程就是将策略分布函数转化为确定化（分布函数变成冲激）。 

**Exercise 3.14**
如下图，若已知灰色部分的state-value，那么中间蓝色的state-value为多少？

![image-20230928174842836](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230928174842836.png)

根据关于$v_{\pi}(s)$的贝尔曼方程
$$
v_{\pi}(s) = \sum_{a}\pi(a \mid s) \sum_{s', r}p(s', r \mid s, a)(r + \gamma v_{\pi}(s'))
$$
由于其向上下左右的奖励皆为0，故可约去加和前的第一项，且此时不再包含$r$项，即
$$
v_{\pi}(s) = \sum_{a}\pi(a \mid s) \sum_{s'}p(s' \mid s, a)\gamma v_{\pi}(s')
$$
对于$p(s' \mid s, a)$来说，其在选择固定（向上下左右）的情况下，前往某个状态是确知的，故概率皆为1
$$
v_{\pi}(s) = \frac{1}{4}(0.7  -0.4 + 0.4 + 2.3) \cdot 0.9 = 0.7
$$
但是可以完全不用按照这个公式，而是对bellman方程的理解来计算。bellman方程的本质是对从当前阶段到下一阶段得到的奖励期望和下一阶段的价值期望之和得到，由于本例中奖励为0，故直接求出后继所有状态的价值的均值即可。

**Bellman Equation**
前面对于bellman方程的描述更加偏重于数学推导，以下辅助以回溯图的方式对bellman方程进行推导、理解和记忆。

贝尔曼方程的基本形式是递归，用所有后继状态计算得到当前状态的value，该递归方程的核心在于将当前的收益分解为动作选择后的奖励和所有后继状态收益的期望和。所以关于$q_{\pi}$的方程如下
$$
q_{\pi}(s, a) = \sum\limits_{r}r~p(r \mid s, a) + \gamma\sum\limits_{s'}p(s' \mid s, a)\sum\limits_{a'}\pi(a' \mid s')q_{\pi}(s', a')
$$


![image-20231003104427036](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231003104427036.png)

再例如下回溯图，对于某一对$s$和$a$，那么下一时刻得到的奖励的期望可以很轻易算出，见下式的左边；同时对于所有可到达的$s'$，同样也要算出其状态的期望，见下式的右边
$$
\begin{array}{rcl}
q_{\pi}(s, a)&=&\mathbb{E}[R_{t + 1} \mid S_t = s, A_t = a] + \gamma\mathbb{E}[G_{t + 1} \mid S_t = s, A_t = a] \\
&=&\sum\limits_{r}p(r \mid s, a)\cdot r + \gamma \sum\limits_{s'}p(s' \mid s, a)v_{\pi}(s') \\
&=& r(s, a) + \gamma \sum\limits_{s'}p(s' \mid s, a) v_{\pi}(s') \\
&=& \sum\limits_{s'}p(s' \mid s, a)(r(s, a, s') + \gamma v_{\pi}(s'))
\end{array}
$$


![image-20230928184057813](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230928184057813.png)

### 3.6 Optimal Policies and Optimal Value Functions

**Optimal Policies**
定义一个策略$\pi$要好于另外一个策略$\pi’$，则需要让$v_{\pi}(s) \geq v_{\pi'}(s)$对于状态空间中的所有状态$s$都成立。也即一个较好的策略，要让所有的状态下的状态价值函数$v_{\pi}$都要较大。那么optimal policies就是所有策略中最大的那个策略，即
$$
v_*(s)\doteq\max_\pi v_\pi(s)
$$
那么对于状态价值$v_{\pi}$最佳的策略$\pi_{*}$是否也能使得$q_{\pi}$最大呢？从定性的角度判断，$q_{\pi}$是在$v_{\pi}$的基础上增加了一个确定性的选择$a$，进而对所有后继状态进行期望计算而得。那么若选择了optimal policies，则在进行所有后继状态进行期望计算的时候就会选择所有后继状态$s'$的$v_{*}(s')$，由于$v_{*}$是最大的，而对于$q_{*}$来说，最佳策略仅仅只对$v_{*}$进行影响，所以$q_{*}$也是策略所有$q_{\pi}$中最大的。从以下式子理解
$$
q_{*}(s, a) = \mathbb{E}[R_{t + 1} + \gamma v_{*}(S_{t + 1}) \mid S_t = s, A_t = a]
$$
更简单来说，上式中，$q_*$可以视为关于$v_*$的单调函数，所以$v_*$是最大的，自然$q_*$也是最大的。

**Bellman Optimality Equation**
**直觉上，$v_*(s)$是选择了当前状态的最佳动作的$q_*(s, a)$（反证法可证明）。**并且此时的策略分布函数一定是个冲激函数（假设最优策略具有唯一性），则有
$$
\begin{array}{rcl}
v_{*}(s)&=& \max\limits_{a}q_{*}(s,a)  \\
&=&\max\limits_{a}\mathbb{E}_{\pi_*}[G_t\mid S_t=s,A_t=a] \\
&=&\max\limits_a\mathbb{E}_{\pi_*}[R_{t+1}+\gamma G_{t+1}\mid S_t{=}s,A_t{=}a] \\
&=&\max\limits_a\mathbb{E}[R_{t+1}+\gamma v_*(S_{t+1})\mid S_t=s,A_t=a] \\
&=&\max\limits_a\sum_\limits{s^{\prime},r}p(s^{\prime},r|s,a)\big[r+\gamma v_*(s^{\prime})\big]
\end{array}
$$
同理，对于最优动作价值函数，由于已经确定了当前状态为$s$和选择的动作$a$，所以第一项所获得的奖励$r(s, a)$保持不变。**最优动作价值函数的最优体现在在后继状态$s'$中选择了最优状态价值$v_{*}(s')$，也即在每个后继状态$s'$中选择了使得当前状态价值最大的动作。**如下
$$
\begin{array}{rcl}
q_*(s,a) &=&\mathbb{E}\Big[R_{t+1}+\gamma\max\limits_{a'}q_*(S_{t+1},a')\mid S_t=s,A_t=a\Big]\ \\
&=&\sum_\limits{s^{\prime},r}p(s^{\prime},r\mid s,a)\Big[r+\gamma\max\limits_{a^{\prime}}q_*(s^{\prime},a^{\prime})\Big]
\end{array}
$$
上两个式子便是分别关于$v_*$和$q_*$的贝尔曼最佳方程。其形式上在于遇到需要计算关于$A_t$的期望$\sum\limits_a \pi(a \mid s)$的时候（即在任何选择动作的时候）都使用了$\pi_*$规定下的最佳动作$\max_a$（greedy策略），以下回溯图展示了这一点


![image-20231003125850582](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231003125850582.png)

对于有限MDP来说（Finite Markov Decision Processes），bellman optimal equation本质上是一组非线性方程组（bellman equation是线性方程组，由此其可写为矩阵的形式），有多少组状态就有多少个方程，同时也有多少个未知量$v_*$或$q_*$，原则上我们能够解出方程组中的唯一解。

一旦我们解出了方程组的解，例如$v_*$，那么我们可以通过这个解来找到最佳策略。根据$v_*$找到策略的方法是：对于每一个状态$s$，我们只要能够找到使得下一个状态的$v_*(s')$最大的动作即可（one-step search，一步搜索），这个动作可能有一个或者多个，也就是说，只要让$\pi(a \mid s)$中所有最优动作的概率不为0，就是最佳策略。简单来说，就是做一次试验，然后检查结果，并记录结果最好的那次试验。

同样的，如果我们解出来的是$q_*$，那么得到最佳策略的方式更为简单，此时甚至不再需要一步搜索，对于某个状态$s$，只需要找到一个或多个最佳的动作$a$，使得$q_{\pi}(s, a)$最大即可。相比于知道$v_*$来得到最佳策略，这种方式能够更加简单且不需要知道系统的转移函数$p$（环境的动态信息）来得到最佳策略（因为不需要预演一步搜索的情况）。

使用$q_*$或者$v_*$得到最佳策略$\pi_*$最大的好处就是局部最优即是全局最优，即你只需要关注下一个状态的状态价值函数$v_{*}(s')$或者当前状态下不同动作的动作价值函数$q_*(s, a)$是否最大即可，不用考虑将来的奖励。因为$v_*$和$q_*$本身已经包含了未来所有的奖励信息。即在这里使用贪心，得到的解就是最优解。

通过解贝尔曼最优方程组来找到最佳策略是一个方法，但是其必须满足以下的条件才有可行性：

- 提前知道转移函数$p$，也即知晓了环境动态特性；
- 系统具有有马尔可夫性（能使用MDP的系统都行）；
- 要有足够的算力（因为解决这种非线性方程组通常是使用暴力的方法）。

**Exercise 3.25**
Given an equation for $v_*$ in terms of $q_*$.
$$
v_*(s)=\max\limits_{a}q_*(s,a)\quad\text{for all}~s\in\mathcal{S}
$$
**Exercise 3.26**
Give an equation for $q_*$ in terms of $v_*$ and the four-argument $p$.
$$
q_*(s,a) = \sum_{s', r}p(s',r\:|s,a)\big(r+\gamma v_*(s')\big)\quad\text{for all}\:s\in\mathcal{S},a\in\mathcal{A}(s)
$$
**Exercise 3.27**
Give an equation for $\pi_*$ in terms of $q_*$.
$$
\pi_*(s)=\operatorname*{argmax}_{a}q_*(s,a)\quad\text{for all}\:s\in\mathcal{S}
$$

**Exercise 3.28**
Give an equation for $\pi_*$ in terms of $v_*$ and the four-argument $p$.
$$
\pi_*(s)=\underset{a}{\operatorname*{argmax}}\sum_{s', r}p(s',r|s,a)\big(r+\gamma v_*(s')\big)\quad\text{for all}\:s\in\mathcal{S}
$$
该式子也即前面所提到的根据$v_{*}$进行一步搜索，从而得到最优策略$\pi_*$。

**Exercise 3.29**
Rewrite the four Bellman equations for the four value functions ($v_{\pi}$, $v_*$, $q_{\pi}$, and $q_*$) in terms of the three argument function $p(s' \mid s, a)$ and the two-argument function $r(s, a)$.
$$
\begin{aligned}
v_\pi(s)&=\sum_{a}\pi(a|s)\left(r(s,a)+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')\right)\quad\text{for all}\:s\in\mathcal{S} \\
v_*(s)&=\max\limits_{a}\left(r(s,a)+\gamma\sum_{s'}p(s'|s,a)v_*(s')\right)\quad\text{for all}\:s\in\mathcal{S} \\
q_{\pi}(s,a)&=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\sum_{a'}\pi(a'|s')q_{\pi}(s',a')\quad\text{for all}\:s\in\mathcal{S},a\in\mathcal{A}(s) \\
q_*(s,a)&=r(s,a)+\gamma\sum_{s'}p(s'|s,a)\max\limits_{a'}q_*(s',a')\quad\text{for all}\:s\in\mathcal{S},a\in\mathcal{A}(s)
\end{aligned}
$$
务必根据回溯图记忆，这样才能真正理解公式的意义。
