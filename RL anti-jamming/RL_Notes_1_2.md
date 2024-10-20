## Chapter 1 Introduction

### 1.1 & 1.2

Feature of reinforcement learning: 

- trial-and-error search
- delayed reward.

RL is problem, solution & field.

A learning agent must be able to:

- sense the state of its environment;
- take actions that affect the state; 
- have a goal. 

​	Markov decision processes include three factors.

Supervised learning: 

- from training set of labeled examples;
- to identify a category (generalize);
- but not adequate for learning from interaction. 

Unsupervised learning: 

- finding structure hidden in collections of unlabeled data; 
- RL is to maximize a reward signal instead of finding structure. 

One challenge of RL: 
the trade-off between **exploration and exploitation**.

- trying past actions which are effective means high reward; 
- but we should explore in order to make better action selection in the future. 

Another feature: *consider the whole problem of a goal-directed agent interacting with an uncertain environment.*

Interactive, goal-seeking agent can also be a component of a larger behaving system.

The agent's actions are permitted to affect the future state of the environment, thereby correct actions require taking into account indirect, delayed consequences of actions.
The agent can judge progress toward its goal based on what is sense directly.

### 1.3 Elements of Reinforcement Learning

Four main elements of a RL system is a policy, a reward, a value function and a model.

A policy:

- it is a mapping from perceived states of the environment to actions to be taken when in those states; 
- it defines the behavior of an agent when in some particular states; 
- it may be stochastic, specifying probabilities for each action. 

A reward signal: 

- the agent will receive a reward from environment after a action; 
- it indicates what is good in an immediate sense; 

- the agent's sole objective is to maximize the reward over a long run; 
- the reward signal will change the policy, low reward means changing policy next time;
- reward signals may be stochastic functions of the state of the environment and the action taken.

A value function: 

- it indicates what is good in the long run; 
- value of a state is the total amount of reward an agent can expect to accumulate over the future; 
- we most concerned about the value, actions choices are made based on value judgements; 
- a method for efficiently estimating values is the most important in RL algorithms.

A model of the environment: 

- this allows inferences to be made about how the environment will behave; 
- models are used for planning, when using models, we call model-based methods as opposed to simpler mode-free which are explicitly trial-and-error.

## Chapter 2 Multi-armed Bandits

### 2.2 Action-value Methods & 2.3 The 10-armed Testbed

Action-value方法即为只有action和其返回的reward而形成的value，没有环境状态等其他的信息，所以其实RL中最为简单的模型，其也更加依赖于估计预测值的方法。

**Sample-Average Method**
在一般的情况下，我们所选择某action后反馈的reward并不是唯一确定的，其带有的随即特性使得我们可以将其视为随机变量，进而可用PDF来描述。我们定义该随机变量的期望就是其真值，即
$$
q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]
$$
所以根据统计的相关知识，我们可以通过多次的测量值来估计真值，即
$$
Q_t(a)\doteq\frac{\text{sum of rewards when }a\text{ taken prior to }t}{\text{number of times }a\text{ taken prior to }t}=\frac{\sum_{i=1}^{t-1}R_i\cdot\mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1}\mathbb{1}_{A_i=a}}
$$
这里的$\mathbb{1}_{A_i = a}$表示当下标$A_i = a$为真时，该值为1，否则为0。故上式所描述的事情为当时间为$1$到$t-1$不断增加（就是状态不断地再递进）时，当期间的某个状态时所选择的action为a时，更新该action的预测值为新增加测量值后的平均值。

**Greedy Actions**
一个最简单的action-value方法即为再每回的动作选择中，都选择具有最大estimation的action，然后再用sample-average方法更新estimation供下回的动作选择使用。表达式即为
$$
A_t\doteq\underset{a}{\operatorname*{\arg\max}}Q_t(a)
$$
其中的$\underset{a}{\operatorname*{\arg\max}}$表示选择具有最大值的$a$，进而带入后面的表示式，最终为$Q_t(a)$。

注意该方法并不是在最开始随机选择一个动作，然后之后就一直选择该动作。若是我们统一设置所有动作的default值为所有动作真值的均值，那么第一回的选择可能会返回一个小于该均值的反馈值。根据sample-average方法，我们在下一回选择前对当前的action进行更新，此时该动作的预测值会比其他动作的预测值会更低，所以再下一回的选择中，我们会更换为更高预测值的动作，从而实现了动作的更换。

**$\varepsilon$-greedy Methods**
在greedy actions的基础上，我们增加一个参数$\varepsilon$，使得每回的动作选择中，都会有概率$\varepsilon$不进行greedy的策略，转而选择与每个动作预测值独立的其他所有动作（包括现在最佳的actions）。显而易见，随着选择次数的增加，最终所有action的预测值$Q_t(a)$都会趋近于真值$q_*(a)$。

为了检验$\varepsilon$-greedy方法的有效性，设计实验如下：随机生成2000回的10-armed bandit，对于每个bandit，在$t$时刻中选择某个动作$A_t$，其反馈的真值$q_*(A_t)$（真值是反馈值的期望）服从标准高斯分布。再次基础上，动作$A_t$反馈的值为均值$q_*(A_t)$，方差为1的高斯分布。也就是说，某次动作所返回的值包含两个随机事件，即均值服从标准高斯分布，方差为1的高斯分布。再每回测验中，使用greedy方法、$\varepsilon$为0.1的$\varepsilon$-greedy方法和$\varepsilon$为0.01的$\varepsilon$-greedy方法。实验结果如下

![image-20230915102752146](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230915102752146.png)

![image-20230915102806305](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230915102806305.png)

上面的三张图中，第一张图表示某回10-armed bandit中每个动作反馈值的图例；第二张图和第三张图均为测试结果

- 前者为随着步数的增加，每步所获得的平均奖励的图例。由于三种方法均属于greedy或是其变种，所以最开始的几步具有几乎重合的曲线。但是随着步数的增加，greedy方法的缺点就展露无遗，他会陷入一个次好的怪圈而再也跳不出来。反观$\varepsilon$-greedy的两种策略
- 后者为随着步数的增加，该步所能选择到最佳action的概率。最开始，由于缺少对于所有actions的先验知识，所以选到Optimal action的概率相对较少，但是随着步数的增加，探索不断增加（即使是greedy方法，也会在刚开始的时候进行探索），找到Optimal action的概率不断升高，所以曲线不断升高。其核心在于随着步数的增加，探索的成分也在不断地增加。

对于上述的实验结果，有以下几个结论

- $\varepsilon$从某种意义上可以看作是RL中exploration-exploitation中exploration中的比重，$\varepsilon$越大，表示探索得越多。所以对于第一幅平均奖励得图示来说，$\varepsilon$越大（即探索的越多），自然找到最佳action的速度最快，收敛到平稳的平均奖励（指图示中平均奖励增长缓慢的那段曲线）更快；
- 但是由于算法本身的限制，其在到达平稳平均奖励时，仍然会进行探索，从而导致此时仍有机会不能选到最佳action，最终平稳时的平均奖励会更低。

书上题目有一个问题，当$\varepsilon$-greedy长时间地运行后最终稳定的平均奖励为多少，简单的思路见iPad上的文档

*P28 This of course implies that the probability of selecting the optimal action converges to greater than $1 - \varepsilon$?*

### 2.4 Incremental Implementation

本节将通过前面所使用到的sample-average求估计值的方法推广至一般形式，推导如下
$$
\begin{array}{rcl}
Q_{n+1} &=& \frac{1}{n}\sum_{i=1}^nR_i \\
&=&\frac1n\left(R_n+\sum_{i=1}^{n-1}R_i\right) \\
&=&\frac1n\left(R_n+(n-1)\frac1{n-1}\sum_{i=1}^{n-1}R_i\right) \\
&=&\frac1n\Big(R_n+(n-1)Q_n\Big) \\
&=&Q_n+\frac1n\Big[R_n-Q_n\Big]
\end{array}
$$
该形式完全符合
$$
\text{NewEstimate}\leftarrow \text{OldEstimate}+\text{StepSize}\Big[\text{Target}-\text{OldEstimate}\Big]
$$
的形式，该形式是极为重要的形式，我们可以通过改变该公式的某些参数，从而改变对于预测值的估计性能。比如$\text{StepSize}$取为$1 / n$且$\text{Target}$取为第$n$步所选择的action后所获得的奖励$R_n$，那么预测值估计的递推式就完全和sample-average一模一样。

### 2.5 Tracking a Nonstationary Problem

对于一个平稳的bandit problem，选择sample-average来作为估计预测值的方法是十分合理的，因为此时actions后的奖励不会随着时间的推移而发生变化，在一个长时间的过程中，我们最终可以采用这个方法来获得该action的奖励真值$Q_t(a) \rightarrow q_*(a)$。

若让$\alpha_n(a)$表示获得奖励$R_n$后更新预测值时所用的$\text{StepSize}$，数学可以证明，当$\alpha_n(a)$满足以下关系时，预测值$Q_t(a)$会最终收敛到$q_*(a)$
$$
\sum\limits_{n=1}^\infty\alpha_n(a)=\infty\quad\text{and}\quad\sum\limits_{n=1}^\infty\alpha_n^2(a)<\infty
$$
所以一个典型的满足上面两个条件的$\text{StepSize}$即为$\alpha_n(a) = 1 / n$。但是收敛速度会很慢（书上说的）。

*The first condition is required to guarantee that the steps are large enough to eventually overcome any initial conditions or random fluctuations. The second condition guarantees that eventually the steps become small enough to assure convergence.*
*如何证明这个结论呢？*

**Exponential Recency-weighted average**
对于更为常见的非平稳bandit problem，再使用sample-average去更新预测值就不那么合适了，因为sample-average使得每次的奖励$R_i$最后都同等地贡献于预测值。非平稳的奖励使得我们需要提高预测值中近期奖励的权重，我们可以通过将$\text{StepSize}$更改为常数来实现这一操作，即此时的$\text{StepSize}$为常数$\alpha$，此时$Q_n$的递推式变为
$$
Q_{n + 1} = Q_n + \alpha \big[R_n - Q_n\big]
$$
继续对这一式子进行推导
$$
\begin{array}{rcl}Q_{n+1}&=&Q_n+\alpha\Big[R_n-Q_n\Big]\\
&=&\alpha R_n+(1-\alpha)Q_n\\
&=&\alpha R_n+(1-\alpha)\left[\alpha R_{n-1}+(1-\alpha)Q_{n-1}\right]\\
&=&\alpha R_n+(1-\alpha)\alpha R_{n-1}+(1-\alpha)^2Q_{n-1}\\
&=&\alpha R_n+(1-\alpha)\alpha R_{n-1}+(1-\alpha)^2\alpha R_{n-2}+\\ 
&& ~~~~ \cdots+(1-\alpha)^{n-1}\alpha R_1+(1-\alpha)^nQ_1\\
&=&(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i
\end{array}
$$
可以看到上式最终变为了加权平均的形式，之所以是加权平均，这是因为$(1 - \alpha)^n + \sum_{i = 1}^n \alpha (1 - \alpha)^{n - i} = 1$，其满足归一化性质。根据这个式子，可以看到随着$i$的减小，其对于$R_i$的权重$\alpha (1 - \alpha)^{n - i}$指数级减小，意味着预测值越来越由临近的奖励来决定。

当$\alpha = 1$时，表达式变为
$$
\begin{array}{rcl}
Q_{n + 1} &=& (1 - \alpha)^n Q_1 + \sum_{i = 1}^n \alpha(1 - \alpha)^{n - i} R_i \\
&=& 0^n \cdot Q_1 + \sum_{i = 1}^n 1 \cdot 0^{n - i} \cdot R_i \\
&=& R_n
\end{array}
$$
即此时预测值只取决于最近的奖励$R_n$。

**Exercise 2.4**
一般式中$\text{StepSize}$不为常数，而随着步数改变，记为$\alpha_k$，即状态为$Q_k$时，选择某action得到奖励为$R_k$，从而更新到$Q_{k + 1}$。上述仍为加权平均的形式，求出表达式。
$$
\begin{array}{rcl}
Q_{n + 1} &=& Q_n + \alpha_k\big[R_n - Q_n\big] \\
&=& \alpha_n R_n + (1 - \alpha_n) Q_n \\
&=& \alpha_n R_n + (1 - \alpha_n) [Q_{n - 1} + \alpha_{n - 1}(R_{n - 1} - Q_{n - 1})] \\
&=& \alpha_n R_n + \alpha_{n - 1}(1 - \alpha_n)R_{n - 1} + (1 - \alpha_{n - 1})(1 - \alpha_n)Q_{n - 1} \\
&=& \alpha_n R_n + \sum\limits_{i = 1}^{n - 1} \alpha_i R_i \big[\prod\limits_{j = i + 1}^{n}(1 - \alpha_j)\big] + \prod\limits_{k = 1}^{n}(1 - \alpha_k)Q_1
\end{array}
$$

### 2.6 Optimistic Initial Values

对初始值的讨论前面鲜少进行，实际上，之前的所有估计预测值的方法都在某种程度上依赖于初始值的确定，其均是有偏估计。只是，对于sample-average来说，其的有偏性在选择了action之后立刻就消失了。而对于常数$StepSize$来说，随着时间的推移，初始值$Q_1(a)$的权重会不断地减小。

但是书上说有偏性并不会带来什么缺点，有时反倒是个优点。但是，其仍旧有个最大的缺点，若不将所有初始值简单地设置为0，那么这些初始值就必须有我们确定，有时这也方便我们将先验信息方便地给予系统。

将初始值设置得偏大可以鼓励最开始exploration的进行，因为偏大的初始值会使得greedy方法将所有的action都选择一遍，也即系统在最初的时候进行了大量的exploration。

![image-20230916123846862](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230916123846862.png)

上图为简单的有关较大初始值的试验，可以看到相比于初始值为0的灰色曲线，初始值为5的蓝色曲线在系统最初的时候性能偏低，这是因为较大的初始值鼓励系统在最开始的时候进行大量的探索，但是由于大量的探索总会包含到Optimal action，所以会有某些步（集中在前k步）选择到Optimal action的概率很高。最终蓝色的曲线的表现会比灰色的曲线更好，这是由于后期探索再不断地减少，前期的大量探索已经是系统获得了大量的信息。

但是初始值设置的只能影响到系统最初的状态，面对需要后面探索的需求就无法进行了，比如说非平稳bandit问题。

**Exercise 2.6**
为什么会有最开始的峰值？
最开始agent经过了大量的exploration，该exploration降低了最开始的Optimistic Initial Value，其中具有越高真值（就是Optimal action）的action降低得越少，所以agent大概率会在k次选择后选择到Optimal action。但是随着选择到Optimal action得次数越来越多，大初值得影响就越来越少，然而其他的action还强烈地被大初值影响着，最终使得Optimal action的预测值要低于其他action，从而导致选择Optimal action比例的下降，进而产生了峰值。

**Exercise 2.7**
在联系2.4的基础上，使得$\alpha_n = \alpha / \bar{o}_n$，其中$\bar{o}_n=\bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1}),\quad\mathrm{for~}n\geq0,\quad\mathrm{with~}\bar{o}_0=0$，证明其为加权平均和，且无偏（与初始值无关）。
我就用我纯自己写的复杂的过程
首先关注$\bar{o}_n$
$$
\begin{array}{rcl}
\bar{o}_n&=&\bar{o}_{n-1}+\alpha(1-\bar{o}_{n-1}) \\
&=& \bar{o}_{n - 1}(1 - \alpha) + \alpha
\end{array}
$$
观察
$$
\bar{o}_0 = 1, \quad\bar{o}_1 = \alpha, \quad \bar{o}_2 = \alpha + \alpha(1 - \alpha), \\
\quad \bar{o}_3 = \alpha + \alpha(1 - \alpha) + \alpha(1 - \alpha)^2 \cdots
$$
故有
$$
\bar{o}_n = \alpha\big[1 + (1 - \alpha) + \cdots + (1 - \alpha)^{n - 1}\big]
$$
则
$$
\alpha_n = \frac{1}{1 + (1 - \alpha) + \cdots + (1 - \alpha)^{n - 1}}
$$
使用Exercise 2.4的结论，即
$$
\begin{array}{rcl}
Q_{n + 1} &=& \alpha_n R_n + \sum\limits_{i = 1}^{n - 1} \alpha_i R_i \big[\prod\limits_{j = i + 1}^{n}(1 - \alpha_j)\big] + \prod\limits_{k = 1}^{n}(1 - \alpha_k)Q_1
\end{array}
$$
可知，当$n = 1$时，$\alpha_1 = 1$，故
$$
\prod\limits_{k = 1}^{n}(1 - \alpha_k)Q_1 = (1 - \alpha_1)\prod\limits_{k = 2}^{n}(1 - \alpha_k)Q_1 = 0
$$
当$i > 0$时
$$
\begin{array}{rcl}
\prod\limits_{j = i + 1}^{n}(1 - \alpha_j) &=& \prod\limits_{j = i + 1}^{n}(1 - \frac{1}{1 + (1 - \alpha) + \cdots + (1 - \alpha)^{j - 1}}) \\
&=& \prod\limits_{j = i + 1}^{n}(1 - \alpha)\frac{1 + \cdots + (1 - \alpha)^{j - 2}}{1 + \cdots + (1 - \alpha)^{j - 1}} \\
&=& (1 - \alpha)^{n - i} \frac{1 + \cdots + (1 - \alpha)^{i - 1}}{1 + \cdots + (1 - \alpha)^{i}} \cdot
\frac{1 + \cdots + (1 - \alpha)^{i}}{1 + \cdots + (1 - \alpha)^{i + 1}} \cdots
\frac{1 + \cdots + (1 - \alpha)^{n - 2}}{1 + \cdots + (1 - \alpha)^{n - 1}} \\
&=& (1 - \alpha)^{n - i} \frac{1 + \cdots + (1 - \alpha)^{i - 1}}{1 + \cdots + (1 - \alpha)^{n - 1}}
\end{array}
$$
故
$$
\begin{array}{rcl}
Q_{n + 1} &=& \alpha_n R_n + \sum\limits_{i = 1}^{n - 1} \alpha_i R_i \big[\prod\limits_{j = i + 1}^{n}(1 - \alpha_j)\big] + \prod\limits_{k = 1}^{n}(1 - \alpha_k)Q_1 \\
&=& \frac{R_n}{1 + (1 - \alpha) + \cdots + (1 - \alpha)^{n - 1}}  + \sum\limits_{i = 1}^{n - 1}R_i \frac{(1 - \alpha)^{n - i}}{1 + (1 - \alpha) + \cdots + (1 - \alpha)^{i - 1}}  \frac{1 + \cdots + (1 - \alpha)^{i - 1}}{1 + \cdots + (1 - \alpha)^{n - 1}} \\
&=& \frac{\sum\limits_{i = 1}^{n}(1 - \alpha)^{n - i}}{1 + \cdots + (1 - \alpha)^{n - 1}} R_i\\
&=& \frac{\sum\limits_{i = 1}^{n}\alpha(1 - \alpha)^{n - i}}{1 - (1 - \alpha)^n} R_i
\end{array}
$$
证毕。

网上的解答（比我的过程更简洁）

[《RLAI》第二版Chapter 2 的 Exercises 2.7 题解]: https://zhuanlan.zhihu.com/p/250369155

### 2.7 Upper-Confidence-Bound Action Selection

标题的中文名称为基于置信度上界的动作选择。前面说到的greedy方法，由于缺少探索，虽然在短期内看起来最好，但是最终容易陷入到次最好的怪圈而跳不出来；$\varepsilon$-greedy方法在greedy的方法之上，强制要求系统进行探索，但是其探索的方法以平等的姿态选择所有actions，不带任何偏好。一个更好的方法是根据每个action所具有的成为最佳动作的潜力来决定是否选择该action，这股潜力可以分为两项，一是关于该action的预测值，具有更高预测值的action显然具有更大的潜力；二是对于该action的置信度。在greedy方法的基础上，其选择action的公式如下
$$
A_t\doteq\underset{a}{\operatorname*{argmax}}\left[Q_t(a)+c\sqrt{\frac{\ln t}{N_t(a)}}\:\right]
$$
相比于greedy方法的选择action公式，其增加了一项有关不确定性的度量，该度量与某action的预测值共同形成了对于该action成为最佳action的潜力上界。其基本思想就是认为置信度越低（即越不确定）的action成为最佳的可能性越大，所以其称为基于置信度上界的动作选择。

具体来看度量置信度的那项，当某个action从未被选中，那么其$N_t(a) = 0$，此时认为其潜力无限。也就是说第一次选择的时候，每个action的置信度都是无穷，正如2.6节所提到的Optimistic Initial Values一样，其鼓励系统最初的大量探索。随着某个action被选择次数的增加，分母项不断增加，置信度也越来越小（不确定度降低）。分子中的$\ln t$随着$t$的增加而增加，之所以取了自然对数，是因为要让其增加速率越来越低。参数$c$决定了置信度在整个潜力中的占比，*开根号不知道为什么，实际经验中调整的？*

同样是适用之前的试验，得到如下图片

![image-20230918120047440](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20230918120047440.png)

在平稳的bandit问题上，UCB的表现要更好。但是UCB还具有和最大初始值相近的问题，就是难以适用在非平稳的问题上。因为UCB和带最大初始值的greedy一样，都只能在早期获得的关于action的信息，而后大概率不再更新。

### 2.8 Gradient Bandit Algorithms

前面所展示的方法都是依赖于agent对action预测值的估计而进行的。现在，我们不依赖于对action的估计，转而对每个action，引入一个新的参数，称之为preference，用$H_t(a)$来表示。其代表着在状态$t$下，对于某个action的偏好程度。注意，应该将$H_t(a)$与奖励$R_t(a)$区分开来，这俩原则上没有直接的关系，$H_t(a)$也并不依赖于$R_t(a)$而存在。在后面的算法中可以看到，$H_t(a)$的更新是依靠$R_t(a)$来进行的。

有了$H_t(a)$的存在，agent之后的选择策略将使用$H_t(a)$来进行。agent选择action时不再像和greedy等方法一样，确定性的选择具有最大的估计预测值$Q_t(a)$的action（即使诸如$\varepsilon$-greedy仍会有随机的元素，但是其本质还是在确定性的选择的基础上上增加的随机因素），而是通过某种概率分布来进行选择，例如某个action的$H_t(a)$比其他更大，那么其概率分布所占比重就大，代表着agent更容易选择到这个动作。我们使用soft-max分布来作为关于action的概率分布，表达式如下
$$
\begin{aligned}\Pr\{A_t{=}a\}&\doteq\frac{e^{H_t(a)}}{\sum_{b=1}^ke^{H_t(b)}}\doteq\pi_t(a)\end{aligned}
$$
使用这个soft-max分布的好处是偏好$H_t(a)$不在是个绝对的概念，其的大小是相对的。比如，让所有的$H_t(a)$同时加上1000，那么改变之后的的soft-max分布得益于指数的加和可分开为乘积的性质，其的概率分布是不变的。
*soft-max分布可能还有很多性质和应用，以后再说。。。*

![gradient1](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\gradient1.jpg)

![gradient2](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\gradient2.jpg)

![gradient3](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\gradient3.jpg)

起初，对于每个动作，其$H_t(a)$皆为0，等价于选择每个动作的概率都是相同的。随后，对于每次的选择$A_t$，都有奖励$R_t$，然后更新$H_t(a)$如下
$$
\begin{aligned}
H_{t+1}(A_t)&\doteq H_t(A_t)+\alpha\big(R_t-\bar{R}_t\big)\big(1-\pi_t(A_t)\big),\quad&\text{and}\\
H_{t+1}(a)&\doteq H_t(a)-\alpha\big(R_t-\bar{R}_t\big)\pi_t(a),\quad&\text{for all }a\neq A_t
\end{aligned}
$$
其中，$\bar{R}_t$是当前时刻之前所有奖励$R_i$的平均。当本次奖励$R_t$的值大于$\bar{R}_t$时，$H_{t + 1}(A_t)$增加，其他动作的$H_{t + 1}$减小，进而改变概率。

书上还有一个关于本方法的试验，懒得写了。。。
