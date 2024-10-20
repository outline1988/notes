## Chapter 6 Temporal-Difference Learning

### 6.1 TD Prediction

总体来说，我们求解最佳策略$\pi_*$的方法都是policy iteration，各种方法例如DP和MC都是基于GPI的变式下进行的。GPI包含evaluation和improvement，其中improvement在不同的方法中都使用贪心算法改进策略（除了MC中的on-policy算法是为了舍弃了exploring start的假设而妥协使用了$\varepsilon$-greedy进行改进），不同方法实现policy iteration的核心在于policy evaluation（也就是prediction problem）。
$$
\begin{aligned}
v_{\pi}(s)& \doteq\mathbb{E}_\pi[G_t\mid S_t{=}s]  \\
&=\mathbb{E}_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_{t}=s]
\end{aligned}
$$
MC方法使用第一行进行evaluation，也即根据大数定理，从模拟生成的episodes中找到多个观测值$G_t$，以平均统计量的方式估计$v_{\pi}(s)$，缺点是我们必须等待episode生成完毕才能知道某刻的$G_t$，并且在使用重要性采样的off-policy中，有绝大多数的episodes（与目标策略有着不同的轨迹）无法派上用场，这些轨迹在重要性采样的场景下，我们难以将其内在的信息挖掘出来；DP方法使用第二行进行evaluation，在完全已知环境动态特性$p$的前提下，$v_{\pi}$作为等式的不动点可以用迭代的方式将初始随机的估计值向真值逼近，在逼近也就是迭代的过程中，DP使用bootstrap方法不断地通过先前的预测信息来得到新的预测信息。

时许差分（Temporal-Difference，TD）主要使用下式进行
$$
\begin{aligned}
v_{\pi}(s) &= \mathbb{E}_{\pi}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_{t}=s] \\
&= \mathbb{E}_{\pi}[R_{t+1}\mid S_{t}=s] + \gamma \mathbb{E}_{\pi}[v_{\pi}(S_{t+1})\mid S_{t}=s]
\end{aligned}
$$
TD方法利用了MC的采样特性，让其能够适应未知动态特性的情况（model-free）；同时，又利用了DP方法自举的特性，利用马上得到的即时奖励和下一时刻状态的价值预测，从而避免了如MC方法一样必须等待一幕的完成才能进行更新的缺点，使其能够在某episode中的每个时间间隔都能进行更新。最简单形式的TD方法用如下式子进行更新
$$
V(S_t)\leftarrow V(S_t)+\alpha\Big[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\Big]
$$
即若某episode进行到了状态$S_t$，则使用下一时刻的即时奖励$R_{t + 1}$和唯一的（同一episode中只有一个后继状态）下一状态$S_{t + 1}$的价值估计量$V(S_{t + 1})$来对当前状态$S_{t}$进行更新。其中的$R_{t + 1}$是使用类似于MC方法采样而来的，后面的$V(S_{t + 1})$用到了DP的自举特性。可以简单理解为TD方法是对MC方法以DP方法自举的特性改进而来的，即MC方法的本质是通过试验得到的整个episodes的轨迹，从而取得某个状态$S_t$的回报$G_{t}$。TD方法同样是希望得到该状态$S_t$的回报$G_t$，只不过其不依赖于整条轨迹，而利用DP方法自举的特性只使用下一时刻的奖励$R_{t + 1}$来更新状态价值，即使用$R_{t+1}+\gamma V(S_{t+1})$来作为MC方法中$G_t$的替代。上述的方法我们称之为TD(0)或者one-step TD。

下图为TD(0)的回溯图，不同于MC的回溯图需要列举所有的状态和动作直到终端状态，TD(0)的回溯图只展示了一个动作和其后继状态，也清楚的表明这该种方法是只利用两个预测值$R_{t + 1}$和$V(S_{t + 1})$来进行更新。

![image-20231016154821760](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231016154821760.png)

对于当前状态$S_{t}$的预测值$V(S_{t})$的更新只是用了一个后继状态价值$V(S_{t + 1})$，区别于DP中使用了所有后继状态的期望更新（expected updates），我们称其为样本更新（sample updates）。

与所有的更新方式相同，我们采用$NewEstimate\leftarrow OldEstimate+StepSize\Big[Target-OldEstimate\Big]$的结构来进行更新，其中括号内的称之为误差，我们定义TD(0)误差为
$$
\delta_t\doteq R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$
TD(0)误差$\delta_t$实际上是在当前时刻$t$就会产生的，但只有我们走到下一个时刻$t + 1$获知后继状态$S_{t + 1}$时才能够正真的得知。

如果在某个episode中，预测值$V$不会更新（例如MC方法，其只在episode完成之后才更新，TD方法就是在episode进行当中更新的），那么Monte Carlo误差可以由TD(0)误差表示
$$
\begin{aligned}
G_{t}-V(S_{t})& =R_{t+1}+\gamma G_{t+1}-V(S_{t})+\gamma V(S_{t+1})-\gamma V(S_{t+1})   \\
&=\delta_t+\gamma\big(G_{t+1}-V(S_{t+1})\big) \\
&=\delta_t+\gamma\delta_{t+1}+\gamma^2\left(G_{t+2}-V(S_{t+2})\right) \\
&=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\big(G_T-V(S_T)\big) \\
&=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\left(0-0\right) \\
&=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
\end{aligned}
$$
其中MC误差来源于使用某次episode的$G_t$来更新对于$V(S_t)$的更新表达式中。

**Exercise 6.1**
推导出预测数组$V_t$会在同一episode不断更新的MC误差表达式
$$
\begin{aligned}
G_t-V_t(S_t)&=R_{t+1}+\gamma G_{t+1}-V_t(S_t)+\gamma V_t(S_{t+1})-\gamma V_t(S_{t+1})  \\
&=\delta_t+\gamma(G_{t+1}-V_t(S_{t+1})) \\
&=\delta_t+\gamma\big(G_{t+1}-V_{t+1}(S_{t+1})\big)+\gamma\big(V_{t+1}(S_{t+1})-V_t(S_{t+1})\big) \\
&=\delta_t+\gamma\delta_{t+1}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\left(G_T-V(S_T)\right)+\sum_{k=t}^{T-1}\gamma^{k-t+1}\big(V_{k+1}(S_{k+1})-V_k(S_{k+1})\big) \\
&=\delta_t+\gamma\delta_{t+1}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}\left(0-0\right)+\sum_{k=t}^{T-1}\gamma^{k-t+1}\left(V_{k+1}(S_{k+1})-V_k(S_{k+1})\right) \\
&=\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k+\sum_{k=t}^{T-1}\gamma^{k-t+1}\left(V_{k+1}(S_{k+1})-V_k(S_{k+1})\right)
\end{aligned}
$$
这个式子是直接引入更新后的预测值数组$V_{k + 1}$来表示，没有使用到更新前后$V$的关系。

直观的理解TD方法相比于MC方法的好处，对于某一个episode，更新某一个状态的价值不一定非得等待该episode完成后，得到了这个状态在本episode的真值再来进行更新，随之而来转入下一时刻得到的即时奖励同样也包含了一部分在该状态价值的有用信息，所以也能够使用即时奖励进行更新。TD方法还有的好处就是其受到的扰动更小，对于MC来说，MC将整个轨迹视为一个整体，某次的更新需要受到该状态之后轨迹上所有状态的影响，该状态价值的方差要受到其后轨迹所有状态方差的累积。而对于TD来说，其只依赖于下一状态的价值和转入下一状态的到的奖励，预测价值所受到的扰动也要更小。TD方法对于局部轨迹有更好的适应性，因为其只根据一段很小轨迹上的信息进行更新，小轨迹意味其更容易被包含在更大的轨迹当中，在一个新的场景中若其中部分与之前类似，则可以使用之前学到过的信息。

### 6.2 Advantages of TD Prediction Methods

TD方法不需要知道环境的动态特性（model free）。

TD是一种online的方法，其不需要如同MC方法一样要等待整个episode的完成再进行更新。并且MC必须要有很多episodes，而TD方法对于continuing tasks也同样胜任。

TD方法的收敛速度相对于MC来说大大提高（最小二乘？）。

**Example 6.2**
考虑如下随机游走过程

![image-20231017174305973](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231017174305973.png)

对于除了终端状态之外的状态，其向左向右的概率是相同的，我们使用MC和TD方法分别估计其各个状态的价值，从而对MC和TD方法进行一个对比。

首先我们计算其真值，由于该MDP过程状态较少，采用解析的方法求得，公式如下
$$
\mathbf{v_{\pi}} = (\mathbf{I} - \gamma \mathbf{P_{\pi}})^{-1} \mathbf{r_{\pi}}
$$
其中
$$
\mathbf{r_{\pi}} = 
\begin{bmatrix}
0 \\
0 \\
0 \\
0 \\
0.5 \\
0 \\
\end{bmatrix}

,\quad

\mathbf{P_{\pi}} = 
\begin{bmatrix}
0 & 0.5 & 0 & 0 & 0 & 0.5 \\
0.5 & 0 & 0.5 & 0 & 0 & 0 \\
0 & 0.5 & 0 & 0.5 & 0 & 0 \\
0 & 0 & 0.5 & 0 & 0.5 & 0 \\
0 & 0 & 0 & 0.5 & 0 & 0.5 \\
0 & 0 & 0 & 0 & 0 & 0 \\

\end{bmatrix}
$$
对于向量$\mathbf{r_{\pi}}$来说，只有再状态E时有0.5的概率转移到终端状态并获得奖励1，所以有且仅有该状态的奖励为0.5。其中矩阵$\mathbf{P_{\pi}}$按顺序分别代表的状态是A至E，最后包含终端状态T（终端状态也要包含）。将所有的终端状态合并为一种即可，终端状态无法转移至其他状态，所以矩阵的最后一行为0，只有状态A和状态E能够转移至终端状态，所以最后一列的第一行和第五行为0.5。

通过矩阵计算，最终解为
$$
\mathbf{v_{\pi}} = 
\begin{bmatrix}
0.1667 \\
0.3333 \\
0.5000 \\
0.6667 \\
0.8333 \\
0 \\
\end{bmatrix}
$$
分别表示状态A至E的真值为$\frac16,\frac26,\frac36,\frac46$和$\frac56$。

使用MC方法和TD方法分别用来预测该MDP各个状态的价值，得到的结果如下

![image-20231017180518639](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231017180518639.png)

左图使用TD(0)方法，每种颜色的线代表TD(0)所使用的episodes数，可以看到，随着episodes的数量增加，其预测的值不断地逼近真值。右图为TD方法和MC方法关于收敛速度的比较，图示纵坐标为估计值与真值的均方误差，对于TD来说，选取不同的步长参数$\alpha$可以改变其收敛的性能，一般来说，$\alpha$越小，精确度越高，但收敛的速度也越慢。MC方法全面落后于TD方法。并且也要注意到，不管是MC还是TD，收敛的结果都和$\alpha$高度相关，现在先记住这个结论，即收敛结果与步长$\alpha$有关。

### 6.3 Optimality of TD(0)

如果我们只能从有限的经验来进行学习，也就是说只有有限个episodes或者说是有限个时刻，我们使用一种新的方式对估计价值$V$进行更新。对于MC方法来说，我们从已知的数据中统计所有某一状态的真实的$G_t$，并分别使用$G_t - V(S_t)$来计算其增量，由此我们可以得到该状态的多个增量，将这多个增量求和后作为该状态的增量对其估计价值进行更新，可以得到一个新的价值，然后再更新，再得到一个新的价值，如此以往，能将该状态的价值收敛到一个确定的数字（所有$G_t$的均值），对于状态空间的所有状态都这么做，就能使所有状态的估计价值趋近于某一固定的数。同样的方法也适用于TD(0)，只不过这时统计的是某个状态所有的即时奖励和后继状态估计价值之和，再对其与本状态估计价值做差从而得到增量。这种方法我们称之为**batch upadating**，可以看到，对于MC来说，我们每批统计的$G_t$是固定不变的（$R_{t + 1} + V(S_{t + 1})$会因为每批不同的$V(S_t)$而发生变化），我们将其作为目标来使用const $\alpha$方法来更新估计价值$V(S_t)$，当步长$\alpha$足够小时，能使其最终收敛到一个确定的值。

以MC方法为例（$G_t$是固定不变的），我们使用以下式子进行更新
$$
\begin{aligned}
V(S_t) &\leftarrow V(S_t) + \alpha[\sum\limits_{n = 0}^{N - 1}G_t - N V(S_t)] \\
&\leftarrow V(S_t) + \alpha N[\frac{1}{N}\sum\limits_{n = 0}^{N - 1}G_t - V(S_t)] \\
\end{aligned}
$$
当$\alpha$足够小时，$\alpha N$小于1，而目标值$\frac{1}{N} \sum\limits_{n = 0}^{N - 1}G_t$永远固定不变，所以根据第二章提到过的固定步长更新表达式

$$
\begin{array}{rcl}Q_{n+1}&=&Q_n+\alpha\Big[R_n-Q_n\Big]\\
&=&(1-\alpha)^nQ_1+\sum_{i=1}^n\alpha(1-\alpha)^{n-i}R_i
\end{array}
$$
最终会使得$V(S_t) = \frac{1}{N} \sum\limits_{n = 0}^{N - 1}G_t$，即趋向这个固定值。所以batch upadating最终导致的结果是$V(S	_t)$趋向一个固定值，而在MC方法和TD(0)方法中，这两个固定值是不一样的，固定值的不同从某种程度上反映了MC和TD(0)不一样的特性，在正常的更新中，虽然最终趋向的值相同，但是前进的方向是不同的。

**Example 6.4**
假设对于某一MDP过程，我们有了以下的episodes

![image-20231018212145309](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018212145309.png)

我们要通过这些episodes来估计状态A和状态B的价值。首先由于状态B的下一时刻为终端状态，所以很轻易的计算出$V(B) = \frac{3}{4}$，因为八个B中共有6个B给出了奖励1。那么我们该如何对A进行预测呢？一种方法是使用TD(0)的方法，由于我们已经预测了状态B的价值，而已知的episode中只有A百分百在获得0奖励的同时转到状态B，所以我们可以利用已得到的状态B的信息，也即$V(B) = \frac{3}{4}$，从而也就推断$V(A) = \frac{3}{4}$。若用MC方法来预测，那么所有的episodes中只有一个episode包含状态A的信息，并且此时状态A的价值为0，所以我们就估计$V(A) = 0$。显然，使用了更多信息的TD(0)拥有更好的估计性能，MC只使用了一个episode的信息，由此虽然在训练集中MC能获得最小的误差（这里误差为0），但我们不认为其对未来数据拥有更好的匹配。我觉得MC方法最大的坏处就是由于其对于轨迹的严格要求，而TD(0)，能使用任意其他轨迹上对应状态来进行更新，导致其能获得的能用的信息往往要比TD(0)少得多，所以MC方法通常具有更大的方差。

使用TD(0)方法来计算上例状态B的价值还可以这样理解，我们将已知的所有数据建模成一个Markov过程，也就是其最大似然（maximum-likelihood）模型

![image-20231018104442430](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018104442430.png)

似然函数是给定参数，关于观测值的PDF，不同的参数值会导致不同的PDF，我们找到某个参数能够使得当前的PDF拥有最大程度的可能性生成当前的数据，则该参数就被称之为最大似然估计。均值在很多场景下是经典的最大似然估计，所以我们采用观测值的均值来简化这个Markov过程。使用已有数据进行建模如上图，就是保证了当前的各个参数（包括转移概率，奖励等等）能够最有可能生成当前的观测值。有了此建模假设，再求其对应的状态A和状态B的价值就很轻易了，由于此模型为最大似然模型，所以用此模型得到的参数也是最大似然估计。在基于最大似然的模型下，我们预测状态A和状态B的价值的正确性就只取决于最大似然模型的正确性，一旦模型正确，估计的参数也一定正确。这样的估计我们称之为确定性等价估计（certainty-equivalence estimate），而batch TD(0)方法最终趋向的就是确定性等价估计（batch MC方法趋向的是回报的均值）。

所以综上，certainty-equivalence estimate（batch TD(0)最终收敛的结果）的好处在于两个方面：精度更高和速度更快。精度更高体现在其为最大似然估计，数据越多，所用到的信息相比于batch MC方法要大得多；速度更快体现在其只用了即时回报和后继状态的估计值，相比于batch MC使用整条轨迹上的信息，前者所受到的扰动要更小，方差更小，也即收敛越快。

同样这也解释了TD(0)为什么比MC更快，因为普通的更新的部分都会在某种程度上与batch方法的重合，即使普通的更新方法与bach更新方法最终收敛到的值不同，但是多少也有着相同的前进方向。

**Exercise 6.7**
设计一个off-policy版本的TD(0)方法，即使用behavior policy的数据来估计target policy的价值
$$
V(S_t)\leftarrow V(S_t)+\alpha\rho_{t:t}\Big[R_{t+1}+\gamma V(S_{t+1})-V(S_t)\Big]
$$
由于TD(0)方法每次更新都只需要一个即使奖励数据$R_t$即可进行更新，所以只需要将$R_t$从策略$b$下的数据转化为策略$\pi$下的数据即可，即使用重要性因子$\rho_{t:t} = \frac{\pi(A_t \mid S_t)}{b(A_t \mid S_t)}$。*目前暂时不知道这样为什么是对的。。。当误差为随机变量，其内只包含一个随机变量，即奖励。*

### 6.4 Sarsa: On-policy TD Control

我们最终要将使用TD(0)的evaluation的部分合并在整个policy iteration中，其同样是在GPI的框架下进行的。在MC方法中，由于平衡exploration和exploitation的问题（若policy使用确定性的贪心算法，那么有些状态动作对永远不会被选择，也即缺少exploration），我们提出了on-policy和off-policy方法。同样在TD方法中也有这两种方法的区别，现在我们介绍的就是TD的on-policy方法。

**Sarsa Prediction**
首先要进行的步骤是将前面prediction问题中对状态价值$v_{\pi}(s)$的估计转换为对动作价值$q_{\pi}(s, a)$的估计，我们在使用某个episode进行对$v_{\pi}(s)$的估计时，关注的是从状态$S_t$转移到状态$S_{t + 1}$，其中$S_t$是我们要进行更新的状态，$S_{t + 1}$是我们要访问其价值的状态。对于$q_{\pi}(s, a)$的估计只需要在状态价值估计的基础上增加对动作$A_t$和$A_{t + 1}$的考量和记录，即我们将关注点从状态到状态的转移改变到状态动作对到状态动作对的转移，若发生了试验序列$S_t, A_t, R_{t + 1}, S_{t + 1}, A_{t + 1}$，我们使用以下式子进行更新
$$
Q(S_t, A_t) \leftarrow Q(S_t ,A_t) + \alpha \big[ R_{t + 1} + \gamma Q(S_{t + 1}, A_{t + 1}) - Q(S_t, A_t) \big]
$$
我们将状态价值预测的问题转换为了动作价值预测的问题，也即将表$V$转换为了表$Q$，相应的，公式中的所有$V$也都替换成了$Q$。因为每次更新都是用到了五个元素$(S_t, A_t, R_{t + 1}, S_{t + 1}, A_{t + 1})$，所以该种方法被称之为Sarsa，其回溯图如下

![image-20231018150853709](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018150853709.png)

从某个动作$A_t$出发，想要更新当前的$Q(S_t, A_t)$，需要得到即使奖励$R_{t + 1}$和下一时刻的状态动作对的预测值$Q(S_{t + 1}, A_{t + 1})$。

**Sarsa: On-policy TD Control**
有了以上的铺垫，我们就可以进行迭代，伪代码如下

![image-20231018153532310](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018153532310.png)

不同于MC的offline方法，Sarsa以online的方式更新预测值。在每个episode的开始，我们都要选择一个$S$作为该episode的初始状态，且按照已存的$Q(s, a)$（经过前几个episodes更新过，或是完全随机的初始值）选择动作$A$，由此完成对于该episode的初始化（即在进入每个episode内部循环之前都要提供一个状态$S$和动作$A$），进入子循环的迭代。在每次的子循环中，我们直接执行在本循环的开始前获得了$S$和$A$，进而得到即时奖励$R$和使得原状态转移到一个新的状态$S'$，我们再次根据当前的$Q(s, a)$来选择（例如$\varepsilon$-greedy）动作$A'$，由此集齐一个五元组$(S, A, R, S', A')$，可对$Q(s, a)$进行更新，并在循环的末尾中令下一个循环开始的$S$和$A$分别为现在循环的$S'$和$A'$，所以本次循环中最开始的$S$和$A$也是由上个循环末尾同样的操作而来（如果第一次进入子循环，那么就是初始化来的）。注意到在本次循环更新$Q(s, a)$前得到的$S', A'$作为下一个循环开始的$S, A$，而下一个循环的$S', A'$又是按照本次循环更新之后的$Q(s, a)$来选择策略（例如$\varepsilon$-greedy），所以更新表达式（在GPI的模式下）可写为（以$\varepsilon$-greedy更新为例）
$$
Q(S_t, A_t) \leftarrow Q(S_t ,A_t) + \alpha \big[ R_{t + 1} + \gamma Q'(S_{t + 1}, A_{t + 1}) - Q(S_t, A_t) \big]
$$
其中$Q'(s, a)$有$1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|}$的概率为$\max\limits_aQ(S_{t + 1}, a)$，有分别$\frac{\varepsilon}{|\mathcal{A}(s)|}$的概率选择任意其他所有对应的$Q(S_{t + 1}, a)$，$Q'(S_{t + 1}, A_{t + 1})$选择的策略是上个循环更新后的策略，而$Q(S_t, A_t)$选择的策略是上个循环更新前的策略。这里之所以选择$Q'(s, a)$，是为了将$\varepsilon$-greedy显式的体现在表达式中。

可以看到，我们不断地使用$\varepsilon$-greedy更新的策略来得到数据，并且对同一策略进行改进，所以Sarsa为on-policy方法，其最终能将策略收敛至$\varepsilon$-soft中最优的策略。在实际问题中也常常让$\varepsilon$随着episode的进行而不断减小，从而不断逼近最优策略。

### 6.5 Q-learning: Off-policy TD Control

如果真正理解了Sarsa的更新表达式，Q-learning的更新表达式就十分好理解了，Q-learning的更新表达式如下
$$
Q(S_t, A_t) \leftarrow Q(S_t ,A_t) + \alpha \big[ R_{t + 1} + \gamma \max\limits_aQ(S_{t + 1}, A_{t + 1}) - Q(S_t, A_t) \big]
$$
该表达式和Sarsa的更新表达式有略微的不同：
一是将Sarsa更新表达式中的$Q'(S_{t + 1}, A_{t + 1})$转换为了$\max\limits_aQ(S_{t + 1}, A_{t + 1})$，而前者是按照$\varepsilon$-greedy策略进行的，后者更只是在前者的基础上去掉$\varepsilon$，即完全使用greedy策略。注意到Sarsa由于产生数据和改进的策略（即在式子中得到$Q'(S_{t + 1}, A_{t + 1})$的方式）都是同一策略，所以其为on-policy方法。那么对于Q-learning来说，为了保证所有的状态动作对都能被访问到（这是保证收敛最基本的要求，书上没说这是为什么，我的猜想是DP方法的自举特性使用到了所有其他状态的价值，而TD方法只是DP方法利用数据的模拟，其内在需要用到所有状态的数据，当然需要让所有的状态动作对都能被访问到），产生数据的策略为$\varepsilon$-greedy；由于Q-learning并没有正真意义上到达了使得$Q(S_{t + 1}, A_{t + 1})$最大的后继状态，使用最大的$Q(S_{t + 1}, A_{t + 1})$进行更新隐含着使用greedy方法产生后继状态这一过程，而这一后继状态并没有真正的被使用，所以产生数据和改进的策略不是同一个策略，为off-policy方法。
二是Sarsa中$Q(S_t, A_t)$使用的更新策略为上个循环更新前的策略，而$Q'(S_{t + 1}, A_{t + 1})$使用的策略是上个循环更新后的策略，也就是使用当前的Q表进行更新；而Q-learning的表达式中，前面的$Q(S_t, A_t)$和后面的$\max\limits_aQ(S_{t + 1}, A_{t + 1})$都是由当前的Q表进行更新的。综上，Sarsa前一项的策略为旧策略，后一项的策略才为当前新的策略，所以Sarsa所进行的GPI从某种意义来说有一定的延迟（在实践中这并不重要），而Q-learning前后策略都是当前新的策略。
第三点的不同更为本质，Sarsa在进行更新时，实际已经采取了$S_{t + 1}, A_{t + 1}$作为接下来的决策序列，这也是第二点不同的本质原因，由于在更新前采用了$S_{t + 1}, A_{t + 1}$，所以$S_{t + 1}, A_{t + 1}$作为下次循环的开头必然使用的是对于下次循环来说旧的策略。

Q-learning在基于Sarsa的改进其实很自然，首先是对于一个五元组$(S_t, A_t, R_{t + 1}, S_{t + 1}, A_{t + 1})$，我们使用$Q(S_{t + 1}, A_{t + 1})$来更新$Q(S_t, A_t)$实际上是根据TD(0)的思想，由于TD(0)是在MC基础上的改进，所以其仍然保留着需要实际产生轨迹的条件。现在我们放宽这个条件，也即用来更新的$S_{t + 1}, A_{t + 1}$是否真实产生其实不重要，所以我们用不着真正实施$S_{t + 1}, A_{t + 1}$，由此自然而然会想到选用$\max\limits_aQ(S_{t + 1}, A_{t + 1})$作为对下一步的预想，从而进行更新。

Q-learning即使在选择$\varepsilon$-greedy的改进策略下，也使得最终的动作价值收敛之最优。不同于Sarsa，由于其是使用了$\varepsilon$-greedy的on policy算法，所以最终收敛的策略也是在$\varepsilon$-soft策略中最优的策略。

**Exercise 6.12**
暂时不考虑充分遍历所有状态动作对这件事，如果将Sarsa和Q-learning的动作选择都改为greedy，那么此时Sarsa和Q-learning再每次的更新中所做的动作是否完全相同？

答案显然是否定的，因为Sarsa更新表达式的前后所用的策略是不同的，即后项的策略是前项策略更新后的，而Q-learning前后所用的策略都是当前最新的。故若前项$Q(S_t, A_t)$保持相同，即Sarsa和Q-learning要准备更新同一个Q表，那么Sarsa在后项的抉择中由于使用前项更新后的策略，所以可能会与Q-learning不同。

以下是Q-learning的伪代码

![image-20231018210510805](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018210510805.png)

Q-learning和Sarsa的伪代码最大的不同在于Q-learning只需要一个状态和动作对即可进行更新，而Sarsa需要一个五元组才能进行更新，所以从代码层面上，Q-learning的代码更好写。Q-learning的回溯图如下

![image-20231018211049700](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231018211049700.png)

为了预测当前动作状态对的动作价值，以后继状态所有的动作价值中最大的那个进行更新。

### 6.6 Expected Sarsa

前面说到，Sarsa和Q-learning最本质的不同在于Sarsa在状态$S_{t + 1}$时，真的采用了更新表达式中的策略$A_{t + 1}$，也即$S_{t + 1}, A_{t + 1}$作为下次循环的开头$S_t, A_t$，必然是使用对于当时的Q表来说旧的策略。并且由于表达式中后项$Q(S_{t + 1}, A_{t + 1})$是按照当时更新策略实际选择的，所以往往会造成更新后的$Q(S_t, A_t)$有较大的方差。为了对Sarsa进行改进，我们放弃将实际选择作为更新表达式的后项，也即此时不再需要提前决定下步怎么走，而是如同Q-learning一样假想出下步的选择，同时将后项改为期望的形式，也即下式
$$
\begin{aligned}
Q(S_{t},A_{t})& \leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\mathbb{E}_\pi[Q(S_{t+1},A_{t+1})\mid S_{t+1}]-Q(S_t,A_t)\Big]  \\
&\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)-Q(S_t,A_t)\Big]
\end{aligned}
$$
也就是期望Sarsa（Expected Sarsa）的更新表达式。其即吸收了Q-learning不用实际走出下一步的优点，也舍弃了Sarsa以实际选择的下一步来作为更新表达式的第二项从而导致的较大的方差（因为只有一次观测）。如果说Expected Sarsa是on-policy策略，其原因在于$\mathbb{E}_\pi[Q(S_{t+1},A_{t+1})\mid S_{t+1}]$仍然是按照$\pi$策略下的意义来求得期望的，不像off-policy的Q-learning直接选择最大值来作为第二项，使其与$\pi$无关。所以实际上可以将Q-learning看成Expected Sarsa的推广，即不管将on-policy的对$\pi$求期望，改为对其他策略求期望，在Q-learning中，这个其他策略就是greedy策略，在Sarsa中，用于更新的只是策略$\pi$下的一个样本，并且下一步还强制要求走这个样本。同时Expected Sarsa也可以是off-policy，若在期望更新表达式中所选用的权重不是behavior policy，而是其他，那么此时就是off-policy。可以将Expected Sarsa视为Sarsa在状态$S_{t + 1}$时为了选择$A_{t + 1}$而进行了大量的试验，用均值来消除Sarsa可能在选择$A_{t + 1}$所带来的随机性，本质上是对Sarsa的提升，故Expected Sarsa收敛的结果也和Sarsa一样。对于选择其他加权方式的Expected Sarsa，可将其视为选择了另一种策略$b$多次选择$A_{t + 1}$而来的均值。

**Example 6.6**
下面为某一任务示意图，任务的目标是从起点顺利到达重点，期间不能进入“悬崖”（灰色部分），否则重新回到起点，并扣除一百分。为了鼓励能够尽快到达终点，每做一个动作，就扣一分。

![image-20231019142604688](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231019142604688.png)

我们分别使用$\varepsilon = 0.1$的$\varepsilon$-greedy策略分别对Sarsa和Q-learning进行评估，迭代到最后，得到的策略上图所示，得到的性能如下图所示

![image-20231019145636156](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231019145636156.png)

可以看到，在此场景下，Q-learning的性能没有Sarsa好，这是因为Q-learning使得策略收敛到最佳策略，而由于Q-learning本身的bahavior策略是$\varepsilon$-soft策略，在以最优策略执行时还伴有随机的探索，而在此场景下，最优策略的探索意味着有很高的风险进入“悬崖”，所以奖励分会更低。而Sarsa收敛到的策略是在$\varepsilon$-soft策略空间中的最优策略，也即这个最优策略已经考虑到了行动所带有的固定的随机性，所以在此场景下Sarsa的性能要更好。

那么Expected Sarsa在此场景下表现如何呢？下图添加了Expected Sarsa在此场景下的数据

![image-20231019150725056](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231019150725056.png)

可以看到，Expected Sarsa在收敛之后从始至终保持着较好的策略，无论$\alpha$的变化。同时，其又拥有比普通的Sarsa更快的收敛速度。随着$\alpha$增大的时候，Sarsa的性能会直线下降，我觉得这是由于$\alpha$从某种程度上表示这新值在旧值中的比重，因为Sarsa在$S_{t + 1}$是选择的$A_{t + 1}$具有很大的随机性（方差更大），所以将新值占的比重更多就会增加预测的随机性，所以导致了性能的下降，而Expected Sarsa虽然增加了一点计算复杂度，但随之而来的好处是在一定程度上消除了选择动作$A_{t + 1}$所带来的随机性。

Sarsa最大的缺点在于进行更新的后继状态是真正执行的状态，而此状态会导致过大的方差，从而导致性能的下降。而Q-learning和Expected Sarsa都是使用后继状态的某种统计量（前者是最大值，后者是期望），统计量的使用降低了后继状态的随机性。Sarsa的这一缺点在步长$\alpha$大的时候就会变得尤为明显，原因就是上面所说的步长代表着新值的比重。

### 6.7 Maximization Bias and Double Learning

目前为止我们所讨论的迭代算法都在构建target policy的过程中使用了最大化（maximization），例如在Q-learning中使用了greedy算法，Sarsa中使用$\varepsilon$-greedy算法。当我们从预测的动作价值中选择一个最大值时，我们也正在估计其就是最大动作价值的预测值，但是选择众多预测值中的最大来估计最大价值会产生一个偏差，我们称之为最大化偏差（maximization）。

**Example 6.7**
有如下的MDP，起始状态为A，选择动作向右会导致直接达到终端状态，且没有奖励；选择动作向左虽然也不能获得奖励，但可以使得状态A转移到状态B，在状态B下，有多种动作可供选择，每种动作得到的奖励都服从一个均值为0.1方差为1的高斯分布，然后到达终端状态。

![image-20231019164228173](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231019164228173.png)

在状态A中，Q-learning根据$Q(A, \text{left})$和$Q(A, \text{right})$的相对大小来使用$\varepsilon$-greedy选择策略生成episode。可知$Q(A, \text{right}) = 0$，若使用Q-learning算法来更新$Q(A, \text{left})$，则需要知道$\max\limits_a Q(B, a)$，而$Q(B, a)$呈均值为-0.1，方差为1的正态分布，所以有极大的可能会用高于0的动作价值来更新$Q(A, \text{left})$，由此造成在最初的episodes中，Q-learning极为拥护在状态A时向左的动作，而遗憾得到了错误的答案，这就是maximization bias。只有当试验次数足够多的时候，Q-learning才会意识到选择左边的骗局，但是现在仍然存在最大化偏差，所以仍然离最优的百分之五有点距离。

我觉得导致maximization bias的问题在于使用预测值的最大来估计最大真值需要同时估计两个信息，一是最大真值的位置在哪（即哪个动作），二是最大真值的值是什么，使用预测值的最大的位置来估计真值最大的位置是无偏的，但是直接使用预测值的最大来预测最大的真值一定是有偏的，而后者就是maximization bias产生的原因所在。解决问题的思路利用我们可得知的预测值最大的位置（这个估计量是无偏估计量），在另一组样本中相应位置的值作为最大真值的预测值，这样最大真值的估计值也是无偏的。

我们使用两组预测值$Q_1$和$Q_2$，并分别交叉赋予两组预测值不同的任务，$Q_1$用来确认最大值出现的位置，$Q_2$用来提供相应位置的值。比如在Q-learning的场景下，每轮的更新会以平等的概率0.5选择以下式子更新
$$
Q_1(S_t,A_t)\leftarrow Q_1(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma Q_2\big(S_{t+1},\underset{a}{\operatorname*{\arg\max}}Q_1(S_{t+1},a)\big)-Q_1(S_t,A_t)\Big] \\
 \text{and} \quad Q_2(S_t,A_t)\leftarrow Q_2(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma Q_1\big(S_{t+1},\underset{a}{\operatorname*{\arg\max}}Q_2(S_{t+1},a)\big)-Q_2(S_t,A_t)\Big]
$$
每次都是以$Q_1$的值更新$Q_2$（我觉得相反也是可以的），要么相反，所以$Q_1$和$Q_2$完全镜像。用于产生数据的behavior target可以选择其中一个进行，或是两个一起进行，如$Q_1 + Q_2$。这种方法就是double learning，使用double learning的效果如上图的绿色曲线。下面是Double Q-learning的算法框图

![image-20231019172012096](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231019172012096.png)

**Exercise 6.13**
写出Double Expected Sarsa更新表达式
$$
Q_1(S_t, A_t)\leftarrow Q_1(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_a\pi_2(a\mid S_{t+1})Q_1(S_{t+1},a)-Q_1(S_t,A_t)\Big] \\
\text{and} \quad Q_2(S_t,A_t)\leftarrow Q_2(S_t,A_t)+\alpha\Big[R_{t+1}+\gamma\sum_a\pi_1(a \mid S_{t+1})Q_2(S_{t+1},a)-Q_2(S_t,A_t)\Big]
$$
这里就是以$Q_1$更新$Q_1$，此时$Q_2$负责找位置。其中$\pi_1$和$\pi_2$分别对应$Q_1$和$Q_2$下的$\varepsilon$-greedy策略，因为$Q_1$和$Q_2$并不相同，所以对应策略也不相同，这里的$\pi_1$和$\pi_2$就相当于前面的最大值出现的位置。

如此情况下必然也存在Double Sarsa，那么其更新表达式为
$$
Q_1(S_t, A_t) \leftarrow Q_1(S_t ,A_t) + \alpha \big[ R_{t + 1} + \gamma Q'_1(S_{t + 1}, A_{t + 1}) - Q_1(S_t, A_t) \big]
$$
其中$Q'_1(S_{t + 1}, A_{t + 1})$中的$A_{t + 1}$是由$\pi_2$来进行确定的，也即使用$Q_2(S_{t + 1}, a)$的$\varepsilon$-greedy贪心。$Q_2(S_t, A_t)$的更新表达式与上式刚好相反。

### 6.8 Games, Afterstates, and Other Special Cases

学不动了，以后再学。。。
