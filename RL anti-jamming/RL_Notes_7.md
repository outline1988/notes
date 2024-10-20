## Chapter 7 $\boldsymbol{n}$-step Bootstrapping

### 7.1 $\boldsymbol{n}$-step TD Prediction

前面我们所讨论的TD算法皆为one-step TD，也就是说，为了得到新的估计值$V(S_{t})$，我们使用该状态的后继状态的估计值$V(S_{t + 1})$以及从当前状态$S_t$转移到后继状态$S_{t + 1}$所得到的奖励$R_{t + 1}$之和作为目标值来更新$V(S_t)$。MC方法使用$G_t$也就是剩余所有奖励之和来更新当前状态价值估计$V(S_{t})$。那么一个很自然的想法就是，即我们不只使用一个奖励，也不使用全部奖励，而是使用任意给定的$n$个奖励来进行更新，介于one-step TD和MC之间，我们称之为$n$-step TD。回溯图如下

![image-20231021171234793](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231021171234793.png)

上图的最左边为one-step TD方法，最右边为MC方法，介于这两个之间的就是$n$-step TD。更正式的来说，假设我们在忽略动作的情况下生成了以下序列$S_t, R_{t + 1}, S_{t + 1}, R_{t + 2}, S_{t + 2}, \dots , R_T, S_T$，MC方法直接使用所有奖励之和来更新价值函数
$$
G_t\doteq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T
$$
而one-step方法使用$\gamma V(S_{t + 1})$来替代了$\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots+\gamma^{T-t-1}R_T$这个部分，也即
$$
G_{t:t+1}\doteq R_{t+1}+\gamma V_t(S_{t+1})
$$
这里的$G_{t : t + n}$是我们新定义的符号，其表示当前时刻$t$到达时刻$t + n$中所有的奖励$R_{t + 1}, R_{t + 2}, \dots , R_{t + n}$保留，然后再与$t + n$时刻转移到的状态的估计价值$V(S_{t + n})$来更新$V(S_t)$，所以$G_{t : t + 1}$只使用了一个奖励$R_{t + 1}$和估计价值$V(S_{t + 1})$来更新。所以很显然，用于$n$-step TD更新的目标值为
$$
G_{t : t + n} = R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots + \gamma^{n - 1}R_{t + n} + \gamma^n V(S_{t + n})
$$
很显然，必须保证$n \geq 1$且到达的最终状态$S_{t + n}$不能超过终端状态$S_T$，也即$t + n < T$，不去等号的原因是为了将MC方法与$n$-step TD进行区别。

注意，由于$n$-step TD在更新时刻$t$时，需要用到当前时刻$t$到$t + n$的数据，所以我们只能在$t + n$时刻才能对$S_t$进行更新。在$t + n$时刻还没更新之前，状态$S_t$的价值已经迭代到了$V_{t + n - 1}(S_t)$（如果时刻$t + 1$到$t + n$中，没有再遇到与$S_{t}$相同的状态，那么$V_{t + n - 1}(S_t) = V_t(S_t)$），所以更新式如下
$$
V_{t + n}(S_t) \doteq V_{t + n - 1}(S_t) + \alpha \big[ G_{t : t + n} - V_{t + n - 1}(S_{t + n}) \big]
$$
每一次迭代都只有一个状态的价值更新，其他状态的价值保持不变，但是为了标号方便每次更新都要对所整组$V$的下标进行更新，所以在时刻$t + n$更新状态$S_{t}$的价值时，能够用到的价值应该是更新到$t + n$时刻之前的价值$V_{t + n - 1}(S_{t + n})$。$n$-step TD算法的伪代码如下

![image-20231021223909730](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231021223909730.png)

在某一个episode的循环中，若当前时刻为$t$，则我们按照策略$\pi$来选择动作以观测下一个状态$S_{t + 1}$并同时获得并记录$R_{t + 1}$，我们判断$S_{t +1}$是否为终端状态，若为终端状态，我们将记录终端时刻$T$为当前的$t + 1$。在获得了$R_{t + 1}$之后，和估计价值$V$一起，我们就能够计算出$G_{t + 1 - n : t + 1}$（内包含$n$个奖励），从而对状态$S_t$进行更新。但是在此之前，有可能会有无法找到该更新哪个状态的情况（$t < n$时，此时前面$n$步的状态超出了边界），所以首先需要判断$t + 1 - n$是否不低于0，这里使用$\tau$与0来比较是为了方便编程。若能够找到需要更新的状态（即前面$n$步的状态），则首先计算奖励的折扣累计，然后计算状态$V_{\tau + n}$的值，即$\tau + n$（也就是$t + 1$）为终端时刻$T$，则为0，否则查表。当$\tau$为$T - 1$时，即为该更新的状态为终端状态的前一个状态时，结束继续循环。注意到$n$-step TD方法所需要循环的次数为$T + n - 1$，对于最后的$n$个循环，episode无法提供完整的回报信息，所以超过$T$的奖励和预测值在计算的时候都以0来处理。

在本代码中，每次的循环到的时刻$t$都是在取得后继时刻$t + 1$的有关参数后再来开展的。

**Exercise 7.1**
使用one-step TD的误差$\delta_t$来表示$n$-step误差$G_{t : t + n} - V(S_t)$
$$
\begin{aligned}
G_{t : t + n} - V(S_t) &= R_{t + 1} + \gamma G_{t + 1 : t + n} + \gamma V(S_t) - \gamma V(S_{t + 1}) \\
&= \delta_t + \gamma (G_{t + 1 : t + n} - V(S_{t + 1})) \\
&= \delta_t + \gamma(\delta_{t + 1} + \gamma(G_{t + 2 : t + n} - V(S_{t + 2}))) \\
&= \delta_t + \gamma \delta_{t + 1} + \gamma^2 (G_{t + 2 : t + n} - V(S_{t + 2})) \\
&= \delta_t + \gamma \delta_{t + 1} + \gamma^2 \delta_{t + 2} + \cdots + \gamma^n (G_{t + n : t + n} - V(S_{t + n})) \\
&= \delta_t + \gamma \delta_{t + 1} + \gamma^2 \delta_{t + 2} + \cdots + \gamma^n (V(S_{t + n}) - V(S_{t + n})) \\
&= \delta_t + \gamma \delta_{t + 1} + \gamma^2 \delta_{t + 2} + \cdots + \gamma^{n - 1}\delta_{t + n - 1} \\
&= \sum\limits_{k = t}^{t + n - 1}\gamma^{k - t} \delta_{k}
\end{aligned}
$$
对于$n$-step TD方法来说，有一个特殊的性质能够保证每次的更新都能往着真值更近一步（至少不会后退），即以下不等式成立
$$
\max\limits_s \big| \mathbb{E}[G_{t : t + n} \mid S_t = s] - v_{\pi}(s) \big| \leq \gamma^n \max\limits_s \big| V_{t + n - 1}(s) - v_{\pi}(s) \big|
$$
也即使用$n$-step TD得到的目标值$G_{t : t + n}$在一阶矩的意义（期望）下，所有状态的最大误差要小于更新前的预测值$V_{t + n - 1}$的所有状态中最大的误差的$\gamma^n$倍。这种性质我们称之为error reduction property，在该性质下，我们能够正式的证明$n$-step TD能够最终收敛到正确的真值。（你不证明我也知道你会收敛的啦。。。）

**Example 7.1**
我们选择之前使用过的随机游走问题来展示$n$-step TD方法的prediction问题，这次我们将状态数量从5增加至19，并且仍然保持只有到达最右边终端状态时的奖励为1，其余的状态转移包括到达最左边终端状态时的奖励都为0，仍然设置起点状态为最中间的状态，所有状态的初始值都设置为0.5。我们想要探究的是选择怎样的$n$最为合适，试验结果如下

![image-20231022202358617](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231022202358617.png)

上图的实验使用最初的10个episodes，以不同的参数$n$和$\alpha$来对状态价值$V$进行更新，并将得到的结果与实际算出来的真值做均方根误差（root mean square，RMS），重复100次最后取平均值。由于只使用了较少的episodes，所以我们并不期望其最终的收敛结果能够完全逼近真值。但是，当选择参数$n = 4$，$\alpha = 0.4$左右时，其在有限数据的情况下能达到的效果是最好的。

*我认为$n$的选择同时影响收敛速度和精度（方差），随着$n$的增大，状态价值的收敛速度会越来越快。在本例子中，只有终端状态的前一状态转移到终端状态时才能获得真正有用的信息（奖励）。我们可以从第一次episode进行完毕之后清楚的感受到，若使用one-step TD方法，一次episode之后，只有终端状态的前一个状态才被更新，只有在多个episodes之后，终端状态所得到的奖励才会被扩散到其他所有状态。所以也可以从一次学习的广度来理解$n$这个参数，one-step一次只从该状态相邻的一个状态来学习，而$n$-step状态一次从相邻的$n$个状态的奖励信息进行学习，所以自然而然，从更大范围学习的算法收敛速度会更快，更特别的MC方法一个episode对所有访问过的状态进行了更新。所以我们对$n$-step TD学习的方式有更深刻的认识，也即状态价值收敛到真值的过程就是每个状态在随着更新的过程中将自身的信息（奖励）扩散到其他状态，而参数$n$决定了扩散的速度。*

*难道$n$越大越好？当然不是，收敛速度增大带来的坏处就是随机性增加，从原来只有一个奖励具有随机性的one-step转换为了具有$n$个奖励随机性的$n$​-step，理所当然的随机性在不断的累积，所以从图示中可以看到，虽然收敛速度变快了，但相应的误差也越来越大。*

上述描述是初学时的感受，结论可能不完全甚至是错的，这里补充后续我对于MC方法和TD(0)方法的分析以及$n$-step TD如何将这两个方法统一在一起。总的来说，MC方法拥有更高的**一次样本利用率**，对于一个样本，使用MC方法可以使得该样本在紧接的更新中被一条轨迹中的所有状态使用，该轨迹的所有状态在一次更新中都使用到了这个样本所包含的信息；而对于TD(0)来说，一次更新只能使得一个状态获得该样本所拥有的信息量（即使如此，其后续的更新会使得这个信息量最终覆盖到轨迹中的其他状态）。TD(0)方法可使用**更多的样本数量**，不管是否在同一个episode，只要出现了该状态的信息，其在使用时序差分的更新方式时就能够使用，说明最小二乘的实例可以清楚的展现，而MC只能使用到同一个episode的信息，所以在相同数量episodes的情况下，TD(0)能够使用更多的信息。随着$n$的增加，样本的利用率在提高，但是能用的信息量就减少了，所以通常来说存在一个最佳的$n$使得样本的利用率和能用的信息量达到一种平衡，最大化收敛的速度。更多能用的信息量要好于更高的利用率，所以通常TD(0)的收敛速度要比MC高。所有本质上我们说TD(0)比MC收敛更快，亦或是什么MC方法的方差大，都是在说TD(0)相比于MC方法使用了更多的样本信息。TD(0)用$V(S_{t + 1})$来取代MC方法的$G_{t + 1 : T}$，而$V(S_{t + 1})$几乎使用到了之前所有的样本数据，几乎可以认为就是之前所有样本的某种加权均值，而$G_{t + 1 : T}$仅仅只是一些奖励的加和。

### 7.2 $\boldsymbol{n}$-step Sarsa

与将one-step Sarsa的prediction问题转到control时思路一样，我们还是先从状态价值的估计转移到动作价值的估计问题上。动作价值的prediction只与状态价值的predition有略微的区别，即动作状态价值需要使用的预测值为$Q(S_{t + n}, A_{t + n})$，所以在episode中我们在状态$S_{t + n}$的时候，仍然还需要实际的选择并使用动作$A_t$，此时回报$G_{t : t + n}$的表达式为
$$
G_{t + t : n} = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1} R_{t + n} + \gamma^n Q(S_{t + n}, A_{t + n})
$$
所以更新表达式也变为了
$$
Q_{t + n}(S_{t}, A_{n}) = Q_{t + n - 1}(S_{t}, A_{t}) + \alpha \big[ G_{t : t + n} - Q_{t + n - 1}(S_t, A_t) \big]
$$
由此可写出伪代码为

![image-20231022201203724](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231022201203724.png)

该程序和上面的$n$-step prediction的程序很像，只有在细微的地方有所区别，一是使用了$\varepsilon$-greedy来更新策略；二是要实际选择上一个循环生成的$A_{t + 1}$（在本次循环中为$A_t$），即在选择完动作并判断下一时刻是否为终端状态后，若不是则根据当前动作价值使用$\varepsilon$-greedy方法继续选择下一个动作（在prediction问题中，已知都是使用给定$\pi$来生成episode的，并且在$S_{t + 1}$不需要选择动作），且这个动作将在下一个循环中实际被选择。

$n$-step Sarsa的回溯图如下

![image-20231022201713050](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231022201713050.png)

**$\boldsymbol{n}$-step Expected Sarsa**
对于最右边的$n$-step Expected Sarsa来说，其更新表达式为
$$
Q_{t + n}(S_{t}, A_{t}) = Q_{t + n - 1}(S_t, A_t) + \alpha \big[ G_{t: t + n} - Q_{t + n - 1}(S_t, A_t) \big]
$$
其中
$$
G_{t:t+n} = R_{t+1} + \gamma R_{T+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n \bar{V}_{t+n-1}(S_{t+n})
$$
其中
$$
\bar{V}_{t + n - 1}(S_{t + n}) = \sum\limits_a \pi(a \mid s)Q(S_{t + n}, a)
$$
我们将$\bar{V}_{t + n - 1}(S_{t + n})$称之为期望近似价值（expected approximate value）。

### 7.3 $\boldsymbol{n}$-step Off-policy Learning

回忆在MC中的off-policy方法，我们使用importance sampling来将更exploration的策略$b$下的序列产生的$G_t$使用重要性因子$\rho_{t : T - 1} = \prod\limits_{k = t}^{T - 1}\frac{\pi(S_k \mid A_k)}{b(S_k \mid A_k)}$修正为策略$\pi$下的数据，由此以incremental的方式进行更新（用算术均值或是按照重要性因子加权平均）。我们仍然按照importance sampling的思路套用到$n$-step TD方法，在on-policy下，prediction问题的更新表达式为
$$
V_{t + n}(S_t) = V_{t + n - 1}(S_t) + \alpha \big[ G_{t : t + n} -V_{t + n - 1}(S_t) \big]
$$
其中
$$
G_{t : t +n} = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1} R_{t + n} + \gamma^n V_{t + n - 1}(S_{t + n})
$$


使用了important sampling后，prediction问题的更新表达式为
$$
V_{t + n}(S_t) = V_{t + n - 1}(S_t) + \alpha \rho_{t : t + n - 1} \big[ G_{t : t + n} - V_{t + n - 1}(S_{t + n}) \big]
$$
其中（重要性因子下标的终端时间为$t + n - 1$，之所以要除去最后一项，是因为我们所获奖励最多到$R_{t + n}$，而这项奖励是由$S_{t + n - 1}$和$A_{t + n -1}$进行修正的）
$$
\rho_{t : t + n - 1} = \prod_{k}^{\min(t + n - 1, T - 1)} \frac{\pi(S_k \mid A_k)}{b(S_k \mid A_k)}
$$
需要注意的是，$G_{t : t + n}$这一项中只有奖励项即$R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1} R_{t + n}$是在策略$b$下产生的，而其余的$\gamma^n V_{t + n - 1}(S_{t + n})$是在策略$\pi$下的数据。之所以讲重要性权重放在误差项的前面，我也不理解，我觉得应该放在回报中的奖励项中，因为只有这些项是有behavior策略产生的，需要修正。同样，behavior更具探索性体现在所有策略$\pi$下产生的中间轨迹$S_t, A_t, R_{t + 1},\dots,S_{t + n - 1}, A_{t + n - 1}$都要可有策略$b$产生。

由此我们就能写出动作价值的更新表达式
$$
Q_{t + n}(S_t, A_t) = Q_{t + n - 1}(S_t, A_t) + \alpha \rho_{t + 1 : t + n - 1}\big[ G_{t:t + n} - Q_{t + n - 1}(S_t, A_t) \big]
$$
这里的$G_{t : t + n}$的含义同上。重要性因子$\rho_{t + 1 : t + n}$的起始项之所以要加一，原因和MC方法那一节相同，即都不对状态$S_t$和动作$A_t$下产生的奖励$R_{t + 1}$进行修正，因为策略$\pi$不一定能产生$S_t, A_t$轨迹，而且我们也不关心策略$\pi$其到底能不能产生这个轨迹，由此只对第一项之后的项进行修正；书上的重要性因子在这里为$\rho_{t:t + n}$，我觉得是错误的，因为$\rho_{t + n} = \frac{\pi(S_{t + n} \mid A_{t + n})}{b(S_{t + n} \mid A_{t + n})}$，而我们又没有实际的产生奖励$R_{t + n + 1}$，又怎么用$\rho_{t + n}$来修正呢？

使用off-policy的$n$-step Sarsa方法伪代码如下

![image-20231024111556178](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231024111556178.png)

off-policy的$n$-step Sarsa和on-policy的$n$-step Sarsa行为几乎一模一样，仅仅在计算重要性权重的部分中有所区别。所以$n$-step Sarsa既可以为on-policy，也可以为off-policy。而one-step Sarsa只能为on-policy，因为重要性因子不会对第一项奖励进行修正。

### 7.5 Off-policy Learning Without Importance Sampling: The $\boldsymbol{n}$-step Tree Backup Algorithm

前面说介绍的$n$-step TD方法都是在Sarsa的框架下进行介绍的，包括其最自然的on-policy $n$-step Sarsa，和使用importance sampling改进后的off-policy $n$-step Sarsa。

迄今为止，我所讨论的model-free算法使用自举（bootstrap）思想的仅有TD方法，包括其中的TD(0)和$n$-step TD算法，这些TD方法有一个共同特点是，我们只使用了一个预测值来完成对于目标预测值的估计。对于DP来说，我们穷举尽了状态空间内所有的状态，才完成了一次的更新，那么在TD中，我们是否也可以利用更多的预测值来进行自举更新呢？观察如下回溯图

![image-20231024144520473](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231024144520473.png)

One-step Expected Sarsa的表达式如下，我们将在其的基础上发展出树回溯算法（tree-backup algorithm）
$$
Q_{t + 1}(S_t, A_t) = Q_{t}(S_t, A_t) + \alpha\big[ R_{t + 1} + \gamma\sum\limits_a \pi(S_{t + 1} \mid a) Q_t(S_{t + 1}, a) - Q_t(S_t, A_t) \big]
$$
期望Sarsa自举了状态$S_t$后继状态$S_{t + 1}$所有的价值函数$Q(S_{t + 1}, a)$。在$n$-step TD（以2-step TD为例）的情况下，$S_{t + 1}$接下来的动作$A_{t + 1}$等都知道，所以我们将上式中误差项内求和的$Q(S_{t + 1}, A_{t + 1})$使用我们实际得到的轨迹来代替，也即
$$
\begin{aligned}
\sum\limits_a \pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) &= \sum\limits_{a \neq A_{t + 1}}\pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) + \pi(S_{t + 1}, A_{t + 1})Q_t(S_{t + 1}, A_{t + 1}) \\
&= \sum\limits_{a \neq A_{t + 1}}\pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) + \pi(S_{t + 1}, A_{t + 1})( R_{t + 2} + \gamma \sum\limits_{a}\pi(S_{t + 2}, a) Q_t(S_{t + 2}, a)) \\
\end{aligned}
$$
令
$$
\begin{aligned}
G_{t : t + 2} &= R_{t + 1} + \gamma\sum\limits_a \pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) \\
&= R_{t + 1} + \gamma\big[ \sum\limits_{a \neq A_{t + 1}}\pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) + \pi(S_{t + 1}, A_{t + 1})( R_{t + 2} + \gamma \sum\limits_{a}\pi(S_{t + 2}, a) Q_t(S_{t + 2}, a)) \big] \\
&= R_{t + 1} + \gamma\pi(S_{t + 1}, A_{t + 1})R_{t + 2} + \gamma \sum\limits_{a \neq A_{t + 1}}\pi(S_{t + 1}, a) Q_t(S_{t + 1}, a) + \gamma^2 \pi(S_{t + 1}, A_{t + 1})\sum\limits_{a}\pi(S_{t + 2}, a) Q_t(S_{t + 2}, a)
\end{aligned}
$$
上式就是$2$-step的tree-backup算法，形式很复杂，但是内核很简单，就是在期望Sarsa的基础上将实际产生的序列应用到期望的计算上。

对于$n$-step的tree-backup算法来说，为了将表达式简化，我们使用递归的方式来重新写出其表达式
$$
G_{t : t + n} \doteq R_{t + 1} + \gamma \sum\limits_{a \neq A_{t + 1}}\pi(a \mid S_{t + 1})Q_{t + n - 1}(S_{t + 1}, a) + \gamma \pi(A_{t + 1} \mid S_{t + 1})G_{t + 1 : t+ n}
$$
程序的伪代码如下

![image-20231024152146447](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231024152146447.png)

*Exercise 7.4和Exercise 7.11待补充*
