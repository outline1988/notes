### Natural Policy Gradients

传统的策略梯度下降以如下更新表达式对每一个状态动作对在同一个episode中进行更新
$$
\boldsymbol{\theta}_{k + 1} = \boldsymbol{\theta}_{k} + \alpha\nabla J(\boldsymbol{\theta}_{k})
$$
但是由于$J(\boldsymbol{\theta})$关于参数$\boldsymbol{\theta}$的性质十分不好，很容易出现参数一次更新过大而跑出峰值，即overshooting问题。一个很自然的想法就是通过限制一次更新的大小$\Delta \boldsymbol{\theta}$来防止overshooting
$$
\Delta\boldsymbol{\theta}^*=\underset{\|\Delta\boldsymbol\theta\|\leq\epsilon}{\operatorname*{\arg\max}}J(\boldsymbol\theta+\Delta\boldsymbol\theta)
$$
但是实际上，由于无法确定目标函数$J(\boldsymbol\theta)$到底是有多敏感的，并且实际上就是有可能十分敏感。限制一个很小的$\epsilon$会导致十分缓慢的收敛速度（即产生了undershooting问题），同时也不能保证避免overshooting的问题。下图解释了为什么即使$\epsilon$很小也会造成目标函数的剧烈变化

![image-20240325191238846](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240325191238846.png)

通常我们认为，参数$\boldsymbol\theta$先形成$\pi_{\boldsymbol\theta}$，再形成$J(\boldsymbol\theta)$，而前者是导致overshooting问题的罪魁祸首，后者的微小变化直觉来看不会对目标函数造成巨大的波动。$\pi_{\boldsymbol\theta}$是一个pdf，不同的参数对pdf的影响不同，如上图左右两个图像在参数空间中距离是相同的，但是反映在pdf的差异上却截然不同，直观上，左边的差异程度要远大于右边。

既然不能通过限制参数空间的距离来把握目标函数的变化，同时目标函数对于策略这个pdf的敏感程度是不高的，故我们可以通过限制pdf的差异来把握目标函数的变化
$$
\Delta\boldsymbol{\theta}^*=\underset{\mathcal{D}_{\text{KL}}(\pi_\boldsymbol\theta\|\pi_{\boldsymbol\theta + \Delta\boldsymbol\theta})\leq\epsilon}{\operatorname*{\arg\max}}J(\boldsymbol\theta+\Delta\boldsymbol\theta)
$$
其中（因为此时是每一个状态动作对更新一次，所以计算KL散度所用到的都是对应状态的策略）
$$
\mathcal{D}_{\text{KL}}(\pi_\boldsymbol\theta\|\pi_{\boldsymbol\theta + \Delta\boldsymbol\theta}) = \mathbb{E}_{\pi_{\boldsymbol\theta}}\big[ \ln \frac{\pi_\boldsymbol\theta(x)}{\pi_{\boldsymbol\theta + \Delta\boldsymbol\theta}(x)} \big]
$$
表示KL散度，其能够衡量两个pdf之间的距离，在$\Delta\boldsymbol\theta$很小的时候，其二阶导就为$\pi_{\boldsymbol\theta}$的Fisher Information Matrix（下面也有相应证明），即
$$
I(\boldsymbol\theta)=\nabla_\boldsymbol\theta^2\mathcal{D}_{\text{KL}}(\pi_\boldsymbol\theta(x)\mid\mid\pi_{\boldsymbol\theta+\Delta\boldsymbol\theta}(x))|_{\Delta\boldsymbol\theta=0}
$$
我们将限制策略差异的优化表达式进行拉格朗日松弛，即
$$
\Delta\boldsymbol{\theta}^*=\underset{\Delta\boldsymbol\theta}{\operatorname*{\arg\max}} \Big[J(\boldsymbol\theta+\Delta\boldsymbol\theta)  - \lambda \big(\mathcal{D}_{\text{KL}}(\pi_\boldsymbol\theta\|\pi_{\boldsymbol\theta + \Delta\boldsymbol\theta} ) - \epsilon\big) \Big]
$$
为了求解出$\Delta\boldsymbol\theta$，我们将上述表达式对$\Delta\boldsymbol\theta$进行求导，并让其为0。为了方便求解，在求导之前，先进行泰勒近似，此时令$\boldsymbol\theta=\boldsymbol\theta_{\text{old}}+\Delta\boldsymbol\theta $
$$
\begin{aligned}
\Delta\boldsymbol{\theta}^* &= \underset{\Delta\boldsymbol\theta}{\operatorname*{\arg\max}} \Big[
J(\boldsymbol\theta_{\text{old}}) + \nabla J(\boldsymbol\theta) \Big|_{\boldsymbol\theta = \boldsymbol\theta_{\text{old}}} \Delta\boldsymbol\theta - \lambda \big(\frac{1}{2} \Delta\boldsymbol\theta^{\top} \nabla^2\mathcal{D}_{\text{KL}}(\pi_{\boldsymbol\theta_{\text{old}}}  \mid\mid \pi_{\boldsymbol\theta} ) \Big|_{\boldsymbol\theta = \boldsymbol\theta_{\text{old}}} \Delta\boldsymbol\theta - \epsilon\big)  \Big] \\
&= \underset{\Delta\boldsymbol\theta}{\operatorname*{\arg\max}} \Big[
J(\boldsymbol\theta_{\text{old}}) + \nabla J(\boldsymbol\theta) \Big|_{\boldsymbol\theta = \boldsymbol\theta_{\text{old}}}  \Delta\boldsymbol\theta - \frac{\lambda }{2} \Delta\boldsymbol\theta^{\top} I(\boldsymbol\theta_{\text{old}}) \Delta\boldsymbol\theta + \lambda\epsilon  \Big] \\
\end{aligned}
$$

注意，对于KL散度来说，其0阶和1阶导数都为0。通过一阶导求其极大值，此时再令$\boldsymbol\theta = \boldsymbol\theta_{\text{old}}$
$$
\begin{aligned}
0 &= \frac{\partial}{\partial \Delta\boldsymbol\theta}\Big[J(\boldsymbol\theta) + \nabla J(\boldsymbol\theta)   \Delta\boldsymbol\theta - \frac{\lambda }{2} \Delta\boldsymbol\theta^{\top} I(\boldsymbol\theta) \Delta\boldsymbol\theta + \lambda\epsilon \Big] \\
&= \nabla J(\boldsymbol\theta) - \lambda I(\boldsymbol{\theta}) \Delta\boldsymbol\theta
\end{aligned}
$$
由此
$$
\Delta\boldsymbol\theta = \frac{1}{\lambda} I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)
$$
注意，此时得出来的结果仅仅代表目标函数二阶泰勒近似后的最优值，并不代表原目标函数的最优值。同时这个最优值是从原目标函数增加一个KL散度惩罚项而来的，根据拉格朗日的对偶性，可以认为这个最优值就是原来参数在某个KL散度限制下的置信域最优值的近似。这个置信域的大小受拉格朗日系数$\lambda$确定，而其在最优解中仅仅以一个系数呈现。若我们忽略这个系数，并更改为另外一个系数$\alpha$，则这个$\alpha$从某种程度上就是代表了置信域的大小。简单来说，目标函数增加一个惩罚项使得最优值限制在某个置信域内，但是这个置信域的大小无从确定，由$\lambda$确定；同时，二阶的泰勒近似造成了这个最优值在远处有一定的偏差，好处就是KL散度的惩罚项变为了可求的FIM；最后，我们可以通过省略$\lambda$并且增加$\alpha$的方式来控制这个置信域的大小。也就是说，**自然策略梯度就是使用了KL散度作为置信域的梯度优化算法**，其更新表达式为
$$
\boldsymbol{\theta}_{k + 1} = \boldsymbol{\theta}_{k} + \alpha\tilde\nabla J(\boldsymbol{\theta}_{k})
$$
其中
$$
\tilde{\nabla}J(\boldsymbol\theta) = I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)
$$
对于常数$\alpha$来说，由于我们限制了KL散度的范围为$\epsilon$，由此可反解出$\alpha$
$$
\begin{aligned}
\epsilon &= \mathcal{D}_{\text{KL}}(\pi_\boldsymbol\theta\|\pi_{\boldsymbol\theta + \Delta\boldsymbol\theta}) \\
&= \frac{1}{2}\Delta \boldsymbol\theta^{\top} I(\boldsymbol\theta) \Delta \boldsymbol\theta \\
&= \frac{1}{2} \big[\alpha I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta) \big]^{\top}  I(\boldsymbol\theta) \alpha I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta) \\
&= \frac{\alpha^2}{2} \nabla J^{\top}(\boldsymbol\theta) I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)
\end{aligned}
$$
故
$$
\alpha = \sqrt{\frac{2 \epsilon}{\nabla J^{\top}(\boldsymbol\theta) I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)}}
$$
综上所述，自然策略梯度相比于普通的策略梯度，其在$\nabla J(\boldsymbol\theta)$前增加了FIM的逆进行修正，同时拥有了一个动态变化的步进，使其参数的更新不会使得策略的变化在给定的范围内。

关于物理实现，最终要的就是解出$I(\boldsymbol\theta)$，我们知道，除了二阶导的表达式，FIM还可由以下方式计算得出
$$
I(\boldsymbol\theta) = \mathbb{E}\big[\nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top} \big]
$$
其中$\pi_{\boldsymbol\theta}$为当前参数下的策略，这个是已知的，由于一个episode包含多个更新，即每一个状态动作对都更新一次，则
$$
\hat{I}(\boldsymbol\theta) = \nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top}
$$

综上所述，传统的PG算法将目标函数在当前参数进行一阶近似，并使用固定步长前进的优化算法。NPG在PG的基础上增加了关于策略层面的置信域，该置信域使用二阶近似，目标函数仍然使用一阶近似。在没有其他保证的情况下，目标函数的一阶近似可能会导致不好的结果，示意图如下

![image-20240326161452215](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240326161452215.png)

首先，在初始点中，目标函数进行了一阶近似；同时，我们通过KL散度的限制得到了置信域的范围，这个置信域的范围就是目标函数变化不多的区域，虽然KL散度也进行了二阶近似，但是认为二阶近似的误差省略，最主要的误差来源于使用样本估计FIM的误差，这个误差的影响就是超过KL散度限制的区域也被包含进来了或在KL散度限制内的区域没被包含进来。由此，NPG并不能保证更新之后目标函数一定更优。同时，FIM和其逆还有很大的存储量和计算量的要求。

### Trust Region Policy Optimization

首先进行一波前置知识的学习

**Fisher Information Matrix**
$$
I(\theta) = -\mathbb{E}\big[ \frac{\partial^2 \ln p(x; \theta)}{\partial \theta^2}  \big] = \mathbb{E}\big[ \big(\frac{\partial \ln p(x; \theta)}{\partial \theta} \big)^2 \big]
$$
**KL Divergence**
$$
\mathcal{D}_{KL}(p(x; \theta) \mid\mid p(x; \theta + \delta)) = \mathbb{E}[\ln \frac{ p(x; \theta)}{ p(x; \theta + \delta)} ]
$$
对KL表达式中的$\ln p(x; \theta + \delta)$二阶泰勒展开
$$
\ln p(x; \theta + \delta) \approx \ln p(x; \theta) + \frac{\partial}{\partial \theta} \ln p(x; \theta) \delta + \frac{\partial^2}{\partial \theta^2} \ln p(x; \theta) \frac{\delta^2}{2}
$$
由此
$$
\begin{aligned}
\mathcal{D}_{\text{KL}}(p(x; \theta) \mid\mid p(x; \theta + \delta)) &= \mathbb{E}[ \ln\frac{ p(x; \theta)}{ p(x; \theta + \delta)} ] \\
&= -\mathbb{E}\big[\ln p(x; \theta) + \frac{\partial}{\partial \theta} \ln p(x; \theta) \delta + \frac{\partial^2}{\partial \theta^2} \ln p(x; \theta) \frac{\delta^2}{2} - \ln p(x; \theta) \big] \\
&=  - \mathbb{E}\big[\frac{\partial}{\partial \theta} \ln p(x; \theta) \big] \delta - \mathbb{E}\big[\frac{\partial^2}{\partial \theta^2} \ln p(x; \theta) \big]\frac{\delta^2}{2} \\
&= I(\theta) \frac{\delta^2}{2}
\end{aligned}
$$
这就是Fisher信息和KL散度的关系，也即$I(\theta)$就是衡量了两个无穷接近参数下pdf的差异程度的极限。这个差异程度可以用KL散度来表明，其中相差了$\frac{\delta^2}{2}$这个系数。

**关于优势函数在新旧策略$\tilde{\pi}$和$\pi$下的恒等式**
$$
\sum\limits_{t = 0}^{\infty} \gamma^t a_{\pi}(S_t, \pi'(S_t)) = \eta(\tilde{\pi}) - \eta(\pi)
$$
其中$\eta(\pi)$在第二章提过（其实是刚加上的），这里第一次提到优势函数advantage function，其表达式为
$$
a_{\pi}(S_t, A_t) = q_{\pi}(S_t, A_t) - v_{\pi}(S_t, A_t)
$$
具体含义为在状态$S_t$选择$A_t$后所获得的奖励相对于平均奖励有多少优势，其尺度与普通奖励的尺度相同。

我们现在的目标是判别新旧策略$\tilde{\pi}$和$\pi$孰优孰弱，在Policy Improvement的想法是对于每一个状态$s$，都有$v_{\pi}(\tilde{s})\geq v_{\pi}(s)$，这是一个很强的约束。在此处，我们对于两个策略之间的判断是使用$\eta(\tilde\pi) \geq \eta(\pi)$来进行的，相比于前面的方式，这个对于更优策略的要求要弱很多。故将两个策略下的价值函数做差
$$
\begin{aligned}
\eta(\tilde\pi) - \eta(\pi)&= v_{\tilde\pi}(s_0) - v_{\pi}(s_0) \\
&= \mathbb{E}_{\tilde\pi}\big[\sum\limits_{t = 0}^{\infty} \gamma^t R_{t + 1}\big] - \mathbb{E}_{\pi}\big[\sum\limits_{t = 0}^{\infty} \gamma^t R_{t + 1}\big] \\
&= \sum\limits_{t = 0}^{\infty} \gamma^t \Big(\mathbb{E}_{\tilde\pi}\big[ R_{t + 1} \big] - \mathbb{E}_{\pi}\big[ R_{t + 1} \big]\Big)
\end{aligned}
$$
现在考虑一个优势函数，该优势函数整体是在旧策略$\pi$的条件下生成的，但其最开始的一步是由新策略$\tilde\pi$生成的，这与Policy Improvement Theorem中的证明很类似
$$
\begin{aligned}
a_{\pi}(S_t, \tilde\pi(S_t)) &= q_{\pi}(S_t,\tilde\pi(S_t)) - v_{\pi}(S_t) \\
&= \mathbb{E}_{\tilde\pi}[R_{t + 1} + \gamma v_{\pi}(S_{t + 1})] - v_{\pi}(S_{t + 1}) \\
&= \mathbb{E}_{\tilde\pi}[R_{t + 1}] + \gamma v_{\pi}(S_{t + 1})- v_{\pi}(S_{t + 1}) \\
&= \mathbb{E}_{\tilde\pi}[R_{t + 1}] - \mathbb{E}_{\pi}[R_{t + 1}]
\end{aligned}
$$
故将上面两个式子结合，可得到
$$
\eta(\tilde\pi) - \eta(\pi) = \sum\limits_{t = 0}^{\infty} \gamma^t \Big[a_{\pi}(S_t, \tilde\pi(S_t))\Big]
$$
故新旧策略平均回报之差为很多个以新策略为第一步，剩余都是旧策略的优势函数，其中所有的$S_t$都是由旧策略产生的，故原式子还可以写为
$$
\eta(\tilde\pi) - \eta(\pi) = \mathbb{E}_{\tilde\pi}\bigg[\sum\limits_{t = 0}^{\infty} \gamma^t \Big[a_{\pi}(S_t, A_t)\Big]\bigg]
$$
需要注意的是，上式等号右边的轨迹遵循的是策略$\tilde{\pi}$，也就是产生的所有$S_t$和$A_t$都是由策略$\tilde{\pi}$造成的，相比于使用$\tilde{\pi}(S_t)$的表达式，$\mathbb{E}_{\tilde{\pi}}$更能体现$\tilde{\pi}$产生的轨迹这个概念。但是为什么会产生两个咋一看挺对，仔细一想又不太对（整体的轨迹和局部的轨迹之间的差别），但是最终还是对的等式呢？原因在于在计算$\eta(\tilde\pi) - \eta(\pi)$和$\mathbb{E}_{\tilde\pi}\big[ R_{t + 1} \big] - \mathbb{E}_{\pi}\big[ R_{t + 1} \big]$之间关系时并没有强调轨迹是由$\pi$还是$\tilde{\pi}$产生的，但是前后状态的关系一定要紧密（前后关状态是由同一个策略产生的）。但是在得到$a_{\pi}(S_t, \tilde\pi(S_t))$时，每一个局部轨迹都是由$\tilde{\pi}$产生的，又由于$\eta(\tilde\pi) - \eta(\pi)$要求我们状态要紧密，所以最后整体的轨迹都又$\tilde{\pi}$产生。

**TRPO**
若目标函数$J(\boldsymbol\theta)$是由$\pi_{\boldsymbol\theta}$（不敏感）控制，且$\pi_{\boldsymbol\theta}$由是由$\boldsymbol\theta$（敏感）控制，这种目标函数的形成方式符合强化学习中基于策略的大多数目标函数。则我们可以使用NPG来替代传统的PG，以防止由于$\pi_{\boldsymbol\theta}$对$\boldsymbol\theta$的过度敏感而导致更新的失败。NPG告诉我们，为了限制参数$\boldsymbol{\theta}$的改变对策略$\pi$有太多的影响，在更新时对$\boldsymbol\theta$的变化范围定义一个与KL散度有关的置信域，并在这个置信域中找到最佳值。最终在对目标函数和约束条件都使用了泰勒近似的情况下，使得NPG的更新表达式如下，此时一个episode只更新一次，$I(\boldsymbol\theta)$变为所有状态下pdf的FIM的均值（相应的约束变为了平均KL散度），$J(\boldsymbol\theta)$变为所有状态动作对下的优势函数与策略乘积之和
$$
\boldsymbol{\theta}_{k + 1} = \boldsymbol{\theta}_{k} + \sqrt{\frac{2 \epsilon}{\nabla J^{\top}(\boldsymbol\theta) I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)}} I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)
$$
但是由于对目标函数使用了一阶近似，对约束条件使用了二阶近似，更新后的参数

1. 无法保证能够有效的提高目标函数；

2. 无法保证更新后的参数满足最初的KL散度限制。

一种思路就是在发现有上述问题的时候，将置信域缩小，看看缩小后得到的新解能否在满足最初KL散度限制下的情况下，还能使得性能提升。按照这种思路，则面临三个问题

1. 如何缩小置信域，并在新的置信域上取得最优值；
2. 如何判断新解满足KL限制；
3. 如何判断新解能够提升性能。

对于第一个问题，减小置信域的等价结论是缩小步长。原因是一阶近似的目标函数一定是与当前参数相切的单调平面，改变置信域的大小不能改变这个单调平面，但是却限制了在这个单调平面的最优值的位置，这个位置一定与之前置信域下的最优值共线。所以可以简单通过缩小步长来同时实现缩小置信域和在置信域上去的新最优值的操作。

对于第二个问题，我们只需要在对两个参数下得到的分布做一次KL散度的计算即可。这个具体的KL散度计算，需要使用旧参数产生的轨迹中的所有状态，对这些状态对应的KL散度都进行新旧策略下的KL散度计算，最终再取平均，即使用下式
$$
\begin{aligned}
\mathcal{\bar D}_{\text{KL}}^{\rho_{\boldsymbol\theta_{\text{old}}}}(\boldsymbol\theta_{\text{old}}, \boldsymbol{\theta}) &= \mathbb{E}_{s \sim \rho_{\boldsymbol\theta_{\text{old}}}}\big[ \mathcal{D}_{\text{KL}}\big( \pi_{\boldsymbol\theta_{\text{old}}}(\cdot \mid s_t) \mid\mid \pi_{\boldsymbol\theta}(\cdot \mid s_t)\big ) \big] \\
&\approx \frac{1}{T} \sum\limits_{t = 0}^{T - 1} \mathcal{D}_{\text{KL}}\big( \pi_{\boldsymbol\theta_{\text{old}}}(\cdot \mid s_t) \mid\mid \pi_{\boldsymbol\theta}(\cdot \mid s_t)\big )
\end{aligned}
$$
由于此时对KL散度的计算只存在样本近似，所以相比于最初更新时使用的二阶泰勒近似，近似程度要好不少。

对于第三个问题，首先排除的做法是使用新策略再获取一些数据，来判定这些数据的性能好不好。故我们希望能使用旧策略下的样本数据来评判新策略到底好不好，TRPO是使用如下关系进行的（这个等式前面证明过）
$$
\begin{aligned}
\eta(\tilde\pi) - \eta(\pi) &= \mathbb{E}_{\tilde\pi}\bigg[\sum\limits_{t = 0}^{\infty} \gamma^t \Big[A_{\pi}(s_t, a_t)\Big]\bigg] \\
&= \sum\limits_{t = 0}^{\infty} \sum\limits_{s} \gamma^t \Pr(s_0 \rightarrow s, t, \tilde{\pi}) \sum\limits_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a)
\end{aligned}
$$
令
$$
\rho_{\tilde{\pi}}(s) = \sum\limits_{t = 0}^{\infty} \gamma^t \Pr(s_0 \rightarrow s, t, \tilde{\pi})
$$
则
$$
\begin{aligned}
\eta(\tilde\pi) - \eta(\pi) &= \sum\limits_{s} \rho_{\tilde{\pi}}(s) \sum\limits_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a) \\
&= \mathbb{E}_{s_t \sim \rho_{\tilde{\pi}}, a_t \sim \tilde{\pi}}\big[ A_{\pi}(s_t, a_t) \big]
\end{aligned}
$$
若只是对新旧策略进行评判，我们只需要让右式与0进行比较即可。观察右边的式子，虽然优势函数$A_{\pi}$是在旧策略$\pi$下的，但是其却要求所有出现的$s_t$和$a_t$是在新策略下产生的，所以最终无法计算。这里，我们进行一次近似，将$\rho_{\tilde{\pi}}$替换为$\rho_{\pi}$，也即
$$
\begin{aligned}
\eta(\tilde\pi) - \eta(\pi) &= \sum\limits_{s} \rho_{\tilde{\pi}}(s) \sum\limits_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a)\\
& \approx \sum\limits_{s} \rho_{\pi}(s) \sum\limits_{a} \tilde{\pi}(a \mid s) A_{\pi}(s, a) \\
&= \mathbb{E}_{s_t \sim \rho_{\pi}}\big[\sum\limits_{a} \tilde{\pi}(a \mid s_t) A_{\pi}(s_t, a) \big] \\
&= \mathbb{E}_{s_t \sim \rho_{\pi}, a_t \sim \pi} \big[\frac{\tilde{\pi}(s_t \mid a_t)}{\pi(s_t \mid a_t)} A_{\pi}(s_t, a_t) \big] \\
&= \mathcal{L}_{\pi}(\tilde{\pi})
\end{aligned}
$$
表达式前后满足零阶和一阶相等，所以该近似在局部可认为与原式相等。同时注意到最后一行，我们使用了重要性采样，重要性采样不影响随机变量的无偏性，但其带来了更大的方差。经过近似，在新旧策略和就策略下的轨迹都直到的情况下，我们就能够将其计算出来，由此解决了第三个问题。

最后需要注意的是，在之前的讨论中，我们并没有规定$\nabla J(\boldsymbol\theta)$是以何种形式出现的。在TRPO的原文中，其最终优化的是在约束$\mathcal{\bar D}_{\text{KL}}^{\rho_{\boldsymbol\theta_{\text{old}}}}(\boldsymbol\theta_{\text{old}}, \boldsymbol{\theta}) \leq \epsilon$下的$\mathcal{L}_{\boldsymbol\theta_{\text{old}}}(\boldsymbol\theta)$的最优值$\boldsymbol\theta$，所以其本意是直接对$\mathcal{L}_{\boldsymbol\theta_{\text{old}}}(\boldsymbol\theta)$进行优化，但是这个表达式中包含了新策略，所以没法直接对其进行优化，看了代码后，发现代码直接将分子新策略置为旧策略，也即
$$
\begin{aligned}
\mathcal{L}_{\boldsymbol\theta_{\text{old}}}(\boldsymbol\theta) &= \mathbb{E}_{s_t \sim \rho_{\pi}, a_t \sim \pi} \big[ \frac{\pi_{\boldsymbol\theta}(s_t \mid a_t)}{\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)} A_{\pi}(s_t, a_t) \big] \\
& \approx \mathbb{E}_{s_t \sim \rho_{\pi}, a_t \sim \pi} \big[ \frac{\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)}{\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)} A_{\pi}(s_t, a_t) \big]
\end{aligned}
$$
同时将分母的旧策略变为常数，由此，在每个episode的一次更新中，梯度为
$$
\begin{aligned}
\nabla \mathcal{L}_{\boldsymbol\theta_{\text{old}}}(\boldsymbol\theta) &\approx \mathbb{E}_{s_t \sim \rho_{\pi}, a_t \sim \pi} \big[ \frac{\nabla\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)}{\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)} A_{\pi}(s_t, a_t) \big] \\
&\approx \frac{1}{T} \sum\limits_{t = 0}^{T - 1} \frac{\nabla\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)}{\pi_{\boldsymbol\theta_\text{old}}(s_t \mid a_t)} A_{\pi}(s_t, a_t)\\
&= \nabla J(\boldsymbol\theta)
\end{aligned} \\
$$
正好就是普通的策略梯度更新表达式，所以我们认为TRPO就是添加了线性搜索和验证的功能的NPG。

关于为什么同样使用了近似，并且同样是在KL散度的约束下，使用NPG算出来的结果就存疑，而使用$\mathcal{L}_{\boldsymbol\theta_{\text{old}}}(\boldsymbol\theta)$算出来的结果就认为是正确的呢？这当然有近似多少的问题，但是最主要的数学原理在于，后者在KL约束的条件下，是原目标函数$\eta(\tilde\pi)$的下届，即
$$
\begin{gathered}
\eta(\tilde{\pi})\geq L_{\pi}(\tilde{\pi})-C\mathcal{D}_{\text{KL}}^{\max}(\pi,\tilde{\pi}),\\\mathrm{where~}C=\frac{4\epsilon\gamma}{(1-\gamma)^2}
\end{gathered}
$$
将$\mathcal{D}_{\mathrm{KL}}^{\max}(\pi,\tilde{\pi})$近似为$\mathcal{\bar D}_{\text{KL}}^{\rho_{\tilde\pi}}(\pi,\tilde{\pi})$，并将系数$C$根据拉格朗日对偶性转化为约束条件，由此问题变为了
$$
\begin{gathered}
\max_\theta \mathcal{L}_{\boldsymbol\theta_\text{old}} ( \boldsymbol\theta ) \\ 
\text{subject to}~\mathcal{\bar{D}}_{\text{KL}} ^ { \rho _ { \boldsymbol\theta _ \text{old }}}(\boldsymbol\theta_{\text{old}},\boldsymbol\theta)\leq\delta
\end{gathered}
$$
现在说一下如何求FIM逆的问题，先说如何计算FIM，前面也提到了，可以使用下式进行计算
$$
\hat{I}(\boldsymbol\theta) = \frac{1}{T}\sum\limits_{t = 0}^{T - 1}\nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top}
$$
但是由于$I(\boldsymbol\theta)$的维数过于大，所以通常不直接求出FIM并表示出来。我们可以使用conjugate gradient方法来求出其逆与某个向量相乘的结果，其基本的形式为
$$
A\mathbf{x} = \mathbf{b}
$$
若我们已知$\mathbf{b}$，且对于任意向量$\mathbf{v}$，我们都可以计算出$A \mathbf{v}$（注意此时不要求将矩阵$A$完整表示出来，只要求出其对应向量的乘积即可），则我们可以使用共轭梯度法求出$\mathbf{x} = A^{-1} \mathbf{b}$。那么具体到求$I^{-1}(\boldsymbol\theta) \nabla J(\boldsymbol\theta)$的问题，只需要能够知道$I(\boldsymbol\theta)$与任意向量相乘的结果就行。

那么我们如何求出$I(\boldsymbol\theta)$​与任意向量相乘的结果呢？下面这个方法通过FIM的定义式得到
$$
\begin{aligned}
I(\boldsymbol\theta) \mathbf{v} &= \mathbb{E}\big[\nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top} \big] \mathbf{v} \\
&= \mathbb{E}\big[\nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \big(\nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top} \mathbf{v} \big) \big]  \\
& \approx \frac{1}{T}\sum\limits_{t = 0}^{T - 1}\nabla{ \ln \pi_{\boldsymbol\theta} }(a_t \mid s_t) \big( \nabla{ \ln \pi_{\boldsymbol\theta}(a_t \mid s_t) } ^ {\top} \mathbf{v} \big)
\end{aligned}
$$
第二个方法通过KL散度的Hessian矩阵来求出FIM。第一步，先将KL散度的一阶导与任意向量点积；第二步，在对第一步得到的点积结果求导，得到的向量即为FIM与对应任意相乘的结果，具体的数学表达式太麻烦了，这里就不写了。

对比这两个方法，会发现方法一同时使用了状态和动作，而方法二只是用了状态 ，因为只是用状态相当于知道了当前状态所有动作的分布，所以我认为方法二会更精准一点。



