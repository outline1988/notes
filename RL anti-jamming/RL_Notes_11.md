## Chapter 11 Off-policy Methods with Approximation

 写到这里，我试图想重新归纳并更加深刻地来理解前面所学的那么多的算法，包括Sarsa、Expected Sarsa和Q-learning等。

**Understanding from the Perspective of GPI**
广义策略迭代（Generalized Policy Iteration，GPI）以两种过程交替进行，一是Policy Evaluation，二是Policy Improvement。其能使得策略最终收敛为最优策略的核心是Policy Improvement theorem，即使用greedy方式对策略进行改进一定能使得策略更好。GPI中Policy Evaluation的正确与否是价值函数最终能否收敛的关键，动态规划下Policy Evaluation以下式进行迭代
$$
q_{k + 1}(s, a) = \sum\limits_{s', r} p(s', r \mid s, a)(r + \sum\limits_{a'}\pi(a' \mid s')q_{k}(s', a'))
$$
其依赖于是bellman equation的不动点的性质。再经过策略改进定理并循环往复，便能使得策略收敛至最优策略。

在具体model-free的实现中，多数算法将动态规划中需要环境动态特性的迭代式转化为依赖于样本的迭代，其背后的数学原理是大数定理（这里写一个思考：任何关于PDF的表达式，其都能通过大量的样本来进行体现）。Sarsa和Expected Sarsa都可以用这种方式理解（包括增加bootstrap深度的$n$-step TD和$n$​-step Tree Backup算法）。

聚焦于predicition的问题，将tabular拓展至function approximation。函数近似可以简述为在于大量的样本数据下，使用特定模型（线性）使得损失函数$\overline{\mathrm{VE}}(\mathbf{w})$最小，损失函数表达式如下
$$
\overline{\mathrm{VE}}(\mathbf{w})\doteq\sum_{s\in\mathcal{S}}\mu(s)\left[v_\pi(s)-\hat{v}(s,\mathbf{w})\right]^2
$$
在on-policy的情况下，使用策略$\pi$生成的样本数据对$\overline{\mathrm{VE}}(\mathbf{w})$进行随机梯度下降（SGD），由于样本数据与$\mu(s)$和$v_{\pi}(s)$ 均源自于策略$\pi$，故其能够满足prediction中收敛的要求（即使是semi-gradient descent，也有相关证明证明其能收敛），从而找到最优$\mathbf{w}$​。

off-policy使用behavior policy生成的样本数据对target policy的价值函数来进行更新，behavior policy条件的增加，价值函数收敛的难度直线上升。off-policy的收敛，面临两个挑战，一是要处理更新目标不同的问题，即用bahavior policy的数据原则上应该用以对应价值函数的更新，而不是另外target policy的价值函数，$\overline{\mathrm{VE}}(\mathbf{w})$中$v_{\pi}(s)$​的要求无法达到。万幸的是，重要性采样将两个不同策略下的数据进行转换，解决了这个问题，但随之而来的代价是样本数据的方差增大。此外，强行使用target policy对behavior policy下的样本数据进行Expected Sarsa也能缓解这个问题带来的不收敛。

第二个问题由funtion approximation造成，注意看$\overline{\mathrm{VE}}(\mathbf{w})$表达式中的$\mu(s)$，其应该为target policy下的状态分布，直接使用bahavior policy生成的样本数据必然带来$\mu(s)$的失配，从而导致prediction时的不收敛。

**Understanding from the Perspective of Law of Large Numbers**
前面提到了用大数定理的思想将由bellman equation转换而来的迭代式转换为了使用样本数据的更新表达式，以同样的方式对bellman optimality equation进行处理。首先将bellman optimality equation转换为value iteration的迭代表达式
$$
q_{k + 1}(s, a) = \sum\limits_{s', a} p(s', r \mid s, r)(r + \max\limits_{a'} q_{k}(s', a'))
$$
由此我们能够直接转为Q-learning的更新的目标（注意真正的Q-learning是以$Target$为目标的增量式更新）
$$
Target = R_{t + 1} + \max\limits_a Q(S_{t + 1}, a)
$$
bellman optimality equation的不动点性质，保证了其能最终收敛。

### 11.1 Semi-gradient Methods

如果我们忽略第二个挑战，即behavior policy生成的样本数据无法匹配target policy的状态分布，从而导致在function approximation中的发散（事实上，一些场合下这些方法能够收敛）。并且使用前面几章tabular中克服第一个挑战的方法，如重要性采样等，就可以很自然地得到如下off-policy $n$-step Sarsa更新表达式
$$
\mathbf{w}_{t+n}=\mathbf{w}_{t+n-1}+\alpha\rho_{t+1}\cdots\rho_{t+n-1}\left[G_{t:t+n}-\hat{q}(S_t,A_t,\mathbf{w}_{t+n-1})\right]\nabla\hat{q}(S_t,A_t,\mathbf{w}_{t+n-1})
$$
其中
$$
\rho_{t + k} = \frac{\pi(A_{t + k} \mid S_{t + k})}{b(A_{t + k} \mid S_{t + k})}
$$
特别的，对于one-step Expectation Sarsa来说，更新表达式如下
$$
\mathbf{w}_{t+1}=\mathbf{w}_{t}+\alpha\delta_t\nabla\hat{q}(S_t,A_t,\mathbf{w}_{t})
$$
其中
$$
\delta_t = R_{t + 1} + \sum\limits_a\pi(a \mid S_{t + 1}) \hat{q}(S_{t + 1},a,\mathbf{w}_{t})  -\hat{q}(S_t,A_t,\mathbf{w}_{t})
$$
one-step Sarsa无法使用重要性采样，因为无论对于target policy还是behavior policy，对状态动作对$S_t$和$A_t$来说，选择$A_t$都是绝对的。

### 11.2 Examples of Off-policy Divergence

我们现在来面对导致发散的第二个挑战，即off-policy和on-policy不同进而状态分布不同所导致的价值函数无法收敛的问题。behavior policy存在的意义是为了弥补target policy探索性的不足，所以behavior policy可能能够生成的轨迹一定包括target policy的轨迹，换句话说，behavior policy在完全能够覆盖target policy的同时还可能生成其他的轨迹，而这些target policy无法生成的轨迹最终体现在这些样本数据无法用于target policy的更新。所以就会存在这样一种情况：在behavior生成的一长串轨迹中，只有零散的部分轨迹能够用于最终的更新，这在tabular问题中不会产生什么问题，因为此时Q表上的每一个元素都是独立地更新的。

然而，若使用了function approximation，由于其自带有的泛化的能力，可能导致一个状态价值函数的增大会造成另外一个状态相应增大，例如下面这个例子

![image-20240226154952341](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240226154952341.png)

若权重$w$的初始权重为正数，且状态的转移产生的奖励都为0，则使用TD(0)或是DP等自举的方法对从左到右的状态转移进行价值函数的更新，会使得两个状态的价值函数同时增大。若正好下次产生的数据，即从右边状态转移到某个状态的轨迹是target policy无法完成的，那么右边状态的价值函数就永远不会发生更新，从而只存在左边状态转移到右边状态，最终价值函数发散。
