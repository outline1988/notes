## Chapter 10 On-policy Control with Approximation

现在，我们需要将prediction问题转换到control的问题上，前面对于状态价值的函数的估计也相应转换为对于动作价值函数的估计，也即$\hat{q}(s, a, \mathbf{w}) \approx q_{*}(s, a)$。

### 10.1 Episodic Semi-gradient Control & 10.2 Semi-gradient $\boldsymbol{n}$-step Sarsa

将Sarsa算法从表格型拓展至function approximation是一件十分直观的事情，因为只需要将Q表更换为一个近似函数，将每次与环境互动得到的$S_t, A_t \mapsto U_t$作为近似函数的训练样本进行更新，更新表达式如下
$$
\mathbf{w}_{t + 1} = \mathbf{w}_{t} + \alpha \big[R_{t + 1} + \gamma \hat{q}(S_{t + 1}, A_{t + 1}, \mathbf{w}_{t}) - \hat{q}(S_t, A_t, \mathbf{w}_{t})\big] \nabla \hat{q}(S_t, A_t, \mathbf{w}_{t})
$$
并且与之配套的，如何选择动作也需要进行轻微的修改，在这里只需要遍历所有动作，得到当前状态的所有动作价值后使用$\varepsilon$-greedy选择动作即可。注意，**遍历的做法势必使得当前的操作需要在动作空间离散且不十分巨大**的情况下进行；同时，有前面的章节可以知道，此为semi梯度更新。伪代码如下

![image-20240223200436623](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223200436623.png)

从伪代码可以看到，Sarsa with function approximation与表格型的Sarsa算法区别不大，仅在涉及Q表的地方转换为了近似函数。

**Example 10.1**
很常见的一个例子：Mountain Car Task。如下图

![image-20240223200826728](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240223200826728.png)

小车在初始随机在山谷的某个地方生成，目标是通过加油、减速和不做动作使得小车能够上去最右边的山顶，期间若跑至最左边，则车子在山谷最低处以零速度生成。使得这个游戏变得更为困难的条件是车子引擎的动力无法支撑起车的重量，所以使得小车需要借助左边的惯性来冲到右边的山顶。

我们定义动作$A_t$的三个取值+1、-1和0分别代表加速、减速和不做动作，令$x_{t}$为当前的位置且限制在$[-1.2, 0.5]$，$\dot{x}_{t}$为当前的速度且限制在$[-0.07, 0.07]$，并且上述三个变量满足
$$
\begin{aligned}
x_{t+1}&\doteq \text{bound}\big[x_t+\dot{x}_{t+1}\big] \\
\dot{x}_{t+1}&\doteq \text{bound}\big[\dot{x}_t+0.001A_t-0.0025\cos(3x_t)\big]
\end{aligned}
$$
经过上述算法训练，就可得到上述的结果图。由于权重初始化为0，所以在初始的情况会具有极大的探索性，并且由于前期的训练几乎无法完成目标，所以做的任何动作都会使得对当前价值函数的预测减小，从而鼓励小车不断得探索其他的动作，表现就是回不断得在山谷之间来回摆动，因为小车的趋势是向着最低的山谷，所以可以预测价值函数的最低点就是在山谷，并且越远离山谷，价值函数的预测越高，所以小车只能往着没尝试过的两边去探索。

**Semi-gradient $\boldsymbol{n}$-step Sarsa**
将one step Sarsa拓展至$n$-step Sarsa也很自然，只要仿照tabular的$n$-step Sarsa表达式
$$
Q(S_t, A_t) = Q(S_t, A_t) + \alpha \big[ G_{t : t + n} - Q(S_t, A_t) \big]
$$
将使用随机梯度下降的参数更新表达式写为
$$
\mathbf{w}_{t + n} = \mathbf{w}_{t + n - 1} + \alpha\big[ G_{t : t + n} - \hat{q}(S_t, A_t, \mathbf{w}_{t + n - 1}) \big] \nabla \hat{q}(S_t, A_t, \mathbf{w}_{t + n - 1})
$$
其中
$$
G_{t : t + n} = R_{t + 1} + \gamma R_{t + 2} + \cdots + \gamma^{n - 1} R_{t + n} + \gamma^n \hat{q}(S_{t + n}, A_{t + n}, \mathbf{w}_{t})
$$
伪代码如下

![image-20240225192157288](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240225192157288.png)

同样将Semi-gradient $n$-step Sarsa运用到前面小车的例子，可以得到以下结果

![image-20240225192315012](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240225192315012.png)

*后面还有一些关于折扣因子的讨论，没看懂，以后再看。。。*



