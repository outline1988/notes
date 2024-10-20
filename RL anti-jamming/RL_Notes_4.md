## Chapter 4 Dynamic Programming

### 4.1 Policy Evaluation (Prediction)

现在我们将根据一个给定的策略$\pi$来计算出此时所有状态的价值（比如说$v_{\pi}(s)$）。对于一个给定的策略$\pi$和已知的环境动态$p$，我们可列出bellman方程如下
$$
\begin{aligned}
v_{\pi}(s) &= \sum\limits_a \pi(a \mid s) \sum\limits_{s', r}p(s' ,r \mid s, a)(r + \gamma v_{\pi}(s')) \\
\end{aligned}
$$
若$v_{\pi}$是真值，则其在$\gamma < 1$和有终端状态的情况下满足上述的等式（当作结论记住），也就是说，对于某个状态$s$的价值，我们可以用即时奖励的期望和所有后继状态（也就是所有状态）的价值的加权和来等效表示。那么对于所有的状态，我们就可列出$|\mathcal{S}|$的未知数的$|\mathcal{S}|$个方程，从原则上来说这样的线性方程是可以解出来的。

**Analytical Policy Evaluation**
现在通过bellman方程的一般形式来求得其矩阵形式，从而再给定$\pi$的情况下，求得其解析解
$$
\begin{aligned}
v_{\pi}(s) &= \sum\limits_a \pi(a \mid s) \sum\limits_{s', r}p(s' ,r \mid s, a)(r + \gamma v_{\pi}(s')) \\
&= \sum\limits_a \pi(a \mid s)\sum\limits_{r}p(r \mid s, a) + \gamma \sum\limits_a\pi(a \mid s)\sum_{s'}p(s' \mid s, a)v_{\pi}(s') \\
&= r_{\pi}(s) + \gamma \sum\limits_{s'}p_{\pi}(s, s')v_{\pi}(s')
\end{aligned}
$$
这里的$p_{\pi}(s, s')$就是代表MDP退化成MP后的概率转移函数。同时我们发现，上式形式同矩阵与向量相乘的表达式及其相似，故令
$$
\mathbf{v_{\pi}} = 
\begin{bmatrix}
v_{\pi}(s_1) \\
v_{\pi}(s_2) \\
\vdots \\
v_{\pi}(s_N) \\
\end{bmatrix}

,\quad

\mathbf{r_{\pi}} = 
\begin{bmatrix}
r_{\pi}(s_1) \\
r_{\pi}(s_2) \\
\vdots \\
r_{\pi}(s_N) \\
\end{bmatrix}

,\quad 

\mathbf{P_{\pi}} = 
\begin{bmatrix}
p(s_1, s_1) & p(s_1, s_2) & \cdots & p(s_1, s_N) \\
p(s_2, s_1) & p(s_2, s_2) & \cdots & p(s_2, s_N) \\
\vdots & \vdots & \ddots & \vdots \\
p(s_N, s_1) & p(s_N, s_2) & \cdots & p(s_N, s_N) \\
\end{bmatrix}
$$
则可得
$$
\mathbf{v_{\pi}} = \mathbf{r_{\pi}} + \gamma \mathbf{P_{\pi}} \mathbf{v_{\pi}} \\
\mathbf{v_{\pi}} = (\mathbf{I} - \gamma \mathbf{P_{\pi}})^{-1} \mathbf{r_{\pi}}
$$
此即为bellman方程解析解。

**Iterative Policy Evaluation**
暴力解出上述非线性方程组对于我们来说是不可接受的，所以采用迭代的方法来求出$v_{\pi}$，首先我们先给予每一个状态的价值任意的初值（因为终端状态价值始终为0，所以唯有终端状态初值必须设为0），然后使用如下的迭代表达式来进行迭代
$$
\begin{array}{rcl}
v_{k+1}(s)&\doteq&\mathbb{E}_\pi[R_{t+1}+\gamma v_k(S_{t+1})\mid S_t=s]\\
&=&\sum\limits_a\pi(a|s)\sum\limits_{s',r}p(s',r|s,a)\Big[r+\gamma v_k(s')\Big] \\
&=& r_{\pi}(s, a) + \sum\limits_{s'}p_{\pi}(s, s')v_{k}(s')
\end{array}
$$
在某一次迭代中，新值$v_{k + 1}$由即时期望和所有旧值$v_k$的线性组合（加权平均）求出，我们也称这样的更新过程为期望更新（expected updates）。可以证明，当$k \rarr \infin$时，$v_k$会最终收敛到$v_{\pi}$，初始值$v_0$是完全与策略$\pi$和环境动态$p$无关的值，随着迭代的进行，初始值$v_0$的随机成分在上述的迭代式中不断地被$\pi$和$p$稀释，最终$v_k$收敛到与初值无关的$v_{\pi}$。迭代结束的标志是$v_{k + 1}$和$v_{k}$相等，即此时$v_{\pi}=v_{k + 1} = v_{k}$，带回进入迭代式中，刚好就是bellman equation。

对于实际的迭代程序，可以使用两种写法，第一种写法使用两个数组，在迭代的过程中交替代表$v_k$和$v_{k + 1}$，也就是说，某个数组中每次迭代的$v_{k + 1}$都是完全都是由另外一个数组，也就是旧值来进行计算的；另一种写法就是只使用一个数组，对于某一个状态的期望更新也将使用当前数组的值来进行，也就是说，某个状态的期望更新不完全由旧值来决定，而是掺杂了新更新的值，从某种程度上，这种就地（in place）的算法收敛的速度会更快，当然此时数组中状态的顺序对收敛速度的影响也十分巨大。

**Example 4.1**
如下图，终止状态分别为左上和右下，余的所有状态对于任何转移，奖励都为-1，若某个动作会使得边缘状态跳出棋盘，那么仍继续保持这个状态，令策略为等概率的向四周转移，求每个状态的价值$v_{\pi}$。

![image-20231005140146521](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231005140146521.png)

以下为解答

![image-20231005140432691](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231005140432691.png)

假设所有状态的初始值都为0（当然终端状态必须为0），然后根据iterative policy evaluation来计算每个状态的迭代值，这里以状态1为例子
$$
v_{\pi}(s) = r_{\pi}(s) + \sum\limits_{s'}p_{\pi}(s, s')v_{\pi}(s') \\
\begin{aligned}
v_{1}(1) &= -1 + \frac{1}{4}(0 + 0 + 0 + 0) = 0 \\
v_{2}(1) &= -1 + \frac{1}{4}(0 -1 -1 -1) = -1.7 \\
v_{3}(1) &= -1 + \frac{1}{4}(0 - 1.7 - 2 - 2) = -2.4 \\
v_{4}(1) &= -1 + \frac{1}{4}(0 - 2.4 - 2.9 - 2.9) = -2.9\\
&\cdots
\end{aligned}
$$
以上述例子来说，每次的迭代，都会使得终端状态的价值连同即使奖励期望一起，不断地向周围侵蚀，最终使得初始值的权重不断缩小，而即使奖励和转移函数占据主导，最终收敛到$v_{\pi}$。

**Exercise 4.1**
求$q_{\pi}(11, \text{down})$和$q_{\pi}(7, \text{down})$？
$$
q_{\pi}(s, a) = r(s, a) + \sum\limits_{s'}p(s' \mid s, a)v_{\pi}(s') \\
\begin{aligned}
q_{\pi}(11, \text{down}) &= -1 + 0 = -1 \\
q_{\pi}(7, \text{down}) &= -1 + (-14) = -15
\end{aligned}
$$
**Exercise 4.3**
写出关于$q_{\pi}$的迭代表达式
$$
\begin{aligned}
q_{k+1}(s,a)& =\mathbb{E}_\pi[R_{t+1}+\gamma q_k(S_{t+1},A_{t+1})\mid S_t=s,A_t=a]  \\
&=\sum_{s^{\prime},r}p(s^{\prime},r|s,a)\left[r+\gamma\sum_{a^{\prime}}\pi(a^{\prime}|s^{\prime})q_k(s^{\prime},a^{\prime})\right]
\end{aligned}
$$

### 4.2 Policy Improvement

策略评估（policy evaluation）是在给定某个策略的情况下，通过某种方式（这里使用iterative policy evaluation）来得到所有状态的状态价值。但是强化学习的最终目的是寻找optimal policy，所以我们需要在通过给定的价值的情况下以某种方式改进原先的策略，使得新的策略能够不断地向最优价值趋近。

**Policy Improvement Theorem**
当给定的新旧策略$\pi'$和$\pi$，怎样评估这两个策略的好坏。为了简单起见，我们假设旧策略$\pi$的价值函数完全已知，同时新旧策略具有确定性，即分布函数都是冲激（deterministic，这个假设为的是后续的证明方便）。最简单的方式是，既然我们都明确知道了策略$\pi'$，那么重新使用policy evaluation的方法再求一遍价值函数$v_{\pi'}$不就行了，当然这可行，但是计算量太大，并且对改进策略没有启发作用。

回到刚才的假设，对于某一状态$s$，我们只需要判断$v_{\pi}(s)$和$v_{\pi'}(s)$的相对大小即可确定新旧策略在该状态的好坏程度。观察$v_{\pi}(s)$，这个价值函数体现了从当前状态$s$开始，以$s$的条件下根据策略$\pi(s)$选择动作后所获得回报期望。如果我们仅仅只改变其下一步的策略为$\pi'(s)$，后续的所有策略选择仍然按照$\pi$来进行，那么直觉上，从此方法得到的回报期望与改变前的回报期望做对比，当前状态的新旧策略的好坏程度就高下立判了。
$$
v_{\pi}(s) \leq q_{\pi}(s, \pi'(s))
$$
上不等式中，$q_{\pi}(s, \pi'(s))$体现的是仅仅改变下一步的策略为$\pi'(s)$后的回报期望，所以只要上述成立，就能证明对于当前状态$s$来说，策略$\pi'(s)$要好于策略$\pi(s)$。单一的状态推广到所有状态时，不等式都成立，那么就足以说明$\pi'$要好于$\pi$。

以下是证明，我们要证明的是，若对于所有状态$s$，$v_{\pi}(s) \leq q_{\pi}(s, \pi'(s))$都成立，则$v_{\pi}(s) \leq v_{\pi'}(s)$都成立
$$
\begin{aligned}
v_{\pi}(s) &\leq q_{\pi}(s,\pi'(s))  \\
&=\mathbb{E}\big[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_{t}=s,A_{t}=\pi'(s)\big]   \\
&=\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_{t}=s\big] \\
&\leq\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma q_{\pi}(S_{t+1},\pi'(S_{t+1}))\mid S_{t}=s\big]   \\
&=\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma\mathbb{E}_{\pi'}[R_{t+2}+\gamma v_{\pi}(S_{t+2})|S_{t+1},A_{t+1}=\pi'(S_{t+1})]\mid S_{t}=s\big] \\
&=\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma R_{t+2}+\gamma^2v_\pi(S_{t+2})\mid S_t=s\big] \\
&\leq\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3v_{\pi}(S_{t+3})\mid S_{t}=s\big] \\
&\leq\mathbb{E}_{\pi'}\big[R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\gamma^3R_{t+4}+\cdots\mid S_t=s\big] \\
&=v_{\pi'}(s).
\end{aligned}
$$
这个证明可以简单理解为，若对于所有状态$s$，$v_{\pi}(s) \leq q_{\pi}(s, \pi'(s))$都成立，那么我们将$q_{\pi}(s, \pi'(s))$中包含$\pi$的部分都换成$\pi'$，也即$v_{\pi}(s) \leq q_{\pi'}(s, \pi'(s))$也成立，也就是$v_{\pi}(s) \leq v_{\pi'}(s)$成立。

**Policy Improvement**
上述的结论对于我们如何改进策略$\pi$具有十分启发式的意义，我们按照下述贪心（greedy）的方式对策略进行改进
$$
\begin{array}{rcl}
\pi^{\prime}(s)&\doteq&\underset{a}{\operatorname*{argmax}}q_\pi(s,a)  \\
&=&\underset{a}{\operatorname*{argmax}}\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})\mid S_{t}=s,A_{t}=a] \\
&=&\underset{a}{\operatorname*{argmax}}\sum_\limits{s^{\prime},r}p(s^{\prime},r|s,a)\Big[r+\gamma v_\pi(s^{\prime})\Big],
\end{array}
$$
由policy improvement theorem可知，我们改进的目标是对于所有的状态，让不等式$v_{\pi}(s) \leq q_{\pi}(s, \pi'(s))$成立，所以在知道了$v_{\pi}(s)$和$q_{\pi}(s, a)$后，直接选择让$q_{\pi}(s, a)$最大的$a$作为新的策略，也就是上式。由于$v_{\pi}(s) = \sum\limits_a\pi(a \mid s)q_{\pi}(s, a)$，所以我们选择具有最大的$q_{\pi}(s, a)$，一定能保证不等式成立（最次也是等号的发生）。

假设出现了某种策略$\pi$，其按照上述方式改进后的策略$\pi'$，出现了$v_{\pi} = v_{\pi'}$，根据上述的贪心方式，则有以下式子成立
$$
\begin{array}{rcl}
v_{\pi^{\prime}}(s)&=&\max\limits_a\mathbb{E}[R_{t+1}+\gamma v_{\pi^{\prime}}(S_{t+1})\mid S_t=s,A_t=a]\\
&=&\max\limits_a\sum\limits_{s^{\prime},r}p(s^{\prime},r|s,a)\Big[r+\gamma v_{\pi^{\prime}}(s^{\prime})\Big]
\end{array}
$$
这恰恰就是bellman optimal equation，既然已经满足最优方程了，那么求解就结束了。

### 4.3 Policy Iteration

最开始随机指定一个策略$\pi_0$，然后对其policy evaluation求出$v_{\pi_0}$，在对其进行policy improvement得到$\pi_1$，如此循环往复，最终策略就会收敛到$\pi_*$。
$$
\pi_0\xrightarrow{\mathrm{E}}v_{\pi_0}\xrightarrow{\mathrm{I}}\pi_1\xrightarrow{\mathrm{E}}v_{\pi_1}\xrightarrow{\mathrm{I}}\pi_2\xrightarrow{\mathrm{E}}\cdots\xrightarrow{\mathrm{I}}\pi_*\xrightarrow{\mathrm{E}}v_*
$$
在已知环境动态$p$的情况下，我们最开始随机选择一个策略$\pi_0$，然后对该策略进行policy evaluation也就是使用DP方法多次迭代后得到$v_{\pi_0}$，再根据此时的$v_{\pi_0}$求出$q_{\pi_0}(s, a)$（环境动态已知），由此进policy improvement得到新的$\pi_1$，如此循环往复，最终得到最优价值$v_*$和最优策略$\pi_*$。

对于关于$q_{\pi}$的迭代，其本质是一样的，只不过再policy evaluation的时候把对$v_{k}$的迭代转化为了对$q_{k}$的迭代，即使用
$$
q_{k + 1}(s, a) = \sum\limits_{s', r}p(s', r \mid s, a)(r + \gamma \sum\limits_{a'}\pi(a \mid s)q_{k}(s', a'))
$$
此迭代式在policy evaluation的时候求出$q_{\pi}$。

实际上，应该将policy iteration重点放在对policy进行迭代的方式上，而不应该将价值看得太重。policy evaluation只不过总体policy迭代的一个子步骤，evaluation的目的还是在于求出$v_{\pi}$从而帮助得到一个更好的策略，不同的方法有不同的evaluation步骤。

### 4.4 Value Iteration

书中将value iteration重点放在其是由policy iteration发展而来的基础上进行介绍的。为了更方便的进行记忆，我将直接从bellman optimal equation来进行介绍，bellman optimal equation如下
$$
\begin{aligned}
v_{*}(s, a) &= \max\limits_a q_{*}(s, a) \\
&= \max\limits_a \sum\limits_{s', r}p(s', r \mid s, a)(r + \gamma v_{*}(s'))
\end{aligned}
$$
就如iterative policy evaluation一样，我们直接将此方程转换为迭代式
$$
\begin{aligned}
v_{k + 1}(s, a) &= \max\limits_a q_{k}(s, a) \\
&= \max\limits_a \sum\limits_{s', r}p(s', r \mid s, a)(r + \gamma v_{k}(s'))
\end{aligned}
$$
与iterative policy evaluation的理解方式类似，初始值的比重最终会随着环境的动态特性$p$和最优选择$\max_a$的迭代而不断地减小，使得$v_k$最后能够收敛到$v_*$。

**Relation Between Policy Iteration and Value Iteration**
对于policy iteration来说，其中间步骤的policy evalution需要进行多次的迭代，事实上，某次迭代中取得的$\pi$对于最终的收敛性并没有影响。所以我们也未必需要进行很精确的policy evaluation，即多次迭代从而将某次策略的价值函数都求出来。极端情况下，甚至直接使用初值进行policy improvement也能使其最终收敛。

更加详细来说，某次我们随机赋予一个初值$v_0$，我们不假设有一个随机的策略，而是直接通过这个初值进行policy improvement，即
$$
\begin{aligned}
v_1(s) &= \max\limits_aq_0(s, a)\\
&= \max\limits_a \sum\limits_{s', r}p(s', r \mid s, a)(r + \gamma v_{0}(s')) \\

\end{aligned}
$$
在这里，我们使用了greedy方法来进行improvement，从而得到新值$v_1$。我们使用同样的方式求得$v_2, v_3 \cdots$，我们会发现，对于前面的policy iteration来说，我们并没有完整的进行policy evaluation，而只进行了半步，即将状态价值转化为动作价值从而使用greedy进行improvement。

value iteration巧妙的同时结合了policy evaluation和policy improvement，每次迭代需要两次遍历，依次是evaluation中的遍历$s'$，另一次是improvement中的遍历$a$。同样我们也可以偶尔在evaluation环节中添加其的迭代次数，能够使得$v_k$更快的收敛到$v_*$。

### 4.5 Asynchronous Dynamic Programming & 4.6 Generalized Policy Iteration

**Asynchronous Dynamic Programming**
前面所介绍的迭代算法最大的问题就是需要系统性的遍历整个状态空间来完成依次迭代，其最后也是一次性的取得了所有状态的价值函数。然而若遇到状态空间过大，以及一些状态的价值函数我们并不关心的情况，就会遇到计算量爆炸的问题。我们可以采用异步动态规划算法来解决这个问题，异步算法来源于迭代式对收敛条件的宽松，其不要求收敛迭代式中的所有进行计算的状态有过相同的次数的迭代经历，基于此，我们可以灵活的安排顺序或是省略无关竟要的状态的迭代次数，来达到收敛某些特定状态的效果。关于异步动态规划算法，书中只是进行了概括性的介绍，并无详细的细节或例子，所以异步算法写到这里。

**Generalized Policy Iteration**
无论是policy iteration还是value iteration，我们都可看成是policy evaluation和policy improvement交互的过程。对于每一次迭代的policy，其都要根据一个新的与此时policy不匹配（这里不匹配的policy都是通过greedy方法得到）的value function来改进；而对于每一次迭代的value function，其又要根据一个新的，与此时value function不匹配的policy来更新；只有当某次迭代的value function和policy相匹配的时候，value function和policy才不会继续再改变，保持稳定，才达到了bellman optimal equation，由此也是value function和policy都达到了optimal。

evaluation和improvement的关系可看作竞争和合作，竞争在于其每次的迭代都会使得value function或是policy朝着与此前不匹配的方向前进，但是长期下来，evaluation和improvement迭代最后到达了一个汇合点，最终区趋于稳定。

![image-20231007120711916](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231007120711916.png)

如上图所示，evaluation和improvement可视为两条相交直线的一条，直线上的某点代表着某次迭代的状态。每次迭代后，都会使得直线上某点的状态转移到另一条直线的某点，并且会不断的趋向于最后的共同目标，即最优解。

**Bootstrap**
自举（bootstrap）也是DP的一个特性，自举的意思是对某状态value的更新都要依赖于其他的状态，即靠其他的状态的值才能迭代的算出当前状态的value。
