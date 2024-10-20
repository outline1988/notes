## Chapter 5 Monte Carlo Methods

### 5.1 Monte Carlo Prediction

在第四章中，我们使用DP的方法来进行policy iteration，以此找到最优策略，但此方法要求我们对于环境有一个全面的认知，也即动态特性$p$要已知，才能够根据$p$使用iterative policy evaluation来进行policy iteration中的policy evaluation环节，也即给定一个策略$\pi$，来计算出此时所有状态的价值函数$v_{\pi}$。此时由于已知的环境动态特性$p$，状态价值$v_{\pi}$可以转换为$q_{\pi}$从而其完全足够用以improvement。

如果环境动态特性$p$难以求得，DP的方法将难以进行下去，虽然policy improvement不依赖于$p$，但是DP的policy evaluation要求我们必须知道$p$。在这样的情况下，我们可以使用蒙特卡洛方法（Monte Carlo method，MC method）来完成policy evaluation的环节（当然单单只求状态价值在没有环境动态的情况下，是不足以进行improvement，但是prediction内在的含义是相同的）。

与DP方法的直接进行迭代计算价值函数不同，MC方法从在大量试验所获得的“经验（experience）”来对状态的价值函数进行估计。在这里我们要求此时的MDP一定为episodic tasks，由此其可被分为一个个相互独立的episode。在某次的episode中，若其出现了状态$s$，那么我们称到达了$s$（visit to $s$）。当然，一个episode可能会有多个状态$s$的出现，我们将状态$s$的第一次出现称为首达$s$（first visit to $s$）。所以一个自然的想法就是，既然我们拥有了大量试验之后大量独立的episodes，对于某个特定状态$s$，我们只要将所有episodes中首达$s$的价值的平均估计量作为我们对于该状态$s$的预测，这就是首达蒙特卡洛方法（first-visit MC method）。首达MC方法的数学基础是大数定理，而首达的性质保证了不同观测值的独立性。所以当试验次数趋于无穷时，平均估计量作为该状态$s$价值的无偏估计，是能够收敛到真值的。

我们也可以用上同一episode中多次出现的状态$s$，即我们将所有episode中所有到达$s$的价值的平均估计量作为该状态价值的估计量，其同样也能收敛到该状态的价值，这种方法为every-visit MC method。虽然同一episode中不同位置的相同状态$s$不能保证其具有完全独立的性质，但类比随机过程的均值各态历经性，不同时刻随机变量之间的相关性随着时间差具有某种下降的趋势，同样也可用于估计均值。

MC方法还有一个特性就是其非自举（bootstrap）的性质，也即对于某一个状态价值的估计是独立于其他状态的价值的，由此我们就可将关注点放在我们所需要的一些状态中了。

### 5.2 Monte Carlo Estimation of Action Values

在DP中，之所以只算出state value就可以依次进行policy improvement是因为我们知道了环境动态特性$p$，由此通过$q_{\pi}(s, a)$和$v_{\pi}(s)$关于$p$的关系就可以求得action value。但是在MC方法中，环境动态不可知，故单单只求出状态价值是不行的，我们需要求出动作价值。使用MC求动作价值的思路同样是大量的试验，只不过这次将注意力放在状态动作对，即我们不再是记录某个状态的出现，而是记录某个状态和某个动作的共同出现，得到此时的价值，最后求其平均估计量。其同样也有first visit和every visit的方式，这里不再赘述。

但是这里存在的问题是，必然存在某种策略，在该策略下，再多的试验都无法取得全部的状态和动作对的价值，由此便无法进行improvement（对于没有出现过的状态动作对来说，就不会对其进行更新，也即一直保持不动）。解决的方式就是在k-armed bandit那一章节中的保持探索（maintaining exploration）思想。为了方便，我们引入一个假设叫探索性开端（exploring starts），即每一对的状态和动作都会以非零的概率作为某个episode的开端，由此能保证在无限次的试验中，每种状态和动作都会被选上，从而得到所有状态动作对的价值。

### 5.3 Monte Carlo Control

在MC方法计算出$q_{\pi}(s, a)$的基础上，我们就能够顺利按照GPI的思想来得到最优策略（这里先假设GPI中的evaluation是完全的），仅仅将policy evaluation中的迭代换成MC方法（基于exploring starts和无限多次试验这两个假设），过程如下
$$
\pi_0\xrightarrow{\mathrm{E}}v_{\pi_0}\xrightarrow{\mathrm{I}}\pi_1\xrightarrow{\mathrm{E}}v_{\pi_1}\xrightarrow{\mathrm{I}}\pi_2\xrightarrow{\mathrm{E}}\cdots\xrightarrow{\mathrm{I}}\pi_*\xrightarrow{\mathrm{E}}v_*
$$
关于此的详细描述都在第四章，这里不再赘述。

前面基于的两个假设分别是1. exploring stars；2. 无限多次试验。我们暂时保持第一个假设，考虑将第二个假设给去除。实际上，无限多次的试验同iterative policy evaluation中无限多次的迭代所遇到的困境一样，在现实中均难以完成，并且无限多次的目的都是为了能够精确进行policy evaluation。所以我们当然可以在GPI的思想指导下，将无限次转化为有限次，书上提供两条思路：让有限次试验得到的平均估计量足够的逼近于真值，这些就是参数估计相关的知识了；或者干脆更加极端，只用一次的试验来估计某个状态动作价值，就像value iteration一样。由此得到的伪代码如下

![image-20231010221230645](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231010221230645.png)上述伪代码计算均值的部分可以仿照第二章的思路，用增量更新的方式来求解均值。由于一次episode难以覆盖所有的状态动作对，所以我们需要用上之前policy evaluation的数据，即使这些数据都不是在同一个策略$\pi$下产生的，但是GPI告诉我们其最后能够收敛。我觉得可以将均值计算中更多的权重给予更近策略所得到的观测值来提高收敛速度，反正书上没这么说，按照GPI的原则反正最后也能收敛。

最后，我们称这样的算法为探索开端的蒙特卡洛方法（Monte Carlo ES），并且这种方法对于能否收敛到其不动点仍然没有严格的数学证明。定性的理解就是GPI是神！咋都能收敛。

### 5.4 Monte Carlo Control without Exploring Starts

从总体上来说，有两种方法可以摆脱exploring starts的假设，分别为同轨策略（on-policy）和离轨策略（off-policy）。on-policy意味着用于evaluate和improvement的策略是同一策略，而off-policy并非是同一策略（例如用于生成数据的evaluation的策略不同于improvement改进的策略）。在蒙特卡洛方法中，evaluation用某一给定的策略来生成episodes，而improvement也是用该策略对应的价值函数来进行改进，所以上述的MC方法也属于on policy。本小节所讨论的都是on-policy策略。

为了让策略更具有探索性，我们将上述使用MC方法的policy iteration中所有出现的策略都更改为更加“soft”的策略，此时策略不再是确定性（deterministic）策略，对于任意状态，而所有动作都具有非零的概率被选择。基于soft policy的前提下我们使用k-armed bandit中的$\varepsilon$-greedy思路，即在确定性greedy的基础上，每次的选择都会有$\varepsilon$的概率不用贪心策略，而是等概率的选择所有动作，即此时其他的非贪心动作有概率$\frac{\varepsilon}{|\mathcal{A}(s)|}$被选上，而原来的greedy动作被选上的概率为$1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|}$。$\varepsilon$-greedy策略是$\varepsilon$-soft策略的一种情况，后者对于所有状态所有动作的要求为$\pi(a \mid s) \geq \frac{\varepsilon}{|\mathcal{A}(s)|}$。

虽然中间使用的都是soft策略，但是在GPI的帮助下，最终依然能够收敛到**soft策略中的最佳策略**，将上述伪代码按照$\varepsilon$-greedy方法进行修改，得到新伪代码如下

![image-20231011201029422](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231011201029422.png)

与之前伪代码唯一不同的地方就是在于policy improvement不再是以确定性的方式，而是使用$\varepsilon$-greedy的方法进行。

**Policy Improvement Thorem in Soft Policies**
前面的策略改进都是基于对$q_{\pi}$的贪心而得到的确定性策略，根据policy improvement thorem，其最后能够收敛到符合optimal bellman equation的最优策略。在soft策略的背景下，使用$\varepsilon$-greedy对原先的soft policy进行改进，该定理依然成立，只不过最终收敛到的是所有soft策略下的最优策略，证明如下
$$
\begin{array}{rcl}
q_\pi(s,\pi'(s)) &=& \sum\limits_a\pi'(a|s)q_\pi(s,a) \\
&=& \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_aq_\pi(s,a)+(1-\varepsilon)\max\limits_aq_\pi(s,a) \\
&=& \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_aq_\pi(s,a) + \max\limits_aq_\pi(s,a) -  \max\limits_a\varepsilon q_\pi(s,a) \\
&\geq& \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_aq_\pi(s,a) + \sum\limits_a \pi(s \mid a) q_\pi(s,a) -  \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_aq_\pi(s,a) \\
&=& v_{\pi}(s)
\end{array}
$$
有了这个结论，我们就知道在soft策略的条件下，只要进行$\varepsilon$-greedy的改进，改进后的效果一定不会比改进前的差。现在来证明等号发生的情况，即$\pi$和$\pi'$在改进定理的情况下相同，则其都为所有soft策略中最优的soft策略（确定性策略的改进定理在等号出现的时候，迭代式的形式变为了最优贝尔曼方程，所以此时一定是最优价值）。

为了更好的理解soft策略，我们假设出现了一个新的环境，其仅仅只与原来的环境有略微的不同，即每次选择动作之后，有$1 - \varepsilon$的概率与原环境的表现完全相同，但还有$\varepsilon$的概率表现为与原环境中重新等概率选择所有动作的表现相同。在这样的新环境下，根据bellman optimal方程，其必定存在一个最优价值$\widetilde v_*$，并且这个最优价值满足以下关系
$$
\begin{array}{rcl}
\widetilde v_*(s) &=&  \max\limits_a \widetilde q_*(s, a) \\
&=& (1 - \varepsilon)\max\limits_a \widetilde q_*(s, a) + \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_a \widetilde q_*(s, a) \\
\end{array}
$$
这个式子中第一行是根据bellman optimal equation写出的，第二行是根据这个新环境本身所具有的特性写出的。

回到原始环境，在soft策略的条件下，如果按照soft策略下的改进最后等号相同（$\pi' = \pi$），则有
$$
q_{\pi}(s, a) = (1-\varepsilon)\max\limits_aq_\pi(s,a) + \frac{\varepsilon}{|\mathcal{A}(s)|}\sum\limits_aq_\pi(s,a) \\
$$
可以发现其形式与新环境的最优表达式一模一样。同时，即使两个环节只在细节上稍有差别，但其表现出来的转移特性完全相同，所以新环境存在最优意味着原环节也存在最优，且新旧环境最优时价值相等，即$q_{\pi} = q_*$。所以当使用$\varepsilon$-greedy方法进行policy evaluation，最终会在等号出现的时候达到最优。**该证明的思路在于将$\varepsilon$-greedy的方法不施加于策略的改进，而是将其嵌入到环境的特性，在将这个环境套用到确定性的policy improvement thorem。这启示我们，若有什么修改的方式对于所有的策略都进行施加，那么可将修改策略的方式嵌入环境，转而使用确定性的policy improvement thorem，由此得到的结论就是，以这种方式进行的迭代能够得到在特定修改方式施加于所有的策略中能达到的最佳策略。**

但注意此最优只是在所有soft策略中是最优的，这个策略只能是依然再探索的接近最优的策略，但的的确确使我们抛弃了exploring starts的不可实现的假设。

### 5.5 Off-policy Prediction via Importance Sampling

在off-policy中，进行policy evaluation和policy improvement的策略不是同一策略，我们称用于evaluation的策略（例如在MC中生成数据）为behavior policy而用于improvement的策略为target policy，本节我们所讨论的都是off-policy learning。

**Importance Sampling**
考虑一个随机变量$X$，其拥有一个较为复杂的概率密度函数$f_X(x)$，如果我们能得到该分布的观测值，根据大数定理，我们可以对该随机变量（或其函数）的均值进行估计，估计量如下
$$
\mathbb{E}[f(X)] = \int f(x) p_X(x) \mathrm{d}x \approx \frac{1}{N}\sum\limits_{i = 1}^{N} f(x_i)
$$
其中约等号右边的式子是根据大数定理而来的求均值的离散版本。

但是如果计算机难以模拟复杂的分布函数$f_{X}(x)$，而现在我们有一个简单的，计算机可生成的随机变量$Y$，其具有概率密度函数$f_{Y}(y)$。由此，我们适当修改上述等式
$$
\begin{array}{rcl}
\mathbb{E}[f(X)] &=& \int f(x) p_X(x) \mathrm{d}x \\
&=& \int f(x) \frac{p_X(x)}{p_Y(x)} p_Y(x) \mathrm{d}x \\
&\approx& \frac{1}{N}\sum\limits_{i = 1}^{N} f(y_i) \frac{p_X(y_i)}{p_Y(y_i)}
\end{array}
$$
其中$y_i$表示由随机变量$Y$产生的样本。由此我们可以由简单分布产生的数据来计算复杂分布的期望，其过程就是在每一个简单分布产生的样本（或其函数）乘一个重要性权重$\rho_i$，这个权重与两个随机变量概率密度在该样本观测值$y_i$取值的比例有关。概率密度的取值可以简单的理解为该值的概率，所以这个式子给予我们的启发就是若有某一随机变量的某一观测值，通过该随机变量在该点取值的概率与另一随机变量在同一观测值的概率的比值（所以实际上也并不一定需要知道PDF，仅仅取值的概率即可）来将这个随机变量的观测值修正为另一随机变量的观测值（说观测值不准确），由此进行均值的估计。

怎么理解重要性采样呢？先考虑一下为什么能用大数定理来求均值，对于某一特定的随机变量，其概率密度值大的地方必然代表着该值的邻域发生的概率较大，体现在多次的试验中就是观测值在该领域集中的较多。我们将概率密度的横轴分为若干个较小的间隔，每次观测得到的观测值意味着在对应间隔上加一，即该邻域在统计图中高度加一。进行多次观测后，该统计图中每个邻域的累积次数可以画出一个近似该随机变量PDF的曲线。直接将所有的观测值求和近似于每个邻域的代表值（该领域随意一个值）乘上对应次数再求和，和使用PDF求平均的形式相同，只不过此时PDF变为了某领域发生的次数。详细可以参考统计信号处理基础P146附录7A蒙特卡洛方法。

重要性采样可以将求某一随机变量的均值转换为具有不同分布的另一随机变量的均值的机理在于，将原随机变量的观测值乘上系数之后，它将原本近似于原随机变量PDF的次数统计图转换为了与目标随机变量PDF近似的次数统计图。这个系数就是目标PDF在该领域的取值（理解为该领域的概率）与原PDF在该邻域的取值的比值上，这个重要性系数体现在实际的试验观测中就是为每一个观测值乘上这个重要性系数（整体上就是次数统计图上不同邻域的高度发生了不同程度的改变）为该观测值所处领域的次数修正至目标的PDF。

使用重要性采样应该牢记这样两个事实，使用简单分布$Y$生成的样本若是复杂分布$X$难以生成（概率很小），那么此时重要性因子很小，相当于$Y$对计算$E[f(X)]$的贡献很小；若是复杂分布$X$可能生成的数据简单分布$Y$难以生成，那么则表示由$Y$很难生成包含$X$信息量很多的样本。这两件事的本质都是简单分布$Y$生成的样本数据由于$X$本身很难生成，所以所包含的关于$X$的信息很少，最终导致估计精度下降。**所以实践中简单分布$Y$要尽可能全覆盖$X$所能产生的数据，此条能够保证在无穷的$Y$下样本能够保证使得$X$的均值收敛；其次，$Y$尽量少生成$X$难以生成的数据，如此保证$Y$所生成的大部分数据都具有$X$的信息量；最后，为了进一步提高$Y$产生样本所能包含$X$的信息量，应该让$Y$与$X$​尽可能接近。**

上面所讨论的缺陷可以简单总结为方差过大的问题。

**Prediction Using Importance Sampling**
我们给定一个策略$\pi$，希望能够得到其$v_{\pi}$（目前还没有涉及到$q_{\pi}$），但是我们即不知道环境动态特性$p$，也不能使用$\pi$来生成episodes，我们拥有的仅有另外一个策略$b$和使用策略$b$生成的一系列episodes，这种情况显然属于off-policy，策略$\pi$是target policy而策略$b$是behavior policy。

所以我们使用importance sampling，我们取已有的策略$b$生成的某次episode，我们要计算$\mathbb{E}[G_t \mid S_t = s]$，这个episode包含了状态$S_t$，其后的所有轨迹为$S_t, A_t, S_{t + 1}, A_{t + 1}, \cdots ,S_T$，我们要将该轨迹修正为在符合目标策略$\pi$下的轨迹（等效于为不同策略下的$G_t$进行修正），也即使用策略$\pi$和策略$b$在该特定轨迹的概率之比
$$
\begin{aligned}
&\operatorname*{Pr}\{A_{t + 1},S_{t+1},A_{t+1},\ldots,S_{T}\mid S_{t},A_{t:T-1}\sim\pi\}  \\
&=~\begin{aligned}\:\pi(A_t|S_t)p(S_{t+1}|S_t,A_t)\pi(A_{t+1}|S_{t+1})\cdots p(S_T|S_{T-1},A_{T-1})\end{aligned} \\
&=~\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k) 
\end{aligned}
$$

$$
\begin{aligned} 
\rho_{t:T-1}\doteq\frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)p(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1}b(A_k|S_k)p(S_{k+1}|S_k,A_k)}=\prod_{k=t}^{T-1}\frac{\pi(A_k|S_k)}{b(A_k|S_k)}.
\end{aligned}
$$

经过了修正之后，我们就可以计算策略$\pi$下的回报均值为$\mathbb{E}[\rho_{t:T-1}G_t \mid S_t = s]$。我们将所有的episodes首尾相连连成一条链，其中包含若干个起始状态和终止状态，将$\mathcal{J}(s)$定义为所有episodes中第一次（或者所有，取决于first-visit还是every-visit）到达状态$s$的时刻；$T(t)$表示当前时刻$t$所处的episode的终端状态时刻。

**Ordinary Importance Sampling**
由上述，假设均使用first-vist方法，则以下式子成立
$$
V(s)\doteq\frac{\sum_{t\in\mathcal{J}(s)}\rho_{t:T(t)-1}G_t}{|\mathcal{J}(s)|}.
$$
这就是使用策略$b$下的轨迹修正后得到的关于$v_{\pi}(s)$的估计统计量，因为其为均值的形式，我们称其为ordinary importance sampling。该统计估计量为$v_{\pi}$的无偏估计，但是其拥有很大的方差甚至无穷大的方差，因为重要性权重$\rho$的方差是无界的（因为有可能episodes的长度无限长，而每个子$\rho$都是大于1的），必然导致$V(s)$的方差也是无界的；注意到该式子使用的是**所有符合状态为$s$的episodes**，不管此时轨迹策略$\pi$下有没有可能发生，若没有，无非就是重要性权重$\rho$为0，故此轨迹对分子没贡献，但是其会对分母有贡献，而最终导致结果总时偏小。也就是说，即使策略$b$所包含策略$\pi$的信息很少，也依然会对估计量进行影响，使得其偏小，这是我们在非无限样本（无限次总会收敛到真值的）中不愿意看见的。

**Weighted Importance Sampling**
一种特殊的权重形式如下，同样使用first-visit方法
$$
V(s)\doteq\frac{\sum_{t\in\mathcal{J}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t\in\mathcal{J}(s)}\rho_{t:T(t)-1}}
$$
该估计统计量虽然为有偏估计，但是重要性权重$\rho$的无界性不会导致$V(s)$的无界，也即其拥有更小的方差。数学可以证明，其为渐进无偏估计。相比于ordinary，在少数episodes可用时所生成的估计量与真值有更小的误差（书P106，Example 5.4），weighted要更为常用。直观上，当$\rho$为0时，其对加权重要性采样的式子的分子和分母同时无贡献。

理解上面的式子应该从**策略$b$为着手点**，我们所要的状态首先是出现在策略$b$中，其次再来计算其后续轨迹的重要性因子，若其后的轨迹使用$\pi$无法产生，则此时重要性因子为0。对于every-visit来说，两种形式下的估计统计量都是有偏的。

**注意，到目前为止我们只是用重要性采样估计$v_{\pi}(s)$，并没有涉及到状态动作对的$q_{\pi}(s, a)$，策略$b$生成的轨迹一定是能够覆盖到策略$\pi$的，策略$b$最终能被采用到用于估计$\pi$的轨迹其重要性权重不为0，也即被采用的轨迹使用策略$\pi$也能生成，改进的地方在于使用了策略$b$生成而没有用策略$\pi$。所以在状态价值估计这个问题中，importance sampling并没有解决策略$\pi$探索不足的问题，但是在估计动作价值的时候解决了。**

**Example 5.5**
若存在某一MDP，使用off-policy方法并且其target policy和behavior policy如下

![image-20231013123719069](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231013123719069.png)

我们令随机变量$X$为对策略$b$修正后的回报，即$X = \rho_{t : T(t) - 1} G_{t}$，其中这里的$G_t$是属于策略$b$下的回报。由上可知，其均值
$$
\mathbb{E}[X] = \mathbb{E}[\rho_{t : T(t) - 1} G_{t}] = v_{\pi}(s)
$$
其方差为
$$
\operatorname{Var}[X]\doteq\mathbb{E}\Big[\left(X-\bar{X}\right)^2\Big]=\mathbb{E}\Big[X^2-2X\bar{X}+\bar{X}^2\Big]=\mathbb{E}[X^2]-\bar{X}^2
$$

$$
\begin{aligned}
\mathbb{E}[X^2] &= \mathbb{E}_b\left[\left(\prod_{t=0}^{T-1}\frac{\pi(A_t|S_t)}{b(A_t|S_t)}G_0\right)^2\right] \\
&=\frac12\cdot0.1\left(\frac1{0.5}\right)^2 \\
&+\frac12\cdot0.9\cdot\frac12\cdot0.1\left(\frac1{0.5}\frac1{0.5}\right)^2 \\
&+\frac12\cdot0.9\cdot\frac12\cdot0.9\cdot\frac12\cdot0.1\left(\frac1{0.5}\frac1{0.5}\frac1{0.5}\right)^2 \\
&+\cdots  \\
&=0.1\sum_{k=0}^\infty0.9^k\cdot2^k\cdot2=\:0.2\sum_{k=0}^\infty1.8^k=\:\infty
\end{aligned}
$$

在具体计算$X$的二阶原点矩时，由于策略$\pi$必不可能选择right，轨迹包含right的重要性权重$\rho$都为0，故可忽略；其次在left动作下回到原状态的奖励为0，故只用考虑终端状态的奖励1。造成上述$X$二阶原点矩为无穷的原因在于target policy回到原状态的概率与behavior policy回到原状态的概率之比与回到原状态的概率之积要大于1。

由于$X$的方差是无穷的，所以直接使用其平均估计量必然会导致其有效性很低。所以更为实用的是使用weighted importance sampling。好处在于在削弱无偏性为渐进无偏性后，其无界的方差升级到了有界的方差。

**Exercise 5.6**
给出关于$q_{\pi}(s, a)$的估计值$Q(s, a)$，同样在有大量策略$b$的序列下
$$
Q(s, a) = \frac{\sum_{t \in \mathcal{J}(s, a)}\rho_{t + 1 : T(t) - 1} G_t}{\sum_{t \in \mathcal{J}(s, a)}\rho_{t + 1 : T(t) - 1}}
$$
这里进行简单推导，由前可知从时刻$t$开始产生的轨迹为$S_t, A_t, S_{t + 1}, A_{t + 1}, \cdots ,S_T$，并且由于$Q(s, a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$对于选择动作$A_t$没有随机性，所以重要因子只取决于从$S_{t + 1}$开始后的轨迹，即
$$
\begin{aligned}
&\operatorname*{Pr}\{S_{t+1},A_{t+1}, \ldots,S_{T} \mid S_{t}, A_{t + 1:T-1}\sim\pi\}  \\
&= p(S_{t+1}\mid S_t,A_t) \pi(A_{t+1}\mid S_{t+1})\cdots p(S_T\mid S_{T-1},A_{T-1}) \\
&= p(S_{t+1}\mid S_t,A_t) \prod_{k = t + 1}^{T-1}\pi(A_k \mid S_k)p(S_{k+1} \mid S_k,A_k) 
\end{aligned}
$$

$$
\begin{aligned} 
\frac{p(S_{t+1}\mid S_t,A_t) \prod_{k=t + 1}^{T-1}\pi(A_k\mid S_k)p(S_{k+1}\mid S_k,A_k)}
{p(S_{t+1}\mid S_t,A_t)\prod_{k=t + 1}^{T-1}b(A_k\mid S_k)p(S_{k+1}\mid S_k,A_k)}=\prod_{k=t + 1}^{T-1}\frac{\pi(A_k\mid S_k)}{b(A_k\mid S_k)} = \rho_{t + 1:T-1}
\end{aligned}
$$

对于该式子的描述为：从策略$b$产生的episodes出发，找到所有的状态$s$和动作$a$对，然后得到该状态动作对后面的轨迹，若该轨迹在策略$\pi$中不可能发生（则说明这条轨迹对于计算策略$\pi$下的参数没有任何信息量），此时重要性因子为0；若可能发生，则计算重要性因子，因为动作$a$已经作为条件给定了，其后面的顺序（不包含动作$a$）仍然有可能在$\pi$中发生，所以即使在策略$\pi$下，某动作状态对不会发生，但其仍然能够计算出该状态动作对的动作价值函数预测$Q(s, a)$。使用策略$b$能算出策略$\pi$下不可能出现的状态动作对的机理在于策略$b$在包含了策略$\pi$的所有轨迹的同时，其仍然包含着策略$\pi$所没有的状态动作对，而计算此时的状态动作对的价值不需要将此刻的状态动作对的轨迹所包含，而是其后续的轨迹与即时奖励之和，这些后续的轨迹$\pi$是有可能产生的，而即使奖励是属于策略$b$的，不需要修正。所以若要保证在无穷的策略$b$下的样本能够计算出策略$\pi$的参数，那么则要保证策略$b$尽量包含策略$\pi$所能产生的全部轨迹，以及策略$b$本身要有全部的状态动作对出现。

所以新策略$b$用于计算$\pi$状态价值或动作价值的轨迹仍然都是属于$\pi$可能出现过的轨迹，状态价值中，轨迹从$S_t$出发；而动作价值中，$b$中的轨迹从$S_t, A_t$出发，但只要对$b$中的轨迹从$S_{t + 1}$出发后做修正，然后再加上$S_t, A_t$的奖励即可，从$S_{t + 1}$出发的轨迹$\pi$是有可能出现的，所以上述的$\rho_{t + 1 : T(t) - 1} G_t$中，重要性因子没有修正第一项，第一项实际上不属于策略$\pi$，但是因为要improvement必须要所有的动作，而策略$\pi$却不可能发生这样的动作，所以只能让策略$b$的一次奖励与修正后的属于策略$\pi$的后续轨迹的回报之和作为策略$\pi$下那个原本不可能发生的状态动作对的价值，此时策略$\pi$仍然是占主导。

### 5.6 Incremental Implementation

 若我们拥有序列$G_{1}, G_{2}, \dots, G_{n - 1}$和其对应的权重序列$W_1, W_2, \dots, W_{n - 1}$，并且此时已经计算得到
$$
V_n\doteq\frac{\sum_{k=1}^{n-1}W_kG_k}{\sum_{k=1}^{n-1}W_k},\quad n\geq2
$$
此时来了一个新的$G_n$和其权重$W_n$，我们要按照以上式子计算$V_{n + 1}$，怎样用增量式的思路来计算呢？
$$
\begin{aligned}
V_{n+1}& =\frac{\sum_{k=1}^nW_kG_k}{\sum_{k=1}^nW_k}  \\
&=\frac{W_nG_n+\sum_{k=1}^{n-1}W_kG_k}{W_n+\sum_{k=1}^{n-1}W_k} \\
&=\frac{W_nG_n+\left(\sum_{k=1}^{n-1}W_k\right)V_n+W_nV_n-W_nV_n}{W_n+\sum_{k=1}^{n-1}W_k} \\
&=V_n+\frac{W_nG_n-W_nV_n}{\sum_{k=1}^nW_k} \\
&=V_n+\frac{W_n}{\sum_{k=1}^nW_k}[G_n-V_n] \\
&=V_n+\frac{W_n}{C_n}[G_n-V_n], \quad n\geq1
\end{aligned}
$$
其中
$$
C_{n + 1} = C_n + W_{n + 1}, \quad n \geq 0 ~~ and ~~ C_0 = 0
$$
$C_n$是所有权重$W_1, W_2, \dots, W_{n}$的累加，当$n = 1$时
$$
V_2 = V_1 + \frac{W_1}{C_1}[G_1 - V_1] = G_1
$$
$V_n$代表着前面$n - 1$项的加权平均和，所以$V_2$代表前面1项的加权平均和，也就是$G_1$；$V_1$代表前面0项的和，所以任意。每当有一个新的$G_n$和其权重$W_n$时，我们直接套用上式子得到$V_{n + 1}$，然后再计算出$C_{n + 1}$即可。

由此，按照这个思路，使用weighted importance sampling的MC方法prediction伪代码如下

![image-20231013174432036](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231013174432036.png)

上述伪代码并不是针对某一特定的状态动作对，而是只要遇到了一个随意的状态动作，就对其进行更新。对于某一个episode，程序从后往前遍历，一旦遇到了策略$\pi$产生不了轨迹，也即$W = 0$，则进入下一个episode。$W$在最后进行更新，表示计算$Q(s, a)$时的权重不用考虑第一项的奖励，也即权重因子为$\rho_{t + 1 : T(t) - 1}$。

### 5.7 Off-policy Monte Carlo Control

有了以上的铺垫，便可设计使用off-policy方法找到最优策略的算法如下

![image-20231013193056662](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20231013193056662.png)

从策略$b$的某一episode视角来看，对该episode从后往前遍历。在第一个循环中，程序针对的是$S_{T - 1}$和$A_{T - 1}$的动作价值，由于终端回报$G_T$永远为0，且终端回报永远能被目标策略的轨迹所包含，所以第一个循环可直接更新动作价值，更新后紧接着进行贪心improvement；第一个循环中的重要性权重$W$永远为1，这是由于第一次循环仅仅只有策略$b$产生的回报$R_{T}$，故权重因子为1。进入后面的循环前，默认将刚才的$S_{T - 1}$和$A_{T - 1}$加入到轨迹，由此必然需要判断加入后的轨迹是否是$\pi$可能生成的，也即判断$A_{T - 1}$与$\pi(S_{T - 1})$是否相等；判断完成后，再进行权重的更新，权重更新在最后是因为估计$Q(s, a)$的第一个即时回报权重为1，所以当前状态动作对的概率之比还不需要添加，最后添加是保证下一次循环的使用。注意到在GPI的原则下，用于weighted importance sampling的回报并非是同一target policy的回报，但GPI保证了其仍然能收敛。

正真用于估计$\pi$的在策略$b$下产生的episodes，对于其的要求是极其苛刻的，其尾轨迹必须与策略$\pi$产生的轨迹所重合，然而对于一个markov process来说，两个轨迹重合的概率是极其极其低的，这就导致实际中几乎不可能使用behavior policy的episodes生成足够多的可供target behavior使用的episodes。特别是如果出现了某些状态通常在较早的时候出现，那么能够遇到相同轨迹的可能性就更低了（随着轨迹长度的增加，轨迹重合的概率降低）。所以在实际编写中，我们应该把target behavior的greedy方法看成是$\varepsilon$特别小的$\varepsilon$-greedy方法。由此才能保证behavior policy方法产生的episodes能尽可能的为$\pi$的改进所用。虽然这样我们使用了策略$b$的所有轨迹，但是绝大部分的轨迹的重要性权重$\rho$由于不能与策略$\pi$重合，所以其极其低，最终对于价值函数的估计的贡献也是很少的。所以off-policy方法缺陷很大，无法应用到实际中。*这条可以参见重要性采样小节的最后一段，结论是要是的策略$b$在与策略$\pi$接近的同时尽量具有探索性，其有一个探索性和收敛速度的trade off，可能可以用其他方法找到trade off的平衡点，这里插个眼。*
