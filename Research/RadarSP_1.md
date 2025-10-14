### 噪声模型和信噪比

雷达的噪声可以分为外部噪声和内部噪声，外部噪声即与回波信号一起进入接收机的噪声，一般来自于宇宙，常常忽略；内部噪声即一般的电子元器件都会产生的热噪声，这种噪声在整个雷达系统中占大头，所以主要讨论内部热噪声。

虽然热噪声往往来源于每级电子系统内部，但是为了方便，将噪声统一建模为与回波共同进入输入端，此时输入端的噪声功率谱密度为
$$
S_n(f) = kT
$$
即为功率谱为常数的高斯白噪声，其中$k$是玻尔兹曼常数，一般为$k = 1.38 \times 10^{-23} \, \text{J/K}$，$T$为当前的温度。

接收机系统的带宽一般与发射信号的带宽相同，若是接收机带宽更大，那么便会引入更强功率的噪声，若是更小，则不能完全接收回波信号。一个典型的接收机系统的传递函数如下图所示

![image-20240828103842567](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240828103842567.png)

根据能量守恒的原则将不规则的函数形状转换为矩形窗，由此等效带宽定义为
$$
\beta_n = \frac{\int |H(f)|^2 \mathrm{d} f}{\max\{ |H(f)|^2 \}} = \frac{1}{G_s} \int |H(f)|^2 \mathrm{d} f
$$
由于接收机系统一定是有源的，所以其存在一个增益$G_s$。至此，我们便可以得到输出端噪声功率为
$$
P_n = k T \beta_n G_s
$$
上述的输出噪声功率的表达式中，温度$T$作为变量来描述噪声功率是不合适的，为了更加方便的描述该噪声的功率，引入一个标准额定温度$T_0 = 290 \, \text{K} = 16.85\,^\circ \text{C}$，此时实际的温度可表示为$T = T_0 + T_e$，故噪声功率为
$$
N_0 = k T_0 \beta_n G_s + k T_e \beta_n G_s
$$
其中，$T_e$代表有效温度，使用该量便可以完整表示温度的信息，进而描述当前的噪声功率。

此外，除了使用加性的有效温度来描述噪声功率，还可以使用噪声系数$F_n$，定义如下
$$
F_n = \frac{P_n}{kT_0 \beta_n G_s} = \frac{T_0 + T_e}{T_0}
$$
即系统实际输出功率与假定为标准温度下的系统输出噪声功率的比值， 在前述的模型下，就是实际温度与标准温度的比值。直观的理解为当我们使用标准温度进行噪声功率的讨论时，对于实际系统的噪声，还需要添加一个修正，一种方法是使用加性温度修正，另一种方法是使用乘性噪声系数修正。

由雷达方程可知，接收机输入的瞬时回波功率为
$$
P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4 \pi)^3 R^4}
$$
进入接收机系统后其得到一个增益$G_s$，最后再与输出噪声功率相比便可得到信噪比表达式
$$
\text{SNR} = \frac{P_o}{P_n} = \frac{G_s P_r}{k T_0 \beta_n G_s F_n} = \frac{P_r}{k T_0 \beta_n F_n} = \frac{P_t G_t G_r \lambda^2 \sigma}{(4 \pi)^3 R^4 k T_0 \beta_n F_n}
$$
一般来说，之后当回波的信噪比大于某一信噪比门限时，目标才会被检测到，当该门限带入为上式的信噪比中时，便可得到一个关于最大检测距离的约束公式。

**双基地雷达方程（The bistatic radar equation）**

首先分析回波的瞬时功率
$$
P_r = \frac{P_t G_t}{4 \pi R_1 ^2} \sigma \frac{1}{4 \pi R_2^2} \frac{\lambda^2}{4 \pi} G_r
$$
噪声功率与普通雷达方程一样，所以回波的信噪比为
$$
\text{SNR} = \frac{P_t G_t G_r \lambda^2 \sigma}{(4 \pi)^3 R_1^2 R_2^2 kT_0 \beta_n F_n}
$$
对比两个方程，可以看出，普通雷达方程相当于双基地的一种特殊情况，即$R_1 = R_2$。 



### 停跳近似-错

雷达发射的窄带信号可统一表示为
$$
s_t(t) = u(t) \exp(\mathrm{j} 2 \pi f_0 t)
$$
其中，$u(t)$为复包络，其包含了基带信号带宽等的所有信息。

对于一个正在运动的目标，其相对雷达的距离为关于时间$t$的函数$R(t)$，由此可得到回波信号的表达式为
$$
s_r(t) = s_t(t - \frac{2 R(t)}{c}) = u(t - \frac{2 R(t)}{c}) \exp\left[\mathrm{j} 2 \pi f_0 (t - \frac{2 R(t)}{c})\right]
$$
停跳近似的假设在于将复包络内关于$t$的变化拿掉，即复包络中目标的行为好像“停”了一样
$$
s_r(t) \approx u(t - \frac{2 R}{c}) \exp\left[\mathrm{j} 2 \pi f_0 (t - \frac{2 R(t)}{c})\right]
$$
由于窄带信号载频远远大于带宽$f_0 \gg B$，所以同等目标速度的情况下，载频中由于目标运动而造成的相位调制要远远多于基带复信号的目标运动相位调制，再加上一般目标的运动不快，所以在同一个脉冲中，基带复信号中目标运动造成的相位调制几乎没有变化，而载频中的目标运动相位调制要明显得多，故将复包络的相位变化忽略（即假设复包络碰到了一个禁止目标，在下一个脉冲才会突然变化），而载频遇到了一个正在运动的目标，所以载频的频率会增加一个多普勒频移（但是在慢速目标的假设下，忽略对于脉宽等的变化），所以整体上呈现在单脉冲中目标包络不变，而载频增加一个多普勒频移的效果。

在下一个脉冲来到的时候，复包络确实由于目标在一个脉冲重复周期中前进的距离而发射相位的变化（变化很微小）

当雷达的多个回波脉冲排列为快慢时间轴的形式时，窄带信号由于带宽小，中频采样率小，所以距离单元相对来说很大，在量级为几百的脉冲数量的情况下，目标也不会脱离一个距离单元。
$$
c T_s \gg M T_r v \Rightarrow c \gg v B M k T_r = v f_s \text{CPI}
$$
其中，$v$为目标速度，$B$为发射带宽，$M$为积累脉冲数，$f_r$为脉冲重复周期，$k$满足$f_s = kB$，即采样率与带宽是同一个量级的，上式条件不满足的话就很容易发生距离徙动问题，一般出现在宽带雷达中。

现在先不考虑停跳近似，来探究一下多普勒频率出现背后的物理机理。如果目标的速度朝向雷达，对于相邻两个脉冲，按照快慢时间轴排列，第二个脉冲相比于第一个脉冲要提前$\Delta t = v T_r$时间，这段时间很短（因为目标速度$v$不快，$T_r$很短），同样，脉冲内部的相位也要提前这么一小段时间。对于窄带信号的脉冲来说，其内相位调制包含两个部分，一是基带相位调制，这一部分与信号的带宽有关；二是载频的相位调制。由于窄带信号载频远大于带宽，所以$\Delta t$这段时间对于载频的相位影响很大，在一定的条件下，将$\Delta t$引起的基带信号的相位变化忽略，那么此时相邻两个脉冲之间的相位差就仅由$2 \pi f_0 \frac{2R(t)}{c} \Rightarrow 2 \pi f_d t$决定，由此我们便可以通过测复指数信号的频率来确定多普勒频移。

很遗憾，我这里说的全错。。。

### 停跳近似

**信号模型（载频相同，相位连续）**
对于单脉冲的雷达发射信号，第1个脉冲的波形可以表示为
$$
s_0(t) = A(t) \exp(\mathrm{j} 2 \pi f_0 t + \varphi_0)
$$
其中，$A(t)$为复包络，其在一个脉宽范围$[0, T_e]$内（区别于一般的范围在$[-T_e / 2, T_e / 2]$的复包络$u(t)$）。

对于一个正在运动的目标，若其相对雷达的距离为关于时间的函数$R(t)$，则第一个脉冲所返回的回波为
$$
\begin{aligned}
r_0(t) &= s_0(t - \frac{2 R(t)}{c}) = A(t - \frac{2 R(t)}{c}) \exp\left[\mathrm{j} 2 \pi f_0 (t - \frac{2 R(t)}{c}) + \varphi_0\right]  \\
&\approx A(t - \frac{2 R(0)}{c}) \exp(-j\frac{4\pi R(t)}{\lambda}) \exp(\mathrm{j} 2 \pi f_0 t + \varphi_0)
\end{aligned}
$$
这里将时间在第一个脉冲的周期内$[0, T_r]$，复包络由于目标运动而发生的相位变化忽略不计。直观来说，就是发送脉冲的瞬间，目标对于复包络来说，好像静止了。对于载频来说，其保留所有目标运动特性引起的相位调制。上式中，$\exp(-j 4\pi R(t) / \lambda)$称之为相位历程，其包含了目标所有的运动特性。也就是说，对于运动的目标，复包络当其静止在当前脉冲发射的时刻；而对于载频来说，其会由于目标的运动而引起一个相位调制，称为相位历程，这个相位历程是能够通过下混频给恢复出来的。

例如当目标做匀速运动$R(t) = R_0 - vt$时，经过解调后，第一个脉冲的基带回波信号变为
$$
x_0(t) = A(t - \frac{2 R_0}{c}) \exp(-\mathrm{j} \frac{4\pi R_0}{\lambda}) \exp(\mathrm{j} 2 \pi (\frac{2v}{\lambda}) t)
$$
可以发现，复包络的变化仅仅近似为一个延迟，去除载频后，留下了由于目标运动导致载频的相位调制，即相位历程。

第$m + 1$的脉冲发射信号为
$$
s_m(t) = A(t - m T_r) \exp(\mathrm{j} 2 \pi f_0 t + \varphi_0)
$$
注意上式中仅有复包络项进行了延迟，载频仍然保持不变（相参体制）。此时回波信号为
$$
\begin{aligned}
r_m(t) &= A(t - mT_r - \frac{2 R(t)}{c})\exp\left[\mathrm{j} 2 \pi f_0 (t - \frac{2 R(t)}{c}) + \varphi_0\right] \\
&\approx A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R(t)}{\lambda}) \exp(\mathrm{j} 2 \pi f_0 t + \varphi_0)
\end{aligned}
$$
可以看到，第$m$个脉冲与第$0$个脉冲的回波信号仅仅有复包络的位置不同，复包络的变化不光有脉冲之间发射时间间隔的延迟，还有由于目标运动的延迟。对于单脉冲的复包络来说，目标是静止的，但对于下一个脉冲的复包络来说，目标好像瞬间跳到下一个脉冲时刻的位置，并仍然保持静止，这就是停跳近似。由于相位历程是载频所引起的，载频在整个时间轴都保持相同形式，所以相位历程也在整个时间轴保持相同形式。下混频后，得到第$m + 1$的脉冲的基带信号。
$$
x_m(t) = A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-\mathrm{j}\frac{4\pi R(t)}{\lambda})
$$
**信号模型（载频相同，相位不连续但已知）**

注意在前面的讨论中，所有脉冲共同使用一个载频$\exp(\mathrm{j}2 \pi f_0 t + \varphi_0)$，若不同脉冲之间的载频并不连续，也即第$m$个脉冲的回波信号为
$$
\begin{aligned}
s_r(t; m) &= A(t - mT_r - \frac{2 R(t)}{c})\exp\left[\mathrm{j} 2 \pi f_0 (t - \frac{2 R(t)}{c}) + \varphi_m\right] \\
&\approx A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R(t)}{\lambda}) \exp(\mathrm{j} 2 \pi f_0 t + \varphi_m)
\end{aligned}
$$
可以看到，若每段脉冲使用对应的下混频参考信号（相位严格相同）$\exp(\mathrm{j} 2 \pi f_0 t + \varphi_m)$，则仍然能保持相位历程是连续的，仍然能够MTD。

**信号模型（载频不同，相位必然不连续，但已知）**
若使用脉间捷变信号，即不同脉冲之间的载频和相位$\exp(\mathrm{j} 2 \pi f_m t + \varphi_m)$是不同的，由此得到回波模型
$$
\begin{aligned}
s_r(t; m) &= A(t - mT_r - \frac{2 R(t)}{c})\exp\left[\mathrm{j} 2 \pi f_m (t - \frac{2 R(t)}{c}) + \varphi_m\right] \\
&\approx A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R(t)}{\lambda_m}) \exp(\mathrm{j} 2 \pi f_m t + \varphi_m) \\
&= A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R_0}{\lambda_m}) \exp(\mathrm{j} 2 \pi (\frac{2v}{\lambda_m}) t ) \exp(\mathrm{j} 2 \pi f_m t + \varphi_m)
\end{aligned}
$$
使用对应参考信号$\exp(\mathrm{j} 2 \pi f_m t + \varphi_m)$下混频后，得到
$$
\begin{aligned}
s(t; m) &= A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R(t)}{\lambda_m}) \\
&= A(t - mT_r - \frac{2 R(mT_r)}{c})\exp(-j\frac{4\pi R_0}{\lambda_m}) \exp(\mathrm{j} 2 \pi (\frac{2v}{\lambda_m}) t ) 
\end{aligned}
$$
对于一般慢速目标来说，每个脉冲相当于有了一个新的相位历程，在慢时间轴的采样相当于解一组方程组，未知参数包含在了$R(t)$中。

对于匀速目标而言，若不改变脉间频率，则相位历程相当于一个初相固定的正弦信号（复指数信号），包含未知数$R_0$和$v$，由于未知数的位置很好，很方便在距离维和速度维搜索。

对于匀速目标且脉间频率改变的模型而言，在慢时间轴上的采样同样相当于解一组方程组，但是由于分母中存在$\lambda_m$的关系，不能再像单频一样做距离维和速度维的搜索了。*为什么不将信号同时做一个指数的计算呢？*
$$
\left[\exp(-j\frac{4\pi R(t)}{\lambda_m}) \right]^{\lambda_m / \lambda_0} = \exp(-j\frac{4\pi R(t)}{\lambda_0})
$$

**采样**
前面讨论的是连续信号从射频到载频的信号模型，接下来对该基带信号进行采样。假设共有$M$个脉冲，并且在每次脉冲发射后的$2 R_s / c$时刻进行采样（每个脉冲先采一个点），即需要要求在$M T_r$的时间内，脉冲的回波延迟不会超过这个时刻，即目标应在距离范围$[R_s - c \tau / 2, R_s]$内（因为是从矩形窗前沿开始计时的），此时第$m + 1$个脉冲的采样值为
$$
s_m(m T_r + \frac{2 R_s}{c}) = A\left(\frac{2}{c}[R_s - R(m T_r)]\right) \exp(-\mathrm{j} \frac{4 \pi }{ \lambda} R\left(m T_r + \frac{2 R_s}{c}\right))
$$
由于复包络内$\frac{2}{c}[R_s - R(m T_r)]$变化极其微小（目标速度不快），故相位的变化忽略不计，所以就相当于在相位历程中等间隔$T_r$采样。当目标在匀速运动时，相位历程就为一个单频的复指数信号，各个采样点为
$$
\begin{aligned}
\exp(-\mathrm{j} \frac{4 \pi }{ \lambda} R\left(m T_r + \frac{2 R_s}{c}\right)) &= \exp(-\mathrm{j} \frac{4 \pi }{ \lambda} R_0  )\exp(\mathrm{j}2 \pi \left(\frac{2v}{\lambda}\right)\left(m T_r + \frac{2 R_s}{c}\right)) \\
&= \exp(\mathrm{j} 2 \pi f_d T_r m + \phi_0)
\end{aligned}
$$
相当于对复指数信号等间隔采样。

综上所述，不管目标是怎样的运动特性，使用快慢时间轴排列的方式都相当于对相位历程的等间隔采样。

**自相关后的距离徙动**
前面说到，在慢时间轴上采样（采样周期$T_r$）相当于对相位历程进行采样。然而，相位历程并不是在整个时间轴上都是有值的，其只在个别位置，即复包络的位置有值。同时，若按照快慢时间轴排列后，根据停跳近似，相邻脉冲之间目标的运动会使得不同脉冲的复包络在快时间轴上相对变化，即脉冲无法对准。我们只能保证在所有脉冲都有值的同一快时间轴采样，才能保证均匀采样，从而进行积累。

为了保证不发生距离徙动，即整列的数据都能进行积累，需要保证
$$
|R(MT_r) - R(0)| < \frac{c T_e}{2}
$$
即一整个CPI的时间$MT_r$内，目标走过的距离要少于一个脉宽的距离（距离单源），这样在整个快时间轴这么多的距离单元中，总有一个距离单元是能够采样到有用的点，如图所示

![7E77098E2F537E022FF3A9FAEEB3643E](D:\qq\820936392\FileRecv\MobileFile\7E77098E2F537E022FF3A9FAEEB3643E.png)

实际上MTD发生在快时间轴的脉冲压缩之后，脉冲压缩之后使得原本的脉宽$T_e$变为$1 / B$，所以上式重新修改为
$$
|R(MT_r) - R(0)| < \frac{c}{2 B}
$$
由于基带数据的采样率通常就是带宽$B$，所以$c / 2 B$就是一个距离单元的长度，换句话说，每当目标在一个CPI的时间内运动距离大于一个距离单元（距离分辨率）时，就会发生距离徙动。

此时还有一个问题亟待解决，每个快时间轴上做脉冲压缩（自相关）后，是否仍然能保持相位历程的相位关系呢？为了简化问题，只考虑相邻两个脉冲的同一个距离单元，假设没有发生距离徙动，那么这两个脉冲在距离单元上是对齐的，第一个脉冲的相关后为
$$
y_0(t) = \int_{-\infty}^{\infty} A(\tau - \frac{2 R(0)}{c})\exp(-\mathrm{j}\frac{4 \pi R(\tau)}{c}) A^*(-t + \tau) \mathrm{d} \tau \\
$$
第$m + 1$个脉冲的自相关结果为
$$
\begin{aligned}
y_m(t) &= \int_{-\infty}^{\infty} A(\tau - mT_r- \frac{2 R(mT_r)}{c}) \exp(-\mathrm{j} \frac{4 \pi R(\tau)}{c}) A^*(-t + \tau) \mathrm{d} \tau \\
&= \int_{-\infty}^{\infty} A(\tau' - \frac{2 R(mT_r)}{c}) \exp(-\mathrm{j} \frac{4 \pi R(\tau' + mT_r)}{c}) A(-t + \tau' + mT_r) \mathrm{d} \tau'
\end{aligned}
$$
假设在发射脉冲$2 R_s / c$时间后进行采样，则
$$
y(mT_r + \frac{2 R_s}{c}; m) = \int_{-\infty}^{\infty} A(\tau'- \frac{2 R(mT_r)}{c}) \exp(-\mathrm{j} \frac{4 \pi R(\tau' + mT_r)}{c}) A(-\frac{2 R_s}{c} + \tau') \mathrm{d} \tau'
$$
与前面一个脉冲自相关后的结果进行对比
$$
y\left((m - 1)T_r + \frac{2 R_s}{c}; m - 1\right) = \int_{-\infty}^{\infty} A(\tau'- \frac{2 R\left((m - 1)T_r\right)}{c}) \exp(-\mathrm{j} \frac{4 \pi R(\tau' + (m - 1)T_r)}{c}) A(-\frac{2 R_s}{c} + \tau') \mathrm{d} \tau'
$$
若将复包络内$R(mT_r)$而引起的相位变化忽略，那么上下两个式仅仅只差相位历程，对于匀速运动目标，相位历程$\exp(-\mathrm{j} \frac{4 \pi R(t)}{c}) = \exp(-\mathrm{j} \frac{4 \pi R_0}{c})\exp(\mathrm{j} 2 \pi f_d T_r m)$，那么相邻脉冲仅仅差别了$\exp(\mathrm{j} 2 \pi f_d T_r)$的相位，仍然能够进行积累。

其实从本质上来说，脉冲压缩为每一个快时间轴的对应距离单源做了对应的同一线性处理，而相位项知识一个常系数，在线性处理的时候可以提出来，所以相位的相对关系保持不变。

综上所述，即使快时间轴做了脉冲压缩，但是由于相邻脉冲的相位历程的相位差与积分元无关，故可以作为常数处理，而使得脉压之后仍然保持相同的相位差。注意两个信号自相关的表达式应该为
$$
y(t) = \int_{-\infty}^{\infty} x_1(\tau) x_2^*(-t + \tau) \mathrm{d} \tau
$$
此时$x_1(t)$固定不动而$x_2(t)$在滑窗。其他的表达式还需要学。

### 匹配滤波器（很乱，还没整理）

先说两个重要结论

**1. 正定二次型的时频域相互转换**
一个及其常见的二次型为
$$
\begin{aligned}
T(\mathrm{x}) &= \mathrm{x}^{T} C^{-1} \mathrm{s} = \int_{-1 / 2}^{1 / 2} \frac{X^*(f) S(f)}{P_{ww}(f)} \mathrm{d} f\\
&= \mathrm{s}^{T}C^{-1}\mathrm{x} = \int_{-1 / 2}^{1 / 2} \frac{S^*(f) X(f)}{P_{ww}(f)} \mathrm{d} f
\end{aligned}
$$
例1：广义匹配滤波器的优化目标（高斯均值偏移系数）为
$$
d^2 = \mathrm{s}^{T}C^{-1}\mathrm{s} = \int_{-1 / 2}^{1 / 2} \frac{|S(f)|^2}{P_{ww}(f)} \mathrm{d} f
$$
例2：一个WSS随机矢量过一个LTI系统的方差为（注意此时的$\mathrm{h}$与正常的顺序是相反的，正真的滤波器设计需要再反折）
$$
\begin{aligned}
E\left[(\mathrm{w}^{T} \mathrm{h})^2 \right] &= \mathrm{h}^{T}C\mathrm{h} \\
&= \int_{-1 / 2}^{1 / 2} P_{ww}(f) |H(f)|^2 \mathrm{d} f
\end{aligned}
$$
相应的，连续的版本为（这里$\mathrm{h}$对应的时域是$h(-t)$）
$$
\begin{aligned}
R_y(0) &= h(t) * R_x(t) * h(-t) \\
&= \int_{-\infty}^{\infty} P_{ww}(f) |H(f)|^2 \mathrm{d} f
\end{aligned}
$$
连续频域和离散频域的差别只在于积分的上下限。

**2. 信噪比**
一般而言，接收机接收到的信号是一个确知信号加上噪声的形式，如下
$$
x(t) = s(t) + w(t)
$$
故$x(t)$本身而言是一个随机信号，该随机信号的信噪比定义如下
$$
\eta(t) = \frac{E^2[x(t)]}{\mathrm{var}(x(t))}
$$
若该信号$x(t)$经过一个冲击响应为$h(t)$的系统，得到输出信号
$$
\begin{aligned}
y(t) = x(t) * h(t) &= \int_{-\infty}^{\infty}x(\tau) h(t - \tau) \mathrm{d} \tau \\ 
&= \int_{-\infty}^{\infty} X(f) H(f) \exp(\mathrm{j} 2 \pi f t) \mathrm{d} f
\end{aligned}
$$
输出信号$y(t)$的信噪比根据公式
$$
\eta_o(t) = \frac{E^2[y(t)]}{\mathrm{var}(y(t))} = \frac{E^2\left[\int_{-\infty}^{\infty}x(\tau) h(t - \tau) \mathrm{d} \tau\right]}{E\left[(\int_{-\infty}^{\infty}w(\tau) h(t - \tau) \mathrm{d} \tau )^2\right]}
$$
分子的形式很好确定，因为平方在期望的外面，而期望内部是线性表达式，困难的是分母如何求解。分母需要求解一个点乘形式的二阶矩，若切换为**离散**形式，由于协方差矩阵的存在，故很好进行转换，即
$$
\begin{aligned}
E\left[(\mathrm{w}^{T} \mathrm{h})^2 \right] &= E\left[\mathrm{h}^{T}\mathrm{w}\mathrm{w}^{T}\mathrm{h}\right] \\
&= \mathrm{h}^{T}E[\mathrm{w}\mathrm{w}^{T}]\mathrm{h}\\
&= \mathrm{h}^{T}C\mathrm{h} 
\end{aligned}
$$
然而在连续域中，随机过程经过了一个LTI系统，时域上描述为
$$
\begin{aligned}
E\left[(\int_{-\infty}^{\infty}w(\tau) h(t - \tau) \mathrm{d} \tau )^2\right] &= h(-t) * R_w(t) * h(t) \Big|_{t = 0} \\
&= R_{w_o}(0)
\end{aligned}
$$
**匹配滤波器的最大输出信噪比（离散版本）**
$$
\eta = \frac{E^2\left[ y[N - 1]; \mathcal{H}_1 \right]}{\mathrm{var}(y[N - 1]; \mathcal{H}_1)} = \frac{\mathrm{s}^T \mathrm{s}}{\sigma^2}
$$
其中$y[N - 1] = \sum\limits_{k = 0}^{N - 1} x[k] h[N - 1 - k]$，且$x[k] = s[k] + w[k]$。这里的信号能量为某一时刻信号能量的平方，噪声是某一时刻信号的方差。

**匹配滤波器连续**
$$
\chi=\frac{\left|y(T_{M})\right|^{2}}{n_{p}}=\frac{\left|\int X(f)H(f)\exp{(\mathrm{j}2 \pi fT_{M})}\mathrm{d}f\right|^{2}}{N_0\int \left|H(f)\right|^{2}\mathrm{d}f}
$$
上式的来源依然是某时刻功率的平方比上该时刻的方差，不过上式没有显示的表现出来，具体来说匹配滤波器的输出
$$
y'(t) = \int x'(\tau) h(t - \tau) \mathrm{d} \tau \\
x'(t) = x(t) + w(t) \\
\begin{aligned}
E\left[ y'(t) \right] &= \int E\left[ x'(\tau) \right] h(t - \tau) \mathrm{d} \tau = \int x(\tau) h(t - \tau) \mathrm{d} \tau\\
&= \int X(f) H(f) \exp(j 2 \pi f t) \mathrm{d}f \\
&= y(t)
\end{aligned}
$$
这里直接使用确定信号进5行计算功率是由于噪声的零均值特性，忽略了求期望的操作。

同样，噪声的功率就是$y'(t)$的方差
$$
\begin{aligned}
\mathrm{var}(y'(t)) &= E\left[ \left(y'(t) - y(t)\right)^2 \right] \\
&= E\left[ \left( \int\left(x'(\tau) - x(\tau)\right) h(t - \tau) \mathrm{d}\tau \right)^2 \right] \\
&= E\left[ \left( \int w(\tau) h(t - \tau) \mathrm{d}\tau \right)^2 \right] = E\left[ w_o^2(t) \right] \\
&= R_o(0) = \int P_{w}(f) |H(f)|^2 \mathrm{d} f \\
&= N_0 \int |H(f)|^2 \mathrm{d} f 
\end{aligned}
$$
综上所述，一个随机信号某一时刻的信噪比可以定义为该时刻信号期望的平方比上该时刻信号的方差。

**部分匹配滤波**
为了在雷达信号处理的过程中降低信噪比，从仿真的角度更好地对TBD进行模拟，可以使用部分匹配滤波造成的失配来完成，一个带宽为$B$的LFM信号，使用带宽为$B / N$的LFM来进行匹配。

先分析噪声，从离散时域上
$$
{E}[(\mathrm{w}^T \mathrm{h})^2] = \mathrm{h}^T C \mathrm{h} = \sigma^2 \mathrm{h}^T \mathrm{h}
$$
而信号功率
$$
E^2[\mathrm{h}^T\mathrm{x}] = (\mathrm{h}^T \mathrm{h})^2
$$
故从结果上，信号增加能量的平方倍，噪声增加能量倍，故最终信噪比增加能量倍。

从连续频域上来说，输出信号为（假设非因果，时延为0）
$$
y(0) = \int S(f) H(f) \mathrm{d} f = \int \left| S(f) \right|^2 \mathrm{d} f \\

|y(0)|^2 = \left[ \int \left| S(f) \right|^2 \mathrm{d} f \right]^2
$$

信号为随机过程进入一个LTI系统
$$
n_p = \int P_w(f) \left| H(f) \right|^2 \mathrm{d} f = N_0 \left[ \int \left| S(f) \right|^2 \mathrm{d} f \right]
$$
故最终不管是连续域还是离散域，一个白噪声进入一个LTI系统，则白噪声的功率谱密度提升传递函数的能量倍数，只不过在连续域中，白噪声能量谱密度为$N_0$而在离散域中，功率谱密度就是方差$\sigma^2$。对于信号来说，若是刚好匹配，则得到输出的幅度为原始信号能量，功率为原始信号能量的平方。所以，对于匹配滤波器的输出，其信号功率为原始信号能量的平方，而噪声功率为功率谱密度的信号能量倍，最终信噪比为原始信号能量与白噪声功率谱密度之比。

通过匹配滤波器前后信噪比的变化，可以确定一个处理增益（PG）的定义。在离散域中，输入信噪比为$(\varepsilon / N) /  {\sigma^2}$，输出信噪比为$\varepsilon  /  {\sigma^2}$，处理增益$\mathrm{PG} = N$；连续域中，输入的信噪比为$(\varepsilon / T) / (B N_0)$，输出信噪比为$\varepsilon / N_0$，处理增益$\mathrm{PG} = BT$为时宽带宽积。

注意，连续高斯白噪声是一个理想的概念，其由于频谱无限所以现实中不存在，现实中都是离散版本的高斯白噪声，所以在将连续域的噪声功率时，我使用$B N_0$来彰显这一点，将该连续域以$B$的采样率来离散化之后，就得到了一个离散的高斯白噪声，其方差才描述为$\sigma^2 = B N_0$。

### 雷达接收信号的随机与确定性分析

雷达的接收信号可以被建模如下
$$
x[n] = s[n] + w[n]
$$
其中，$w[n]$是WGN。而$s[n]$常常被建模为WSS随机过程，那么既然雷达发射的信号是已知的，为什么仍然建模为随机信号呢？

一个典型的例子是相位均匀分布的正弦信号是WSS随机过程，同时对于接收端来说，接收信号收到了一些随机性参数（参数为随机变量）的控制，比如随机时延，随机相位等等，故建模为WSS随机过程。但是与此同时，建模为随机过程只使用了前二阶的统计特性，我们应该还掌握更深层次的信息，所以从某种程度上假设为WSS随机过程损失了一下信息量。

*但是对于雷达信号来说，每次都建模为WSS随机过程是否一直都是有效的呢？*



### 脉冲体制的MTD与连续波的互模糊函数对距离和速度搜索的关系和比较

