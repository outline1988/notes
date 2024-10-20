### 序列的matlab表示

在matlab中，一个序列需要用两个等长的向量表示，一是样值向量，也就是纵坐标的值，二是位置向量，也就是横坐标的值。

如下生成一个固定数字角频率正弦序列

```matlab
% initial agr
x_len = 25;
dig_omega = pi / 8;
% implementation
n_x = 0 : x_side_len - 1;	% 位置向量
x = sin(dig_omega * n_x);	% 样值向量
stem(n_x, x, '.');
```

 其中stem函数是用来画出序列的图像的，第一个参数位置向量，第二个参数样值向量。

### 序列卷积和多项式乘法

使用conv函数可以实现两个序列的卷积和于其等效的多项式乘法。
```matlab
u = [1, 1, 1, 1];
v = [2, 2, 2];
y = conv(u, v);
```

卷积得到的长度是两个序列长度之和再减1。

在此基础上可以编写带有位置向量的序列卷积，该算法解释如下
第一是确定样值向量，不论在哪个位置卷积，都是其中一个序列的翻转向左（n为负）向右（n为正）滑过的结果，所以最终得到的样值向量是相等的，直接用conv函数来生成。
其次是确定位置向量，以两个序列的位置向量都从大于0的位置开始，其中之一的序列关于y轴翻转之后需要再右移该序列第一个样值的位置和另一序列第一个样值的位置之和，该值就是卷积完后的序列的第一样值位置，而至于最后一个位置的值就可以很轻松的根据第一位置和长度得到，也可以通过最后一个位置之和得到。

故可以写出下列函数
```matlab
% 带有位置向量的卷积函数
function [y, y_n] = convu(u, u_n, v, v_n)
    y = conv(u, v);
    
    y_n_begin = u_n(1) + v_n(1);
    y_n_end = u_n(end) + v_n(end);
    y_n = y_n_begin : y_n_end;
```

### 求解差分方程

搭配使用filter函数和filtic函数可以求解任意常系数差分方程的解。

每个常系数差分方程可有以下方程表示
$$
\sum_{i = 0}^{N} A_i y(n - i) = \sum_{i = 0}^{M} B_i x(n - i)
$$
其中$A_0$一般为1，就算不为1，在求解的时候也会将其归一化。

```matlab
y_n = filter(B, A, x_n);		% 求解零状态响应
y_n = filter(B, A, x_n, x_y_i)	% 求解全相应，初值信息在x_i中
```

需要注意的是，matlab中的求解都是都数值求解，故输入的$x(n)$和输出的$y(n)$的是等长的有限序列，一般将$x(n)$表示为单位序列$\delta(n)$，并在matlab中进行`x_n = [1, zeros(1, 30)]`这样的初始化，其中30为任意你想要的长度，最终得到的输出$y(n)$的长度为31。

关于`x_i`，其为一序列包含了$x(n)$和$y(n)$的初值信息，用如下方法求得
```matlab
x_y_i = filtix(B, A, y_i, x_i);
```

其中`x_i`和`y_i`分别为$x(-1) \cdots x(-N)$和$y(-1) \cdots y(-N)$的序列，最终`x_y_i`的长度为M和N的较大值，若$x(n)$因果序列则省略不写，例如下例求解
$$
y(n) - 0.8y(n - 1) = x(n), \space y(-1) = 1
$$

```matlab
% 使用filter函数对例141进行求解
% y(n) - a * y(n - 1) = x(n)
% 左边系数为A，右边系数为B
% filter函数先右边系数再左边系数
A = [1, -0.8];
B = [1, 0];
y_i = [1, 0];
x_n = [1, zeros(1, 29)];	% 最终得到长度30的序列
n = 0 : 29;

x_y_i = filtic(B, A, y_i);
y_n = filter(B, A, x_n, x_y_i);
stem(n, y_n);
```

### 一个理想采样恢复的程序

理想采样信号就是冲激串函数和模拟信号乘积，而其在频域上的特点就是模拟信号频域的以$2 \pi/T_s$的周期延拓，为了保证不出现频谱混叠，则要求$\Omega_{max} < \Omega_s - \Omega_{max}$即$\Omega_s > 2 \Omega_s$，这便是奈奎斯特采样定理。

将理想采样信号还原为模拟信号，就是将其通过理想低通滤波器（门宽为$\Omega_s$的矩形窗），将频域中的一个重复分量分离出来，最后经过推导可以得到恢复的模拟信号为
$$
\begin{aligned}
x_{a}(t) &=\sum_{n=-\infty}^{n=\infty} x(n) \frac{\sin [ \pi\left(t-n T_{s}\right)/T_s]}{[\pi\left(t-n T_{s}\right)/T_s]} \\
&=\sum_{n=-\infty}^{n=\infty} x(n) g\left(t-n T_{s}\right)
\end{aligned}
$$
下列代码根据上述推导得到的结论，由$x(n)$内插而还原。
```matlab
clear;
% 恢复理想采样的模拟信号，由于采样点数不是无限，只有中间一部分能够较为完整的恢复
f_s = 1e6;      % 采样周期为1Mhz
T_s = 1 / f_s;

f_0 = 400e3;    % 被采信号频率400khz
T_0 = 1 / f_0;

n_length = 20;
n = 0 : n_length - 1;      % 采集n_length个点
x_n = sin(2 * pi * (f_0 / f_s) * n + pi / 6);

t_length = 5000;    
t = linspace(-n_length * T_s, n_length * T_s, t_length);

y = zeros(1, length(t));    % 应该可以用向量方式生成

% 返回一个矩阵，列的维数和n相同，行的维数和相同，及对于每一个延迟n。
y = x_n * g_inter(t, n, T_s);

t_begin = length(t) / 2;
t_end = ( (n_length - 1) / (n_length * 2) ) * t_length + t_begin;
plot(t(t_begin : t_end), y(t_begin : t_end));
```

要想通过内插来完美还原$x_a(t)$，需要满足两个条件，一是频域有限，由此不会造成混叠，二是时域有限，由此才能够实际采样，但是频域有限的信号往往时域无限，频域有限的信号往往对应着时域无线，所以实际中难以完成完美的理想还原。

### 离散时间傅里叶变换（DTFT）

**使用矩阵相乘求DTFT**
$$
X(e^{jw}) = \sum_{n = -\infty}^{+\infty}x(n)\mathrm{e}^{-jwn}
$$
对于DTFT来说，其时域是有限的序列，而频域是连续的函数，所以需要首先创建三个矩阵，以R_N为例（N=4），如下
```matlab
x = [1, 1, 1, 1];
n = [0 : 3];
w = linspace(-2.8 * pi, 2.8 * pi, 1000);
```

由于$w$是连续的，所以你可以暂时将其放一边

对于求和符号来说，最终得到的是一个数，所以总体的形式是横向量乘列向量的形式，由公式可知，横向量是$x(n)$，列向量是$e^{-jwn}$。

对于$e^{-jwn}$来说，最后得到的是列向量，所以$n$应该以列向量形式出现。至于$w$，为了不影响总体的列向量，其只能作为列向量的横向拓展，程序如下
```matlab
X = x * exp(-j * n' * w);	% n'为列向量，n' * w为列向量的横向拓展
```

将上述代码封装成函数
```matlab
function [X_w, w] = dtft(x_n, n_x, w)
    if (~iscolumn(n_x))
        n_x = n_x';
    end
    X_w = x_n * exp(-1j * n_x * w);
end
```

**使用fft函数得到DTFT**
可知，DFT变换就是在DTFT的基础上，N点等间隔进行采样，如果点数足够多，那么该序列的DFT就与能还原其DTFT。

那么如何能够在不改变频谱的基础上增加采样点数呢？如果增加采样率，可以提高点数，但是会导致频谱的最高频率发生改变。
故最好的办法是延长采样时间，但是由于DTFT是对无限长的序列进行的，实际中我们只能选取只在部分位置有值的序列来进行DTFT，所以实际上只需要不断地补零就可以了。

```matlab
% 只能在一个区间上进行，所以w必须在0到2 * pi
function [X_w, w] = dtft(x_n, n_x, w)
    if n_x(1) ~= 0
        x_n = [ones(1, n_x(1)), x_n];
    end
    X_w = fft(x_n, length(w));
end
```

有缺陷，但是我不想改：）。


### 使用DTFT画频谱

序列的DTFT变换实际上是对某个模拟信号进行冲激理想采样后的模拟傅里叶变换，用卷积的方法来做可以得到
$$
\hat{x_a}(t) = x_a(n T_s) = x_a(t)\cdot\sum_{n = -\infty}^{+\infty}\delta(t - n T_s)
$$
其傅里叶变换为，即原始模拟信号的周期延拓
$$
\hat{X_a}(j\Omega) = \frac{1}{T_s} \cdot X_a(j\Omega) * \delta(\Omega - n\frac{2\pi}{T_s})
$$
用定义的方法，见书p61页（高西全版本），得到的结果如下
$$
\hat{X_a}(j\Omega) = \sum_{n = -\infty}^{+\infty}x_a(nT_s)\mathrm{e}^{-j\Omega n T_S}
$$
将模拟角频率和数字角频率的关系用上，即可得到与序列的DTFT一模一样的表达式，即
$$
X(e^{jw}) = DTFT[x(n)] = \hat{X_a}(j\Omega)
$$
也就是说，通过采样得到的序列$x(n)$的DTFT变换和用理想冲击采样得到的采样信号的傅里叶变换的表达式是一摸一样的。而冲激采样信号的频谱恰恰是原始模拟信号的周期延拓。既然我们可以通过截断冲击采样的频谱来得到模拟信号的频谱，那当然能过够通过阶段采样序列$x(n)$的数字频谱来得到模拟信号的频谱，代码如下

```matlab
clear;
% 画出模拟信号的频谱图
% 生成一个模拟信号，取方波序列的半个周期加一门宽
f_0 = 10e3;     % 模拟频率10khz
T_0 = 1 / f_0;
% x_a = 0.5 * (square(2 * pi * f_0 * (t + 0.25 * T_0)) + 1);

% 采样参数
f_s = 200e3;    % 采样频率200khz
T_s = 1 / f_s;

t_sample_length = 0.6 * T_0;    % 要采样模拟信号的范围

% 理论上用dtft还原限制频率的频谱需要无穷的点，所以给出模拟信号只在中心0.5周期的地方有值，其他地方为0
n_length = t_sample_length * f_s;   % 采样点数计算

n = -n_length / 2 : n_length / 2;
n_length = length(n);   % 更新n的长度

% 采样
x_n = 0.5 * (square(2 * pi * f_0 * (n * T_s + 0.25 * T_0)) + 1);

dig_w = linspace(-pi, pi, 1000);	% 数字角频率
X_w = dtft(x_n, n, dig_w);

ana_w = dig_w * f_s / (2 * pi);		% 模拟角频率


plot(ana_w / 1e3, T_s * abs(X_w));
xlabel('kHz');
```

需要注意的是，完美的得到模拟信号的频谱图需要负无穷到正无穷的时间，即n的范围是负无穷到正无穷，所以上述代码只实现了门函数的频谱还原，因为该函数只有中间半个周期的地方有值，并且还受到了一定程度的频谱混叠。

### 使用FFT画频谱

**fft**

```matlab
X_k = fft(x_n, N);
x_n = ifft(X_k, N);
```

进行N点的DFT变换和IDFT变化。

虽然又很多种形式，但是最主要的就是记住这种形式，其他的到时候查就可以了。

该函数为`x_n`做`N`长度的DFT变换`N`过长则补零，默认的位置向量都是从零开始的。

**fftshift**

可以知道，使用matlab画出模拟信号的频谱图的原理在于对采样序列的DTFT是其模拟信号频谱的周期延拓，所以可以只取一个周期，再加上足够多的点来近似模拟频谱。

而DFT是DTFT的等间隔采样，若做DFT的点数足够多，也能近似表示DTFT（当然从定义出发的前一个方法殊途同归），不过由于DFT只在0到N-1上有值，所以还有配合`fftshift`函数来进行频谱的偏移。

`fftshift`对fft和ifft后的结果进行操作，而`ifftshift`还原`fftshift`操作后的结果。

下面程序用来画出LFM的频谱图

```matlab
clear; close all;
T = 10e-6;    % 脉冲宽度
B = 8e6;           % 带宽 25Mhz
f_0 = 10e6;         % 载波频率 10Mhz
f_s = 3 * (f_0 + B / 2);        % 采样率

N = T * f_s;
u = B / T;

% 生成一个周期(-T/2, T/2)的复包络
t = linspace(-T / 2, T / 2, N);

u_t = (1 / sqrt(T)) * exp(1j * pi * u * t.^2);

s_t = u_t .* exp(1j * 2 * pi * f_0 * t);

S_f = fftshift(fft(s_t));
f = linspace(-f_s / 2, f_s / 2, N);

figure;
plot(f / 1e6, abs(S_f));
xlabel('Mhz');
grid on; axis tight;
```

最终你只需要调用`fftshift(fft(s_t))`，然后适当调整横坐标，就可以得到模拟信号的频谱图。

以为调整横坐标很简单，但实际上你还是不会调。

fftshift是怎样平移的？
我们知道，对于N点有限长共轭对称序列，其对称的地方在$[1, N-1]$的部分，而fftshift的平移就是关于其右半边（-1）的部分移到最左边。

```matlab
Xeven = [1 2 3 4 5 6];
fftshift(Xeven);
% ans = [4 5 6 1 2 3];
```

```matlab
Xodd = [1 2 3 4 5 6 7];
fftshift(Xodd);
% ans = [5 6 7 1 2 3 4];
```

fftshift的实现源代码，即向右循环移位$\lfloor \frac{N}{2} \rfloor$（向下取整）个元素。
```matlab
x = circshift(x, floor(size(x) / 2));
```

怎样平移很坐标呢？
找到中心频率，即$\lfloor \frac{N}{2} \rfloor$对应的频率

```matlab
k = 0 : N - 1;
f = (k - floor(N / 2)) * f_s / N;
```

把握住最终的范围为$[0, f_s)$，或者$[-f_s / 2, f_s / 2)$。左闭区右开，所以遇到$N$为偶数比如4，则$2 / 4$的频率属于负频率，正频率或者负频率（不包含DC分量）的数量都是$\lfloor \frac{N}{2} \rfloor$。

fftshift是按照上述这样画的，但是periodogram去正频率就有点差别。
### 矩阵的创建和重构

**linspace**

```matlab
y = linspace(x1, x2, n);
y = linspace(x1, x2);
```

第一个函数生成从x1到x2等间隔的n个点所形成的`1*n`的行向量，使用`‘`可以将其转置。

比如，生成从0到2pi的n个点所形成的列向量

```c
n = 100;
v = linspace(0, 2 * pi, n)';
```

**rand**

```matlab
x = rand;		% 返回一个区间(0, 1)的数
x = rand(x); 	% 返回n*n的随机数矩阵
```

可以通过参数的类型来记住rand函数

- 没有参数就是返回一个0到1的数。
- 一个参数就是返回那个参数的方阵。

**翻转（flip）**

```matlab
flipud(A);	% 上下翻转 flip up-down
fliplr(A);  % 左右翻转 flip left-right
```

`flipud`函数上下翻转，所以对行向量操作不改变。

同理`fliplr`函数不对列向量做出改变。

**单位阵**

```matlab
I = eye(n);
```

返回$n \times n$的单位阵。

**循环平移矩阵**

```matlab
Y = circshift(A, K);
Y = circshift(A, K, dim);
```

A是被循环移位的矩阵；K是平移的大小；dim是维度，对于二维矩阵，dim=1对应列、dim=2对应行。

K为正数则向右向下平移。

**翻转和循环平移求出序列的共轭对称序列**

```matlab
% 返回该序列的共轭对称序列和共轭反对称序列
function [x_e_n, x_o_n] = conjsym(x_n)
    x_e_n = ( x_n + conj( fliplr( circshift(x_n, -1) ) ) ) / 2;
    x_o_n = -1j * ( x_n - conj( fliplr( circshift(x_n, -1) ) ) ) / 2;
end
```

**拼接（cat）**

```matlab
C = cat(dim, A, B);
```

按照增加dim维度长度的原则拼接数组。

例如对于矩阵，dim为1时，由于矩阵的第一维为行，所以最终得到的结果是行增加的拼接方式。

**大量拼接向量或矩阵（repmat）**

```matlab
B = repmat(A, sz1, sz2);
```

用A作为子元素形成一个以sz1和sz2为新“尺寸”的矩阵。

如果只是简单的重复，与1矩阵相乘也可以得到同样的效果，且性能和可读性更好。

**取整操作**

```matlab
Y = ceil(X);	% 朝着正无穷取
Y = fix(X);		% 朝着0取
Y = floor(X);	% 朝着负无穷取
Y = round(X);	% 朝着最近的整数取
```

朝0取在正数中即使朝者负无穷取，而在负数中即使朝着正无穷取。

### 设计模拟低通滤波器

**freqs**

```matlab
omega = linspace(0, 2 * pi * 14e3, 1000);
H_a_jomega = freqs(B, A, omega);
```

通过多项式系数和位置向量来得到对应系统函数$H_a(s)$。

**Analog Butterworth**
Butterworth模拟低通滤波器的关键参数的$N$和$\Omega_C$，所给的指标都是为了求这两个参数。

1. 确定参数，使用下列函数确定阶数$N$和3db截止频率$\Omega_c$
   ```matlab
   [N, omega_c] = buttord(omega_p, omega_s, alpha_p ,alpha_s, 's');
   ```

2. 通过确定的参数来计算多项式的系数向量
   ```matlab
   [B, A] = butter(N, omega_c, 's');
   ```

3. 通过多项式和位置向量（代表模拟频率）来计算幅频值
   ```matlab
   omega = linspace(0, 2 * pi * 14e3, 1000);
   H_a_jomega = freqs(B, A, omega);
   ```

   关于freqs函数的介绍在前面。

```matlab
% 根据指标设计巴特沃斯滤波器
% 模拟滤波器指标
[f_p, alpha_p, f_s, alpha_s] = deal(5e3, 2, 12e3, 30);

[N, omega_c] = buttord(2 * pi * f_p, 2 * pi * f_s, alpha_p ,alpha_s, 's');
[B ,A] = butter(N, omega_c, 's');

omega = linspace(0, 2 * pi * 14e3, 1000);
H_a_jomega = freqs(B, A, omega);

plot(omega / (2 * pi * 1e3), abs(H_a_jomega));
```

**Analog Chebyshev Ⅰ**
Chebyshev Ⅰ型的模拟低通滤波器关键参数为阶数$N$、通带最大衰减$\alpha_p$和通带边界频率$$\Omega_p$$。

```matlab
% 用切比雪夫1型滤波器
[f_p, alpha_p, f_s, alpha_s] = deal(5e3, 2, 12e3, 30);
[N, omega_p] = cheb1ord(2 * pi * f_p, 2 * pi * f_s, alpha_p, alpha_s, 's');

[B, A] = cheby1(N, alpha_p, omega_p, 's');

omega = linspace(0, 2 * pi * 14e3, 1000);
H_a_jomega = freqs(B, A, omega);

plot(omega, abs(H_a_jomega));
```

注意参数，步骤基本和Butterworth相同。

**Analog Chebyshev Ⅱ**

关键参数为阶数$N$、阻带最小衰减$\alpha_s$和通带截止频率$$\Omega_s$$。

```matlab
[N, omega_s] = cheb2ord(2 * pi * f_p, 2 * pi * f_s, alpha_p, alpha_s, 's');
[B, A] = cheby2(N, alpha_s, omega_s, 's');

omega = linspace(0, 2 * pi * 15e3, 1000);
H_a_2_jomega = freqs(B, A, omega);
```

**Analog elliptic**
关键参数为$N$、通带最大衰减$\alpha_p$、阻带最小衰减$\alpha_s$和通带边界频率$$\Omega_p$$，共四项。

```matlab
% 椭圆滤波器
[f_p, alpha_p, f_s, alpha_s] = deal(5e3, 2, 12e3, 30);

[N, omega_p] = ellipord(2 * pi * f_p, 2 * pi * f_s, alpha_p, alpha_s, 's');
[B, A] = ellip(N, alpha_p, alpha_s, omega_p, 's');

omega = linspace(0, 2 * pi * 15e3, 1000);
H_a_jomega = freqs(B, A, omega);

plot(omega, abs(H_a_jomega));
```

### 设计任意模拟滤波器

**低通到高通**
可以以模拟低通滤波器为蓝本，通过频率变换得到任意样式滤波器，复变量变换公式如下
$$
s = \frac{\Omega^{'}_{0}\Omega_{0}}{s^{'}}
$$
其中$s'$为旧低通滤波器的复变量，$s$为新高通滤波器的复变量，频率变换的本质是复变函数的映射。

观察频率变换公式，其中$\Omega^{'}_{0}$和$\Omega_{0}$是事先确定的，并且这两个量刚好分别为低通的量和与之相对应的高通的量，只要确定了这两个量，就可以得到确切的频率变换。此时你能够确切知道高通的$\Omega_{0}$与低通的$\Omega^{'}_{0}$是相匹的，但是高通还有一些量无法与低通滤波器相对应，所以你可以将高通的量带入变换公式，就可以得到相应的低通的量，再根据低通指标设计滤波器即可。由此通过公式的转换就可以实现指标的唯一对应。

所以上述可以归结为以下步骤

- 确定$\Omega^{'}_{0}$和$\Omega_{0}$，即提前相对应，因为$\Omega^{'}_{0}$的取值是任意的，所以通常取为$j$。
- 通过高通的指标算出其他的低通指标。
- 根据所有的低通指标设计滤波器。
- 再将滤波器的系统函数通过频率变换公式的映射得到新的系统函数。

将上述步骤写成程序
```matlab
close all; clear;

[alpha_p, alpha_s, omega_ph, omega_sh] = deal(0.1, 40, 2 * pi * 4e3, 2 * pi * 1e3);

% 归一化低通滤波器设计
lambda_p = 1;		% 将omega_ph与lambda_p（取值为1）提前相对应，从而确定了频率变换公式
lambda_s = lambda_p * omega_ph / omega_sh;	% 计算其他的低通指标

[butt_N, lambda_c] = buttord(lambda_p, lambda_s, alpha_p, alpha_s, 's');
[Bl, Al] = butter(butt_N, lambda_c, 's');

[Bh, Ah] = lp2hp(Bl, Al, omega_ph);		% 使用lp2hp函数，该函数某人低通的1对应第三个参量所对应的高通量，即omega_ph

omega = linspace(0, 2 * pi * 6e3, 1000);
H_hp_jomega = freqs(Bh, Ah, omega);

plot(omega / (2 * pi * 1e3), abs(H_hp_jomega));
xlabel('kHz'); grid on;
```

其实记住一句话就好，低通的频率1永远对应着高通中的lp2hp的第三个参量。

所以你可以认为`lp2hp`函数的频率变换是基于低通频率的1和高通的第三个参量而变换的。

**其他的频率变换用直接法设计**
简单说说`lp2bp`函数

```matlab
[Bbp, Abp] = lp2bp(Blp, Alp, omega_0, B_w);
```

需要的第三和第四参量是确定的中心频率$\Omega_0$和通带带宽$B_w$，与`lp2hp`函数的第三个参量可以任意确定不同（只要低通的频率1与第三个参量相对应），`lp2bp`函数的第三和第四个参量几乎是提前定好的，这时候你只能以带通的 $\Omega_{pl}$和低通归一的$\lambda_p = 1$相关联。

为了方便记忆，将`lp2hp`的关联参量也设为高通的$\Omega_p$和低通的$\lambda_p = 1$关联。

低通到带通频率转换公式如下（由于对称性，负号不写）
$$
\lambda = \lambda_p \frac{\Omega_0^2 - \Omega^2}{\Omega B_w}, \space\lambda_p = 1
$$
低通到带通频率转换公式如下
$$
\lambda = \lambda_s \frac{\Omega B_w}{\Omega_0^2 - \Omega^2}, \space\lambda_s = 1
$$

### 一个简单的计算引出的三种方法

在设计巴特沃斯滤波器时，需要计算如下式子
$$
G_a(p) = \frac{1}{\prod_{k = 0}^{N - 1}(p - p_k)}
$$
对于matlab来说，$p$是虚轴上的一系列离散的点，其为一向量，只要取足够多的数，就可以模拟曲线，而$p_k$是巴特沃斯滤波器的左半边极点，是长度为$N$的向量。要让matlab算出上式的值，需要先得到每个$p-p_k$的向量，再对应元素相乘，由此可以用下列程序算出
```matlab
% 使用for循环计算
G_a_p = ones(1, length(p));
for each = k
    G_a_p = G_a_p .* (p - p_k(each + 1));
end
G_a_p = 1 ./ G_a_p;
```

此方法简单暴力，但不够优雅，没有充分利用matlab中矩阵的优势。

故尝试用矩阵计算，如下
```matlab
G_a_p_matrix = repmat(p, N, 1) - repmat(p_k', 1, length(p));
G_a_p = 1 ./ prod(G_a_p_matrix);
```

其基本原理是构造两个矩阵从而实现对应元素进行正确的操作，需要事先在草稿纸上演算得出该程序，但是在实际的测试中其性能要比循环版本的略差，原因在于需要 创建两个很大的矩阵，故考虑利用与1向量相乘的方法求新的构造矩阵，如下
```matlab
G_a_p_matrix = ones(N, 1) * p - p_k' * ones(1, length(p));
G_a_p = 1 ./ prod(G_a_p_matrix);
```

其原理与前一程序的原理完全相同，使用了矩阵相乘使其性能较好于使用`repmat`函数，于循环版本相当，但是代码更简洁，易读性不好说。。。（你只要知道他能正确的运行就好了）

### 模拟转数字滤波器

**脉冲响应不变法**
脉冲响应不变法是将模拟系统函数$H_a(s)$按照一定规则转化为数字系统函数$H_d(z)$的方法，由此可以实现由模拟滤波器至数字滤波器的转换。其原理为将模拟系统函数$H_a(s)$转化为模拟脉冲$h_a(t)$后，根据适当的采样率$f_s$对模拟脉冲进行采样，从而得到序列（理论无限长），再将该序列转化为$H_d(z)$从而得到数字滤波器。

其他理论推导请看书。

matlab中使用`impinvar`函数来实现脉冲响应法
```matlab
[B_z, A_z] = impinvar(B_s, A_s, f_s);
```

以下为例子
```matlab
close all; clear;
% 设计数字低通滤波器
[w_p, w_s, alpha_p, alpha_s] = deal(0.2 * pi, 0.35* pi, 1, 20);
f_s = 100;
[omega_p, omega_s] = deal(f_s * w_p, f_s * w_s);

% 模拟低通滤波器
[butt_N, omega_c] = buttord(omega_p, omega_s, alpha_p, alpha_s, 's');
[B_slp, A_slp] = butter(butt_N, omega_c, 's');

% 模拟滤波器转化为数字滤波器
[B_zlp, A_zlp] = impinvar(B_slp, A_slp, f_s);
[H_lp_w, w] = freqz(B_zlp, A_zlp, 1000);

plot(w / pi, abs(H_lp_w));
xlabel('\pi');
```

采样率是提前自己设置的，有了采样率才能将数字指标转化为模拟指标。

~~不知道为什么高通模拟不能转化为数字滤波器。~~因为你用脉冲响应法，会频谱混叠！

**双线性变换法**
双线性变换法的原理可简单叙述为在频率压缩的基础上再进行脉冲响应不变法（也就是时域采样），通过下式进行频率的非线性压缩
$$
\Omega = \frac{2}{T_s}\tan(\frac{T_s}{2} \Omega_1)
$$
将原来范围为$(-\infty, \infty)$压缩至$(-\pi / T_s, \pi / T_s)$，其特点是离原点近时，线性度高，离原点远时，非线性度高。采样率$f_s$越大，则压缩后的范围越宽，也意味着线性度高的范围更宽，所以理论上采样率越大越好。

其他推导请看书。

matlab中使用`bilinear`函数来实现双线性变换法
```matlab
[B_z, A_z] = bilinear(B_s, A_s, f_s);
```

以例子实现了数字带阻滤波器的设计
```matlab
close all; clear;
% 用双线性变换法实现数字带阻滤波器的设计
f_s = 44100;
T_s = 1 / f_s;
% 预设模拟滤波器指标
[omega_pl, omega_sl, omega_su, omega_pu] = deal(2 * pi * 3500, ...
                                                2 * pi * 4500, ...
                                                2 * pi * 7500, ...
                                                2 * pi * 9643);
[w_pl, w_sl, w_su, w_pu] = deal(omega_pl / f_s, ...
                                omega_sl / f_s, ...
                                omega_su / f_s, ...
                                omega_pu / f_s);
% 预畸变矫正后的指标
fun_compress = @(w) (2 / T_s) * tan(w / 2);
[omega_pl, omega_sl, omega_su, omega_pu] = deal(fun_compress(w_pl), ...
                                                fun_compress(w_sl), ...
                                                fun_compress(w_su), ...
                                                fun_compress(w_pu));
[omega_pl, omega_sl, omega_su, omega_pu];
[alpha_p, alpha_s] = deal(1, 40);
B_w = omega_su - omega_sl;
omega_0 = sqrt(omega_sl * omega_su);

% 模拟低通滤波器
lambda_s = 1;
lambda_p = lambda_s * omega_pl * B_w / (omega_0^2 - omega_pl^2);
[butt_N, lambda_c] = buttord(lambda_p, lambda_s, alpha_p, alpha_s, 's');
[B_lp, A_lp] = butter(butt_N, lambda_c, 's');

[B_bs, A_bs] = lp2bs(B_lp, A_lp, omega_0, B_w);

[B_bs_z, A_bs_z] = bilinear(B_bs, A_bs, f_s);
[H_bs_w, f] = freqz(B_bs_z, A_bs_z, 1000, f_s);

plot(f / 1000, abs(H_bs_w));
xlabel('kHz');
```



### 脉冲响应

*如果Z变换的分式的系数都是实数，那么其逆Z变换的序列一定也是是序列，可以用初值和终值定理来证明，同理如果拉普拉斯变换的分式的系数都是实数，那么其反拉普拉斯变换一定是实函数。*

**impz**
matlab可以用`impz`函数来生成系统函数$H_d(z)$对应时域的序列（因果）。

```matlab
[h, t] = impz(B_z, A_z, n, f_s);
```

该函数返回以前两个参数为分子分母系数的系统函数$H_d(z)$的时域序列，该序列长度为`n`，`f_s`参数是用来与`n`搭配生成模拟的时域位置向量`t = [0 * f_s, 1 * f_s, ..., (n - 1) * f_s]'`。

如果不填`n`，函数会自动帮你找到一个合适的长度。

如果不填`n`，但是想要显示实际时间轴，可以`n = []`。

**residue**

```matlab
[r, p, k] = residue(b, a);
```

对于一个系统函数$H_a(s)$，对其部分分式展开。
$$
\frac{b(s)}{a(s)}=\frac{b_{m} s^{m}+b_{m-1} s^{m-1}+\ldots+b_{1} s+b_{0}}{a_{n} s^{n}+a_{n-1} s^{n-1}+\ldots+a_{1} s+a_{0}}=\frac{r_{n}}{s-p_{n}}+\ldots+\frac{r_{2}}{s-p_{2}}+\frac{r_{1}}{s-p_{1}}+k(s)
$$

`residue` 的输入是由多项式 `b = [bm ... b1 b0]` 和 `a = [an ... a1 a0]` 的系数组成的向量。输出为留数 `r = [rn ... r2 r1]`、极点 `p = [pn ... p2 p1]` 和多项式 `k`。对于大多数教科书问题，`k` 为 `0` 或常量。

也可将部分分式合成
```matlab
[b, a] = residue(r, p, k);
```

**$H_a(s)$的时域信号（只含一级极点）**
$H_d(z)$有办法通过`impz`函数来求其的时域序列，那么$H_a(s)$有没有办法求他的时域函数呢，虽然matlab中没有专门的函数，但是也可以通过理论的推导来画出时域序列。

可以知道
$$
H_a(s) = \sum_{i = 1}^{n}\frac{r_i}{s - p_i}
$$
其中n为其一级极点（只含一级极点）的数量，那么可以轻松算出其反拉普拉斯变换
$$
h_a(t) = \sum_{i = 1}^{n}r_i \exp{(p_i t)}
$$
所以知道了极点$p_i$和留数$r_i$，就可以求出其时域序列，在matlab中的求解方法类似于DTFT的求法（几乎一摸一样），如下
```matlab
B = [0, 1];
A = [1, -2];
[r, p] = residue(B, A);
t = linspace(0,100,1000);
h = r' * exp(p .* t);

plot(t, real(h));
```

注意调用函数生成的向量都是列向量。

### 匿名函数

只会简单的使用
```matlab
fun_compress = @(w) (2 / T_s) * tan(w / 2);

omega = fun_compress(w);	% 通过句柄调用匿名函数
```

我将匿名函数的语法分为三部分，第一部分是赋值号左边，即函数句柄名，函数定义后可用函数句柄来调用函数，类似于C语言的函数指针；第二部分是输入变量，类似于verilog的端口列表，用来定义输入变量的；第三部分就是函数定义的具体内容。

匿名函数适用于一些简单的表达式，其优点是可以方便的使用之前存在的变量。

### FIR数字滤波器设计

**fir1**

```matlab
b = fir1(N - 1, w_c, 'type', windows(N));
```

返回值`b`若表示系统函数的分子，而`N - 1`则代表系统函数的阶数也同样表示时域长度为`N`的序列，两种表达等价。

`w_c`用来表示截止频率，`w_c`一般为通带边界频率和阻带截止频率的平均值。若`w_c`为二维向量，则生成为带通或带阻。

**窗函数法设计FIR数字滤波器**
以下为用窗函数法设计数字滤波器的步骤

- 根据阻带最大衰减$\alpha_s$来确定窗函数类型
- 根据窗函数的过渡带宽来求出最小的奇数$N$。
- 根据滤波器的指标求出$w_c$，其为通带频率和阻带频率的平均值，不同的滤波器类型对应不同的$w_c$，标量或者二维向量。
- 使用窗函数的子程序生成窗，如果窗有参数那就另外求。

**fir2**

```matlab
b = fir2(N - 1, f, A, windows(N));
```

以`plot(f, A)`所画出来的图形为理想幅度特性，使用频率采样法进行滤波器的设计。首先按照线性相位的要求进行采样$N$点，对这些点进行IDFT得到时域，再对时域进行`windows(N)`的加窗。



### 随机数

**randn**

```matlab
noise = randn(n);
noise = randn(sz1, sz2);
```

第一个返回n维方阵的标准高斯分布的随机数。

第二个返回$sz_1 \times sz_2$的矩阵，同样是标准高斯分布的随机数。

### 随机统计量

**mean & var**

```
E_A = mean(A);
var_A = var(A);
```

若A是观测值向量，则分别返回期望和方差。

可以选择添加`mean`的第二个参数，选择求均值的方向，如A为矩阵，而第二个参数为2，那么就会得到一个每行均值的向量。

若A是矩阵，则每一列代表一个随机变量，该列的元素则是该随机变量的观测值，所以上述函数返回一个行向量，其每一个元素代表对应随机变量的期望和方差。

**cov & corrcoef**

```matlab
cov_A = cov(A);
corrcoef_A = corrcoef(A);
```

若A是观测值向量，即其只代表一个随机变量的观测值，则上述函数分别返回该向量的方差（也就是自己和自己的协方差）和相关系数（协方差用两个标准差之积来归一化，固定为1）。

若A是矩阵，仍然将列看作随机变量，列的元素表示随机变量的观测值，返回随机变量个数维的方阵，其代表着协方差阵和相关系数阵（对角矩阵，协方差阵的对角均为方差，而相关系数阵的对角均为1）。

### 特征值分解

```matlab
[V, D] = eig(A);
```

V是归一化后的特征列向量组成的矩阵，D是对角线为特征值的矩阵。

此函数的特征值没有进行排列，所以还需以下操作进行排列

```matlab
[Vs, Ds] = eig(R_x);

[d, ind] = sort(diag(Ds), 'descend');
D = Ds(ind, ind);
V = Vs( : , ind);
```

先用sort函数获取降序排列的索引数组d，再通过d对Ds和Vs重新排列。

### PCA主成分分析

- 得到相关矩阵，使用`cov()`函数。
- 对相关矩阵进行特征值分解，得到最大特征值的特征向量。
- 对原来的数据进行特征前面得到的特征向量的方向进行投影。

```matlab
close all; clear;
% PCA分析

% 先创建样本
data_ori = randn(2, 100);
theta = pi / 4;

len_expand = [10, 0; 0, 7];
rot_45 = cos(theta) * eye(2) + sin(theta) * [0, -1; 1, 0];

data =  rot_45 * len_expand * data_ori;

% scatter(data(1, : ), data(2, : ), '.');
% 计算相关阵并特征值分解
R_x = cov(data');

[Vs, Ds] = eig(R_x);

[d, ind] = sort(diag(Ds), 'descend');
D = Ds(ind, ind);
V = Vs( : , ind);

% 投影
pri_dir = V( : , 1);
result = pri_dir * (pri_dir' * data);

hold on;
scatter(data(1, : ), data(2, : ), '.');
scatter(result(1, : ), result(2, : ), '.');
```

### 使用两个一维向量张成一个矩阵

常用两个一维向量张成一个矩阵，通常矩阵的特性是按某列或者某行的某一个参数不变，一般形式为行向量乘列向量的形式，列向量代表着生成矩阵列的变化，行向量代表着生成矩阵行的变化。

比如在参数的估计中（low rank matrix approximation），代价函数的形式为
$$
J(\tilde{x}) = (r - f(\tilde{x}))^{T}(r-f(\tilde{x}))= \sum_{n = 1}^{N}(r[n] - f_n(\tilde{x}))^2
$$
上式实际上就是观测值和估计值的Least Squares（最小二乘），即遍历所有被估计参数$\tilde{x}$，与生成观察值$r$相同方式生成某一特定被估参数$\tilde{x}$的序列$\tilde{r}$，通过比较实际的观测$r$和自己生成的模拟$\tilde{r}$的距离，来表示代价。

例如，被估计参数为常量，且只被噪声干扰
$$
r = A + e
$$
则其代价函数为，而$\tilde{A}$取遍了所有可能出现的值
$$
J(\tilde{A}) = \sum_{n = 1}^{N}(r - \tilde{A})^2
$$
在matlab中，可以用循环来计算一些列的根据$\tilde{A}$的变化而变化的$J(\tilde{A})$，但是更优雅的方式是使用矩阵运算。将行向量$r$以列的维度张成为矩阵，而行数就是$\tilde{A}$的长度，可以通过左乘1列向量来获得每行都相同矩阵。对于$\tilde{A}$，应该将每个确定的$\tilde{A}$为行，而变化的其他参数为列，这里没有变化的其他参数，所以统一为1行向量，最终的结果就是列向量的$\tilde{A}$乘行向量的1向量。

```matlab
temp = ones(length(A_tilde), 1) * y - A_tilde' * ones(1, length(y));
temp = temp .* temp;
J_A_tilde = mean(temp, 2);	% 按照行方向做均值
```

上面代码的第二第三行做了均方操作。

再来一个例子，估计
$$
r = \cos(wn) + e
$$
中的$w$，程序如下
```matlab
temp = ones(length(w_tilde), 1) * y - cos(w_tilde' * n);
temp = temp .* temp;
J_w_tilde = mean(temp, 2);
```

第一行的左边部分与之前一样，右边部分略微不同，不同的地方在于此次的每一行$\tilde{w}$不变化的迭代，有个变化的参数$n$，不过按照之前的方法，描述列变化的向量的放左边列向量，描述行变化的向量放右边行向量。

### 模拟信号的FT到数字信号的DFT到底经过了什么

以一个余弦波为例，其信号具有以下形式
$$
x(t) = \cos(\Omega_0 t)
$$
其在频域上的表现（直接做连续傅里叶变换）就是在频点$\Omega_0$上有一个冲击$\delta(\Omega - \Omega_0)$。

我们知道，离散信号的DTFT等价于对模拟信号进行冲激采样，并对其做FT。也就是说，从某种程度上，连续信号的FT和离散信号的DTFT在满足采样定理且带宽有限的情况下，二者是等价的，其相对应积分和求和的上下限都是无穷，例如上述的余弦波在取样之后的DTFT是在对应频点上的无限堆叠。

但是理想是丰满，现实总时骨感，在数字信号处理中，是无法进行无限点的采样的，故只能取适当的点数。这些适当点数的离散信号和上述上下限都为无限点数的信号在时域上的数学描述就是加窗，而时域上的加窗在频域上表现为卷积。若信号取上述的余弦信号并且窗函数取矩形窗，那么就是在频域中的冲激函数和Sa函数的卷积，最终的结果为Sa函数。也就是说，在实际的数字信号处理中，从模拟信号采集的任何有限离散信号已经默认在时域中加上了窗，在频域中卷上了Sa函数。此时**波形分辨率**的概念孕育而生，因为数字信号处理所处理的信号可以认为其永远加上了窗，而频域中Sa的宽度取决于窗的长度，所以既有可能出现两个信号的频率过于接近而导致由于窗函数长度不够，Sa函数过宽而混叠在一起的情况（实际上无论两个频率相隔多远，都会出现混叠的情况，只不过频率挨得近混叠得更严重从而分辨不出两个频率）。定义波形分辨率为$\Delta f = 1 / T_{span}$，其中$T_{span} = N \cdot f_s$是被采样信号的时域的长度。根据前述的概念，波形分辨率正使描述窗函数的长度，也即频域中Sa宽度，只有Sa函数的宽度足够小，也即时域$T_{span}$足够长，也即采样点数$N$足够多（$f_s$保持不变的情况下），才能在频域中分辨这两个信号。

那么补零又是个什么操作呢，我们知道DFT实际上是对DTFT的等间隔采样，而由于DFT自带的有限长度特性，其默认所有被采集的信号已经进行了一波频谱泄露了，在已经频谱泄露的基础上做DTFT对再对其等间隔采样，最终得到的就是DFT。补零的操作实际上是减小已经频谱泄露完之后DTFT的采样间隔，所以并没有任何补充信息的作用。但是补零仍然是有作用的，当两个频率都能卡在Sa函数的零点处时，会出现好像没有发生频谱泄露的情况，此时对于我们的谱分析是更有利的。

所以综上所述，为了区分两个很近频率的信号，一是要让Sa函数的宽度足够小，即通过增加采样点数的方式，而是通过补零，人为的制造好像没有发生频谱泄露的效果，从而更有利于进行分析。
