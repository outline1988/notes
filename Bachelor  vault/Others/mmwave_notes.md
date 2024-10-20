**毫米波**
30-300GHz，对应为1-10mm的波长

**毫米波雷达的调制方式**
LFMSK（Linear Frequency Modulated Shift Keying）这里不讨论

FMCW（Frequency Modulated Continuous Wave）

**波形图**

下为带宽（频率）-时间关系图（波形特点）

![image-20221005152218773](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221005152218773.png)

![image-20221005152257589](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221005152257589.png)

综合各种因素，最终选择FMCW（fast），其中一个周期称为一个chirp，连续的几个chirps形成一个frame（帧）。

Frequency Modulated Continuous Wave

![image-20221006133140318](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221006133140318.png)

一帧有多个chirp，占空比一般小于百分之五十

**运行demo**
awr1843的sdk中包含了demo文件（以.bin为后缀结尾），详情参考
https://e2echina.ti.com/blogs_/b/the_process/posts/awr1843boost-mmw-demo

注意当前最新（2022.10.6）的版本为3.5，而上述博客所示mmWave Demo Visualizer网址版本为3.4，所以需要打开3.5版本的Visualizer网址，方法就是将网址后面的3.4改为3.5。

- bin后缀的文件为你需要烧录的文件
- 烧录时，你需要将板子的SOP调成烧录模式101，使用uniflash软件进行烧录
- 将板子的SOP调成001，即为命令模式，借助网页版本的mmWave Demo Visualizer来进行配置参数和命令的发送和接收，并数据可视化
  

**FMCW测距**
对于每一个其距离、距离分辨率和最大距离公式如下
![image-20221006150025296](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221006150025296.png)

对于距离分辨率，即两个物体的径向距离挨得多近时雷达分辨不出来，其来源于观察时间的长短（时域的持续时间TSP），在频谱上的频域分辨率就是观察时间的倒数，由此可以推得距离分辨率。其取决于带宽，带宽越大，一可以理解为观察时间变长了。

 对于最大距离，其来源于采样率的限制，由于距离的信息来源于中频信号IF的频率，所以当距离很远时，IF很大，然而却没有足够的采样频率来匹配，所以造成了最大的检测距离（雷达原理的最大距离是由于功率的衰减而造成的）。其取决于频率的变化速率，频率变化得越快，表示相同距离所产生的中频越大，所以需要越高的采样率。

**从FFT序列下标还原距离**

![image-20221006152358820](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221006152358820.png)

![image-20221006152326668](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20221006152326668.png)

即计算出相隔点的频率step，其与分辨率有关（原因不详），如果采样点数和FFT的点数相同，则分辨率=step，如果不相同，则参考上述公式。上述公式的带宽B是由（采样点数/采样率）来确定的。

实际的距离就是采样下标×距离精度（与距离分辨率有关）

**编译demo**
首先配置环境，打开在`\mmwave_sdk_03_05_00_04\packages\scripts\windows`中的`setenv.bat`文件。需要注意两个地方，以下

```c
set MMWAVE_SDK_DEVICE=awr18xx	// 1. 改成对应的设备
set MMWAVE_SDK_TOOLS_INSTALL_PATH=E:/Project/mmwave	// 2. 改成实际sdk安装的路径
```

### python

**python pickle库**
可以将python的对象转换成字节流存在文件中，下次使用的时候在将文件加载即可。

```python
import pickle
pickle.dump(obj, file[, protocol])
new_obj = pickle.load(file)		# new_obj和obj完全相同
```

protocol为序列化使用的协议版本

- 0：ASCII协议，所序列化的对象使用可打印的ASCII码表示；
- 1：老式的二进制协议；
- 2：2.3版本引入的新二进制协议，较以前的更高效。其中协议0和1兼容老版本的python

protocol默认值为`pickle.DEFAULT_PROTOCOL`，在python3中为3。

pickle最大的用处就是将python的数据类型（包括内置和自定义）转换成字节流从而存在本地文件中，但是其只能用于pickle内部，而不具有通用性，其完全就是当保存数据而来的，相当于是一种只能由python使用的协议。

**python 内置struct库**
strucu库作为python内置的标准库之一，可以将python**内置的类型**（如整型、元组等等）转换为字节流，该字节流就是计算机底层所表示的字节流，与pickle不同的是，其更为通用，可用于数据传输。

```python
import struct
byte_code = struct.pack(format, data1, data2 ...)
new_data = struct.unpack(format, byte_code)
```

`struct.pack()`将后面的数据打包成元组后翻译为字节流。

[【pickle库详解】](https://www.cnblogs.com/baby-lily/p/10990026.html)

**set**

创建set

```python
set1 = {1, 2, 3, 3}
set2 = set([1, 2, 3, 3])
```

使用set
```python
>>> set1, set2
({1, 2, 3}, {1, 2, 3})
>>> set1 & set2
{1, 2, 3}
>>> set1 | set2
{1, 2, 3}
```

对于减法，会消除第一个操作数中与第二个操作数相同的元素

```python
>>> set2 = {1, 2, 3, 4}
>>> set1 - set2
set()
```

**字节流和字符流**
字节流就是全部都用8位的ascii组合而成的类型

字符流就是用16-32位的unicode（如utf-8等）编码组合而成的类型，就是常用的字符串。

在python中分别为以下类型
```python
>>> byte1 = b'0x01'
>>> char1 = 'a'
>>> string1 = 'abc'
>>> print(type(byte1), type(char1), type(string1))
<class 'bytes'> <class 'str'> <class 'str'>
```



$$
\Omega_{MN} = \frac{\mu_{MN} E}{k}
$$

$$
\begin{array}{l}
x_{1}(n)=R_{4}(n) \\
x_{2}(n)=\left\{\begin{array}{ll}
n+1, & 0 \leq n \leq 3 \\
8-n & 4 \leq n \leq 7 \\
0 & \text { other } 
\end{array}\right. \\
x_{3}(n)=\left\{\begin{array}{ll}
4-n & 0 \leq n \leq 3 \\
n-3 & 4 \leq n \leq 7 \\
0 & \text { other } 
\end{array}\right. \\
x_{4}(n)=\cos \frac{\pi}{4} n \\
x_{5}(n)=\sin \frac{\pi}{8} n \\
x_{6}(n)=\cos 8 \pi n+\cos 16 \pi n+\cos 20 \pi n
\end{array}
$$


$$
H_{d}\left(e^{j \omega}\right)=\left\{\begin{array}{ll}
e^{-j \omega_2 a} - e^{-j \omega_1 a} & \omega_1 < \omega < \omega_2 \\
0 & others
\end{array}\right.
$$

$$
\begin{aligned}
h_{d}(n) & =\frac{1}{2 \pi} \int_{-\pi}^{\pi} H_{d}\left(e^{j \omega}\right) e^{j \omega n} \\
& =\frac{\sin \omega_{2}(n-a)}{\pi(n-a)} - \frac{\sin \omega_{1}(n-a)}{\pi(n-a)}
\end{aligned}
$$




$$
H_d(e^{jw}) = 
$$




















