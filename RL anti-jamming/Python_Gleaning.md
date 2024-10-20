### 文件路径

**绝对路径**
文件路径分为绝对路径和相对路径，绝对路径更好理解，所以先讲绝对路径。绝对路径在不同的操作系统中有所不同，对于windows来说，一个文件路径示例如下

```
D:\files\data\ndvi.tif
```

windows中，绝对路径必定以盘符开头`D:`，用反斜杠（非除法）`\`分隔。

linux中，以斜杠（除法）`/`为根源，可以理解为所有的文件都在一个盘符下。

**相对路径**

![image-20240410143543380](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240410143543380.png)

`.\`表示当前盘符（通常默认不写），`..\`表示上一级目录。

**python的路径表示**
python可以同时识别windows和linux下的路径，并且可以相互转换（核心就是windows下的`\`和linux下的`/`是一样的）。唯一需要注意的是，windows的分隔符`\`在python中还有转译字符的意义，所以在python中使用windows的路径格式，只能以两种方式进行

```python
path = r"D:\files\data\ndvi.tif"  # 绝对路径
path = r".\data\ndvi.tif"  # 相对路径，当前位置位于files文件内
path = r"data\ndvi.tif"  # 与上一行等价，通常将.\省略

path = "D:/files/data/ndvi.tif"  # 使用linux的分隔符/，python也认

path = "D:\\files\\data\\ndvi.tif"  # 去除转译
```

**常用的文件夹操作**

```python
import os

path = r".\logs"
if not os.path.isdir(path):
    os.mkdir(path)  
```

上述这个代码很常见，但是只适合在已存在的目录中创建一个文件夹的目的，其没有同时创建深层文件夹的能力。比如上式中，首先会检查`logs`这个文件夹在当前路径是否存在，同时`mkdir`只能创建一级目录，且要求前面的路径必须存在，所以上述这个代码的使用条件就是在已存在的路径中创建一个文件夹。

```python
path = os.path.join("parent_directory", "subdirectory", "file.txt")
print(path)
```

`os.path.join()`函数用来按照参数列表的顺序创建一个路径，具体是相对路径还是绝对路径，取决于传入的第一个参数是相对还是绝对。

### matplotlib使用

使用matplotlib库画图首先需要引入库，官方推荐的调用库的方式为
```python
import matplotlib.pyplot as plt
```

其次，具体使用`plt`画图的方式可分为两种：一类为matlab式，即类似于matlab中面向过程的语法；另一类为面向对象式，本文重点介绍后者。

对于一个画图对象来说，有两个实例需要关注，figure类和axes类，figure代表一个窗口（更形象一点就是一个画板），axes（注意不要和坐标轴axis混为一谈）代表具体的图画（画板内的画布），决定着你要具体画什么东西，使用以下方式来创建这两个实例
```python
fig, ax = plt.subplots(1, 1, figsize=(10, 5), layout='constrained')
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
```

调用上行代码默认只创建一个fig，并由具体的参数来决定ax的数量，例如该代码表示创建一个ax，所以返回的ax变量直接为axes实例。但是对于注释的那一行来说，其创建了一行两列的两个axes对象，所以该方法返回的第二个代表axes实例的参数为一个列表，可以使用元组赋值的方式拆开列表内的具体axes实例。

#### plot

创建好了一个fig和若干个axes对象之后，就可以在每个axes的实例内画图了。首先最重要的就是直接画图这个功能
```python
x = np.linspace(0, 2, 100)

fig, ax = plt.subplots(figsize=(10, 10 * 0.618))

ax.plot(x, x, label='linear')
ax.plot(x, x ** 2, label='quadratic')
ax.plot(x, x ** 3, label='cubic')
ax.legend(loc='best')

ax.set_xlabel('x label')
ax.set_ylabel('y label')

ax.set_title('simple plot')

plt.show()
```

如上代码块所示，我们直接使用axes类型的实例`ax`上直接调用`plot()`方法，并且传入了一个`label`参数。由此，我们完成了对于主体数据绘制的目的。
其次，我们需要对该数据绘图有具体的标注，分别使用`set_xlabel()`、`set_ylabel()`、`set_title()`等方法来进行，因为这是对于axes类型实例的操作，所以上述方法都是依靠于`ax`变量进行调用；同时，由于同时对一个axes类型的实例画了三条曲线，并且传入了对于的图注，所以我们需要使用`ax.lengend()`方法来展示图注。
最后再使用`plt.show()`进行展示。

除了xy轴以及标题等标识之外，还有`ax.text()`方法用来在相应的坐标位置放置一个说明性的文字；
使用`ax.grid(True)`来打开画布中的方格。

使用`ax.annotate()`用来标注一个箭头，并附上说明文字
```python
x = np.linspace(-5, 5, 100)

mu, sigma = 0, 1
y = np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))


fig, ax = plt.subplots(figsize=(10, 10 * 0.618))
ax.plot(x, y)
ax.set_title(r'Gaussian distribution with $\mu=0$ and $\sigma^2=1$')
ax.set_xlabel('x')
ax.set_ylabel('y')

ax.text(-4, 0.3, r'$f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$', fontsize=15)

max_point = np.array( [ 0, 1 / (sigma * np.sqrt(2 * np.pi)) ] )
ax.annotate('local_max', xy=max_point, xytext=max_point + (0.8, 0.05), 
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=15)
ax.axis([-5, 5, 0, 0.5])

ax.grid(True)
```

得到结果

![image-20240808122330845](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240808122330845.png)

使用`ax.set_xticks()`用来自定义横轴要显示的数据，`ax.set_yticks()`同理。

使用`ax.set_yscale('log')`用来转换至对数坐标轴。

#### mesh

`np.meshgrid()`通常用来创建一个二维坐标，`x`代表二维坐标系中的x轴的刻度，`y`代表二维坐标系中的y轴刻度，由于二维坐标系中通常x轴位于底下向右增大，而y轴位于右边向上增大，所以与正常的矩阵索引方式有所不同（x轴位于右边向下增大，y轴位于上面向右增大），所以`np.meshgrid()`返回的`x`和`y`都应该以二维坐标系的基准来排列。

```python
x, y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
z = np.cos(x**2 + y**2)

fig, ax = plt.subplots(figsize=(8, 6))
pc = ax.pcolormesh(x, y, z)

fig.colorbar(pc, ax=ax)


plt.show()
```

在创建了一个mesh图后，使用`fig.colorbar()`可展示colorbar，需要参数mesh图的实例（`ax.pcolormesh()`返回），以及对应的axes实例。