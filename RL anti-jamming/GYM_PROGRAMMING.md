## PY编程相关

### RL环境配置

配环境的时候没写，现在懒得详细说了，总的来说就是

- 使用anaconda创建虚拟环境，这个虚拟环境实际上就是一个python的环境，anaconda允许你在一台电脑上有多个python环境，用于强化学习的环境我命名为`rl_env`；
- 接着需要安装必要的库，主要是`gym`和`pytorch`，都在对应的虚拟环境下安装，具体细节上网搜；
- 在使用pycharm编程时要要选择对应的虚拟环境；
- vscode主要用来运行`.ipynb`文件。

### gym环境

**MultiDiscrete类**
在 Gym 中，`gym.spaces.MultiDiscrete` 是一个用于表示多个离散空间的类。

`MultiDiscrete` 类表示一个多维离散空间，其中每个维度都有不同的离散取值范围。它常用于表示一组离散的动作选择，每个维度可以有不同的离散取值个数。

具体来说，`MultiDiscrete` 类的实例具有以下属性和方法：

属性：

- `nvec`：一个整数列表，表示每个维度的离散取值个数。

下面是一个示例，展示如何使用 `MultiDiscrete` 类：

```python
import gym
from gym.spaces import MultiDiscrete

# 创建一个 MultiDiscrete 对象
action_space = MultiDiscrete([3, 4, 2])

# 检查动作空间的属性
print(action_space.nvec)  # 输出：[3, 4, 2]

# 从动作空间中采样一个动作
action = action_space.sample()  # 返回的是一个np的向量，表示具体的某个动作
print(action)  # 输出：[0 1 0]

# 检查动作是否在动作空间中
print(action_space.contains(action))  # 输出：True
```

本质上这个类就是在np数组的基础上再封装了一个类，其支持`sample()`和`contains()`方法。传入类的参数`nvec`可以理解为再底层创建了一个大小为`nvec`的np数组。

2

`np.ravel_multi_index` 是 NumPy 库中的一个函数，用于将多维索引转换为一维索引。

在 NumPy 中，数组的元素可以通过索引来访问和操作。对于多维数组，每个元素都有一个对应的多维索引，表示其在数组中的位置。`np.ravel_multi_index` 函数允许将这样的多维索引转换为一维索引，以便在一维数组中定位对应的元素。

该函数的语法如下：

```python
numpy.ravel_multi_index(multi_index, dims, mode='raise', order='C')
```

参数说明：

- `multi_index`：表示多维索引的数组或元组。
- `dims`：表示数组维度的整数值或元组。
- `mode`：指定越界索引的处理模式。可选值为 'raise'、'wrap' 或 'clip'。
- `order`：指定多维索引的顺序。可选值为 'C'（按行展开）或 'F'（按列展开）。

下面是一个示例，展示如何使用 `np.ravel_multi_index` 函数：

```python
import numpy as np

# 创建一个二维数组
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 将多维索引 (1, 2) 转换为一维索引
index = np.ravel_multi_index((1, 2), arr.shape)

print(index)
# 输出: 5
```

在这个示例中，我们创建了一个二维数组 `arr`，然后使用 `np.ravel_multi_index` 函数将多维索引 `(1, 2)` 转换为一维索引 `index`。转换后，我们可以使用一维索引来访问数组中对应的元素。

总结：`np.ravel_multi_index` 是 NumPy 库中的一个函数，用于将多维索引转换为一维索引。它对于定位多维数组中的元素非常有用。通过指定多维索引和数组维度，函数可以计算出对应的一维索引值。

特殊技巧，计算空间大小
```python
n_action_ridx = np.ravel_multi_index(env.action_space.nvec - 1, env.action_space.nvec) + 1
n_state_ridx = np.ravel_multi_index(env.observation_space.nvec - 1, env.observation_space.nvec) + 1
```

第一个参量表示多维下标的最后一个值，所以要在numpy数组的基础上减一，第二个参数表示多维空间的形状，最后加一是由于下标是从0开始数的。

3

`np.unravel_index` 是 NumPy 库中的一个函数，用于将一维索引转换为多维索引。

在某些情况下，我们可能需要将一维数组的索引转换回原始多维数组的索引。`np.unravel_index` 函数可以帮助我们实现这个转换。它接受一个一维索引值和表示多维数组形状的参数，并返回对应的多维索引。

该函数的语法如下：

```python
numpy.unravel_index(indices, shape, order='C')
```

参数说明：

- `indices`：要转换的一维索引值或一维数组。
- `shape`：表示多维数组形状的整数元组。
- `order`：指定多维索引的顺序。可选值为 'C'（按行展开）或 'F'（按列展开）。

下面是一个示例，展示了 `np.unravel_index` 函数的使用方式：



```python
import numpy as np

# 定义一维索引值
idx = 5

# 定义多维数组的形状
shape = (3, 4)

# 将一维索引转换为多维索引
result = np.unravel_index(idx, shape)

print(result)
# 输出: (1, 1)
```

在这个示例中，我们将一维索引值 `idx` 转换为多维索引，使用的多维数组形状为 `(3, 4)`。转换后，我们得到的结果是 `(1, 1)`，表示在形状为 `(3, 4)` 的多维数组中，索引值 `5` 对应的多维索引为 `(1, 1)`。

通常情况下，`np.unravel_index` 函数在处理强化学习等领域中的动作选择时非常有用，可以将一维动作索引转换为多维动作索引，以便在多维动作空间中进行操作。

总结：`np.unravel_index` 是 NumPy 库中的一个函数，用于将一维索引转换为多维索引。它可以在处理多维数组时对索引进行转换，通过提供一维索引值和多维数组形状，返回对应的多维索引。

其最后返回的时`tuple`类型，故最终还需要使用`np.array()`来转换成np类型。

3

`np.prod` 是 NumPy 库中的一个函数，用于计算数组中所有元素的乘积。

该函数的语法如下：



```python
numpy.prod(a, axis=None, dtype=None, keepdims=<no value>, initial=<no value>)
```

参数说明：

- `a`：要计算乘积的数组。
- `axis`：指定沿着哪个轴计算乘积。如果未提供该参数，则会将所有元素相乘得到一个标量值。
- `dtype`：指定输出的数据类型。
- `keepdims`：指定是否保持输出数组的维度和输入数组相同。默认为 False。
- `initial`：指定初始乘积的值。默认为 None。

下面是几个示例，展示了 `np.prod` 函数的使用方式：

```python
import numpy as np

# 计算一维数组的乘积
arr1 = np.array([1, 2, 3, 4])
result1 = np.prod(arr1)
print(result1)  # 输出: 24

# 计算二维数组的乘积
arr2 = np.array([[1, 2], [3, 4]])
result2 = np.prod(arr2)
print(result2)  # 输出: 24

# 指定轴计算乘积
arr3 = np.array([[1, 2], [3, 4]])
result3 = np.prod(arr3, axis=0)
print(result3)  # 输出: [3 8]

# 保持维度和输入数组相同
arr4 = np.array([[1, 2], [3, 4]])
result4 = np.prod(arr4, keepdims=True)
print(result4)  # 输出: [[24]]

# 指定初始乘积的值
arr5 = np.array([1, 2, 3, 4])
result5 = np.prod(arr5, initial=10)
print(result5)  # 输出: 240
```

在这些示例中，我们使用 `np.prod` 函数计算了不同维度的数组的乘积。可以通过指定轴来计算特定维度上的乘积，也可以通过设置 `keepdims` 参数来保持输出数组的维度。还可以通过设置 `initial` 参数来指定初始乘积的值。

总结：`np.prod` 是 NumPy 库中用于计算数组中所有元素乘积的函数。它可以对多维数组进行计算，支持指定轴、保持维度和设置初始乘积值等功能。

4

`np.random.choice` 是 NumPy 库中的一个函数，用于从给定的一维数组或整数范围中进行随机抽样。

该函数的语法如下：

```python
numpy.random.choice(a, size=None, replace=True, p=None)
```

参数说明：

- `a`：用于抽样的一维数组或整数范围。如果是一维数组，可以是列表、元组或 NumPy 数组。
- `size`：指定抽样的大小。可以是整数值，表示要抽样的元素个数；也可以是元组或整数列表，表示要抽样的多维数组形状。
- `replace`：指定是否允许抽样时有重复元素。如果为 True，表示允许重复抽样；如果为 False，则不允许重复抽样。
- `p`：用于指定抽样概率的一维数组。如果提供了 `p`，则抽样时会按照指定的概率进行抽取。默认情况下，每个元素被选择的概率相等。

下面是几个示例，展示了 `np.random.choice()` 函数的使用方式：

```python
import numpy as np

# 从一维数组中随机抽样
arr1 = np.array([1, 2, 3, 4, 5])
result1 = np.random.choice(arr1, size=3)
print(result1)  # 输出: [4 2 4] （可能是随机的结果）

# 从整数范围中随机抽样
result2 = np.random.choice(10, size=5)
print(result2)  # 输出: [2 7 9 1 3] （可能是随机的结果）

# 允许重复抽样
result3 = np.random.choice(arr1, size=5, replace=True)
print(result3)  # 输出: [4 1 5 4 2] （可能是随机的结果）

# 按照指定概率抽样
result4 = np.random.choice(arr1, size=5, p=[0.1, 0.2, 0.3, 0.2, 0.2])
print(result4)  # 输出: [3 2 1 2 3] （可能是随机的结果）
```

在这些示例中，我们使用 `np.random.choice` 函数进行了不同类型的随机抽样。可以从一维数组或整数范围中进行抽样，并可以指定抽样的大小、是否允许重复抽样以及抽样概率。

总结：`np.random.choice` 是 NumPy 库中用于从一维数组或整数范围中进行随机抽样的函数。它可以根据指定的参数进行抽样，并返回抽样结果。可以指定抽样大小、是否允许重复抽样，以及按照指定概率进行抽样。

生成episode时可用该函数进行按照策略的概率选择动作
```python
action_ridx = np.random.choice(choices_ridxs, p=policy[state_ridx])  # 从动作空间下标列表choices_ridxs中随机选择一个动作，按照policy的概率
```

其中`policy[state_ridx])`的大小和`choices_ridxs`等大，且前者要满足归一化要求。

5

`np.cumsum` 是 NumPy 库中的一个函数，用于计算数组的累积和。

该函数的语法如下：

```python
numpy.cumsum(a, axis=None, dtype=None)
```

参数说明：

- `a`：要计算累积和的数组。
- `axis`：指定沿着哪个轴计算累积和。如果未提供该参数，则会将所有元素相加得到一个一维数组。
- `dtype`：指定输出的数据类型。

在给定的示例中，`rewards` 是一个一维数组。通过 `rewards[::-1]`，我们将 `rewards` 数组进行反转，然后使用 `np.cumsum` 计算反转后数组的累积和，最后再将结果反转回来。

具体示例如下：

```python
import numpy as np

rewards = np.array([1, 2, 3, 4, 5])

# 将 rewards 数组反转，并计算反转后数组的累积和
cumulative_sum = np.cumsum(rewards[::-1])[::-1]

print(cumulative_sum)
# 输出: [15 14 12 9 5]
```

在这个示例中，我们首先将 `rewards` 数组反转得到 `[5, 4, 3, 2, 1]`，然后使用 `np.cumsum` 计算反转后数组的累积和，得到 `[5, 9, 12, 14, 15]`。最后，我们再将结果数组反转回来，得到 `[15, 14, 12, 9, 5]`，即为原始 `rewards` 数组的逆向累积和。

这种操作常用于计算序列或时间序列中的累积和，特别是在强化学习等领域中，用于计算累积奖励或返回。

总结：`np.cumsum` 是 NumPy 库中用于计算数组累积和的函数。通过指定数组和轴，可以计算沿着指定轴的累积和。在给定的示例中，通过反转数组、计算累积和，然后再次反转，实现了原始数组的逆向累积和。

6

`zip(state_ridxs, action_ridxs, returns)` 是用于将多个可迭代对象进行逐个配对的 Python 内置函数。

该函数接受多个可迭代对象作为参数，并返回一个迭代器，每次迭代返回一个元组，其中包含来自各个可迭代对象的元素。

在给定的示例中，`state_ridxs`、`action_ridxs` 和 `returns` 都是可迭代对象，可能是列表、元组或其他类似的对象。通过 `zip` 函数，我们将这三个可迭代对象逐个配对，并创建一个包含元组的迭代器。

具体示例如下：

```python
state_ridxs = [1, 2, 3]
action_ridxs = [4, 5, 6]
returns = [0.5, 1.0, 1.5]

result = zip(state_ridxs, action_ridxs, returns)  # 同时多个可迭代进行时，迭代这个

for item in result:
    print(item)
# 输出:
# (1, 4, 0.5)
# (2, 5, 1.0)
# (3, 6, 1.5)
```

在这个示例中，我们有三个可迭代对象 `state_ridxs`、`action_ridxs` 和 `returns`，它们分别包含了索引值、动作索引和返回值。通过 `zip` 函数，我们将它们进行逐个配对，并得到一个迭代器 `result`。在每次迭代中，我们可以访问一个元组，其中包含来自各个可迭代对象的元素。在这个示例中，我们使用循环遍历迭代器，并打印每个元组。

这种使用方式常见于需要同时迭代多个可迭代对象，并进行配对操作的场景。在机器学习、数据分析和其他数据处理任务中经常用到。

总结：`zip` 是 Python 内置函数，用于将多个可迭代对象进行逐个配对。通过将多个可迭代对象作为参数传递给 `zip` 函数，可以创建一个迭代器，每次迭代返回一个包含来自各个可迭代对象的元素的元组。在给定的示例中，`zip(state_ridxs, action_ridxs, returns)` 将 `state_ridxs`、`action_ridxs` 和 `returns` 进行逐个配对，并返回一个迭代器，其中每个元组包含来自各个可迭代对象的元素。

7

`np.sum(policy, axis=1)` 是 NumPy 库中的一个函数，用于计算数组沿指定轴的元素和。

该函数的语法如下：

```
numpy.sum(a, axis=None, dtype=None, keepdims=False)
```

参数说明：

- `a`：要计算元素和的数组。
- `axis`：指定沿着哪个轴计算元素和。如果未提供该参数，则会将所有元素相加得到一个标量。
- `dtype`：指定输出的数据类型。
- `keepdims`：如果为 True，则保持输出的维度和输入数组的维度相同。

在给定的示例中，`policy` 是一个二维数组。通过 `np.sum(policy, axis=1)`，我们计算了 `policy` 数组沿第二个轴（轴索引为 1）的元素和。

具体示例如下：

```python
import numpy as np

policy = np.array([[0.2, 0.3, 0.5],
                   [0.1, 0.4, 0.5]])

# 计算 policy 数组沿第二个轴的元素和
sum_axis_1 = np.sum(policy, axis=1)

print(sum_axis_1)
# 输出: [1. 1.]
```

在这个示例中，我们有一个二维数组 `policy`，其中包含两个子数组。通过指定 `axis=1`，我们计算了 `policy` 数组沿第二个轴（即列）的元素和。在每个子数组上，相应的元素被相加得到一个标量值。计算结果为 `[1.0, 1.0]`，表示每个子数组的元素和。

这种使用方式常见于需要沿指定轴对数组进行求和的场景，特别是在统计、数据分析和机器学习中。

总结：`np.sum(policy, axis=1)` 是 NumPy 库中用于计算数组沿指定轴的元素和的函数。通过指定数组和轴，可以计算沿指定轴的元素和。在给定的示例中，通过 `np.sum(policy, axis=1)` 计算了 `policy` 数组沿第二个轴的元素和，返回一个包含每个子数组元素和的一维数组。

这里主要是要记住`axis=0`或`axis=1`时怎样进行的。对于矩阵来说，前者代表上下，后者代表左右。

8

`np.allclose` 是 NumPy 库中的一个函数，用于比较两个数组是否在指定的容差范围内相等。

该函数的语法如下：

```
numpy.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)
```

参数说明：

- `a`：第一个数组。
- `b`：第二个数组。
- `rtol`：相对容差（可选，默认值为 1e-05）。两个元素之间的差异相对于较大的绝对值的容忍度。
- `atol`：绝对容差（可选，默认值为 1e-08）。两个元素之间的最大差异值的容忍度。
- `equal_nan`：是否将 NaN 视为相等（可选，默认为 False）。如果为 True，则 NaN 将被视为相等。

`np.allclose` 返回一个布尔值，指示两个数组是否在指定的容差范围内相等。如果两个数组的对应元素之间的差异小于等于容差范围（即 `|a - b| <= atol + rtol * |b|`），则返回 True，否则返回 False。

具体示例如下：

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.array([1.001, 2.002, 3.003])

# 检查 a 和 b 是否在容差范围内相等
result = np.allclose(a, b, rtol=1e-02, atol=1e-01)

print(result)
# 输出: True
```

在这个示例中，我们有两个数组 `a` 和 `b`。通过调用 `np.allclose(a, b, rtol=1e-02, atol=1e-01)`，我们比较了这两个数组是否在容差范围内相等。由于 `b` 数组中的元素与 `a` 数组中的对应元素之间的差异小于等于容差范围（容忍度为 `rtol=1e-02` 和 `atol=1e-01`），所以返回结果为 True。

这种函数常用于比较两个数组在数值上是否相近，特别是在涉及浮点数运算的情况下，由于浮点数精度的限制，直接进行相等比较可能不准确，因此可以使用 `np.allclose` 函数进行近似相等的比较。

总结：`np.allclose` 是 NumPy 库中用于比较两个数组是否在指定的容差范围内相等的函数。通过指定两个数组以及相对容差和绝对容差，可以判断两个数组是否在容差范围内相等。返回一个布尔值，指示比较结果。

如何评价两个数组的大小关系还不知道。

9

在 Python 中，f-string 是一种字符串格式化的方式，用于将变量值插入到字符串中。f-string 是在 Python 3.6 版本中引入的。

f-string 使用形如 `f"..."` 的语法，其中字符串的前缀是字母 "f"。在 f-string 中，可以在字符串中使用花括号 `{}` 来引用变量，并在运行时将其替换为相应的值。

在给定的示例中，使用了 f-string 来构建一个包含动态内容的字符串。具体来说，`f"Episode {episode}/{num_episodes}: #updates={len(state_ridxs)} return={min(returns)}"` 中的花括号 `{}` 内的表达式会被替换为相应的变量值。

例如，如果 `episode` 的值为 3，`num_episodes` 的值为 10，`state_ridxs` 的长度为 100，`returns` 是一个列表，其中包含一些返回值，那么这个 f-string 在运行时会被解析为类似于以下的字符串：

"Episode 3/10: #updates=100 return=0.5"

其中，`{episode}` 被替换为变量 `episode` 的值 3，`{num_episodes}` 被替换为变量 `num_episodes` 的值 10，`{len(state_ridxs)}` 被替换为 `state_ridxs` 的长度 100，`{min(returns)}` 被替换为 `returns` 列表中的最小值 0.5。

总结：f-string 是一种字符串格式化的方式，在字符串中插入变量值。使用 `f"..."` 的语法，其中花括号 `{}` 内的表达式会在运行时被替换为相应的变量值。这种格式化方式使得构建包含动态内容的字符串更加简洁和直观。

一个新的字符串格式化方式。最前面的f代表format。

10

要找到向量中所有最大值的下标，可以使用 NumPy 库中的函数 `np.where()`。该函数可以根据指定的条件，返回满足条件的元素的下标。

下面是一个示例，展示如何找到向量中所有最大值的下标：

```
import numpy as np

# 创建一个一维数组
v = np.array([3, 2, 5, 2, 1, 5, 5])

# 找到最大值
max_value = np.max(v)

# 找到所有最大值的下标
max_indices = np.where(v == max_value)[0]

print(max_indices)
# 输出: [2 5 6]
```

在这个示例中，我们首先创建了一个一维数组 `v`，包含了元素 `[3, 2, 5, 2, 1, 5, 5]`。然后，使用 `np.max()` 函数找到了数组中的最大值，即 5。接下来，使用 `np.where()` 函数和条件 `v == max_value`，找到了满足条件的所有元素的下标。最后，打印出最大值的下标 `[2, 5, 6]`。

注意，`np.where()` 返回的结果是一个包含满足条件的元素下标的元组。如果只关心一维数组中的下标，可以通过索引 `[0]` 来获取所需的结果。

如果向量中存在多个最大值，并且希望得到所有最大值的下标，可以使用 `np.where()` 函数来实现。

总结：使用 NumPy 的 `np.where()` 函数可以根据条件找到向量中满足条件的元素的下标。通过找到最大值，并使用等于条件来找到所有最大值的下标。

11

在 Python 中，`enumerate` 是一个内置函数，用于在迭代过程中同时获取元素的索引和值。它返回一个可迭代对象，该对象生成一个索引和对应值的元组。

`enumerate` 函数的语法如下：

```python
enumerate(iterable, start=0)
```

其中，`iterable` 是一个可迭代对象，例如列表、元组、字符串等。`start` 是可选参数，表示索引的起始值，默认为 0。

下面是一个简单的示例，展示了如何使用 `enumerate` 函数：

```python
fruits = ["apple", "banana", "orange"]

for index, fruit in enumerate(fruits):
    print(index, fruit)
```

在上述示例中，我们有一个名为 `fruits` 的列表，使用 `enumerate` 函数在迭代过程中获取每个水果的索引和值。在 `for` 循环中，`index` 变量存储索引值，`fruit` 变量存储对应的水果值。然后，我们打印出索引和水果的组合。

输出结果为：

```python
0 apple
1 banana
2 orange
```

通过使用 `enumerate` 函数，可以更方便地同时访问索引和元素值（自动获取迭代器相应的索引），适用于需要迭代并获取索引的情况，例如在循环中需要记录元素的位置或执行某些操作。

12

在 Python 中，`None` 是一个特殊的对象，表示空值或缺失值。在给定的上下文中，`x[:, None]` 的作用是为数组 `x` 添加一个新的维度。

具体来说，`None` 用作切片操作中的占位符，表示在该位置插入一个新的维度。在这个例子中，`x[:, None]` 将数组 `x` 变成一个二维数组，其中第二个维度的长度为 1。

下面是一个简单的示例，演示了如何使用 `None` 来添加新的维度：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5])
print(x.shape)  # 输出: (5,)

x = x[:, None]
print(x.shape)  # 输出: (5, 1)

print(x)
```

在上述示例中，我们有一个一维数组 `x`，包含元素 `[1, 2, 3, 4, 5]`。通过使用 `x[:, None]`，我们在 `x` 上插入了一个新的维度，将其变成一个列向量。`x.shape` 的输出结果从 `(5,)` 变为 `(5, 1)`，表示数组 `x` 现在是一个 5 行 1 列的二维数组。

输出结果为：

```python
(5,)
(5, 1)
[[1]
 [2]
 [3]
 [4]
 [5]]
```

这种用法在某些情况下是很有用的，例如在进行某些矩阵运算或与其他形状要求相符的数据进行操作时。

这种做法通常是将一个一维的向量（这里视为行向量）转换为一个列向量

13

`itertools.combinations_with_replacement` 是 Python 中 `itertools` 模块提供的一个函数，用于生成可重复组合的迭代器。它接受一个可迭代对象 `x_t` 和一个整数 `degree`，并生成所有从 `x_t` 中选取 `degree` 个元素的可重复组合。

下面是一个示例，演示了如何使用 `itertools.combinations_with_replacement` 函数：

```python
import itertools

x_t = ['a', 'b', 'c']
degree = 2

for items in itertools.combinations_with_replacement(x_t, degree):
    print(items)
```

在上述示例中，我们有一个列表 `x_t` 包含元素 `['a', 'b', 'c']`，并指定 `degree` 为 2。通过使用 `itertools.combinations_with_replacement(x_t, degree)`，我们生成了所有可重复组合的迭代器。在 `for` 循环中，我们遍历迭代器并打印每个组合。

输出结果为：

```python
('a', 'a')
('a', 'b')
('a', 'c')
('b', 'b')
('b', 'c')
('c', 'c')
```

这些输出表示从 `x_t` 中选取 2 个元素的可重复组合。例如，('a', 'a') 表示选取两个 'a'；('a', 'b') 表示选取一个 'a' 和一个 'b'，依此类推。

`itertools.combinations_with_replacement` 在处理组合问题时非常有用，特别是需要考虑重复元素的情况。

14

`itertools.combinations_with_replacement` 是 Python 中 `itertools` 模块提供的一个函数，用于生成可重复组合的迭代器。它接受一个可迭代对象 `x_t` 和一个整数 `degree`，并生成所有从 `x_t` 中选取 `degree` 个元素的可重复组合。

下面是一个示例，演示了如何使用 `itertools.combinations_with_replacement` 函数：

```
import itertools

x_t = ['a', 'b', 'c']
degree = 2

for items in itertools.combinations_with_replacement(x_t, degree):
    print(items)
```

在上述示例中，我们有一个列表 `x_t` 包含元素 `['a', 'b', 'c']`，并指定 `degree` 为 2。通过使用 `itertools.combinations_with_replacement(x_t, degree)`，我们生成了所有可重复组合的迭代器。在 `for` 循环中，我们遍历迭代器并打印每个组合。

输出结果为：

```
('a', 'a')
('a', 'b')
('a', 'c')
('b', 'b')
('b', 'c')
('c', 'c')
```

这些输出表示从 `x_t` 中选取 2 个元素的可重复组合。例如，('a', 'a') 表示选取两个 'a'；('a', 'b') 表示选取一个 'a' 和一个 'b'，依此类推。

`itertools.combinations_with_replacement` 在处理组合问题时非常有用，特别是需要考虑重复元素的情况。

15

`functools.reduce` 是 Python 中 `functools` 模块提供的一个函数，用于对可迭代对象进行累积操作。它接受一个二元函数（例如 lambda 函数）和一个可迭代对象，将函数应用于可迭代对象中的元素，从左到右依次累积计算结果。

在给定的上下文中，`functools.reduce(lambda x, y: x * y, items)` 使用 lambda 函数对 `items` 中的元素进行累积乘法操作。

下面是一个示例，演示了如何使用 `functools.reduce` 和 lambda 函数进行累积乘法操作：

```python
import functools

items = [1, 2, 3, 4, 5]

result = functools.reduce(lambda x, y: x * y, items)
print(result)
```

在上述示例中，我们有一个列表 `items` 包含元素 `[1, 2, 3, 4, 5]`。通过使用 `functools.reduce(lambda x, y: x * y, items)`，我们对列表中的元素进行累积乘法操作。lambda 函数定义了一个二元操作，将 `x` 和 `y` 相乘。`functools.reduce` 函数将该操作应用于列表中的元素，从左到右依次进行累积计算。

输出结果为：

```python
120
```

这个结果表示对列表中的元素进行累积乘法操作的结果。在本例中，1 * 2 * 3 * 4 * 5 的结果为 120。

`functools.reduce` 在需要对可迭代对象进行累积操作时非常有用，例如求和、乘积、最大值等。

16

`np.asarray()` 和 `np.array()` 都是 NumPy 提供的函数，用于将输入转换为 NumPy 数组。它们之间的区别如下：

1. **处理方式**：`np.asarray()` 会尽可能地将输入转换为 NumPy 数组，如果输入已经是一个 NumPy 数组，则返回自身。如果输入是一个支持数组协议的对象（例如列表、元组、其他数组类），则会创建一个新的 NumPy 数组，并将数据复制到该数组中。相比之下，`np.array()` 可以接受多种输入类型，并根据输入的情况来决定是创建一个新的数组还是返回一个已有的数组。
2. **默认行为**：`np.asarray()` 默认情况下会共享输入数据的内存，即如果输入已经是一个 NumPy 数组，则返回的数组与输入数组共享相同的数据。相对而言，`np.array()` 默认情况下会复制输入数据，即始终创建一个新的数组，并将数据复制到该数组中。
3. **参数控制**：`np.asarray()` 和 `np.array()` 都接受一些可选参数，用于控制数组的创建和转换行为。例如，可以使用 `dtype` 参数指定所创建数组的数据类型，或使用 `order` 参数指定数组的存储顺序。

下面是一个示例，演示了 `np.asarray()` 和 `np.array()` 的区别：

```python
import numpy as np

# 使用 np.asarray() 将列表转换为数组
lst = [1, 2, 3]
arr1 = np.asarray(lst)
print(arr1)

# 使用 np.array() 将列表转换为数组
arr2 = np.array(lst)
print(arr2)

# 共享内存的例子
arr3 = np.asarray(arr2)
arr4 = np.array(arr2)
arr2[0] = 99
print(arr3)  # 输出: [99  2  3]
print(arr4)  # 输出: [1 2 3]
```

在上述示例中，我们首先使用 `np.asarray()` 和 `np.array()` 将列表 `lst` 转换为 NumPy 数组。结果表明，两个函数都能成功地将列表转换为数组。

另外，我们还展示了共享内存的例子。通过使用 `np.asarray()` 和 `np.array()` 转换已有的数组 `arr2`，并对原始数组 `arr2` 进行修改，我们可以观察到共享内存的情况。`np.asarray()` 返回的数组 `arr3` 与 `arr2` 共享相同的数据，因此对 `arr2` 的修改会反映在 `arr3` 中。而 `np.array()` 返回的数组 `arr4` 是一个新的数组，对 `arr2` 的修改不会影响 `arr4`。

综上所述，`np.asarray()` 和 `np.array()` 的主要区别在于处理方式和默认行为，但它们都是用于将输入转换为 NumPy 数组的有用工具。选择使用哪个函数取决于具体的需求和预期的行为。

17

`np.linalg.pinv()` 是 NumPy 提供的函数，用于计算矩阵的伪逆（也称为广义逆或 Moore-Penrose 逆）。在给定的上下文中，`np.linalg.pinv(x_train)` 表示将矩阵 `x_train` 计算其伪逆，并将结果赋值给变量 `self.w`。

伪逆是矩阵运算中的一种扩展，用于处理不可逆矩阵或奇异矩阵。它在最小二乘问题、线性回归和解决方程组等任务中具有重要应用。

下面是一个示例，演示了如何使用 `np.linalg.pinv()` 计算矩阵的伪逆：

```python
import numpy as np

# 假设 x_train 是一个矩阵
x_train = np.array([[1, 2],
                    [3, 4]])

self.w = np.linalg.pinv(x_train)
print(self.w)
```

在上述示例中，我们有一个二维矩阵 `x_train`，形状为 (2, 2)。通过使用 `np.linalg.pinv()`，我们计算了矩阵 `x_train` 的伪逆，并将结果赋值给变量 `self.w`。

输出结果为：

```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

这个结果表示矩阵 `x_train` 的伪逆。伪逆可以看作是矩阵的一种广义逆，具有类似逆矩阵的性质，但可以适用于不可逆或奇异矩阵的情况。

通过计算矩阵的伪逆，可以解决各种线性代数问题，并用于数据分析、信号处理、优化等领域。在给定的上下文中，将 `np.linalg.pinv(x_train)` 的结果赋值给 `self.w` 可能是为了在某种模型或算法中使用伪逆来进行计算或优化。

17

在给定的代码中，首先使用 `np.meshgrid()` 函数生成了两个二维坐标矩阵 `w0` 和 `w1`，这两个矩阵的形状都是 (100, 100)。然后，使用 `np.array()` 函数将 `w0` 和 `w1` 组合成一个三维数组 `w`，并通过 `transpose()` 函数调整了维度顺序。

下面是一个示例，演示了如何生成 `w` 数组：

```python
import numpy as np

w0, w1 = np.meshgrid(
    np.linspace(-1, 1, 100),
    np.linspace(-1, 1, 100))

w = np.array([w0, w1]).transpose(1, 2, 0)

print(w.shape)
```

在上述示例中，我们使用 `np.linspace(-1, 1, 100)` 生成了包含 100 个从 -1 到 1 等间距的值的一维数组，并将其传递给 `np.meshgrid()` 函数的两个参数。

然后，我们使用 `np.array()` 将 `w0` 和 `w1` 组合成一个三维数组，其中 `w0` 和 `w1` 分别对应于数组 `w` 的第一个维度和第二个维度。通过 `transpose(1, 2, 0)`，我们调整了维度顺序，使得 `w` 的形状变为 (100, 100, 2)。

输出结果为：

```
(100, 100, 2)
```

这表明 `w` 数组的形状是 (100, 100, 2)，其中前两个维度对应于 `w0` 和 `w1` 的形状，而第三个维度对应于 `w` 数组的组合维度。

这样生成的 `w` 数组常用于表示二维坐标网格或参数组合的情况，例如在机器学习中的参数搜索、函数计算或图形绘制等场景中。

18

`np.linalg.solve()` 是 NumPy 提供的一个函数，用于求解形如 `A @ X = B` 的线性方程组，其中 `A` 是一个方阵，`X` 和 `B` 是矩阵或向量。

具体而言，`np.linalg.solve()` 的作用是通过数值方法求解线性方程组，找到满足等式关系的未知向量或矩阵 `X`。

下面是一个示例，演示了如何使用 `np.linalg.solve()` 解决线性方程组：

```python
import numpy as np

A = np.array([[2, 3], [4, 5]])
B = np.array([1, 2])

X = np.linalg.solve(A, B)

print(X)
```

在上述示例中，我们定义了一个 2x2 的方阵 `A` 和一个长度为 2 的向量 `B`。我们想要找到一个未知向量 `X`，使得 `A @ X = B`。

通过调用 `np.linalg.solve(A, B)`，我们使用 `np.linalg.solve()` 函数解决了这个线性方程组，并得到了向量 `X`。

输出结果为：

```
[-1.   0.5]
```

这表明解 `X` 的值是 `[-1.   0.5]`，满足等式关系 `A @ X = B`。

`np.linalg.solve()` 函数内部使用了数值方法，例如 LU 分解或 QR 分解，来求解线性方程组。它在科学计算、线性代数和数值计算等领域中被广泛使用。请注意，方阵 `A` 必须是非奇异的（可逆的），否则求解过程可能会失败。

19

`np.clip()` 是 NumPy 库中的一个函数，用于将数组的值限制在指定的范围内。

函数的语法如下：

```python
np.clip(a, a_min, a_max, out=None)
```

参数说明：

- `a`：要进行限制的数组。
- `a_min`：限制的下界，即将数组中的值限制为不小于 `a_min`。
- `a_max`：限制的上界，即将数组中的值限制为不大于 `a_max`。
- `out`：可选参数，指定输出的数组，其形状和类型与输入数组 `a` 相同。

`np.clip()` 函数将数组 `a` 中的每个元素与指定的上下界进行比较，并将其限制在指定的范围内。如果元素小于下界 `a_min`，则将其设为 `a_min`；如果元素大于上界 `a_max`，则将其设为 `a_max`；如果元素在指定范围内，则保持不变。

下面是一个示例，演示了如何使用 `np.clip()` 函数将数组的值限制在指定的范围内：

```python
import numpy as np

arr = np.array([2, 4, 6, 8, 10])
clipped_arr = np.clip(arr, 3, 7)

print(clipped_arr)
```

输出：

```
[3 4 6 7 7]
```

在上述示例中，我们创建了一个示例的数组 `arr`。然后，使用 `np.clip()` 函数将数组的值限制在范围 `[3, 7]` 内。结果数组 `clipped_arr` 中的元素小于 3 的被限制为 3，大于 7 的被限制为 7，而在指定范围内的元素保持不变。

通过 `np.clip()` 函数，我们可以方便地对数组进行值的限制，以确保数组的值在指定范围内。

20

`np.unique` 是NumPy库中的一个非常有用的函数，它用于找出数组中的所有唯一值，并可以选择性地返回这些唯一值出现的频率。这个函数在数据分析和处理中特别有用，尤其是当你需要识别或计数数组中不同元素的时候。

函数的基本用法如下：

```python
unique_elements = np.unique(array)
```

这将返回一个新的数组，其中包含 `array` 中所有独特的、排序后的值。

如果你还想知道每个独特值在原数组中出现的次数，可以使用 `return_counts` 参数：

```python
unique_elements, counts = np.unique(array, return_counts=True)
```

这里，`unique_elements` 是一个数组，包含了原数组中的所有唯一值，而 `counts` 是一个相同长度的数组，表示每个独特值在原数组中出现的次数。

此外，`np.unique` 还有一些其他参数，可以帮助你进行更特定的操作，如：

- `axis` 参数，用于在多维数组中沿指定轴查找唯一值。
- `return_index` 参数，返回原数组中每个唯一值首次出现的索引。
- `return_inverse` 参数，返回一个数组，用于重构原数组。

例如：

```
pythonCopy code
unique_elements, indices = np.unique(array, return_index=True)
```

这将返回唯一元素以及它们在原数组中首次出现的位置。

### python编程小技巧

```python
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
```

此后每次调用`policy`这个函数所使用的参数都是当前的的参数，也即`Q`会随着调用时间的不同而不同。

这个现象反映了 Python 中的闭包（closure）概念。闭包指的是一个函数记住了它所在的词法作用域，即使函数在其作用域外被调用。在这个特定的情况中，`policy` 函数（由 `make_epsilon_greedy_policy` 创建）是一个闭包，因为它引用了它外部作用域中的变量 `Q`、`epsilon` 和 `nA`。

当你通过 `make_epsilon_greedy_policy` 创建 `policy` 函数时，`policy` 会记住并引用它被创建时的 `Q` 字典。这不是通过拷贝 `Q` 的值实现的，而是通过引用。这意味着 `policy` 内部使用的 `Q` 与外部的 `Q` 是同一个对象。因此，当外部的 `Q` 更新时（例如，通过学习算法更新 Q-value），`policy` 函数内部引用的 `Q` 也会随之变化。

这种行为是 Python 函数作用域和对象引用行为的自然结果，它在设计如强化学习算法这类需要记住并更新状态估计的应用时特别有用。这允许 `policy` 函数动态地适应 `Q` 的变化，反映最新的学习成果，而不需要每次 `Q` 更新时都重新创建 `policy` 函数。

这种技巧在实现复杂的算法时非常有用，因为它减少了需要手动同步不同部分的状态或信息的需要。通过利用闭包和引用传递，算法的不同部分可以自然地保持同步。

因为一般的表格型在每次更新Q表时候，都要手动更新policy（如果你的policy和Q表是同步的话），那么这个技巧就可以将Q表与policy所绑定，每次查询policy中的某个状态时，则自动使用最新的Q表。其实也等效于传入被闭包的参数，效果是一样的，只不过这样做更简洁，因为传入的参数更少。

## Gymnasium

### Wrapper

![image-20240406154353768](C:\Users\outline\AppData\Roaming\Typora\typora-user-images\image-20240406154353768.png)

