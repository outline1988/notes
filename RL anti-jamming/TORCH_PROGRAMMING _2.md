## 正式的学习pytorch

### 自动梯度的原理

在torch中，任何张量都属于两类：一类为leaf node，即叶子节点，正如名字所说，他是在计算图中的最末端，这类数据不来源于任何其他的数据，而只是凭空生成的，你可以理解为他就是一个最为最为底层的自变量；另一类为非leaf节点，这类数据是由其他数据或其他方式生成的，也即这类张量本身是某个自变量的函数。

关于更多自动梯度的原理，可以参考[一文详解pytorch的“动态图”与“自动微分”技术](https://zhuanlan.zhihu.com/p/351687500)。

关于自动梯度，张量的一个属性特别关键`.grad_fn`，可以将其视为一个指针，它指向获得到本张量操作的函数，所以他就是计算图中的箭头。对于leaf节点来说，他是凭空生成的，故`grad_fn=None`表示他没有指向前面任何一个操作。

### 网络更新的步骤

使用以下网络作为例子
```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入层到隐藏层
        self.fc2 = nn.Linear(5, 2)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**前向传播**
要更新一个网络，首先要进行前向传播，在前向传播中，你应当视输入数据和网络参数都为leaf节点，也即它们都不包含以往的计算信息。前向的传播就是在创建一个计算图

```python
y = net(x)
loss = criterion(y, label)
```

**反向传播**
计算图在前向传播的过程中创建完成之后，就要进行反向传播，以得到计算图中各个节点tensor的梯度信息。注意，为了防止有些参数（因为我们要求参数的梯度，所以就要清楚参数的历史信息）之前的反向传播已经得到了一个梯度信息，所以在此之前要清除以往的梯度信息。清除`net.zero_grad()`一定是一个函数操作。

```python
net.zero_grad()
loss.backward()
```

**网络参数更新：backward后访问grad**
创建计算图的目的是为了求出关于参数的梯度，自计算图的创建和反向传播完成之后，我们实际上已经获得了网络参数的梯度信息了，所以在之后对参数的更新中，我们不再需要创建计算图了，所以使用`with torch.no_grad()`定义的块中进行梯度更新

```python
lr = 0.003
with torch.no_grad():
    for param in net.parameters():
        grads = param.grad
        param.data = param.data - lr * grads
```

上述代码完成了梯度的一次更新。注意，`net.parameters()`生成的是一个迭代器，依次返回每一层参数tensor的某个封装，我们可以通过`.data`访问实际网络参数的张量表示，也可以通过`.grad`访问实际网络参数的梯度。还需要注意的是，使用`.data`访问得到的张量只包含了数据本身，你可以通过改变这个数据来改变网络参数，但该数据不包含有梯度信息。要得到梯度信息，应该直接用`param.grad`访问而不是`param.data.grad`，后者是不行的。

**网络参数更新：使用自动微分函数**
先使用反向传播backward在使用grad方法得到的梯度信息，会由于在backward中一次性影响传播节点的梯度。使用自动微分函数能够直接计算图中某个节点的梯度信息，而不会影响其他。

在网络参数的更新中，`torch.autograd.grad()`方法直接支持使用`net.parameters()`生成的迭代器来进行梯度的计算，并且也返回一个迭代器。使用这个迭代器的方法和`net.parameters()`一样，使用自动微分函数不会累积梯度信息。
```python
y = net(x)
loss = criterion(y, label)

grads = torch.autograd.grad(loss, net.parameters())

with torch.no_grad():
    for param, grad in zip(net.parameters(), grads):
        param.data -= lr * grad
```

使用这种方式不需要像其他两种方式一样手动清空梯度信息。

在使用`torch.autograd.grad()`函数时，默认情况下，新得到的参数是leaf节点，但是你可以手动设置`create_graph=True`，由此，得到的梯度也被增加到了网络图中。不要太复杂化一阶导的操作，就和其他加减法一样，他也被增加到了计算图中而已。

**网络参数更新：使用优化器**
优化器的方式进行优化，更为顶层

```python
optim = torch.optim.SGD(net.parameters(), lr)

y = net(x)
loss = criterion(y, label)

net.zero_grad()
loss.backward()
optim.step()
```

可以视为`optim`为第一种网络参数更新的升级版，共同点是两者都要手动`backward()`。

使用`backward()`的更新方式可以认为更加底层，因为需要手动反向传播和手动梯度更新，也就是第一种和第三种。但是第二种可以认为只要计算图建立起来了，他就能够计算，不用在意之前积累的梯度信息。
