### Basics of Supervised Learning

有监督学习的前提是有一堆带有标注的数据集，即$\mathcal{D} = \{(\mathbf{x}, \mathbf{y})_k\}$，其中$\mathbf{x}$和$\mathbf{y}$都是随机变量。通常，数据集中的输入取自$p(\mathbf{x})$，整个带有标注的数据则取自联合概率分布$p(\mathbf{x}, \mathbf{y})$，这些是客观存在的。在有监督学习的问题下，我们关注的是如何将输入$\mathbf{x}$转换为$\mathbf{y}$，而这一过程就是$p(\mathbf{y} | \mathbf{x})$。简要来说，这个世界上客观存在了$p(\mathbf{x})$和$p(\mathbf{x}, \mathbf{y})$，则有了转换关系$p(\mathbf{y} | \mathbf{x})$。

我们现在希望拟合这个转换关系$p(\mathbf{y} | \mathbf{x})$，再加上我们能够轻松获取输入的分布（不断地采样），我们就能得到$p(\mathbf{x}, \mathbf{y})$。我们希望用一个可有参数控制的非线性函数$f_{\boldsymbol\theta}$来拟合这个转换关系（这个非线性函数通常就是神经网络）。拟合的基础在于定义了一个损失函数，如下为单个数据的样本的损失函数定义
$$
\mathcal{L}(\hat{\mathbf{y}}_i, \mathbf{y}_i) = \mathcal{L}(f_{\boldsymbol\theta}(\mathbf{x}_i), \mathbf{y}_i) = \mathcal{L}\big(\boldsymbol\theta, (\mathbf{x}_i, \mathbf{y}_i) \big)
$$
所有数据的损失函数
$$
\sum_i \mathcal{L}\big(\boldsymbol\theta, (\mathbf{x}_i, \mathbf{y}_i) \big) = \mathcal{L}(\boldsymbol\theta, \mathcal{D})
$$
故将数据来替代期望符号的时候，一个任务的损失函数就是由一个网络参数$\boldsymbol\theta$和一个数据集$\mathcal{D}$共同决定。简单来说，在数据集给定的情况下，一个损失$\mathcal{L}$与一个网络参数$\boldsymbol\theta$完全对应。所以在最终数据集给定的情况下，一个网络参数，就对应一个损失，只不过正真计算损失函数的时候，考虑数据集很大，只会取一个batch。

### What is a Task?

一个任务就是一个集合，如下
$$
\mathcal{T}_i = \{ p_i(\mathbf{x}), p_i(\mathbf{x}, \mathbf{y}), \mathcal{L}_i \}
$$
注意，我们定义一个任务不是用一个数据集来定义，而是用一个生成数据集的分布来定义，所以这里定义的任务要更加广泛。
