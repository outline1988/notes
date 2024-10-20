## tianshou

### Batch && Buffer

**Batch**
Batch是tianshou自己使用的一个基本的数据结构，可以将其视为一个高级特化的字典，通常来说，这个字典包括 `obs`，`act`，`rew`，`terminated`，`truncated`，`obs_next`等字段，这也是agent与环境一次交互的所有数据。每个字段通常包含一个第一维度数量相同的np数组，各表示不同交互的数据。其中`truncated`表示的是环境中由于其他情况没能到达的终端状态的情况，通常为交互次数超出最大限制，这是环境自己定义的。

在Batch的基础上，可派生出不同的protocol，比如，对于`BatchWithReturnsProtocol`来说，其要求Batch的字段中除了基本字段，还要有`return`字段，这样做的意义是实现了Batch不同状态之间的解耦。

**Buffer**
Buffer可视为Batch的封装，通常一个Batch是用来实际更新策略的数据，而这个Batch就采样自Buffer。所以Buffer用来装所有的交互数据，从Buffer中采样得到Batch后，再用Batch更新策略。

### Policy

在tianshou中，每一个DRL算法都以某个Policy的类出现，这里以实例化一个PG类为例，其过程如下
```python
state_shape = 4
action_shape = 2
# Usually taken from an env by using env.action_space
action_space = gym.spaces.Box(low=-1, high=1, shape=(2,))
net = Net(state_shape, hidden_sizes=[16, 16], device="cpu")
actor = Actor(net, action_shape, device="cpu").to("cpu")
optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
dist_fn = torch.distributions.Categorical

policy: BasePolicy
policy = PGPolicy(actor=actor, optim=optim, dist_fn=dist_fn, action_space=action_space)
```

#### Net
tianshou中的网络有基本的模块

```python
from tianshou.utils.net.common import Net
```

这个Net模块是MLP模块的包装，通常，以如下方式进行实例化
```python
net = Net(state_shape, hidden_sizes=[16, 16], device="cpu")
```

可以将其视为一个缺少输出的网络，在定义时只需要传入输入层大小和中间隐层大小，其中，隐层时以列表的形式传入的。最后再指定设备。

**Actor**
在policy based的算法中，一个Actor就是一个策略网络。在前面完成Net模块的实例化后，我们最终要为其增加一个输出从而形成一个真正的策略网络Actor

```python
actor = Actor(net, action_shape, device="cpu").to("cpu")
```

如上，我们为`net`增加了输出层的大小，最终形成了一个策略网络Actor。关于`device`，记住固定这样的格式吧。

**Policy**
实例化一个Policy，相当于完成了所有关于DRL算法所需要准备，包括网络的优化器`optim`等等

```python
policy: BasePolicy
policy = PGPolicy(actor=actor, optim=optim, dist_fn=dist_fn, action_space=action_space)
```

1. 首先还是传入刚才定义的`actor`参数。

2. 其次，关于`optim`，就是定义如何更新网络，一般都选择
   ```python
   optim = torch.optim.Adam(actor.parameters(), lr=0.0003)
   ```

3. `dist_fn`是关于actor最终输出的类型，策略网络输出的都是关于动作空间的分布，这里的动作空间为离散的，所以定义为

   ```python
   dist_fn = torch.distributions.Categorical
   ```

4. 最后不知道的是为什么要传入`action_space`，无所谓，记住完了。

打印这个policy，最终会输出清晰的结构
```cmd
PGPolicy(
  (actor): Actor(
    (preprocess): Net(
      (model): MLP(
        (model): Sequential(
          (0): Linear(in_features=4, out_features=16, bias=True)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=16, bias=True)
          (3): ReLU()
        )
      )
    )
    (last): MLP(
      (model): Sequential(
        (0): Linear(in_features=16, out_features=2, bias=True)
      )
    )
  )
)
```

**Algorithm Updating**
这里，我们伪造一个实际的假数据，将其用于策略的更新，从而实现一边策略学习的过程

```python
dummy_buffer = ReplayBuffer(size=10)
env = gym.make("CartPole-v1")
```

首先定义一个Buffer，用于存放数据
```python
obs, info = env.reset()
for i in range(3):
    act = policy(Batch(obs=obs[np.newaxis, :])).act.item()
    obs_next, rew, _, truncated, info = env.step(act)
    # pretend ending at step 3
    terminated = i == 2
    info["id"] = i
    dummy_buffer.add(
        Batch(
            obs=obs,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            obs_next=obs_next,
            info=info,
        ),
    )
    obs = obs_next
```

这里强制让其终止在第三步，紧接着手机剩下数据
```python
obs, info = env.reset()
for i in range(3, 10):
    act = policy(Batch(obs=obs[np.newaxis, :])).act.item()
    obs_next, rew, _, truncated, info = env.step(act)
    # pretend this episode never end
    terminated = False
    info["id"] = i
    dummy_buffer.add(
        Batch(
            obs=obs,
            act=act,
            rew=rew,
            terminated=terminated,
            truncated=truncated,
            obs_next=obs_next,
            info=info,
        ),
    )
    obs = obs_next
```

由此便完成了数据的收集，**注意第二个episode是不完整的**。

我们使用一行代码就可以将数据用于策略的更新了
```python
policy.update(sample_size=0, buffer=dummy_buffer, batch_size=10, repeat=6)
```

其中，`sample_size`表示为从Buffer中抽取的Batch数量。在抽取完成之后，这些Batch还会进一步被分割为一个个的miniBatch，每次的更新，就是使用`miniBatch`来进行。`repeat`表示抽取的Batch要重复多少次更新。

这行代码调用`BasePolicy.update()`方法，这个方法是内定的，具体代码如下

```python
def update(
    self,
    sample_size: int | None,
    buffer: ReplayBuffer | None,
    **kwargs: Any,
) -> TTrainingStats:
    if buffer is None:
        return TrainingStats()  # type: ignore[return-value]
    start_time = time.time()
    batch, indices = buffer.sample(sample_size)
    self.updating = True
    batch = self.process_fn(batch, buffer, indices)
    training_stat = self.learn(batch, **kwargs)
    self.post_process_fn(batch, buffer, indices)
    if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    self.updating = False
    training_stat.train_time = time.time() - start_time
    return training_stat
```

- 首先对Buffer使用`sample_size`参数进行采样得到一个Batch，然后将Batch和Buffer共同传入`process_fn()`，通常这个函数都在子类中实现了重定义，比如`PGPolicy`中，该函数计算出了Batch中每个状态的回报，为各个状态实现了解耦。
- `learn()`方法为更新主体，其在各个方法所需参数不同的情况下，统一在`**kwargs`中，完成了对策略网络的更新。其中也有参数是共同的，也就是用于更新的Batch。

### PGPolicy

**`forward()`**
该函数实现策略网络的前向更新，其接受Batch类型，返回Batch中每个状态在策略中关于动作的分布。一般来说`state=None`，具体定义如下

```python
def forward(
    self,
    batch: ObsBatchProtocol,
    state: dict | BatchProtocol | np.ndarray | None = None,
    **kwargs: Any,
) -> DistBatchProtocol:
    """Compute action over the given batch data by applying the actor.

    Will sample from the dist_fn, if appropriate.
    Returns a new object representing the processed batch data
    (contrary to other methods that modify the input batch inplace).
    """
    logits, hidden = self.actor(batch.obs, state=state)

    if isinstance(logits, tuple):
        dist = self.dist_fn(*logits)
    else:
        dist = self.dist_fn(logits)

    act = dist.sample()
    return cast(DistBatchProtocol, Batch(logits=logits, act=act, state=hidden, dist=dist))

```

首先进行`actor`的网络，得到一个原生输出`logits`，其与最后的概率成比例。这个原生输出后会经过`dist_fn()`而形成一个分布，离散的情况下，`dist_fn()`通常为`torch.distributions.Categorical`。

在得到了一个分布的前提下，进一步对该分布进行采样，得到一个动作。最后多个数据共同形成一个符合`DistBatchProtocol`协议的Batch进行返回。

**`process_fn()`**
该方法是在更新参数`learn()`方法之前进行的，表示对Batch进行预处理，在这里对Batch的每个状态求得return从而实现解耦。

**`learn()`**
该方法就是最终实现参数更新的主体了

```python
def learn(  # type: ignore
    self,
    batch: BatchWithReturnsProtocol,
    batch_size: int | None,
    repeat: int,
    *args: Any,
    **kwargs: Any,
) -> TPGTrainingStats:
    losses = []
    split_batch_size = batch_size or -1
    for _ in range(repeat):
        for minibatch in batch.split(split_batch_size, merge_last=True):
            self.optim.zero_grad()
            result = self(minibatch)  # 相当于一次forward
            dist = result.dist  # 输出的分布，每个状态都有
            act = to_torch_as(minibatch.act, result.act)  # 输入的batch的各种动作转换为torch格式
            ret = to_torch(minibatch.returns, torch.float, result.act.device)
            log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
            loss = -(log_prob * ret).mean()
            loss.backward()
            self.optim.step()
            losses.append(loss.item())

    loss_summary_stat = SequenceSummaryStats.from_sequence(losses)

    return PGTrainingStats(loss=loss_summary_stat)  # type: ignore[return-value]
```

- 首先对`Batch.split()`函数进行介绍
  ```python
  split(size: int, shuffle: bool = True, merge_last: bool = False)
  ```

  该方法将一个Batch拆分成多个minibatch，然后生成每个minibatch的迭代器。其中，`size`为-1时表示整段Batch生成器；`shuffle`表示是否要打乱后再生成；`merge_last`表示当Batch的长度不为`size`的整数倍时，最后一个miniBatch会比`size`的数量要更多。例如
  ```python
  batch_ = Batch({"a": [4, 5, 6, 7, 8], "b": (1, 2, 3, 4, 5)})
  for minibatch in batch_.split(2, shuffle=True, merge_last=True):
      print(minibatch)
  # output:
  # Batch(
  #     a: array([5, 4]),
  #     b: array([2, 1]),
  # )
  # Batch(
  #     a: array([7, 6, 8]),
  #     b: array([4, 3, 5]),
  # )
  ```

- 然后就是梯度更新的标准格式

  - `optim.zero_grad()`
  - 计算loss
  - `loss.backward()`
  - `optim.step`

  其中，loss的计算是核心，对于PG来说，我们要更新的为以下表达式
  $$
  \hat{g}=\frac{1}{|\mathcal{D}|}\sum_{\tau\in\mathcal{D}}\sum_{t=0}^{T - 1}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})R(\tau)
  $$
  我们已经得到了每个状态$s_t$的动作$a_t$和对应的回报$R(\tau)$。回报可以直接从`minibatch.return`得到；对数概率需要先得到当前状态的策略输出分布，由于网络的输出是一个batch，而分布包含在这个batch中，故`dist = self(minibatch.obs).dist`，得到了分布直接带入动作即可得到对数概率`log_prob = dist.log_prob(act)`，进行这些操作的时候要注意那些函数要自动微分，哪些只是常数。

### Collector

**Policy Evaluation**
在训练完成一个策略之后，通常需要让这个策略进行测试，即与环境交互，看看测试性能。Collector模块以Policy和env为参数，以实现策略环境交互的功能

```python
test_collector = Collector(policy, test_envs)
```

其中`policy`是之前介绍过的Policy模块的实例化，`test_envs`通常是并行环境，由以下方式建立（记住就好）
```python
test_envs = DummyVectorEnv(
    [lambda: gym.make("CartPole-v1") for _ in range(2)]
)
```

由此，便可通过Collector中的各种方法来实现交互并测试的功能
```python
collect_result = test_collector.collect(n_episode=9)
collect_result.pprint_asdict()

# output: 
# CollectStats
# ----------------------------------------
# {'collect_speed': 1801.7701748664606,
#  'collect_time': 0.10600686073303223,
#  'lens': array([26, 29, 18, 23, 13, 14, 19, 25, 24]),
#  'lens_stat': {'max': 29.0,
#                'mean': 21.22222222222222,
#                'min': 13.0,
#                'std': 5.202088849208722},
#  'n_collected_episodes': 9,
#  'n_collected_steps': 191,
#  'returns': array([26., 29., 18., 23., 13., 14., 19., 25., 24.]),
#  'returns_stat': {'max': 29.0,
#                   'mean': 21.22222222222222,
#                   'min': 13.0,
#                   'std': 5.202088849208722}}

# Reset the collector
test_collector.reset()
collect_result = test_collector.collect(n_episode=9, random=True)
collect_result.pprint_asdict()

# output:
# CollectStats
# ----------------------------------------
# {'collect_speed': 3981.468165248562,
#  'collect_time': 0.05500483512878418,
#  'lens': array([16, 19, 28, 14, 42, 14, 26, 18, 42]),
#  'lens_stat': {'max': 42.0,
#                'mean': 24.333333333333332,
#                'min': 14.0,
#                'std': 10.498677165349081},
#  'n_collected_episodes': 9,
#  'n_collected_steps': 219,
#  'returns': array([16., 19., 28., 14., 42., 14., 26., 18., 42.]),
#  'returns_stat': {'max': 42.0,
#                   'mean': 24.333333333333332,
#                   'min': 14.0,
#                   'std': 10.498677165349081}}
```

**Data Collecting**
与策略评估时所用的Collector模块一样，只不过由于手机数据时，还需要有一个Buffer容器来存储，所以实例化Collector类时，要传入第三个Buffer参数

```python
train_env_num = 5
train_envs = DummyVectorEnv([lambda: gym.make('CartPole-v1') for _ in range(train_env_num)])

replaybuffer = VectorReplayBuffer(total_size=100, buffer_num=train_env_num)  # replaybuffer的参数要与env对应

train_collector = Collector(policy=policy, env=train_envs, buffer=replaybuffer)
```

紧接着实际采集数据至buffer中
```python
train_collector.reset()
replaybuffer.reset()

n_step = 64
collector_result = train_collector.collect(n_step=n_step)

collector_result.pprint_asdict()
print(replaybuffer)  # replaybuffer中是均匀分配Buffer中的各种容量来实现vector的效果的
```

注意此时的收集以`n_step`为单位，不需要完全收集完成整个episode，只要到达限定步数就行。

其次，vector样式的buffer将一个很长的数组均匀分成很多端，每个并行的环境的数据就占据其中一段。暂时不知道这样是如何实现后续的采样的。。。

### Trainer

在之前的模块都完成之后，可以手动进行更新
```python
train_collector.reset()
train_envs.reset()
test_collector.reset()
test_envs.reset()
replayBuffer.reset()

n_episode = 10
for _i in range(n_episode):
    evaluation_result = test_collector.collect(n_episode=n_episode)
    print(f"Evaluation mean episodic reward is: {evaluation_result.returns.mean()}")
    train_collector.collect(n_step=2000)
    # 0 means taking all data stored in train_collector.buffer
    policy.update(sample_size=None, buffer=train_collector.buffer, batch_size=512, repeat=1)
    train_collector.reset_buffer(keep_statistics=True)
```

大致过程如下

- 首先进行一个大循环，每次大循环内执行一次evaluate和update；
- 在evaluate阶段，使用test_collector采集测试数据，一般测试阶段采集episodes；
- 在update阶段，首先先收集`n_step`的数据；
- 这`n_step`的数据并非全部用来进行更新，而是随机抽取其中`sample_size`个数量`step`进行update；
- 在具体内部的`update`中（一次大update），将这`sample_size`数量的数据拆分为一个个minibatch对参数进行更新，直到minibatch用光所有采样的batch；
- 上一步步骤重复repeat次。
- 最后清空`replaybuffer`内的数据。

**Trainer**
可以直接使用Train模块来统一完成上述过程

```python
train_collector.reset()
train_envs.reset()
test_collector.reset()
test_envs.reset()
replayBuffer.reset()

result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=10,
    step_per_collect=2000,
    step_per_epoch=1,
    batch_size=512,
    repeat_per_collect=1,
    episode_per_test=10,
).run()
```

其中需要注意的参数是

- `max_epoch`表示整体的最大轮次，相当于前面大循环次数；
- `step_per_epoch`每个大epoch中，一共要交互的数据数量；
- `step_per_collect`每次用于update的交互次数，即在一个epoch中，每过`step_per_collect`次交互，就update一次，此时update会用上所有的`step_per_collect`数据。在上面的代码中，默认`step_per_collect`和`step_per_epoch`相同，所以一次epoch只进行一次update。
- `batch_size`，一次小更新所用的数据量，相当于前面的`batch_size`；
- `repeat_per_collect`每次大更新，采样后的数据重复运用几次，相当于前面的`repeat`参数；
- `episode_per_test`用于测试的`episodes`数，相当于前面的`test_n_episodes`。

每次更新中，若有`test_in_train`和`stop_fn`参数，那么在训练的过程中，会使用训练的数据来测试策略的效果，若策略的效果达到了`stop_fn`的条件，那么则会再收集`test`数据来得到回报，若这测试数据的回报也达到了`stop_fn`的条件，那么更新结束，训练完成。

### `.run()`后究竟发生了什么

由于Trainer的基类`BaseTrainer`，总体定下的方式就是迭代器训练，所以当`.run()`运行后，首先跳转到`__next__()`方法，

**`BaseTrainer.__next__()`**
运行一次迭代器就是一个epoch的训练（也即一个进度条），所以`__next__()`方法内包含了一个epoch内所有的训练过程。

1. 首先进入`self.train_step()`函数，这个函数用于进行数据的采集，即`self.train_collector.collect()`函数，收集`step_per_collect`次数据。采集完后，判断数据是否满足`self.stop_fn()`的条件，若满足，则进入测试步骤`test_episode()`。若测试的数据还满足`stop_fn()`条件，则直接结束训练，否则进入下一步骤。

2. 收集完数据之后，进入`self.policy_update_fn()`方法，该方法根据不同的训练方式而不同（on policy和off policy，这里主要以on policy为例）。在`OnpolicyTrainer`中，直接进行`self.policy.update()`的更新。

   - 这个`update()`方法由Policy的基类`BasePolicy()`进行定义；
   - 首先从步骤1中采集数据的容器`buffer`采样一个`batch`，通常都是采集整个`buffer`的数据为`batch`（前面`buffer`采集到了`step_per_collect`个数据，所以这里也采样这么多的`batch`）；
   - 将这个`batch`送入`process_fn()`方法进行预处理，这个`process_fn`是由算法层面定义的，比如PG中该方法用于计算回报；
   - 预处理完`batch`，紧接着进入`self.learn()`，这个函数就是具体实现不同DRL算法核心所在了，在on policy的训练中，反正就是给了`step_per_collect`个数据的`batch`进行训练；
   - 训练完成之后在进入`post_process_fn`，到此`self.policy_update_fn()`方法结束。

3. 到此，一个`epoch`结束。

## 类简述

### Net类

Net模块通常是对MLP的包装，用以实现特定的DRL算法，net通常用于网络的backbone部分，即该网络的输出层大小为hidden_layers[-1]的大小，最后输出一个logits。

由于其是对于MLP的包装，其拥有几个可选项
```python
state_shape: int | Sequence[int],  # 输入层大小，必须填
action_shape: TActionShape = 0,  # 通常不用填，因为网络是作为backbone，即从输入中提取特征
hidden_sizes: Sequence[int] = (),  # 隐藏层大小，必填
norm_layer: ModuleType | Sequence[ModuleType] | None = None,  # 用某种方式来对某层在激活前进行归一化（不太懂怎么使用）
norm_args: ArgsType | None = None,
activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,  # 每层的激活函数
act_args: ArgsType | None = None,
device: str | int | torch.device = "cpu",
softmax: bool = False,  # 最后的输出（hidden_size[-1]大小）后在进行softmax
concat: bool = False,
num_atoms: int = 1,
dueling_param: tuple[dict[str, Any], dict[str, Any]] | None = None,
linear_layer: TLinearLayer = nn.Linear,  # 线性模型的模板
```

典型的使用如下
```python
net = Net(  # 不包含输出层，只包含输入和隐藏层的MLP backbone
    args.state_shape,
    hidden_sizes=args.hidden_sizes,
    activation=nn.Tanh,  
    device=args.device,
)
```

只需要传入四个数据即可作为一个MLP backbone来使用了。



### ActorProb类

用来实现连续空间下的actor网络，即输出action_shape数量的高斯分布的两个参数，均值和方差，这里还未形成正真的高斯分布。其会使用到之前所定义的net网络作为backbone，然后在对其进行加工后形成最终的actorprob。

可选参数如下
```python
preprocess_net: nn.Module,  # 之前定义的net网络，作为backbone，一定要输入
action_shape: TActionShape,  # 动作空间的维度，一定要输入
hidden_sizes: Sequence[int] = (),  # 在backbone的基础上再增加隐藏层，通常不会增加，所以默认为空
max_action: float = 1.0,  # 对最终的tanh(logits)后的输出进行缩放[-max_action, max_action]中，所以必须搭配tanh使用
device: str | int | torch.device = "cpu",
unbounded: bool = False,  # 是否无界，无界则不适用tanh进行限制，有界则使用tanh搭配max_action缩放到[-max_action, max_action]内
conditioned_sigma: bool = False,  # sigma是否与输入有关，若是，则sigma单独为一个网络，和self.mu的网络相同，若不是，则为一个单独的参数进行更新，学习完毕后，无论怎样的输入，输出的sigma都固定不变。
preprocess_net_output_dim: int | None = None,
```

简单来说，actorprob的类实现了将net网络backbone的作用继续延申到neck和head结构，直至完成作为连续空间actor输出高斯分布的作用。

典型使用如下
```python
actor = ActorProb(
    net, 
    args.action_shape, 
    unbounded=False, 
    device=args.device, 
    max_action=1
).to(args.device)
```

这里将unbounded改为了False，则最后会使用tanh对其进行缩放。

### Critic类

Critic类则要简单得多，因为其只用作对状态价值函数的估计，所以再有了backbone之后，最终的输出大小直接为1，典型使用如下
```python
critic = Critic(
    Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        activation=nn.Tanh,
    ),
    device=args.device,
).to(args.device)
```

ActorCritic类将前面的Actor类和Critic类整合在一起，用以更好的将两个网络的参数传入优化器中。

## 网络

```python
from typing import Any
from collections.abc import Sequence  # 用于任何序列式的类型注解，通常就是list
ModuleType = type[nn.Module]  # ModuleType以后就是nn.Module的类型注解，用以定义函数或者类时的显式表达
ArgsType = tuple[Any, ...] | dict[Any, Any] | Sequence[tuple[Any, ...]] | Sequence[dict[Any, Any]]  # 传入函数的参数类型，因为传入函数的参数一般都是元组tuple或者字典dict，所以ArgsType首先定义了tuple[Any, ...]，其中Any表示任何类型，...表示长度任意；其次dict[Any, Any]表示任意类型的键值对。由于基于MLP的norm参数和activation参数是个序列的话，那么长度必须和hidden_sizes匹配，以对应每一层，所以在使用Sequence来重复。最后，| 表示类型可以是这四种类型中的一种。
TLinearLayer: TypeAlias = Callable[[int, int], nn.Module]  # TypeAlias表示类型别名，在这里指Callable[[int, int], nn.Module]类型的别名是TLinearLayer，其中Callable[[int, int], nn.Module]表示输入两个int类型，返回nn.Module类型的可调用对象，就是指nn.Linear

def miniblock(
    input_size: int,
    output_size: int = 0,
    norm_layer: ModuleType | None = None,
    norm_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    activation: ModuleType | None = None,
    act_args: tuple[Any, ...] | dict[Any, Any] | None = None,
    linear_layer: TLinearLayer = nn.Linear,
) -> list[nn.Module]:
    """Construct a miniblock with given input/output-size, norm layer and activation."""
    layers: list[nn.Module] = [linear_layer(input_size, output_size)]
    if norm_layer is not None:
        if isinstance(norm_args, tuple):
            layers += [norm_layer(output_size, *norm_args)]
        elif isinstance(norm_args, dict):
            layers += [norm_layer(output_size, **norm_args)]
        else:
            layers += [norm_layer(output_size)]
    if activation is not None:
        if isinstance(act_args, tuple):
            layers += [activation(*act_args)]
        elif isinstance(act_args, dict):
            layers += [activation(**act_args)]
        else:
            layers += [activation()]
    return layers


class MLP(nn.Module):
    """Simple MLP backbone.

    Create a MLP of size input_dim * hidden_sizes[0] * hidden_sizes[1] * ...
    * hidden_sizes[-1] * output_dim

    :param input_dim: dimension of the input vector.
    :param output_dim: dimension of the output vector. If set to 0, there
        is no final linear layer.
    :param hidden_sizes: shape of MLP passed in as a list, not including
        input_dim and output_dim.
    :param norm_layer: use which normalization before activation, e.g.,
        ``nn.LayerNorm`` and ``nn.BatchNorm1d``. Default to no normalization.
        You can also pass a list of normalization modules with the same length
        of hidden_sizes, to use different normalization module in different
        layers. Default to no normalization.
    :param activation: which activation to use after each layer, can be both
        the same activation for all layers if passed in nn.Module, or different
        activation for different Modules if passed in a list. Default to
        nn.ReLU.
    :param device: which device to create this model on. Default to None.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param flatten_input: whether to flatten input data. Default to True.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        hidden_sizes: Sequence[int] = (),
        norm_layer: ModuleType | Sequence[ModuleType] | None = None,  # 使用ModuleType作为类型注解意味着传入的参数不是一个示例，而是一个类的类型，比如直接传入nn.Module就不是一个示例，而是一个类类型。
        norm_args: ArgsType | None = None,
        activation: ModuleType | Sequence[ModuleType] | None = nn.ReLU,  # 有默认值，就增加None
        act_args: ArgsType | None = None,
        device: str | int | torch.device | None = None,  # 暂时不关注这个
        linear_layer: TLinearLayer = nn.Linear,
        flatten_input: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        if norm_layer:  # 首先判断是不是list
            if isinstance(norm_layer, list):
                assert len(norm_layer) == len(hidden_sizes)
                norm_layer_list = norm_layer
                if isinstance(norm_args, list):
                    assert len(norm_args) == len(hidden_sizes)
                    norm_args_list = norm_args
                else:
                    norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
            else:
                norm_layer_list = [norm_layer for _ in range(len(hidden_sizes))]
                norm_args_list = [norm_args for _ in range(len(hidden_sizes))]
        else:
            norm_layer_list = [None] * len(hidden_sizes)
            norm_args_list = [None] * len(hidden_sizes)
        if activation:
            if isinstance(activation, list):
                assert len(activation) == len(hidden_sizes)
                activation_list = activation
                if isinstance(act_args, list):
                    assert len(act_args) == len(hidden_sizes)
                    act_args_list = act_args
                else:
                    act_args_list = [act_args for _ in range(len(hidden_sizes))]
            else:
                activation_list = [activation for _ in range(len(hidden_sizes))]
                act_args_list = [act_args for _ in range(len(hidden_sizes))]
        else:
            activation_list = [None] * len(hidden_sizes)
            act_args_list = [None] * len(hidden_sizes)
        hidden_sizes = [input_dim, *list(hidden_sizes)]
        model = []
        for in_dim, out_dim, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
            strict=True,
        ):
            model += miniblock(in_dim, out_dim, norm, norm_args, activ, act_args, linear_layer)
        if output_dim > 0:
            model += [linear_layer(hidden_sizes[-1], output_dim)]
        self.output_dim = output_dim or hidden_sizes[-1]
        self.model = nn.Sequential(*model)
        self.flatten_input = flatten_input

    @no_type_check
    def forward(self, obs: np.ndarray | torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if self.flatten_input:
            obs = obs.flatten(1)
        return self.model(obs)
```

MLP一个网络的实现，适用性很强，建议死记，经典永流传。

### 类型注解

```python
from typing import Any
from collections.abc import Sequence  
ModuleType = type[nn.Module]  



tuple[Any, ...]  # 任意类型，任意长度的元组
dict[Any, Any]  # 任意类型键值对的
Sequence[int]  # 某个类型的list
classtype = type[aclass]  # 某个类的类型
ArgsType = tuple[Any, ...] | dict[Any, Any] | Sequence[tuple[Any, ...]] | Sequence[dict[Any, Any]]  # 从一些类型中挑选
TLinearLayer: TypeAlias = Callable[[int, int], nn.Module]  # TypeAlias表示类型别名，在这里指Callable[[int, int], nn.Module]类型的别名是TLinearLayer，其中Callable[[int, int], nn.Module]表示输入两个int类型，返回nn.Module类型的可调用对象，就是指nn.Linear
```

