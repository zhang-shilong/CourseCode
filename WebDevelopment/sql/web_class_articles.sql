create table articles
(
    id            int unsigned auto_increment
        primary key,
    username      text     null,
    title         text     null,
    markdown_text text     null,
    html_text     text     null,
    read_time     int      null,
    publish_time  datetime null
);

INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (1, 'zsl', 'Welcome', '## Welcome

这里是 Articles 页面。您可以在左边选择文章，或登录后点击 New Article 添加新文章。
', '&lt;h2 id=&quot;h2-welcome&quot;&gt;&lt;a name=&quot;Welcome&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Welcome&lt;/h2&gt;&lt;p&gt;这里是 Articles 页面。您可以在左边选择文章，或登录后点击 New Article 添加新文章。&lt;/p&gt;
', 0, '2022-06-23 18:38:22');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (15, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (16, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (18, 'zsl', 'Corpus annotation for mining biomedical events from literature', '# Corpus annotation for mining biomedical events from literature

从文献中挖掘生物医学事件的语料库注释

&gt; Kim, JD., Ohta, T. &amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. *BMC Bioinformatics* **9,** 10 (2008). https://doi.org/10.1186/1471-2105-9-10

## GENIA 本体

GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。

GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 

## 事件注释的例子

使用 GENIA 本体对下面这句话进行事件注释（event annotation）。

The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.

第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：

```
(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
```

下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：

```
(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
```

事件中的 Theme 是一个属性或实体，由**属性受事件影响**的一个或多个实体填充。

在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。

## GENIA 本体和 Gene Ontology

GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。

', '&lt;h1 id=&quot;h1-corpus-annotation-for-mining-biomedical-events-from-literature&quot;&gt;&lt;a name=&quot;Corpus annotation for mining biomedical events from literature&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Corpus annotation for mining biomedical events from literature&lt;/h1&gt;&lt;p&gt;从文献中挖掘生物医学事件的语料库注释&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;Kim, JD., Ohta, T. &amp;amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. &lt;em&gt;BMC Bioinformatics&lt;/em&gt; &lt;strong&gt;9,&lt;/strong&gt; 10 (2008). &lt;a href=&quot;https://doi.org/10.1186/1471-2105-9-10&quot;&gt;https://doi.org/10.1186/1471-2105-9-10&lt;/a&gt;&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-genia-&quot;&gt;&lt;a name=&quot;GENIA 本体&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体&lt;/h2&gt;&lt;p&gt;GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。&lt;/p&gt;
&lt;p&gt;GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 &lt;/p&gt;
&lt;h2 id=&quot;h2-u4E8Bu4EF6u6CE8u91CAu7684u4F8Bu5B50&quot;&gt;&lt;a name=&quot;事件注释的例子&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;事件注释的例子&lt;/h2&gt;&lt;p&gt;使用 GENIA 本体对下面这句话进行事件注释（event annotation）。&lt;/p&gt;
&lt;p&gt;The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.&lt;/p&gt;
&lt;p&gt;第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;事件中的 Theme 是一个属性或实体，由&lt;strong&gt;属性受事件影响&lt;/strong&gt;的一个或多个实体填充。&lt;/p&gt;
&lt;p&gt;在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。&lt;/p&gt;
&lt;h2 id=&quot;h2-genia-gene-ontology&quot;&gt;&lt;a name=&quot;GENIA 本体和 Gene Ontology&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体和 Gene Ontology&lt;/h2&gt;&lt;p&gt;GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。&lt;/p&gt;
', 5, '2022-06-23 22:38:33');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (19, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (20, 'zsl', 'Untitled article 2022/06/23 23:25:31', '## 无题测试

当没有 `#` 一级标题时，自动命名标题为 Untitled article + 发表时间。', '&lt;h2 id=&quot;h2-u65E0u9898u6D4Bu8BD5&quot;&gt;&lt;a name=&quot;无题测试&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;无题测试&lt;/h2&gt;&lt;p&gt;当没有 &lt;code&gt;#&lt;/code&gt; 一级标题时，自动命名标题为 Untitled article + 发表时间。&lt;/p&gt;
', 0, '2022-06-23 23:25:31');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (22, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (23, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (24, 'zsl', 'Corpus annotation for mining biomedical events from literature', '# Corpus annotation for mining biomedical events from literature

从文献中挖掘生物医学事件的语料库注释

&gt; Kim, JD., Ohta, T. &amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. *BMC Bioinformatics* **9,** 10 (2008). https://doi.org/10.1186/1471-2105-9-10

## GENIA 本体

GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。

GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 

## 事件注释的例子

使用 GENIA 本体对下面这句话进行事件注释（event annotation）。

The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.

第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：

```
(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
```

下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：

```
(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
```

事件中的 Theme 是一个属性或实体，由**属性受事件影响**的一个或多个实体填充。

在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。

## GENIA 本体和 Gene Ontology

GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。

', '&lt;h1 id=&quot;h1-corpus-annotation-for-mining-biomedical-events-from-literature&quot;&gt;&lt;a name=&quot;Corpus annotation for mining biomedical events from literature&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Corpus annotation for mining biomedical events from literature&lt;/h1&gt;&lt;p&gt;从文献中挖掘生物医学事件的语料库注释&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;Kim, JD., Ohta, T. &amp;amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. &lt;em&gt;BMC Bioinformatics&lt;/em&gt; &lt;strong&gt;9,&lt;/strong&gt; 10 (2008). &lt;a href=&quot;https://doi.org/10.1186/1471-2105-9-10&quot;&gt;https://doi.org/10.1186/1471-2105-9-10&lt;/a&gt;&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-genia-&quot;&gt;&lt;a name=&quot;GENIA 本体&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体&lt;/h2&gt;&lt;p&gt;GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。&lt;/p&gt;
&lt;p&gt;GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 &lt;/p&gt;
&lt;h2 id=&quot;h2-u4E8Bu4EF6u6CE8u91CAu7684u4F8Bu5B50&quot;&gt;&lt;a name=&quot;事件注释的例子&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;事件注释的例子&lt;/h2&gt;&lt;p&gt;使用 GENIA 本体对下面这句话进行事件注释（event annotation）。&lt;/p&gt;
&lt;p&gt;The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.&lt;/p&gt;
&lt;p&gt;第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;事件中的 Theme 是一个属性或实体，由&lt;strong&gt;属性受事件影响&lt;/strong&gt;的一个或多个实体填充。&lt;/p&gt;
&lt;p&gt;在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。&lt;/p&gt;
&lt;h2 id=&quot;h2-genia-gene-ontology&quot;&gt;&lt;a name=&quot;GENIA 本体和 Gene Ontology&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体和 Gene Ontology&lt;/h2&gt;&lt;p&gt;GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。&lt;/p&gt;
', 5, '2022-06-23 22:38:33');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (25, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (26, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (27, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (28, 'zsl', 'Corpus annotation for mining biomedical events from literature', '# Corpus annotation for mining biomedical events from literature

从文献中挖掘生物医学事件的语料库注释

&gt; Kim, JD., Ohta, T. &amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. *BMC Bioinformatics* **9,** 10 (2008). https://doi.org/10.1186/1471-2105-9-10

## GENIA 本体

GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。

GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 

## 事件注释的例子

使用 GENIA 本体对下面这句话进行事件注释（event annotation）。

The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.

第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：

```
(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
```

下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：

```
(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
```

事件中的 Theme 是一个属性或实体，由**属性受事件影响**的一个或多个实体填充。

在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。

## GENIA 本体和 Gene Ontology

GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。

', '&lt;h1 id=&quot;h1-corpus-annotation-for-mining-biomedical-events-from-literature&quot;&gt;&lt;a name=&quot;Corpus annotation for mining biomedical events from literature&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Corpus annotation for mining biomedical events from literature&lt;/h1&gt;&lt;p&gt;从文献中挖掘生物医学事件的语料库注释&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;Kim, JD., Ohta, T. &amp;amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. &lt;em&gt;BMC Bioinformatics&lt;/em&gt; &lt;strong&gt;9,&lt;/strong&gt; 10 (2008). &lt;a href=&quot;https://doi.org/10.1186/1471-2105-9-10&quot;&gt;https://doi.org/10.1186/1471-2105-9-10&lt;/a&gt;&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-genia-&quot;&gt;&lt;a name=&quot;GENIA 本体&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体&lt;/h2&gt;&lt;p&gt;GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。&lt;/p&gt;
&lt;p&gt;GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 &lt;/p&gt;
&lt;h2 id=&quot;h2-u4E8Bu4EF6u6CE8u91CAu7684u4F8Bu5B50&quot;&gt;&lt;a name=&quot;事件注释的例子&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;事件注释的例子&lt;/h2&gt;&lt;p&gt;使用 GENIA 本体对下面这句话进行事件注释（event annotation）。&lt;/p&gt;
&lt;p&gt;The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.&lt;/p&gt;
&lt;p&gt;第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;事件中的 Theme 是一个属性或实体，由&lt;strong&gt;属性受事件影响&lt;/strong&gt;的一个或多个实体填充。&lt;/p&gt;
&lt;p&gt;在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。&lt;/p&gt;
&lt;h2 id=&quot;h2-genia-gene-ontology&quot;&gt;&lt;a name=&quot;GENIA 本体和 Gene Ontology&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体和 Gene Ontology&lt;/h2&gt;&lt;p&gt;GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。&lt;/p&gt;
', 5, '2022-06-23 22:38:33');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (29, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (30, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (31, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (32, 'zsl', 'Corpus annotation for mining biomedical events from literature', '# Corpus annotation for mining biomedical events from literature

从文献中挖掘生物医学事件的语料库注释

&gt; Kim, JD., Ohta, T. &amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. *BMC Bioinformatics* **9,** 10 (2008). https://doi.org/10.1186/1471-2105-9-10

## GENIA 本体

GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。

GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 

## 事件注释的例子

使用 GENIA 本体对下面这句话进行事件注释（event annotation）。

The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.

第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：

```
(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
```

下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：

```
(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
```

事件中的 Theme 是一个属性或实体，由**属性受事件影响**的一个或多个实体填充。

在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。

## GENIA 本体和 Gene Ontology

GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。

', '&lt;h1 id=&quot;h1-corpus-annotation-for-mining-biomedical-events-from-literature&quot;&gt;&lt;a name=&quot;Corpus annotation for mining biomedical events from literature&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Corpus annotation for mining biomedical events from literature&lt;/h1&gt;&lt;p&gt;从文献中挖掘生物医学事件的语料库注释&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;Kim, JD., Ohta, T. &amp;amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. &lt;em&gt;BMC Bioinformatics&lt;/em&gt; &lt;strong&gt;9,&lt;/strong&gt; 10 (2008). &lt;a href=&quot;https://doi.org/10.1186/1471-2105-9-10&quot;&gt;https://doi.org/10.1186/1471-2105-9-10&lt;/a&gt;&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-genia-&quot;&gt;&lt;a name=&quot;GENIA 本体&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体&lt;/h2&gt;&lt;p&gt;GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。&lt;/p&gt;
&lt;p&gt;GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 &lt;/p&gt;
&lt;h2 id=&quot;h2-u4E8Bu4EF6u6CE8u91CAu7684u4F8Bu5B50&quot;&gt;&lt;a name=&quot;事件注释的例子&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;事件注释的例子&lt;/h2&gt;&lt;p&gt;使用 GENIA 本体对下面这句话进行事件注释（event annotation）。&lt;/p&gt;
&lt;p&gt;The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.&lt;/p&gt;
&lt;p&gt;第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;事件中的 Theme 是一个属性或实体，由&lt;strong&gt;属性受事件影响&lt;/strong&gt;的一个或多个实体填充。&lt;/p&gt;
&lt;p&gt;在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。&lt;/p&gt;
&lt;h2 id=&quot;h2-genia-gene-ontology&quot;&gt;&lt;a name=&quot;GENIA 本体和 Gene Ontology&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体和 Gene Ontology&lt;/h2&gt;&lt;p&gt;GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。&lt;/p&gt;
', 5, '2022-06-23 22:38:33');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (33, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (34, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (35, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (36, 'zsl', 'Corpus annotation for mining biomedical events from literature', '# Corpus annotation for mining biomedical events from literature

从文献中挖掘生物医学事件的语料库注释

&gt; Kim, JD., Ohta, T. &amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. *BMC Bioinformatics* **9,** 10 (2008). https://doi.org/10.1186/1471-2105-9-10

## GENIA 本体

GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。

GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 

## 事件注释的例子

使用 GENIA 本体对下面这句话进行事件注释（event annotation）。

The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.

第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：

```
(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
```

下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：

```
(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
```

事件中的 Theme 是一个属性或实体，由**属性受事件影响**的一个或多个实体填充。

在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。

## GENIA 本体和 Gene Ontology

GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。

', '&lt;h1 id=&quot;h1-corpus-annotation-for-mining-biomedical-events-from-literature&quot;&gt;&lt;a name=&quot;Corpus annotation for mining biomedical events from literature&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Corpus annotation for mining biomedical events from literature&lt;/h1&gt;&lt;p&gt;从文献中挖掘生物医学事件的语料库注释&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;Kim, JD., Ohta, T. &amp;amp; Tsujii, J. Corpus annotation for mining biomedical events from literature. &lt;em&gt;BMC Bioinformatics&lt;/em&gt; &lt;strong&gt;9,&lt;/strong&gt; 10 (2008). &lt;a href=&quot;https://doi.org/10.1186/1471-2105-9-10&quot;&gt;https://doi.org/10.1186/1471-2105-9-10&lt;/a&gt;&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-genia-&quot;&gt;&lt;a name=&quot;GENIA 本体&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体&lt;/h2&gt;&lt;p&gt;GENIA 事件注释依赖于两种本体：事件本体（event ontology）和术语本体（term ontology）。&lt;/p&gt;
&lt;p&gt;GENIA 事件本体定义并分类了 GENIA 领域感兴趣的事件，而 GENIA 术语本体定义了导致或贯穿事件的事物。 粗略地说，事件本体为谓词（例如，“结合”、“磷酸化”等）提供词汇表，而术语本体则是用来描述事件的参数（如蛋白质）。 &lt;/p&gt;
&lt;h2 id=&quot;h2-u4E8Bu4EF6u6CE8u91CAu7684u4F8Bu5B50&quot;&gt;&lt;a name=&quot;事件注释的例子&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;事件注释的例子&lt;/h2&gt;&lt;p&gt;使用 GENIA 本体对下面这句话进行事件注释（event annotation）。&lt;/p&gt;
&lt;p&gt;The binding of I kappa B/MAD-3 to NF-kappa B p65 is sufficient to retarget NF-kappa B p65 from the nucleus to the cytoplasm.&lt;/p&gt;
&lt;p&gt;第一个 box 中是对原句进行术语注释（term annotation）的结果，其中蓝色表示蛋白质分子，绿色表示细胞组件。每一个术语可以表示为一个键值对的 n 元组：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: T36, Class: Protein_molecule, Name: I kappa B/MAD-3)
(Id: T37, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T38, Class: Protein_molecule, Name: NF-kappa B p65)
(Id: T39, Class: Cell_component, Name: nucleus)
(Id: T40, Class: Cell_component, Name: cytoplasm)
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;下面三个 box 是对术语注释进行事件注释的结果。浅红色表示注释者用作绑定事件的文本证据的线索，我们的注释原则之一要求每个事件都有这样一个线索词的支持。黄色表示其他支持词。每一个事件也有一个唯一的 ID：&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(Id: E5, Class: Binding, ClueType: binding, Theme: T36, Theme: T37)
(Id: E6, Class: Localization, ClueType: retarget, Theme: T38, ClueGoal: T40)
(Id: E7, Class: Positive_regulation,
  ClueType: is sufficient to,
  Theme: E6 (Localization, Theme: T38),
  Cause: E5 (Binding, Theme: T36, Theme: T37))
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;事件中的 Theme 是一个属性或实体，由&lt;strong&gt;属性受事件影响&lt;/strong&gt;的一个或多个实体填充。&lt;/p&gt;
&lt;p&gt;在这种表示中，来自 GENIA 术语本体的实体显示在矩形框中，而来自GENIA 事件本体的实体显示在圆圈中。黑色、红色和蓝色箭头分别表示事件与其主题（theme）、原因（cause）和地点（location）之间的联系。&lt;/p&gt;
&lt;h2 id=&quot;h2-genia-gene-ontology&quot;&gt;&lt;a name=&quot;GENIA 本体和 Gene Ontology&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;GENIA 本体和 Gene Ontology&lt;/h2&gt;&lt;p&gt;GENIA 事件本体的多数实体取自于 GO，但由于 GENIA 感兴趣的领域比 GO 窄得多，所以只提取了 GO 中的一个子集。GENIA 中还添加了三个事件类：Gene Expression、Artificial Process 和 Correlation，以扩展 GO 无法表示的事件。&lt;/p&gt;
', 5, '2022-06-23 22:38:33');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (37, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (38, 'zsl', 'Linux 系统笔记', '# Linux 系统笔记

## 进程控制

### 运行后台进程

为了在后台运行进程，需要在命令最后添加 `&amp;` 符号：

```bash
command &amp;
```

这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：

```text
[1] 28
```

后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：

```bash
command &gt; /dev/null 2&gt;&amp;1 &amp;
command 1&gt; output 2&gt;log &amp;
```

### 前后台进程控制

使用 `jobs` 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。

如果需要把后台进程转移到前台，需要使用 `fg` 命令。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
fg %1
```

如果需要终止进程，使用 `kill` 命令，以进程 id 作为参数：

```bash
kill -9 28
```

如果需要把前台进程转移到后台，需要两步操作：

1. 通过按键 `Ctrl + Z` 暂停前台进程。
2. 输入命令 `bg` 将进程转移到后台。

### 保持后台进程运行

如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。

方法一是从 Shell 进程控制中移除任务，使用命令 `disown`。如果有多个任务在后台，需要加上 “`%` + 任务 id”作为参数。

```bash
disown %1
```

通过使用 `jobs -l` 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 `ps aux` 命令。

方法二是忽略挂起信号，使用命令 `nohup`。`nohup` 命令后面跟着另外一个程序作为参数，将会忽略所有的`SIGHUP`（挂起）信号。`SIGHUP` 信号用来发送给进程，用来通知终端已经关闭了。

使用 `nohup` 命令来在后台运行命令，需要输入:

```bash
nohup command &amp;
```

如果没有指定重定向输出，命令输出将会重定向到 `nohup.out` 文件。

```text
nohup: ignoring input and appending output to &#039;nohup.out&#039;
```

如果登出或者关闭终端，进程不会被终止。

', '&lt;h1 id=&quot;h1-linux-&quot;&gt;&lt;a name=&quot;Linux 系统笔记&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Linux 系统笔记&lt;/h1&gt;&lt;h2 id=&quot;h2-u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;进程控制&lt;/h2&gt;&lt;h3 id=&quot;h3-u8FD0u884Cu540Eu53F0u8FDBu7A0B&quot;&gt;&lt;a name=&quot;运行后台进程&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;运行后台进程&lt;/h3&gt;&lt;p&gt;为了在后台运行进程，需要在命令最后添加 &lt;code&gt;&amp;amp;&lt;/code&gt; 符号：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;这个 Shell 进程的任务 id（job id）和进程 id 将被打印在终端：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;[1] 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;后台进程将会不断地将信息打印在终端上，为了禁止标准输入和标准输出，需要将输出重定向：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;command &amp;gt; /dev/null 2&amp;gt;&amp;amp;1 &amp;amp;
command 1&amp;gt; output 2&amp;gt;log &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u524Du540Eu53F0u8FDBu7A0Bu63A7u5236&quot;&gt;&lt;a name=&quot;前后台进程控制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;前后台进程控制&lt;/h3&gt;&lt;p&gt;使用 &lt;code&gt;jobs&lt;/code&gt; 命令显示当前 Shell 会话中所有停止的和后台的任务状态，输出包括任务 id、进程 id、任务状态和启动任务的命令。&lt;/p&gt;
&lt;p&gt;如果需要把后台进程转移到前台，需要使用 &lt;code&gt;fg&lt;/code&gt; 命令。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;fg %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要终止进程，使用 &lt;code&gt;kill&lt;/code&gt; 命令，以进程 id 作为参数：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;kill -9 28
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要把前台进程转移到后台，需要两步操作：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;通过按键 &lt;code&gt;Ctrl + Z&lt;/code&gt; 暂停前台进程。&lt;/li&gt;&lt;li&gt;输入命令 &lt;code&gt;bg&lt;/code&gt; 将进程转移到后台。&lt;/li&gt;&lt;/ol&gt;
&lt;h3 id=&quot;h3-u4FDDu6301u540Eu53F0u8FDBu7A0Bu8FD0u884C&quot;&gt;&lt;a name=&quot;保持后台进程运行&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;保持后台进程运行&lt;/h3&gt;&lt;p&gt;如果你丢失了连接，或者你退出了 Shell 会话，后台进程将会被终止。有很多方法可以保证进程运行直到交互的 Shell 结束。&lt;/p&gt;
&lt;p&gt;方法一是从 Shell 进程控制中移除任务，使用命令 &lt;code&gt;disown&lt;/code&gt;。如果有多个任务在后台，需要加上 “&lt;code&gt;%&lt;/code&gt; + 任务 id”作为参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;disown %1
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;通过使用 &lt;code&gt;jobs -l&lt;/code&gt; 来确认任务已经从任务表中移除。为了列出正在运行的进程，请使用 &lt;code&gt;ps aux&lt;/code&gt; 命令。&lt;/p&gt;
&lt;p&gt;方法二是忽略挂起信号，使用命令 &lt;code&gt;nohup&lt;/code&gt;。&lt;code&gt;nohup&lt;/code&gt; 命令后面跟着另外一个程序作为参数，将会忽略所有的&lt;code&gt;SIGHUP&lt;/code&gt;（挂起）信号。&lt;code&gt;SIGHUP&lt;/code&gt; 信号用来发送给进程，用来通知终端已经关闭了。&lt;/p&gt;
&lt;p&gt;使用 &lt;code&gt;nohup&lt;/code&gt; 命令来在后台运行命令，需要输入:&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-bash&quot;&gt;nohup command &amp;amp;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果没有指定重定向输出，命令输出将会重定向到 &lt;code&gt;nohup.out&lt;/code&gt; 文件。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-text&quot;&gt;nohup: ignoring input and appending output to &amp;#39;nohup.out&amp;#39;
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果登出或者关闭终端，进程不会被终止。&lt;/p&gt;
', 4, '2022-06-23 22:24:11');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (39, 'zsl', 'PyTorch 深度学习实战', '# PyTorch 深度学习实战

```bibtex
@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
```

## 张量

张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。

### 构造和索引张量

```python
torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
```

```python
a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
```

### tensor 的属性

- `ndim`
- `shape`
- `dtype`
- `deivce`
- `data`
- `grad`
- `grad_fn`

**张量的元素类型**

张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。

- `torch.float32` / `torch.float`：32 位浮点数；
- `torch.float64` / `torch.double`：64 位双精度浮点数；
- `torch.float16` / `torch.half`：16 位半精度浮点数；
- `torch.int8`：8 位有符号整数；
- `torch.uint8`：8 位无符号整数；
- `torch.int16` / `torch.short`：16 位有符号整数；
- `torch.int32` / `torch.int`：32 位有符号整数；
- `torch.int64` / `torch.long`：64 位有符号整数；
- `torch.bool`：布尔型。

张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。

张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。

元素类型可以通过相应的方法转换：

```python
double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
```

两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。

PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。

```python
points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
```

**张量的存储位置**

张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 `to()` 更改存储位置。

```python
points.to(device=&quot;cpu&quot;)
points.cpu()

points.to(device=&quot;cuda&quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&quot;cuda:0&quot;)
points.cuda(0)
```

如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 `torch.cuda.is_available()` 设置变量 device 的值。

```python
device = torch.device(&quot;cuda&quot;) if touch.cuda.is_available() else torch.device(&quot;cpu&quot;)
```

### tensor 相关的函数

**tensor.size()**

查看张量的大小，等同于 `tensor.shape`。

**tensor.transpose()**

置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 `t()`。

```python
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
```

**tensor.permute()**

改变张量维度的次序。

**tensor.is_contiguous()**

检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。

**tensor.contiguous()**

返回新的连续张量。

**tensor.unsqueeze()**

张量升维函数。参数表示在哪个地方加一个维度。

```python
input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
```

**就地操作方法**

以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 `zero_()`，`scatter_()` 等。

**torch.scatter()**

生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：

- **dim**：沿着哪个维度进行索引；
- **index**：用来 scatter 的元素索引；
- **src**：用来 scatter 的源元素，可以是一个标量或一个张量。

```python
target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
```

**torch.randperm()**

将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。

```python
n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
```

### 张量的存储视图

PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
```

可以索引存储区取得，且修改存储区实例会对原张量产生修改。

```python
points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
```

子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。

### 序列化张量

可以通过 `torch.save()` 将张量保存到文件，通常以 t 作为后缀名，再通过 `torch.load()` 读取。可以传递文件地址或文件描述符。

### 广播机制

如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。

如果遵守以下规则，则两个 tensor 是“可广播的”：

- 每个 tensor 至少有一个维度；
- 遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：
    - tensor 维度相等；
    - tensor 维度不等且其中一个维度为 1；
    - tensor 维度不等且其中一个维度不存在。

如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：

- 如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；
- 两个 tensor 扩展维度的过程是将数值进行复制。

## 人工神经网络

神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以**权重**（weight），加上一个常数**偏置**（bias）），然后应用一个固定的非线性函数，即**激活函数**。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。

### 激活函数

激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：

- 在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。
- 在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。

激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。

激活函数主要分为**饱和激活函数**（Saturated Neurons）和**非饱和激活函数**（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：

1.  非饱和激活函数可以解决梯度消失问题；
2.  非饱和激活函数可以加速收敛。

#### Sigmoid 与梯度消失

Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。

Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。

#### ReLU 与死亡神经元

ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。

Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。**用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。**

ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，**该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。**

Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。

使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了**梯度方向锯齿问题**。

#### 理想的激活函数

理想的激活函数应满足两个条件：

1.  输出的分布是零均值的，可以加快训练速度；
2.  激活函数是单侧饱和的，可以更好的收敛。

两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：

$$
egin{equation}
    f(x) =
    egin{cases}
        x, &amp; x gt 0 \\
        alpha (e^{x}-1), &amp; x le 0 \\
    end{cases}
end{equation}
$$

但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。

#### 梯度爆炸

梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。

如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。

### 反向传播

#### 自动求导

自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。

调用 `backward()` 实现自动求导，需要将被求导变量设置为 `requires_grad=True`。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
```

如果提示标量输出才能反向求导，需要为 `backward()` 添加一个 tensor 参数。

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
```

如果需要多次自动求导，需要为 `backward()` 添加参数 `retain_graph=True`。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。

```python
x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
```

#### 动态图

目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。

对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。

### train/eval 模式切换

`model.train()` 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 `model.eval()` 或 `model.train(mode=False)` 启用测试。

### 卷积神经网络（CNN）

#### 卷积

torch.nn 提供了一维、二维和三维的卷积，其中 `torch.nn.Conv1d()` 用于时间序列，`torch.nn.Conv2d()` 用于图像，`torch.nn.Conv3d()` 用于体数据和视频。

相比于线性变换，卷积操作为神经网络带来了以下特征：

- 邻域的局部操作；
- 平移不变性；
- 模型的参数大幅减少。

卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。

&gt; 无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。

假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：

$$OH = frac{H+2P-FH}{S}+1$$

$$OW = frac{W+2P-FW}{S}+1$$

卷积操作可以用于检测图像的特征。

```python
conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
```

使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。

#### 池化

我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。

下采样可选择的操作有：

- 平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；
- 最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；
- 使用带步幅的卷积，具有一定的应用前景。

torch.nn 提供了一维、二维和三维的池化操作，分别是 `torch.nn.MaxPool1d()`、`torch.nn.MaxPool2d()` 和 `torch.nn.MaxPool3d()`。

## torch.nn

torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。

### torch.nn.Sequential

torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。

调用 `model.parameters()` 将从内部模块收集权重和偏置，还可以使用 `model.named_parameters()` 方法获取非激活函数的模块信息。

```python
seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
```

torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。

```python
named_seq_model = nn.Sequential(OrderedDict([  
    (&quot;hidden_linear&quot;, nn.Linear(1, 8)),  
    (&quot;hidden_activation&quot;, nn.Tanh()),  
    (&quot;output_linear&quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
```

我们可以调用 `numel()` 方法，得到每个张量实例中元素的数量。

```python
numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
```

### torch.nn.Module

PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。

torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 `tanh()` 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 `max_pool2d()`）。

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)
        
    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&lt;AddmmBackward&gt;)
```

## torch.utils

### torch.utils.data.Dataset

我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。

torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。

要自定义自己的 Dataset 类，至少要重载两个方法，`__len__` 和 `__getitem__`，其中 `__len__` 返回的是数据集的大小，`__getitem__` 实现索引数据集中的某一个数据。

```python
class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &#039;tensor_data[0]: &#039;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &#039;len os tensor_dataset: &#039;, len(tensor_dataset)
# len os tensor_dataset:  4
```

### torch.utils.data.DataLoader

torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &#039;one batch tensor data: &#039;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &#039;len of batchtensor: &#039;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
```

## torch.optim

PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。

优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。

```python
optimizer = optim.SGD(model.parameters(), lr=1e-2)
```

在训练过程中先调用 `optimizer.zero_grad()` 清空梯度，再调用 `loss.backward()` 反向传播，最后调用 `optimizer.step()` 更新模型参数。

## 模型设计

### 增加模型参数

模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。

### 正则化

添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。

L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。

L2 正则化也称为**权重衰减**，L2 正则化对参数 $w_{i}$ 的负梯度为 $-2 	imes lambda 	imes w_{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。

在 PyTorch 中，L1 和 L2 正则化很容易实现。

```python
# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
```

torch.optim.SGD 优化器已存在 `weight_decay` 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。

### Dropout

Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。

Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。

在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。

### 批量归一化

批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。

PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。

### 跳跃连接

跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。

实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。
', '&lt;h1 id=&quot;h1-pytorch-&quot;&gt;&lt;a name=&quot;PyTorch 深度学习实战&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;PyTorch 深度学习实战&lt;/h1&gt;&lt;pre&gt;&lt;code class=&quot;lang-bibtex&quot;&gt;@book{stevens2020deep,
  title={Deep learning with PyTorch},
  author={Stevens, Eli and Antiga, Luca and Viehmann, Thomas},
  year={2020},
  publisher={Manning Publications}
}
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-u5F20u91CF&quot;&gt;&lt;a name=&quot;张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量&lt;/h2&gt;&lt;p&gt;张量是 PyTorch 中基本的数据结构。神将网络将张量作为输入，并生成张量作为输出，且内部和优化过程中的所有操作都是张量之间的操作。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6784u9020u548Cu7D22u5F15u5F20u91CF&quot;&gt;&lt;a name=&quot;构造和索引张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;构造和索引张量&lt;/h3&gt;&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;torch.tensor([1.0, 2.0, 3.0])
# tensor([1., 2., 3.])
torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
# tensor([[4., 1.],
#         [5., 3.],
#         [2., 1.]])
torch.zeros(3)
# tensor([0., 0., 0.])
torch.ones(3)
# tensor([1., 1., 1.])
&lt;/code&gt;&lt;/pre&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
a[1, 1]
# tensor(3.)
a[1:]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, :]
# tensor([[5., 3.],
#         [2., 1.]])
a[1:, 0]
# tensor([5., 2.])
a[None]  # 相当于 a.unsqueeze(0)
# tensor([[[4., 1.],
#          [5., 3.],
#          [2., 1.]]])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 的属性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 的属性&lt;/h3&gt;&lt;ul&gt;
&lt;li&gt;&lt;code&gt;ndim&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;shape&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;dtype&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;deivce&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;data&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad&lt;/code&gt;&lt;/li&gt;&lt;li&gt;&lt;code&gt;grad_fn&lt;/code&gt;&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;&lt;strong&gt;张量的元素类型&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量构造函数通过 dtype 参数指定包含在张量中的数字的数据类型。以下是 dtype 参数可能的取值。&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;code&gt;torch.float32&lt;/code&gt; / &lt;code&gt;torch.float&lt;/code&gt;：32 位浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float64&lt;/code&gt; / &lt;code&gt;torch.double&lt;/code&gt;：64 位双精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.float16&lt;/code&gt; / &lt;code&gt;torch.half&lt;/code&gt;：16 位半精度浮点数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int8&lt;/code&gt;：8 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.uint8&lt;/code&gt;：8 位无符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int16&lt;/code&gt; / &lt;code&gt;torch.short&lt;/code&gt;：16 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int32&lt;/code&gt; / &lt;code&gt;torch.int&lt;/code&gt;：32 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.int64&lt;/code&gt; / &lt;code&gt;torch.long&lt;/code&gt;：64 位有符号整数；&lt;/li&gt;&lt;li&gt;&lt;code&gt;torch.bool&lt;/code&gt;：布尔型。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;张量的默认数据类型是 32 位浮点数。在神经网络中发生的计算通常是用 32 位浮点精度执行的。采用更高的精度，如 64 位，并不会提高模型精度，反而需要更多的内存和计算时间。16 位半精度浮点数的数据类型在标准 CPU 中并不存在，而是由现代 GPU 提供的。如果需要的话，可以切换到半精度来减少神经网络占用的空间，这样做对精度的影响也很小。&lt;/p&gt;
&lt;p&gt;张量可以作为其他张量的索引，在这种情况下，PyTorch 期望索引张量为 64 位的整数。创建一个将整数作为参数的张量时，默认会创建一个 64 位的整数张量。因此，我们将把大部分时间用于处理 32 位浮点数和 64 位有符号整数。&lt;/p&gt;
&lt;p&gt;元素类型可以通过相应的方法转换：&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;double_points = torch.zeros(10, 2).double()
double_points = torch.zeros(10, 2).to(torch.double)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;两种方法是等价的，PyTorch 底层会检查转换是否有必要，如果必要，则执行转换。&lt;/p&gt;
&lt;p&gt;PyTorch 张量与 NumPy 数组共享底层缓冲区，二者具备互操作性。互操作基本上不需任何开销。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points_np = points.numpy()  # 返回一个 NumPy 数组
points = torch.from_numpy(points_np)  # 通过 NumPy 数组新建张量
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;张量的存储位置&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量可以存储在 CPU 和各种 GPU 上，通过构造张量时的 divice 参数确定，或通过 &lt;code&gt;to()&lt;/code&gt; 更改存储位置。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points.to(device=&amp;quot;cpu&amp;quot;)
points.cpu()

points.to(device=&amp;quot;cuda&amp;quot;)  # 默认为 GPU:0
points.cuda()

points.to(device=&amp;quot;cuda:0&amp;quot;)
points.cuda(0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果 GPU 可用，将所有张量移动到 GPU 是一个很好地选择。可以通过 &lt;code&gt;torch.cuda.is_available()&lt;/code&gt; 设置变量 device 的值。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;device = torch.device(&amp;quot;cuda&amp;quot;) if touch.cuda.is_available() else torch.device(&amp;quot;cpu&amp;quot;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-tensor-&quot;&gt;&lt;a name=&quot;tensor 相关的函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;tensor 相关的函数&lt;/h3&gt;&lt;p&gt;&lt;strong&gt;tensor.size()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;查看张量的大小，等同于 &lt;code&gt;tensor.shape&lt;/code&gt;。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.transpose()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;置换两个维度。可以作为单独的函数或张量实例的方法。二维张量的转置可简写为 &lt;code&gt;t()&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a_t = a.transpose(0, 1)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;tensor.permute()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;改变张量维度的次序。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.is_contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;检查张量是否是连续张量。一个张量的值在存储区中从最右的维度开始向前排列被定义为连续张量。连续张量效率较高，通过改进数据内部性可以提高性能。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.contiguous()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;返回新的连续张量。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;tensor.unsqueeze()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;张量升维函数。参数表示在哪个地方加一个维度。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;input = torch.arange(0, 6)
input.unsqueeze(0)
# tensor([[0, 1, 2, 3, 4, 5]])
input.unsqueeze(0).shape
# torch.Size([1, 6])
input.unsqueeze(1)
# tensor([[0],
#         [1],
#         [2],
#         [3],
#         [4],
#         [5]])
input.unsqueeze(1).shape
# torch.Size([6, 1])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;就地操作方法&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;以下划线结尾的方法是就地操作方法，可以直接对对象值产生改变，包括 &lt;code&gt;zero_()&lt;/code&gt;，&lt;code&gt;scatter_()&lt;/code&gt; 等。&lt;/p&gt;
&lt;p&gt;&lt;strong&gt;torch.scatter()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;生成张量的独热编码。scatter(dim, index, src) 的参数有 3 个：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;&lt;strong&gt;dim&lt;/strong&gt;：沿着哪个维度进行索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;index&lt;/strong&gt;：用来 scatter 的元素索引；&lt;/li&gt;&lt;li&gt;&lt;strong&gt;src&lt;/strong&gt;：用来 scatter 的源元素，可以是一个标量或一个张量。&lt;/li&gt;&lt;/ul&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;target_onehot = torch.zeros(target.shape[0], 10)
target_onthot.scatter_(1, target.unsqueeze(1), 1.0)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;&lt;strong&gt;torch.randperm()&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;将 0~n-1（包括 0 和 n-1）随机打乱后获得的数字序列，用于分割数据集。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;n_samples = data.shape[0]
shuffled_indices = torch.randperm(n_samples)
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-u5F20u91CFu7684u5B58u50A8u89C6u56FE&quot;&gt;&lt;a name=&quot;张量的存储视图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;张量的存储视图&lt;/h3&gt;&lt;p&gt;PyTorch 中将张量中的值分配到连续的内存块中，并由 torch.Storge 实例使用偏移量和每个维度的步长对存储区进行索引。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points_storge = points.storge()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;可以索引存储区取得，且修改存储区实例会对原张量产生修改。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;points = torch.tensor([[4.0, 1.0], [5.0, 3.0], [2.0, 1.0]])
points.storge_offset()  # 偏移量
# 0
points.stride()  # 步长
# (2, 1)
points.size()  # 大小
# torch.Size([3, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;子张量与原张量共享内存空间，仅通过偏移量、步长和大小获得新的张量实例。修改子张量时，原张量也会改变。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E8Fu5217u5316u5F20u91CF&quot;&gt;&lt;a name=&quot;序列化张量&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;序列化张量&lt;/h3&gt;&lt;p&gt;可以通过 &lt;code&gt;torch.save()&lt;/code&gt; 将张量保存到文件，通常以 t 作为后缀名，再通过 &lt;code&gt;torch.load()&lt;/code&gt; 读取。可以传递文件地址或文件描述符。&lt;/p&gt;
&lt;h3 id=&quot;h3-u5E7Fu64ADu673Au5236&quot;&gt;&lt;a name=&quot;广播机制&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;广播机制&lt;/h3&gt;&lt;p&gt;如果一个 PyTorch 操作支持广播，则其 tensor 参数可以自动扩展为相等大小（不需要复制数据）。通常情况下，小一点的数组会被 broadcast 到大一点的，这样才能保持大小一致。&lt;/p&gt;
&lt;p&gt;如果遵守以下规则，则两个 tensor 是“可广播的”：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;每个 tensor 至少有一个维度；&lt;/li&gt;&lt;li&gt;遍历 tensor 所有维度时，从末尾随开始遍历，两个 tensor 存在下列情况：&lt;ul&gt;
&lt;li&gt;tensor 维度相等；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度为 1；&lt;/li&gt;&lt;li&gt;tensor 维度不等且其中一个维度不存在。&lt;/li&gt;&lt;/ul&gt;
&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;如果两个 tensor 是“可广播的”，则计算过程遵循下列规则：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;如果两个 tensor 的维度不同，则在维度较小的 tensor 的前面增加维度，使它们维度相等。对于每个维度，计算结果的维度值取两个 tensor 中较大的那个值；&lt;/li&gt;&lt;li&gt;两个 tensor 扩展维度的过程是将数值进行复制。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-u4EBAu5DE5u795Eu7ECFu7F51u7EDC&quot;&gt;&lt;a name=&quot;人工神经网络&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;人工神经网络&lt;/h2&gt;&lt;p&gt;神经网络的基本构件是神经元，其核心就是输入的线性变换（将输入乘以&lt;strong&gt;权重&lt;/strong&gt;（weight），加上一个常数&lt;strong&gt;偏置&lt;/strong&gt;（bias）），然后应用一个固定的非线性函数，即&lt;strong&gt;激活函数&lt;/strong&gt;。神神经网络使我们能够在没有明确模型的情况下近似处理高度非线性的问题。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;激活函数&lt;/h3&gt;&lt;p&gt;激活函数是向神经网络中引入非线性因素，通过激活函数神经网络就可以拟合各种曲线。激活函数有两个作用：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;在模型的内部，激活函数允许输出函数在不同的值上有不同的斜率，这是线性函数无法做到的。&lt;/li&gt;&lt;li&gt;在网络的最后一层，激活函数的作用是将前面的线性运算的输出集中到给定的范围内。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;激活函数有很多种，包括平滑函数 tanh、Softplus 和 Sigmoid 等，非平滑函数 Hardtanh、ReLU 和 LeakyReLU 等。Sigmoid 激活函数曾经被广泛使用，现在已经不常用了，除非明确希望将输出规范到 0~1。ReLU 目前被认为是性能最好的通用激活函数之一。&lt;/p&gt;
&lt;p&gt;激活函数主要分为&lt;strong&gt;饱和激活函数&lt;/strong&gt;（Saturated Neurons）和&lt;strong&gt;非饱和激活函数&lt;/strong&gt;（One-sided Saturations）。Sigmoid 和 Tanh 是饱和激活函数，而 ReLU 以及其变种为非饱和激活函数。非饱和激活函数主要有如下优势：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;非饱和激活函数可以解决梯度消失问题；&lt;/li&gt;&lt;li&gt;非饱和激活函数可以加速收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;h4 id=&quot;h4-sigmoid-&quot;&gt;&lt;a name=&quot;Sigmoid 与梯度消失&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Sigmoid 与梯度消失&lt;/h4&gt;&lt;p&gt;Sigmoid 极容易导致梯度消失问题。饱和神经元会使得梯度消失问题雪上加霜，假设神经元输入 Sigmoid 的值特别大或特别小，对应的梯度约等于 0，即使从上一步传导来的梯度较大，该神经元权重和偏置的梯度也会趋近于 0，导致参数无法得到有效更新。&lt;/p&gt;
&lt;p&gt;Sigmoid 激活函数不是关于原点中心对称的（zero-centered），而 Tanh 解决了这一问题。但二者都容易产生梯度消失的问题。&lt;/p&gt;
&lt;h4 id=&quot;h4-relu-&quot;&gt;&lt;a name=&quot;ReLU 与死亡神经元&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;ReLU 与死亡神经元&lt;/h4&gt;&lt;p&gt;ReLU 激活函数的提出就是为了解决梯度消失问题。ReLU 的梯度只可以取两个值：0 或 1，当输入小于 0 时，梯度为 0；当输入大于 0 时，梯度为 1。好处就是：ReLU 的梯度的连乘不会收敛到 0，连乘的结果也只可以取两个值：0 或 1 。如果值为 1，梯度保持值不变进行前向传播；如果值为 0 ,梯度从该位置停止前向传播。&lt;/p&gt;
&lt;p&gt;Sigmoid 函数是双侧饱和的，即朝着正负两个方向函数值都会饱和；但 ReLU 函数是单侧饱和的，即只有朝着负方向，函数值才会饱和。&lt;strong&gt;用一个常量值 0 来表示检测不到特征是更为合理的，像 ReLU 这样单侧饱和的神经元就满足要求。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;ReLU 尽管稀疏性可以提升计算高效性，但同样也可能阻碍训练过程。通常，激活函数的输入值有一偏置项(bias)，假设 bias 变得太小，以至于输入激活函数的值总是负的，那么反向传播过程经过该处的梯度恒为 0,对应的权重和偏置参数此次无法得到更新。如果对于所有的样本输入，&lt;strong&gt;该激活函数的输入都是负的，那么该神经元再也无法学习，称为神经元”死亡“问题。&lt;/strong&gt;&lt;/p&gt;
&lt;p&gt;Leaky ReLU 的提出就是为了解决神经元”死亡“问题，Leaky ReLU 与 ReLU 很相似，仅在输入小于 0 的部分有差别，ReLU 输入小于 0 的部分值都为 0，而 Leaky ReLU 输入小于 0 的部分，值为负，且有微小的梯度。&lt;/p&gt;
&lt;p&gt;使用 Leaky ReLU 的好处就是：在反向传播过程中，对于 Leaky ReLU 激活函数输入小于零的部分，也可以计算得到梯度，这样就避免了&lt;strong&gt;梯度方向锯齿问题&lt;/strong&gt;。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7406u60F3u7684u6FC0u6D3Bu51FDu6570&quot;&gt;&lt;a name=&quot;理想的激活函数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;理想的激活函数&lt;/h4&gt;&lt;p&gt;理想的激活函数应满足两个条件：&lt;/p&gt;
&lt;ol&gt;
&lt;li&gt;输出的分布是零均值的，可以加快训练速度；&lt;/li&gt;&lt;li&gt;激活函数是单侧饱和的，可以更好的收敛。&lt;/li&gt;&lt;/ol&gt;
&lt;p&gt;两个条件都满足的激活函数为 ELU（Exponential Linear Unit），其表达式如下：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;&lt;br&gt;egin{equation}&lt;br&gt;    f(x) =&lt;br&gt;    egin{cases}&lt;br&gt;        x, &amp;amp; x gt 0 &lt;br&gt;        alpha (e^{x}-1), &amp;amp; x le 0 &lt;br&gt;    end{cases}&lt;br&gt;end{equation}&lt;br&gt;&lt;/p&gt;
&lt;p&gt;但由于 ELU 的计算量较大，仍应优先选择 ReLU 和 Leaky ReLU。&lt;/p&gt;
&lt;h4 id=&quot;h4-u68AFu5EA6u7206u70B8&quot;&gt;&lt;a name=&quot;梯度爆炸&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;梯度爆炸&lt;/h4&gt;&lt;p&gt;梯度误差是在神经网络训练期间计算的方向和梯度，神经网络以正确的方向和数值更新网络权重。在深度网络或递归神经网络中，梯度误差可能在更新过程中累积，造成非常大的梯度。这反过来会导致网络权重的大量更新，进而导致网络不稳定。在极端情况下，权重值可能变得太大，以至于溢出并导致 NaN 值现成梯度爆炸现象。&lt;/p&gt;
&lt;p&gt;如果发生梯度爆炸的问题，可以选择减少网络层数和减小 batch size，使用 L1 和 L2 正则化，对权重大小设置阈值等方法。&lt;/p&gt;
&lt;h3 id=&quot;h3-u53CDu5411u4F20u64AD&quot;&gt;&lt;a name=&quot;反向传播&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;反向传播&lt;/h3&gt;&lt;h4 id=&quot;h4-u81EAu52A8u6C42u5BFC&quot;&gt;&lt;a name=&quot;自动求导&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;自动求导&lt;/h4&gt;&lt;p&gt;自动求导是 PyTorch 中非常重要的特性，能够让我们避免手动去计算非常复杂的导数，这能够极大地减少了我们构建模型的时间。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;backward()&lt;/code&gt; 实现自动求导，需要将被求导变量设置为 &lt;code&gt;requires_grad=True&lt;/code&gt;。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward()
x.grad
# tensor([2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果提示标量输出才能反向求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加一个 tensor 参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([1., 2., 3.], requires_grad=True)
y = x * 2
y.backward(torch.ones_like(x))
x.grad
# tensor([2., 2., 2.])
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;如果需要多次自动求导，需要为 &lt;code&gt;backward()&lt;/code&gt; 添加参数 &lt;code&gt;retain_graph=True&lt;/code&gt;。这是由于为了节省内存，PyTorch 会在计算图计算完成后丢弃。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;x = torch.tensor([3.], requires_grad=True)
y = x * 2
y.backward(retain_graph=True)
x.grad
# tensor([2.])
y.backward()
x.grad
# tensor([4.])
&lt;/code&gt;&lt;/pre&gt;
&lt;h4 id=&quot;h4-u52A8u6001u56FE&quot;&gt;&lt;a name=&quot;动态图&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;动态图&lt;/h4&gt;&lt;p&gt;目前神经网络框架分为静态图框架和动态图框架，PyTorch 和 TensorFlow、Caffe 等框架最大的区别就是他们拥有不同的计算图表现形式。 TensorFlow 使用静态图，这意味着我们先定义计算图，然后不断使用它，而在 PyTorch 中，每次都会重新构建一个新的计算图。&lt;/p&gt;
&lt;p&gt;对于使用者来说，两种形式的计算图有着非常大的区别，同时静态图和动态图都有他们各自的优点，比如动态图比较方便 debug，使用者能够用任何他们喜欢的方式进行 debug，同时非常直观，而静态图是通过先定义后运行的方式，之后再次运行的时候就不再需要重新构建计算图，所以速度会比动态图更快。&lt;/p&gt;
&lt;h3 id=&quot;h3-train-eval-&quot;&gt;&lt;a name=&quot;train/eval 模式切换&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;train/eval 模式切换&lt;/h3&gt;&lt;p&gt;&lt;code&gt;model.train()&lt;/code&gt; 让你的模型知道现在正在训练。像 dropout、batchnorm 层在训练和测试时的作用不同，所以需要使它们运行在对应的模式中。调用 &lt;code&gt;model.eval()&lt;/code&gt; 或 &lt;code&gt;model.train(mode=False)&lt;/code&gt; 启用测试。&lt;/p&gt;
&lt;h3 id=&quot;h3--cnn-&quot;&gt;&lt;a name=&quot;卷积神经网络（CNN）&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积神经网络（CNN）&lt;/h3&gt;&lt;h4 id=&quot;h4-u5377u79EF&quot;&gt;&lt;a name=&quot;卷积&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;卷积&lt;/h4&gt;&lt;p&gt;torch.nn 提供了一维、二维和三维的卷积，其中 &lt;code&gt;torch.nn.Conv1d()&lt;/code&gt; 用于时间序列，&lt;code&gt;torch.nn.Conv2d()&lt;/code&gt; 用于图像，&lt;code&gt;torch.nn.Conv3d()&lt;/code&gt; 用于体数据和视频。&lt;/p&gt;
&lt;p&gt;相比于线性变换，卷积操作为神经网络带来了以下特征：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;邻域的局部操作；&lt;/li&gt;&lt;li&gt;平移不变性；&lt;/li&gt;&lt;li&gt;模型的参数大幅减少。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;卷积运算对输入数据应用滤波器（卷积核）。对于输入数据，卷积运算以一定间隔滑动滤波器的窗口并应用，然后将结果保存到输出的对应位置。应用滤波器的位置间隔称为步幅（stride）。在进行卷积层的处理之前，可能需要进行填充（padding）操作，在边界创建重影像素（ghost pixel）。若 stride=1 且 padding=1 时，输出图像与输入图像的大小完全相同。&lt;/p&gt;
&lt;blockquote&gt;
&lt;p&gt;无论是否使用填充，权重和偏置的大小都不会改变。填充卷积的主要原因是分离卷积和改变图像大小。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;p&gt;假设输入大小为 $(H,W)$，滤波器大小为 $(FH,FW)$，输出大小为 $(OH,OW)$，填充为 $P$，步幅为 $S$，满足如下算式：&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OH = frac{H+2P-FH}{S}+1&lt;/p&gt;
&lt;p class=&quot;editormd-tex&quot;&gt;OW = frac{W+2P-FW}{S}+1&lt;/p&gt;
&lt;p&gt;卷积操作可以用于检测图像的特征。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)
with torch.no_grad():
    conv.weight[:] = torch.tensor([-1., 0., 1.],
                                  [-1., 0., 1.],
                                  [-1., 0., 1.])
    conv.bias.zero_()
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;使用上述滤波器后，图像的垂直边缘被增强了。因此，我们可以使用更复杂的滤波器，检测例如水平、对角线边缘、十字或棋盘等图案。计算机视觉专家的工作一直以来就是设计出最有效的过滤器组合，使图像中突出的某些特征和物体能够被识别出来。&lt;/p&gt;
&lt;h4 id=&quot;h4-u6C60u5316&quot;&gt;&lt;a name=&quot;池化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;池化&lt;/h4&gt;&lt;p&gt;我们建议使用小卷积核，如 $3 	imes 3$、$5 	imes 5$，以实现峰值局部性。但是如何使我们的网络看到更大范围的图像呢？一种方法是使用大的卷积核，但是过大的卷积核回收敛到旧的全连接层，仿射变换失去了卷积的所有优良性质。另一种方法是在卷积之后堆叠另一个卷积，同时在连续卷积之间对图像进行下采样。&lt;/p&gt;
&lt;p&gt;下采样可选择的操作有：&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平均池化，取 4 个像素的平均，是早期方法，已经不受欢迎；&lt;/li&gt;&lt;li&gt;最大池化（Max pooling），取 4 个像素的最大值，是目前最常用的方法之一；&lt;/li&gt;&lt;li&gt;使用带步幅的卷积，具有一定的应用前景。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;torch.nn 提供了一维、二维和三维的池化操作，分别是 &lt;code&gt;torch.nn.MaxPool1d()&lt;/code&gt;、&lt;code&gt;torch.nn.MaxPool2d()&lt;/code&gt; 和 &lt;code&gt;torch.nn.MaxPool3d()&lt;/code&gt;。&lt;/p&gt;
&lt;h2 id=&quot;h2-torch-nn&quot;&gt;&lt;a name=&quot;torch.nn&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn&lt;/h2&gt;&lt;p&gt;torch.nn 中包含了创建各种神经网络结构所需的模块（在其他框架中，通常称为层）。&lt;/p&gt;
&lt;h3 id=&quot;h3-torch-nn-sequential&quot;&gt;&lt;a name=&quot;torch.nn.Sequential&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Sequential&lt;/h3&gt;&lt;p&gt;torch.nn 提供了一种通过 torch.nn.Sequential 容器来连接模型的方式。该模型将第一个模块所期望的输入指定为 torch.nn.Sequential 的一个参数，将中间输出传递给后续模块，并产生最后一个模块返回的输出。&lt;/p&gt;
&lt;p&gt;调用 &lt;code&gt;model.parameters()&lt;/code&gt; 将从内部模块收集权重和偏置，还可以使用 &lt;code&gt;model.named_parameters()&lt;/code&gt; 方法获取非激活函数的模块信息。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;seq_model = nn.Sequential(
    nn.Linear(1, 13),
    nn.Tanh(),
    nn.Linear(13, 1)
)
seq_model
[param.shape for param in seq_model.parameters()]
# [torch.Size([13, 1]), torch.Size([13]), torch.Size([1, 13]), torch.Size([1])]
for name, param in seq_model.named_parameters():  
    print(name, param)
# 0.weight Parameter containing:
# tensor([[ 0.9076],
#         [ 0.4156],
#         [ 0.6652],
#         [-0.1614],
#         [ 0.8295],
#         [-0.5206],
#         [ 0.5455],
#         [-0.5800],
#         [-0.8789],
#         [-0.2983],
#         [ 0.9619],
#         [-0.0695],
#         [ 0.3376]], requires_grad=True)
# 0.bias Parameter containing:
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
# 2.weight Parameter containing:
# tensor([[-0.2325,  0.1765, -0.2058, -0.0837, -0.0175, -0.0500, -0.1919, -0.0714,
#          -0.0514, -0.1212, -0.1691,  0.2013,  0.0659]], requires_grad=True)
# 2.bias Parameter containing:
# tensor([-0.1888], requires_grad=True)
seq_model[0].bais
# tensor([ 0.9150, -0.5848,  0.9955,  0.8735, -0.7206,  0.8271, -0.2286,  0.5375,
#          0.0871, -0.0139, -0.5513,  0.7496,  0.4974], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.nn.Sequential 也接受 collections.OrderedDict 作为参数，可以用它来命名传递给 torch.nn.Sequential 的每个模块，此时可以将子模块作为 torch.nn.Sequential 的属性来访问一个特定的参数。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;named_seq_model = nn.Sequential(OrderedDict([  
    (&amp;quot;hidden_linear&amp;quot;, nn.Linear(1, 8)),  
    (&amp;quot;hidden_activation&amp;quot;, nn.Tanh()),  
    (&amp;quot;output_linear&amp;quot;, nn.Linear(8, 1))  
]))  
named_seq_model.output_linear.bias
# tensor([0.3302], requires_grad=True)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;我们可以调用 &lt;code&gt;numel()&lt;/code&gt; 方法，得到每个张量实例中元素的数量。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;numel_list = [p.numel() for p in model.parameters()]
sum(numel_list), numel_list
# (18090, [432, 16, 1152, 8, 16384, 32, 64, 2])
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-nn-module&quot;&gt;&lt;a name=&quot;torch.nn.Module&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.nn.Module&lt;/h3&gt;&lt;p&gt;PyTorch 模块派生自基类 torch.nn.Moudle，一个模块可以用一个或多个参数实例作为属性，这些参数实例时张量，它们的值在训练过程中得到了优化。一个模块还可以有一个或多个子模块作为属性，并且它还能够追踪它们的参数。&lt;/p&gt;
&lt;p&gt;torch.nn 没有实现一些重要的功能，如 reshape，此时不能使用 torch.nn.Sequential 构建网络，需要子类化 nn.Moudle。部分没有参数的子模块不需要实例化，可以直接调用 torch 顶级命名空间（如 &lt;code&gt;tanh()&lt;/code&gt; 这些常用函数）或 torch.nn.functional 内的函数式 API（如小众函数 &lt;code&gt;max_pool2d()&lt;/code&gt;）。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = F.max_pool2d(torch.tanh(self.conv2(out)), 2)
        out = out.view(-1, 8 * 8 * 8)
        out = torch.tanh(self.fc1(out))
        out = self.fc2(out)
        return out

model = Net()
model(img.unsqueeze(0))
# tensor([[-0.0157,  0.1143]], grad_fn=&amp;lt;AddmmBackward&amp;gt;)
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-utils&quot;&gt;&lt;a name=&quot;torch.utils&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils&lt;/h2&gt;&lt;h3 id=&quot;h3-torch-utils-data-dataset&quot;&gt;&lt;a name=&quot;torch.utils.data.Dataset&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.Dataset&lt;/h3&gt;&lt;p&gt;我们要自定义自己数据读取的方法，就需要继承 torch.utils.data.Dataset，并将其封装到 DataLoader 中。torch.utils.data.Dataset 表示该数据集，继承该类可以重载其中的方法，实现多种数据读取及数据预处理方式，而 torch.utils.data.DataLoader 封装了 Data 对象，实现单（多）进程的迭代器输出数据集。&lt;/p&gt;
&lt;p&gt;torchvision.datasets 提供了 MNIST、CIFAR-10 等经典计算机视觉数据集，这些数据集都作为 torch.utils.data.Dataset 的子类返回。&lt;/p&gt;
&lt;p&gt;要自定义自己的 Dataset 类，至少要重载两个方法，&lt;code&gt;__len__&lt;/code&gt; 和 &lt;code&gt;__getitem__&lt;/code&gt;，其中 &lt;code&gt;__len__&lt;/code&gt; 返回的是数据集的大小，&lt;code&gt;__getitem__&lt;/code&gt; 实现索引数据集中的某一个数据。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;class TensorDataset(Dataset):
    # TensorDataset继承Dataset, 重载了__init__, __getitem__, __len__
    # 实现将一组Tensor数据对封装成Tensor数据集
    # 能够通过index得到数据集的数据，能够通过len，得到数据集大小

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 生成数据
data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)

# 将数据封装成Dataset
tensor_dataset = TensorDataset(data_tensor, target_tensor)

# 可使用索引调用数据
print &amp;#39;tensor_data[0]: &amp;#39;, tensor_dataset[0]
# tensor_data[0]:  (
#  0.6804
# -1.2515
#  1.6084
# [torch.FloatTensor of size 3]
# , 0.2058754563331604)

# 可返回数据len
print &amp;#39;len os tensor_dataset: &amp;#39;, len(tensor_dataset)
# len os tensor_dataset:  4
&lt;/code&gt;&lt;/pre&gt;
&lt;h3 id=&quot;h3-torch-utils-data-dataloader&quot;&gt;&lt;a name=&quot;torch.utils.data.DataLoader&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.utils.data.DataLoader&lt;/h3&gt;&lt;p&gt;torch.utils.data.DataLoader 将 Dataset 或其子类封装成一个迭代器，可以迭代输出 Dataset 的内容，同时可以实现多进程、shuffle（打乱顺序）、不同采样策略，数据校对等等处理过程。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;tensor_dataloader = DataLoader(tensor_dataset,   # 封装的对象
                               batch_size=2,     # 输出的batchsize
                               shuffle=True,     # 随机输出
                               num_workers=0)    # 只有1个进程

# 以for循环形式输出
for data, target in tensor_dataloader: 
    print(data, target)

# 输出一个batch
print &amp;#39;one batch tensor data: &amp;#39;, iter(tensor_dataloader).next()
# one batch tensor data:  [
#  0.6804 -1.2515  1.6084
# -0.1156 -1.1552  0.1866
# [torch.FloatTensor of size 2x3]
# , 
#  0.2059
#  0.6452
# [torch.DoubleTensor of size 2]
# ]

# 输出batch数量
print &amp;#39;len of batchtensor: &amp;#39;, len(list(iter(tensor_dataloader)))
# len of batchtensor:  2
&lt;/code&gt;&lt;/pre&gt;
&lt;h2 id=&quot;h2-torch-optim&quot;&gt;&lt;a name=&quot;torch.optim&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;torch.optim&lt;/h2&gt;&lt;p&gt;PyTorch 的 optim 模块提供了一组随时可用的优化器（optimizer），用于更新参数和最小化损失函数的值。优化器使用 PyTorch 的自动求导特性来计算每个参数的梯度，这取决于该参数对最终输出的贡献，并允许用户在复杂的正向传播中依赖动态传播图自动求导。&lt;/p&gt;
&lt;p&gt;优化器初始化时需要传入模型的可学习参数，以及其他超参数如 lr，momentum 等。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;optimizer = optim.SGD(model.parameters(), lr=1e-2)
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;在训练过程中先调用 &lt;code&gt;optimizer.zero_grad()&lt;/code&gt; 清空梯度，再调用 &lt;code&gt;loss.backward()&lt;/code&gt; 反向传播，最后调用 &lt;code&gt;optimizer.step()&lt;/code&gt; 更新模型参数。&lt;/p&gt;
&lt;h2 id=&quot;h2-u6A21u578Bu8BBEu8BA1&quot;&gt;&lt;a name=&quot;模型设计&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;模型设计&lt;/h2&gt;&lt;h3 id=&quot;h3-u589Eu52A0u6A21u578Bu53C2u6570&quot;&gt;&lt;a name=&quot;增加模型参数&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;增加模型参数&lt;/h3&gt;&lt;p&gt;模型参数越多，模型所能管理的输入的可变性就越大。但与此同时，模型出现过拟合的可能性也越大，因为模型可以使用更多的参数来记忆输入中不重要的方面。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6B63u5219u5316&quot;&gt;&lt;a name=&quot;正则化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;正则化&lt;/h3&gt;&lt;p&gt;添加正则化（regularization）项是稳定泛化的一种方法，其目的在于减小模型本身的权重，从而限制训练对它们增长的影响。换句话说，这是对较大权重的惩罚。这使得损失更平滑，并且从拟合单个样本中获得的收益相对减少。&lt;/p&gt;
&lt;p&gt;L1 正则化是模型中所有权重的绝对值之和，L2 正则化是模型中所有权重的平方和，它们都通过一个较小的因子 $lambda$ 进行缩放（该因子是训练前设置的超参数）。&lt;/p&gt;
&lt;p&gt;L2 正则化也称为&lt;strong&gt;权重衰减&lt;/strong&gt;，L2 正则化对参数 $w&lt;em&gt;{i}$ 的负梯度为 $-2 	imes lambda 	imes w&lt;/em&gt;{i}$。在损失函数中加入 L2 正则化，相当于在优化步骤中将每个权重按照当前值的比例衰减。注意，权重衰减适用于网络中的所有参数，包括偏置。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，L1 和 L2 正则化很容易实现。&lt;/p&gt;
&lt;pre&gt;&lt;code class=&quot;lang-python&quot;&gt;# L1 regularization
loss = loss_fn(outputs, labels)
l1_lambda = 0.001
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm

# L2 regularization
loss = loss_fn(outputs, labels)
l2_lambda = 0.001
l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
loss = loss + l2_lambda * l2_norm
&lt;/code&gt;&lt;/pre&gt;
&lt;p&gt;torch.optim.SGD 优化器已存在 &lt;code&gt;weight_decay&lt;/code&gt; 参数，该参数对应 $2 	imes lambda$，在更新过程中直接执行权重衰减。&lt;/p&gt;
&lt;h3 id=&quot;h3-dropout&quot;&gt;&lt;a name=&quot;Dropout&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;Dropout&lt;/h3&gt;&lt;p&gt;Dropout 是一种对抗过拟合的有效策略，它将网络每轮训练迭代中的神经元随机部分清零。&lt;/p&gt;
&lt;p&gt;Dropout 在每次迭代中有效地生成具有不同神经元拓扑的模型，使得模型中的神经元在过拟合过程中协调记忆过程的机会更少。另一种观点是，Dropout 在整个网络中干扰了模型生成的特征，产生了一种接近于增强的效果。&lt;/p&gt;
&lt;p&gt;在 PyTorch 中，我们可以通过在非线性激活与后面的线性或卷积模块之间添加一个 torch.nn.Dropout 模块在模型中实现 Dropout。作为一个参数，我们需要指定输入归零的概率。如果是卷积，我们将使用专门的 torch.nn.Dropout2d 或者 torch.nn.Dropout3d，将输入的所有通道归零。&lt;/p&gt;
&lt;h3 id=&quot;h3-u6279u91CFu5F52u4E00u5316&quot;&gt;&lt;a name=&quot;批量归一化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;批量归一化&lt;/h3&gt;&lt;p&gt;批量归一化（batch normalization）利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。这有助于避免激活函数的输入过多地进入函数的饱和部分，从而消除梯度并减慢训练速度。该方法的论文作者指出，使用批量归一化可以消除或减轻对 Dropout 的需要。&lt;/p&gt;
&lt;p&gt;PyTorch 提供了 torch.nn.BatchNorm1d、torch.nn.BatchNorm2d 和 torch.nn.BatchNorm3d 来实现批量归一化，使用哪种模块取决于输入的维度。由于批量归一化的目的是重新调整激活的输入，因此其位置是在线性变换（或卷积）和激活函数之后。&lt;/p&gt;
&lt;h3 id=&quot;h3-u8DF3u8DC3u8FDEu63A5&quot;&gt;&lt;a name=&quot;跳跃连接&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;跳跃连接&lt;/h3&gt;&lt;p&gt;跳跃连接（skip connections），会跳跃神经网络中的某些层，并将一层的输出作为下一层的输入。&lt;/p&gt;
&lt;p&gt;实验发现深度网络出现了退化问题（Degradation problem）：网络深度增加时，网络准确度出现饱和，甚至出现下降。引入跳跃连接是为了解决不同架构中的不同问题。在 ResNets 的情况下，跳跃连接解决了前述的退化问题，而在 DenseNets 的情况下，它确保了特征的可重用性。&lt;/p&gt;
', 57, '2022-06-23 22:26:08');
INSERT INTO web_class.articles (id, username, title, markdown_text, html_text, read_time, publish_time) VALUES (40, 'zsl', 'FEDSA', '# FEDSA

&gt; W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, &quot;FEDSA: A Data Federation Platform for Law Enforcement Management,&quot; _2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)_, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.

&gt; RESTful
&gt; [RESTful API 一种流行的 API 设计风格](https://restfulapi.cn/)
&gt; REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。

## 摘要

大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。

## I. Introduction

数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。

我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:

- 平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。
- 它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。
- 它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。
- 平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。
- 平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。
- 平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。

在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。

## II. Related Research

近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。

在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。

## III. System Overview

图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:

- Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。
- Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。
- Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。
- CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。
- Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。
- Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。
- Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。
- Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。

## IV. Distinctive System Features

在本节中，我们将详细介绍 FEDSA 的不同特征。

### A. API

FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。

### B. 联邦查询语言

FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;person&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
      &quot;filter&quot;:{&quot;first_name&quot;:&quot;Abel&quot;,&quot;height&quot;:{&quot;$gte&quot;:&quot;160&quot;}} 
} 
```

### C. 基于规则的查询重写

为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。

在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:

在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示

```
{ 
 &quot;query&quot;:{ 
 &quot;scope&quot;:[&quot;P&quot;,&quot;Q&quot;], 
 &quot;output&quot;:{&quot;project&quot;:{}}, 
 &quot;window&quot;:{&quot;limit&quot;:&quot;100&quot;}, 
 &quot;filter&quot;:{&quot;p_2&quot;:&quot;a&quot;,&quot;q_2&quot;:&quot;b&quot;} 
}
```

被转换为相应的 SQL 查询，可以在源数据模式上执行:

```
(select * from T1 where T1.t_12 = &quot;a&quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &quot;b&quot;*)
on (T2.t_21 = object_link.t_21)
```

### D. 可扩展异构数据源支持

FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。

### E. 安全性

由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。

### F. 进程驱动的联合函数创建

通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。

图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。

### G. 系统分布与优化

为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。

## V. Evaluation

目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。

### A. 查询执行的性能

查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。

#### 一般查询
在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。

研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。

#### 编排服务查询
在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。

## VI. Conclusion

在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。
', '&lt;h1 id=&quot;h1-fedsa&quot;&gt;&lt;a name=&quot;FEDSA&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;FEDSA&lt;/h1&gt;&lt;blockquote&gt;
&lt;p&gt;W. Li, Z. Feng, W. Mayer, G. Grossmann, A. K. Kashefi and M. Stumptner, “FEDSA: A Data Federation Platform for Law Enforcement Management,” &lt;em&gt;2018 IEEE 22nd International Enterprise Distributed Object Computing Conference (EDOC)&lt;/em&gt;, 2018, pp. 21-27, doi: 10.1109/EDOC.2018.00013.&lt;/p&gt;
&lt;p&gt;RESTful&lt;br&gt;&lt;a href=&quot;https://restfulapi.cn/&quot;&gt;RESTful API 一种流行的 API 设计风格&lt;/a&gt;&lt;br&gt;REST 是 Representational State Transfer 的缩写，如果一个架构符合 REST 原则，就称它为 RESTful 架构。RESTful 架构可以充分的利用 HTTP 协议的各种功能，是 HTTP 协议的最佳实践。RESTful API 是一种软件架构风格、设计风格，可以让软件更加清晰，更简洁，更有层次，可维护性更好。&lt;/p&gt;
&lt;/blockquote&gt;
&lt;h2 id=&quot;h2-u6458u8981&quot;&gt;&lt;a name=&quot;摘要&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;摘要&lt;/h2&gt;&lt;p&gt;大数据时代，数据联合研究领域面临新的挑战。随着新型数据源和数据格式的出现，终端用户需要执行复杂的搜索和数据分析任务，因此需要灵活的数据联合、定制的安全机制和高性能的处理(如近实时查询)。为了解决这些挑战，本文提出了一个名为 FEDSA 的数据联合平台，并报告了其初步实现。该平台的显著特征包括流程驱动的数据联合、数据联合作为服务、基于高级公共数据模型的简单查询语言、所有联合服务的数据安全保护、查询重写和完全分发。我们将演示这些特性如何应对挑战，讨论当前实现的性能，并概述未来的扩展。&lt;/p&gt;
&lt;h2 id=&quot;h2-i-introduction&quot;&gt;&lt;a name=&quot;I. Introduction&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;I. Introduction&lt;/h2&gt;&lt;p&gt;数据联合(或通常称为数据库联合)的概念是数据集成的一种方法，在这种方法中，中间件提供对许多异构数据源的统一访问。通过使用高级查询语言，用户可以组合、对比、分析和操作他们的数据，就像他们被提供一个虚拟数据仓库一样。数据联合提供的功能在需要从各种异构系统(如政府组织中使用的系统)查询信息的大型组织中非常有用。在大数据时代，新的挑战层出不穷。例如，捕获和收集新类型数据的能力的增强来自于支持不同格式的新系统的开发。最终用户经常要面对一个更加复杂的系统环境，这使得信息搜索和执行数据分析任务更加复杂。这需要灵活的数据联合、定制的安全机制和高性能处理(例如，近实时查询)，特别是在执法机构领域。在一个典型的执法环境中，有多个机构在不同的管辖级别上运作。例如，遵循以州为基础的政策和法律的州一级机构与适用不同政策和法律的联邦一级机构合作。这些机构都使用不同的系统来生成、维护和搜索关键数据。例如，如果一个机构调查涉及州际犯罪和税务欺诈的案件，官员需要分别访问相应的州警察部门和其他机构的数据，而不损害数据的独立性和权限。如果调查涉及嫌疑人，机构可能需要首先发现与嫌疑人有关联的人，例如亲属、同事等，然后可能确定银行账户、车辆、住房和财产信息，这些信息由不同的政府部门或公司管理。如果对嫌疑人签发逮捕令，警察需要找到嫌疑人的地址，识别嫌疑人，在执行逮捕令之前，他们需要知道嫌疑人是否可能拥有任何枪支。这些信息可以是文本、图像或任何其他格式，需要尽快获得，以使警察的逮捕过程尽可能高效和安全。通过使用连接不同执法机构的所有数据源的数据联合平台，可以比目前在机构之间来回进行多个通信的做法更有效地进行上述活动。&lt;/p&gt;
&lt;p&gt;我们开发了一个名为 FEDSA 的数据联合平台，旨在支持执法上下文中的搜索和分析任务。除了向终端用户提供透明性、集成异构数据源、提供添加新数据源的灵活性等数据联合的传统目标外，FEDSA 还提供了一些独特的特性:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;平台顶层的集成工作流系统使最终用户能够以流程驱动的方式灵活地创建自定义数据联合函数。&lt;/li&gt;&lt;li&gt;它使用数据联合作为服务的概念来公开其所有功能，其中所有数据联合服务都是通过具有统一请求和响应结构的 RESTful API 来访问的。&lt;/li&gt;&lt;li&gt;它在公共数据模型(CDM)上应用了一种与 SQL 具有类似表达能力的查询语言。&lt;/li&gt;&lt;li&gt;平台的查询重写组件基于规则，可用于 CDM 和外部数据源之间的复杂查询重写，覆盖模式和属性。&lt;/li&gt;&lt;li&gt;平台应用跨服务的数据访问控制，并允许用户通过定制的身份验证机制访问平台。&lt;/li&gt;&lt;li&gt;平台的所有主要组件都是以松散耦合的方式设计和实现的，并且可以完全分布式。&lt;/li&gt;&lt;/ul&gt;
&lt;p&gt;在其余部分中，我们将对第二节的相关研究进行回顾。我们在第三节描述了平台的总体设计和架构。在第四节中，我们介绍了 FEDSA 不同方面的详细设计。在第五节中，我们将演示该平台在一个典型场景中的应用。然后，我们在第六节总结了我们的测试系统部署和平台的性能基准。最后，我们在第七节概述了进一步的工作并进行了总结。&lt;/p&gt;
&lt;h2 id=&quot;h2-ii-related-research&quot;&gt;&lt;a name=&quot;II. Related Research&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;II. Related Research&lt;/h2&gt;&lt;p&gt;近年来，由于大数据技术作为一种数据集成方式的快速发展，联邦数据访问的话题再次成为热门话题。由于大数据应用程序以数据为中心的本质，许多解决方案都是特定于内容的。例如，在[3]和[4]中，数据联合的研究重点是生物医学数据。同时，业界和学术界也开发了一些通用的集成解决方案。在[5]中，建立了一个利用数据联合方法进行研究数据虚拟池和分析的软件平台。在[6]、[7]和[8]中，提出了三种 sql 引擎，它们能够支持对多个异构源的查询。尽管这些解决方案支持数据联合，但在应用于我们的问题时存在一些限制:首先，[6]和[7]中的解决方案是为数据集成目的而设计的，这需要提取和重新定位远程数据。在我们的场景中，由执法机构维护的数据非常敏感，在许多情况下，不允许保留本地副本或不可行。其次，在所有这三个解决方案中，用户都需要了解每个数据源的模式，以便执行类似 sql 的查询。对于不是 SQL 查询专家的用户来说，执行查询可能很困难，更不用说不知道底层的模式了。第三，这些工具缺乏用新功能扩展联邦功能的能力。执法任务因情况而异，因此数据联合过程中的灵活性至关重要。&lt;/p&gt;
&lt;p&gt;在数据联合系统中，数据源被连接成一个统一的系统。要执行联邦查询并以一致的格式显示多个数据源的结果，需要克服几个主要的挑战:(1)需要一种可以通过最终用户界面使用的高级查询语言，然后(2)通过重写查询将其分发到源系统。这需要(3)语义匹配，以便能够实现单个结果的一致格式。针对这些挑战分别进行了一些研究。在高级查询语言领域，主要存在两大类语言:类 sql 语言和面向文档的语言。在前者中，典型的例子有[6]，[7]，[9]，[8]。这些语言要么是对原始 SQL 的扩展，要么是采用 SQL 的语法，因此使用过 SQL 语言的用户可以很容易地熟悉它们。由于[10]和[11]等 NoSQL 数据库的开发，面向文档的查询语言变得越来越流行。与类似 sql 的语言相比，这些查询语言具有类似的查询能力，可以很容易地应用于 RESTful API 中。在查询重写(也称为基于本体的数据访问和查询翻译)领域，最近的数据集成解决方案通常采用基于规则的方法。例如，在[12]中，为了管理大规模语义数据，维护了表示数据语义的用户定义的模式映射规则，其中使用 Datalog 进行查询转换。在语义匹配领域，已经开发了一些工具来解决模式匹配和关系匹配问题。例如，信息集成工具 Karma[13]使用户能够快速地将数据模式映射到公共模型中。该算法已成功应用于解决语义抽取问题[14]。&lt;/p&gt;
&lt;h2 id=&quot;h2-iii-system-overview&quot;&gt;&lt;a name=&quot;III. System Overview&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;III. System Overview&lt;/h2&gt;&lt;p&gt;图 1 显示了 FEDSA 的总体架构图。它由以下部分组成:&lt;/p&gt;
&lt;ul&gt;
&lt;li&gt;Workflow Management Component，工作流管理组件。该组件调用 FEDSA 的基本服务，即由 FEDSA 主门户提供的服务，并通过利用轻量级工作流管理引擎 Camunda[15]的强大功能来执行复杂的数据联合任务。该组件提供了一个简单的图形化工作流界面，用于创建和调整数据联合流程。一旦用户熟悉了 FEDSA 提供的基本服务，他们就可以设计和执行快速执行复杂任务的工作流，而无需特殊的 IT 技能。新创建的工作流可以在数据库中命名和维护，并重用来构建服务。与 FEDSA 的其他组件不同，该组件拥有自己的门户和位于主门户之上的 API。&lt;/li&gt;&lt;li&gt;Main Portal，主门户。FEDSA 的主门户作为单个接入点，通过 RESTful API 提供对内部组件的访问。当收到查询时，系统对查询进行分析，即对查询进行反汇编、提取属性和查询条件，并调用相应的后续组件进行进一步处理。当任务完成并收集了结果时，在通过 API 返回给面向用户的前端之前，结果会被添加一些额外的信息，比如状态(成功或失败)、每个源上的执行时间、每个源上的结果数量。&lt;/li&gt;&lt;li&gt;Query Parser，查询解析器。查询解析器分析每个查询，标识所请求的信息和条件，并标识联邦中的相关数据源。受支持的常见查询服务包括通过查询条件或链接信息查找实体、查找文档和查找文档内容。查询被转换为内部表示，以便于后续的查询重写和联邦执行。最后，解析后的查询被发送到 FEDSA 的 planner 组件进行进一步处理。&lt;/li&gt;&lt;li&gt;CDM Services，CDM 服务。查询是在公共数据模型(CDM)之上表示的，该模型从单个数据源的模式中抽象出来。针对执法机构的 CDM 是基于所提供的模式和需求(如提供起源信息)创建的。关于 CDM 的细节以及它是如何创建的不在本文的讨论范围之内，这里也不做讨论。CDM 服务的主要功能是通过 RESTful 服务管理 CDM 模式。&lt;/li&gt;&lt;li&gt;Planner。Planner 组件将查询分解为可在单个源上执行的子查询，并计划任务来转换和合并收集到的结果。当规划器接收到查询时，它确定查询中涉及的实体和关系，并从映射存储库中检索相关的数据源。查询被发送到任务管理器进行作业调度。当执行查询时，任务管理器将收集结果并将其发送给计划器。在这里，将合并来自各个数据源的结果，消除重复项，并根据与查询的相关性对结果进行排序。为此可以使用不同的算法。经过排序的结果将返回到主门户的 API façade 进行进一步处理。&lt;/li&gt;&lt;li&gt;Mapping Repository，映射存储库。映射存储库组件存储关于联合中的数据源的信息，例如实体类型、端点信息以及用于模式映射和查询转换的映射规则。如图 1 所示，映射存储库为 FEDSA 的其他组件提供配置和控制输入。映射存储库支持以下功能:(1)在查询规划期间方便识别相关数据源，(2)提供访问特定数据源所需的端点信息，以及(3)提供每个源适配器所需的模式映射和查询翻译规则。此外，映射存储库还通过提供对源元数据的访问的主门户公开外部 RESTful API，从而允许用户从联合中动态添加和删除源(或者在源模式发生变化时更改映射规则)。&lt;/li&gt;&lt;li&gt;Task Manager，任务管理器。任务管理器作为子查询的调度器，收集各个源的子查询结果。当收到查询时，任务管理器将子查询发送到相应数据源的源适配器，源适配器执行子查询并返回中间结果，然后将中间结果转发给规划者进行集成和排序。在这个组件中，可以应用不同的工具和不同的调度策略(例如，单个查询的顺序或并行执行)。&lt;/li&gt;&lt;li&gt;Source Adapter，源适配器。源适配器组件接受 CDM 中表达的子查询，将其重写为源的原生模式上的查询，并在数据源上执行查询。在 FEDSA 中，有几个源适配器可用，每个源适配器都是为特定类型的数据源设计的，比如关系数据库(PostgreSQL)、Elastic Search、HDFS 等。当子查询被转发到源适配器时，源适配器在映射存储库中搜索相应的规则，以执行模式和属性转换，并将转换后的子查询发送到源执行。当接收到执行结果时，源适配器将在映射存储库中再次搜索相应的规则，将结果的模式和属性转换回 CDM 模式。&lt;/li&gt;&lt;/ul&gt;
&lt;h2 id=&quot;h2-iv-distinctive-system-features&quot;&gt;&lt;a name=&quot;IV. Distinctive System Features&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;IV. Distinctive System Features&lt;/h2&gt;&lt;p&gt;在本节中，我们将详细介绍 FEDSA 的不同特征。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-api&quot;&gt;&lt;a name=&quot;A. API&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. API&lt;/h3&gt;&lt;p&gt;FEDSA 使用数据联邦作为服务概念来提供其所有功能，其中所有数据联邦服务都是(1)通过具有统一请求和响应结构的 RESTful API 来访问的。(2)通用查询功能的主要查询过程组件的通用查询 API;(3)用于访问 CDM 结构信息的通用数据模型 API;(4)映射库组件的映射库 API，用于配置源连接和编辑映射规则。所有 API 都遵循 REST 标准并提供 CRUD 支持。&lt;/p&gt;
&lt;h3 id=&quot;h3-b-&quot;&gt;&lt;a name=&quot;B. 联邦查询语言&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;B. 联邦查询语言&lt;/h3&gt;&lt;p&gt;FEDSA 对一般查询使用面向文档的查询语言。本质上，它采用了 MongoDB 查询语法的一个子集。与类似 SQL 的查询语言相比，它具有类似的查询能力，并被设计用于 RESTful 服务。更重要的是，虽然学习类似 SQL 的语言需要一些关系代数知识，但由于使用嵌套数据结构来表达查询标准，我们的查询语言在表达实体和关系方面更直观，方便非资讯科技使用者(例如执法机构的警务人员)使用。在我们的场景中，这种特性是非常需要的。每个请求的查询条件包含四个参数，分别是 scope、filter、output 和 window。参数 scope 决定要搜索的实体类型。例如，人，车辆，地点，案例，等等。filter 参数包含详细的查询条件。参数 output 约束结果有效负载中显示的内容。参数 window 包含窗口约束，如结果的偏移量和限制。例如，下面的查询检索前 100 个 Person 实体及其所有属性，其中实体的名字是“Abel”，它们的高度不小于 160cm。&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;person&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
      &amp;quot;filter&amp;quot;:{&amp;quot;first_name&amp;quot;:&amp;quot;Abel&amp;quot;,&amp;quot;height&amp;quot;:{&amp;quot;$gte&amp;quot;:&amp;quot;160&amp;quot;}} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-c-&quot;&gt;&lt;a name=&quot;C. 基于规则的查询重写&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;C. 基于规则的查询重写&lt;/h3&gt;&lt;p&gt;为了实现基于本体的透明数据访问，采用了基于规则的查询重写方法。这种方法包括离线规则生成阶段和在线重写步骤。&lt;/p&gt;
&lt;p&gt;在规则生成步骤中，从 CDM 生成映射规则到每个数据源，并在映射存储库中维护映射规则，以捕获 CDM 和每个源数据模型之间的关系。图 2 显示了如何在 CDM 部分的一个子集和一个关系数据库模式示例之间生成规则的示例。从这些映射中，翻译规则以半交互的方式派生出来。在前面描述的映射存储库中维护产生的类似于 datalog 的规则。CDM 转换为 DS 的映射规则如下:&lt;/p&gt;
&lt;p&gt;在查询转换的在线阶段，每个涉及数据源的源适配器应用映射和转换规则来进行模式转换。基于第一步生成的规则，应用[16]中提出的一种方法，将查询在 CDM 表示和数据源的查询语言之间进行转换。例如，查询如下所示&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;{ 
 &amp;quot;query&amp;quot;:{ 
 &amp;quot;scope&amp;quot;:[&amp;quot;P&amp;quot;,&amp;quot;Q&amp;quot;], 
 &amp;quot;output&amp;quot;:{&amp;quot;project&amp;quot;:{}}, 
 &amp;quot;window&amp;quot;:{&amp;quot;limit&amp;quot;:&amp;quot;100&amp;quot;}, 
 &amp;quot;filter&amp;quot;:{&amp;quot;p_2&amp;quot;:&amp;quot;a&amp;quot;,&amp;quot;q_2&amp;quot;:&amp;quot;b&amp;quot;} 
}
&lt;/code&gt;&lt;/pre&gt;&lt;p&gt;被转换为相应的 SQL 查询，可以在源数据模式上执行:&lt;/p&gt;
&lt;pre&gt;&lt;code&gt;(select * from T1 where T1.t_12 = &amp;quot;a&amp;quot;)
join object_link
on (T1.t_11 = object_link.t_11)
join (select * from T2 where T2.t_22 = &amp;quot;b&amp;quot;*)
on (T2.t_21 = object_link.t_21)
&lt;/code&gt;&lt;/pre&gt;&lt;h3 id=&quot;h3-d-&quot;&gt;&lt;a name=&quot;D. 可扩展异构数据源支持&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;D. 可扩展异构数据源支持&lt;/h3&gt;&lt;p&gt;FEDSA 通过使用几个特定于源的适配器，提供跨多个异构数据源的数据联合服务。对于每种类型的数据源，都有一个源适配器可用;但是，如果需要，可以为单个数据源提供额外的源适配器。例如，在 Elastic Search 中，不同索引中的文档内容可以以不同的嵌套格式存储，这可能需要不同的适配器。适配器通过将在总体 CDM 中制定的查询转换为特定于源的查询来支持一般查询。此外，FEDSA 还支持无法直接从高级语言查询的特定数据源，其中提供了一些特定于源的服务。为了满足执法场景，表 1 中列出了当前实现的源适配器。需要注意的是，选择 Elastic Search 表示索引搜索引擎，选择 PostgreSQL 表示关系数据库，选择 HDFS 表示分布式文件系统，选择 MongoDB 表示 NoSQL 数据库。&lt;/p&gt;
&lt;h3 id=&quot;h3-e-&quot;&gt;&lt;a name=&quot;E. 安全性&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;E. 安全性&lt;/h3&gt;&lt;p&gt;由于所收集的大部分数据可能很敏感，并且受到访问限制，因此必须将安全性和访问控制机制设计到整个联邦体系结构中。在 FEDSA 中，访问控制框架为所有联邦服务提供保护。在将结果返回给用户之前，必须对每个请求进行身份验证，并检查信息访问的权限。由于源之间的身份验证和授权机制可能不同，平台支持特定于源和特定于用户的身份验证凭据。在撰写本文时，框架的身份验证方面已经完全实现，而授权是由数据源直接传递的。但是，通过利用 FEDSA 的映射存储库组件，理论上可以实现基于规则的整体授权机制，并且可以成为 FEDSA 未来工作的一部分。&lt;/p&gt;
&lt;h3 id=&quot;h3-f-&quot;&gt;&lt;a name=&quot;F. 进程驱动的联合函数创建&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;F. 进程驱动的联合函数创建&lt;/h3&gt;&lt;p&gt;通过创建调用 FEDSA 基本服务的工作流，FEDSA 用户可以为复杂任务创建和执行定制服务。这种灵活性是通过集成 Camunda 工作流引擎提供的，其功能通过类似于 FEDSA 的主查询门户的门户公开的。此门户遵循与主门户相同的 API 设计原则。因此，使用工作流组件创建的服务可以被其他应用程序以与基本服务相同的方式使用。&lt;/p&gt;
&lt;p&gt;图形化的工作流设计界面使用户能够快速设计完成复杂任务的工作流。图 3 显示了名为 FindNearbyEntities 的工作流示例的图形化表示。给定实体链接到其他实体，该工作流确定通过链接可到达的实体最多有 k 跳，并从数据源获得实体的完整记录。这个工作流对于检索与相关人员或实体关联的关键实体的信息特别有用。当为特定的请求创建工作流实例时，会迭代计算结果，如图 4 中的伪代码所示。首先对实体初始 id、实体类型、跳数、结果集等属性进行初始化。其次，调用 FEDSA 的基本 FindAdjacentEntities，它返回与给定种子实体直接链接的特定类型的所有实体的 id。重复此步骤，直到达到要求的跳数。最后，返回结果集。新创建的工作流可以在数据库中命名和维护，以便重用。此外，现有的工作流可以修改和扩展。例如，可以扩展图 3 所示的工作流，不仅可以检索实体 id，还可以从相应的数据源检索实际的实体(参见图 5)。&lt;/p&gt;
&lt;h3 id=&quot;h3-g-&quot;&gt;&lt;a name=&quot;G. 系统分布与优化&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;G. 系统分布与优化&lt;/h3&gt;&lt;p&gt;为了减少延迟并适应分布式数据管理环境，需要一个分布式架构。这使得联邦编配组件可以与数据源共存，异步集成模式有助于提高可伸缩性。为此，FEDSA 的所有主要组件都是以松散耦合的方式设计和实现的。具体来说，工作流管理组件和映射存储库组件是分布式的，其余的组件独立工作，无需调用彼此的函数。因此，FEDSA 有可能以完全分布式模式部署。如图 1 所示，可以以分片方式部署多个计划任务管理器对，以支持多个查询的并发处理。此外，在规划器、任务管理器和源适配器中缓存关键元数据和映射信息，可以最大限度地减少在查询重写期间访问映射存储库的需要，否则这将成为瓶颈。如果映射存储库被更新，缓存将被刷新。&lt;/p&gt;
&lt;h2 id=&quot;h2-v-evaluation&quot;&gt;&lt;a name=&quot;V. Evaluation&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;V. Evaluation&lt;/h2&gt;&lt;p&gt;目前 FEDSA 的实施起到了验证概念的作用。一些组件目前正在使用所提议方法的简化版本。因此，在评价中，我们给出了几个有代表性的变量和服务的数字。系统部署在 3 个 OpenStack 服务器上，分别为 app 服务器、index 节点和 data 节点。具体来说，除了映射存储库之外，所有主要的 FEDSA 组件都托管在应用程序服务器中，而映射存储库托管在数据节点中。应用服务器中还部署了一个 web UI 来模拟用户应用程序的活动。索引节点中部署 1 台弹性搜索服务器。数据节点中部署 1 台 PostgreSQL 服务器、1 台 MongoDB 服务器和 1 台 HDFS 服务器。&lt;/p&gt;
&lt;h3 id=&quot;h3-a-&quot;&gt;&lt;a name=&quot;A. 查询执行的性能&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;A. 查询执行的性能&lt;/h3&gt;&lt;p&gt;查询响应时间是衡量系统性能的最重要的指标之一，因为大多数查询都要求接近实时的性能。我们评估 FEDSA 的两种主要查询类型的查询执行时间，即一般查询和编排服务查询。我们不报告 CDM 服务和映射存储库服务的结果，因为这些服务遵循类似的处理管道，因此查询执行时间应该与一般查询相当。&lt;/p&gt;
&lt;h4 id=&quot;h4-u4E00u822Cu67E5u8BE2&quot;&gt;&lt;a name=&quot;一般查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;一般查询&lt;/h4&gt;&lt;p&gt;在 FEDSA 提供的所有通用查询中，测试了三个查询服务，它们是 FindEntities、GetTextContent 和 GetBinaryContent。FindEntities 服务根据联合查询语言中指定的条件返回实体。GetTextContent 和 GetBinaryContent 是特定于源的服务，它们分别返回从 HDFS 提取的内容和二进制格式的原始文件。表 2 显示了发送到 FindEntities 服务的查询的执行时间，这些查询只涉及 PostgreSQL，只涉及 Elastic Search，并且分别涉及 PostgreSQL 和 Elastic Search。在涉及不同数据源的每个测试中，查询是相同的。每个查询执行两次:一次是用特定于源的查询语言制定的，并直接在源系统上执行，另一次是通过 FEDSA 门户在 CDM 上发出查询。&lt;/p&gt;
&lt;p&gt;研究结果显示了几个发现。首先，对于只涉及 PostgreSQL 的查询，它表明对于有更多匹配的查询，直接通过 JDBC 执行的执行时间确实会增加，但增量没有 FEDSA 查询那么显著。这主要是由于将结果转换回 CDM 模式的过程。只涉及 Elastic Search 的查询支持这一发现，因为在这些测试中，不需要对结果进行转换，并且对于匹配更多的查询，执行时间也不会高很多。其次，通过 FEDSA 发送的查询会导致较长的执行时间，但开销并不大(通常不超过数百毫秒)。第三，涉及两个数据源的联邦查询，执行时间略大于单个数据源执行时间的总和。这与任务管理器中应用的顺序调度策略是一致的。表 3 显示了对特定于 hdfs 的 GetTextContent 和 GetBinaryContent 服务的查询的执行时间。结果表明，在相同的网络条件下，对 GetBinaryContent 服务的查询产生的执行时间要短得多，而对 GetTextContent 服务的查询在从文件中提取文本上消耗了大量的时间。&lt;/p&gt;
&lt;h4 id=&quot;h4-u7F16u6392u670Du52A1u67E5u8BE2&quot;&gt;&lt;a name=&quot;编排服务查询&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;编排服务查询&lt;/h4&gt;&lt;p&gt;在工作流服务组件的评估中，我们使用了在第四节中描述的 FindNearbyEntities 示例。在表 4 中，第一行显示了 GetAdjacentEntities 的执行时间，这是 FindNearbyEntities 服务调用的基本服务，第二到第四行显示了 FindNearbyEntities 1-3 跳的执行时间。研究结果显示了几个发现。首先，FindNearbyEntities 服务的执行时间明显长于它调用的基本服务，这表明管理工作流实例的开销可能很大。其次，当跳数增加时，FindNearbyEntities 服务的执行时间稳步增加，尽管速度低于初始开销。这一结果表明，尽管启动工作流实例的开销可能很大，但调用 FEDSA 的基本服务的后续开销相对较小，这取决于被调用的基本服务的执行时间。在上面进行的所有测试中，大多数查询都可以在 1 秒内执行，这表明使用 FEDSA 可以实现接近实时响应的目标。&lt;/p&gt;
&lt;h2 id=&quot;h2-vi-conclusion&quot;&gt;&lt;a name=&quot;VI. Conclusion&quot; class=&quot;reference-link&quot;&gt;&lt;/a&gt;&lt;span class=&quot;header-link octicon octicon-link&quot;&gt;&lt;/span&gt;VI. Conclusion&lt;/h2&gt;&lt;p&gt;在本文中，我们提出了一个名为 FEDSA 的数据联合解决方案及其初始实现，旨在满足执法场景中与信息收集和探索相关的交互式查询需求。文中介绍了几个不同的特性，它们可以满足用户需求的几个方面。我们的实验表明，目前由 FEDSA 提供的大多数服务都可以在 1 秒内完成，这表明我们的体系结构的开销是可以接受的。联邦食品和工业安全局的实施仍在进行中。我们的目标是用排序、合并和过滤机制来细化 planner 组件的实现，用进一步的调度策略来扩展任务管理器，力求优化响应时间，增强查询翻译实现，使查询集更丰富。并充分开发了一个具有身份验证和授权机制的联合安全框架。此外，还需要对 FEDSA 在一组实际数据集和查询配置文件上的全面实现和分布式部署进行综合评估。&lt;/p&gt;
', 50, '2022-06-23 22:39:20');