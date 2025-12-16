+++
date = '2025-12-16T22:09:11+08:00'
title = 'Linear Attention基础-理论篇'
author = "sword865"
type = "post"
tags = ["Attention", "LLM", "Linear Attention"]
topics = ["Efficient Attention", "LLM"]
+++

准备写一些关于线性注意力的文章，对相关理论和工程(Kernel)实践做一些梳理，这一片是关于基础理论的。

# Softmax Attention到线性注意力

## Softmax Attention与O(N^2)复杂度

标准的Transformer使用的是Softmax Attention。给定查询（Query）、键（Key）、值（Value）矩阵 $Q, K, V \in \mathbb{R}^{N \times d}$，其中 $N$ 是序列长度，$d$ 是特征维度（通常 $d \ll N$）。Attention的计算公式为：

$$ Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

让我们仔细分析一下这个计算过程的维度变化：
1.  **计算相似度矩阵**：$QK^T$。
    *   $Q$ 是 $N \times d$，$K^T$ 是 $d \times N$。
    *   相乘得到 $N \times N$ 的矩阵。这一步的计算复杂度是 $O(N^2 d)$。
2.  **应用Softmax**：对每一行进行归一化，维度不变，仍为 $N \times N$。
3.  **加权求和**：乘以 $V$。
    *   $N \times N$ 的矩阵乘以 $N \times d$ 的矩阵 $V$。
    *   结果是 $N \times d$。这一步的计算复杂度也是 $O(N^2 d)$。

**瓶颈所在**：
由于Softmax是非线性的，我们必须先完整地计算出 $N \times N$ 的Attention Matrix。
*   **计算复杂度**：$O(N^2 d)$。随着序列长度 $N$ 的增加，计算量呈平方级增长。
*   **显存占用**：$O(N^2)$（如果不使用FlashAttention等优化）。需要存储 $N \times N$ 的矩阵。
*   **KV Cache**：在推理时，为了避免重复计算，我们需要缓存所有的 $K$ 和 $V$，显存占用为 $O(Nd)$。

对于长序列（Long Context）场景，这个 $O(N^2)$ 的复杂度是不可接受的。

## 线性注意力 (Linear Attention)

线性注意力的核心思想是：**如果我们能去掉Softmax，或者将其替换为某种核函数 $\phi(\cdot)$，我们就可以利用矩阵乘法的结合律优化计算。**

假设我们将相似度函数定义为 $\text{sim}(q, k) = \phi(q)^T \phi(k)$，其中 $\phi(\cdot)$ 是一个特征映射函数（Feature Map），将输入映射到某个特征空间。那么Attention公式变为：

$$ Attention(Q, K, V) = \left( \phi(Q) \phi(K)^T \right) V $$

这里 $\phi(Q)$ 和 $\phi(K)$ 的维度仍然是 $N \times d$（假设特征映射不改变维度，或者映射到 $D$ 维）。

**利用结合律 (Associative Property)**：
矩阵乘法满足结合律 $(AB)C = A(BC)$。我们可以改变计算顺序：

$$ Attention(Q, K, V) = \phi(Q) \left( \phi(K)^T V \right) $$

让我们看看现在的计算过程：
1.  **先计算 $\phi(K)^T V$**：
    *   $\phi(K)^T$ 是 $d \times N$，$V$ 是 $N \times d$。
    *   相乘得到一个 $d \times d$ 的矩阵。我们称之为 **状态矩阵 (State Matrix)** 或 **KV Memory**。
    *   计算复杂度：$O(N d^2)$。
2.  **再左乘 $\phi(Q)$**：
    *   $\phi(Q)$ 是 $N \times d$，状态矩阵是 $d \times d$。
    *   相乘得到 $N \times d$ 的结果。
    *   计算复杂度：$O(N d^2)$。

**优势**：
*   **总复杂度**：$O(N d^2)$。由于 $d$ 通常是一个较小的常数（如64, 128），这相对于序列长度 $N$ 是**线性**的。
*   **显存占用**：我们只需要维护一个 $d \times d$ 的状态矩阵，与 $N$ 无关。

## 循环神经网络形式 (RNN View)

线性注意力不仅计算高效，而且可以写成RNN的形式，这对于自回归生成（Inference）非常友好。

我们可以将上述矩阵运算写成累加的形式。对于第 $t$ 个时刻的输出 $o_t$：

$$ o_t = \sum_{i=1}^t \text{sim}(q_t, k_i) v_i = \sum_{i=1}^t (\phi(q_t)^T \phi(k_i)) v_i $$

提取出与 $i$ 无关的项 $\phi(q_t)$：

$$ o_t = \phi(q_t)^T \sum_{i=1}^t \phi(k_i) v_i^T $$

我们可以定义状态 $S_t = \sum_{i=1}^t \phi(k_i) v_i^T$。这个状态 $S_t$ 是一个 $d \times d$ 的矩阵。
显然，它可以递归地计算：

$$ S_t = S_{t-1} + \phi(k_t) \phi(v_t)^T $$
$$ o_t = \phi(q_t)^T S_t $$

这就是线性注意力的RNN形式：
1.  **更新 (Update)**：用当前的 $k_t$ 和 $v_t$ 更新状态 $S$。
2.  **查询 (Query)**：用当前的 $q_t$ 从状态 $S$ 中提取信息。

这解释了为什么线性注意力在推理时极其高效：它不需要像Softmax Attention那样重新访问所有的历史KV Cache，只需要维护一个固定大小的状态 $S$。

# 线性注意力的设计思路与框架

经典的线性注意力虽然看上去可行，但是效果并不好，需要找到一些入手点，在整个公式的这个基础上进行进一步的优化。

## 状态空间模型 (SSM)

状态空间模型（State Space Models, SSM）起源于控制理论，最近在深度学习中复兴（如S4, Mamba）。它提供了一个统一的框架来理解RNN和CNN。

**连续时间视角 (Continuous-Time View)**：
SSM通常定义在连续时间上，描述了一个系统如何根据输入 $x(t)$ 更新其内部状态 $h(t)$ 并产生输出 $y(t)$：
$$ h'(t) = A h(t) + B x(t) $$
$$ y(t) = C h(t) $$
其中 $A$ 是状态转移矩阵，$B$ 是输入投影，$C$ 是输出投影。

**离散化 (Discretization)**：
为了在计算机上处理离散的序列数据（如文本），我们需要将连续系统离散化。常用的方法是**零阶保持 (Zero-Order Hold, ZOH)** 或 **双线性变换 (Bilinear Transform)**。
离散化后，公式变为递归形式：
$$ h_t = \bar{A} h_{t-1} + \bar{B} x_t $$
$$ y_t = C h_t $$
其中 $\bar{A}$ 和 $\bar{B}$ 是 $A, B$ 和采样步长 $\Delta$ 的函数。例如，在ZOH下，$\bar{A} = \exp(\Delta A)$。

**与Linear Attention的联系**：
仔细观察离散化后的SSM公式，它与Linear Attention的RNN形式惊人地相似：
*   Linear Attention: $S_t = I \cdot S_{t-1} + \phi(k_t) \phi(v_t)^T$
*   SSM: $h_t = \bar{A} h_{t-1} + \bar{B} x_t$

如果我们将Linear Attention中的状态衰减（如果有）看作 $\bar{A}$，将键值对的更新看作输入项，两者在数学形式上是高度统一的。
**Mamba** 的创新之处在于引入了**选择性机制（Selection Mechanism）**，让参数 $B, C, \Delta$ 成为输入 $x_t$ 的函数（即 $B(x_t), C(x_t), \Delta(x_t)$）。这使得模型能够根据当前的内容动态地决定“记住什么”和“忽略什么”，从而解决了传统LTI（线性时不变）系统无法进行上下文推理的问题。

## Key的正交假设与Delta Rule

在标准的线性Attention更新规则 $S_t = S_{t-1} + k_t v_t^T$ 中，我们简单地将新的键值对叠加到状态矩阵中。
这种更新方式类似于神经科学中的 **Hebbian Learning**（赫布学习规则）："Cells that fire together, wire together"。我们简单地增加 $k_t$ 和 $v_t$ 之间的关联强度。

**潜在问题**：
这隐含了一个假设：Key之间是正交的。
如果 $k_t$ 与之前的 $k_{t-1}$ 高度相似（不正交），那么 $S_{t-1} k_t$（即模型对当前Key的旧预测）将不为零。简单的叠加会导致信息的冗余和干扰，使得检索时出现噪声。这里有两种常见的信息冲突：

1.  **单个Key的多义性 (Polysemy of a Single Key)**：
    在长序列中，同一个Key（或者语义上非常相似的Key）可能会出现多次，但对应不同的Value。例如，单词 "Apple" 可能在一段话中指代水果，在另一段话中指代公司。
    在简单的线性叠加 $S_t = \sum k_i v_i^T$ 中，这些不同的Value会被直接相加。当我们用 "Apple" 去查询时，得到的是所有历史Value的混合（平均），而不是特定上下文下的那个Value。这导致了语义的模糊。

2.  **不同Key的相关性 (Correlation between Different Keys)**：
    理想情况下，我们希望Key之间是正交的，这样更新一个Key不会影响其他Key。但实际上，Key向量通常存在复杂的协方差结构（Covariance）。
    如果 $k_t$ 与之前的 $k_{j}$ 高度相关（非正交），那么向状态 $S$ 中添加 $k_t v_t^T$ 不仅更新了 $k_t$ 的方向，也会改变 $k_{j}$ 方向上的投影。这意味着，新信息的写入会干扰旧信息的读取，产生“串扰” (Crosstalk)。

### Transformer的处理方式

标准的Softmax Attention通过**加权平均**来处理这种冲突。它显式地计算 Query 与所有历史 Key 的相似度，然后对 Value 进行加权。
*   对于多义性，Softmax可以通过上下文（Query）和位置编码来区分同一个Key的不同实例，给予不同的权重。
*   对于相关性，Softmax是非线性的，它不需要将历史压缩成一个线性算子，因此不存在“写入新Key破坏旧Key结构”的问题（它只是增加了一个新的参考点）。

而Linear Attention试图将所有历史压缩进一个固定大小的线性算子 $S$，就必须面对这种压缩带来的冲突。

### Delta Rule (DeltaNet)

为了解决这个问题，**DeltaNet** 等工作引入了 **Delta Rule**（也称为Widrow-Hoff规则或LMS算法）。
更新公式变为：
$$ S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^T $$

让我们分解这个公式：
1.  **预测 (Prediction)**：$S_{t-1} k_t$ 是模型根据旧状态对当前Key $k_t$ 的预测值（Retrieve Value）。
2.  **误差 (Error)**：$e_t = v_t - S_{t-1} k_t$ 是真实值 $v_t$ 与预测值之间的差异。
3.  **修正 (Correction)**：我们只将这个“误差”部分更新到状态矩阵中。

在下一个章节中，我们会提到：这本质上是在进行一步**在线梯度下降 (Online Gradient Descent)**，试图最小化当前输入下的重构误差。这使得模型在有限的容量（$d \times d$）下，能够更精确地存储和检索信息，避免了重复信息的累积。

#### 类似的加权平均

如果我们观察状态 $S$ 在 $k_t$ 方向上的投影（假设 $k_t$ 是单位向量），更新公式可以近似理解为：
$$ \text{NewValue} \approx (1 - \beta_t) \cdot \text{OldValue} + \beta_t \cdot v_t $$
这实际上是一个**指数移动平均 (Exponential Moving Average, EMA)**。

*   **Softmax Attention**：是**空间上的加权平均**。在推理时（Query Time），它同时看到所有历史，并根据相似度分配权重。
*   **Delta Rule**：是**时间上的加权平均**。在更新时（Update Time），它通过 $\beta_t$（类似于学习率或卡尔曼增益）动态决定是“保持旧记忆”还是“重写为新记忆”。
    *   当同一个Key出现多次（多义性）时，Delta Rule 允许模型根据 $\beta_t$ 逐渐遗忘旧的含义，聚焦于最新的含义，从而在效果上实现了对不同Value的“选择”或“加权”。



## 状态的遗忘机制

标准的线性Attention（如Linear Transformer）通常对所有历史一视同仁，即 $S_t = \sum k_i v_i^T$。这意味着最早的信息和最近的信息具有相同的权重。
然而，在语言建模中，**局部性 (Locality)** 通常很重要：最近的上下文往往比遥远的上下文更相关。

**RetNet (Retentive Network)** 引入了指数衰减（Exponential Decay）机制：
$$ S_t = \gamma S_{t-1} + k_t v_t^T $$
其中 $\gamma \in (0, 1)$ 是衰减因子。

**直观理解**：
*   **软滑动窗口 (Soft Sliding Window)**：这相当于给历史信息加了一个指数衰减的权重窗口。越久远的信息，权重越小（$\gamma^k$），逐渐被“遗忘”。
*   **相对位置编码 (Relative Positional Encoding)**：这种机制巧妙地编码了相对位置信息。对于第 $n$ 个位置的查询 $q_n$ 和第 $m$ 个位置的键 $k_m$ ($m < n$)，它们之间的相互作用会被乘以 $\gamma^{n-m}$。这自然地表达了“距离越远，影响越弱”的归纳偏置。

这种遗忘机制不仅提高了模型对局部信息的关注度，还增强了数值稳定性，防止状态 $S_t$ 的值无限增长。

# Test Time Regression (TTR) 框架

关于TTR视角有一篇很好的论文：《Test-time regression: a unifying framework for designing sequence models with  associative memory》，该视角将Attention机制视为在测试时（Inference time）进行的回归任务。这个框架帮助我们理解为什么Linear Attention在某些任务上表现不如Softmax Attention。

我们将Attention看作是学习一个函数 $f: \mathbb{R}^d \to \mathbb{R}^d$，使得 $f(k_i) \approx v_i$。

## Nonparametric Regression (Softmax Attention)

Softmax Attention对应于**非参数回归（Nonparametric Regression）**，或者称为**对偶形式 (Dual Form)**。

*   **机制**：模型显式地存储了所有的历史训练样本 $(K, V)$。
*   **预测**：当一个新的查询 $q$ 到来时，它通过计算 $q$ 与所有样本 $k_i$ 的相似度（核函数 $K(q, k_i)$）来加权平均 $v_i$。
    $$ o = \sum_i \frac{K(q, k_i)}{\sum_j K(q, k_j)} v_i $$
*   **特点**：
    *   **背诵**：它不需要压缩数据，而是直接利用原始数据。
    *   **容量**：随着序列长度 $N$ 线性增长。理论上，只要 $N$ 足够大，它可以完美回忆起任何见过的细节。
    *   **代价**：推理成本高昂 ($O(N)$)，因为每次都要遍历所有样本。

## Parametric Regression (Linear Attention)

Linear Attention对应于**参数回归（Parametric Regression）**，或者称为**原始形式 (Primal Form)**。

*   **机制**：我们不再存储所有的原始数据，而是维护一个固定大小的参数矩阵 $S$（即RNN状态）。
*   **学习**：这个过程可以看作是使用在线梯度下降（Online Gradient Descent）或最小二乘法来更新模型参数 $S$，使其能够拟合历史数据 $(k, v)$ 的映射关系。
    $$ S_t = S_{t-1} + k_t v_t^T $$
*   **预测**：当查询 $q$ 到来时，我们直接使用学习到的模型进行预测：
    $$ o = S q $$
*   **特点**：
    *   **拟合**：它试图将所有历史信息压缩进一个固定大小的矩阵 $S$ 中，学习一个函数来从过key来提取value。
    *   **容量**：受限于状态 $S$ 的大小（$d \times d$）。如果历史信息过于复杂或庞大，状态矩阵可能会饱和，导致旧信息的灾难性遗忘。
    *   **代价**：推理成本固定 ($O(1)$)，极其高效。

### 梯度下降视角的详细推导

为了更深入地理解这一点，我们可以显式地写出优化目标。
假设我们的目标是学习一个矩阵 $S$，使得对于所有的历史时刻 $i$，都有 $S k_i \approx v_i$。
我们可以定义在时刻 $t$ 的瞬时损失函数为最小二乘误差：
$$ L_t(S) = \frac{1}{2} \| S k_t - v_t \|^2 $$

我们使用**随机梯度下降 (SGD)** 来更新 $S$：
$$ S_t = S_{t-1} - \eta \nabla_S L_t(S_{t-1}) $$

计算梯度：
$$ \nabla_S L_t(S) = (S k_t - v_t) k_t^T $$

代入更新公式（设学习率 $\eta$ 为步长）：
$$ S_t = S_{t-1} + \eta (v_t - S_{t-1} k_t) k_t^T $$

**这正是Delta Rule的形式！**

*   **标准Linear Attention的特例**：
    如果我们假设 $S_{t-1}$ 初始化为0，或者更强地假设 **Key之间是正交的**（即 $S_{t-1} k_t \approx 0$，旧状态对当前Key没有响应），那么公式简化为：
    $$ S_t = S_{t-1} + \eta v_t k_t^T $$
    这正是标准的Linear Attention更新规则（通常 $\eta=1$）。

*   **结论**：
    *   **Standard Linear Attention** 对应于假设了Key正交性的“懒惰”更新。
    *   **DeltaNet / Linear Recurrent Unit** 对应于完整的梯度下降更新，它显式地计算并消除了预测误差（残差）。

实际上，从这个视角出发，使用不同的优化算法如带动量的SGD、L2正则化等测量，可以推导出其他一些linear attention的算法架构。

**总结**：
这个视角解释了Linear Attention在处理“大海捞针”（Needle In A Haystack）任务时的劣势。因为压缩必然带来有损，当需要精确检索某个特定的历史细节时，参数化的Linear Attention可能无法从压缩的状态中完美还原，而非参数化的Softmax Attention则可以直接查阅原始记录。

# 一些常见的Trick

最后聊一聊为了弥补Linear Attention相对于Softmax Attention的不足（如缺乏局部关注能力、数值稳定性等），研究者们引入的一些Trick。

## 因果卷积 (Causal Convolution)

**问题**：
Softmax Attention由于指数函数的存在，天然倾向于关注局部（最近的邻居通常相似度较高）。而Linear Attention的关注分布通常比较平滑，容易忽略局部的细节。
此外，RNN形式的状态 $S_t$ 只能捕捉到 $t$ 时刻之前的信息，对于“当前时刻”的局部特征捕捉能力较弱。

**解决方案**：
像**Mamba**、**RWKV**和**RetNet**等模型通常会在进入状态空间之前，先对输入进行一个短窗口的**因果卷积（Causal Conv1d）**（例如窗口大小为3或4）。
$$ x'_t = \text{Conv1d}(x_{t-k:t}) $$

**作用**：
1.  **增强局部性**：这相当于显式地让模型“看到”最近的几个Token。
2.  **平移不变性**：卷积操作具有平移不变性，有助于捕捉局部的n-gram特征。
3.  **弥补RNN缺陷**：它充当了一个“预处理”步骤，捕捉高频的局部特征，而让RNN状态专注于捕捉低频的全局依赖。

## DPLR (Diagonal-Plus-Low-Rank) 矩阵

**问题**：
在SSM中，我们需要处理长序列。如果状态转移矩阵 $A$ 初始化不当，梯度可能会在反向传播中消失或爆炸（Vanishing/Exploding Gradients）。此外，计算 $A$ 的幂次或卷积核需要高效的算法。

**解决方案**：
HiPPO理论指出，特定的矩阵结构可以最优地记忆历史。S4模型提出将 $A$ 参数化为**对角矩阵加上低秩矩阵（Diagonal Plus Low Rank, DPLR）**的形式：
$$ A = \Lambda - pp^* $$

**作用**：
1.  **计算效率**：这种结构允许我们利用Woodbury矩阵恒等式和Cauchy Kernel，将矩阵向量乘法和卷积核的计算复杂度降低到 $O(N \log N)$ 或 $O(N)$。
2.  **数值稳定性**：它使得矩阵 $A$ 的特征值分布在左半复平面，保证了系统的稳定性（不会发散）。

虽然在Mamba中，为了硬件效率（GPU Tensor Cores），作者简化为了纯对角矩阵（Diagonal），但DPLR在理论推导和早期SSM模型中起到了至关重要的作用。

## 位置编码 (Positional Encoding)

现在的Linear Attention都是使用NoPE的策略，但是这不意味着Linear Attention就一定不可以使用位置编码：

**RetNet** 提出了一种巧妙的方法，将位置编码融入到状态的衰减中。通过给 $S_t$ 乘以一个复数衰减因子 $e^{i\theta}$，实现了类似RoPE的效果，同时保持了递归形式的高效性。
$$ S_t = \gamma e^{i\theta} S_{t-1} + k_t v_t^T $$
这种方法被称为 **xPos** 的变体或其在RNN中的自然延伸。它使得模型能够根据Token之间的相对距离动态调整关注强度。
