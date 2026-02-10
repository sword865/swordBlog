+++
date = '2026-02-10T11:26:44+08:00'
title = 'Linear Attention基础-部署篇'
author = "sword865"
type = "post"
tags = ["Attention", "LLM", "Linear Attention"]
topics = ["Efficient Attention", "LLM", "Serving"]
+++

系列的前两篇文章写了线性注意力基本的的理论和工程(Kernel)，但是相关的内容只是针对开发环境的一些讨论，实际上在大规模部署时，我们为了更好地利用资源，往往需要做很多优化，这篇文章就从这些落地优化的角度聊一下线性注意力，以及为什么我们会认为现在线性注意力的infra只是在初级阶段。

# 量化

## LLM.int8() 和 Massive Activations

早在ChatGPT爆火前，就有一些研究讨论了Transform架构在量化时遇到的问题，在[LLM.int8()](https://arxiv.org/pdf/2208.07339)这篇论文中，就提到在Transformer的激活值中有部分离群点会影响量化效果，因此建议把离群点和其他值分开处理：

<img width="800"  src="/images/2026/20260210/llm_int8.png" class="center" />

这一现象被称为Massive Activations，目前业务对这个现象进行了大量的研究，其中两个主要因素就是Softmax 和 RoPE。

**Softmax 归一化**：Softmax有着强制要求“总和为1”的特性，但是在实际的LLM运算中，我们往往有更复杂的需求（比如有事我们输入模型的问题比较简单，一个60层的网络可能在前40层就已经完成了计算，对于后面层我们需要的是一个什么都不做的操作，那么在残差网络的帮助下我们其实需要一个0输出），为了满足这种需求就会出现类似Attention Sink的现象，导致注意力集中于单一无关标记的，同时产生**massive activation**。

**RoPE位置编码**：在文章[Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding](https://arxiv.org/pdf/2502.01563)就提到：RoPE在Query/Key的通道维度对上应用了带有频率性角度参数的旋转变换，该变化依赖相对位置，同时在某些维度的投影上显著放大或缩小分量，从而导致部分维度“成簇集中”，出现了异常的spike值。 实际上，该文章也在实验中发现凡是采用 RoPE 的模型（如 LLaMA、Qwen、Gemma、Falcon 等）都会展示这种 massive values stripes，而不使用 RoPE 的模型（如 GPT-2、OPT、Jamba）则没有这一现象。

## Linear Attention 量化

现在，在Linear Attention中没有了Softmax和RoPE的干扰，因此大大缓解了这个问题。[A Unified View of Attention and Residual Sinks](https://arxiv.org/pdf/2601.22966v1)等研究发现：将 Softmax Attention 替换为线性注意力后，隐藏状态的峰值激活从全 Softmax 模型约 6000 降到 510，意味着“massive activations”基本消失。不过，残差归一化层可能引入新的residual sinks，因此对于整个模型尤其是FFN层而言，仍存在离群激活的现象。

总体而言，线性注意力消除了 Softmax 导致的部分量化难点，但针对其他激活异常（如残差汇）仍需额外优化.

相关的研究可以参考[Mamba-PTQ](https://arxiv.org/pdf/2407.12397) 等。

从量化的角度来说，Linear Attention似乎是在部署上占有优势的。

然而实际部署的问题不只是量化。

# Prefix Cache

如果说量化并非LLM部署时的必选项，那么Prefix Cache一定是大规模部署时绕不过的一个策略。

线性注意力对Prefix Cache的支持显然和Full-Attention比要差很多，首先state往往是in place更新，其实单个state的大小一般有2个MB左右，也远大于一个attention block的大小。

不过尽管如此，社区还是在推动Prefix Cache的落地，以vLLM为例[#issues/26201](https://github.com/vllm-project/vllm/issues/26201)就在跟踪这方便的优化。

不过整体而言，由于Prefix Cache引入了更大的block size，Linear Attention的Prefix Cache命中率必然是有所下降的。

# 投机采样

虽然没有Prefix Cache那么重要，投机采样仍然是实际部署时候最有效的加速手段。

线性注意力对投机采样的支持并不好，这里包括比如对tree结构的支持，对并行的验证的支持等。
也有一些策略进行研究。比如 [When Linear Attention Meets Autoregressive Decoding](https://arxiv.org/pdf/2406.07368)