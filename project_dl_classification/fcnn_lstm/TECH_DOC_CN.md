# FCNN + BiLSTM 局部放电识别技术讲解（中文版）

面向研发同学的深度解读，重点回答三个问题：
1. **为什么要把 FCNN 和 BiLSTM 结合？**
2. **它们是通过什么数学/神经网络机制完成分类？**
3. **组合方式是什么，数据在网络里具体怎么流动？**

本文基于 `project_dl_classification/fcnn_lstm` 目录中的代码，逐层拆解原理与实现。

---

## 1. 组合动机：CNN 负责“看局部”，LSTM 负责“记顺序”

### 1.1 FCNN（全卷积网络）的特点
- **局部感受野**：`Conv1d` 在时间轴上滑动，通过核权重 $w$ 与局部片段 $x_{t:t+k-1}$ 做卷积：
  $$
  y_t = \sum_{i=0}^{k-1} w_i \cdot x_{t+i}
  $$
  能自动学习局部脉冲形状的匹配模板。
- **共享权重**：同一组卷积核对所有时间步共享，提升泛化能力、降低参数量。
- **下采样能力**：使用 stride=2 的卷积相当于在时域做抽样，压缩序列长度、提升噪声鲁棒性。

### 1.2 BiLSTM 的特点
- LSTM 单元通过门控机制（输入门 $i_t$、遗忘门 $f_t$、输出门 $o_t$）调控记忆：
  $$
  f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
  $$
  能将较长时间跨度的信息保存在状态 $c_t$ 中。
- **双向结构**：BiLSTM 同时沿前向、反向遍历序列，将“过去”和“未来”的上下文拼接，适合捕捉放电事件前后的形态变化。

### 1.3 为什么要组合？
- 单独 FCNN：擅长提局部形状，但缺乏长短期依赖。
- 单独 LSTM：能建模时序，但原始长序列直接送入 LSTM 训练慢、易过拟合。
- **组合方案**：先用 FCNN 压缩并提取高层局部特征，再让 LSTM 在压缩后的序列上建模长程依赖 → 同时兼顾局部识别与全局语境。

---

## 2. 数据流与网络结构

### 2.1 数据准备
- `PartialDischargeDataset` 将 `raw_signal` min-max 归一化：$\hat{x} = (x - \min)/( \max - \min)$。
- 返回形状 `(1, L_i)` 的张量和标签 `label`。
- `custom_collate_with_lengths`：
  1. 记录每条序列长度 $L_i$。
  2. `pad_sequence` 对齐成 $L_{\max}$。
  3. 转为 `(N, C=1, L_{\max})`，并返回 `lengths` 张量。

### 2.2 FCNN 前端

| 层 | 超参数 | 作用 |
| --- | --- | --- |
| Conv1d (1→32) | k=5, stride=2, padding=2 | 对原始信号做低级滤波 + 下采样 |
| Conv1d (32→64) | k=5, stride=2, padding=2 | 扩大感受野，再次下采样 |
| Conv1d (64→128) | k=3, stride=2, padding=1 | 进一步压缩序列，得到 128 通道特征 |

卷积后的长度通过公式计算：
$$
L_{\text{next}} = \left\lfloor \frac{L + 2p - d(k-1)}{s} \right\rfloor + 1
$$
代码中的 `_lengths_after_conv_stack` 就是逐层套用该公式，得到 `lengths_prime`，保证后续 LSTM 只读取有效时间步。

### 2.3 BiLSTM 主干

处理流程：
1. 将卷积输出 `(N, 128, L_3)` 转置为 `(N, L_3, 128)`，符合 `batch_first` 的 LSTM 输入规范。
2. 使用 `pack_padded_sequence(x, lengths_prime)` 把 padding 截断，避免无效时间步干扰。
3. LSTM 每个时间步执行：
   $$
   \begin{aligned}
   i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) \\
   f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) \\
   o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) \\
   \tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) \\
   c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
   h_t &= o_t \odot \tanh(c_t)
   \end{aligned}
   $$
4. 双向拼接：`h_fwd`（正向最后一步）与 `h_bwd`（反向最后一步）concat → `feat ∈ ℝ^{N×2H}`。

### 2.4 分类头

```
feat → Linear(2H → H) → ReLU → Dropout → Linear(H → num_classes)
```

等价于对 `feat` 乘以权重矩阵 $W \in \mathbb{R}^{K \times 2H}$，得到 logits：
$$
z_k = W_k^\top \cdot \text{feat} + b_k
$$
`CrossEntropyLoss` 将 logits 与 one-hot 标签比较，通过 softmax+负对数似然计算损失，驱动反向传播调整 CNN/LSTM 的权重。

---

## 3. 组合方式总结

1. **串联式架构**：FCNN → BiLSTM → MLP。卷积输出直接成为 LSTM 的输入特征。
2. **长度对齐机制**：通过 `_conv_cfg` 记录的 (k, s, p, d) 配置计算 `lengths_prime`，再交给 `pack_padded_sequence`，确保时间维一致。
3. **梯度传递**：分类损失对最终 logits 求导，梯度沿 MLP → LSTM → Conv1d 反向传播，实现端到端联合训练。

---

## 4. 两个子网络如何“学会”分类？

1. **FCNN 学习什么？**
   - 卷积核在训练中被更新为“局部模板”。比如捕捉 PD 脉冲的上升沿/下降沿、噪声形状。
   - 下采样后的特征图保留“局部事件是否出现”的信息，用通道维度编码不同形状。

2. **BiLSTM 学习什么？**
   - 通过门控机制学习“事件出现的顺序、间隔、持续时间”，例如“先出现一次大幅度放电，随后是振荡衰减”。
   - 双向结构把前后文都考虑进去，适合识别上下文依赖强的局部放电模式。

3. **联合学习过程**
   - 前向传播：FCNN 输出 $F$，LSTM 输出隐藏状态 $h$，MLP 输出 logits。
   - 反向传播：`dLoss/dlogits` 传给 MLP、再传给 LSTM 的隐藏状态和权重、最终传到卷积核，使得“能够正确分类的局部模式”被强化，“误导分类的模式”被抑制。

---

## 5. 训练与推理要点

1. **长度张量**：`lengths_prime` 必须与 batch 内样本一一对应，否则 `pack_padded_sequence` 会报错或导致错位。
2. **梯度裁剪**：`clip_grad_norm_(model.parameters(), 1.0)` 防止 LSTM 在长序列上梯度爆炸。
3. **归一化策略**：当前版本是逐样本 min-max，如需跨批次一致性，可改为 `RobustScaler` 并保存参数。
4. **推理阶段**：与训练相同的预处理&打包流程，否则卷积长度和 LSTM 时间轴会错配。

---

## 6. 延伸建议

1. **Explainability**：可在卷积输出基础上做 CAM/Grad-CAM，观察 FCNN+BiLSTM 联合后关注的时间段。
2. **多任务学习**：在 `feat` 上并联多个头，实现 PD 类型、置信度等多输出。
3. **数据增广**：对 `raw_signal` 做平移、加噪等操作，提高模型鲁棒性。

---

通过以上拆解，你应该清楚：
- FCNN 如何把海量时间序列压缩成语义丰富的局部特征；
- BiLSTM 如何利用门控结构记住事件顺序；
- 两者串联如何实现“既懂局部形状，又懂长程依赖”的 PD 分类器。

掌握这套思路，就能在实际工程中自如地调整卷积堆叠、LSTM 层数或分类头，打造满足不同场景的时间序列智能模型。***

---

## 7. 为何精度优于单纯 FCNN？

实测表明：FCNN+BiLSTM 组合可达到约 99% 的分类精度，而纯 FCNN 约为 94%。原因主要包括：

1. **双向上下文建模**：BiLSTM 在时间正反两个方向解析信号，能够捕捉“放电事件前兆 + 事件后的衰减”这样的长程依赖。纯 FCNN 的感受野受核大小限制，很难完整覆盖这些关联。
2. **层次特征提取**：前端 FCNN 用 stride=2 的卷积逐层压缩序列，相当于在不同尺度上提取特征再交给 LSTM。相比直接对原序列做卷积，组合模型的特征更具判别性。
3. **有效长度对齐**：通过 `_lengths_after_conv_stack` 和 `pack_padded_sequence`，LSTM 仅处理真实的时间步。FCNN baseline 在处理 padded 区域时会把无效数据也纳入统计，容易稀释重要特征。
4. **梯度传播路径**：BiLSTM 的门控机制在长序列上更稳定，能将分类信号反向传给 FCNN，使卷积核学习到更贴合放电事件的局部模板。

综合来看，FCNN 负责高效提取局部形态，BiLSTM 负责刻画持续时间、先后顺序，两者协同即可大幅提升识别准确率。***
