# 基于 FCNN 的局部放电识别教学笔记

同学们好，这份教程会带着大家完整走一遍 `project_dl_classification/fcnn` 项目，从为什么要选 FCNN 开始，到参数如何在模块间传递、模型构建细节、Dataset 与 `collate_fn` 的作用，再到最后如何画出可解释 AI 的 CAM（Class Activation Map）。跟着步骤做，你就能把 GUI 标注出来的 Parquet 数据训练成一个能识别局部放电（PD）的智能模型。

---

## 1. 为什么选择 FCNN？

1. **时间不变特性**：局部放电信号往往形状稳定但位置不固定，FCNN 通过全卷积 + 全局平均池化（GAP）天然具备平移不变性，能从整段时间序列抓取模式。
2. **轻量且鲁棒**：相比 RNN/LSTM，FCNN 没有循环依赖，前向和反向计算都更快，在长序列上不易出现梯度衰减。
3. **解释性友好**：卷积核直接作用在时间轴，结合 GAP 很容易构建 CAM，可视化每个时间点对分类的贡献，对高压行业的合规性至关重要。

---

## 2. 超参数如何在工程内传递？

超参数集中在 `fcnn_hyperparameters.py`：

```python
in_channels = 1
nr_featuremaps = 2
kernel_size = 7
batch_size = 32
lr = 0.001
epochs = 100
```

在 `fcnn_train.py` 中统一导入：

```python
from fcnn_hyperparameters import (
    in_channels, nr_featuremaps, kernel_size,
    batch_size, lr, epochs
)
```

调用链如下：

1. `FCNN(in_channels, nr_featuremaps, kernel_size)` 设置卷积层的形状。
2. `DataLoader(..., batch_size=batch_size, collate_fn=collate_fn)` 控制批次大小与补齐策略。
3. `optim.Adam(model.parameters(), lr=lr)` 设置优化器学习率。
4. 训练循环迭代 `epochs` 次完成全量学习。

保持这一集中式配置能让团队调参时更有条理，也方便写自动化脚本。

---

## 3. FCNN 模型构建细节

模型定义在 `fcnn_model.py`，结构非常紧凑：

```python
self.conv1 = nn.Conv1d(in_channels, nr_featuremaps, kernel_size,
                       stride=1, padding='same', bias=False)
self.relu1 = nn.ReLU()
self.conv2 = nn.Conv1d(nr_featuremaps, nr_featuremaps, kernel_size,
                       stride=1, padding='same', bias=False)
self.relu2 = nn.ReLU()
self.gap = nn.AdaptiveAvgPool1d(1)
self.fc1 = nn.Linear(nr_featuremaps, 2)
```

几个关键点：

1. **padding='same'** 保证卷积前后序列长度一致，省去手动计算边界。
2. **Kaiming 初始化** 让 ReLU 输出分布稳定，避免训练初期梯度爆炸或消失。
3. **GAP + 全连接** 代替传统的 Flatten + 多层全连接，参数更少，还能直接计算 CAM。
4. **`return_conv` 开关**：`forward` 方法支持返回卷积输出，为 CAM 提供必要的特征图。

---

## 4. Dataset 与 `collate_fn` 的作用

### 4.1 Dataset：如何读入 GUI 标注的 Parquet

`PartialDischargeDataset` 直接接收 Pandas DataFrame：

```python
sample = self.dataframe.iloc[idx]
signal = torch.tensor(sample['tds'][::4], dtype=torch.float32)
label = torch.tensor(sample['cluster_id'] + 1, dtype=torch.long)
return signal, label
```

- `tds` 是 GUI 导出的时间序列；示例里做了 4 倍下采样（`[::4]`）。
- `cluster_id` 可能以 `-1/0` 开始，因此加 1 让标签从 0 起步，满足 `CrossEntropyLoss` 的要求。

`get_sample_for_id` 还能随机抽取指定类别的样本，方便在验证阶段定位问题。

### 4.2 collate_fn：批次内自动补齐

时间序列通常长短不一，直接打包会失败。因此我们自定义 `collate_fn`：

```python
def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0.0)
    return padded_inputs.view(padded_inputs.size(0), 1, -1), torch.tensor(labels)
```

步骤讲解：

1. `pad_sequence` 会按最长序列长度补齐其他样本，填充值设为 0。
2. `view(batch, 1, -1)` 给 Conv1d 增加通道维度（`C=1`）。
3. 返回的 `labels` 是 `torch.long`，可直接传给交叉熵。

通过自定义 `collate_fn`，我们就不用手动对齐序列，大幅减轻数据预处理负担。

---

## 5. 从训练到评估

1. **读取数据**：`df = pd.read_parquet(...)`。
2. **切分**：80% 训练、10% 验证、10% 测试。
3. **构建 DataLoader**：为训练集打开 `shuffle=True`，验证/测试设为 `False`。
4. **训练循环**：
   - 将批次放到 `device`（CUDA/CPU）。
   - 前向计算 `output = model(x_tr)`。
   - 计算 `loss = criterion(output, y_tr)` 并反向传播。
   - 每 100 个 batch 打印一次训练/验证损失。
5. **测试集评估**：遍历 `dataloader_test`，统计准确率。

建议大家在实践中补充如下改进：  
- 使用 `torch.utils.data.random_split` 让划分更随机；  
- 保存最佳验证 Loss 的模型权重；  
- 利用 `torchmetrics` 计算 Precision/Recall。

---

## 6. 绘制可解释 AI 的 CAM（深入原理）

### 6.1 为什么 CAM 能表示贡献度？

最后一层卷积输出记为 $( \mathbf{F} \in \mathbb{R}^{C \times L} )$，其中 \(C\) 是通道数（`nr_featuremaps`），\(L\) 是时间步。GAP 会对每个通道做平均：
$[
g_c = \frac{1}{L}\sum_{t=1}^{L} F_{c,t}, \quad c=1,\dots,C
]$

线性层（全连接层）以矩阵形式计算：
$[
\mathbf{z} = \mathbf{W}\mathbf{g} + \mathbf{b}
]$
其中 $(\mathbf{W}\in \mathbb{R}^{K \times C})$（\(K\) 为类别数），$(\mathbf{g}\in \mathbb{R}^{C})$。这一步体现了“全连接层权重矩阵与输入向量相乘”：

1. 先把 $(\mathbf{g})$ 写成列向量；
2. 对于第 \(k\) 类，logit 为 $(z_k = \sum_{c=1}^{C} W_{k,c} g_c + b_k)$。

如果我们把 GAP 换成“按时间步保留原始值”，则每个时间步的贡献度可以写成：
$$[
\text{CAM}_k(t) = \sum_{c=1}^{C} W_{k,c} \cdot F_{c,t}
]$$

注意这个式子和上面求 logit 的式子结构一致，只是把“均值”改成“逐时间步”。因此 CAM 值越大，代表该时间点在该类别的线性组合中占比越高，相当于贡献度。

### 6.2 代码如何复现数学计算？

```python
logits, conv_out = model(x.view(1, 1, -1), return_conv=True)
weights = model.fc1.weight.data        # => W ∈ ℝ^{K×C}
class_weights = weights[torch.argmax(logits)]
cam = (class_weights.view(-1, 1) * conv_out.squeeze(0)).sum(dim=0)
```

1. `return_conv=True` 让 `model` 返回 $( \mathbf{F} )$。
2. `class_weights` 抽出对应类别的 $(W_{k,\cdot})$。
3. `class_weights.view(-1,1) * conv_out` 就是对每个通道做逐元素乘法，相当于 $(W_{k,c}F_{c,t})$。
4. `sum(dim=0)` 将所有通道贡献加总，得到长度为 \(L\) 的 $(\text{CAM}_k(t))$。

### 6.3 注意力可视化原理

CAM 本质上把全连接层的权重“拉回”到卷积特征图上，类似把注意力指向那些让 logit 最大化的时间点。因为卷积特征图在时间轴上保留了局部模式的位置，线性层权重则告诉我们每个模式对分类的加权程度，两者结合就能勾勒出“模型在看哪里”。所以按照代码中的乘法 + 求和步骤，就能把模型决策过程转化为可视化的注意力曲线。

### 6.4 教学建议

绘制 CAM 后请学生观察：

- CAM 的峰值是否和人工标注的 PD 区间吻合；
- 若模型预测错误，CAM 是否偏离真实 PD 区域，以此分析模型偏差。

这不仅帮助理解模型，也能作为检验数据质量的辅助工具。

---

## 7. 实战小贴士

1. **数据清洗**：确保 Parquet 中的 `tds` 是数值向量，`cluster_id` 没有缺失值。
2. **批次长度**：若某些序列特别长，可以考虑在 `collate_fn` 里做截断或分桶（bucketing）。
3. **可视化仓库**：将 CAM 图和卷积核图保存到 `reports/`，方便复盘。
4. **模型导出**：使用 `torch.save(model.state_dict(), 'checkpoints/fcnn.pt')`，并记录超参数版本。

---

## 8. 总结

我们已经完成了从 GUI 标注数据到 FCNN 训练、解释全流程的教学。重点再次回顾：

1. **FCNN 的优势**：平移不变、计算高效、易于解释。
2. **参数传递链路**：集中管理、模块间清晰传参。
3. **模型构建细节**：双卷积 + GAP + 全连接，配合合理初始化。
4. **Dataset & collate_fn**：解决变长序列的读取与补齐。
5. **CAM 可解释性**：帮助工程师验证模型关注的是不是 PD 信号本身。

按照这套流程，你可以自信地把 FCNN 模型部署到局部放电检测场景中，为高压设备的健康监测提供可靠支撑。加油！
