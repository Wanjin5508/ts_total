

## 想顺利上手：特征工程 / Shapelet / 词袋 / ROCKET，需要先掌握哪些知识？

我按“你必须掌握（可直接开干）→加分项（做得更稳/更强）”列一个最小知识栈。

---

# A. 通用机器学习底座（所有方法都依赖）

### 必须掌握

1. **监督分类基本框架**
    
	- 训练/验证/测试分离
	    
	- 交叉验证（尤其是 **GroupKFold**）
	    
	- 指标：Precision/Recall/F1、PR-AUC（PD 常更看重召回）、混淆矩阵
    

2. **防泄漏的工程化建模**
    

	- `Pipeline`：标准化、特征选择、模型必须打包
	    
	- 特征选择必须嵌套在 CV 内（尤其 tsfresh 这种）
    

3. **常用可解释模型**
    

	- Logistic Regression（L1/L2 正则）
	    
	- Linear SVM
	    
	- 决策树 / RandomForest / GBDT（再配合特征重要性、Permutation importance、SHAP）
    

4. **类别不平衡处理（HFCT 很常见）**
    

	- class_weight、阈值移动、PR 曲线选点
	    
	- 分组条件下的重采样要谨慎（避免跨组“造样本”）
    

### 加分项

- 概率校准（Platt/Isotonic）用于“置信度阈值”更可信
    
- 误差分析：按 run/工况分桶看错在哪
    

---

# B. 时间序列/信号处理底座（特征工程与切片质量的根基）

### 必须掌握

1. **采样与滤波**
    

	- 采样率、奈奎斯特、混叠
	    
	- FIR/IIR 基本概念，带通/带阻（工频及谐波）、零相位滤波（filtfilt）
    

2. **窗口化与谱分析**
    

	- 窗函数、谱泄漏
	    
	- FFT、功率谱密度 PSD（Welch）
	    
	- 频带能量/谱质心/带宽/谱熵等可解释频域特征
    

3. **触发/检测与分割（你已经做了，但建议补上理论闭环）**
    

	- 阈值触发、能量触发、峰度触发
	    
	- 去基线、去趋势、去直流、归一化（尤其 z-normalization 对 shapelet/词袋很关键）
    

### 加分项

- STFT 与时频图的基本读法
    
- 小波 / 小波包：尺度—频带对应关系、能量特征与能量熵
    

---

# C. Shapelet（子序列可解释）需要的知识点

### 必须掌握

1. **子序列、距离与归一化**
    

	- 子序列提取（sliding subsequence）
	    
	- 距离：欧氏距离为主（配合 z-normalization）
	    
	- 为什么要 z-normalize：消除幅值与偏移影响，让 shapelet 捕捉形状
    

2. **Shapelet Transform 思路**
    

	- 学 shapelet（判别子序列）→ 计算到 shapelet 的最小距离 → 变成特征 → 用线性/树模型分类  
	    解释输出就是：**哪些 shapelet 触发了分类**，并能定位回原始波形片段。
    

### 加分项

- shapelet 的选择准则（信息增益/ANOVA/F-stat）
    
- 复杂度控制（候选数量、长度集合、采样策略）
    

---

# D. 词袋/字典方法（Bag-of-Patterns / WEASEL 系）需要的知识点

### 必须掌握

1. **离散化/符号化**
    

	- 滑窗
	    
	- 把连续序列转成符号序列（常见：SAX、SFA）
	    
	- n-gram / 词频（BoW/TF）
    

2. **高维稀疏特征 + 线性模型**
    

	- Logistic Regression / Linear SVM 在稀疏空间里特别强
	    
	- 基本的特征筛选（卡方/信息增益）概念
    

解释输出：**哪些“模式词”最能区分 PD/噪声**，也能回溯到对应的局部片段（虽然比 shapelet 更“符号级”）。

### 加分项

- 多尺度窗口（不同窗口长度产生不同“词”）
    
- 正则化与特征选择对稳定性的影响
    

---

# E. ROCKET / MiniRocket 需要的知识点

### 必须掌握

1. **一维卷积的直觉（不需要深度学习全套）**
    

- 卷积核在时序上扫过，得到响应
    
- dilation（空洞）与不同感受野
    
- ROCKET 的两个核心统计：`max` 和 `proportion of positive values`
    

2. **线性分类器**
    

- RidgeClassifier / Logistic Regression  
    ROCKET 的强大依赖于“随机卷积特征 + 线性模型”。
    

解释性相对弱一些，但你可以把它作为**性能对照上限**：证明“不是因为不用 DL 才做不好”。
