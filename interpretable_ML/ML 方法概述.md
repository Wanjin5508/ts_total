下面把你给的 **Middlehurst et al. “Bake off redux …” (DMKD, 2024)** 里涉及的时间序列分类方法，按论文的类别做一个 **Markdown 化归纳**：每类包含 **(1) 预处理/数据要求**、**(2) 该类内方法的准确率排名（按论文表格的 ACC 排名）**、**(3) 你该在占位符处放哪张原文图**、以及 **(4) 可直接套你 HFCT 片段数据的 Python 代码（以 `aeon` 为主）**。  
（说明：论文的排名来自其在 112 个 UCR UTSC 数据集上、30 次 resample 的平均表现；你的 HFCT 结果会因信号分布与采样方式不同而变化。）

---

## 0. 你 HFCT 片段数据的统一约定（适配论文这些方法）

**你的数据结构（按你描述）**

- 每个样本：一个不等长（或等长）的 1D 信号片段 `x_i`（列表/np.array）
    
- 标签：`y_i ∈ {0,1}`，0=噪声，1=PD
    
- 建议额外列：`group_id`（测量批次/测量文件 ID，用于 Group Split，避免泄漏）
    

**关键点：等长 vs 不等长**

- 论文实验大多在 **等长** UCR 上做；很多算法/实现默认也更偏向等长。
    
- 少数算法（例如 RDST）原生支持不等长（`capability:unequal_length = Yes`）。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html "https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html"))
    
- `ProximityForest` 的 aeon 实现明确是 **univariate + equal-length**。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/stable/examples/classification/distance_based.html "https://www.aeon-toolkit.org/en/stable/examples/classification/distance_based.html"))
    

---

## 1) Distance-based（基于距离）

### 1.1 数据预处理 / 输入要求

- 核心：定义序列间距离（典型如 DTW），再做 1-NN 或基于距离的树/集成。论文回顾了 DTW、EE、PF、ShapeDTW、GRAIL 等。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- 常见预处理：
    
    - **每段独立 z-normalize（可选）**：对 DTW 类方法常见（是否做取决于你是否希望保留幅值信息作为判别线索）。
        
    - **缺失值不允许**（一般）。
        
    - **长度**：理论上 DTW 可处理不等长，但很多工程实现/加速距离不支持；实践里建议先统一长度，或选能吃不等长输入的实现。
        

### 1.2 类内准确率排名（论文 Table 2 的 ACC）

（括号内是论文表里的 ACC 排名号）

1. **PF (Proximity Forest)**: 0.837 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/2 "Table 2 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **EE (Elastic Ensemble)**: 0.811 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/2 "Table 2 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **1NN-DTW**: 0.756 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/2 "Table 2 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **ShapeDTW**: 0.742 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/2 "Table 2 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
5. **GRAIL**: 0.727 (5) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/2 "Table 2 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

> 论文也提到 ShapeDTW 的 shape descriptor 可包含 slope / wavelet / PAA 等（你领导提的 wavelet coefficient 在这里属于典型 shape descriptor 家族）。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))

### 1.3 图占位符（放论文原图）

- `Fig. 6`：Distance-based 方法谱系/关系图（overview）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 7`：Distance-based 排名图（ranked accuracy）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 6 (overview of distance-based classifiers) from Middlehurst et al. 2024 -->
<!-- FIG PLACEHOLDER: Insert Fig. 7 (ranked test accuracy of distance-based classifiers) from Middlehurst et al. 2024 -->
```

### 1.4 代码（distance-based）

> 依赖：`pip install aeon`；GRAIL 额外 `pip install grailts`（论文提到的 SINK kernel / SVM 流派有 Python 版实现）。([GitHub](https://github.com/TheDatumOrg/grail-python "https://github.com/TheDatumOrg/grail-python"))

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

# -----------------------
# 0) Load your parquet
# -----------------------
df = pd.read_parquet("your_hfct_segments.parquet")
# assume: df["signal"] is list[float], df["label"] is 0/1, df["group_id"] identifies measurement batch/file
X_raw = [np.asarray(x, dtype=np.float32) for x in df["signal"].tolist()]
y = df["label"].to_numpy()
groups = df["group_id"].to_numpy()

# -----------------------
# 1) Optional: resample to fixed length for equal-length-only estimators
# -----------------------
def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    # linear interpolation resampling (fast, no scipy dependency)
    if len(x) == target_len:
        return x
    xp = np.linspace(0.0, 1.0, num=len(x), dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(xq, xp, x).astype(np.float32)

def to_aeon_equal_length_2d(X_list, target_len: int) -> np.ndarray:
    # aeon accepts 2D: (n_cases, n_timepoints) for univariate equal-length
    Xr = np.stack([resample_1d(x, target_len) for x in X_list], axis=0)
    return Xr

# -----------------------
# 2) Group split
# -----------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_raw, y, groups=groups))
X_train_raw = [X_raw[i] for i in train_idx]
X_test_raw  = [X_raw[i] for i in test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# choose a target length (example: median length)
target_len = int(np.median([len(x) for x in X_train_raw]))
X_train_eq = to_aeon_equal_length_2d(X_train_raw, target_len)
X_test_eq  = to_aeon_equal_length_2d(X_test_raw,  target_len)

def eval_clf(name, clf, Xtr, Xte):
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)
    print(
        f"{name:>18s} | "
        f"ACC={accuracy_score(y_test, pred):.3f} "
        f"BALACC={balanced_accuracy_score(y_test, pred):.3f} "
        f"F1={f1_score(y_test, pred):.3f}"
    )

# -----------------------
# 3) Distance-based models (aeon)
# -----------------------
from aeon.classification.distance_based import (
    KNeighborsTimeSeriesClassifier, ElasticEnsemble, ProximityForest, ShapeDTW
)

# 1NN-DTW
eval_clf("1NN-DTW", KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw"), X_train_eq, X_test_eq)

# Elastic Ensemble (EE)
eval_clf("EE", ElasticEnsemble(), X_train_eq, X_test_eq)

# Proximity Forest (PF) - aeon implementation is univariate + equal-length
eval_clf("PF", ProximityForest(n_trees=100, n_splitters=5, random_state=42), X_train_eq, X_test_eq)

# ShapeDTW
eval_clf("ShapeDTW", ShapeDTW(), X_train_eq, X_test_eq)

# -----------------------
# 4) GRAIL (external: grailts)
# -----------------------
# pip install grailts
try:
    from grailts import GRAIL  # API may differ by version; see the package README
    from sklearn.svm import LinearSVC

    # Example skeleton: learn representation -> Linear SVM
    # NOTE: adjust to the exact grailts API; this block is a template.
    grail = GRAIL()  # configure landmarks/kernels per your needs
    Z_train = grail.fit_transform(X_train_raw)  # often works on list of arrays
    Z_test  = grail.transform(X_test_raw)

    svm = LinearSVC()
    svm.fit(Z_train, y_train)
    pred = svm.predict(Z_test)
    print(f"{'GRAIL+LinearSVM':>18s} | ACC={accuracy_score(y_test, pred):.3f}")
except Exception as e:
    print("GRAIL template not executed (install/adjust grailts API):", e)
```

---

## 2) Feature-based（基于特征向量）

### 2.1 数据预处理 / 输入要求

- 思路：把整段序列做 **series→vector** 的特征提取，然后用传统分类器（RF/Rotation Forest 等）。论文在 feature-based 小节明确用 pipeline 视角组织。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- 常见预处理：
    
    - **缺失值处理**（TSFresh/Catch22 常有 replace NaN 选项）
        
    - **标准化**：取决于你是否要保留幅值；也可对部分特征做 robust scaling
        
    - **长度**：特征提取理论上可对不等长工作，但很多封装希望等长；工程上“先统一长度”最省心。
        

### 2.2 类内准确率排名（论文 Table 3 的 ACC）

1. **FreshPRINCE**: 0.855 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/3 "Table 3 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **TSFresh**: 0.799 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/3 "Table 3 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **Catch22**: 0.795 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/3 "Table 3 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **Signatures**: 0.787 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/3 "Table 3 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 2.3 图占位符

- `Fig. 8`：feature pipeline 示意图（transform + classifier）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 9`：feature-based 方法关系图（overview）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 10`：feature-based 排名图（ranked accuracy）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 8 (feature extraction pipeline) -->
<!-- FIG PLACEHOLDER: Insert Fig. 9 (overview of feature-based classifiers) -->
<!-- FIG PLACEHOLDER: Insert Fig. 10 (ranked test accuracy of feature-based classifiers) -->
```

### 2.4 代码（feature-based）

```python
from aeon.classification.feature_based import (
    FreshPRINCEClassifier, TSFreshClassifier, Catch22Classifier, SignatureClassifier
)

# Use equal-length arrays (X_train_eq / X_test_eq) for simplest integration
eval_clf("FreshPRINCE", FreshPRINCEClassifier(random_state=42), X_train_eq, X_test_eq)
eval_clf("TSFresh",     TSFreshClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("Catch22",     Catch22Classifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("Signatures",  SignatureClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
```

---

## 3) Interval-based（基于区间/子区间统计）

### 3.1 数据预处理 / 输入要求

- 思路：从固定 offset 或随机区间抽取若干子区间，对每个区间算统计量/频域量，再用树模型集成。论文定义与动机写得很清楚（抗噪、抓相位依赖片段）。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- 常见预处理：
    
    - 多数实现默认 **等长**（区间位置相对固定/可比）
        
    - 频域特征（如 periodogram）往往内置；你外部不一定需要先做 FFT
        

### 3.2 类内准确率排名（论文 Table 4 的 ACC）

1. **QUANT**: 0.867 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **DrCIF**: 0.864 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **R-STSF**: 0.864 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **CIF**: 0.848 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
5. **STSF**: 0.846 (5) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
6. **RISE**: 0.806 (6) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
7. **TSF**: 0.802 (7) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 3.3 图占位符

- `Fig. 14`：interval-based 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- （可选）`Fig. 15`：QUANT vs DrCIF/TSF 的散点对比（论文用于说明改进幅度）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 14 (ranked test accuracy of interval-based classifiers) -->
<!-- Optional FIG PLACEHOLDER: Insert Fig. 15 (scatter comparisons for interval methods) -->
```

### 3.4 代码（interval-based）

```python
from aeon.classification.interval_based import (
    QUANTClassifier,
    DrCIFClassifier,
    CanonicalIntervalForestClassifier,      # CIF
    SupervisedTimeSeriesForest,             # STSF
    RSTSF,
    RandomIntervalSpectralEnsembleClassifier,  # RISE
    TimeSeriesForestClassifier              # TSF
)

eval_clf("QUANT", QUANTClassifier(random_state=42), X_train_eq, X_test_eq)
eval_clf("DrCIF", DrCIFClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("R-STSF", RSTSF(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("CIF", CanonicalIntervalForestClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("STSF", SupervisedTimeSeriesForest(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("RISE", RandomIntervalSpectralEnsembleClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("TSF", TimeSeriesForestClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
```

---

## 4) Shapelet-based（基于 shapelet / 子序列原型，可解释性强）

### 4.1 数据预处理 / 输入要求

- 核心：shapelet 是“对分类最有判别力的短子序列原型”，最终模型可以解释为“哪些形状在起作用”。
    
- RDST 在 aeon 文档里明确 **支持不等长**（很关键，适合你 HFCT 片段长度不一致的现实）。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html "https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html"))
    

### 4.2 类内准确率排名（论文 Table 6 的 ACC）

1. **RDST**: 0.876 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/6 "Table 6 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **STC**: 0.864 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/6 "Table 6 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **MrSQM**: 0.863 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/6 "Table 6 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **RSF**: 0.801 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/6 "Table 6 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 4.3 图占位符

- `Table 5`：shapelet 方法差异对照表 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 18`：shapelet-based 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Table 5 (key differences in shapelet-based algorithms) -->
<!-- FIG PLACEHOLDER: Insert Fig. 18 (ranked test accuracy of shapelet-based classifiers) -->
```

### 4.4 代码（shapelet-based）

> 注：论文里的 “RSF” 在 `aeon` 1.3.0 的 shapelet-based 列表中**不作为现成分类器暴露**（你可以用下面给的“简化 RSF”自己跑一个 baseline），但 RDST / STC / MrSQM 都有现成实现（MrSQM 在 aeon 被归到 dictionary-based 模块）。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/latest/api_reference/classification.html "https://www.aeon-toolkit.org/en/latest/api_reference/classification.html"))

```python
from aeon.classification.shapelet_based import RDSTClassifier, ShapeletTransformClassifier  # STC
from aeon.classification.dictionary_based import MrSQMClassifier

# RDST supports unequal length (so you can also feed X_train_raw/X_test_raw)
eval_clf("RDST", RDSTClassifier(random_state=42, n_jobs=-1), X_train_raw, X_test_raw)

# STC / MrSQM easiest with equal-length
eval_clf("STC", ShapeletTransformClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("MrSQM", MrSQMClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)

# -----------------------
# Optional: a simplified RSF baseline (not the exact paper RSF)
# -----------------------
from sklearn.tree import DecisionTreeClassifier

class SimpleRSF:
    """
    Simplified Random Shapelet Forest baseline:
    - sample random shapelets from training data for each tree
    - represent each series by min Euclidean distance to each shapelet
    - train a decision tree
    """
    def __init__(self, n_trees=50, n_shapelets=20, min_len=10, max_len=200, random_state=42):
        self.n_trees = n_trees
        self.n_shapelets = n_shapelets
        self.min_len = min_len
        self.max_len = max_len
        self.rng = np.random.default_rng(random_state)
        self.trees_ = []
        self.shapelets_ = []  # list[list[np.ndarray]]

    @staticmethod
    def _mindist(x, s):
        L = len(s)
        if len(x) < L:
            return np.inf
        # sliding windows
        best = np.inf
        for i in range(0, len(x) - L + 1):
            d = np.linalg.norm(x[i:i+L] - s)
            if d < best:
                best = d
        return best

    def _transform(self, X_list, shapelets):
        Z = np.zeros((len(X_list), len(shapelets)), dtype=np.float32)
        for i, x in enumerate(X_list):
            for j, s in enumerate(shapelets):
                Z[i, j] = self._mindist(x, s)
        return Z

    def fit(self, X_list, y):
        self.trees_.clear()
        self.shapelets_.clear()
        X_list = [np.asarray(x, dtype=np.float32) for x in X_list]
        for _ in range(self.n_trees):
            # sample shapelets
            shapelets = []
            for _ in range(self.n_shapelets):
                x = X_list[self.rng.integers(0, len(X_list))]
                L = int(self.rng.integers(self.min_len, min(self.max_len, len(x)) + 1))
                start = int(self.rng.integers(0, len(x) - L + 1))
                shapelets.append(x[start:start+L])
            Z = self._transform(X_list, shapelets)
            tree = DecisionTreeClassifier(random_state=int(self.rng.integers(0, 1_000_000)))
            tree.fit(Z, y)
            self.trees_.append(tree)
            self.shapelets_.append(shapelets)
        return self

    def predict(self, X_list):
        preds = []
        for tree, shapelets in zip(self.trees_, self.shapelets_):
            Z = self._transform(X_list, shapelets)
            preds.append(tree.predict(Z))
        preds = np.stack(preds, axis=0)
        # majority vote
        return (preds.mean(axis=0) >= 0.5).astype(int)

rsf = SimpleRSF(n_trees=50, n_shapelets=25, min_len=10, max_len=200, random_state=42)
rsf.fit(X_train_raw, y_train)
pred = rsf.predict(X_test_raw)
print(f"{'SimpleRSF':>18s} | ACC={accuracy_score(y_test, pred):.3f} F1={f1_score(y_test, pred):.3f}")
```

---

## 5) Dictionary-based（词袋/字典，离散化 + 线性分类器；也很适合解释“哪些词/模式重要”）

### 5.1 数据预处理 / 输入要求

- 论文给了典型流程：滑窗→离散化成“词”→词频直方图→分类器。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- **离散化/词袋法本质上就包含了你领导举例的“wavelet coefficient”同类思想**：把连续信号映射到“可计数的模式特征”空间（这里常用 SFA/DFT 等）。
    
- 典型要求：
    
    - 缺失值不允许
        
    - 多数实现偏好等长（便于窗口/参数搜索），不等长也能做但要更小心参数与窗口比例
        

### 5.2 类内准确率排名（论文 Table 8 的 ACC）

1. **WEASEL v2.0**: 0.874 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **TDE**: 0.861 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **WEASEL v1.0**: 0.845 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **BOSS**: 0.834 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
5. **cBOSS**: 0.833 (5) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 5.3 图占位符

- `Fig. 20`：TS→dictionary model 的示意图（窗口/离散化/计数）([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 24`：dictionary-based 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 20 (dictionary model transformation diagram) -->
<!-- FIG PLACEHOLDER: Insert Fig. 24 (ranked test accuracy of dictionary-based classifiers) -->
```

### 5.4 代码（dictionary-based）

```python
from aeon.classification.dictionary_based import (
    WEASEL_V2, TemporalDictionaryEnsemble, WEASEL, BOSSEnsemble, ContractableBOSS
)

eval_clf("WEASEL_v2", WEASEL_V2(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("TDE", TemporalDictionaryEnsemble(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("WEASEL_v1", WEASEL(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("BOSS", BOSSEnsemble(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("cBOSS", ContractableBOSS(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
```

---

## 6) Convolution / Kernel-based（ROCKET 系：随机卷积核特征 + 线性分类器；速度/精度很强）

### 6.1 数据预处理 / 输入要求

- ROCKET 家族本质是：大量随机卷积核 → 激活图 → summary features → RidgeClassifierCV。论文把它单列成 convolution based。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- 典型要求：
    
    - 通常偏好等长（卷积核固定）
        
    - 建议按段做标准化（或至少去 DC），否则卷积特征可能被幅值偏置主导（但这也可能是你 PD vs 噪声的重要判别信息——需要你实验决定）
        

### 6.2 类内准确率排名（论文 Table 10 的 ACC）

1. **MR-Hydra**: 0.884 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **MultiROCKET**: 0.881 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **MiniROCKET**: 0.874 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **Hydra**: 0.870 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
5. **ROCKET**: 0.868 (5) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
6. **Arsenal**: 0.866 (6) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 6.3 图占位符

- `Fig. 28`：convolution-based 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 28 (ranked test accuracy of convolution-based classifiers) -->
```

### 6.4 代码（convolution-based）

```python
from aeon.classification.convolution_based import (
    MultiRocketHydraClassifier, MultiRocketClassifier, MiniRocketClassifier,
    HydraClassifier, RocketClassifier, Arsenal
)

eval_clf("MR-Hydra", MultiRocketHydraClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("MultiROCKET", MultiRocketClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("MiniROCKET", MiniRocketClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("Hydra", HydraClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("ROCKET", RocketClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("Arsenal", Arsenal(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
```

---

## 7) Deep learning（论文也评了，但你若“回到传统 ML 可解释”可先不做主线）

### 7.1 类内准确率排名（论文 Table 12 的 ACC）

1. **H-InceptionTime**: 0.876 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/12 "Table 12 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **InceptionTime**: 0.874 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/12 "Table 12 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **LiteTime**: 0.869 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/12 "Table 12 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **ResNet**: 0.833 (4) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/12 "Table 12 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
5. **CNN**: 0.727 (5) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/12 "Table 12 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 7.2 图占位符

- `Fig. 33`：deep learning 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 33 (ranked test accuracy of deep learning classifiers) -->
```

### 7.3 代码（deep learning，若你要跑）

> 需要额外依赖（通常是 TensorFlow/Keras）；如果你只做传统 ML，这段可以先跳过。

```python
import inspect
from aeon.classification.deep_learning import (
    TimeCNNClassifier, ResNetClassifier, InceptionTimeClassifier, LITETimeClassifier
)

# CNN in paper ~= TimeCNNClassifier
# LiteTime in paper ~= LITETimeClassifier

eval_clf("CNN(TimeCNN)", TimeCNNClassifier(random_state=42), X_train_eq, X_test_eq)
eval_clf("ResNet", ResNetClassifier(random_state=42), X_train_eq, X_test_eq)
eval_clf("InceptionTime", InceptionTimeClassifier(random_state=42), X_train_eq, X_test_eq)
eval_clf("LiteTime", LITETimeClassifier(random_state=42), X_train_eq, X_test_eq)

# H-InceptionTime: some aeon versions expose custom-filter options via network params;
# below is a safe template that tries to enable "use_custom_filters" if available.
sig = inspect.signature(InceptionTimeClassifier)
if "use_custom_filters" in sig.parameters:
    eval_clf("H-InceptionTime", InceptionTimeClassifier(use_custom_filters=True, random_state=42), X_train_eq, X_test_eq)
else:
    print("H-InceptionTime: your aeon version may not expose 'use_custom_filters'; use InceptionTime as fallback.")
```

---

## 8) Hybrid（HIVE-COTE/集成：精度很强，但可解释性与工程复杂度更高）

### 8.1 数据预处理 / 输入要求

- Hybrid 本质是把多个表示（interval/shapelet/dictionary/rocket 等）模块化组合，论文在 4.8 节给了比较和结构图（如 HIVE-COTE v2 的结构/流程）。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- 工程建议：
    
    - 先用单一强基线（如 QUANT / WEASEL2 / RDST / MR-Hydra）把数据管线跑通，再上 HC2。
        

### 8.2 类内准确率排名（论文 Table 13 的 ACC）

1. **HC2**: 0.891 (1) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/13 "Table 13 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
2. **HC1**: 0.879 (2) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/13 "Table 13 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **TS-CHIEF**: 0.878 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/13 "Table 13 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **RIST**: 0.878 (3) ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/13 "Table 13 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

### 8.3 图占位符

- `Fig. 35`：HIVE-COTE v2 ensemble 结构图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 36`：HIVE-COTE meta-ensemble 流程图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
- `Fig. 37`：hybrid 排名图 ([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1 "Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

```markdown
<!-- FIG PLACEHOLDER: Insert Fig. 35 (HIVE-COTE v2 structure) -->
<!-- FIG PLACEHOLDER: Insert Fig. 36 (HIVE-COTE progression flowchart) -->
<!-- FIG PLACEHOLDER: Insert Fig. 37 (ranked test accuracy of hybrid classifiers) -->
```

### 8.4 代码（hybrid）

> `HC1/HC2/RIST` 在 aeon 里有；`TS-CHIEF` 不在 aeon 1.3.0 的 hybrid 列表中（若你确实要复现 TS-CHIEF，通常要用论文作者的实现/其他库）。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/latest/api_reference/classification.html "https://www.aeon-toolkit.org/en/latest/api_reference/classification.html"))

```python
from aeon.classification.hybrid import HIVECOTEV1, HIVECOTEV2, RISTClassifier

# HC1 ~= HIVECOTEV1, HC2 ~= HIVECOTEV2 (naming in aeon)
eval_clf("HC1(HC-v1)", HIVECOTEV1(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("HC2(HC-v2)", HIVECOTEV2(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)
eval_clf("RIST", RISTClassifier(random_state=42, n_jobs=-1), X_train_eq, X_test_eq)

print("TS-CHIEF: not provided as aeon classifier in 1.3.0; use author implementation if needed.")
```

---

## 9) 论文里的“Best-in-category”对你选型的直接启示（最省事的起步路线）

论文后面把 **每类最优**拿出来对比（8 类 best-of-category）。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/14 "Table 14 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))  
对你的 HFCT（片段二分类、且你关心可解释）我建议你按优先级：

1. **RDST**：shapelet 可解释 + 支持不等长（非常贴合你数据现实）。([aeon-toolkit.org](https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html "https://www.aeon-toolkit.org/en/latest/api_reference/auto_generated/aeon.classification.shapelet_based.RDSTClassifier.html"))
    
2. **QUANT / DrCIF**：传统树模型特征，工程稳，很多统计/频域特征天然解释。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/4 "Table 4 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
3. **WEASEL v2 / TDE**：词袋法很强，也能解释“哪些词/模式贡献最大”。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/8 "Table 8 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    
4. **MR-Hydra / MultiROCKET**：若你更关心精度+速度（但解释性弱于 shapelet/interval）。([Springer](https://link.springer.com/article/10.1007/s10618-024-01022-1/tables/10 "Table 10 | Bake off redux: a review and experimental evaluation of recent time series classification algorithms | Data Mining and Knowledge Discovery | Springer Nature Link"))
    

---

如果你愿意，我下一步可以把上面代码进一步“工程化”成一个统一入口（配置式选择算法、自动判断是否需要重采样、自动产出特征重要性/shapelet 可视化、并把结果按 batch 分组汇总），直接嵌进你现在的 HFCT pipeline。

