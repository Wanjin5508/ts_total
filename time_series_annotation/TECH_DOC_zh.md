# 时间序列标注 GUI —— 技术概览（中文版）

## 1. 目标与范围

`time_series_annotation` 包实现了一个基于 PyQt5/pyqtgraph 的桌面应用，用于查看多通道时间序列、标注感兴趣区间并导出标注结果。本文档说明系统结构、数据流以及扩展各核心组件的方式。

## 2. 架构速览

| 模块 | 主要职责 |
| --- | --- |
| `main_window.py` | 构建 GUI，连接各类控件，持有 `Dataset` 与 `TimeSeriesPlot` 实例，统筹预筛选、手工标注、导入导出以及分块导航。 |
| `timeseries_plot_qt.py` | 自定义 `pg.PlotWidget` 子类，负责曲线绘制、十字准线、区间选择、峰值展示、区间渲染以及与主窗口的信号通信。 |
| `dataset_async.py` | 异步加载 `.npy/.npz` 文件，将数据切分为固定大小的块，按 `sample_rate` 下采样，并提供块内/原始索引互转工具。 |
| `comment_rectangle.py` | 定义交互式区间遮罩 `CommentLinearRegionItem`，支持选择、高亮、Ctrl 拖拽以及导出状态切换。 |
| `preselection.py` | 提供 `PreSelector` 启发式算法，扫描信号寻找异常片段并生成候选区间。 |
| `export.py` | 维护按记录器划分的标注 DataFrame，去重区间，并将合并结果写入 Parquet。 |
| `annotation.py` | 保存 UI 与导出流程中使用的标签词表。 |
| `color_scheme.py` | 定义曲线、预选区间、选中区间等场景的统一配色。 |
| `paths.py` | 集中管理原始/滤波数据及导出目录的路径配置。 |
| `pipeline.py` | 批处理低通滤波工具，用于离线滤波与绘图校验。 |
| `augmentation.py` | 针对导出 Parquet 数据的基础增强函数（平移、缩放）。 |
| `performance_test.py` | 便捷脚本，用于载入并分析 `cProfile` 输出。 |

## 3. 数据流

1. **数据源配置**：在启动 GUI 前，将一个或多个 `.npy/.npz` 路径写入 `Dataset.file_path`（可在引导脚本中完成）。路径根配置集中在 `paths.py`。  
2. **分块加载**：`Dataset`（`dataset_async.py`）内存映射/加载每个文件，计算 `chunk_cnt`，并异步执行 `read_chunk`，实现多通道并行读取。每个块经 `array2df` 变为 Polars DataFrame，可按 `sample_rate` 下采样。  
3. **可视化**：`MainWindow`（`main_window.py`）以 DataFrame 元组实例化 `TimeSeriesPlot`。绘图会按通道着色，叠加十字准线，提供鼠标交互，并将区间更新、偏移测量等信号发回主窗口。  
4. **标注流程**：用户可绘制或接受 `PreSelector` 生成的区间，调整边界、填写注释，并以 `CommentLinearRegionItem` 形式存储。这些控件记录所属通道 (`recid`)、原始采样索引、UI 状态及导出状态。  
5. **导出**：选中区间使用 `Dataset.backorigin()` 转回绝对索引，通过 `OutputDF`（`export.py`）入库。导出时按记录器拼接 DataFrame、去重，并在 `export/` 目录下写入带时间戳的 Parquet。  
6. **后处理**：`pipeline.py`（滤波）与 `augmentation.py`（平移/缩放片段）等离线脚本基于导出数据做 ML 训练或 QA。性能分析可借助 `performance_test.py`。  

## 4. GUI 层细节

- **主界面布局**（`time_series_annotation/main_window.py`）  
  - 采用垂直分割器：上方为绘图，下方按“区间选择 / 绘图与计算 / 导入与导出”分组堆叠控件。  
  - 按记录器维护多组参数（`distance`、`height`、`window_size` 等），以便自动化与手动调节互不影响。  
  - 分块管理依赖 `user_chunk_size`、`user_sample_rate` 创建 `Dataset`，导航时更新 `Dataset.chunk_idx` 并通过 `TimeSeriesPlot.get_new_chunk()` 重绘。  
  - 暴露大量槽函数处理按钮/文本框事件（`on_preselection`、`on_pick_all`、`on_clear_plot`、`on_chunk_size_given` 以及导入导出动作），同时操作数据后端与遮罩层。  
  - 菜单栏连接文件选择（`select_files`）与保存/导出流程。  

- **绘图组件**（`time_series_annotation/timeseries_plot_qt.py`）  
  - 继承自 `pg.PlotWidget`，增加共享十字准线、偏移信息面板及基于 `Dataset.file_path` 的图例。  
  - 同步维护多类列表：曲线、包络、峰值、预选区间、导出区间（按 chunk 索引）。  
  - 区间集合变化时发出 `line_region_item_list_update`，便于主窗口刷新计数或状态。  
  - 支持曲线位移、显示峰值/包络、鼠标事件（点击选区、拖动绘制、移动十字准线），并在切换 chunk 时同步导出区间的显隐。  
  - 与 `CommentLinearRegionItem` 协同设定配色、响应选中事件，导出区间会以灰色显示并锁定。  

- **区间控件**（`time_series_annotation/comment_rectangle.py`）  
  - `pg.LinearRegionItem` 子类，附加 `recid`、`comment`、原始索引与状态标记。  
  - 支持 Ctrl 拖拽微调、右键恢复、点击发射信号，主窗口可据此展示或编辑注释。  
  - 使用 `color_scheme.py` 中的颜色区分默认、选中与已导出状态。  

## 5. 自动化与分析

- **预筛选**（`time_series_annotation/preselection.py`）  
  - `PreSelector` 将当前 chunk 按 `segment_length` 切片，计算峰峰值与正信号均值，根据 `height` 阈值及 IQR 统计判定异常段。  
  - 生成的 `(start_idx, end_idx)` 列表可直接转换为 `CommentLinearRegionItem` 供用户批量确认。  

- **峰值与包络**（`timeseries_plot_qt.py`）  
  - 借助 SciPy (`hilbert`, `find_peaks`) 计算包络并标亮峰值。  
  - 包络颜色与主窗口中的 `reference_line_combobox` 选项对应，为跨通道对齐提供参考。  
  - 峰值索引可用于辅助预筛选或导出。  

- **数据集工具**（`dataset_async.py`）  
  - `backorigin()` 将 chunk 相对索引转换为原始数组索引，确保导出区间与原始信号对齐。  
  - `origin2chunk()` 执行逆转换，便于导入功能跳转到包含指定区间的 chunk。  
  - `AnalyzerBase` 为电压归一化或多通道分析提供基础类。  

## 6. 导出与后处理

- `OutputDF`（`time_series_annotation/export.py`）按记录器维护 Polars DataFrame，通过 `unique(subset=["recid", "signal"])` 去重，并将结果写入 Parquet。  
- 导出目录默认位于代码旁的 `export/`，可通过 `default_export_dir` 调整。  
- `time_series_annotation/augmentation.py` 提供基础数据增强：  
  - `shift_signal`：按样本数循环平移每段信号。  
  - `scale_amplitude`：等比例缩放幅度，用于数据增强或归一化实验。  

## 7. 运行方式

1. **准备路径**：设置 `Dataset.file_path`（可在 `main_window.py` 实例化 `MainWindow` 前或独立脚本中完成），指向 `.npy` 或 `.npz` 录波文件。  
2. **安装依赖**：PyQt5、pyqtgraph、numpy、scipy、polars、pandas。  
3. **启动应用**：在仓库根目录运行  
   ```bash
   python -m time_series_annotation.main_window
   ```  
   若包不可见，需调整 `PYTHONPATH`。  
4. **加载数据**：在 GUI 内通过 *File → Load .npy or .npz Files* 更新 `Dataset.file_path` 并触发重载。  
5. **标注与导出**：使用预筛选、手动绘制、注释框以及导出按钮，在 `export/` 下生成 Parquet。  

## 8. 可扩展性建议

- **新增分析能力**：基于 `AnalyzerBase` 或在 `MainWindow` 注入新的预处理流程，对 DataFrame 元组做转换后再传给 `TimeSeriesPlot`。  
- **自定义预筛选**：扩展 `PreSelector.detect_abnormal_intervals()`，返回 `(start_idx, end_idx)` 列表即可，无需改动主界面逻辑。  
- **附加元数据**：如需记录严重等级或标签，可扩展 `CommentLinearRegionItem` 字段，并同步更新 `OutputDF.schema`。  
- **性能建议**：调节 `Dataset.chunk_size` 平衡加载耗时与交互流畅度；使用 `performance_test.py` 进行 cProfile 分析。  

## 9. 相关脚本

- `pipeline.py` —— 演示离线低通滤波（Butterworth、零相位）与快速绘图，方便 QA。  
- `performance_test.py` —— 载入 `restats` 并按累计耗时列出热点，用于调优 chunk 或渲染路径。  
- `augmentation.py` —— 目前提供平移/缩放示例，可扩展至噪声注入、随机裁剪等增强策略。  

---

本中文版文档与英文版内容等价，帮助新成员快速理解 GUI 体系、标注数据流以及可扩展的切入点。
