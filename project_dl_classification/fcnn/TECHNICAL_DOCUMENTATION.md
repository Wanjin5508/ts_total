# FCNN-Based Partial Discharge Classifier — Technical Documentation

## 1. Executive Summary

The Fully Convolutional Neural Network (FCNN) project under `project_dl_classification/fcnn` ingests annotated time-series segments exported as Parquet files from the `time_series_annotation` GUI. The model predicts whether a segment contains partial discharge (PD) events. This document describes the end-to-end design, implementation details, and operational considerations for engineering, ML, and operations teams.

## 2. System Context

```
GUI Annotator (.parquet, labels) ──▶ FCNN Pipeline ──▶ Deployed PD Classifier
                                             │
                                   Explainability (CAM)
```

- **Inputs**: Parquet table with at least two columns:
  - `tds`: raw time-domain signal (1-D array or list of samples).
  - `cluster_id`: integer label emitted by annotators (`-1`/`0` etc. are remapped internally).
- **Outputs**:
  - Trained FCNN weights (`state_dict`).
  - Evaluation metrics (loss curves, test accuracy).
  - Class Activation Maps (CAM) for interpretability.

## 3. Code Structure

| File | Responsibility |
| --- | --- |
| `fcnn_model.py` | Defines the FCNN architecture (Conv1d blocks + GAP + linear head) and initialization strategy. |
| `fcnn_dataset.py` | Wraps the Parquet dataframe into a PyTorch `Dataset` and custom `collate_fn` for padding variable-length signals. |
| `fcnn_hyperparameters.py` | Central repository for core knobs (channels, kernel size, learning rate, epochs). |
| `fcnn_train.py` | Orchestrates data split, training loop, validation/test evaluation, CAM visualization, and kernel plotting. |

## 4. Data Pipeline

1. **Load Parquet**: `pandas.read_parquet('data/processed/partial_discharge_data.parquet')`.
2. **Split**: 80/10/10 into train/val/test via dataframe slicing.
3. **Dataset**: `PartialDischargeDataset` converts each row to:
   - `input`: downsampled signal `sample['tds'][::4]` → `torch.FloatTensor`.
   - `target`: integer class derived from `cluster_id`.
4. **Collate**: `collate_fn` packs a batch of variable-length tensors using `pad_sequence`, returning `(batch, 1, seq_len)` for Conv1d compatibility.
5. **DataLoaders**: standard PyTorch loaders with shuffling for train set.

## 5. Model Architecture

```
Input (B, C=1, L) 
 └─ Conv1d -> ReLU
 └─ Conv1d -> ReLU
 └─ AdaptiveAvgPool1d(1)  # Global Average Pooling
 └─ Linear -> logits (num_classes = 2)
```

- No fully connected layers between convolution blocks and GAP, preserving temporal invariance.
- Weight initialization:
  - Kaiming Normal for Conv1d layers to stabilize ReLU activations.
  - Xavier Uniform (`gain=0.1`) for the final linear layer.

## 6. Training & Validation

- **Optimizer**: `torch.optim.Adam`.
- **Loss**: `nn.CrossEntropyLoss`.
- **Device Placement**: automatic CUDA/CPU selection with `model.to(device)` and per-batch transfer.
- **Logging**:
  - `lossi`, `tr_loss`, `vl_loss` arrays track batch and averaged losses.
  - Validation sampling supports both random batches and targeted class sampling via `dataset_val.get_sample_for_id`.
- **Hyperparameters** (default):
  - `in_channels=1`, `nr_featuremaps=2`, `kernel_size=7`, `batch_size=32`, `lr=1e-3`, `epochs=100`.

## 7. Evaluation & Explainability

1. **Test Loop**: iterates over `dataloader_test`, computes accuracy, and aggregates loss.
2. **Class Activation Map (CAM)**:
   - Forward pass with `return_conv=True` to collect final conv feature maps.
   - Multiply feature maps by class-specific weights from the linear head to obtain contribution scores per time step.
   - Visualize with Plotly scatter plot, color-coding signal points by CAM intensity.
3. **Kernel Visualization**: Iterates through `Conv1d` layers, plotting learned kernels per feature map for debugging.

## 8. Operational Considerations

- **Data Contracts**: Ensure GUI-exported Parquet files preserve `tds` as numeric sequences and `cluster_id` as integral labels. Missing columns or inconsistent sampling rates will break the dataset logic.
- **Versioning**: Track model checkpoints alongside the GUI release that produced the annotations to maintain traceability.
- **Performance**:
  - Memory usage scales with the padded sequence length inside a batch. Consider bucketing by length for larger deployments.
  - GPU acceleration is optional but recommended for long sequences (>10k samples).
- **Explainability**: CAM plots should be archived for compliance when deploying in regulated HV environments.

## 9. Future Enhancements

- Integrate data normalization/scaling in `PartialDischargeDataset`.
- Generalize the FCNN head to multi-class PD taxonomy (beyond binary).
- Replace manual validation sampling with a proper validation dataloader iteration.
- Package training/evaluation into CLI entry points for CI/CD integration.

---
For questions, contact the ML Engineering team or refer to the repository README for environment setup instructions.
