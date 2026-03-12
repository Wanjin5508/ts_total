# FCNN+BiLSTM Partial Discharge Classifier — Technical Documentation

## 1. Executive Summary

The `project_dl_classification/fcnn_lstm` package implements a hybrid Fully Convolutional Neural Network (FCNN) plus bidirectional LSTM model for time-series classification of partial discharge (PD) signals. The system consumes GUI-labeled Parquet data, normalizes variable-length signals, and produces a robust classifier with sequence awareness. This document details the architecture, data pipeline, training procedure, and operational considerations for engineering and ML teams.

## 2. System Context

```
Time-Series Annotation GUI  ──▶  Parquet (raw_signal, label)
                                    │
                                    ▼
          Data Loader (normalize, pad, lengths) ──▶ FCNN + BiLSTM ──▶ PD predictions
                                                         │
                                           Metrics, Explainability hooks
```

- **Inputs**: DataFrame columns `raw_signal` (1-D waveform) and `label` (integer class).
- **Outputs**: Trained `CNN_LSTM_Classifier` weights, evaluation metrics, optional embeddings for downstream tasks.

## 3. Repository Structure

| File | Responsibility |
| --- | --- |
| `fcnn_lstm_dataset.py` | Dataset class, normalization, sampling helpers, and collate functions (with or without sequence lengths). |
| `fcnn_lstm_model.py` | Defines the hybrid model: strided Conv1d stack feeding a BiLSTM with packed sequences and linear head. |
| `fcnn_lstm_train.py` | Training and evaluation routines (`train_epoch`, `eval_epoch`) including gradient clipping and accuracy tracking. |
| `utils.py` | Placeholder for shared utilities (e.g., `EarlyStopping`) consumed by training scripts. |

## 4. Data Pipeline

1. **Load Parquet**: Upstream code (not shown) reads GUI exports into a DataFrame.
2. **Dataset** (`PartialDischargeDataset`):
   - Normalizes each `raw_signal` to `[0, 1]` via min-max scaling to stabilize learning across varying amplitudes.
   - Returns `(tensor(1, L_i), label)` pairs where `L_i` differs per sample.
3. **Collate**:
   - `custom_collate_with_lengths` pads sequences to `L_max`, preserves the original lengths tensor, and shapes inputs to `(N, C=1, L_max)` required by Conv1d.
4. **DataLoader**: Wraps dataset with batch size, shuffling, and the custom collate to deliver `(batch, labels, lengths)` tuples to the trainer.

## 5. Model Architecture

### 5.1 Convolutional Front-End

```
Conv1d(1→32, k=5, stride=2, padding=2) + ReLU
Conv1d(32→64, k=5, stride=2, padding=2) + ReLU
Conv1d(64→128, k=3, stride=2, padding=1) + ReLU
```

- Strided convolutions downsample the temporal dimension by 2× per layer (overall 8×), expanding the receptive field while suppressing noise.
- `_conv_cfg` documents the kernel/stride/padding per layer, enabling deterministic computation of post-conv sequence lengths.

### 5.2 BiLSTM Backbone

- `input_size=128` (channels from final Conv1d), `hidden_size` configurable, `num_layers≥1`, `bidirectional=True`.
- Packed sequences (`pack_padded_sequence`) ensure the LSTM processes only valid time steps, preventing padded zeros from degrading gradients.
- Hidden states from the last forward/backward directions are concatenated to form `feat ∈ ℝ^{N×hidden_out}`.

### 5.3 Classification Head

```
Linear(hidden_out → hidden_out/2) → ReLU → Dropout
Linear(hidden_out/2 → num_classes)
```

- Optional LayerNorm is commented but easily re-enabled for stabilization.

## 6. Sequence-Length Handling

`CNN_LSTM_Classifier._lengths_after_conv_stack` applies the Conv1d length formula iteratively:
$$[
L_{\text{next}} = \left\lfloor \frac{L + 2p - d\cdot(k-1)}{s} \right\rfloor + 1
]$$

The resulting `lengths_prime` are clamped to the actual conv output length and fed into `pack_padded_sequence`, ensuring LSTM alignment matches convolutional downsampling.

## 7. Training Procedure

- **Optimizer / Loss**: Typically `Adam` + `CrossEntropyLoss` (configurable in caller script).
- **Gradient Management**: `nn.utils.clip_grad_norm_` with threshold `1.0` prevents exploding gradients when combining CNN and LSTM.
- **Metrics**: `train_epoch` and `eval_epoch` return `(loss, accuracy)` for logging dashboards.
- **Device Support**: Batches and lengths tensors are moved to CUDA when available.

## 8. Operational Guidance

- **Scalability**: Strided convolutions reduce sequence length early, keeping LSTM throughput manageable for long raw signals.
- **Normalization**: Per-sample min-max normalization is suitable when amplitude varies across capture sessions; for production, consider dataset-level scaling or robust scalers saved alongside checkpoints.
- **Explainability**: The last conv activations (available by tapping `model.fcnn`) can be reused to compute CAM/Grad-CAM for regulatory reporting.
- **Early Stopping**: Plug `utils.EarlyStopping` into the training loop to avoid overfitting on relatively small annotated datasets.

## 9. Future Enhancements

- Add explicit data splitting, logging, and checkpointing in `fcnn_lstm_train.py`.
- Support multi-class PD taxonomies and class imbalance mitigation (e.g., focal loss or reweighting).
- Provide end-to-end scripts for CAM visualization leveraging both convolutional and recurrent representations.

## 10. Performance Notes vs. FCNN Baseline

Empirical testing shows the FCNN+BiLSTM hybrid reaches ~99% accuracy, outperforming the standalone FCNN (~94%). Key drivers:

- **Richer temporal context**: The BiLSTM processes bidirectional sequences, capturing long-range dependencies between discharge pulses that a purely convolutional model (with limited receptive field) may miss.
- **Strided convolutional front-end**: By reducing sequence length before the recurrent layers, the model gains hierarchical features while keeping the LSTM tractable, enabling better generalization without overfitting.
- **Packed-sequence training**: Ignoring padded timesteps preserves gradient quality, whereas the FCNN baseline may suffer from padding artifacts when averaging across the entire temporal axis.

The combined effect is sharper discrimination of subtle PD patterns, especially when events are defined by both local morphology and temporal ordering.

For additional context or onboarding assistance, contact the ML Engineering team or refer to project-level READMEs.
