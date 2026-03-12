# Time Series Annotation GUI — Technical Overview

## 1. Purpose & Scope

The `time_series_annotation` package implements a PyQt5/pyqtgraph desktop application for reviewing multi-channel time-series data, marking interesting intervals, and exporting the resulting annotations. This document explains how the system is structured, how data flows through it, and how to extend the main components.

## 2. Architecture at a Glance

| Module | Responsibility |
| --- | --- |
| `main_window.py` | Builds the GUI, wires widgets, owns `Dataset` and `TimeSeriesPlot` instances, orchestrates preselection, annotation, export, and chunk navigation. |
| `timeseries_plot_qt.py` | Custom `pg.PlotWidget` subclass handling data visualization, crosshairs, selection, peak display, interval drawing, and communication back to the main window. |
| `dataset_async.py` | Loads `.npy/.npz` files asynchronously, slices data into fixed-size chunks, downsamples, and exposes helper utilities for mapping between chunk-relative and absolute indices. |
| `comment_rectangle.py` | Defines `CommentLinearRegionItem`, the interactive interval overlay supporting selection, dragging (with Ctrl), and export state toggling. |
| `preselection.py` | Contains heuristics (`PreSelector`) that scan signals for abnormal segments and build candidate intervals for review. |
| `export.py` | Manages per-recorder annotation dataframes, deduplicates intervals, and writes merged results to Parquet. |
| `annotation.py` | Holds the label vocabulary used throughout the UI/export workflow. |
| `color_scheme.py` | Provides consistent color palettes for plotted signals and interval states. |
| `paths.py` | Centralizes filesystem locations for raw/filtered datasets and export roots. |
| `pipeline.py` | Batch utilities for low-pass filtering raw data and saving/plotting the results. |
| `augmentation.py` | Simple signal augmentation helpers (shift, scale) operating on exported parquet datasets. |
| `performance_test.py` | Convenience script for loading and inspecting `cProfile` output. |

## 3. Data Flow

1. **Source configuration** — Before launching the GUI, populate `Dataset.file_path` (e.g., in a bootstrap script) with one or more `.npy/.npz` files. File metadata roots live in `paths.py`.
2. **Chunked loading** — `Dataset` (`dataset_async.py`) memory-maps/loads each file, computes `chunk_cnt`, and spawns asynchronous `read_chunk` tasks so multiple channels load in parallel. Each chunk becomes a Polars dataframe via `array2df`, optionally downsampled by `sample_rate`.
3. **Visualization** — `MainWindow` (`main_window.py`) instantiates `TimeSeriesPlot` with the tuple of dataframes. The plot renders each recorder in its own color, overlays crosshairs, exposes mouse-driven selection, and emits signals (interval updates, offset measurements) back to the window.
4. **Annotation workflow** — Users draw or accept preselected intervals (`PreSelector`), edit bounds, add comments, and store selections as `CommentLinearRegionItem` overlays. These widgets keep track of their recorder (`recid`), original sample indices, UI state, and export status.
5. **Export** — Selected intervals are converted back to absolute indices with `Dataset.backorigin()` and stored through `OutputDF` (`export.py`). When exporting, per-recorder dataframes are concatenated, deduplicated, and written to timestamped parquet files under `export/`.
6. **Post-processing** — Offline scripts in `pipeline.py` (filtering) and `augmentation.py` (shifting/scaling annotated snippets) operate on exported data for ML training or QA. Performance profiling can be reviewed via `performance_test.py`.

## 4. GUI Layer Details

- **Main layout** (`time_series_annotation/main_window.py`)  
  - Uses a vertical splitter: the plot occupies the top pane, controls live in grouped sections below (“Interval Selection”, “Plot and Calculation”, “Import and Export”).  
  - Keeps per-recorder state (`distance`, `height`, `window_size`, etc.) so automation and manual adjustments are scoped per signal.  
  - Chunk management: `user_chunk_size` and `user_sample_rate` drive `Dataset` creation, while chunk navigation updates `Dataset.chunk_idx` and refreshes the plot via `TimeSeriesPlot.get_new_chunk()`.  
  - Exposes numerous slots for buttons/line edits (`on_preselection`, `on_pick_all`, `on_clear_plot`, `on_chunk_size_given`, export/import actions) to modify both the data backend and the plot overlays.  
  - Connects the menu bar to file loading helpers (see `select_files` flow) and saving/export routines.

- **Plot widget** (`time_series_annotation/timeseries_plot_qt.py`)  
  - Derives from `pg.PlotWidget`, adds shared crosshair, offset info panel, and a legend keyed by `Dataset.file_path`.  
  - Maintains multiple lists in parallel: plotted curves, envelope items, peak markers, preselected intervals, and exported rectangles indexed by chunk.  
  - Emits `line_region_item_list_update` whenever interval collections change, enabling the main window to reflect counts or statuses.  
  - Houses utility methods for shifting traces, toggling peak/envelope overlays, handling mouse events (click to select, drag for intervals, move crosshair), and syncing export rectangles when chunks change.  
  - Coordinates with `CommentLinearRegionItem` instances to set colors, respond to selection, and ensure exported intervals render in grey and become immutable.

- **Interval items** (`time_series_annotation/comment_rectangle.py`)  
  - Subclass of `pg.LinearRegionItem` with extra metadata (`recid`, `comment`, `origin_start/end`, flags).  
  - Supports Ctrl-drag to refine intervals, right-click to reset exported state, and signal emission upon clicks, so `MainWindow` can display/edit comments in its forms.  
  - Uses color palettes from `color_scheme.py` to uniformly distinguish default, selected, and exported intervals.

## 5. Automation & Analytics

- **Preselection** (`time_series_annotation/preselection.py`)  
  - `PreSelector` splits the current chunk into fixed-length segments (`segment_length`), computes peak-to-peak amplitudes and positive signal averages, and marks segments whose amplitude or average exceed neighbors by configurable ratios (`height` threshold, IQR statistics).  
  - The generated `[start, end]` intervals feed directly into the plot as candidate `CommentLinearRegionItem`s, which the user can accept/reject en masse.

- **Peaks & envelopes** (`timeseries_plot_qt.py`)  
  - Uses SciPy (`hilbert`, `find_peaks`) to derive envelopes and highlight peaks.  
  - Envelopes furnish context when aligning spikes across channels; colors correspond to `reference_line_combobox` selections in the main window.  
  - Peak indexes feed into preselection or export exports to capture amplitude-dense areas.

- **Dataset helpers** (`dataset_async.py`)  
  - `backorigin()` converts chunk-relative indices back to the original array space using the chunk size and `sample_rate`, ensuring exported intervals align with the raw recordings.  
  - `origin2chunk()` performs the inverse, letting import routines jump to the chunk containing a previously exported interval.  
  - The `AnalyzerBase` class offers a starting point for voltage normalization or multi-channel analysis pipelines.

## 6. Export & Post-Processing

- `OutputDF` (`time_series_annotation/export.py`) stores one Polars dataframe per recorder, deduplicates rows via `unique(subset=["recid", "signal"])`, and materializes parquet files.  
- The export directory defaults to `export/` adjacent to the code but can be changed via `default_export_dir`.  
- Augmentation utilities (`time_series_annotation/augmentation.py`) operate on exported parquet datasets for downstream ML:  
  - `shift_signal` circularly shifts each stored interval by a sample count.  
  - `scale_amplitude` scales each value, useful for synthetic augmentation or normalization tests.

## 7. Running the Application

1. **Prepare file paths** — Set `Dataset.file_path` (e.g., in `main_window.py` before instantiating `MainWindow`, or via an auxiliary bootstrap script) to point to `.npy` or `.npz` recordings.  
2. **Install dependencies** — PyQt5, pyqtgraph, numpy, scipy, polars, pandas.  
3. **Launch** — From the repository root:  
   ```bash
   python -m time_series_annotation.main_window
   ```  
   Adjust `PYTHONPATH` if necessary so the package resolves.  
4. **Load data** — Use *File → Load .npy or .npz Files* to update `Dataset.file_path` at runtime and trigger `Dataset` reloads.  
5. **Annotate & export** — Utilize preselection, manual selections, comment entry, and export buttons to generate parquet files under `export/`.

## 8. Extensibility Guidelines

- **Adding analysis features** — Build on `AnalyzerBase` or inject new processors in `MainWindow` that operate on the tuple of dataframes before passing them to `TimeSeriesPlot`.  
- **Custom preselection** — Extend `PreSelector.detect_abnormal_intervals()` with additional scoring functions; the main window only requires a list of `(start_idx, end_idx)` per recorder.  
- **Additional metadata** — Enhance `CommentLinearRegionItem` to store extra fields (e.g., severity, tags) and update `OutputDF.schema` accordingly.  
- **Performance considerations** — Keep chunk sizes manageable (`Dataset.chunk_size`) to balance load times and interaction smoothness; profile UI heavy paths with `performance_test.py`.

## 9. Related Scripts

- `pipeline.py` — Demonstrates an offline, reproducible low-pass filtering pipeline (Butterworth, zero-phase) and quick plotting for QA.  
- `performance_test.py` — Loads `cProfile` stats (`restats`) and prints cumulative-time hot spots; useful when tuning chunk sizes or rendering paths.  
- `augmentation.py` — Minimal but illustrative; expand with noise injection, random cropping, or other ML-friendly transforms.

---

This document should give new contributors enough context to navigate the GUI stack, understand how annotations travel from raw files to exported parquet, and identify hook points for additional tooling.
