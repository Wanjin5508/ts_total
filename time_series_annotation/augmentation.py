import polars as pl
import numpy as np


def parquet_to_df(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)


def shift_signal(df: pl.DataFrame, shift_amount_in_samples: int) -> pl.DataFrame:
    shifted_df = df.with_columns(
        pl.col("signal").map_elements(lambda x: np.roll(x, shift_amount_in_samples).tolist())
        .alias("signal")
    )

    return shifted_df


def scale_amplitude(df: pl.DataFrame, scale_factor: int) -> pl.DataFrame:
    scaled_df = df.with_columns(
        pl.col("signal").map_elements(lambda x: [i * scale_factor for i in x])
        .alias("signal")
    )

    return scaled_df
