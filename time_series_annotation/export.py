import os
import typing
from datetime import datetime
import polars as pl
import numpy as np
import polars.datatypes as DataType
from dataclasses import dataclass
from typing import Tuple


class OutputDF:
    def __init__(self, num_rec: int, selected_intervals: list=[], default_export_dir="export"):
        self.schema = {
            "recid": pl.Int64,
            "signal": pl.List(pl.Int64),  # for np array
            "comments": pl.Utf8,
            "file_path": pl.Utf8,
            "range": pl.List(pl.Int64)  # for tuple of (start, end)
        }

        self.output_df = pl.DataFrame(
            schema=self.schema  # strict= False
        )

        self.num_rec = num_rec
        self.output_df_list = [pl.DataFrame(schema=self.schema) for i in range(num_rec)]
        self.default_export_dir = default_export_dir
        self._ensure_export_dir()

    def _ensure_export_dir(self):
        if not os.path.exists(self.default_export_dir):
            os.makedirs(self.default_export_dir)

    def init_output_df(self):
        self.output_df = pl.DataFrame(
            schema=self.schema  # strict= False
        )

    def init_output_df_list(self):
        self.output_df_list = [pl.DataFrame(schema=self.schema) for i in range(self.num_rec)]

    def add_row(self, recid, signal, comment, file_path, start, end):
        """
        add a section of source data as a row in the output dataframe

        :param signal: np array
        :param comment: str
        :param file_path: str
        :param start, end: int
        """

        new_row = pl.DataFrame(
            [
                {
                    "recid": recid,
                    "signal": signal.tolist(),
                    "comments": comment,
                    "file_path": file_path,
                    "range": [start, end]
                }
            ],
            schema=self.schema
        )

        # self.output_df = pl.concat([self.output_df, new_row])
        # self.output_df = self.output_df.vstack(new_row)
        # print(f'len(self.output_df_list) {len(self.output_df_list)}')
        # self.output_df_list[recid] = self.output_df_list[recid].filter(~pl.col("signal").is_in(new_row["signal"]))

        # self.output_df_list[recid].remove(pl.col("signal") == signal.tolist())
        # for row in self.output_df_list[recid].iter_rows():
        #     if row[1] == new_row[1]:
        #         row[2] == new_row[2]

        self.output_df_list[recid] = self.output_df_list[recid].vstack(new_row)
        self.output_df_list[recid] = self.output_df_list[recid].unique(subset=["recid", "signal"], maintain_order=True, keep="last")
        self.get_concated_df()

    def get_concated_df(self):
        self.output_df = pl.concat(self.output_df_list)

    def export_to_parquet(self, existing_file=None, target_name=None):
        self.get_concated_df()
        if existing_file is None:
            # export in a new parquet file with timestamp as name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f'export_{timestamp}.parquet'
            file_path = os.path.join(self.default_export_dir, target_name)
            to_export = self.output_df
        else:
            # export to the existing file
            file_path = existing_file
            # for idx in range(len(self.output_df_list)):
            #     self.output_df_list[idx].write_parquet(file_path)
            existing_df = pl.read_parquet(file_path)

            to_export = pl.concat([self.output_df, existing_df])
            to_export = to_export.with_columns(
                (pl.col("recid") * 2).alias("recid")
            )
        to_export.write_parquet(file_path)

        self.init_output_df()
        self.init_output_df_list()

    def get_dataframe(self, recid: int) -> pl.DataFrame:
        return self.output_df_list[recid]

    def get_dataframe_list(self):
        return self.output_df_list
