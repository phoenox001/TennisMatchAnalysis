# dask_wrapper.py

import os

# Set this to True to use Dask, False to use Pandas
USE_DASK = os.environ.get("USE_DASK", "0") == "1"

if USE_DASK:
    import dask.dataframe as dd
    import dask
    import numpy as np
    import dask.array as da
    from dask.base import compute

    print("Using Dask DataFrame")

    # Wrappers for common operations
    read_csv = dd.read_csv
    concat = dd.concat
    merge = dd.merge
    to_datetime = dd.to_datetime

    def to_pandas(df):
        return df.compute()

    def dropna(df, **kwargs):
        return df.dropna(**kwargs)

    def fillna(df, **kwargs):
        return df.fillna(**kwargs)

    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)

    def groupby(df, *args, **kwargs):
        return df.groupby(*args, **kwargs)

    def apply(df, func, **kwargs):
        return df.map_partitions(lambda d: d.apply(func, **kwargs), meta=df)

    def astype(df, dtype_dict):
        return df.astype(dtype_dict)

    def rename(df, **kwargs):
        return df.rename(**kwargs)

    def drop(df, **kwargs):
        return df.drop(**kwargs)

    def loc(df, *args):
        return df.loc[*args]

    def compute_df(df):
        return df.compute()

    def dask_merge_asof(left, right, **kwargs):
        import pandas as pd

        # Falls Dask-DataFrames, konvertiere sie zu Pandas
        if hasattr(left, "compute"):
            left = left.compute()
        if hasattr(right, "compute"):
            right = right.compute()

        merged = pd.merge_asof(left, right, **kwargs)

        return dd.from_pandas(merged, npartitions=10)

    def dask_notna(df_or_series):
        import pandas as pd

        # Works for pandas or dask DataFrame/Series
        if isinstance(df_or_series, dd.DataFrame) or isinstance(
            df_or_series, dd.Series
        ):
            return df_or_series.map_partitions(lambda part: part.notna())
        else:
            return pd.notna(df_or_series)

    def DateOffset(*args, **kwargs):
        import pandas as pd

        # Just return pandas DateOffset object — this is not dataframe specific
        return pd.DateOffset(*args, **kwargs)

    def NaT():
        import pandas as pd

        # pd.NaT ist ein Singleton, es gibt keine dask-Alternative, also einfach zurückgeben
        return pd.NaT

    def notna(s):
        import pandas as pd

        if isinstance(s, pd.Series):
            return s.notna()
        elif isinstance(s, dd.Series):
            # map_partitions gibt eine Dask Series mit bool zurück
            return s.map_partitions(lambda part: part.notna())
        else:
            raise TypeError(f"Unsupported type {type(s)}")

    def from_pandas(data, npartitions):
        return dd.from_pandas(data, npartitions=npartitions)

else:
    import pandas as dd
    import numpy as np
    from pandas import to_datetime

    print("Using Pandas DataFrame")

    read_csv = dd.read_csv
    concat = dd.concat
    merge = dd.merge
    to_pandas = lambda df: df

    def dropna(df, **kwargs):
        return df.dropna(**kwargs)

    def fillna(df, **kwargs):
        return df.fillna(**kwargs)

    def sort_values(df, *args, **kwargs):
        return df.sort_values(*args, **kwargs)

    def groupby(df, *args, **kwargs):
        return df.groupby(*args, **kwargs)

    def apply(df, func, **kwargs):
        return df.apply(func, **kwargs)

    def astype(df, dtype_dict):
        return df.astype(dtype_dict)

    def rename(df, **kwargs):
        return df.rename(**kwargs)

    def drop(df, **kwargs):
        return df.drop(**kwargs)

    def loc(df, *args):
        return df.loc[*args]

    def iloc(df, *args):
        return df.iloc[*args]

    def compute_df(df):
        return df
