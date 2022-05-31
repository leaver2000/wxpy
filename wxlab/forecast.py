from types import GeneratorType

from typing import NamedTuple, Type
from datetime import datetime
import pandas as pd
import pandas as pd
import xarray as xr

import pyarrow.parquet as pq
import pyarrow as pa


class Table:
    @staticmethod
    def from_pandas(pandas_object: pd.DataFrame | pd.Series):
        ...

    def to_pandas(self) -> pd.DataFrame | pd.Series:
        ...


class BBox(NamedTuple):
    minx: float = -130.0 % 360
    maxx: float = -60.0 % 360
    miny: float = 20.0

    maxy: float = 55.0


def _load(file: str, **kwargs: pd.Timestamp) -> xr.Dataset:
    ds: xr.Dataset = xr.load_dataset(file, engine="pynio")
    if not kwargs:
        return ds

    return ds.expand_dims(kwargs)


class Grib:
    """transform grib data to various types"""

    def __init__(
        self,
        data: pd.Series | dict[datetime, str] | datetime,
        mapping={
            "lv_ISBL0": "hPa",
            "lat_0": "lat",
            "lon_0": "lon",
            "TMP_P0_L100_GLL0": "temp",
            "UGRD_P0_L100_GLL0": "u_wind",
            "VGRD_P0_L100_GLL0": "v_wind",
        },
    ) -> None:

        if isinstance(data, str):
            if data.endswith(".parquet"):
                table: Table = pq.read_table(data)
                self._data: xr.Dataset = table.to_pandas().to_xarray()

            elif data.endswith((".GR2", ".gr2")):
                self._data = _load(data).rename(mapping)
            else:
                raise NotImplementedError("is file type is not supported")

        elif hasattr(data, "items"):
            self._data = (
                # this allows us to open several grib files in a generator function
                # because we are using grib data that is for a point in time in many cases
                # the mapping object shopuld use a timestamp as the key to the path.
                _load(file, validTime=[pd.to_datetime(timestamp).value]).rename(mapping)
                for timestamp, file in data.items()
            )
        else:
            raise NotImplementedError

    def to_dataset(self, bbox: BBox = BBox()) -> xr.Dataset:
        """the first step in reading a grib"""

        if isinstance(self._data, GeneratorType):
            ds = xr.concat(self._data, dim="validTime")
            condition = (ds.lon >= bbox.minx) & (ds.lon <= bbox.maxx) & (ds.lat >= bbox.miny) & (ds.lat <= bbox.maxy)
            self._data = ds.where(condition, drop=True)

        return self._data

    def to_dataframe(self) -> pd.DataFrame:
        if self._data is None:
            self.to_dataset()

        df = self._data.to_dataframe()

        df.columns.set_names("elements", inplace=True)

        return (
            df.unstack("hPa")
            .swaplevel(0, 1, axis=1)  # .reorder_levels(["hPa", "elements"], axis=1)
            .reorder_levels(["validTime", "lat", "lon"])
        )

    def to_table(self) -> Table:
        df = self.to_dataframe().stack("hPa")
        return pa.Table.from_pandas(df)

    def to_parquet(self, filepath: str):
        table = self.to_table()
        return pq.write_table(table, filepath)


def read_parquet(file: str, kwargs) -> pd.DataFrame:
    return pq.read_table(file, **kwargs).to_pandas()


def to_dataset(
    fp: pd.Series,
    bbox: BBox = BBox(),
    mapping={
        "lv_ISBL0": "hPa",
        "lat_0": "lat",
        "lon_0": "lon",
        "TMP_P0_L100_GLL0": "temp",
        "UGRD_P0_L100_GLL0": "u_wind",
        "VGRD_P0_L100_GLL0": "v_wind",
    },
    **kwargs,
):
    def generate_dataset() -> xr.Dataset:
        for timestamp, file in fp.items():
            ds: xr.Dataset = xr.load_dataset(file, engine="pynio")

            yield ds.expand_dims({"validTime": [timestamp.value]}).rename(mapping)

    ds = xr.concat(generate_dataset(), dim="validTime")
    if isinstance(bbox, BBox):
        condition = (ds.lon >= bbox.minx) & (ds.lon <= bbox.maxx) & (ds.lat >= bbox.miny) & (ds.lat <= bbox.maxy)
        ds = ds.where(condition, drop=True)
    return ds


def to_dataframe(ds: xr.Dataset, **kwargs):
    if not isinstance(ds, xr.Dataset):
        ds = to_dataset(ds, **kwargs)

    df = ds.to_dataframe()
    df.columns.set_names("elements", inplace=True)

    return df.unstack("hPa").reorder_levels(["validTime", "lat", "lon"]).reorder_levels(["hPa", "elements"], axis=1)
