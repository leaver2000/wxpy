from typing import Iterable, Type, NamedTuple
from datetime import datetime
import json

import pandas as pd
from geopandas import GeoDataFrame
import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from shapely.geometry import Polygon


class BBox(NamedTuple):
    minx: float = -130.0 % 360
    maxx: float = -60.0 % 360
    miny: float = 20.0
    maxy: float = 55.0


class Probsevere:
    def __init__(self, data: str | pd.Series | dict[datetime, str] | pd.DataFrame):
        if isinstance(data, str):
            if data.endswith(".parquet"):
                df: pd.DataFrame = pq.read_table(data).to_pandas()
                if "coordinates" in df.columns:
                    df.insert(0, "geometry", [Polygon(map(tuple, x)) for x in df["coordinates"]])
                    df = GeoDataFrame(df.drop("coordinates", axis=1), geometry="geometry")
                self._data = df

            else:
                raise NotImplementedError
        if isinstance(data, pd.DataFrame):
            self._data = data
        else:
            df = to_dataframe(data, datetime_dtype=np.int64)
            condition = df.columns[df.columns != "geometry"]
            df[condition] = df[condition].astype(np.float32)
            self._data = df

    def to_dataframe(self) -> pd.DataFrame:
        return self._data.copy()

    def to_table(self):
        df = self.to_dataframe()

        try:
            # if isinstance(df, GeoDataFrame):
            df["coordinates"] = np.array([tuple(x.coords) for x in df.geometry.exterior], dtype=object)
            df = df.drop("geometry", axis=1)
        except AttributeError:
            ...
        return pa.Table.from_pandas(df)

    def to_parquet(self, filepath: str):
        table = self.to_table()
        return pq.write_table(table, filepath)

    def to_xarray(self) -> xr.Dataset:
        return self.to_dataframe().to_xarray()


@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, dataframe: pd.DataFrame) -> None:

        self._index = dataframe.index

    @property
    def lat(self) -> NDArray[np.float32]:
        return self._index.unique("lat").to_numpy().astype(np.float32)

    @property
    def lon(self) -> NDArray[np.float32]:
        return self._index.unique("lon").to_numpy().astype(np.float32)


def reshape_geometry(probsevere: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    ps = probsevere.copy()

    def index_abs_argmin(
        ps_bounds: pd.DataFrame,  # (27070, 2)
        grid: NDArray[np.float32],  # (281,)
    ) -> NDArray[np.float32]:  # (27070, 2)
        """
        >>> ps_bounds:GeoDataFrames.geometry.bounds[["minx", "maxx","miny", "maxy"]]
        >>> galwem_grid: NDArray[np.float32]
        """

        # first shaped the probsevere and galwm so that have a common axis
        ps_shaped = ps_bounds.to_numpy()[:, np.newaxis]  # (27070, 1, 2)
        galwem_shaped = grid[:, np.newaxis]  # (141, 1)
        delta = abs(galwem_shaped - ps_shaped)  # (27070, 281, 2)
        # in the delta find the smallest diffrence here -- ^
        index_nearest = np.argmin(delta, axis=1)  # (27070, 2)
        # use the index position of the smallest diff to a grid point
        # to index the grid for a max and min
        return grid[index_nearest]  # (27070, 2)

    ps[["WEST", "EAST"]] = index_abs_argmin(ps.bounds[["minx", "maxx"]] % 360, target.geo.lon)
    # and S,N
    ps[["SOUTH", "NORTH"]] = index_abs_argmin(ps.bounds[["miny", "maxy"]], target.geo.lat)
    return ps.drop(["geometry"], axis=1).set_index(["WEST", "EAST", "NORTH", "SOUTH"], append=True).astype(np.float32)


def align_time(probsevere: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """aligns the time in the probsevere forecast"""

    def sync_time(stack: NDArray[np.int64], source_time: pd.Index) -> NDArray[np.int64]:
        delta = abs(stack - source_time.to_numpy()).astype("timedelta64[ns]")
        condition = delta < pd.to_timedelta(3, unit="h")
        value = np.where(condition, source_time, np.nan).astype(np.int64)
        return np.nanmax(value, axis=1)

    ps = probsevere.copy().reset_index("validTime")

    ps["validTime"] = sync_time(
        ps["validTime"].to_numpy()[:, np.newaxis],
        forecast.index.unique("validTime"),
    )

    return ps.set_index("validTime", append=True)  # .reorder_levels(["validTime", "WEST", "EAST", "NORTH", "SOUTH"])


def meshgrid(df: pd.DataFrame, on_level: list[str] = ["WEST", "SOUTH"]) -> pd.DataFrame:

    west, east, north, south = (df.index.get_level_values(name) for name in ["WEST", "EAST", "NORTH", "SOUTH"])

    condition = (north == south) & (west == east)
    # mesh_grid_condition
    df.loc[condition] = df.loc[condition].reset_index().groupby(["validTime", "WEST", "EAST", "NORTH", "SOUTH"]).mean()
    return df.droplevel(on_level).rename_axis(["validTime", "lon", "lat"]).fillna(0).astype(np.float32).swaplevel(1, 2)


def to_dataframe(files: pd.Series | list[str], datetime_dtype: Type[type] = np.int64) -> pd.DataFrame:
    """from an iterable collection of filepaths to probsevere return a dataframe"""
    if not isinstance(files, list):
        files = files.tolist()

    def generate() -> Iterable[GeoDataFrame]:
        """load function for probsevere dataset"""
        for file in files:
            with open(file, mode="r", encoding="utf8") as fc:
                feat = json.load(fc)
                df = GeoDataFrame.from_features(feat["features"])
                df["validTime"] = feat["validTime"]
                yield df

    ps = pd.concat(generate(), ignore_index=True)

    ps["validTime"] = pd.to_datetime(ps["validTime"], format="%Y%m%d_%H%M%S %Z", utc=True).astype(datetime_dtype)

    ps["AVG_BEAM_HGT"] = ps["AVG_BEAM_HGT"].str.replace(r"[A-Za-z]", "", regex=True).apply(pd.eval)

    ps[["MAXRC_EMISS", "MAXRC_ICECF"]] = (
        ps[["MAXRC_EMISS", "MAXRC_ICECF"]]
        .stack()
        .str.extract(r"(?:\()([a-z]*)(?:\))")
        .replace({"weak": 1, "moderate": 2, "strong": 3})
        .fillna(0)
        .unstack(-1)
        .droplevel(0, axis=1)
    )

    return ps.set_index(["validTime", "ID"])
