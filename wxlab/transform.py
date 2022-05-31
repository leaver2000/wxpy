__all__ = ["ProbSevere", "Galwem", "unpack_files", "GeoAccessor", "BBox"]
import json
from typing import Iterable, NamedTuple

import pandas as pd
import xarray as xr
import numpy as np
from geopandas import GeoDataFrame

from numpy.typing import NDArray
from geojson import Feature


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


def unpack_files(
    ALL_GALWEM_FILES: list[str], ALL_PROBSEVERE_FILES: list[str], time_buffer: int = 90
) -> tuple[pd.Series, pd.Series]:
    # GALWEM FILES
    galwem = pd.Series(ALL_GALWEM_FILES, name="GALWEM")

    g_times: pd.DataFrame = galwem.str.extract(r"FH.(?P<forecast_hour>\d{3})_DF__(?P<valid_time>\d{8})")

    galwem.index = pd.to_datetime(g_times["valid_time"]) + pd.to_timedelta(
        g_times["forecast_hour"].astype(int), unit="h"
    )
    # PROBSEVERE FILES
    probsevere = pd.Series(ALL_PROBSEVERE_FILES, name="ProbSevere")
    probsevere.index = pd.to_datetime(
        probsevere.str.replace("_", "T").str.extract(r"(\d*T\d*).json", expand=False).rename("validTimes")
    )
    # TIME BUFFER ON EACH END OF THE PROBSEVERE DATA
    buffer = pd.to_timedelta(time_buffer, unit="m")
    condition = (probsevere.index > galwem.index.min() - buffer) & (probsevere.index < galwem.index.max() + buffer)

    return galwem, probsevere[condition]


def index_abs_argmin(
    bounds: pd.DataFrame,  # (27070, 2)
    grid: NDArray[np.float32],  # (281,)
) -> NDArray[np.float32]:  # (27070, 2)
    """
    >>> bounds:GeoDataFrames.geometry.bounds[["minx", "maxx","miny", "maxy"]]
    >>> galwem_grid: NDArray[np.float32]
    """

    # first shaped the probsevere and galwm so that have a common axis
    stacked_bounds = bounds.to_numpy()[:, np.newaxis]  # (27070, 1, 2)
    stacked_grid = grid[:, np.newaxis]  # (141, 1)
    delta = abs(stacked_grid - stacked_bounds)  # (27070, 281, 2)
    # in the delta find the smallest diffrence here -- ^
    index_nearest = np.argmin(delta, axis=1)  # (27070, 2)
    # use the index position of the smallest diff to a grid point
    # to index the grid for a max and min
    return grid[index_nearest]  # (27070, 2)


def iterfiles(filepaths: "pd.Series[str]") -> Iterable[xr.Dataset | Feature]:

    if filepaths.str.endswith(".json").all():
        for timestamp, file in filepaths.items():
            with open(file, mode="r", encoding="utf-8") as fc:
                feat = json.load(fc)
                for f in feat["features"]:
                    f["properties"]["validTime"] = timestamp
                    yield f

    if filepaths.str.endswith(".GR2").all():
        for timestamp, file in filepaths.items():
            yield xr.open_dataset(file, engine="pynio").expand_dims({"validTime": [timestamp.value]})


class BBox(NamedTuple):
    minx: float = -130.0 % 360
    maxx: float = -60.0 % 360
    miny: float = 20.0
    maxy: float = 55.0


class Base:
    def __repr__(self) -> str:
        return self._frame._repr_html_()

    def _repr_html_(self):
        return self._frame._repr_html_()


class Galwem(Base):
    def __init__(self, files: pd.Series, **kwargs) -> None:

        ds: xr.Dataset = xr.concat(iterfiles(files), dim="validTime")

        self._frame: xr.Dataset = ds.rename(
            {
                "lv_ISBL0": "hPa",
                "lat_0": "lat",
                "lon_0": "lon",
                "TMP_P0_L100_GLL0": "temp",
                "UGRD_P0_L100_GLL0": "u_wind",
                "VGRD_P0_L100_GLL0": "v_wind",
            }
        )

    def to_dataframe(self, bbox: BBox = BBox()) -> pd.DataFrame:
        ds = self._frame
        condition = (ds.lon >= bbox.minx) & (ds.lon <= bbox.maxx) & (ds.lat >= bbox.miny) & (ds.lat <= bbox.maxy)
        df = ds.where(
            condition,
            drop=True,
        ).to_dataframe()

        df.columns.set_names("elements", inplace=True)

        return (
            df.unstack("hPa").reorder_levels(["validTime", "lat", "lon"]).reorder_levels(["hPa", "elements"], axis=1)
        )

    @property
    def index(self):
        return self._frame.index


def itertimes(source: pd.DatetimeIndex, target: pd.DatetimeIndex) -> Iterable[tuple[pd.DatetimeIndex, pd.Timestamp]]:

    time_interval = len(target)

    start_time: NDArray[np.int64] = np.argmin(
        abs(target.values[:, np.newaxis] - source.values) > pd.to_timedelta(time_interval, unit="h"),
        axis=1,
    )

    end_time = np.roll(start_time, -1)
    end_time[-1] = -1

    for (i0, i1), target_hour in zip(zip(start_time, end_time), target):
        yield source[i0:i1], target_hour


class ProbSevere(Base):
    def __init__(self, files: pd.Series) -> None:
        df = GeoDataFrame.from_features(iterfiles(files))
        df["validTime"] = pd.to_datetime(df["validTime"], format="%Y%m%d_%H%M%S %Z", utc=True)
        df["AVG_BEAM_HGT"] = df["AVG_BEAM_HGT"].str.replace(r"[A-Za-z]", "", regex=True).apply(pd.eval)

        df[["MAXRC_EMISS", "MAXRC_ICECF"]] = (
            df[["MAXRC_EMISS", "MAXRC_ICECF"]]
            .stack()
            .str.extract(r"(?:\()([a-z]*)(?:\))")
            .replace({"weak": 1, "moderate": 2, "strong": 3})
            .fillna(0)
            .unstack(-1)
            .droplevel(0, axis=1)
        )
        df = df.set_index(["validTime", "ID"])
        df[df.columns != "geometry"] = df[df.columns != "geometry"].astype(np.float32)
        self._frame = df
        self._bounds = self._frame["geometry"].bounds

    @property
    def frame(self) -> GeoDataFrame:
        return self._frame

    @property
    def index(self) -> pd.Index:
        return self._frame.index

    @property
    def columns(self) -> pd.Index:
        return self._frame.columns

    @property
    def bounds(self) -> pd.DataFrame:
        """
        Returns a DataFrame with columns minx, miny, maxx, maxy values containing the bounds for each geometry.

        See GeoSeries.total_bounds for the limits of the entire series.

        Examples
        >>> from shapely.geometry import Point, Polygon, LineString
        >>> d = {'geometry': [Point(2, 1), Polygon([(0, 0), (1, 1), (1, 0)]),
        ... LineString([(0, 1), (1, 2)])]}
        >>> gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
        >>> gdf.bounds
        minx  miny  maxx  maxy
        0   2.0   1.0   2.0   1.0
        1   0.0   0.0   1.0   1.0
        2   0.0   1.0   1.0   2.0
        """
        return self._bounds

    def to_grid_like(self: "ProbSevere", target: pd.DataFrame):
        frame = self.frame.copy()
        # set grid coordinates W,E
        frame[["WEST", "EAST"]] = index_abs_argmin(self.bounds[["minx", "maxx"]] % 360, target.geo.lon)
        # and S,N
        frame[["SOUTH", "NORTH"]] = index_abs_argmin(self.bounds[["miny", "maxy"]], target.geo.lat)
        # we can now drop the geometry and append our grid points to the index

        self._frame = frame.drop(["geometry"], axis=1).set_index(["WEST", "EAST", "NORTH", "SOUTH"], append=True)
        return self

    def index_like(self, target_index: pd.Index):
        source_times: pd.DatetimeIndex = self.index.unique("validTime")
        target_times: pd.DatetimeIndex = pd.to_datetime(target_index.unique("validTime"))
        # df = self.frame.copy()
        # the grid now has to be synced as its possible
        # to have the same value for for the same grid point
        #
        # -----------------------------------------
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # |                se | sw                |
        # -----------------------------------------
        # |                ne | nw                |
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # |                   |                   |
        # -----------------------------------------
        source = self.frame.copy().droplevel("ID")

        def generate() -> pd.DataFrame:

            for source_datetime, target_hour in itertimes(source_times, target_times):

                frame = source.loc[source_datetime, :].groupby(["WEST", "EAST", "NORTH", "SOUTH"]).mean()

                index_names = frame.index.names

                west, east, north, south = (frame.index.get_level_values(name) for name in index_names)

                mesh_grid_condition = (north == south) & (west == east)
                # NOTE NEEDS VALIDATION
                frame.loc[mesh_grid_condition] = frame.loc[mesh_grid_condition].groupby(index_names).mean()

                frame = frame.droplevel(["WEST", "SOUTH"])

                frame.index = frame.index.set_names(["lon", "lat"])

                frame["validTime"] = target_hour.value

                yield frame.reset_index().set_index(["validTime", "lat", "lon"], append=True)

        df = pd.concat(generate(), axis=0).droplevel(0)

        container = pd.DataFrame(index=target_index, columns=self.columns).fillna(0)

        container.loc[df.index, :] = df
        self._frame = container
        return self

    def to_dataframe(self, dtype=float) -> pd.DataFrame:

        frame = self.frame.copy(deep=True)

        if dtype in (int, float):
            frame["AVG_BEAM_HGT"] = frame["AVG_BEAM_HGT"].str.replace(r"[A-Za-z]", "", regex=True).apply(pd.eval)

            frame[["MAXRC_EMISS", "MAXRC_ICECF"]] = (
                frame[["MAXRC_EMISS", "MAXRC_ICECF"]]
                .stack()
                .str.extract(r"(?:\()([a-z]*)(?:\))")
                .replace({"weak": 1, "moderate": 2, "strong": 3})
                .fillna(0)
                .unstack(-1)
                .droplevel(0, axis=1)
            )

        return frame.astype(dtype=dtype)
