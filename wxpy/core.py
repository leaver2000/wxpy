import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from typing import Iterable
from geojson import FeatureCollection
from numpy.typing import NDArray
import pandas as pd
import xarray as xr
from geopandas import GeoDataFrame
from shapely.geometry import box
import json


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


def dataset_generator(times_files: dict[str, str]) -> Iterable[xr.Dataset]:

    for timestamp, filepath in times_files.items():
        ds: xr.Dataset = xr.load_dataset(filepath, engine="pynio")

        yield ds.expand_dims({"validTime": [timestamp.value]}).rename(
            {
                "lv_ISBL0": "hPa",
                "lat_0": "lat",
                "lon_0": "lon",
                "TMP_P0_L100_GLL0": "temp",
                "UGRD_P0_L100_GLL0": "u_wind",
                "VGRD_P0_L100_GLL0": "v_wind",
            }
        )


def grib2dataset(times_files: dict[str, str]):
    return xr.concat(dataset_generator(times_files), dim="validTime")


def make_bbox():
    bounds: pd.Series = (
        GeoDataFrame(geometry=[box(minx=-130.0, maxx=-60.0, miny=20.0, maxy=55.0)])
        .set_crs(epsg=4326)
        .bounds.rename({0: "bbox"})
        .squeeze()
    )
    # Longitude conversion 0~360 to -180~180
    # https://confluence.ecmwf.int/pages/viewpage.action?pageId=149337515
    bounds[["minx", "maxx"]] = bounds[["minx", "maxx"]] % 360
    return bounds


class DataFrame(GeoDataFrame):
    def in_to(self, target: list[str] | xr.Dataset, bbox=make_bbox()) -> pd.DataFrame:
        if isinstance(target, xr.Dataset):
            ds = target
        elif isinstance(target, pd.DataFrame):
            ...
        elif isinstance(target, list):
            ds: xr.Dataset = xr.concat(dataset_generator(target), dim="validTime")
        target_validtime: pd.DatetimeIndex = pd.to_datetime(ds["validTime"])
        # bool array
        condition: xr.DataArray = (
            (ds.lon >= bbox.minx) & (ds.lon <= bbox.maxx) & (ds.lat >= bbox.miny) & (ds.lat <= bbox.maxy)
        )
        # drop values outside of bounding box
        galwem_conus_ds: xr.Dataset = ds.where(condition, drop=True)

        df: pd.DataFrame = galwem_conus_ds.to_dataframe()

        df.columns.set_names("elements", inplace=True)

        GALWEM = (
            df.unstack("hPa").reorder_levels(["validTime", "lat", "lon"]).reorder_levels(["hPa", "elements"], axis=1)
        )

        ps_geometry: GeoSeries = self["geometry"]
        # set grid coordinates W,E
        self[["WEST", "EAST"]] = index_abs_argmin(ps_geometry.bounds[["minx", "maxx"]] % 360, GALWEM.geo.lon)
        # and S,N
        self[["SOUTH", "NORTH"]] = index_abs_argmin(ps_geometry.bounds[["miny", "maxy"]], GALWEM.geo.lat)
        ps: pd.DataFrame = self.drop(["geometry"], axis=1).set_index(["WEST", "EAST", "NORTH", "SOUTH"], append=True)

        ps_validtimes = self.index.unique("validTime")

        mesh = synctime_meshgrid(ps, target_validtime, ps_validtimes)
        PROBSEVERE = pd.DataFrame(index=GALWEM.index, columns=mesh.columns).fillna(0)
        PROBSEVERE.loc[mesh.index, :] = mesh
        return PROBSEVERE


def index_abs_argmin(
    ps_bounds: pd.DataFrame,  # (27070, 2)
    galwem_grid: NDArray[np.float32],  # (281,)
) -> NDArray[np.float32]:  # (27070, 2)
    """
    >>> ps_bounds:GeoDataFrames.geometry.bounds[["minx", "maxx","miny", "maxy"]]
    >>> galwem_grid: NDArray[np.float32]
    """

    # first shaped the probsevere and galwm so that have a common axis
    ps_shaped = ps_bounds.to_numpy()[:, np.newaxis]  # (27070, 1, 2)
    galwem_shaped = galwem_grid[:, np.newaxis]  # (141, 1)
    delta = abs(galwem_shaped - ps_shaped)  # (27070, 281, 2)
    # in the delta find the smallest diffrence here -- ^
    index_nearest = np.argmin(delta, axis=1)  # (27070, 2)
    # use the index position of the smallest diff to a grid point
    # to index the grid for a max and min
    return galwem_grid[index_nearest]  # (27070, 2)


class ProbSevere(GeoDataFrame):
    def __init__(self, data=None, *args, geometry=None, crs=None, **kwargs):
        super().__init__(data, *args, geometry=geometry, crs=crs, **kwargs)


def reshape(filepaths: list[str]) -> DataFrame:
    def generate() -> Iterable[GeoDataFrame]:
        for fc in _iterpaths(filepaths):
            df = GeoDataFrame.from_features(fc["features"])
            df["validTime"] = fc["validTime"]
            yield df

    df: pd.DataFrame = DataFrame(pd.concat(generate(), ignore_index=True))
    df["validTime"] = pd.to_datetime(df["validTime"], format="%Y%m%d_%H%M%S %Z", utc=True)
    # set a validTime storm_id multiindex
    df.set_index(["validTime", "ID"], inplace=True)

    df.columns.set_names("elements", inplace=True)

    # here just updating sting values with numeric ones
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
    return df  # DataFrame(pd.concat(generate(), ignore_index=True))


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


def time_iterator(g_time: np.ndarray, target_times: pd.Index) -> Iterable[tuple[pd.Timestamp, pd.Index]]:
    time_interval = len(g_time)
    start_time = np.argmin(
        abs(np.array(g_time)[:, np.newaxis] - target_times.values) > pd.to_timedelta(time_interval, unit="h"),
        axis=1,
    )

    end_time = np.roll(start_time, -1)
    end_time[-1] = -1

    for timestamp, tuple_slice in zip(g_time, zip(start_time, end_time)):
        yield timestamp, target_times[slice(*tuple_slice)]


def synctime_meshgrid(probsevere: pd.DataFrame, galwem_validtimes, ps_validtimes) -> pd.DataFrame:
    probsevere = probsevere.droplevel("ID").astype(np.float32)

    def generate():
        for mark_times, target_times in time_iterator(galwem_validtimes, ps_validtimes):
            frame = probsevere.loc[target_times, :].groupby(["WEST", "EAST", "NORTH", "SOUTH"]).mean()

            index_names = frame.index.names

            west, east, north, south = (frame.index.get_level_values(name) for name in index_names)

            mesh_grid_condition = (north == south) & (west == east)
            # NOTE NEEDS VALIDATION
            frame.loc[mesh_grid_condition] = frame.loc[mesh_grid_condition].groupby(index_names).mean()

            frame = frame.droplevel(["WEST", "SOUTH"])

            frame.index = frame.index.set_names(["lon", "lat"])

            frame["validTime"] = mark_times.value  # pd.to_datetime([galwem_valid_hour]).astype(int)[0]

            yield frame.reset_index().set_index(["validTime", "lat", "lon"], append=True)

    return pd.concat(generate(), axis=0).droplevel(0)


def _iterpaths(filepaths: list[str]) -> Iterable[FeatureCollection]:
    """load function for probsevere dataset"""
    for filepath in filepaths:
        with open(filepath, mode="r", encoding="utf8") as fc:
            yield json.load(fc)
