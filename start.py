import os
from glob import glob
import pandas as pd
import numpy as np
from wxpy.core import reshape, grib2dataset

ALL_GALWEM_FILES = sorted(glob(os.path.join("data", "galwem", "*.GR2")))
ALL_PROBSEVERE_FILES = sorted(glob(os.path.join("data", "probsevere", "*.json")))


def unpack_files(time_buffer: int = 90) -> tuple[pd.Series, pd.Series]:
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


import json


from geopandas import GeoDataFrame, GeoSeries
from typing import Iterable, NamedTuple

import xarray as xr


def iterfiles(filepaths: "pd.Series[str]") -> Iterable[xr.Dataset | dict[str, dict[str, str | float | int]]]:

    if filepaths.str.endswith(".json").all():
        for filepath in filepaths:
            with open(filepath, mode="r", encoding="utf-8") as fc:
                feat = json.load(fc)
                for f in feat["features"]:
                    f["properties"]["validTime"] = feat["validTime"]
                    yield f
    if filepaths.str.endswith(".GR2").all():
        for timestamp, file in filepaths.items():
            yield xr.open_dataset(file, engine="pynio").expand_dims({"validTime": [timestamp.value]})


class BBox(NamedTuple):
    minx: float = -130.0 % 360
    maxx: float = -60.0 % 360
    miny: float = 20.0
    maxy: float = 55.0


class ProbSevere:
    def __init__(self, files: pd.Series) -> None:
        df = GeoDataFrame.from_features(iterfiles(files))
        df["validTime"] = pd.to_datetime(df["validTime"], format="%Y%m%d_%H%M%S %Z", utc=True)
        self._frame = df.set_index(["validTime", "ID"])

    def __repr__(self) -> str:
        return self._frame.__repr__()

    @property
    def frame(self):
        return self._frame

    def fit(self, into, bbox: BBox = BBox()):
        return self._frame


class Galwem:
    def __init__(self, files: pd.Series, bbox: BBox = BBox()) -> None:
        # def generate():

        #     for timestamp, filepath in files.items():
        #         ds: xr.Dataset = xr.open_dataset(filepath, engine="pynio")
        #         condition: xr.DataArray = (
        #             (ds.lon_0 >= bbox.minx)
        #             & (ds.lon_0 <= bbox.maxx)
        #             & (ds.lat_0 >= bbox.miny)
        #             & (ds.lat_0 <= bbox.maxy)
        #         )
        #         ds: xr.Dataset = ds.where(condition, drop=True)
        #         yield ds.expand_dims({"validTime": [timestamp.value]}).rename(
        #             {
        #                 "lv_ISBL0": "hPa",
        #                 "lat_0": "lat",
        #                 "lon_0": "lon",
        #                 "TMP_P0_L100_GLL0": "temp",
        #                 "UGRD_P0_L100_GLL0": "u_wind",
        #                 "VGRD_P0_L100_GLL0": "v_wind",
        #             }
        #         )
        # ds =
        ds: xr.Dataset = xr.concat(iterfiles(files), dim="validTime")
        ds: xr.Dataset = ds.where(
            ((ds.lon_0 >= bbox.minx) & (ds.lon_0 <= bbox.maxx) & (ds.lat_0 >= bbox.miny) & (ds.lat_0 <= bbox.maxy)),
            drop=True,
        ).rename(
            {
                "lv_ISBL0": "hPa",
                "lat_0": "lat",
                "lon_0": "lon",
                "TMP_P0_L100_GLL0": "temp",
                "UGRD_P0_L100_GLL0": "u_wind",
                "VGRD_P0_L100_GLL0": "v_wind",
            }
        )
        print(ds)


if __name__ == "__main__":
    galwem, probsevere = unpack_files(time_buffer=90)
    Galwem(galwem.iloc[0:2])
    # ps = ProbSevere(probsevere)
    # print(ps)
    # print(BBox())


if __name__ == "__main__2":
    galwem, probsevere = unpack_files(time_buffer=90)
    galwem_ds = grib2dataset(galwem)
    print(galwem_ds)
    ps = reshape(probsevere).in_to(galwem_ds)

    print(ps)

    print("DONE!!!")
