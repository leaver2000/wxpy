from glob import glob
import pandas as pd
import numpy as np
import xarray as xr
import requests
import re

var_template = "{0}_P0_L100_GLL0"
coordinates = ["lv_ISBL0", "lat_0", "lon_0"]
variables = {var_template.format(x): x for x in ("TMP", "RH", "UGRD", "VGRD", "HGT")}
URL_TEMPLATE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/557ww/prod/557ww.{year:04}{month:02d}{day:02d}/GLOBAL.grib2.{year:04}{month:02d}{day:02d}{hour:02d}.{forecast_hour:04d}"


def build_url(model_run: str, forecast_hour: int = 0):
    run = pd.to_datetime(model_run)
    return URL_TEMPLATE.format(
        **{x: getattr(run, x) for x in ("year", "month", "day", "hour")}, forecast_hour=forecast_hour
    )


def dataset_from_url(url: str) -> xr.Dataset:
    file_path = "tmp/" + re.search(r"\d*\.\d{4}$", url).group()
    r = requests.get(url)
    ds = None
    if r.status_code == 200:
        print("writing file")
        with open(file_path, "wb") as f:
            f.write(r.content)
        ds: xr.Dataset = xr.load_dataset(file_path, engine="pynio")
    else:
        print("bad status code")

    return ds


def run():
    url = build_url("2022-05-30T00:00", forecast_hour=0)
    print(url)
    ds = dataset_from_url(url)
    print(ds)


# https://nomads.ncep.noaa.gov/pub/data/nccf/com/557ww/prod/557ww.20220530/GLOBAL.grib2.2022053000.0000

if __name__ == "__main__":
    run()
