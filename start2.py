"""_summary_

Returns:
    _type_: _description_

Yields:
    _type_: _description_
"""
from typing import Mapping
from datetime import datetime
import requests
import psutil
import time
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
import pandas as pd

scheduler = BlockingScheduler()
template = "data/{0}.parquet"
MRMS_ARCHIVE = "https://mrms.agron.iastate.edu/{year}/{month}/{day}/"


def name_to_datetime(names: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(names.str.replace("_", "T").str.extract(r"(\d*T\d*).json")[0], tz="utc").rename(
        "validTime"
    )


def read_mrms(*args: str) -> pd.DataFrame:
    url = "/".join(["https://mrms.ncep.noaa.gov/data", *args]) + "/?C=M;O=D"
    return pd.read_html(url)[0].dropna()


def read_probsevere() -> pd.DataFrame:
    df = read_mrms("ProbSevere", "PROBSEVERE")
    df.index = name_to_datetime(df.Name)
    return ("https://mrms.ncep.noaa.gov/data/ProbSevere/PROBSEVERE/" + df["Name"]).rename("url")


def to_dataframe(vt_url: Mapping[pd.Timestamp, str]):
    def generate():
        for i, (vt, url) in enumerate(vt_url.items()):
            print(vt.strftime("%Y-%m-%dT%H:%M:%SZ"))
            print(psutil.virtual_memory().percent)
            print(f"count ={i}\n")
            try:
                for feat in requests.get(url).json()["features"]:
                    props = feat["properties"]
                    props["validTime"] = vt
                    props["geometry"] = feat["geometry"]
                    yield props
            except:
                ...

    print("begining file collection, collecting file for....")

    ps = pd.DataFrame(tuple(generate())).set_index(["validTime", "ID"])
    print("all files collected")

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

    ps.loc[:, ps.columns != "geometry"] = ps.loc[:, ps.columns != "geometry"].astype(np.float32)

    print("dataframe processed")
    return ps


@scheduler.scheduled_job(IntervalTrigger(days=1, start_date=datetime.strptime("2022-05-30", "%Y-%m-%d")))
def on_newday():
    print("ON NEW DAY EVENT TRIGGERED")

    available_data = read_probsevere()

    mapping = available_data[available_data.index.day != datetime.utcnow().day]
    print(f"there are {len(mapping)} files queued for download")

    file_name = datetime.utcnow().strftime("%Y-%m-%d")
    to_dataframe(mapping[::-1]).to_parquet(template.format(file_name))
    print(f"file saved as {template.format(file_name)}")


if __name__ == "__main__":
    on_newday()
    scheduler.start()
