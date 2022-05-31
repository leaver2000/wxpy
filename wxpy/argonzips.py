import zipfile
import requests

# import shutil
# import os
import pandas as pd


def generate_urls(
    start="2022-03-01T00:00",
    end="2022-05-01T00:00",
):
    ARGON_TEMPLATE = (
        "https://mrms.agron.iastate.edu/{year}/{month:02d}/{day:02d}/{year}{month:02d}{day:02d}{hour:02d}.zip"
    )
    for x in pd.date_range(start=start, end=end, freq="h"):
        yield ARGON_TEMPLATE.format(**{attr: getattr(x, attr) for attr in ("year", "month", "day", "hour")})


def unzip_probsevere(filename):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        s = pd.Series(zip_ref.namelist())
        for x in s[s.str.contains("PROBSEVERE")]:
            zip_ref.extract(x, f"data/test/{x}")


def get_data():
    for url in generate_urls():
        filename = "data/zips/" + url.split("/")[-1]
        print("getting ", url)
        req = requests.get(url)
        with open(filename, "wb") as output_file:
            output_file.write(req.content)
        print("SAVED")


if __name__ == "__main__":
    get_data()
