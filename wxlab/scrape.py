from typing import Callable, Iterable
import time
import re

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup





class TSragr:
    def __init__(self, base_url: str = None) -> None:

        self._baseurl = base_url

        r = requests.get(base_url)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, "lxml").find_all("a")

        if soup[0].text == "Parent Directory":
            soup = soup[1:]

        self._soup = pd.Series([x.text for x in soup])

    def __repr__(self) -> str:
        return f"{self.url}\n"+ self._soup.__repr__()

    def __getitem__(self, args) -> "TSragr":
        self._soup = self._soup[args]
        return self

    @property
    def url(self):
        url = self._baseurl
        if not url.endswith("/"):
            url = url+ "/"
        return url

    def navto(self, *args: str) -> "TSragr":
        return TSragr(self.url + "/".join(args))

    def navup(self) -> "TSragr":
        return TSragr(re.match(r"^(.*[\/])", self.url).group())

    def inav(self, index: int):
        return TSragr(self.url + self._soup[index])

    def download(self, save_to="./", wait: float = 10):

        soup = self._soup.copy()
        soup.index = self.url + self._soup

        for url, filename in soup.items():
            print("DOWNLAODING FILE")
            local_filename = save_to + filename
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print("FILE SAVED")
            time.sleep(60 * wait)

class Urls:
    def __init__(self, urls: Iterable[str]):
        self._urls = urls  # np.array(urls)

    def __repr__(self):
        return np.array(self._urls).__repr__()

    def download(self, save_to="/media/external/data/", _slice: slice = slice(0, 2), wait: int = 10) -> None:
        for url in self._urls[_slice]:
            local_filename = save_to + url.split("/")[-1]
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            time.sleep(60 * wait)

    @property
    def values(self):
        return self._urls


def read_urls(func: Callable[[], tuple[str, str | None]]):
    url, path_opt = func()

    def inner():
        # read the base url
        r = requests.get(url)
        r.raise_for_status()
        # read the html content of th
        soup = BeautifulSoup(r.content, "lxml")
        latests_run = url + soup.find_all("a")[-1].text

        r = requests.get(latests_run)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, "lxml")

        all_gribs: list[str] = [latests_run + a.text for a in soup.find_all("a")[1:]]

        if path_opt is not None:
            r = requests.get([g for g in all_gribs if g.endswith(path_opt)][0])
            r.raise_for_status()
            alpha = BeautifulSoup(r.content, "lxml").find_all("a")
            all_gribs = [latests_run + path_opt + a.text for a in alpha if a.text.endswith(".grib2")]

        return Urls(all_gribs)

    return inner


@read_urls
def galwem() -> Urls:
    return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/557ww/prod/", None


@read_urls
def hrrr(loc="conus") -> Urls:
    """loc = conus or alaska"""
    return "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod/", loc + "/"
