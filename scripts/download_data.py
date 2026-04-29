#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import urllib.request


URL = "https://github.com/drwiiche/electricity-consumption/raw/refs/heads/master/AEP_hourly.csv"


def main() -> None:
    out = pathlib.Path("data/AEP_hourly.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Data already exists: {out}")
        return
    print(f"Downloading data from {URL}")
    urllib.request.urlretrieve(URL, out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
