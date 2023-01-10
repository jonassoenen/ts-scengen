"""
    Util class to read rlp and slp profiles from the 'rlp_slp_data' folder
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "rlp_slp_data"
RLP_YEARS = list(range(2014, 2018))
SLP_YEARS = list(range(2010, 2014))


def read_slps():
    return {year: read_slp(year) for year in SLP_YEARS}


def read_rlps():
    return {year: read_rlp(year) for year in RLP_YEARS}


def read_rlp(year):
    data_df = pd.read_excel(
        DATA_PATH / f"RLP0N{year}.xls", sheet_name="Sheet1", nrows=2
    )
    assert data_df.columns[0] == "IVEKA", f"RLP0N{year} is not for IVEKA!"
    return pd.read_excel(
        DATA_PATH / f"RLP0N{year}.xls",
        sheet_name="Sheet1",
        skiprows=4,
        usecols=["RLPestU"],
    ).set_index(
        pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:45:00", freq="15min")
    )


def read_slp(year):
    non_residential_data_df = pd.read_excel(
        DATA_PATH / f"SLPU_{year}.xls",
        sheet_name="Electricity Non Residential",
        usecols=["S11 <56KVA", "S12 56-100KVA"],
    )
    residential_data_df = pd.read_excel(
        DATA_PATH / f"SLPU_{year}.xls",
        sheet_name="Electricity Residential",
        usecols=["S21 Res<1.3", "S22 Res>=1.3"],
    )
    slp = pd.concat([residential_data_df, non_residential_data_df], axis=1).set_index(
        pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:45:00", freq="15min")
    )
    return slp
