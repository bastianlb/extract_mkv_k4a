import pandas as pd
import piso
import numpy as np
from dateutil.parser import isoparse
import logging


def custom_date_parser(date):
    # we used custom format.. month-day hrs:mins:sec,ms
    if not isinstance(date, str) or date == "nan":
        return np.nan
    try:
        d, ms = date.split(",")
    except ValueError as e:
        logging.info("Invalid date format: ", date)
        raise e
    assert len(ms) == 3, date
    # put into ISO format
    date = date.replace(" ", "T")
    date = date.replace(",", ".")
    date = date + "+00:00"
    try:
        return isoparse("2021-" + date)
    except ValueError as e:
        logging.info(f"Invalid date format: {date}")
        raise e
    return date


def read_data(fname):
    return pd.read_csv(
        fname,
        usecols=["Trial", "filekey", "Phase", "Start", "End", "notes"],
        dtype={"Trial": str, "filekey": str, "Phase": str, "Start": str, "End": str, "notes": str},
        infer_datetime_format=False, date_parser=custom_date_parser,
        parse_dates=["Start", "End"])


def _in(key, df):
    return not np.all(df["Phase"] == key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fname = "../atlas_workflow_annotations_1401_11:48.csv"
    annotations = read_data(fname)

    for i, trial in enumerate(annotations["Trial"]):
        trial = trial.split("_")
        trial[0] = "21" + trial[0][2:] + trial[0][:2]
        trial.insert(1, "animal")
        annotations.loc[i, ("Trial")] = "_".join(trial)

    for trial in annotations["Trial"].unique():
        filt = annotations[annotations["Trial"] == trial]
        if not (_in("PAO", filt) and _in("TIP", filt)):
            logging.info("Trial incomplete!")
            continue

    # find region of overlap
    for i, trial in annotations.iterrows():
        if trial["Start"] >= trial["End"]:
            print("Trial bad formatting", trial)

    def format(x):
        if len(x) == 1:
            return "0" + x
        return x

    annotations["filekey"] = annotations["filekey"].apply(format)

    # annotations.index = pd.IntervalIndex.from_arrays(annotations["Start"], annotations["End"])
    # annotations = annotations.drop(columns=["Start", "End"])
    to_append = []

    for trial in annotations["Trial"].unique():
        pao = annotations[(annotations["Trial"] == trial) & (annotations["Phase"] == "PAO")]
        tip = annotations[(annotations["Trial"] == trial) & (annotations["Phase"] == "TIP")]
        if (len(pao) == 0 or len(tip) == 0):  # noqa
            logging.info(f"Trial missing PREP annotations: {trial}")
            continue
        assert(pao["filekey"].values[0] == tip["filekey"].values[0])
        pao_i = pd.arrays.IntervalArray([pd.Interval(pd.Timestamp(pao["Start"].values[0]), pd.Timestamp(pao["End"].values[0]))])  # noqa
        tip_i = pd.arrays.IntervalArray([pd.Interval(pd.Timestamp(tip["Start"].values[0]), pd.Timestamp(tip["End"].values[0]))])  # noqa
        inter = piso.intersection(pao_i, tip_i)
        if len(inter) == 0:
            # no overlap between segments
            continue
        to_append.append([trial, pao["filekey"].values[0], "PREP", inter[0].left, inter[0].right, ""])
        diff_pao = piso.difference(pao_i, tip_i)
        diff_tip = piso.difference(tip_i, pao_i)
        # modify phases exclusively
        # process pao
        if len(diff_pao) == 1:
            annotations.loc[pao.index[0], ["Start"]] = diff_pao[0].left
            annotations.loc[pao.index[0], ["End"]] = diff_pao[0].right
        elif len(diff_pao) == 0:
            # phase is contained entirely in overlap
            annotations.drop(pao.index[0], inplace=True)
        elif len(diff_pao) == 2:
            annotations.loc[pao.index[0], ["Start"]] = diff_pao[0].left
            annotations.loc[pao.index[0], ["End"]] = diff_pao[0].right
            to_append.append([trial, pao["filekey"].values[0], "PAO2", diff_pao[0].left, diff_pao[0].right, ""])
        else:
            print(f"Invalid set difference {diff_pao}")
            exit(1)
        # process tip
        if len(diff_tip) == 1:
            annotations.loc[tip.index[0], ["Start"]] = diff_tip[0].left
            annotations.loc[tip.index[0], ["End"]] = diff_tip[0].right
        elif len(diff_tip) == 0:
            # phase is contained entirely in overlap
            annotations.drop(tip.index[0], inplace=True)
        elif len(diff_tip) == 2:
            annotations.loc[tip.index[0], ["Start"]] = diff_tip[0].left
            annotations.loc[tip.index[0], ["End"]] = diff_tip[0].right
            to_append.append([trial, tip["filekey"].values[0], "TIP2", diff_tip[0].left, diff_tip[0].right, ""])
        else:
            print(f"Invalid set difference {diff_tip}")
            exit(1)

    out = pd.DataFrame(to_append, columns=annotations.columns)
    annotations = annotations.append(out)
    annotations.to_csv("../annotations_processed.csv")
