import os
import logging
import pandas as pd

from datetime import timedelta

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath

# "/atlasarchive/atlas/03_animal_trials/210824_animal_trial_02/"

# INPUT_DIR = "/data/input"
INPUT_DIR = "/media/narvis/Elements/03_animal_trials/"
EXPORT_DIR = "/data/2901_atlas_export_rgbd/"


if __name__ == "__main__":
    logging.basicConfig(filename="annotation_export.log",
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("Color Frame Export")
    logging.info(os.listdir(INPUT_DIR))

    set_log_level("info")

    annotations = pd.read_csv("../annotations_processed.csv", parse_dates=["Start", "End"],
                              usecols=["Trial", "filekey", "Phase", "Start", "End"],
                              dtype={"Trial": str, "filekey": str, "Phase": str, "Start": str,
                                     "End": str},
                              infer_datetime_format=True)

    for i, trial in annotations.iterrows():
        recording_dir = trial["Trial"]
        if recording_dir not in os.listdir(INPUT_DIR):
            logging.info("Trial not found: " + recording_dir)
            continue

        phase = trial["Phase"]
        trial_path = os.path.join(INPUT_DIR, recording_dir, "recordings")
        export_path = os.path.join(EXPORT_DIR, recording_dir, trial["Phase"])
        recording_name = "recordings_" + recording_dir[4:6] + recording_dir[2:4] + "_recording_" + str(trial["filekey"])
        dir_path = os.path.join(trial_path, recording_name)
        if "blooper" in trial["Phase"]:
            continue
        if os.path.exists(export_path):
            logging.warning(f"Skipping recording {export_path}, it exists .. continuing")
            continue
        if not os.path.exists(dir_path):
            logging.warning(f"Skipping recording {dir_path}, dir path not found.")
            continue

        start = pd.Timestamp(trial["Start"])
        end = pd.Timestamp(trial["End"])
        phase = trial["Phase"]

        if (end - start) > timedelta(minutes=10):
            # use only the first 10 minutes right now, don't export forever
            end = start + timedelta(minutes=10)

        export_config = ExportConfig()
        export_config.export_color = True
        export_config.export_rgbd = True
        export_config.timesync = True
        # only export 1FPS
        export_config.skip_frames = 5
        export_config.start_ts = timedelta(microseconds=start.value // 1000)
        export_config.end_ts = timedelta(microseconds=end.value // 1000)
        logging.info("Exporting color images for: ")
        logging.info(trial)

        timesync = TimesynchronizerPCPD(export_config)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        os.makedirs(export_path, exist_ok=True)
        logging.info(f"Exporting color images for {export_path}")
        timesync.initialize_feeds(files, MkvPath(export_path))

        timesync.run()

        del timesync

