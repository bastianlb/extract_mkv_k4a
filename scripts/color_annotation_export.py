import os
import logging
import pandas as pd
import glob

from datetime import timedelta

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath

# "/atlasarchive/atlas/03_animal_trials/210824_animal_trial_02/"

# INPUT_DIR = "/data/input"
# INPUT_DIR = "/media/narvis/Elements/03_animal_trials/"
# INPUT_DIR = "/media/narvis/Elements/03_animal_trials/"

INPUT_DIRS = [
    "/mnt/atlas_1/03_animal_trials/", "/mnt/atlas_2/03_animal_trials/",
    "/mnt/recordings/03_animal_trials/"
]
EXPORT_DIR = '/data/tmp'
# EXPORT_DIR = "/archive/2605_di_export/"
# EXPORT_DIR = '/home/tonyw/extract_mkv_k4a/tmp_output'

if __name__ == "__main__":
    logging.basicConfig(filename="annotation_export.log", level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-6s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("Color Frame Export")
    for inp_dir in INPUT_DIRS:
        logging.info(os.listdir(inp_dir))

    set_log_level("info")

    annotations = pd.read_csv("../annotations_processed.csv",
                              parse_dates=["Start", "End"],
                              usecols=["Trial", "filekey", "Phase", "Start", "End"],
                              dtype={
                                  "Trial": str,
                                  "filekey": str,
                                  "Phase": str,
                                  "Start": str,
                                  "End": str
                              },
                              infer_datetime_format=True)

    # [14,24,36,46,53,62,72,83,93,115,128,138,148,162,172, 191]
    # 146 and beyond is calibration 22...
    for i, trial in annotations[0:1].iterrows():
        recording_dir = trial["Trial"]

        INPUT_DIR = next((inp_dir for inp_dir in INPUT_DIRS
                          if os.path.exists(os.path.join(inp_dir, recording_dir))), None)

        if INPUT_DIR is None:
            logging.info("Trial not found: " + recording_dir)
            continue

        phase = trial["Phase"]
        trial_path = os.path.join(INPUT_DIR, recording_dir, "recordings")
        export_path = os.path.join(EXPORT_DIR, recording_dir, trial["Phase"])
        recording_name = "recordings_" + recording_dir[4:6] + recording_dir[
            2:4] + "_recording_" + str(trial["filekey"])
        dir_path = os.path.join(trial_path, recording_name)

        export_config = ExportConfig()
        export_config.export_color = True
        export_config.export_pointcloud = True
        export_config.align_clouds = True
        export_config.timesync = True

        if "blooper" in trial["Phase"]:
            continue

        if os.path.exists(export_path):
            # check for specific files instead of whether the directory exists
            if export_config.export_color and len(
                    glob.glob(export_path + '/**/*.jpg', recursive=True)) > 0:
                logging.warning(
                    f"Skipping recording {export_path}, color images already exist .. continuing")
                continue
            if export_config.export_pointcloud and len(
                    glob.glob(export_path + '/**/*.ply', recursive=True)) > 0:
                logging.warning(
                    f"Skipping recording {export_path}, point clouds already exist .. continuing")
                continue

        if not os.path.exists(dir_path):
            logging.warning(f"Skipping recording {dir_path}, dir path not found.")
            continue

        start = pd.Timestamp(trial["Start"])
        end = pd.Timestamp(trial["End"])
        mid = start + (end - start) / 2
        phase = trial["Phase"]
        new_start = start
        new_end = end
        #new_start = mid - timedelta(seconds=50)

        #if new_start < start:
        #    new_start = start

        #new_end = mid + timedelta(seconds=50)
        #if new_end > end:
        #    new_end = end

        if (end - start) > timedelta(seconds=30):
            # use only the first 15 minutes right now, don't export forever
            end = start + timedelta(seconds=30)

        # only export 0.5FPS
        export_config.skip_frames = 5
        export_config.start_ts = timedelta(microseconds=new_start.value // 1000)
        export_config.end_ts = timedelta(microseconds=new_end.value // 1000)
        logging.info("Exporting data for: ")
        logging.info(trial)

        timesync = TimesynchronizerPCPD(export_config)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        os.makedirs(export_path, exist_ok=True)
        logging.info(f"Exporting data to {export_path}")
        timesync.initialize_feeds(files, MkvPath(export_path))

        timesync.run()

        del timesync
