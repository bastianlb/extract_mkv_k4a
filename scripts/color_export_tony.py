import os
from datetime import timedelta, datetime
import re

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath

DAY = "23"
MONTH = "11"
YEAR = "21"
HOUR = "7"
MINUTE = "26"
SECONDS = "30"
RECORDING = "2"

if __name__ == "__main__":
    print("Color Video export")
    export_config = ExportConfig()
    export_config.export_color = True
    export_config.max_frames_exported = 200
    export_config.skip_frames = 5
    export_config.export_distorted = False
    export_config.export_pointcloud = False
    export_config.timesync = True
    export_config.align_clouds = False
    export_config.export_rgbd = False

    unix_time = datetime(year=int(f'20{YEAR}'),
                         month=int(MONTH),
                         day=int(DAY),
                         hour=int(HOUR) + 1,
                         minute=int(MINUTE),
                         second=int(SECONDS)).timestamp()
    export_config.start_ts = timedelta(
        seconds=unix_time)  # 2nd Nov. 2021, 7.40.33 AM
    export_config.end_ts = export_config.start_ts + timedelta(minutes=2)

    set_log_level("info")

    timesync = TimesynchronizerPCPD(export_config)

    trial_name_regex = f"{YEAR.zfill(2)}{MONTH.zfill(2)}{DAY.zfill(2)}_animal_trial_*"

    directory_atlas_1 = "/mnt/atlas_1/03_animal_trials/"
    directory_atlas_2 = "/mnt/atlas_2/03_animal_trials/"
    trials_1 = os.listdir(directory_atlas_1)
    trials_2 = os.listdir(directory_atlas_2)

    r = re.compile(trial_name_regex)
    trials_1 = list(filter(r.match, trials_1))
    trials_2 = list(filter(r.match, trials_2))
    assert (len(trials_1) == 1 or len(trials_2) == 1)
    if len(trials_1) == 1:
        trial_dir = directory_atlas_1 + trials_1[0]
    else:
        trial_dir = directory_atlas_2 + trials_2[0]

    dir_path = trial_dir + f"/recordings/recordings_{DAY.zfill(2)}{MONTH.zfill(2)}_recording_{RECORDING.zfill(2)}/"
    files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]
    print("Files:", files)

    timesync.initialize_feeds(
        files, MkvPath("/home/tonyw/extract_mkv_k4a/output/test/"))

    timesync.run()

    del timesync