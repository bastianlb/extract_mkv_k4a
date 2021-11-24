import os

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


# "/atlasarchive/atlas/03_animal_trials/210824_animal_trial_02/"

INPUT_DIR = "/data/input"

if __name__ == "__main__":
    print("Color Video export")
    print(os.listdir(INPUT_DIR))
    export_config = ExportConfig()
    export_config.export_color_video = True
    export_config.timesync = True

    set_log_level("info")

    for base_dir in os.listdir(INPUT_DIR):
        if "recording_04" in base_dir:
            continue
        print("Exporting color video for: ", base_dir)

        timesync = TimesynchronizerPCPD(export_config)
        dir_path = os.path.join(INPUT_DIR, base_dir)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        timesync.initialize_feeds(files, MkvPath("/data/export/"))

        timesync.run()

        del timesync
