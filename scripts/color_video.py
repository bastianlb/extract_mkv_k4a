import os

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


# "/atlasarchive/atlas/03_animal_trials/210824_animal_trial_02/"

# INPUT_DIR = "/data/input"
# INPUT_DIR = "/media/lennart/Elements/211207_animal_trial_14/recordings/"
# INPUT_DIR = "/media/lennart/Elements/211207_animal_trial_14/recordings/"
INPUT_DIR = "/media/lennart/d25b2352-7a02-4b0d-ac1e-5fc60c083669/archive_atlas_or01/03_animal_trials/210914_animal_trial_03/recordings/"  # noqa
EXPORT_DIR = "/data/export/"

if __name__ == "__main__":
    print("Color Video export")
    print(os.listdir(INPUT_DIR))
    export_config = ExportConfig()
    export_config.export_color_video = True
    export_config.timesync = True

    set_log_level("info")

    for base_dir in os.listdir(INPUT_DIR):
        print("Exporting color video for: ", base_dir)
        if os.path.exists(os.path.join(EXPORT_DIR, base_dir) + ".avi"):
            continue

        timesync = TimesynchronizerPCPD(export_config)
        dir_path = os.path.join(INPUT_DIR, base_dir)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        timesync.initialize_feeds(files, MkvPath(EXPORT_DIR))

        timesync.run()

        del timesync
