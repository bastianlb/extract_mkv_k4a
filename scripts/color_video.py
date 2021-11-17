import os

# from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


INPUT_DIR = "/atlasarchive/atlas/03_animal_trials/210810_animal_trial_01/recordings/"

if __name__ == "__main__":
    export_config = ExportConfig()
    export_config.export_color_video = True

    # set_log_level("debug")

    for base_dir in os.listdir(INPUT_DIR):
        timesync = TimesynchronizerPCPD(export_config)
        dir_path = os.path.join(INPUT_DIR, base_dir)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        timesync.initialize_feeds(files, MkvPath(os.path.join("../atlas_videos", base_dir)))

        timesync.run()

        del timesync
