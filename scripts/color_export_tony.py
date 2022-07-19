import os
from datetime import timedelta

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


INPUT_DIR = "/mnt/atlas_2/"

if __name__ == "__main__":
    print("Color Video export")
    print(os.listdir(INPUT_DIR))
    export_config = ExportConfig()
    export_config.export_color = True
    export_config.max_frames_exported = 100000
    export_config.skip_frames = 5
    export_config.export_distorted = False
    export_config.export_pointcloud = True
    export_config.timesync = True
    export_config.align_clouds = True
    export_config.export_rgbd = True

    # TODO: start end must be recording dependent...
    #                        1628650257725706448
    #export_config.start_ts = timedelta(microseconds=1628651055148343000 // 1000)
    #export_config.end_ts =   timedelta(microseconds=1628653016064450000 // 1000)
    #                        1628658895596854977

    export_config.start_ts = timedelta(seconds=1639478820) # Tuesday, December 14, 2021 10:47:00 AM GMT
    export_config.end_ts = timedelta(seconds=1639479180) # Tuesday, December 14, 2021 10:53:00 AM
    #export_config.start_ts = timedelta(seconds=1635838893)

    set_log_level("info")

    # for base_dir in ["03_animal_trials/211102_animal_trial_08"]:  #, "recordings_2809_recording_5"]:
    #     print("Exporting color frames for: ", base_dir)

    timesync = TimesynchronizerPCPD(export_config)
    # dir_path = os.path.join(INPUT_DIR, base_dir)
    dir_path = "/mnt/atlas_2/03_animal_trials/211214_animal_trial_17/recordings/recordings_1412_recording_04/"
    files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]
    print("Files:", files)

    timesync.initialize_feeds(files, MkvPath("/home/tonyw/extract_mkv_k4a/output/trial_17_recording_04/"))

    timesync.run()

    del timesync