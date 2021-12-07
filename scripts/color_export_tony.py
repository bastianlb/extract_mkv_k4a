import os
from datetime import timedelta

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


INPUT_DIR = "/media/storage/atlas_recordings/"

if __name__ == "__main__":
    print("Color Video export")
    print(os.listdir(INPUT_DIR))
    export_config = ExportConfig()
    export_config.export_color = True
    export_config.max_frames_exported = 10000
    export_config.skip_frames = 30
    # TODO: start end must be recording dependent...
    #                        1628650257725706448
    # export_config.start_ts = timedelta(microseconds=1628651055148343000 // 1000)
    #export_config.end_ts =   timedelta(microseconds=1628653016064450000 // 1000)
    #                        1628658895596854977

    set_log_level("info")

    for base_dir in ["recordings_2809_recording_2"]:  #, "recordings_2809_recording_5"]:
        print("Exporting color frames for: ", base_dir)

        timesync = TimesynchronizerPCPD(export_config)
        dir_path = os.path.join(INPUT_DIR, base_dir)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        timesync.initialize_feeds(files, MkvPath("/data/datasets/atlas_export/"))

        timesync.run()

        del timesync
