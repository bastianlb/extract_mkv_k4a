import os

from mkv_extractor import set_log_level
from mkv_extractor import TimesynchronizerPCPD, ExportConfig, Path as MkvPath


INPUT_DIR = "/data/input/"

if __name__ == "__main__":
    print("Color Video export")
    print(os.listdir(INPUT_DIR))
    export_config = ExportConfig()
    export_config.export_color = True
    export_config.max_frames_exported = 100
    export_config.skip_frames = 10
    export_config.timesync = True
    # TODO: start end must be recording dependent...
    #                        1628650257725706448
    # export_config.start_ts = 1628651055148343000
    # export_config.end_ts =   1628653016064450000
    #                        1628658895596854977

    set_log_level("info")

    # for base_dir in ["recordings_1108_recording_1"]:  # ,"recordings_2809_recording_5"]:
    for base_dir in os.listdir(INPUT_DIR):
        print("Exporting color frames for: ", base_dir, flush=True)

        timesync = TimesynchronizerPCPD(export_config)
        dir_path = os.path.join(INPUT_DIR, base_dir)
        files = [os.path.join(dir_path, x) + "/" for x in os.listdir(dir_path)]

        timesync.initialize_feeds(files, MkvPath("/data/export/" + base_dir))

        timesync.run()
        del timesync
