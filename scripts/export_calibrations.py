import os

from mkv_extractor import set_log_level
from mkv_extractor import Timesynchronizer, ExportConfig, Path as MkvPath


calibration_config = {
    # "210720_calibration_1": {
    #     "path": "/media/lennart/1.42.6-25556/recording_backup/02_pre_trials/210720_pre_trail_04/calibrations/calibrations_1907_calibration_1/",  # noqa
    #     "feeds": [
    #         "cn01/capture-000000.mkv",
    #         "cn02/capture-000000.mkv",
    #         "cn03/capture-000000.mkv",
    #         "cn04/capture-000000.mkv",
    #     ]
    # },
    # "210720_calibration_2": {
    #     "path": "/media/lennart/1.42.6-25556/recording_backup/02_pre_trials/210720_pre_trail_04/calibrations/calibrations_1907_calibration_2/",  # noqa
    #     "feeds": [
    #         "cn01/capture-000000.mkv",
    #         "cn02/capture-000000.mkv",
    #         "cn03/capture-000000.mkv",
    #         "cn04/capture-000000.mkv",
    #     ]
    # },
    # "210810_calibration_1": {
    #     "path": "/storage/atlas_calibrations/210810_animal_trial_01/calibrations/calibration_01/recordings_narvis/",  # noqa
    #     "feeds": [
    #         "cn01/capture-000000.mkv",
    #         "cn02/capture-000000.mkv",
    #         "cn03/capture-000000.mkv",
    #         "cn04/capture-000000.mkv",
    #     ]
    # },
    # "210824_calibration_1": {
    #     "path": "/storage/atlas_calibrations/210824_animal_trial_02/calibrations/calibration_01/recordings",  # noqa
    #     "feeds": [
    #         "cn01/capture-000000.mkv",
    #         "cn02/capture-000000.mkv",
    #         "cn03/capture-000000.mkv",
    #         "cn04/capture-000000.mkv",
    #     ]
    # },
    # "210824_calibration_2": {
    #     "path": "/storage/atlas_calibrations/210824_animal_trial_02/calibrations/calibration_02/recordings",  # noqa
    #     "feeds": [
    #         "cn01/capture-000000.mkv",
    #         "cn02/capture-000000.mkv",
    #         "cn03/capture-000000.mkv",
    #         "cn04/capture-000000.mkv",
    #     ]
    # },
}


if __name__ == "__main__":
    export_config = ExportConfig()
    export_config.export_extrinsics = True
    export_config.export_color = True

    first_frame = 0
    last_frame = 100
    skip_frames = 20

    set_log_level("trace")

    for key, params in calibration_config.items():
        timesync = Timesynchronizer(first_frame, last_frame, skip_frames,
                                    export_config, True, False)

        base_dir = params["path"]  # noqa

        files = map(lambda x: MkvPath(os.path.join(base_dir, x)),
                    params["feeds"])

        timesync.initialize_feeds(list(files),
                                  MkvPath(os.path.join("../test_output", key)))

        timesync.run()

        del timesync
