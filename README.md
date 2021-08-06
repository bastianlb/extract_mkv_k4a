Artekmed Application Framework for Development and Research
===========================================================

Tool to export frames from an MKV file recorded using the Kinect for Azure SDK

Building:
 - ```conan install .. -s build_type=Debug|Release --build "*"```
 - ```cmake .. -DCMAKE_BUILD_TYPE=Debug```
 - ```make```
 
 Usage:
 - ```source activate_run.sh```
 - ```./bin/export_mkv_k4a --help```
 Run with example config
 - ```./bin/export_mkv_k4a --config ../configs/debug.yml```
 
 ```
 ./bin/extract_mkv_k4a [--magnum-...] [-h|--help] [-c|--config CONFIG] [-l|--loglevel LOGLEVEL]

Arguments:
  -h, --help               display this help message and exit
  -c, --config CONFIG      config file for settings
  -l, --loglevel LOGLEVEL  spdlog logging
                           (default: info)
  --magnum-...             engine-specific options
                           (see --magnum-help for details)
```
