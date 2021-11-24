from conans import ConanFile, CMake
from conans import tools
from conans.tools import os_info, SystemPackageTool
import os, sys
import sysconfig
from io import StringIO

class ExportMKVConan(ConanFile):
    name = "export_mkv_k4a"
    version = "0.1.0"

    description = "export_mkv_k4a"
    url = "https://github.com/TUM-CAMP-NARVIS/export_mkv_k4a"
    license = "GPL"

    short_paths = True
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake", "virtualrunenv"

    options = {
        "with_python": [True, False],
        "with_pcpd": [True, False],
    }

    requires = (
        "opencv/4.5.1@camposs/stable",
        "eigen/3.3.9-r1@camposs/stable",
        "magnum/2020.06@camposs/stable",
        "corrade/2020.06@camposs/stable",
        "kinect-azure-sensor-sdk/1.4.1-r1@camposs/stable",
        "kinect-azure-bodytracking-sdk/1.1.0@vendor/stable",
        "bzip2/1.0.8@conan/stable",
        "zlib/1.2.11-r1@camposs/stable",  # ffmpeg version needs to be overriden
        "fmt/8.0.1",
        "spdlog/1.9.1",
        "yaml-cpp/0.6.3",
        "tbb/2020.3",
        "jsoncpp/1.9.4",
        "happly/cci.20200822",
    )

    default_options = {
        "opencv:shared": True,
        "opencv:with_ffmpeg": True,
        "with_python": True,
        "with_pcpd": False,
        "magnum:with_anyimageimporter": True,
        "magnum:with_tgaimporter": True,
        "magnum:with_anysceneimporter": True,
        "magnum:with_gl_info": True,
        "magnum:with_objimporter": True,
        "magnum:with_tgaimageconverter": True,
        "magnum:with_imageconverter": True,
        "magnum:with_anyimageconverter": True,
        "magnum:with_sdl2application": True,
        "magnum:with_eglcontext": False,
        "magnum:with_windowlesseglapplication": False,
        "magnum:target_gles": False,
        "magnum:with_opengltester": True,
    }

    # all sources are deployed with the package
    exports_sources = "cmake/*", "include/*", "src/*", "CMakeLists.txt"

    def requirements(self):
        if self.options.with_python:
            self.requires("python_dev_config/[>=1.0]@camposs/stable")
            self.requires("pybind11/2.7.1@camposs/stable")
        if self.options.with_pcpd:
            self.requires("Boost/1.75.0-r2@camposs/stable")
            self.requires("pcl/1.11.1-r3@camposs/stable")
            self.requires("rapidjson/1.1.0")
            self.requires("cuda_dev_config/[>=2.0]@camposs/stable")
            self.requires("ringbuffer/0.2.7@artekmed/stable")
            self.requires("rtsplib/0.1.5@artekmed/stable")
            self.requires("enet/1.3.17@camposs/stable")
            self.requires("draco/1.4.1@camposs/stable")
            self.requires("cppfs/1.3.0@camposs/stable")
            self.requires("cereal/1.3.0")
            self.requires("capnproto/0.8.0@camposs/stable")
            self.requires("matroska/1.6.2@camposs/stable")
            self.requires("libjpeg-turbo/2.1.0")
            self.requires("yuv/1749@camposs/stable")
            self.requires("zdepth/0.1@camposs/stable")
            self.requires("optick/1.3.1.0@camposs/stable")
            self.requires("msgpack/3.2.0@camposs/stable")
            self.requires("zmq/4.3.2@camposs/stable")
            self.requires("azmq/1.0.3@camposs/stable")
            self.requires("zstd/1.4.3")
            self.requires("eventbus/3.0.1-r1@camposs/stable")
            self.requires("rttr/0.9.7-dev@camposs/stable")
            self.requires("nvidia-video-codec-sdk/11.0.10@vendor/stable")

    def configure(self):

        if self.settings.os == "Linux":
            self.options["opencv"].with_gtk = True

        if self.settings.os == "Macos":
            self.options['magnum-extras'].with_ui_gallery = False
            self.options['magnum-extras'].with_player = False
            self.options['magnum-plugins'].with_tinygltfimporter = False


        if self.settings.os == "Windows":
            self.options['magnum'].with_windowlesswglapplication = True

    def imports(self):
        self.copy(src="bin", pattern="*.dll", dst="./bin") # Copies all dll files from packages bin folder to my "bin" folder
        self.copy(src="lib", pattern="*.dll", dst="./bin") # Copies all dll files from packages bin folder to my "bin" folder
        self.copy(src="lib", pattern="*.dylib*", dst="./lib") # Copies all dylib files from packages lib folder to my "lib" folder
        self.copy(src="lib", pattern="*.so*", dst="./lib") # Copies all so files from packages lib folder to my "lib" folder
        self.copy(src="lib", pattern="*.a", dst="./lib") # Copies all static libraries from packages lib folder to my "lib" folder
        self.copy(src="bin", pattern="*", dst="./bin") # Copies all applications
        if self.options.with_python:
            with tools.run_environment(self):
                python_version = os.environ.get("PYTHON_VERSION", None) or "3.8"
                self.output.write("Collecting python modules in ./lib/python%s" % python_version)
                self.copy(src="lib/python%s" % python_version, pattern="*", dst="./lib/python%s" % python_version, keep_path=True) # Copies all python modules
                self.copy(src="lib/python", pattern="*", dst="./lib/python", keep_path=True) # Copies all python modules

    def _cmake_configure(self):
        cmake = CMake(self)
        cmake.verbose = True
        cmake.configure()
        return cmake
       
    def build(self):
        cmake = self._cmake_configure()
        cmake.build()

    def package(self):
        cmake = self._cmake_configure()
        cmake.install()
