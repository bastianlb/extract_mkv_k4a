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
    }

    requires = (
        "opencv/4.5.0@camposs/stable",
        "eigen/[3.3.9]@camposs/stable",
        "magnum/2020.06@camposs/stable",
        "corrade/2020.06@camposs/stable",
        "kinect-azure-sensor-sdk/1.4.1@camposs/stable",
        "bzip2/1.0.8@conan/stable",
        "spdlog/1.8.2",
        "yaml-cpp/0.6.3",
        "tbb/2020.3",
        "jsoncpp/1.9.4",
         )

    default_options = {
        "opencv:shared": True,
        "with_python": True,
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
