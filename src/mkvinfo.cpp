#include <thread>
#include <string>
#include <fstream>
#include <iostream>

#include "libyuv.h"
#include "turbojpeg.h"
#include "opencv2/opencv.hpp"
#include "zdepth.hpp"
#include "Eigen/Core"
#include "capnp/serialize.h"
#include "cppfs/fs.h"
#include "cppfs/FilePath.h"
#include "cppfs/FileHandle.h"
#include "serialization/capnproto_serialization.h"
#include "boost/algorithm/string/split.hpp"
#include "boost/algorithm/string/classification.hpp"
#include "pcpd/record/mkv/matroska_read.h"
#include "pcpd/config.h"
#include "pcpd/rttr_registration.h"
#include "pcpd/record/enumerations.h"
#include "pcpd/service/locator.h"
#include "pcpd/record/mkv_single_filereader.h"
#include "pcpd/record/mkv_sync_track_loader.h"

using namespace pcpd;
using namespace pcpd::record;
using namespace rttr;

const std::string NL {"\n"};

// using std::map<std::string, std::string> = SETTING_MAP;

void log(const std::string& msg)
{
    std::cout << msg << std::endl;
}

std::string indent(int n)
{
    std::string s {};
    for(int i = 0; i < n; ++i)
        s += "         ";

    return s;
}

std::string asString(datatypes::PixelFormatType pixel_format)
{
    // https://www.rttr.org/doc/master/classrttr_1_1enumeration.html
    // https://www.axelmenzel.de/projects/coding/rttr/doc/page_tutorial.html

    return std::string {type::get(pixel_format).get_enumeration().value_to_name(pixel_format)};
}

int pixelFormatToBits(datatypes::PixelFormatType pixel_format)
{
    switch(pixel_format)
    {
        case datatypes::PixelFormatType::BGRA:
        case datatypes::PixelFormatType::RGBA:
        case datatypes::PixelFormatType::MJPEG:
            return 32;
        case datatypes::PixelFormatType::RGB:
        case datatypes::PixelFormatType::BGR:
            return 24;
        case datatypes::PixelFormatType::DEPTH:
        case datatypes::PixelFormatType::ZDEPTH:
            return 16;
        case datatypes::PixelFormatType::LUMINANCE:
            return 8;
        default:
            return 0;
    }
}

std::string asString(int level, artekmed::schema::RecordingSchema::Channel::Reader& reader)
{
    std::stringstream ss;

    int ch_idx = reader.getIndex();
    std::string ch_name = reader.getName().cStr();

    std::string il = indent(level);

    ss << il << "index:                " << ch_idx                          << NL;
    ss << il << "name:                 " << ch_name                         << NL;

    auto storage = reader.getStorage();
    
    if(storage.isMkvVideo())
    {
        auto video = storage.getMkvVideo();

        int width = video.getWidth();
        int height = video.getHeight();
        auto pixel_format = serialization::CapnprotoArchive::getPixelFormatC2P(video.getPixelFormat());
        std::string pixel_format_str = asString(pixel_format);
        auto pixel_format_storage = serialization::CapnprotoArchive::getPixelFormatC2P(video.getPixelFormatStorage());
        std::string pixel_format_storage_str = asString(pixel_format_storage);
        std::string codec_name = video.hasCodecName() ? video.getCodecName().cStr() : "";
        int bits_per_element = pixelFormatToBits(pixel_format);
        int stride = (width * bits_per_element) / 8;
        std::string info {"-"};

        if(codec_name == "V_MPEG4/ISO/AVC")
        {
            info = "assume: H264, BGRA, 32-bit";
        }
    
        ss << il << "track_type:           " << "video"                         << NL; 
        ss << il << "width:                " << width                           << NL;
        ss << il << "height:               " << height                          << NL;
        ss << il << "bits_per_element:     " << bits_per_element                << NL;
        ss << il << "stride:               " << stride                          << NL;
        ss << il << "pixel_format:         " << pixel_format_str                << NL;
        ss << il << "pixel_format_storage: " << pixel_format_storage_str        << NL;
        ss << il << "codec:                " << codec_name                      << NL;
        ss << il << "info:                 " << info                            << NL;
    }

    if(storage.isMkvData())
    {
        auto data = storage.getMkvData();

        int max_packet_len = data.getMaxPacketLen();

        ss << il << "track_type:           " << "data"                          << NL;
        ss << il << "max_packet_len::      " << max_packet_len                  << NL;
    }

    return ss.str();
}

/*
SETTING_MAP map(artekmed::schema::RecordingSchema::Channel& channel)
{
    SETTING_MAP map;

    int ch_idx = channel.getIndex();
    std::string ch_name = channel.getName();
    auto storage = channel.getStorage();
    auto video = storage.getMkvVideo();    
    int width = video.getWidth();
    int height = video.getHeight();
    datatypes::PixelFormatType pixel_format = 
        serialization::CapnprotoArchive::getPixelFormatC2P(video.getPixelFormat());
    datatypes::PixelFormatType pixel_format_storage = 
        serialization::CapnprotoArchive::getPixelFormatC2P(video.getPixelFormatStorage());

    map.insert(std::make_pair("index", fmt::format("{0}", ch_idx)));
    // TODO:
    map.insert(std::make_pair("pixel_format", toString(pixel_format)));

    return map;
}
*/

std::string asString(int level, artekmed::schema::RecordingSchema::CalibrationEntry::Reader& reader)
{
    std::stringstream ss;
    std::string device_name = reader.getName().cStr();

    auto tm = serialization::CapnprotoArchive::typeMap<datatypes::DeviceCalibration>();
    auto calib_reader = reader.getCalibration();

    datatypes::DeviceCalibration calib;
    tm.deserialize(calib_reader, calib);

    std::string il = indent(level);

    ss << il << "device:               " << device_name << NL;

    return ss.str();
}

void printInfo(const std::string& mkvFilePath)
{
    MKVPlayerBase mkv{mkvFilePath};
    auto ret1 = mkv.mkv_open();

    if(ret1 == PCPD_RESULT_FAILED)
    {
        log(fmt::format("Could not open MKV file: {0}", mkvFilePath));
        return;
    }

    size_t attachment_size = 2048;
    uint8_t attachment[attachment_size];
    const std::string RECORDING_SCHEMA_NAME = "record_schema.capnp";

    auto ret2 = mkv.mkv_get_attachment(RECORDING_SCHEMA_NAME.c_str(), attachment, &attachment_size);

    if(ret2 == PCPD_BUFFER_RESULT_TOO_SMALL)
    {
        log(fmt::format("Buffer for recording schema attachment to small, need: {0}", attachment_size));
        return;
    }

    if(ret2 == PCPD_BUFFER_RESULT_FAILED)
    {
        log(fmt::format("Could not find recording schema attachment: {0}", RECORDING_SCHEMA_NAME));
        return;
    }

    kj::ArrayPtr<kj::byte> bufferPtr = kj::arrayPtr(attachment, attachment_size);
    kj::ArrayInputStream ins (bufferPtr);      
    ::capnp::InputStreamMessageReader message(ins);

    artekmed::schema::RecordingSchema::Reader reader = message.getRoot<artekmed::schema::RecordingSchema>();
    const std::string& mkv_name = reader.getName();

    std::stringstream ss;

    ss << RECORDING_SCHEMA_NAME             << NL;
    ss << "size:     " << attachment_size   << NL;
    ss << "name:     " << mkv_name          << NL;

    ss << "channels: "                      << NL;
    for(auto channel : reader.getChannels())
    {
        ss << asString(1, channel);
        ss << NL;
    }

    ss << "calibration: "                   << NL;
    for(auto calibration : reader.getCalibrations())
    {
        ss << asString(1, calibration);
        ss << NL;
    }
    
    mkv.mkv_close();

    std::cout << ss.str() << std::endl;
}

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        log("usage: mkvinfo <PATH_TO_MKV_FILE>");
        return -1;
    }
    std::vector<std::string> filepaths;

    MkvSyncTrackLoader loader{nullptr, filepaths};
    std::string mkvFilePath {argv[1]};
    printInfo(mkvFilePath);

    return 0;
}
