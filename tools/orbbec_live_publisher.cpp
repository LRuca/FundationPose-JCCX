#include "libobsensor/ObSensor.hpp"

#include <atomic>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::atomic<bool> g_running{true};

void handle_signal(int) {
    g_running = false;
}

struct Args {
    std::string out_dir = "third_party/FoundationPose/live_orbbec";
    int max_frames = 0;
    int warmup = 5;
    int wait_ms = 100;
};

Args parse_args(int argc, char **argv) {
    Args args;
    for(int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const char *name) -> std::string {
            if(i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if(key == "--out_dir") {
            args.out_dir = need_value("--out_dir");
        }
        else if(key == "--max_frames") {
            args.max_frames = std::stoi(need_value("--max_frames"));
        }
        else if(key == "--warmup") {
            args.warmup = std::stoi(need_value("--warmup"));
        }
        else if(key == "--wait_ms") {
            args.wait_ms = std::stoi(need_value("--wait_ms"));
        }
        else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

void run_cmd(const std::string &cmd) {
    int rc = std::system(cmd.c_str());
    if(rc != 0) {
        throw std::runtime_error("command failed: " + cmd);
    }
}

std::string shell_quote(const std::string &s) {
    std::string out = "'";
    for(char c : s) {
        if(c == '\'') {
            out += "'\\''";
        }
        else {
            out += c;
        }
    }
    out += "'";
    return out;
}

void atomic_rename(const std::string &tmp, const std::string &dst) {
    if(std::rename(tmp.c_str(), dst.c_str()) != 0) {
        throw std::runtime_error("rename failed: " + tmp + " -> " + dst);
    }
}

void write_ppm_rgb(const std::string &path, const uint8_t *rgb, uint32_t width, uint32_t height) {
    std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if(!f) {
        throw std::runtime_error("failed to open " + tmp);
    }
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write(reinterpret_cast<const char *>(rgb), static_cast<std::streamsize>(width * height * 3));
    f.close();
    atomic_rename(tmp, path);
}

void write_pgm16_be(const std::string &path, const uint16_t *depth, uint32_t width, uint32_t height) {
    std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if(!f) {
        throw std::runtime_error("failed to open " + tmp);
    }
    f << "P5\n" << width << " " << height << "\n65535\n";
    std::vector<uint8_t> be(static_cast<size_t>(width) * height * 2);
    for(size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        be[2 * i] = static_cast<uint8_t>((depth[i] >> 8) & 0xff);
        be[2 * i + 1] = static_cast<uint8_t>(depth[i] & 0xff);
    }
    f.write(reinterpret_cast<const char *>(be.data()), static_cast<std::streamsize>(be.size()));
    f.close();
    atomic_rename(tmp, path);
}

void write_text_atomic(const std::string &path, const std::string &content) {
    std::string tmp = path + ".tmp";
    std::ofstream f(tmp);
    if(!f) {
        throw std::runtime_error("failed to open " + tmp);
    }
    f << content;
    f.close();
    atomic_rename(tmp, path);
}

std::shared_ptr<ob::ColorFrame> to_rgb(std::shared_ptr<ob::ColorFrame> color, ob::FormatConvertFilter &filter) {
    if(color->format() == OB_FORMAT_RGB) {
        return color;
    }
    if(color->format() == OB_FORMAT_MJPG) {
        filter.setFormatConvertType(FORMAT_MJPG_TO_RGB);
    }
    else if(color->format() == OB_FORMAT_UYVY) {
        filter.setFormatConvertType(FORMAT_UYVY_TO_RGB);
    }
    else if(color->format() == OB_FORMAT_YUYV) {
        filter.setFormatConvertType(FORMAT_YUYV_TO_RGB);
    }
    else if(color->format() == OB_FORMAT_BGR) {
        filter.setFormatConvertType(FORMAT_BGR_TO_RGB);
    }
    else {
        throw std::runtime_error("unsupported color format");
    }
    auto converted = filter.process(color);
    if(!converted) {
        throw std::runtime_error("color conversion returned null");
    }
    return converted->as<ob::ColorFrame>();
}
} // namespace

int main(int argc, char **argv) try {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    Args args = parse_args(argc, argv);
    run_cmd("mkdir -p " + shell_quote(args.out_dir));

    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();

    auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(colorProfile);

    auto depthProfiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    auto depthProfile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(depthProfile);

    ob::Align align(OB_STREAM_COLOR);
    ob::FormatConvertFilter colorConvert;

    pipeline.start(config);
    auto cameraParam = pipeline.getCameraParam();
    const auto &k = cameraParam.rgbIntrinsic;
    std::ostringstream k_txt;
    k_txt << k.fx << " 0 " << k.cx << "\n"
          << "0 " << k.fy << " " << k.cy << "\n"
          << "0 0 1\n";
    write_text_atomic(args.out_dir + "/cam_K.txt", k_txt.str());

    int seen = 0;
    int published = 0;
    while(g_running && (args.max_frames <= 0 || published < args.max_frames)) {
        auto frameset = pipeline.waitForFrames(args.wait_ms);
        if(frameset == nullptr) {
            continue;
        }
        if(seen++ < args.warmup) {
            continue;
        }

        auto aligned = align.process(frameset);
        if(aligned != nullptr) {
            frameset = aligned->as<ob::FrameSet>();
        }

        auto colorFrame = frameset->colorFrame();
        auto depthFrame = frameset->depthFrame();
        if(colorFrame == nullptr || depthFrame == nullptr) {
            continue;
        }

        auto rgbFrame = to_rgb(colorFrame, colorConvert);
        if(depthFrame->format() != OB_FORMAT_Y16) {
            throw std::runtime_error("depth format is not Y16");
        }

        write_ppm_rgb(
            args.out_dir + "/color.ppm",
            reinterpret_cast<const uint8_t *>(rgbFrame->data()),
            rgbFrame->width(),
            rgbFrame->height());
        write_pgm16_be(
            args.out_dir + "/depth.pgm",
            reinterpret_cast<const uint16_t *>(depthFrame->data()),
            depthFrame->width(),
            depthFrame->height());

        std::ostringstream meta;
        meta << "{\n"
             << "  \"frame_index\": " << published << ",\n"
             << "  \"color_file\": \"color.ppm\",\n"
             << "  \"depth_file\": \"depth.pgm\",\n"
             << "  \"color_timestamp_ms\": " << rgbFrame->timeStamp() << ",\n"
             << "  \"depth_timestamp_ms\": " << depthFrame->timeStamp() << ",\n"
             << "  \"depth_unit\": \"millimeter_uint16\",\n"
             << "  \"color_width\": " << rgbFrame->width() << ",\n"
             << "  \"color_height\": " << rgbFrame->height() << ",\n"
             << "  \"depth_width\": " << depthFrame->width() << ",\n"
             << "  \"depth_height\": " << depthFrame->height() << "\n"
             << "}\n";
        write_text_atomic(args.out_dir + "/frame.json", meta.str());

        std::cout << "published frame " << published
                  << " color=" << rgbFrame->width() << "x" << rgbFrame->height()
                  << " depth=" << depthFrame->width() << "x" << depthFrame->height()
                  << std::endl;
        ++published;
    }

    pipeline.stop();
    return 0;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getName()
              << "\nargs:" << e.getArgs()
              << "\nmessage:" << e.getMessage()
              << "\ntype:" << e.getExceptionType() << std::endl;
    return 1;
}
catch(std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
}
