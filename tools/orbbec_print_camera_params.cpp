#include "libobsensor/ObSensor.hpp"

#include <iostream>
#include <memory>

int main() try {
    ob::Pipeline pipeline;
    auto config = std::make_shared<ob::Config>();

    auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    auto colorProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(colorProfile);

    auto depthProfiles = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
    auto depthProfile = depthProfiles->getProfile(OB_PROFILE_DEFAULT);
    config->enableStream(depthProfile);

    auto colorVideo = colorProfile->as<ob::VideoStreamProfile>();
    auto depthVideo = depthProfile->as<ob::VideoStreamProfile>();
    pipeline.start(config);
    auto cameraParam = pipeline.getCameraParam();
    pipeline.stop();

    const auto &rgb = cameraParam.rgbIntrinsic;
    const auto &depth = cameraParam.depthIntrinsic;

    std::cout << "rgb "
              << rgb.fx << " " << rgb.fy << " " << rgb.cx << " " << rgb.cy
              << " " << rgb.width << " " << rgb.height << std::endl;
    std::cout << "depth "
              << depth.fx << " " << depth.fy << " " << depth.cx << " " << depth.cy
              << " " << depth.width << " " << depth.height << std::endl;
    return 0;
}
catch(ob::Error &e) {
    std::cerr << "function:" << e.getName()
              << "\nargs:" << e.getArgs()
              << "\nmessage:" << e.getMessage()
              << "\ntype:" << e.getExceptionType() << std::endl;
    return 1;
}
