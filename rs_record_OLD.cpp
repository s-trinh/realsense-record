#include <visp3/core/vpImage.h>
#include <visp3/core/vpImageConvert.h>
//#include <visp3/gui/vpDisplayGDI.h>
//#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
//#include <visp3/sensor/vpRealSense.h>
//#include <visp3/sensor/vpRealSense2.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>

#include "hve.h"

//#include <librealsense2/rs.hpp>

int main(int argc, char *argv[])
{
//    rs2::pipeline pipe;

//    // Start streaming with default recommended configuration
//    // The default video configuration contains Depth and Color streams
//    // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
//    pipe.start();

//    return 0;



  float min_depth = 0.29f;
  float max_depth = 16.0f;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--min_depth" && i+1 < argc) {
      min_depth = std::atof(argv[i+1]);
    } else if (std::string(argv[i]) == "--max_depth" && i+1 < argc) {
      max_depth = std::atof(argv[i+1]);
    }
  }

  std::cout << "min_depth: " << min_depth << std::endl;
  std::cout << "max_depth: " << max_depth << std::endl;

  const int width = 848, height = 480, fps = 30;
  rs2::config cfg;
  //cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
  //cfg.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
  cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
  cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGBA8, fps);

  //vpRealSense2 rs;
  //std::cout << "Before open" << std::endl;
  //rs.open(cfg);
  //std::cout << "After open" << std::endl;

  rs2::pipeline pipe;
  std::cout << "START pipe.start(cfg)" << std::endl;
  rs2::pipeline_profile profile = pipe.start(cfg);
  std::cout << "END pipe.start(cfg)" << std::endl;
  rs2::device selected_device = profile.get_device();
  //rs2::pipeline pipe = rs.getPipeline();
  //rs2::pipeline_profile selection = rs.getPipelineProfile();
  //rs2::device selected_device = selection.get_device();
  auto depth_sensor = selected_device.first<rs2::depth_sensor>();

  if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
//        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
  }

  // Filters
  rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
  rs2::colorizer color_filter;        // Colorize - convert from depth to RGB color

  // filter settings
  thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);
  thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);

  // color filter setting
  color_filter.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0);
  color_filter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);
  color_filter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);
  color_filter.set_option(RS2_OPTION_COLOR_SCHEME, 9.0f);   // Hue colorization

  vpImage<uint16_t> I_depth_raw(height, width);
  vpImage<vpRGBa> I_depth(height, width);
  vpImage<vpRGBa> I_color(height, width);

  vpDisplayX d1(I_color, 0, 0, "Color");
  vpDisplayX d2(I_depth, I_color.getWidth(), 0, "Depth");

  struct hve *hardware_encoder;
  struct hve_config hardware_config = {0};
  hardware_config.width = width;
  hardware_config.height = height;
  hardware_config.framerate = fps;
//  hardware_config.device = "/dev/dri/renderD128";
  hardware_config.device = "";

  if( (hardware_encoder = hve_init(&hardware_config)) == NULL) {
    std::cerr << "Failed to do hve_init" << std::endl;
    return 0;
  }

  std::ofstream out_file("output.h264", std::ofstream::binary);
  hve_frame frame = {0};
  AVPacket *packet;

  cv::Mat rgb_color_depth_mat = cv::Mat(cv::Size(width, height), CV_8UC3).setTo(0);
  while (true) {
    rs2::frameset data = pipe.wait_for_frames();      // Wait for next set of frames from the camera
    rs2::frame depth_frame = data.get_depth_frame();  //Take the depth frame from the frameset
    if (!depth_frame) { break; }                      // Should not happen but if the pipeline is configured differently

    rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference
    filtered = thr_filter.process(filtered);
    filtered = color_filter.process(filtered);
    rs2::video_frame filtered_frame = filtered;

    memcpy(rgb_color_depth_mat.ptr<uchar>(), filtered_frame.get_data(), sizeof(unsigned char) * width * height * 3);

    frame.linesize[0] = filtered_frame.get_stride_in_bytes();
    frame.data[0] = (uint8_t*) rgb_color_depth_mat.ptr<uchar>();

    if (hve_send_frame(hardware_encoder, &frame) != HVE_OK)
    {
      std::cerr << "failed to send frame to hardware" << std::endl;
      break;
    }

    int failed = HVE_OK;
    while( (packet = hve_receive_packet(hardware_encoder, &failed)) )
    { //do something with the data - here just dump to raw H.264 file
      std::cout << " encoded in: " << packet->size;
      out_file.write((const char*)packet->data, packet->size);
    }

    if(failed != HVE_OK)
    {
      std::cerr << "failed to encode frame" << std::endl;
      break;
    }

    cv::imshow("Colorized depth", rgb_color_depth_mat);
    cv::waitKey(1);

//    rs.acquire(reinterpret_cast<unsigned char *>(I_color.bitmap), reinterpret_cast<unsigned char *>(I_depth_raw.bitmap), nullptr, nullptr);

//    vpImageConvert::createDepthHistogram(I_depth_raw, I_depth);

    vpDisplay::display(I_color);
    vpDisplay::display(I_depth);
    vpDisplay::flush(I_color);
    vpDisplay::flush(I_depth);

    if (vpDisplay::getClick(I_color, false)) {
      break;
    }
  }

  hve_close(hardware_encoder);

  return 0;
}
