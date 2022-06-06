#include <iostream>
#include <iomanip>
#include <condition_variable>
#include <queue>
#include <thread>
#include <atomic>

#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

static void get_sensor_option(const rs2::sensor& sensor)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    std::cout << "Sensor supports the following options:\n" << std::endl;

    // The following loop shows how to iterate over all available options
    // Starting from 0 until RS2_OPTION_COUNT (exclusive)
    for (int i = 0; i < static_cast<int>(RS2_OPTION_COUNT); i++)
    {
        rs2_option option_type = static_cast<rs2_option>(i);
        //SDK enum types can be streamed to get a string that represents them
        std::cout << "  " << i << ": " << option_type;

        // To control an option, use the following api:

        // First, verify that the sensor actually supports this option
        if (sensor.supports(option_type))
        {
            std::cout << std::endl;

            // Get a human readable description of the option
            const char* description = sensor.get_option_description(option_type);
            std::cout << "       Description   : " << description << std::endl;

            // Get the current value of the option
            float current_value = sensor.get_option(option_type);
            std::cout << "       Current Value : " << current_value << std::endl;

            //To change the value of an option, please follow the change_sensor_option() function
        }
        else
        {
            std::cout << " is not supported" << std::endl;
        }
    }
}

static void change_sensor_option(const rs2::sensor& sensor, rs2_option option_type, float requested_value)
{
    // Sensors usually have several options to control their properties
    //  such as Exposure, Brightness etc.

    // To control an option, use the following api:

    // First, verify that the sensor actually supports this option
    if (!sensor.supports(option_type))
    {
        std::cerr << "This option is not supported by this sensor" << std::endl;
        return;
    }

    // To set an option to a different value, we can call set_option with a new value
    try
    {
        sensor.set_option(option_type, requested_value);
    }
    catch (const rs2::error& e)
    {
        // Some options can only be set while the camera is streaming,
        // and generally the hardware might fail so it is good practice to catch exceptions from set_option
        std::cerr << "Failed to set option " << option_type << ". (" << e.what() << ")" << std::endl;
    }
}

static void reverse_copy_rgb(unsigned char * dst, const unsigned char * const src, int width, int height)
{
  const int channels = 3;
  const unsigned char * src_ptr = src + width*height*channels;
//  while (src != src_ptr) {
//    *(dst++) = *(--src_ptr);
//  }
  for (int i = 0; i < width*height*channels; i+= 3) {
    dst[i+2] = *(--src_ptr);
    dst[i+1] = *(--src_ptr);
    dst[i+0] = *(--src_ptr);
  }
}

namespace
{
// Code adapted from the original author Dan Mašek to be compatible with ViSP image
class FrameQueue
{
public:
  struct cancelled {
  };

  FrameQueue(size_t max_queue_size=std::numeric_limits<size_t>::max())
    : m_cancelled(false), m_cond(), m_queueImg(), m_maxQueueSize(max_queue_size), m_mutex()
  {
  }

  void cancel()
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cancelled = true;
    m_cond.notify_all();
  }

  // Push the image to save in the queue (FIFO)
  void push(const cv::Mat& image)
  {
    std::lock_guard<std::mutex> lock(m_mutex);

    m_queueImg.push(image.clone());

    // Pop extra images in the queue
    while (m_queueImg.size() > m_maxQueueSize) {
      m_queueImg.pop();
    }

    m_cond.notify_one();
  }

  // Pop the image to save from the queue (FIFO)
  cv::Mat pop()
  {
    std::unique_lock<std::mutex> lock(m_mutex);

    while (m_queueImg.empty()) {
      if (m_cancelled) {
        throw cancelled();
      }

      m_cond.wait(lock);

      if (m_cancelled) {
        throw cancelled();
      }
    }

    cv::Mat image(m_queueImg.front());
    m_queueImg.pop();

    return image;
  }

private:
  bool m_cancelled;
  std::condition_variable m_cond;
  std::queue<cv::Mat> m_queueImg;
  size_t m_maxQueueSize;
  std::mutex m_mutex;
};

class StorageWorker
{
public:
  StorageWorker(FrameQueue &queue, const std::string &filename, int width, int height, int fps)
    : m_queue(queue), m_filename(filename), m_width(width), m_height(height), m_fps(fps)
  {
  }

  // Thread main loop
  void run()
  {
    try {
      cv::VideoWriter writer;
      if (!m_filename.empty()) {
        writer = cv::VideoWriter(m_filename, cv::VideoWriter::fourcc('H', '2', '6', '4'), m_fps, cv::Size(m_width, m_height));
        if (!writer.isOpened()) {
          std::cerr << "Cannot write!" << std::endl;
        }
      }

      for (;;) {
        cv::Mat image(m_queue.pop());

        if (!m_filename.empty()) {
           writer.write(image);
        }
      }
    } catch (FrameQueue::cancelled &) {
    }
  }

private:
  FrameQueue &m_queue;
  std::string m_filename;
  int m_width;
  int m_height;
  int m_fps;
};

class DisplayWorker
{
public:
  DisplayWorker(FrameQueue &queue, std::atomic<bool>& abort)
    : m_queue(queue), m_abort(abort)
  {
  }

  // Thread main loop
  void run()
  {
    try {
      for (;;) {
        cv::Mat image(m_queue.pop());

        if (!image.empty()) {
          cv::imshow("D455", image);
        }

        int key = cv::waitKey(1);
        if (key == 27) {
          m_abort = true;
          break;
        }
      }
    } catch (FrameQueue::cancelled &) {
    }
  }

private:
  FrameQueue &m_queue;
  std::atomic<bool>& m_abort;
};

} // Namespace

int main(int argc, char *argv[])
{
  float min_depth = 0.29f;
  float max_depth = 16.0f;
  int mode = 0;
  int fps = 30;
  bool flip = false;
  bool laser = false;
  bool force_white_balance = false;
  bool force_auto_exposure = false;
  bool print_options = false;
  bool align = false;
  std::string output_filename = "";
  bool automatic = false;
  bool automatic2 = false;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--min_depth" && i+1 < argc) {
      min_depth = std::atof(argv[i+1]);
    } else if (std::string(argv[i]) == "--max_depth" && i+1 < argc) {
      max_depth = std::atof(argv[i+1]);
    } else if (std::string(argv[i]) == "--mode" && i+1 < argc) {
      mode = std::atoi(argv[i+1]);
    } else if (std::string(argv[i]) == "--fps" && i+1 < argc) {
      fps = std::atoi(argv[i+1]);
    } else if (std::string(argv[i]) == "--flip") {
      flip = true;
    } else if (std::string(argv[i]) == "--laser") {
      laser = true;
    } else if (std::string(argv[i]) == "--force_white_balance") {
      force_white_balance = true;
    } else if (std::string(argv[i]) == "--force_auto_exposure") {
      force_auto_exposure = true;
    } else if (std::string(argv[i]) == "--print_options") {
      print_options = true;
    } else if (std::string(argv[i]) == "--align") {
      align = true;
    } else if (std::string(argv[i]) == "--output" && i+1 < argc) {
      output_filename = std::string(argv[i+1]);
    } else if (std::string(argv[i]) == "--auto") {
        automatic = true;

        max_depth = 64.0f;
        fps = 15;
        flip = true;
        align = true;

        std::vector<cv::String> filenames ;
        cv::utils::fs::glob(".", "record_*.mp4", filenames);
        std::ostringstream ss;
        ss << "record_" << std::setfill('0') << std::setw(3) << filenames.size() << ".mp4";
        output_filename = ss.str();
    } else if (std::string(argv[i]) == "--auto2") {
        automatic2 = true;

        max_depth = 64.0f;
        fps = 15;
        flip = false;
        align = true;

        std::vector<cv::String> filenames ;
        cv::utils::fs::glob(".", "record_*.mp4", filenames);
        std::ostringstream ss;
        ss << "record_" << std::setfill('0') << std::setw(3) << filenames.size() << ".mp4";
        output_filename = ss.str();
    }
    else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      std::cout << "Options are:" << std::endl;
      std::cout << "\t--min_depth <min_depth>" << std::endl;
      std::cout << "\t--max_depth <max_depth>" << std::endl;
      std::cout << "\t--mode <mode (0, 1, 2)>" << std::endl;
      std::cout << "\t--fps <fps (30, 15, 5)>" << std::endl;
      std::cout << "\t--flip" << std::endl;
      std::cout << "\t--laser" << std::endl;
      std::cout << "\t--force_white_balance" << std::endl;
      std::cout << "\t--force_auto_exposure" << std::endl;
      std::cout << "\t--print_options" << std::endl;
      std::cout << "\t--align" << std::endl;
      std::cout << "\t--output_filename <output>" << std::endl;
      std::cout << "\t--auto" << std::endl;
      std::cout << "\t--auto2 (no flip)" << std::endl;

      std::cout << cv::getBuildInformation() << std::endl;

      return 0;
    }
  }

  std::cout << "automatic: " << automatic << std::endl;
  std::cout << "automatic2: " << automatic2 << std::endl;
  std::cout << "min_depth: " << min_depth << std::endl;
  std::cout << "max_depth: " << max_depth << std::endl;
  std::cout << "flip: " << flip << std::endl;
  std::cout << "laser: " << laser << std::endl;
  std::cout << "force white balance: " << force_white_balance << std::endl;
  std::cout << "force auto exposure: " << force_auto_exposure << std::endl;
  std::cout << "print options: " << print_options << std::endl;
  std::cout << "align: " << align << std::endl;
  std::cout << "output filename: " << output_filename << std::endl;
  std::cout << "fps: " << fps << std::endl;
  std::cout << "mode: " << mode << std::endl;

  int width = 848, height = 480;
  if (mode == 0) {
  } else if (mode == 1) {
    width = 640;
    height = 480;
  } else if (mode == 2) {
    width = 424;
    height = 240;
  }
  std::cout << "image size: " << width << "x" << height << std::endl;

  rs2::config cfg;
  cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16, fps);
  cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
  rs2::align align_to_color(RS2_STREAM_COLOR);

  rs2::pipeline pipe;
  auto prof = cfg.resolve(pipe);
  auto advanced_mode = prof.get_device().as<rs400::advanced_mode>();
  if (!advanced_mode.is_enabled()) {
      std::cout << "Advanced mode is not enabled and will be enabled" << std::endl;
      advanced_mode.toggle_advanced_mode(true);
  }
  auto depth_table = advanced_mode.get_depth_table();
  std::cout << "depthClampMin: " << depth_table.depthClampMin << std::endl;
  std::cout << "depthClampMax: " << depth_table.depthClampMax << std::endl;
  std::cout << "disparityMode: " << depth_table.disparityMode << std::endl;
  std::cout << "disparityShift: " << depth_table.disparityShift << std::endl;
//  depth_table.depthUnits = static_cast<unsigned int>(1000 * max_depth / 16);
  depth_table.depthUnits = 1000; // this should allow seeing at max ~65m
  std::cout << "depthUnits: " << depth_table.depthUnits << std::endl;
  advanced_mode.set_depth_table(depth_table);

  rs2::pipeline_profile selection = pipe.start(cfg);
  rs2::device selected_device = selection.get_device();
  auto depth_sensor = selected_device.first<rs2::depth_sensor>();
  auto color_sensor = selected_device.first<rs2::color_sensor>();

  if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
    if (laser) {
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
    } else {
      depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
    }
  }

  if (force_white_balance) {
    change_sensor_option(color_sensor, rs2_option::RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, 1);
  }
  if (force_auto_exposure) {
    change_sensor_option(depth_sensor, rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
    change_sensor_option(color_sensor, rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
  }

  if (force_auto_exposure) {
    std::cout << "\n\n*****DEPTH SENSOR*****" << std::endl;
    get_sensor_option(depth_sensor);
    std::cout << "\n\n*****COLOR SENSOR*****" << std::endl;
    get_sensor_option(color_sensor);
  }


//  rs400::advanced_mode advanced_device(selection.getDevice());
//  auto depth_table = advanced_device.get_depth_table();
//  depth_table.depthClampMax = 1300; // 1m30 if depth unit at 0.001
//  advanced_device.set_depth_table(depth_table);

  float depth_scale = depth_sensor.get_depth_scale();
  std::cout << "depth scale: " << depth_scale << std::endl;

  cv::Mat rgb_color_depth(height*2, width, CV_8UC3);
  cv::Mat depth_colormap = rgb_color_depth(cv::Range(height, 2*height), cv::Range(0, width));
  cv::Mat depth_normalized(height, width, CV_8UC1);
  cv::Mat depth_raw(height, width, CV_16UC1);

  int offset = 25;
  int ruler_width_scale = 20;
  cv::Size ruler_size(width / ruler_width_scale, height - 2*offset);
  cv::Mat depth_ruler_colormap = rgb_color_depth(cv::Range(height + offset, height + offset + ruler_size.height),
                                                 cv::Range(width - 3*offset - ruler_size.width, width - 3*offset));
  cv::Mat disp(cv::Size(30, 256), CV_8UC1);
  for(int y = 0; y < disp.rows; y++) {
      for(int x = 0; x < disp.cols; x++) {
          disp.at<uchar>(y, x) = static_cast<unsigned char>(255-y);
      }
  }
  cv::Mat turbo, turbo_resize;
  cv::applyColorMap(disp, turbo, cv::COLORMAP_TURBO);
  cv::resize(turbo, turbo_resize, ruler_size);
  float scale_y = ruler_size.height / static_cast<float>(disp.rows);
  std::vector<int> colormap_values = {0, 63, 127, 190, 255};

  // Save
  FrameQueue save_queue(128);
  StorageWorker storage(save_queue, output_filename, width, 2*height, fps);
  std::thread storage_thread(&StorageWorker::run, &storage);

  // Display
  std::atomic<bool> abort(false);
  FrameQueue display_queue(1);
  DisplayWorker display(display_queue, abort);
  std::thread display_thread(&DisplayWorker::run, &display);

  while (!abort) {
    rs2::frameset data = pipe.wait_for_frames();      // Wait for next set of frames from the camera
    if (align) {
      data = align_to_color.process(data);
    }
    rs2::frame color_frame = data.get_color_frame();  //Take the color frame from the frameset
    rs2::frame depth_frame = data.get_depth_frame();  //Take the depth frame from the frameset
    if (!depth_frame) { break; }                      // Should not happen but if the pipeline is configured differently

//    rs2::depth_frame df = depth_frame;
//    std::cout << "du: " << df.get_units() << std::endl;

    depth_raw = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depth_frame.get_data()));
    float min_disp = min_depth / depth_scale;
    float max_disp = max_depth / depth_scale;
    float a = 255 / (max_disp - min_disp);
    float b = -min_disp * a;
    for (int i = 0; i < depth_raw.rows; i++) {
      for (int j = 0; j < depth_raw.cols; j++) {
        ushort disp = depth_raw.at<ushort>(i,j);
        // manually threshold
        if (disp < min_disp || disp > max_disp) {
          depth_raw.at<ushort>(i,j) = 0;
          disp = 0;
        }
        if (disp) {
          depth_normalized.at<uchar>(flip ? depth_raw.rows - i : i, flip ? depth_raw.cols - j : j) = static_cast<uchar>(a*disp + b);
        }
      }
    }
    cv::applyColorMap(depth_normalized, depth_colormap, cv::COLORMAP_TURBO);
    for (int i = 0; i < depth_raw.rows; i++) {
      for (int j = 0; j < depth_raw.cols; j++) {
        ushort disp = depth_raw.at<ushort>(i,j);
        if (disp == 0) {
          depth_colormap.at<cv::Vec3b>(flip ? depth_raw.rows - i : i, flip ? depth_raw.cols - j : j) = cv::Vec3b(0,0,0);
        }
      }
    }
    turbo_resize.copyTo(depth_ruler_colormap);
    for (auto val : colormap_values) {
      float dist = depth_scale * (val - b) / a;
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << dist << "m";
      cv::Point orig(width - 3*offset + 5, static_cast<int>(2*height - offset - val*scale_y));
      cv::putText(rgb_color_depth, oss.str(), orig, 0, 0.4, cv::Scalar(0,0,0), 6);
      cv::putText(rgb_color_depth, oss.str(), orig, 0, 0.4, cv::Scalar(255,255,255), 2);
    }

    const int channels = 3;
    if (flip) {
      reverse_copy_rgb(rgb_color_depth.ptr<uchar>(), static_cast<const unsigned char *>(color_frame.get_data()), width, height);
//      reverse_copy_rgb(rgb_color_depth.ptr<uchar>() + sizeof(unsigned char) * width * height * channels,
//                       static_cast<const unsigned char *>(filtered_frame.get_data()), width, height);
    } else {
      memcpy(rgb_color_depth.ptr<uchar>(), color_frame.get_data(), sizeof(unsigned char) * width * height * channels);
//      memcpy(rgb_color_depth.ptr<uchar>() + sizeof(unsigned char) * width * height * channels,
//             filtered_frame.get_data(), sizeof(unsigned char) * width * height * channels);
    }

    display_queue.push(rgb_color_depth);
    if (!output_filename.empty()) {
      save_queue.push(rgb_color_depth);
    }
  }

  save_queue.cancel();
  storage_thread.join();

  display_queue.cancel();
  display_thread.join();

  return 0;
}
