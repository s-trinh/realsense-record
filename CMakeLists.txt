project(realsense-record)

cmake_minimum_required(VERSION 2.6)

find_package(VISP REQUIRED visp_core visp_sensor visp_io visp_gui)

# https://stackoverflow.com/questions/50760024/cmake-configuration-for-ffmpeg-in-c-project
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(SWSCALE_INCLUDE_DIR libswscale/swscale.h)
find_library(SWSCALE_LIBRARY swscale)

# set the list of source files
#set(cpp_src
#  rs_record.cpp
#)

add_executable(rs_record rs_record.cpp)
target_include_directories(rs_record PRIVATE ${VISP_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})
target_link_libraries(rs_record ${VISP_LIBRARIES} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})

add_executable(rs_record_thread rs_record_thread.cpp)
target_include_directories(rs_record_thread PRIVATE ${VISP_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})
target_link_libraries(rs_record_thread ${VISP_LIBRARIES} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})

add_executable(rs_record_omx rs_record_omx.cpp)
target_include_directories(rs_record_omx PRIVATE ${VISP_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${SWSCALE_INCLUDE_DIR})
target_link_libraries(rs_record_omx ${VISP_LIBRARIES} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${SWSCALE_LIBRARY})

#foreach(cpp ${cpp_src})
#  target_include_directories(decode_encode PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
#  visp_add_target(${cpp})
#endforeach()
