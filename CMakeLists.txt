cmake_minimum_required(VERSION 3.8)
project(stair_estimation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 尋找依賴元件
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 生成自訂訊息 (StairPlanes.msg)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StairPlanes.msg"
  DEPENDENCIES geometry_msgs
)

# 引入標頭檔路徑
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# 建立幾何運算核心函式庫
add_library(stair_geometry_lib
  src/plane_segmentation.cpp
)

# 確保自訂訊息在 C++ 核心庫之前建置完成
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
add_dependencies(stair_geometry_lib ${cpp_typesupport_target})

target_link_libraries(stair_geometry_lib
  rclcpp::rclcpp
  ${PCL_LIBRARIES}
)

# 建立 ROS2 主要執行節點
add_executable(stair_estimator src/stair_estimator.cpp)
target_link_libraries(stair_estimator
  stair_geometry_lib
  rclcpp::rclcpp
  tf2_ros::tf2_ros
  pcl_conversions::pcl_conversions
  pcl_ros::pcl_ros
)

# 鏈結自訂訊息介面到執行檔
rosidl_target_interfaces(stair_estimator ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 設定安裝路徑
install(TARGETS stair_estimator
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()