cmake_minimum_required(VERSION 3.8)
project(stair_estimation)

# 消除 PCL_ROOT 政策警告
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# 1. 尋找依賴元件
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# 2. 生成自訂訊息介面 (StairPlanes.msg)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StairPlanes.msg"
  DEPENDENCIES geometry_msgs
)

# 3. 引入標頭檔路徑
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${tf2_eigen_INCLUDE_DIRS}
  ${pcl_conversions_INCLUDE_DIRS}
  ${pcl_ros_INCLUDE_DIRS}
  ${visualization_msgs_INCLUDE_DIRS} # 補上視覺化標記標頭檔路徑
)

# 4. 建立核心幾何運算函式庫 (plane_segmentation)
add_library(plane_segmentation SHARED
  src/plane_segmentation.cpp
)

# 使用 Plain 簽名鏈結環境與第三方函式庫 (🌟 補上 visualization_msgs 的鏈結變數)
target_link_libraries(plane_segmentation
  rclcpp::rclcpp
  ${PCL_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
)

# 🌟 僅對你專案內部自訂的訊息呼叫此巨集
rosidl_target_interfaces(plane_segmentation ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 5. 建立 ROS2 主要執行節點
add_executable(stair_estimator
  src/stair_estimator.cpp
)

# 將變數與核心庫鏈結至主要執行檔 (🌟 同步補上 visualization_msgs 鏈結變數)
target_link_libraries(stair_estimator
  plane_segmentation
  rclcpp::rclcpp
  tf2_ros::tf2_ros
  ${PCL_LIBRARIES}
  ${tf2_eigen_LIBRARIES}
  ${pcl_conversions_LIBRARIES}
  ${pcl_ros_LIBRARIES}
  ${visualization_msgs_LIBRARIES}
)

# 🌟 僅對你專案內部自訂的訊息呼叫此巨集
rosidl_target_interfaces(stair_estimator ${PROJECT_NAME} "rosidl_typesupport_cpp")

# 6. 設定安裝路徑
install(TARGETS plane_segmentation stair_estimator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()