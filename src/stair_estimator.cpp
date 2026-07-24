#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <unordered_map>
#include <random>
#include <memory>
#include <iomanip>
#include <vector>
#include <tf2_ros/static_transform_broadcaster.h>

#include "stair_estimation/msg/stair_planes.hpp"
#include "plane_segmentation.hpp"
#include "plane_tracker.hpp"

class StairEstimatorNode : public rclcpp::Node {
private:
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<stair_estimation::msg::StairPlanes>::SharedPtr plane_pub_w_;
    rclcpp::Publisher<stair_estimation::msg::StairPlanes>::SharedPtr plane_pub_c_;

    pcl::PointCloud<PointT>::Ptr cloud;
    std::unique_ptr<PlaneSegmentation> plane_segmentation_;
    PlaneTracker plane_tracker_;
    
    PlaneDistances plane_distances_;    // distances of planes relative to the robot (in robot frame).
    stair_estimation::msg::StairPlanes plane_msg_w_;
    stair_estimation::msg::StairPlanes plane_msg_c_;
    
    Eigen::Vector3d v_normal_, h_normal_;
    std::ofstream stair_csv_;

public:
    StairEstimatorNode() : Node("stair_estimator_node") {
        // 初始化訂閱與發布
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);    
        
        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedxm/zed_node/point_cloud/cloud_registered", 1,
            std::bind(&StairEstimatorNode::cloud_cb, this, std::placeholders::_1));
            
        plane_pub_w_ = this->create_publisher<stair_estimation::msg::StairPlanes>("/stair_planes_world", 1);
        plane_pub_c_ = this->create_publisher<stair_estimation::msg::StairPlanes>("/stair_planes_camera", 1);

        // 初始化 cloud, PlaneSegmentation
        cloud = boost::make_shared<pcl::PointCloud<PointT>>();
        plane_segmentation_ = std::make_unique<PlaneSegmentation>(this);

        // 初始化 CSV 檔案
        stair_csv_.open("stair_planes.csv");
        stair_csv_ << "Time,";
        for (int i = 0; i < 5; i++) stair_csv_ << "Vertical"   << i << ",";
        for (int i = 0; i < 5; i++) stair_csv_ << "Horizontal" << i << ",";
        stair_csv_ << "\n";
    }//end StairEstimatorNode

    ~StairEstimatorNode() {
        if (stair_csv_.is_open()) {
            stair_csv_.close();
        }
    }//end ~StairEstimatorNode

    void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::fromROSMsg(*msg, *cloud);
        if (!cloud->isOrganized()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, 
                "Point cloud is not organized. Skipping frame.");
            return;
        }

        // 進行平面分割
        plane_distances_ = plane_segmentation_->segment_planes(cloud);

        // 更新與發布平面資訊
        plane_tracker_.update(plane_distances_);
        plane_msg_w_.vertical = plane_tracker_.get_vertical_averages();
        plane_msg_w_.horizontal = plane_tracker_.get_horizontal_averages();
        v_normal_ = plane_tracker_.get_vertical_normal();
        h_normal_ = plane_tracker_.get_horizontal_normal();

        plane_msg_w_.v_normal.x = v_normal_.x();
        plane_msg_w_.v_normal.y = v_normal_.y();
        plane_msg_w_.v_normal.z = v_normal_.z();
        plane_msg_w_.h_normal.x = h_normal_.x();
        plane_msg_w_.h_normal.y = h_normal_.y();
        plane_msg_w_.h_normal.z = h_normal_.z();
        plane_pub_w_->publish(plane_msg_w_);

        /* Relative to Camera */
        Eigen::Vector3d camera_pos(0.0, 0.0, 0.0);
        try {
            rclcpp::Time cloud_stamp(static_cast<int64_t>(cloud->header.stamp * 1000ULL));
            geometry_msgs::msg::TransformStamped tf_msg = 
                tf_buffer_->lookupTransform("map", "zedxm_base_link", cloud_stamp);

            // 取得機器人當前在 map 座標系下的 X (前後) 與 Z (高度)
            camera_pos.x() = tf_msg.transform.translation.x;
            camera_pos.y() = tf_msg.transform.translation.y;
            camera_pos.z() = tf_msg.transform.translation.z;
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF Exception: %s", ex.what());
            return;
        }

        Eigen::Vector3d n_v(plane_distances_.v_normal.x(), plane_distances_.v_normal.y(), plane_distances_.v_normal.z());
        double camera_projected_v = camera_pos.dot(n_v); // 機器人位置內積樓梯法向量
        plane_msg_c_.vertical.clear();
        for (double v_map : plane_distances_.vertical) {
            // 公式：世界距離 - 機器人投影長度
            plane_msg_c_.vertical.push_back(v_map - camera_projected_v);
        }
        Eigen::Vector3d n_h(plane_distances_.h_normal.x(), plane_distances_.h_normal.y(), plane_distances_.h_normal.z());
        double camera_projected_h = camera_pos.dot(n_h);
        plane_msg_c_.horizontal.clear();
        for (double h_map : plane_distances_.horizontal) {
            // 樓梯在世界的高度 - 機器人當前的高度 = 樓梯相對於機器人的高度！
            plane_msg_c_.horizontal.push_back(h_map - camera_projected_h);
        }
        plane_msg_c_.v_normal.x = plane_distances_.v_normal.x();
        plane_msg_c_.v_normal.y = plane_distances_.v_normal.y();
        plane_msg_c_.v_normal.z = plane_distances_.v_normal.z();
        plane_msg_c_.h_normal.x = plane_distances_.h_normal.x();
        plane_msg_c_.h_normal.y = plane_distances_.h_normal.y();
        plane_msg_c_.h_normal.z = plane_distances_.h_normal.z();
        plane_pub_c_->publish(plane_msg_c_);

        // 當收到新點雲且處理完畢時，輸出尺寸結果到終端機
        write_csv_record();
        print_stair_dimensions();
    }//end cloud_cb

    void write_csv_record() {
        if (!stair_csv_.is_open()) return;

        stair_csv_ << this->get_clock()->now().seconds() << ",";
        for (int i = 0; i < 5; i++) {
            if (i < static_cast<int>(plane_distances_.vertical.size()))
                stair_csv_ << std::fixed << std::setprecision(4) << plane_distances_.vertical[i];
            else 
                stair_csv_ << "0";
            stair_csv_ << ",";
        }
        for (int i = 0; i < 5; i++) {
            if (i < static_cast<int>(plane_distances_.horizontal.size()))
                stair_csv_ << std::fixed << std::setprecision(4) << plane_distances_.horizontal[i];
            else
                stair_csv_ << "0";
            stair_csv_ << ",";
        }
        stair_csv_ << "\n";
    }//end write_csv_record

    // print stair dimensions in terminal
    void print_stair_dimensions() {
        std::cout << "\033[2J\033[1;1H"; // clean terminal
        std::cout << "\033[1;36m=================================================================\033[0m\n";
        std::cout << "\033[1;33m                      STAIR ESTIMATOR REPORT                     \033[0m\n";
        std::cout << "\033[1;36m=================================================================\033[0m\n";

        // 1. first vertical distance (vertical0)
        if (!plane_distances_.vertical.empty()) {
            std::cout << " \033[1;32mBase Vertical Distance (Vert0):\033[0m " 
                      << std::fixed << std::setprecision(4) << plane_distances_.vertical[0] << " m\n";
        } else {
            std::cout << " \033[1;31mBase Vertical Distance (Vert0):\033[0m N/A (No Planes Detected)\n";
        }
        std::cout << "-----------------------------------------------------------------\n";


        // 2. Stair tread (Depth) and riser (Height)
        std::vector<double> vert_diffs;
        std::vector<double> horiz_diffs;

        if (plane_distances_.vertical.size() > 1) {
            for (size_t i = 0; i < plane_distances_.vertical.size() - 1; ++i) {
                vert_diffs.push_back(std::abs(plane_distances_.vertical[i+1] - plane_distances_.vertical[i]));
            }
        }
        if (plane_distances_.horizontal.size() > 1) {
            for (size_t i = 0; i < plane_distances_.horizontal.size() - 1; ++i) {
                horiz_diffs.push_back(std::abs(plane_distances_.horizontal[i+1] - plane_distances_.horizontal[i]));
            }
        }

        // 決定總共要印幾行（取兩者長度的最大值）
        size_t max_rows = std::max(vert_diffs.size(), horiz_diffs.size());

        // 印出欄位 Title
        std::cout << "   \033[1;32m Vertical Diff (Depth)\033[0m       │     \033[1;34m Horizontal Diff (Height)\033[0m  \n";
        std::cout << "--------------------------------┼--------------------------------\n";

        if (max_rows == 0) {
            std::cout << "    (No sufficient planes)      │      (No sufficient planes)\n";
        } else {
            for (size_t i = 0; i < max_rows; ++i) {
                std::string left_col = "";
                std::string right_col = "";

                // 處理左側 垂直欄 資料
                if (i < vert_diffs.size()) {
                    std::stringstream ss;
                    ss << "  [Vert" << i+1 << " - Vert" << i << "] = " 
                       << std::fixed << std::setprecision(4) << vert_diffs[i] << " m";
                    left_col = ss.str();
                } else {
                    left_col = "  -";
                }

                // 處理右側 水平欄 資料
                if (i < horiz_diffs.size()) {
                    std::stringstream ss;
                    ss << "  [Horiz" << i+1 << " - Horiz" << i << "] = " 
                       << std::fixed << std::setprecision(4) << horiz_diffs[i] << " m";
                    right_col = ss.str();
                } else {
                    right_col = "  -";
                }

                // 這裡寬度設為 42，利用 std::left 靠左對齊，確保中間的「│」分隔線永遠在同一行對齊
                std::cout << std::left << std::setw(32) << left_col 
                          << "│" << right_col << "\n";
            }
        }
        std::cout << "\033[1;36m=================================================================\033[0m\n\n";
    }//end print_stair_dimensions
    
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto estimator = std::make_shared<StairEstimatorNode>();
    
    // 使用單/多執行緒 Executor
    // rclcpp::executors::MultiThreadedExecutor executor;
    // rclcpp::executors::SingleThreadedExecutor executor;

    // executor.add_node(estimator);
    
    rclcpp::Rate rate(10);
    
    /* Loop */
    while (rclcpp::ok()) {
        rclcpp::spin_some(estimator);
        // executor.spin_some();

        rate.sleep();
    }//end while

    // executor.spin();
    rclcpp::shutdown();
    return 0;
}