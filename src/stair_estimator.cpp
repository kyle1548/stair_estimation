#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
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
#include <visualization_msgs/MarkerArray.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>
#include <pcl_ros/transforms.h>
#include <unordered_map>
#include <random>
#include <memory>
#include <iomanip>
#include <vector>

#include "corgi_msgs/TriggerStamped.h"
#include "corgi_msgs/StairPlanes.h"
#include "plane_segmentation.hpp"
#include "plane_tracker.hpp"

class StairEstimatorNode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber trigger_sub_;
    ros::Publisher plane_pub_;

    // 使用 std::unique_ptr 自動管理記憶體，避免 Memory Leak
    std::unique_ptr<PlaneSegmentation> plane_segmentation_;
    PlaneTracker plane_tracker_;
    
    PlaneDistances plane_distances_;    // distances of planes relative to the robot (in robot frame).
    corgi_msgs::TriggerStamped trigger_msg_;
    corgi_msgs::StairPlanes plane_msg_;
    
    int cloud_seq_ = 0;
    Eigen::Vector3d v_normal_, h_normal_;
    std::ofstream stair_csv_;

public:
    StairEstimatorNode() {
        // 初始化訂閱與發布
        cloud_sub_   = nh_.subscribe("/zedxm/zed_node/point_cloud/cloud_registered", 1, &StairEstimatorNode::cloud_cb, this);
        trigger_sub_ = nh_.subscribe("trigger", 1, &StairEstimatorNode::trigger_cb, this);
        plane_pub_   = nh_.advertise<corgi_msgs::StairPlanes>("/stair_planes", 1);

        // 初始化 PlaneSegmentation
        plane_segmentation_ = std::make_unique<PlaneSegmentation>();

        // 初始化 CSV 檔案
        stair_csv_.open("stair_planes.csv");
        stair_csv_ << "Time,Trigger,";
        for (int i = 0; i < 5; i++) stair_csv_ << "Vertical"   << i << ",";
        for (int i = 0; i < 5; i++) stair_csv_ << "Horizontal" << i << ",";
        stair_csv_ << "\n";
    }//end StairEstimatorNode

    ~StairEstimatorNode() {
        if (stair_csv_.is_open()) {
            stair_csv_.close();
        }
    }//end ~StairEstimatorNode

    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg) {
        cloud_seq_ = msg->header.seq;
        
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*msg, *cloud);
        if (!cloud->isOrganized()) {
            ROS_WARN_THROTTLE(5, "Point cloud is not organized. Skipping frame.");
            return;
        }

        // 進行平面分割
        plane_distances_ = plane_segmentation_->segment_planes(cloud);

        // 更新與發布平面資訊
        plane_tracker_.update(plane_distances_);
        plane_msg_.horizontal = plane_tracker_.get_horizontal_averages();
        plane_msg_.vertical = plane_tracker_.get_vertical_averages();
        v_normal_ = plane_tracker_.get_vertical_normal();
        h_normal_ = plane_tracker_.get_horizontal_normal();
        
        plane_msg_.v_normal.x = v_normal_.x();
        plane_msg_.v_normal.y = v_normal_.y();
        plane_msg_.v_normal.z = v_normal_.z();
        plane_msg_.h_normal.x = h_normal_.x();
        plane_msg_.h_normal.y = h_normal_.y();
        plane_msg_.h_normal.z = h_normal_.z();
        plane_pub_.publish(plane_msg_);

        // 當收到新點雲且處理完畢時，輸出尺寸結果到終端機
        print_stair_dimensions();
    }//end cloud_cb

    void trigger_cb(const corgi_msgs::TriggerStamped::ConstPtr& msg) {
        trigger_msg_ = *msg;
    }//end trigger_cb

    void write_csv_record() {
        if (!stair_csv_.is_open()) return;

        stair_csv_ << ros::Time::now() << ",";
        stair_csv_ << (int)trigger_msg_.enable << ",";
        for (int i = 0; i < 5; i++) {
            if (i < plane_distances_.vertical.size())
                stair_csv_ << std::fixed << std::setprecision(4) << plane_distances_.vertical[i];
            else 
                stair_csv_ << "0";
            stair_csv_ << ",";
        }
        for (int i = 0; i < 5; i++) {
            if (i < plane_distances_.horizontal.size())
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
            std::cout << "  (No sufficient vertical planes)      │  (No sufficient horizontal planes)\n";
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
    ros::init(argc, argv, "plane_segmentation_node");
    
    // 建立節點物件
    StairEstimatorNode estimator;
    
    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener tfListener(tfBuffer);
    ros::Rate rate(10);

    /* Loop */
    while (ros::ok()) {
        ros::spinOnce();
        
        // 寫入 CSV 紀錄
        estimator.write_csv_record();

        rate.sleep();
    }//end while

    ros::shutdown();
    return 0;
}