#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include "stair_estimation/plane_segmentation.hpp"
#include "stair_estimation/plane_tracker.hpp" 
#include "stair_estimation/msg/stair_planes.hpp"

class StairEstimatorNode : public rclcpp::Node {
public:
    StairEstimatorNode() : Node("stair_estimator") {
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        plane_segmentation_ = std::make_unique<PlaneSegmentation>(shared_from_this(), true);

        cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/zedxm/zed_node/point_cloud/cloud_registered", 10,
            std::bind(&StairEstimatorNode::cloud_callback, this, std::placeholders::_1)
        );

        plane_pub_w_ = this->create_publisher<stair_estimation::msg::StairPlanes>("/stair_estimation/plane_info_w", 10);
        plane_pub_c_ = this->create_publisher<stair_estimation::msg::StairPlanes>("/stair_estimation/plane_info_c", 10);
        
        RCLCPP_INFO(this->get_logger(), "Stair Estimator Node (ROS2) has been successfully initialized.");
    }

private:
    void cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO_ONCE(this->get_logger(), "First point cloud message received!");

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*msg, *cloud);

        pcl::PointCloud<PointT>::Ptr cloud_map(new pcl::PointCloud<PointT>);
        try {
            if (tf_buffer_->canTransform("map", msg->header.frame_id, tf2::TimePointZero)) {
                pcl_ros::transformPointCloud("map", *cloud, *cloud_map, *tf_buffer_);
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "TF 'map' to '%s' not available yet.", msg->header.frame_id.c_str());
                return;
            }
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR(this->get_logger(), "TF Core Exception: %s", ex.what());
            return;
        }

        // 進行平面分割
        plane_distances_ = plane_segmentation_->segment_planes(cloud_map);

        // 更新與發布全域平面資訊 (_w_) -> 經過 Tracker 濾波與追蹤
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

        // 填寫相對座標系特徵 (_c)：使用完美幾何向量投影，免重複翻轉點雲
        Eigen::Vector3d robot_pos(0.0, 0.0, 0.0);
        try {
            if (tf_buffer_->canTransform("map", "zedxm_base_link", tf2::TimePointZero)) {
                auto tf_msg = tf_buffer_->lookupTransform("map", "zedxm_base_link", tf2::TimePointZero);
                robot_pos.x() = tf_msg.transform.translation.x;
                robot_pos.y() = tf_msg.transform.translation.y;
                robot_pos.z() = tf_msg.transform.translation.z;
            }
        } catch (tf2::TransformException &ex) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Base link TF lookup failed: %s", ex.what());
        }

        // 清空舊數據並利用內積重新計算 (修正你提報的 bug 覆蓋，保留 100% 正確的相對計算)
        plane_msg_c_.vertical.clear();
        double robot_projected_v = robot_pos.dot(plane_distances_.v_normal);
        for (double v_map : plane_distances_.vertical) {
            plane_msg_c_.vertical.push_back(v_map - robot_projected_v);
        }

        plane_msg_c_.horizontal.clear();
        double robot_projected_h = robot_pos.dot(plane_distances_.h_normal);
        for (double h_map : plane_distances_.horizontal) {
            plane_msg_c_.horizontal.push_back(h_map - robot_projected_h);
        }

        // 複製相對座標系法向量 (直接採用第一手原始當前幀)
        plane_msg_c_.v_normal.x = plane_distances_.v_normal.x();
        plane_msg_c_.v_normal.y = plane_distances_.v_normal.y();
        plane_msg_c_.v_normal.z = plane_distances_.v_normal.z();
        
        plane_msg_c_.h_normal.x = plane_distances_.h_normal.x();
        plane_msg_c_.h_normal.y = plane_distances_.h_normal.y();
        plane_msg_c_.h_normal.z = plane_distances_.h_normal.z();

        // 4. 同步發布對應通訊 Topic
        plane_pub_w_->publish(plane_msg_w_);
        plane_pub_c_->publish(plane_msg_c_);
    }

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<PlaneSegmentation> plane_segmentation_;
    PlaneTracker plane_tracker_; 

    PlaneDistances plane_distances_;
    Eigen::Vector3d v_normal_;
    Eigen::Vector3d h_normal_;
    stair_estimation::msg::StairPlanes plane_msg_w_;
    stair_estimation::msg::StairPlanes plane_msg_c_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
    rclcpp::Publisher<stair_estimation::msg::StairPlanes>::SharedPtr plane_pub_w_;
    rclcpp::Publisher<stair_estimation::msg::StairPlanes>::SharedPtr plane_pub_c_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<StairEstimatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}