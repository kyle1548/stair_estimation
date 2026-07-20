#ifndef PLANESEG_HPP
#define PLANESEG_HPP

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/integral_image_normal.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Dense>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#define DEBUG 0
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZ PointT_no_color;

struct PlaneDistances {
    std::vector<double> vertical;
    std::vector<double> horizontal;
    Eigen::Vector3d v_normal;
    Eigen::Vector3d h_normal;
};

class PlaneSegmentation {
    public:
        PlaneSegmentation(rclcpp::Node* node, bool debug=DEBUG);
        ~PlaneSegmentation();

        PlaneDistances segment_planes(pcl::PointCloud<PointT>::Ptr cloud);
        void setInputCloud(pcl::PointCloud<PointT>::Ptr cloud);
        void computeNormals();
        void group_by_normals();
        std::pair<std::vector<double>, std::vector<std::vector<int>>> segment_by_distances(Eigen::Vector3d centroid, const std::vector<int>& indices);
        Eigen::Vector3d computeCentroid(const std::vector<int>& indices);
        std::pair<std::vector<double>, std::vector<int>> find_depth_by_h_plane(const std::vector<double>& plane_distances, const std::vector<std::vector<int>>& plane_indices);

        void visualize_stair_planes();
        void visualize_normal_vectors();
        void visualize_extend_planes(const std::vector<double>& h_plane_distances, const std::vector<double>& v_plane_distances);
        void visualize_normal_in_sphere();
        void visualize_h_plane_edges(const std::vector<int>& edge_indices);

    private:
        rclcpp::Node* node_;
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
        pcl::PointCloud<PointT>::Ptr cloud_;
        pcl::PassThrough<PointT> pass_;
        pcl::PointCloud<pcl::Normal>::Ptr normals_;
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> normal_estimator_;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree_;

        Eigen::Vector3d centroid_z, centroid_x;
        std::vector<int> h_point_idx;   // point index in cloud belongs to horizontal planes
        std::vector<int> v_point_idx;   // point index in cloud belongs to vertical planes
        const double cos_threshold_ = std::cos(pcl::deg2rad(10.0));

        bool debug_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stair_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr normal_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr normal_sphere_pub;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr plane_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub;

        int last_marker_count_ = 0; // used to record last published marker count
        int last_plane_marker_count_h_;
        int last_plane_marker_count_v_;
        
        std::ofstream histogram_csv;
};

#endif // PLANESEG_HPP
