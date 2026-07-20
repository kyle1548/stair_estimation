#ifndef PLANE_SEGMENTATION_HPP_
#define PLANE_SEGMENTATION_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <Eigen/Dense>
#include <vector>
#include <utility>

typedef pcl::PointXYZRGB PointT;

struct PlaneDistances {
    std::vector<double> vertical;
    std::vector<double> horizontal;
    Eigen::Vector3d v_normal;
    Eigen::Vector3d h_normal;
};

class PlaneSegmentation {
public:
    PlaneSegmentation(rclcpp::Node::SharedPtr node, bool debug = true);
    
    PlaneDistances segment_planes(pcl::PointCloud<PointT>::Ptr cloud);
    
    std::pair<std::vector<double>, std::vector<int>> find_depth_by_h_plane(
        const std::vector<double>& plane_distances, 
        const std::vector<std::vector<int>>& plane_indices);

    std::vector<double> find_height_by_v_plane(
        const std::vector<double>& plane_distances, 
        const std::vector<std::vector<int>>& plane_indices);

    void visualize_planes(const std::vector<int>& edge_indices, const std::string& frame_id);

private:
    void computeNormals();
    void group_by_normals();
    std::pair<std::vector<double>, std::vector<std::vector<int>>> segment_by_distances(const Eigen::Vector3d& axis, const std::vector<int>& indices);

    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
    pcl::PointCloud<PointT>::Ptr cloud_;
    pcl::PointCloud<pcl::Normal>::Ptr normals_;
    bool debug_;
    
    Eigen::Vector3d centroid_x;
    Eigen::Vector3d centroid_z;

    std::vector<int> h_point_idx;
    std::vector<int> v_point_idx;
    std::vector<std::vector<int>> h_plane_point_indices;
    std::vector<std::vector<int>> v_plane_point_indices;
};

#endif