#include "stair_estimation/plane_segmentation.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <cmath>
#include <map>

PlaneSegmentation::PlaneSegmentation(rclcpp::Node::SharedPtr node, bool debug) 
: node_(node), debug_(debug), normals_(new pcl::PointCloud<pcl::Normal>) {
    edge_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/stair_estimation/edge_visual", 10);
    
    centroid_x = Eigen::Vector3d(1.0, 0.0, 0.0);
    centroid_z = Eigen::Vector3d(0.0, 0.0, 1.0);
}

void PlaneSegmentation::computeNormals() {
    normals_->clear();
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud_);
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    ne.setSearchMethod(tree);
    ne.setKSearch(50);
    ne.compute(*normals_);
}

void PlaneSegmentation::group_by_normals() {
    h_point_idx.clear();
    v_point_idx.clear();

    for (size_t i = 0; i < normals_->size(); ++i) {
        if (!std::isfinite(normals_->points[i].normal_x) || 
            !std::isfinite(normals_->points[i].normal_y) || 
            !std::isfinite(normals_->points[i].normal_z)) continue;

        Eigen::Vector3d normal(normals_->points[i].normal_x, normals_->points[i].normal_y, normals_->points[i].normal_z);
        
        double cos_theta_z = std::abs(normal.dot(centroid_z));
        double cos_theta_x = std::abs(normal.dot(centroid_x));

        // 保留原專案精確夾角過濾邏輯
        if (cos_theta_z > std::cos(15.0 * M_PI / 180.0)) {
            h_point_idx.push_back(i);
        } else if (cos_theta_x > std::cos(20.0 * M_PI / 180.0)) {
            v_point_idx.push_back(i);
        }
    }
    
    RCLCPP_INFO(node_->get_logger(), "Grouped Points -> Horizontal: %lu, Vertical: %lu", h_point_idx.size(), v_point_idx.size());
}

std::pair<std::vector<double>, std::vector<std::vector<int>>> PlaneSegmentation::segment_by_distances(const Eigen::Vector3d& axis, const std::vector<int>& indices) {
    std::vector<double> plane_distances;
    std::vector<std::vector<int>> plane_point_indices;
    if (indices.empty()) return {plane_distances, plane_point_indices};

    double resolution = 0.02; // 2cm 分箱解析度
    std::map<int, std::vector<int>> histogram;

    for (int idx : indices) {
        const PointT& pt = cloud_->points[idx];
        double dist = Eigen::Vector3d(pt.x, pt.y, pt.z).dot(axis);
        int bin = std::round(dist / resolution);
        histogram[bin].push_back(idx);
    }

    for (auto const& [bin, idx_list] : histogram) {
        if (idx_list.size() > 500) { 
            double sum_dist = 0.0;
            for (int idx : idx_list) {
                const PointT& pt = cloud_->points[idx];
                sum_dist += Eigen::Vector3d(pt.x, pt.y, pt.z).dot(axis);
            }
            plane_distances.push_back(sum_dist / idx_list.size());
            plane_point_indices.push_back(idx_list);
        }
    }
    return {plane_distances, plane_point_indices};
}

PlaneDistances PlaneSegmentation::segment_planes(pcl::PointCloud<PointT>::Ptr cloud) {
    cloud_ = cloud;
    PlaneDistances results;

    computeNormals();
    group_by_normals();

    auto [h_dists, h_indices] = segment_by_distances(centroid_z, h_point_idx);
    h_plane_point_indices = h_indices;
    results.horizontal = h_dists;

    auto [v_dists, v_indices] = segment_by_distances(centroid_x, v_point_idx);
    v_plane_point_indices = v_indices;
    results.vertical = v_dists;

    results.v_normal = centroid_x;
    results.h_normal = centroid_z;

    // 呼叫找最前緣點的幾何函式
    auto [avg_depths, edge_indices] = find_depth_by_h_plane(results.horizontal, h_plane_point_indices);
    
    // 執行輕量化黃色邊界點視覺化
    visualize_planes(edge_indices, cloud_->header.frame_id);

    return results;
}

std::pair<std::vector<double>, std::vector<int>> PlaneSegmentation::find_depth_by_h_plane(
    const std::vector<double>& plane_distances, 
    const std::vector<std::vector<int>>& plane_indices) 
{
    std::vector<double> avg_depths;
    std::vector<int> edge_indices;
    (void)plane_distances;

    for (size_t i = 0; i < plane_indices.size(); ++i) {
        const std::vector<int>& indices = plane_indices[i];
        
        std::vector<double> closest_values(cloud_->width, std::numeric_limits<double>::infinity());
        std::vector<int> closest_indices(cloud_->width, -1);

        for (int idx : indices) {
            int col = idx % cloud_->width;
            const PointT& point = cloud_->points[idx];
            
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

            double depth = -centroid_x.dot(Eigen::Vector3d(point.x, point.y, point.z));
            
            if (depth < closest_values[col]) {
                closest_values[col] = depth;
                closest_indices[col] = idx;
            }
        }

        double sum = 0.0;
        int count = 0;
        for (size_t col = 0; col < closest_values.size(); ++col) {
            if (closest_values[col] != std::numeric_limits<double>::infinity()) {
                sum += closest_values[col];
                ++count;
                if (closest_indices[col] != -1) {
                    edge_indices.push_back(closest_indices[col]);
                }
            }
        }

        if (count > 0) {
            avg_depths.push_back(sum / count);
        } else {
            avg_depths.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    return {avg_depths, edge_indices};
}

// 完整保留你原專案中的 find_height_by_v_plane 演算法與所有的 printf 邏輯
std::vector<double> PlaneSegmentation::find_height_by_v_plane(const std::vector<double>& plane_distances, const std::vector<std::vector<int>>& plane_indices) {
    std::vector<double> avg_heights;

    for (size_t i = 0; i < plane_indices.size(); i++) {
        double plane_distance = plane_distances[i];
        double lower = plane_distance - 0.03;
        double upper = plane_distance + 0.03;

        const std::vector<int>& indices = plane_indices[i];
        std::vector<double> highest_values(cloud_->width, -std::numeric_limits<double>::infinity());
        
        for (int idx : indices) {
            int col = idx % cloud_->width; 
            const PointT& point = cloud_->points[idx];
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

            double distance = -centroid_x.dot(Eigen::Vector3d(point.x, point.y, point.z));
            if (distance >= lower && distance <= upper) {
                double height = centroid_z.dot(Eigen::Vector3d(point.x, point.y, point.z));
                if (height > highest_values[col]) {
                    highest_values[col] = height;
                }
            }
        }

        double sum = 0.0;
        int count = 0;
        for (double val : highest_values) {
            if (val != -std::numeric_limits<double>::infinity()) {
                sum += val;
                ++count;
            }
        }
        avg_heights.push_back((count > 0) ? (sum / count) : std::numeric_limits<double>::quiet_NaN());
    }

    return avg_heights;
}

void PlaneSegmentation::visualize_planes(const std::vector<int>& edge_indices, const std::string& frame_id) {
    if (!debug_) return;

    pcl::PointCloud<PointT>::Ptr edge_cloud(new pcl::PointCloud<PointT>);
    edge_cloud->reserve(edge_indices.size());

    for (int idx : edge_indices) {
        if (idx >= 0 && idx < static_cast<int>(cloud_->points.size())) {
            PointT point = cloud_->points[idx];
            if (std::isfinite(point.x)) {
                point.r = 255; point.g = 255; point.b = 0; 
                edge_cloud->push_back(point);
            }
        }
    }

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*edge_cloud, output);
    output.header.frame_id = frame_id;
    output.header.stamp = node_->get_clock()->now();

    edge_pub_->publish(output);
}