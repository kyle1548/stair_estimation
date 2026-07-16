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
#include <tf2_eigen/tf2_eigen.h>
#include <pcl_ros/transforms.h>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <memory> 

#include <Eigen/Dense>
#include "plane_segmentation.hpp"

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointXYZ PointT_no_color;

PlaneSegmentation::~PlaneSegmentation() {
    if (histogram_csv.is_open()) {
        histogram_csv.close();
    }
}

PlaneSegmentation::PlaneSegmentation() :
    normals_(new pcl::PointCloud<pcl::Normal>),
    last_marker_count_(0),
    last_cube_marker_count_h_(0),
    last_cube_marker_count_v_(0)
{
    pass_.setKeepOrganized(true);
    normal_estimator_.setNormalEstimationMethod(normal_estimator_.AVERAGE_3D_GRADIENT);
    normal_estimator_.setMaxDepthChangeFactor(0.01f);
    normal_estimator_.setNormalSmoothingSize(10.0f);

    histogram_csv.open("histogram.csv");
}

void PlaneSegmentation::init_tf() {
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);
    
    pub = nh.advertise<sensor_msgs::PointCloud2>("plane_segmentation", 1);
    normal_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_normals", 1);
    normal_pub2 = nh.advertise<sensor_msgs::PointCloud2>("normal_points", 1);
    plane_pub = nh.advertise<visualization_msgs::MarkerArray>("visualization_plane", 1);
}

PlaneDistances PlaneSegmentation::segment_planes(pcl::PointCloud<PointT>::Ptr cloud) {
    this->setInputCloud(cloud);
    this->computeNormals();
    this->group_by_normals();
    auto [h_plane_distances, h_plane_point_indices] = segment_by_distances(centroid_z, h_point_idx);
    auto [v_plane_distances, v_plane_point_indices] = segment_by_distances(-centroid_x, v_point_idx);

    return {h_plane_distances, v_plane_distances, centroid_z, -centroid_x};
}

void PlaneSegmentation::setInputCloud(pcl::PointCloud<PointT>::Ptr cloud) {
    pass_.setInputCloud(cloud);
    pass_.setFilterFieldName("x");
    pass_.setFilterLimits(0.10, 1.0);
    pass_.filter(*cloud);
    pass_.setFilterFieldName("y");
    pass_.setFilterLimits(-0.4, 0.4);
    pass_.filter(*cloud);
    pass_.setFilterFieldName("z");
    pass_.setFilterLimits(-0.5, 1.0);
    pass_.filter(*cloud);

    try {
        if (tf_buffer_.canTransform("map", cloud->header.frame_id, ros::Time(0), ros::Duration(1.0))) {
            pcl_ros::transformPointCloud("map", *cloud, *cloud, tf_buffer_);
        } else {
            ROS_WARN_THROTTLE(5, "Cannot transform cloud from %s to map. TF not ready.", cloud->header.frame_id.c_str());
        }
    } catch (tf2::TransformException &ex) {
        ROS_ERROR_THROTTLE(5, "TF Exception in setInputCloud: %s", ex.what());
    }

    cloud_ = cloud;
    normal_estimator_.setInputCloud(cloud_);
}

void PlaneSegmentation::computeNormals() {
    normal_estimator_.compute(*normals_);
    for (auto& normal : normals_->points) {
        if (!std::isnan(normal.normal_x) && !std::isnan(normal.normal_y) && !std::isnan(normal.normal_z)) {
            double nx = normal.normal_x;
            double ny = normal.normal_y;
            double nz = normal.normal_z;

            double abs_nx = std::abs(nx);
            double abs_nz = std::abs(nz);

            if (abs_nz >= abs_nx) { 
                if (nz < 0) {   
                    normal.normal_x *= -1;
                    normal.normal_y *= -1;
                    normal.normal_z *= -1;
                }
            } else {    
                if (nx > 0) {   
                    normal.normal_x *= -1;
                    normal.normal_y *= -1;
                    normal.normal_z *= -1;
                }
            }
        }
    }
}

void PlaneSegmentation::group_by_normals() {
    centroid_z = Eigen::Vector3d(0, 0, 1);  
    centroid_x = Eigen::Vector3d(-1, 0, 0); 
    h_point_idx.clear();
    v_point_idx.clear();

    for (size_t i = 0; i < normals_->size(); i++) {
        const auto& n = normals_->points[i];
        if (std::isnan(n.normal_x) || std::isnan(n.normal_y) || std::isnan(n.normal_z)) {
            continue;
        }

        Eigen::Vector3d normal(n.normal_x, n.normal_y, n.normal_z);
        double cos_z = normal.dot(centroid_z);
        double cos_x = normal.dot(centroid_x);

        if (cos_z > cos_threshold_) {   
            h_point_idx.push_back(i);
        } else if (cos_x > cos_threshold_) {    
            v_point_idx.push_back(i);
        }
    }

    centroid_z = !h_point_idx.empty() ? computeCentroid(h_point_idx) : centroid_z;
    centroid_x = !v_point_idx.empty() ? computeCentroid(v_point_idx) : centroid_x;
}

// 🌟 補回來的 computeCentroid 實作
Eigen::Vector3d PlaneSegmentation::computeCentroid(const std::vector<int>& indices) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int idx : indices) {
        const pcl::Normal& n = normals_->points[idx];
        centroid += Eigen::Vector3d(n.normal_x, n.normal_y, n.normal_z);
    }
    if (!indices.empty()) {
        centroid.normalize();
    }
    return centroid;
}

std::pair<std::vector<double>, std::vector<std::vector<int>>> PlaneSegmentation::segment_by_distances(Eigen::Vector3d centroid, const std::vector<int>& indices) {
    const double bin_width = 0.001; 
    const int one_bin_point_threshold = 100;    
    const int total_point_threshold   = 1000;   
    const double merge_threshold = 0.03;    

    std::vector<double> distances;
    std::vector<int> valid_indices;
    distances.reserve(indices.size());
    valid_indices.reserve(indices.size());

    for (int i : indices) {
        const pcl::Normal& n = normals_->points[i];
        if (!std::isnan(n.normal_x) && !std::isnan(n.normal_y) && !std::isnan(n.normal_z)) {
            Eigen::Vector3d position(cloud_->points[i].x,
                                    cloud_->points[i].y,
                                    cloud_->points[i].z);
            double d = centroid.dot(position);  
            distances.push_back(d);
            valid_indices.push_back(i);
        }
    }
    if (distances.empty()) return {};

    auto [min_it, max_it] = std::minmax_element(distances.begin(), distances.end());
    double min_val = *min_it;
    double max_val = *max_it;
    int bin_count = static_cast<int>((max_val - min_val) / bin_width) + 1;
    
    std::vector<int> histogram(bin_count, 0);
    std::vector<double> bin_values(bin_count, 0.0);
    std::vector<std::vector<int>> bin_point_indices(bin_count);
    
    for (size_t i = 0; i < distances.size(); ++i) {
        double d = distances[i];
        int bin = static_cast<int>((d - min_val) / bin_width);
        bin = std::max(0, std::min(bin, bin_count - 1));
        histogram[bin]++;
        bin_values[bin] += d;
        bin_point_indices[bin].push_back(valid_indices[i]);
    }
    
    bool in_range = false;
    std::vector<double> total_counts;
    std::vector<double> mean_distances;
    std::vector<std::vector<int>> candidate_plane_indices;
    double sum_count = 0;
    double sum_value = 0;
    std::vector<int> accumulated_indices;

    for (int i = 0; i < bin_count; ++i) {
        if (histogram[i] >= one_bin_point_threshold) {
            if (!in_range) {
                in_range = true;
                sum_count = histogram[i];
                sum_value = bin_values[i];
                accumulated_indices = bin_point_indices[i];
            } else {
                sum_count += histogram[i];
                sum_value += bin_values[i];
                accumulated_indices.insert(accumulated_indices.end(), bin_point_indices[i].begin(), bin_point_indices[i].end());
            }
        } else if (in_range) {
            if (sum_count >= total_point_threshold) {
                total_counts.push_back(sum_count);
                mean_distances.push_back(sum_value / sum_count);
                candidate_plane_indices.push_back(accumulated_indices);
            }
            in_range = false;
        }
    }
    if (in_range && sum_count >= total_point_threshold) {
        total_counts.push_back(sum_count);
        mean_distances.push_back(sum_value / sum_count);
        candidate_plane_indices.push_back(accumulated_indices);
    }

    if (histogram_csv.is_open()) {
        for (int val : histogram) {
            histogram_csv << val << ",";
        }
        histogram_csv << "\n";
        for (int i = 0; i < bin_count; ++i) {
            histogram_csv << min_val + bin_width * i << ",";
        }
        histogram_csv << "\n";
    }
    
    std::vector<bool> keep(mean_distances.size(), true);
    for (size_t i = 0; i < mean_distances.size(); i++) {
        for (size_t j = i + 1; j < mean_distances.size(); j++) {
            double dist = std::abs(mean_distances[i] - mean_distances[j]);
            if (dist < merge_threshold) {
                if (total_counts[i] >= total_counts[j]) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                }
            }
        }
    }
    
    std::vector<double> final_distances;
    std::vector<std::vector<int>> final_indices;
    for (size_t i = 0; i < mean_distances.size(); i++) {
        if (keep[i]) {
            final_distances.push_back(mean_distances[i]);
            final_indices.push_back(candidate_plane_indices[i]);
        }
    }

    return {final_distances, final_indices};
}

void PlaneSegmentation::visualize_normal_in_space() {
    pcl::PointCloud<PointT>::Ptr cloud_msg(new pcl::PointCloud<PointT>);
    cloud_msg->header.frame_id = "map";
    cloud_msg->height = 1;
    cloud_msg->is_dense = false;

    std::vector<bool> is_horizontal(normals_->size(), false);
    std::vector<bool> is_vertical(normals_->size(), false);

    for (int idx : h_point_idx) {
        if (idx >= 0 && idx < is_horizontal.size()) is_horizontal[idx] = true;
    }
    for (int idx : v_point_idx) {
        if (idx >= 0 && idx < is_vertical.size()) is_vertical[idx] = true;
    }

    cloud_msg->points.reserve(normals_->size());
    for (size_t i = 0; i < normals_->size(); ++i) {
        const auto& n = normals_->points[i];
        if (!pcl::isFinite(n)) continue;

        PointT pt;
        pt.x = n.normal_x;
        pt.y = n.normal_y;
        pt.z = n.normal_z;

        if (is_horizontal[i]) {
            pt.r = 0; pt.g = 0; pt.b = 255;  
        } else if (is_vertical[i]) {
            pt.r = 255; pt.g = 0; pt.b = 0;  
        } else {
            pt.r = 128; pt.g = 128; pt.b = 128;  
        }
        cloud_msg->points.push_back(pt);
    }

    cloud_msg->width = cloud_msg->points.size();
    pcl_conversions::toPCL(ros::Time::now(), cloud_msg->header.stamp);

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_msg, output);
    normal_pub2.publish(output);
}

void PlaneSegmentation::visualize_CubePlanes(const std::vector<double>& h_plane_distances, const std::vector<double>& v_plane_distances) {
    visualization_msgs::MarkerArray marker_array;
    Eigen::Vector3d n_z = centroid_z.normalized();
    Eigen::Vector3d n_x = -centroid_x.normalized();
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::Quaterniond q_z = Eigen::Quaterniond::FromTwoVectors(z_axis, n_z);
    Eigen::Quaterniond q_x = Eigen::Quaterniond::FromTwoVectors(z_axis, n_x);
    
    std_msgs::ColorRGBA blue, red;
    blue.b = 1.0; blue.a = 0.3;
    red.r = 1.0; red.a = 0.3;

    int id = 0;
    for (size_t i = 0; i < h_plane_distances.size(); ++i) {
        double d = h_plane_distances[i];
        Eigen::Vector3d center = d * n_z;

        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.id = id++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.x = q_z.x();
        marker.pose.orientation.y = q_z.y();
        marker.pose.orientation.z = q_z.z();
        marker.pose.orientation.w = q_z.w();

        marker.scale.x = 1.0;  
        marker.scale.y = 1.0;  
        marker.scale.z = 0.001; 
        marker.color = blue;

        marker_array.markers.push_back(marker);
    }
    
    int current_h_count = h_plane_distances.size();
    for (int i = current_h_count; i < last_cube_marker_count_h_; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.id = i; 
        marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(marker);
    }
    last_cube_marker_count_h_ = current_h_count;

    int v_start_id = 1000; 
    for (size_t i = 0; i < v_plane_distances.size(); ++i) {
        double d = v_plane_distances[i];
        Eigen::Vector3d center = d * n_x;

        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = ros::Time::now();
        marker.id = v_start_id + i;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;

        marker.pose.position.x = center.x();
        marker.pose.position.y = center.y();
        marker.pose.position.z = center.z();
        marker.pose.orientation.x = q_x.x();
        marker.pose.orientation.y = q_x.y();
        marker.pose.orientation.z = q_x.z();
        marker.pose.orientation.w = q_x.w();

        marker.scale.x = 1.0;  
        marker.scale.y = 1.0;  
        marker.scale.z = 0.001; 
        marker.color = red;

        marker_array.markers.push_back(marker);
    }
    
    int current_v_count = v_plane_distances.size();
    for (int i = current_v_count; i < last_cube_marker_count_v_; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.id = v_start_id + i;
        marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(marker);
    }
    last_cube_marker_count_v_ = current_v_count;

    plane_pub.publish(marker_array);
}