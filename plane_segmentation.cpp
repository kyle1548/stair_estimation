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


PlaneSegmentation::PlaneSegmentation(bool debug) :
    normals_(new pcl::PointCloud<pcl::Normal>),
    last_marker_count_(0),
    last_plane_marker_count_h_(0),
    last_plane_marker_count_v_(0),
    debug_(debug)
{
    pass_.setKeepOrganized(true);
    normal_estimator_.setNormalEstimationMethod(normal_estimator_.AVERAGE_3D_GRADIENT);
    // normal_estimator_.setNormalEstimationMethod(normal_estimator_.AVERAGE_DEPTH_CHANGE);
    // normal_estimator_.setNormalEstimationMethod(normal_estimator_.COVARIANCE_MATRIX);
    normal_estimator_.setMaxDepthChangeFactor(0.01f);
    normal_estimator_.setNormalSmoothingSize(10.0f);

    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(tf_buffer_);    
    if (debug_) {
        stair_pub = nh.advertise<sensor_msgs::PointCloud2>("visual_stair_points", 1);
        normal_pub = nh.advertise<visualization_msgs::MarkerArray>("visual_normal_vectors", 1);
        normal_sphere_pub = nh.advertise<sensor_msgs::PointCloud2>("visual_normal_sphere", 1);
        plane_pub = nh.advertise<visualization_msgs::MarkerArray>("visual_extended_planes", 1);
        edge_pub = nh.advertise<sensor_msgs::PointCloud2>("visual_stair_edges", 1);

        histogram_csv.open("histogram.csv");
    }//end if
}//end PlaneSegmentation

PlaneSegmentation::~PlaneSegmentation() {
    if (histogram_csv.is_open()) {
        histogram_csv.close();
    }
}//end ~PlaneSegmentation

PlaneDistances PlaneSegmentation::segment_planes(pcl::PointCloud<PointT>::Ptr cloud) {
    this->setInputCloud(cloud);
    this->computeNormals();
    this->group_by_normals();
    auto [v_plane_distances, v_plane_point_indices] = segment_by_distances(-centroid_x, v_point_idx);
    auto [h_plane_distances, h_plane_point_indices] = segment_by_distances(centroid_z, h_point_idx);
    
    // auto [avg_depths, edge_indices] = find_depth_by_h_plane(h_plane_distances, h_plane_point_indices);
    if (debug_) {
        visualize_stair_planes();
        visualize_normal_vectors();
        visualize_extend_planes(h_plane_distances, v_plane_distances);
        visualize_normal_in_sphere();
        // visualize_h_plane_edges(edge_indices);

    }//end if 
    
    return {v_plane_distances, h_plane_distances, -centroid_x, centroid_z};
}//end segment_planes

void PlaneSegmentation::setInputCloud(pcl::PointCloud<PointT>::Ptr cloud) {
    /* Apply ROI filtering */
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

    /* Transform from camera coord to world coord */
    try {
        if (tf_buffer_.canTransform("map", cloud->header.frame_id, ros::Time(0), ros::Duration(0.0))) {
            pcl_ros::transformPointCloud("map", *cloud, *cloud, tf_buffer_);
        } else {
            ROS_WARN_THROTTLE(1, "Cannot transform cloud from %s to map. TF not ready.", cloud->header.frame_id.c_str());
        }
    } catch (tf2::TransformException &ex) {
        ROS_ERROR_THROTTLE(1, "TF Exception in setInputCloud: %s", ex.what());
    }

    /* Set cloud */
    cloud_ = cloud;
    normal_estimator_.setInputCloud(cloud_);
}//end setInputCloud

void PlaneSegmentation::computeNormals() {
    normal_estimator_.compute(*normals_);
    for (auto& normal : normals_->points) {
        if (!std::isnan(normal.normal_x) && !std::isnan(normal.normal_y) && !std::isnan(normal.normal_z)) {
            double nx = normal.normal_x;
            double ny = normal.normal_y;
            double nz = normal.normal_z;

            double abs_nx = std::abs(nx);
            // double abs_ny = std::abs(ny);
            double abs_nz = std::abs(nz);

            if (abs_nz >= abs_nx) { // z為主方向
                if (nz < 0) {   // 翻轉為 +z
                    normal.normal_x *= -1;
                    normal.normal_y *= -1;
                    normal.normal_z *= -1;
                }//end if
            } else {    // x為主方向
                if (nx > 0) {   // 翻轉為 -x
                    normal.normal_x *= -1;
                    normal.normal_y *= -1;
                    normal.normal_z *= -1;
                }//end if
            }//end if else
        }//end if
    }//end for
}//end computeNormals

void PlaneSegmentation::group_by_normals() {
    centroid_z = Eigen::Vector3d(0, 0, 1);  // initial normal vector for horizontal plane
    centroid_x = Eigen::Vector3d(-1, 0, 0); // initial normal vector for vertical   plane
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

        if (cos_z > cos_threshold_) {   // belongs to horizontal planes
            h_point_idx.push_back(i);
        } else if (cos_x > cos_threshold_) {    // belongs to vertical planes
            v_point_idx.push_back(i);
        }//end if else
    }//end for

    // Update centroid
    centroid_x = !v_point_idx.empty() ? computeCentroid(v_point_idx) : centroid_x;
    centroid_z = !h_point_idx.empty() ? computeCentroid(h_point_idx) : centroid_z;
}//end group_by_normals

Eigen::Vector3d PlaneSegmentation::computeCentroid(const std::vector<int>& indices) {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int idx : indices) {
        const pcl::Normal& n = normals_->points[idx];
        centroid += Eigen::Vector3d(n.normal_x, n.normal_y, n.normal_z);
    }//end for
    if (!indices.empty()) {
        centroid.normalize();
    }//end if 
    return centroid;
}//end computeCentroid

std::pair<std::vector<double>, std::vector<std::vector<int>>> PlaneSegmentation::segment_by_distances(Eigen::Vector3d centroid, const std::vector<int>& indices) {
    const double bin_width = 0.001; // 1mm
    const int one_bin_point_threshold = 100;    // 100 points
    const int total_point_threshold   = 1000;   // 1000 points
    const double merge_threshold = 0.03;    // 3cm

    /* Calculate distances */
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
            double d = centroid.dot(position);  // projective distance
            distances.push_back(d);
            valid_indices.push_back(i);
        }//end if
    }//end for
    if (distances.empty()) return {};

    /* Create histogram by distances */
    // Calculate range of histogram
    auto [min_it, max_it] = std::minmax_element(distances.begin(), distances.end());
    double min_val = *min_it;
    double max_val = *max_it;
    int bin_count = static_cast<int>((max_val - min_val) / bin_width) + 1;
    std::vector<int> histogram(bin_count, 0);
    std::vector<double> bin_values(bin_count, 0.0);
    std::vector<std::vector<int>> bin_point_indices(bin_count);
    // Histogram 
    for (size_t i = 0; i < distances.size(); ++i) {
        double d = distances[i];
        int bin = static_cast<int>((d - min_val) / bin_width);
        bin = std::max(0, std::min(bin, bin_count - 1));
        histogram[bin]++;
        bin_values[bin] += d;
        bin_point_indices[bin].push_back(valid_indices[i]);
    }//end for
    
    /* 找出連續高密度 bin */
    bool in_range = false;
    std::vector<double> total_counts;
    std::vector<double> mean_distances;
    std::vector<std::vector<int>> candidate_plane_indices;
    double sum_count = 0;
    double sum_value = 0;
    std::vector<int> accumulated_indices;

    for (int i = 0; i < bin_count; ++i) {   // from smallest distance
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
            }//end if else
        } else if (in_range) {
            if (sum_count >= total_point_threshold) {
                total_counts.push_back(sum_count);
                mean_distances.push_back(sum_value / sum_count);
                candidate_plane_indices.push_back(accumulated_indices);
            }//end if
            in_range = false;
        }//end if else
    }//end for
    if (in_range && sum_count >= total_point_threshold) {
        total_counts.push_back(sum_count);
        mean_distances.push_back(sum_value / sum_count);
        candidate_plane_indices.push_back(accumulated_indices);
    }//end if

    /* Write to histogram csv */
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
    
    /* Only keep max bin in a range (merge_threshold) */
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

    return {final_distances, final_indices}; // from smallest to largest
}//end segment_by_distances

std::pair<std::vector<double>, std::vector<int>> PlaneSegmentation::find_depth_by_h_plane(const std::vector<double>& plane_distances, const std::vector<std::vector<int>>& plane_indices) {
    std::vector<double> avg_depths;
    std::vector<int> edge_point_indices;
    
    // 遍歷每一個被分割出來的水平面
    for (size_t i = 0; i < plane_indices.size(); ++i) {
        const std::vector<int>& indices = plane_indices[i];
        
        // 1. 建立一個陣列，大小等於點雲的寬度 (organized columns)
        // 用來記錄「每一直行中，最靠近機器人的深度值 (最前緣)」
        // 因為我們要找「最前緣（離機器人最近）」，所以初始值設為正無限大
        std::vector<double> closest_values(cloud_->width, INFINITY);
        std::vector<int> closest_indices(cloud_->width, -1);

        // 2. 在這個水平面中，尋找每一直行（col）的最前緣點
        for (int idx : indices) {
            int col = idx % cloud_->width; // 取得在 organized 點雲中的 column 索引
            const PointT& point = cloud_->points[idx];
            
            // 如果點無效，直接跳過
            if (!pcl::isFinite(point)) continue;

            // 計算該點在世界座標下（或對齊後的）正前方深度 X
            // 這裡直接使用 -centroid_x 投影，求得從世界原點往前的投影深度
            double depth = -centroid_x.dot(Eigen::Vector3d(point.x, point.y, point.z));
            
            // 🌟 核心幾何邏輯：找「最前緣」
            // 在這一直行中，如果這個點的深度更小（也就是更靠近機器人），就更新它
            if (depth < closest_values[col]) {
                closest_values[col] = depth;
                closest_indices[col] = idx; // 記錄這個前緣點的索引
            }
        }

        // 3. 計算該水平面「所有有效最前緣點」的平均深度
        double sum = 0.0;
        int count = 0;
        for (size_t col = 0; col < closest_values.size(); ++col) {
            if (closest_values[col] != INFINITY) {
                sum += closest_values[col];
                ++count;
                
                if (closest_indices[col] != -1) {
                    edge_point_indices.push_back(closest_indices[col]);
                }
            }
        }

        // 如果這層有找到有效點，存入平均深度；否則存入 NaN
        if (count > 0) {
            avg_depths.push_back(sum / count);
        } else {
            avg_depths.push_back(std::numeric_limits<double>::quiet_NaN());
        }
    }

    return {avg_depths, edge_point_indices};
}//end find_depth_by_h_plane

void PlaneSegmentation::visualize_stair_planes() {
    pcl::PointCloud<PointT>::Ptr colored_cloud(new pcl::PointCloud<PointT>(*cloud_));
    
    for (size_t i = 0; i < cloud_->size(); ++i) {
        // 沒有有效 normal 的點，塗成黑色
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;
    }
    for (int i : h_point_idx) {
        colored_cloud->points[i].r = 0;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 255; // 水平面塗成藍色
    }
    for (int i : v_point_idx) {
        colored_cloud->points[i].r = 255;
        colored_cloud->points[i].g = 0;
        colored_cloud->points[i].b = 0;   // 垂直面塗成紅色
    }

    /* Publish the result */
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*colored_cloud, output);
    output.header.frame_id = "map";
    stair_pub.publish(output);
}//end visualize_stair_planes

void PlaneSegmentation::visualize_normal_vectors() {
    // 可視化 Marker
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker marker_template;
    marker_template.header.frame_id = "map";
    marker_template.type = visualization_msgs::Marker::ARROW;
    marker_template.action = visualization_msgs::Marker::ADD;
    marker_template.scale.x = 0.005;
    marker_template.scale.y = 0.010;
    marker_template.scale.z = 0.010;
    marker_template.color.r = 0.0;
    marker_template.color.g = 1.0;
    marker_template.color.b = 0.0;
    marker_template.color.a = 1.0;
    marker_template.pose.orientation.x = 0.0;
    marker_template.pose.orientation.y = 0.0;
    marker_template.pose.orientation.z = 0.0;
    marker_template.pose.orientation.w = 1.0;  // 這是必要的！不能為 0

    // 空間分格子平均，降低 Rviz 的 Marker 數量以提升渲染效能
    float grid_size = 0.05;
    std::unordered_map<std::tuple<int, int, int>, pcl::PointNormal, boost::hash<std::tuple<int, int, int>>> grid_map;
    for (size_t i = 0; i < cloud_->size(); ++i) {
        const auto& pt = cloud_->points[i];
        const auto& nm = normals_->points[i];
        if (!pcl::isFinite(pt) || !pcl::isFinite(nm))
            continue;
        int gx = static_cast<int>(std::floor(pt.x / grid_size));
        int gy = static_cast<int>(std::floor(pt.y / grid_size));
        int gz = static_cast<int>(std::floor(pt.z / grid_size));
        auto key = std::make_tuple(gx, gy, gz);
        if (grid_map.find(key) == grid_map.end()) {
            pcl::PointNormal ptn;
            ptn.x = pt.x; ptn.y = pt.y; ptn.z = pt.z;
            ptn.normal_x = nm.normal_x; ptn.normal_y = nm.normal_y; ptn.normal_z = nm.normal_z;
            grid_map[key] = ptn;
        }
    }

    int id = 0;
    /* Point normal (繪製格點法向量) */
    for (const auto& kv : grid_map) {
        const auto& pt = kv.second;

        visualization_msgs::Marker arrow = marker_template;
        arrow.id = id++;

        geometry_msgs::Point start, end;
        start.x = pt.x;
        start.y = pt.y;
        start.z = pt.z;
        end.x = pt.x + 0.03 * pt.normal_x;
        end.y = pt.y + 0.03 * pt.normal_y;
        end.z = pt.z + 0.03 * pt.normal_z;
        arrow.points.push_back(start);
        arrow.points.push_back(end);

        marker_array.markers.push_back(arrow);
    }

    /* Plane normal (繪製整體平面的平均法向量) */
    marker_template.scale.x = 0.025;
    marker_template.scale.y = 0.050;
    marker_template.scale.z = 0.050;
    // 水平面法向量：centroid_z
    if (h_point_idx.size() > 0) {
        visualization_msgs::Marker arrow = marker_template;
        arrow.id = id++;
        geometry_msgs::Point start, end;
        start.x = 0.0;
        start.y = 0.0;
        start.z = 0.0;
        end.x = 0.15 * centroid_z.x();
        end.y = 0.15 * centroid_z.y();
        end.z = 0.15 * centroid_z.z();
        arrow.points.push_back(start);
        arrow.points.push_back(end);

        // 用藍色表示水平面 normal
        arrow.color.r = 0.0;
        arrow.color.g = 0.0;
        arrow.color.b = 1.0;

        marker_array.markers.push_back(arrow);
    }
    // 垂直面法向量：centroid_x
    if (v_point_idx.size() > 0) {
        visualization_msgs::Marker arrow = marker_template;
        arrow.id = id++;
        geometry_msgs::Point start, end;
        start.x = 0.0;
        start.y = 0.0;
        start.z = 0.0;
        end.x = 0.15 * centroid_x.x();
        end.y = 0.15 * centroid_x.y();
        end.z = 0.15 * centroid_x.z();
        arrow.points.push_back(start);
        arrow.points.push_back(end);

        // 用紅色表示垂直面 normal
        arrow.color.r = 1.0;
        arrow.color.g = 0.0;
        arrow.color.b = 0.0;

        marker_array.markers.push_back(arrow);
    }

    // 刪除多餘的舊 marker
    visualization_msgs::Marker delete_marker;
    delete_marker.action = visualization_msgs::Marker::DELETE;
    for (int i = id; i < last_marker_count_; ++i) {
        delete_marker.id = i;
        marker_array.markers.push_back(delete_marker);
    }
    last_marker_count_ = id;  // 記住這次用了幾個 marker

    normal_pub.publish(marker_array);
}//end visualize_normal_vectors

void PlaneSegmentation::visualize_extend_planes(const std::vector<double>& h_plane_distances, const std::vector<double>& v_plane_distances) {
    visualization_msgs::MarkerArray marker_array;
    Eigen::Vector3d n_z = centroid_z.normalized();
    Eigen::Vector3d n_x = -centroid_x.normalized();
    Eigen::Vector3d z_axis(0, 0, 1);
    Eigen::Quaterniond q_z = Eigen::Quaterniond::FromTwoVectors(z_axis, n_z);
    Eigen::Quaterniond q_x = Eigen::Quaterniond::FromTwoVectors(z_axis, n_x);
    // Color setting
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

        marker.scale.x = 1.0;  // width
        marker.scale.y = 1.0;  // height
        marker.scale.z = 0.001; // thickness
        marker.color = blue;

        marker_array.markers.push_back(marker);
    }
    
    int current_h_count = h_plane_distances.size();
    for (int i = current_h_count; i < last_plane_marker_count_h_; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.id = i; 
        marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(marker);
    }
    last_plane_marker_count_h_ = current_h_count;

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
    for (int i = current_v_count; i < last_plane_marker_count_v_; ++i) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "map";
        marker.id = v_start_id + i;
        marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(marker);
    }
    last_plane_marker_count_v_ = current_v_count;

    plane_pub.publish(marker_array);
}//end visualize_extend_planes

void PlaneSegmentation::visualize_normal_in_sphere() {
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
    normal_sphere_pub.publish(output);
}//end visualize_normal_in_sphere

// void PlaneSegmentation::visualize_h_plane_edges(const std::vector<int>& edge_indices) {
//     pcl::PointCloud<PointT>::Ptr edge_cloud(new pcl::PointCloud<PointT>);
//     edge_cloud->reserve(edge_indices.size()); // 預先分配記憶體，提升效率

//     // 2. 🌟 只複製被選為 edge 的點，並將它們塗成黃色 (R=255, G=255, B=0)
//     for (int idx : edge_indices) {
//         PointT point = cloud_->points[idx];
        
//         // 確保點是有效的
//         if (pcl::isFinite(point)) {
//             point.r = 255;
//             point.g = 255;
//             point.b = 0;
//             edge_cloud->points.push_back(point);
//         }
//     }

//     // 3. 填充 ROS 點雲的 header 資訊
//     edge_cloud->header = cloud_->header; // 繼承原始點雲的 timestamp 和 frame_id
//     edge_cloud->width = edge_cloud->points.size();
//     edge_cloud->height = 1;
//     edge_cloud->is_dense = false;
//     pcl_conversions::toPCL(ros::Time::now(), edge_cloud->header.stamp);

//     /* Publish the result */
//     sensor_msgs::PointCloud2 output;
//     pcl::toROSMsg(*edge_cloud, output);
//     // 這裡的 frame_id 會自動繼承原本點雲的 (例如 map 或 zedxm_left_camera_frame)
//     edge_pub.publish(output);
// }//end visualize_h_plane_edges

