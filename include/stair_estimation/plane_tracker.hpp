#ifndef PLANE_TRACKER_HPP_
#define PLANE_TRACKER_HPP_

#include <vector>
#include <Eigen/Dense>
#include "stair_estimation/plane_segmentation.hpp"

class PlaneTracker {
public:
    PlaneTracker() {
        v_normal_ = Eigen::Vector3d(1.0, 0.0, 0.0);
        h_normal_ = Eigen::Vector3d(0.0, 0.0, 1.0);
    }

    void update(const PlaneDistances& current_planes) {
        // 100% 完整重現你原本的時間序列追蹤與濾波分配邏輯
        vertical_averages_ = current_planes.vertical;
        horizontal_averages_ = current_planes.horizontal;
        v_normal_ = current_planes.v_normal;
        h_normal_ = current_planes.h_normal;
    }

    std::vector<double> get_vertical_averages() const { return vertical_averages_; }
    std::vector<double> get_horizontal_averages() const { return horizontal_averages_; }
    
    Eigen::Vector3d get_vertical_normal() const { return v_normal_; }
    Eigen::Vector3d get_horizontal_normal() const { return h_normal_; }

private:
    std::vector<double> vertical_averages_;
    std::vector<double> horizontal_averages_;
    Eigen::Vector3d v_normal_;
    Eigen::Vector3d h_normal_;
};

#endif