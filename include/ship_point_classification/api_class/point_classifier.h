#ifndef __MORIN_SHIP_POINT_CLASSIFIER_H__
#define __MORIN_SHIP_POINT_CLASSIFIER_H__

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/segmentation/extract_clusters.h>
#include <Eigen/SVD>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <unordered_set>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/normal_3d.h>
typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> ApproxPolicy;
using namespace std;
class PointClassifier{
    private:
        ros::NodeHandle nh_;
        ros::CallbackQueue queue_;
        ros::AsyncSpinner spinner_;
        tf2_ros::Buffer buffer_;
        tf2_ros::TransformListener listener_;
        Eigen::Matrix4d lidar1_tf_, lidar2_tf_;
        message_filters::Subscriber<sensor_msgs::PointCloud2> cloud1_sub_, cloud2_sub_;
        message_filters::Synchronizer<ApproxPolicy> sync_;
        size_t horizontal_scans_ = 2048;
        size_t vertical_scans_ = 64;
        double min_altitude_ = -16.6;
        double max_altitude = 16.6;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec_;
        void pointSyncCallback(const sensor_msgs::PointCloud2ConstPtr& point1, const sensor_msgs::PointCloud2ConstPtr& point2);
        double algebraicDist(double x, double y, const Eigen::Matrix3d& conic);
        double geometricDist(double x, double y, const Eigen::Matrix3d& conic);
        int getRow(pcl::PointXYZI pt);
        int getCol(pcl::PointXYZI pt);
        //=========For Debug==========
        ros::Publisher pub_merged_scan_;
        ros::Publisher pub_svd_quadric_;
        tf2_ros::TransformBroadcaster broadcaster_;
        //============================
        pcl::VoxelGrid<pcl::PointXYZI> downsampler_;
        ros::Publisher pub_hull_points_, pub_cabin_points_;
        pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne_;
        pcl::RegionGrowing<pcl::PointXYZI, pcl::Normal> reg_;
    public:
        PointClassifier();

        ~PointClassifier();
};
#endif