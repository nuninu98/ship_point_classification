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

        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec_;
        void pointSyncCallback(const sensor_msgs::PointCloud2ConstPtr& point1, const sensor_msgs::PointCloud2ConstPtr& point2);
        double algebraicDist(double x, double y, const Eigen::Matrix3d& conic);
        //=========For Debug==========
        ros::Publisher pub_merged_scan_;
        ros::Publisher pub_svd_quadric_;
        tf2_ros::TransformBroadcaster broadcaster_;
        //============================
    public:
        PointClassifier();

        ~PointClassifier();
};
#endif