#include <ship_point_classification/api_class/point_classifier.h>

PointClassifier::PointClassifier(): buffer_(), listener_(buffer_), queue_(), spinner_(0, &queue_), cloud1_sub_(nh_, "/ouster1/points", 1), 
cloud2_sub_(nh_, "/ouster2/points", 1), sync_(ApproxPolicy(10), cloud1_sub_, cloud2_sub_){
    nh_.setCallbackQueue(&queue_);
    sync_.registerCallback(boost::bind(&PointClassifier::pointSyncCallback, this, _1, _2));

    lidar1_tf_ = Eigen::Matrix4d::Identity();
    lidar1_tf_(0, 3) = (3.325/ 2.0);
    lidar1_tf_(1, 3) = -(1.415/ 2.0);
    lidar1_tf_(0, 0) = cos(-M_PI/2);
    lidar1_tf_(0, 1) = -sin(-M_PI/2);
    lidar1_tf_(1, 0) = sin(-M_PI/2);
    lidar1_tf_(1, 1) = cos(-M_PI/2);

    lidar2_tf_ = Eigen::Matrix4d::Identity();
    lidar2_tf_(0, 3) = -(3.325/ 2.0);
    lidar2_tf_(1, 3) = (1.415/ 2.0);
    lidar2_tf_(0, 0) = cos(M_PI/2);
    lidar2_tf_(0, 1) = -sin(M_PI/2);
    lidar2_tf_(1, 0) = sin(M_PI/2);
    lidar2_tf_(1, 1) = cos(M_PI/2);
    ec_.setClusterTolerance(1.0);
    ec_.setMinClusterSize(5);
    pub_merged_scan_ = nh_.advertise<sensor_msgs::PointCloud2>("ouster_merged", 1);
    pub_svd_quadric_ = nh_.advertise<visualization_msgs::Marker>("ship_quadric", 1);
    spinner_.start();
}

PointClassifier::~PointClassifier(){
    spinner_.stop();
}

void PointClassifier::pointSyncCallback(const sensor_msgs::PointCloud2ConstPtr& point1, const sensor_msgs::PointCloud2ConstPtr& point2){
    pcl::PointCloud<pcl::PointXYZI> cloud1, cloud2;
    pcl::fromROSMsg(*point1, cloud1);
    pcl::fromROSMsg(*point2, cloud2);

    pcl::PointCloud<pcl::PointXYZI> cloud1_tf, cloud2_tf;
    pcl::transformPointCloud(cloud1, cloud1_tf, lidar1_tf_);
    pcl::transformPointCloud(cloud2, cloud2_tf, lidar2_tf_);

    pcl::PointCloud<pcl::PointXYZI> cloud_sum = cloud1_tf + cloud2_tf;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_range_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    for(const auto& pt : cloud_sum){
        double range = sqrt(pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2));
        if(range > 7.0){
            cloud_range_filtered->push_back(pt);
        }
    }
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(cloud_range_filtered);
    ec_.setSearchMethod(tree);
    ec_.setInputCloud(cloud_range_filtered);
    vector<pcl::PointIndices> cluster_indices;
    ec_.extract(cluster_indices);

    if(cluster_indices.empty()){
        return;
    }
    sort(cluster_indices.begin(), cluster_indices.end(), [](const pcl::PointIndices& p1, const pcl::PointIndices& p2){
        return p1.indices.size() > p2.indices.size();
    });

    pcl::PointCloud<pcl::PointXYZI> ship_cloud;
    for(const auto& id: cluster_indices[0].indices){
        ship_cloud.push_back((*cloud_range_filtered)[id]);
    }
    
    double z_cut = 0.0;
    for(size_t i = 0; i < ship_cloud.size(); i++){
        z_cut += ship_cloud[i].z / ship_cloud.size();
    }

    pcl::PointCloud<pcl::PointXYZI> ship_reflection_cropped;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ship_flatten(new pcl::PointCloud<pcl::PointXYZI>);
    for(const auto& pt : ship_cloud){
        auto point = pt;
        point.z = 0.0;
        if(pt.z > z_cut){
            ship_flatten->push_back(point);
            ship_reflection_cropped.push_back(pt);
        }
    }

    Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
    for(const auto& pt : *ship_flatten){
        Eigen::Vector2d v(pt.x, pt.y);
        v /= ship_flatten->size();
        centroid += v;
    }

    Eigen::MatrixXd disp_mat(2, ship_flatten->size());
    for(size_t i = 0; i < ship_flatten->size(); i++){
        auto pt = (*ship_flatten)[i];
        Eigen::Vector2d v(pt.x, pt.y);
        Eigen::Vector2d error = v - centroid;
        disp_mat.col(i) = error;
    }
    Eigen::Matrix2d cov = disp_mat * disp_mat.transpose();
    cov = cov/ ship_flatten->size();
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto V = svd.matrixV();
    auto S = svd.singularValues().asDiagonal().toDenseMatrix();
    auto U = svd.matrixU();
    // cout<<(cov - U * S * V.transpose())<<endl;
    if(V(0, 0) * V(1, 1) < 0){
        V.col(0) *= -1;
    }
    
    //==================Visualization============
    geometry_msgs::TransformStamped ship_tf;
    ship_tf.header.frame_id = "base_link";
    ship_tf.header.stamp = ros::Time::now();
    ship_tf.child_frame_id = "ship";
    ship_tf.transform.translation.x = centroid(0);
    ship_tf.transform.translation.y = centroid(1);
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation.block<2, 2>(0, 0) = V;
    Eigen::Quaterniond q(rotation);
    ship_tf.transform.rotation.w = q.w();
    ship_tf.transform.rotation.x = q.x();
    ship_tf.transform.rotation.y = q.y();
    ship_tf.transform.rotation.z = q.z();
    broadcaster_.sendTransform(ship_tf);

    visualization_msgs::Marker quadric;
    quadric.header.frame_id = "base_link";
    quadric.header.stamp = ros::Time::now();
    quadric.type = visualization_msgs::Marker::SPHERE;
    quadric.pose.position.x = centroid(0);
    quadric.pose.position.y = centroid(1);
    quadric.pose.orientation = ship_tf.transform.rotation;
    quadric.color.a = 0.7;
    quadric.color.r = 255.0;
    quadric.scale.x = 4 *sqrt(S(0, 0));
    quadric.scale.y = 4* sqrt(S(1, 1));
    quadric.scale.z = 1.0;
    pub_svd_quadric_.publish(quadric);

    sensor_msgs::PointCloud2 merged_scan;
    pcl::toROSMsg(*ship_flatten, merged_scan);
    merged_scan.header.frame_id = "base_link";
    merged_scan.header.stamp = ros::Time::now();
    pub_merged_scan_.publish(merged_scan);
    //===========================================
}