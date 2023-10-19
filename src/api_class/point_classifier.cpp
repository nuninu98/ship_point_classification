#include <ship_point_classification/api_class/point_classifier.h>

PointClassifier::PointClassifier():  queue_(), spinner_(0, &queue_), private_nh_("~"){
    nh_.setCallbackQueue(&queue_);

    string topic;
    private_nh_.param<string>("ship_pointcloud_topic", topic, "/ouster/clustered");
    private_nh_.param<double>("layer_crop_thickness", layer_crop_thickness_, 0.1);
    sub_ship_points_ = nh_.subscribe(topic, 1, &PointClassifier::pointCloudCallback, this);
    ec_.setClusterTolerance(1.0);
    ec_.setMinClusterSize(5);
    pub_merged_scan_ = nh_.advertise<sensor_msgs::PointCloud2>("ouster_merged", 1);
    pub_svd_quadric_ = nh_.advertise<visualization_msgs::Marker>("ship_quadric", 1);

    pub_hull_points_ = nh_.advertise<sensor_msgs::PointCloud2>("hull_points", 1);
    pub_cabin_points_ = nh_.advertise<sensor_msgs::PointCloud2>("cabin_points", 1);

    spinner_.start();
}

PointClassifier::~PointClassifier(){
    spinner_.stop();
}

void PointClassifier::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& point){
    //ros::Time tic = ros::Time::now();
  
    pcl::PointCloud<pcl::PointXYZI> ship_cloud;
    pcl::fromROSMsg(*point, ship_cloud);
    
    double z_cut = 0.0;
    double z_max = -1000.0;
    double layer_crop_thickness_ = 0.1;

    for(size_t i = 0; i < ship_cloud.size(); i++){
        z_cut += ship_cloud[i].z / ship_cloud.size();
    }

    pcl::PointCloud<pcl::PointXYZI> ship_reflection_cropped;
    pcl::PointCloud<pcl::PointXYZI>::Ptr ship_flatten(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for(const auto& pt : ship_cloud){
        auto point = pt;
        point.z = 0.0;
        if(pt.z > z_cut){
            ship_flatten->push_back(point);
            ship_reflection_cropped.push_back(pt);
            if(pt.z > z_max){
                z_max = pt.z;
            }
        }
    }

    for(const auto& pt : ship_reflection_cropped){
        Eigen::Vector3d v(pt.x, pt.y, pt.z);
        v /= ship_flatten->size();
        centroid += v;
    }

    Eigen::MatrixXd disp_mat(2, ship_flatten->size());
    for(size_t i = 0; i < ship_flatten->size(); i++){
        auto pt = (*ship_flatten)[i];
        Eigen::Vector2d v(pt.x, pt.y);
        Eigen::Vector2d error = v - centroid.block<2, 1>(0, 0);
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

    double major_axis = S(0, 0);//2* sqrt(S(0, 0));
    double minor_axis = S(1, 1);//2* sqrt(S(1, 1));
    //===========Extract Conic Matrix=========
    Eigen::Matrix3d conic = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
    H.block<2, 2>(0, 0) = V;
    H(0, 2) = centroid(0);
    H(1, 2) = centroid(1);
    conic(0, 0) = pow(1.0 / major_axis, 2);
    conic(1, 1) = pow(1.0 / minor_axis, 2);
    conic(2, 2) = -1;
    conic = (H.inverse()).transpose() * conic * H.inverse();
    // cout<<conic<<endl;
    // cout<<endl;
    //========================================
    
    vector<pcl::PointCloud<pcl::PointXYZI>> ship_cloud_layer;
    for(double z = z_cut; z <= z_max; z += layer_crop_thickness_){ // Splitting Ship Pointcloud
        pcl::PointCloud<pcl::PointXYZI> layer_cloud;
        for(const auto& pt : ship_reflection_cropped){
            if(pt.z >= z && pt.z < z + layer_crop_thickness_){
                layer_cloud.push_back(pt);
            }
        }
        ship_cloud_layer.push_back(layer_cloud);
    }

    int most_fit_id = -1;
    size_t max_fit_points = 0;
    for(int i = ship_cloud_layer.size()-1; i >=0; i--){
        pcl::PointCloud<pcl::PointXYZI> layer = ship_cloud_layer[i];
        size_t cnt = 0;
        for(const auto& pt : layer){
            //cout<<"TEST: "<<geometricDist(pt.x, pt.y, conic)<<endl;
            if(abs(geometricDist(pt.x, pt.y, conic)) < 0.1){
                cnt++;
            }
        }
        if(max_fit_points < cnt){
            max_fit_points = cnt;
            most_fit_id = i;
        }
    }

    pcl::PointCloud<pcl::PointXYZI> hull_cloud, cabin_cloud;
    for(int id = 0; id <= most_fit_id; id++){
        hull_cloud += ship_cloud_layer[id];
    }
    for(int id = most_fit_id; id < ship_cloud_layer.size(); id++){
        cabin_cloud += ship_cloud_layer[id];
    }
    sensor_msgs::PointCloud2 hull_cloud_ros, cabin_cloud_ros;
    pcl::toROSMsg(hull_cloud, hull_cloud_ros);
    hull_cloud_ros.header.frame_id = point->header.frame_id;
    hull_cloud_ros.header.stamp = ros::Time::now();
    pub_hull_points_.publish(hull_cloud_ros);

    pcl::toROSMsg(cabin_cloud, cabin_cloud_ros);
    cabin_cloud_ros.header.frame_id = point->header.frame_id;
    cabin_cloud_ros.header.stamp = ros::Time::now();
    pub_cabin_points_.publish(cabin_cloud_ros);

    //==================Visualization============
    geometry_msgs::TransformStamped ship_tf;
    ship_tf.header.frame_id = point->header.frame_id;
    ship_tf.header.stamp = ros::Time::now();
    ship_tf.child_frame_id = "ship";
    ship_tf.transform.translation.x = centroid(0);
    ship_tf.transform.translation.y = centroid(1);
    ship_tf.transform.translation.z = centroid(2);
    Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
    rotation.block<2, 2>(0, 0) = V;
    Eigen::Quaterniond q(rotation);
    ship_tf.transform.rotation.w = q.w();
    ship_tf.transform.rotation.x = q.x();
    ship_tf.transform.rotation.y = q.y();
    ship_tf.transform.rotation.z = q.z();
    broadcaster_.sendTransform(ship_tf);

    visualization_msgs::Marker quadric;
    quadric.header.frame_id = point->header.frame_id;
    quadric.header.stamp = ros::Time::now();
    quadric.type = visualization_msgs::Marker::CYLINDER;
    quadric.pose.position.x = centroid(0);
    quadric.pose.position.y = centroid(1);
    if(most_fit_id != -1){
        quadric.pose.position.z = z_cut + layer_crop_thickness_ * most_fit_id;
    }
    else{
        quadric.pose.position.z = centroid(2);
    }
    quadric.pose.orientation = ship_tf.transform.rotation;
    quadric.color.a = 0.7;
    quadric.color.r = 255.0;
    quadric.scale.x = 2* major_axis;
    quadric.scale.y = 2* minor_axis;
    quadric.scale.z = 0.1;
    pub_svd_quadric_.publish(quadric);

    sensor_msgs::PointCloud2 merged_scan;
    //pcl::toROSMsg(ship_cloud_layer[most_fit_id], merged_scan);
    pcl::toROSMsg(ship_reflection_cropped, merged_scan);
    merged_scan.header.frame_id = "base_link";
    merged_scan.header.stamp = ros::Time::now();
    pub_merged_scan_.publish(merged_scan);
    //===========================================
    //cout<<"TIME: "<<(ros::Time::now() - tic).toSec()* 1000.0 <<"ms"<<endl;
}

double PointClassifier::algebraicDist(double x, double y, const Eigen::Matrix3d& conic){
    Eigen::Vector3d point_h(x, y, 1.0);
    return (point_h.transpose() * conic * point_h).value();
}

double PointClassifier::geometricDist(double x, double y, const Eigen::Matrix3d& conic){
    Eigen::Vector3d point_h(x, y, 1.0);
    double err = algebraicDist(x, y, conic);
    // cout<<conic<<endl;
    Eigen::Vector3d Cx = conic * point_h;
    // Eigen::MatrixXd jacobian(2, 1);
    // jacobian(0, 0) = 2* Cx(0);
    // jacobian(0, 1) = 2* Cx(1);
    // auto delta = -jacobian.transpose() * (jacobian * jacobian.transpose()).inverse() * err;
    // return delta.norm();
    return sqrt(pow(err, 2) / (4*(pow(Cx(0), 2) + pow(Cx(1), 2))));
}