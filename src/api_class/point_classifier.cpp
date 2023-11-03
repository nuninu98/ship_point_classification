#include <ship_point_classification/api_class/point_classifier.h>

PointClassifier::PointClassifier(): buffer_(), listener_(buffer_), queue_(), spinner_(0, &queue_), cloud1_sub_(nh_, "/ouster1/points", 1), 
cloud2_sub_(nh_, "/ouster2/points", 1), sync_(ApproxPolicy(10), cloud1_sub_, cloud2_sub_){
    nh_.setCallbackQueue(&queue_);
    sync_.registerCallback(boost::bind(&PointClassifier::pointSyncCallback, this, _1, _2));

    lidar1_tf_ = Eigen::Matrix4d::Identity();
    lidar1_tf_(0, 3) = (1.415/ 2.0);
    lidar1_tf_(1, 3) = -(3.325/ 2.0);
    lidar1_tf_(0, 0) = cos(-M_PI/2);
    lidar1_tf_(0, 1) = -sin(-M_PI/2);
    lidar1_tf_(1, 0) = sin(-M_PI/2);
    lidar1_tf_(1, 1) = cos(-M_PI/2);

    lidar2_tf_ = Eigen::Matrix4d::Identity();
    lidar2_tf_(0, 3) = -(1.415/ 2.0);
    lidar2_tf_(1, 3) = (3.325/ 2.0);
    lidar1_tf_(2, 3) = 0.8;
    lidar2_tf_(0, 0) = cos(M_PI/2);
    lidar2_tf_(0, 1) = -sin(M_PI/2);
    lidar2_tf_(1, 0) = sin(M_PI/2);
    lidar2_tf_(1, 1) = cos(M_PI/2);
    ec_.setClusterTolerance(0.5);
    ec_.setMinClusterSize(100);
    pub_merged_scan_ = nh_.advertise<sensor_msgs::PointCloud2>("ouster_merged", 1);
    pub_svd_quadric_ = nh_.advertise<visualization_msgs::Marker>("ship_quadric", 1);

    pub_hull_points_ = nh_.advertise<sensor_msgs::PointCloud2>("hull_points", 1);
    pub_cabin_points_ = nh_.advertise<sensor_msgs::PointCloud2>("cabin_points", 1);
    downsampler_.setLeafSize(0.4, 0.4, 0.1);

    ne_.setKSearch(5);
    spinner_.start();
}

PointClassifier::~PointClassifier(){
    spinner_.stop();
}

int PointClassifier::getRow(pcl::PointXYZI pt){
    double altitude = atan2(pt.z, sqrt(pow(pt.x, 2) + pow(pt.y, 2))) * (180.0 / M_PI);
    double vertical_resolution = (max_altitude - min_altitude_) / vertical_scans_;
    return (altitude - min_altitude_) / vertical_resolution;
}

int PointClassifier::getCol(pcl::PointXYZI pt){
    double bearing = atan2(pt.y, pt.x) * (180.0 / M_PI);
    double horizontal_resolution = 360.0 / horizontal_scans_;
    int id = (bearing + 180.0) / horizontal_resolution;
    return id % horizontal_scans_;
}

void PointClassifier::pointSyncCallback(const sensor_msgs::PointCloud2ConstPtr& point1, const sensor_msgs::PointCloud2ConstPtr& point2){
    //ros::Time tic = ros::Time::now();
    pcl::PointCloud<pcl::PointXYZI> cloud1, cloud2;
    pcl::fromROSMsg(*point1, cloud1);
    pcl::fromROSMsg(*point2, cloud2);

    pcl::PointCloud<pcl::PointXYZI> cloud1_tf, cloud2_tf;
    pcl::transformPointCloud(cloud1, cloud1_tf, lidar1_tf_);
    pcl::transformPointCloud(cloud2, cloud2_tf, lidar2_tf_);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_merged_range(new pcl::PointCloud<pcl::PointXYZI>);
    *cloud_merged += cloud1_tf;
    *cloud_merged += cloud2_tf;
    for(const auto& pt : *cloud_merged){
        double r = sqrt(pow(pt.x, 2) + pow(pt.y, 2) + pow(pt.z, 2));
        if(r > 7.0 && r < 50.0){
            cloud_merged_range->push_back(pt);
        }
    }
    cv::Mat lidar1_img = cv::Mat::zeros(vertical_scans_, horizontal_scans_, CV_64F);
    cv::Mat lidar2_img = cv::Mat::zeros(vertical_scans_, horizontal_scans_, CV_64F);
    for(const auto& pt : cloud1){
        double range = sqrt(pow(pt.x, 2) + pow(pt.y, 2));
        if(range < 7.0){
            continue;
        }
        int col = getCol(pt);
        int row = getRow(pt);     
        if(row < 0 || row >= lidar1_img.rows){
            continue;
        }
        lidar1_img.at<double>(row, col) = range;
    }
    for(const auto& pt : cloud2){
        double range = sqrt(pow(pt.x, 2) + pow(pt.y, 2));
        if(range < 7.0){
            continue;
        }
        int col = getCol(pt);
        int row = getRow(pt);     
        if(row < 0 || row >= lidar2_img.rows){
            continue;
        }
        lidar2_img.at<double>(row, col) = range;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr edge_points(new pcl::PointCloud<pcl::PointXYZI>);
    for(int col = 0; col < lidar1_img.cols; col++){
        double min_range = 100000.0;
        int min_row = -1;
        for(int row = 0; row < lidar1_img.rows; row++){
            double range = lidar1_img.at<double>(row, col);
            if(range == 0.0){
                continue;
            }
            if(range < min_range){
                min_range = range;
                min_row = row;
            }
        }
        if(min_row != -1){
            double range =lidar1_img.at<double>(min_row, col);
            Eigen::Vector4d point_restore, point_restore_tf;
            double horizontal_resolution = 360.0 / horizontal_scans_;
            double vertical_resolution = (max_altitude - min_altitude_) / vertical_scans_;
            point_restore(0) = range * cos(DEG2RAD(-180.0 + col * horizontal_resolution));
            point_restore(1) = range * sin(DEG2RAD(-180.0 + col * horizontal_resolution));
            point_restore(2) = 0.0; //range* tan(DEG2RAD(min_altitude_ + min_row * vertical_resolution));
            point_restore(3) = 1.0;
            point_restore_tf = lidar1_tf_ * point_restore;
            
            pcl::PointXYZI pt;
            pt.x = point_restore_tf(0);
            pt.y = point_restore_tf(1);
            pt.z = 0.0;
            edge_points->push_back(pt);
        }
    }

    for(int col = 0; col < lidar2_img.cols; col++){
        double min_range = 100000.0;
        int min_row = -1;
        for(int row = 0; row < lidar2_img.rows; row++){
            double range = lidar2_img.at<double>(row, col);
            if(range == 0.0){
                continue;
            }
            if(range < min_range){
                min_range = range;
                min_row = row;
            }
        }
        if(min_row != -1){
            double range =lidar2_img.at<double>(min_row, col);
            Eigen::Vector4d point_restore, point_restore_tf;
            double horizontal_resolution = 360.0 / horizontal_scans_;
            double vertical_resolution = (max_altitude - min_altitude_) / vertical_scans_;
            point_restore(0) = range * cos(DEG2RAD(-180.0 + col * horizontal_resolution));
            point_restore(1) = range * sin(DEG2RAD(-180.0 + col * horizontal_resolution));
            point_restore(2) = 0.0; //range* tan(DEG2RAD(min_altitude_ + min_row * vertical_resolution));
            point_restore(3) = 1.0;
            point_restore_tf = lidar2_tf_ * point_restore;
            
            pcl::PointXYZI pt;
            pt.x = point_restore_tf(0);
            pt.y = point_restore_tf(1);
            pt.z = 0.0;
            edge_points->push_back(pt);
        }
    }
    pcl::search::KdTree<pcl::PointXYZI> edge_kdtree;
    edge_kdtree.setInputCloud(edge_points); 
    downsampler_.setInputCloud(cloud_merged_range);
    downsampler_.filter(*cloud_merged_range);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZI>);
    kdtree->setInputCloud(cloud_merged_range);
    vector<pcl::PointIndices> indices;
    ec_.setSearchMethod(kdtree);
    ec_.setInputCloud(cloud_merged_range);
    ec_.extract(indices);
    
    pcl::PointCloud<pcl::PointXYZI> hull_cloud, cabin_cloud;
    for(int ship_id = 0; ship_id < indices.size(); ship_id++){
        pcl::search::KdTree<pcl::PointXYZI>::Ptr ship_tree(new pcl::search::KdTree<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI> ship_points;
        double z_max = -1000.0;
        double z_min = 1000.0;
        double thick = 0.3;
        for(const auto& id : indices[ship_id].indices){
            ship_points.push_back((*cloud_merged_range)[id]);
            if((*cloud_merged_range)[id].z > z_max){
                z_max = (*cloud_merged_range)[id].z;
            }
            if((*cloud_merged_range)[id].z < z_min){
                z_min = (*cloud_merged_range)[id].z;
            }
        }
        if(z_max <= z_min){
            continue;
        }
        vector<pcl::PointCloud<pcl::PointXYZI>> ship_layers;
        ship_layers.resize(ceil((z_max -z_min) / thick) + 1);
        for(auto pt : ship_points){
            int layer = (pt.z - z_min) / thick;
            ship_layers[layer].push_back(pt);
        }
        // pcl::PointCloud<pcl::PointXYZI>::Ptr ship_points_ptr (new pcl::PointCloud<pcl::PointXYZI>(ship_points));
        // pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
        // ne_.setSearchMethod(ship_tree);
        // ne_.setInputCloud(ship_points_ptr);
        // ne_.compute(*normals);
        // reg_.setSearchMethod(ship_tree);
        // reg_.setInputNormals(normals);
        // reg_.setInputCloud(ship_points_ptr);
        // reg_.setSmoothnessThreshold(10.0 / 180.0 * M_PI);
        // reg_.setCurvatureThreshold(1.0);
        // vector<pcl::PointIndices> clusters;
        // reg_.extract(clusters);
        // sort(clusters.begin(), clusters.end(), [](const pcl::PointIndices& arr1, const pcl::PointIndices& arr2){
        //     return arr1.indices.size() > arr2.indices.size();
        // });
        // for(auto id : clusters[0].indices){
        //     hull_cloud.push_back((*ship_points_ptr)[id]);
        // }
        int best_match_cnt = 0;
        int best_layer = -1;
        for(int layer = ship_layers.size() -1; layer >= 0; layer--){
            int cnt = 0;
            unordered_set<int> edge_matched_ids;
            sort(ship_layers[layer].begin(), ship_layers[layer].end(), [](const pcl::PointXYZI& p1, const pcl::PointXYZI& p2){
                return p1.z > p2.z;
            });
            for(int i = 0; i < ship_layers[layer].size(); i++){
                pcl::PointXYZI search_pt = ship_layers[layer][i];
                search_pt.z = 0.0;
                vector<int> ids;
                vector<float> dists;
                edge_kdtree.nearestKSearch(search_pt, 1, ids, dists);
                if(edge_matched_ids.find(ids[0]) == edge_matched_ids.end()){
                    if(dists[0] < 0.15){
                        cnt++;
                        edge_matched_ids.insert(ids[0]);
                        //hull_cloud.push_back(ship_points[i]);
                    }
                }
            }
            if(best_match_cnt < cnt){
                best_match_cnt = cnt;
                best_layer = layer;
            }
        }
        if(best_layer == -1){
            continue;
        }
        // cout<<"BEST: "<<best_layer<<" /"<<ship_layers.size()<<endl;
        unordered_set<int> edge_matched_ids;
        for(int i = 0; i < ship_layers[best_layer].size(); i++){
            pcl::PointXYZI search_pt = ship_layers[best_layer][i];
            search_pt.z = 0.0;
            vector<int> ids;
            vector<float> dists;
            edge_kdtree.nearestKSearch(search_pt, 1, ids, dists);
            if(edge_matched_ids.find(ids[0]) == edge_matched_ids.end()){
                if(dists[0] < 0.15){
                    edge_matched_ids.insert(ids[0]);
                    hull_cloud.push_back(ship_layers[best_layer][i]);
                }
            }
        }

        for(int layer = 0; layer < best_layer; layer++){
            hull_cloud += ship_layers[layer];
        }
        for(int layer = best_layer + 1; layer < ship_layers.size(); layer++){
            cabin_cloud += ship_layers[layer];
        }
        
        
    }
    sensor_msgs::PointCloud2 ouster_merged_ros;
    pcl::toROSMsg(*cloud_merged_range, ouster_merged_ros);
    ouster_merged_ros.header.stamp = ros::Time::now();
    ouster_merged_ros.header.frame_id = "base_link";
    pub_merged_scan_.publish(ouster_merged_ros);

    sensor_msgs::PointCloud2 hull_cloud_ros;
    pcl::toROSMsg(hull_cloud, hull_cloud_ros);
    hull_cloud_ros.header = ouster_merged_ros.header;
    pub_hull_points_.publish(hull_cloud_ros);

    sensor_msgs::PointCloud2 cabin_cloud_ros;
    pcl::toROSMsg(cabin_cloud, cabin_cloud_ros);
    cabin_cloud_ros.header = ouster_merged_ros.header;
    pub_cabin_points_.publish(cabin_cloud_ros);
    // sensor_msgs::PointCloud2 edge_points_ros;
    // pcl::toROSMsg(*edge_points, edge_points_ros);
    // edge_points_ros.header = ouster_merged_ros.header;
    // pub_hull_points_.publish(edge_points_ros);
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