#include <ship_point_classification/api_class/point_classifier.h>

int main(int argc, char** argv){
    ros::init(argc, argv, "test_point_classifier");
    PointClassifier classifier;
    while (ros::ok())
    {
        ros::spinOnce();
    }
    
}