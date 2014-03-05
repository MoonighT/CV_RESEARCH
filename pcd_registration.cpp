#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/common/io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/pfh.h>

void
downsample(pcl::PointCloud<pcl::PointXYZ>::Ptr &points, float leaf_size,
           pcl::PointCloud<pcl::PointXYZ>::Ptr &downsampled_out)
{
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    vox_grid.setInputCloud(points);
    vox_grid.filter(*downsampled_out);
}

void
compute_surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &points,
            float normal_radius, pcl::PointCloud<pcl::Normal>::Ptr &normal_out)
{
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
    norm_est.setSearchMethod(pcl::search::KdTree<pcl::PointXYZ>::Ptr(new pcl::search::KdTree<pcl::PointXYZ>));
    norm_est.setRadiusSearch(normal_radius);
    norm_est.setInputCloud(points);
    norm_est.compute(*normal_out);
}


void
visualize_normals(const pcl::PointCloud<pcl::PointXYZ>::Ptr points,
                  const pcl::PointCloud<pcl::PointXYZ>::Ptr normal_points,
                  const pcl::PointCloud<pcl::Normal>::Ptr normals)
{
    pcl::visualization::PCLVisualizer viz;
    viz.addPointCloud(points, "points");
    viz.addPointCloud(normal_points, "normal_points");
    viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(normal_points,normals,1,0.02);
    viz.spin();
}

void
detect_keypoints(pcl::PointCloud<pcl::PointNormal>::Ptr &points, float min_scale,
                 int nr_octaves, int nr_scales_per_octave, float min_contrast, 
                 pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints_out)
{
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift_detect;
    sift_detect.setSearchMethod(pcl::search::KdTree<pcl::PointNormal>::Ptr
        (new pcl::search::KdTree<pcl::PointNormal>));
    sift_detect.setScales(min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast(min_contrast);
    sift_detect.setInputCloud(points);
    sift_detect.compute(*keypoints_out);
}

void
PFH_features_at_keypoints(pcl::PointCloud<pcl::PointNormal>::Ptr &points,
                          pcl::PointCloud<pcl::Normal>::Ptr &normals,
                          pcl::PointCloud<pcl::PointWithScale>::Ptr &keypoints,
                          float feature_radius,
                          pcl::PointCloud<pcl::PFHSignature125>::Ptr &descriptors_out)
{
    pcl::PFHEstimation<pcl::PointNormal, pcl::Normal, pcl::PFHSignature125> pfh_est;
    pfh_est.setSearchMethod(pcl::search::KdTree<pcl::PointNormal>::Ptr
        (new pcl::search::KdTree<pcl::PointNormal>));
    pfh_est.setRadiusSearch(feature_radius);
    pcl::PointCloud<pcl::PointNormal>::Ptr keypoints_xyzrgb
        (new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*keypoints, *keypoints_xyzrgb);
    pfh_est.setSearchSurface(points);
    pfh_est.setInputNormals(normals);
    pfh_est.setInputCloud(keypoints_xyzrgb);
    pfh_est.compute(*descriptors_out);
}

void
features_correspondences(pcl::PointCloud<pcl::PFHSignature125>::Ptr &source_descriptors,
                         pcl::PointCloud<pcl::PFHSignature125>::Ptr &target_descriptors,
                         std::vector<int> &correspondences_out,
                         std::vector<float> &correspondences_scores_out)
{
    correspondences_out.resize(source_descriptors->size());
    correspondences_scores_out.resize(source_descriptors->size());
    pcl::search::KdTree<pcl::PFHSignature125> descriptors_kdtree;
    descriptors_kdtree.setInputCloud(target_descriptors);
    const int k = 1;
    std::vector<int> k_indices(k);
    std::vector<float> k_squared_distances(k);
    for(size_t i = 0; i < source_descriptors->size(); ++i)
    {
        descriptors_kdtree.nearestKSearch(*source_descriptors,i,k,k_indices, k_squared_distances);
        correspondences_out[i] = k_indices[0];
        correspondences_scores_out[i] = k_squared_distances[0];
    }
}

int main(int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals1(new pcl::PointCloud<pcl::Normal>);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals2(new pcl::PointCloud<pcl::Normal>);
    //read pcd file
    
    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("bun000.pcd", *cloud1)==-1){
        PCL_ERROR("Couldn`t read file test.pcd.pcd\n");
        return -1;
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("bun045.pcd", *cloud2)==-1){
        PCL_ERROR("Couldn`t read file test.pcd.pcd\n");
        return -1;
    }

    //const float voxel_grid_leaf_size = 0.01;
    //downsample(cloud1, voxel_grid_leaf_size, ds1);
    //downsample(cloud2, voxel_grid_leaf_size, ds2);
    const float normal_radius = 0.03;
    compute_surface_normals(cloud1, normal_radius, normals1);
    compute_surface_normals(cloud2, normal_radius, normals2);

    pcl::PointCloud<pcl::PointNormal>::Ptr pointNormal1(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr pointNormal2(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud1, *normals1, *pointNormal1);
    pcl::concatenateFields(*cloud2, *normals2, *pointNormal2);


    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_out1(new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_out2(new pcl::PointCloud<pcl::PointWithScale>);
    
    detect_keypoints(pointNormal1, 0.001, 6, 4, 0.001, keypoints_out1);
    detect_keypoints(pointNormal2, 0.001, 6, 4, 0.001, keypoints_out2);
    
    cout << "keypoint numbers:" << endl;
    cout << keypoints_out1->size() << endl;
    cout << keypoints_out2->size() << endl;

    
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors_out1(new pcl::PointCloud<pcl::PFHSignature125>);
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors_out2(new pcl::PointCloud<pcl::PFHSignature125>);
    PFH_features_at_keypoints(pointNormal1, normals1, keypoints_out1, 0.03, descriptors_out1);
    PFH_features_at_keypoints(pointNormal2, normals2, keypoints_out2, 0.03, descriptors_out2);

    std::vector<int> correspondences_out;
    std::vector<float> correspondences_scores_out;

    features_correspondences(descriptors_out1, descriptors_out2, correspondences_out, correspondences_scores_out);
    cout << correspondences_out.size();
    //correspondences_scores_out;                     

    //visualize_normals(cloud1, ds1, normals1);
    //visualize_normals(cloud2, ds2, normals2);

    //pcl::visualization::PCLVisualizer viz;
    //viz.addPointCloud(cloud1, "points1");
    //viz.addPointCloud(cloud2, "points2");


    //viz.addPointCloud(normal_points, "normal_points");
    //viz.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(normal_points,normals,1,0.02);
    //viz.spin();
    
    return (0);
    //0.002 4 5 1
}
