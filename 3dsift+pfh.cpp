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
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);

	//read pcd file
	if (pcl::io::loadPCDFile<pcl::PointXYZ> ("bun000.pcd", *cloud_src)==-1){
        PCL_ERROR("Couldn`t read file test.pcd.pcd\n");
        return -1;
    }

    if (pcl::io::loadPCDFile<pcl::PointXYZ> ("bun045.pcd", *cloud_tgt)==-1){
        PCL_ERROR("Couldn`t read file test.pcd.pcd\n");
        return -1;
    }

    //Multiple View-Ports 
    boost::shared_ptr<pcl::visualization::PCLVisualizer> MView (new pcl::visualization::PCLVisualizer ("Aligning")); 
    MView->initCameraParameters (); 
    //View-Port1 
    int v1(0); 
	MView->createViewPort (0.0, 0.0, 0.5, 1.0, v1); 
	MView->setBackgroundColor (0, 0, 0, v1); 
	MView->addText ("Start:View-Port 1", 10, 10, "v1_text", v1); 
	
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green (cloud_src, 0,255,0); 
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red (cloud_tgt, 255,0,0); 
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> white (cloud_tgt, 255,255,255); 
	//View-Port2 
	int v2(0); 
	MView->createViewPort (0.5, 0.0, 1.0, 1.0, v2); 
	MView->setBackgroundColor (0, 0, 0, v2); 
	MView->addText ("Aligned:View-Port 2", 10, 10, "v2_text", v2); 
          
	MView->addPointCloud (cloud_src, green, "source", v1); 
	MView->addPointCloud (cloud_tgt, red, "target", v2); 

	//MView->addPointCloud (cloud_tgt, red, "target2", v2); 
	//MView->addPointCloud (cloud_src, green, "source2", v2); 
	//Properties for al viewports 
	MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source"); 
	MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target"); 
	//MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target2");	
	//MView->spin();

	//remove NAN-Points 
    std::vector<int> indices1,indices2; 
    pcl::removeNaNFromPointCloud (*cloud_src, *cloud_src, indices1); 
    pcl::removeNaNFromPointCloud (*cloud_tgt, *cloud_tgt, indices2);

    //Downsampling 
    PCL_INFO ("Downsampling \n"); 
    //temp clouds src & tgt 
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_src (new pcl::PointCloud<pcl::PointXYZ>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr ds_tgt (new pcl::PointCloud<pcl::PointXYZ>); 
    pcl::VoxelGrid<pcl::PointXYZ> grid; 
    grid.setLeafSize (0.05, 0.05, 0.05); 
    grid.setInputCloud (cloud_src); 
    grid.filter (*ds_src); 
    grid.setInputCloud (cloud_tgt); 
    grid.filter (*ds_tgt);	
    PCL_INFO ("	Downsampling finished \n"); 


    // Normal-Estimation 
    PCL_INFO ("Normal Estimation \n"); 
    pcl::PointCloud<pcl::Normal>::Ptr norm_src (new pcl::PointCloud<pcl::Normal>); 
    pcl::PointCloud<pcl::Normal>::Ptr norm_tgt (new pcl::PointCloud<pcl::Normal>); 
            
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree<pcl::PointXYZ>()); 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree<pcl::PointXYZ>()); 
    
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne; 
    //Source-Cloud 
    PCL_INFO ("	Normal Estimation - Source \n");	
    ne.setInputCloud (cloud_src); 
    ne.setSearchSurface (cloud_src); 
    ne.setSearchMethod (tree_src); 
    ne.setRadiusSearch (0.05); 
    ne.compute (*norm_src); 

    //Target-Cloud 
    PCL_INFO ("	Normal Estimation - Target \n"); 
    ne.setInputCloud (cloud_tgt); 
    ne.setSearchSurface (cloud_tgt); 
    ne.setSearchMethod (tree_tgt); 
    ne.setRadiusSearch (0.03); 
    ne.compute (*norm_tgt); 

    
    //detect keypoints
    pcl::PointCloud<pcl::PointNormal>::Ptr pointNormal1(new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>::Ptr pointNormal2(new pcl::PointCloud<pcl::PointNormal>);
    pcl::concatenateFields(*cloud_src, *norm_src, *pointNormal1);
    pcl::concatenateFields(*cloud_tgt, *norm_tgt, *pointNormal2);


    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_out1(new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>::Ptr keypoints_out2(new pcl::PointCloud<pcl::PointWithScale>);
    
    detect_keypoints(pointNormal1, 0.001, 6, 4, 0.001, keypoints_out1);
    detect_keypoints(pointNormal2, 0.001, 6, 4, 0.001, keypoints_out2);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_src (new pcl::PointCloud<pcl::PointXYZ>); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_tgt (new pcl::PointCloud<pcl::PointXYZ>); 



    cout << "keypoint numbers:" << endl;
    cout << keypoints_out1->size() << endl;
    cout << keypoints_out2->size() << endl;

    (*keypoints_src).points.resize(keypoints_out1->size());
    (*keypoints_tgt).points.resize(keypoints_out2->size());

    for(int i = 0; i < keypoints_out1->size(); i++){
        cout << (*keypoints_out1)[i].x << " ";
        cout << (*keypoints_out1)[i].y << " ";
        cout << (*keypoints_out1)[i].z << " ";
        cout << (*keypoints_out1)[i].scale << endl;
        (*keypoints_src)[i].x = (*keypoints_out1)[i].x;
        (*keypoints_src)[i].y = (*keypoints_out1)[i].y;
        (*keypoints_src)[i].z = (*keypoints_out1)[i].z;
    }
    MView->addPointCloud (keypoints_src, white, "source_key", v1); 

    for(int i = 0; i < keypoints_out2->size(); i++){
        cout << (*keypoints_out2)[i].x << " ";
        cout << (*keypoints_out2)[i].y << " ";
        cout << (*keypoints_out2)[i].z << " ";
        cout << (*keypoints_out2)[i].scale << endl;
        (*keypoints_tgt)[i].x = (*keypoints_out2)[i].x;
        (*keypoints_tgt)[i].y = (*keypoints_out2)[i].y;
        (*keypoints_tgt)[i].z = (*keypoints_out2)[i].z;
    }
    
    MView->addPointCloud (keypoints_tgt, white, "target_key", v2); 

    MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "source_key");
    MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_key");
    
    // PCL_INFO (" Normal Estimation - Source \n");    
    // ne.setInputCloud (ds_src); 
    // ne.setSearchSurface (cloud_src); 
    // ne.setSearchMethod (tree_src); 
    // ne.setRadiusSearch (0.05); 
    // ne.compute (*norm_src); 

    // //Target-Cloud 
    // PCL_INFO (" Normal Estimation - Target \n"); 
    // ne.setInputCloud (ds_tgt); 
    // ne.setSearchSurface (cloud_tgt); 
    // ne.setSearchMethod (tree_tgt); 
    // ne.setRadiusSearch (0.03); 
    // ne.compute (*norm_tgt); 


    //PFH-Source 
    // PCL_INFO ("PFH - started\n");
    // pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh_est_src; 
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_pfh_src (new pcl::search::KdTree<pcl::PointXYZ>()); 
    // pfh_est_src.setSearchMethod (tree_pfh_src); 
    // pfh_est_src.setRadiusSearch (0.1); 
    // pfh_est_src.setSearchSurface (keypoints_src);   
    // pfh_est_src.setInputNormals (norm_src); 
    // pfh_est_src.setInputCloud (keypoints_src); 
    // pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_src (new pcl::PointCloud<pcl::PFHSignature125>); 
    // PCL_INFO (" PFH - Compute Source\n"); 
    // pfh_est_src.compute (*pfh_src); 
    // PCL_INFO (" PFH - finished\n"); 
    // //PFH-Target 
    // pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh_est_tgt; 
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_pfh_tgt (new pcl::search::KdTree<pcl::PointXYZ>()); 
    // pfh_est_tgt.setSearchMethod (tree_pfh_tgt); 
    // pfh_est_tgt.setRadiusSearch (0.1); 
    // pfh_est_tgt.setSearchSurface (keypoints_tgt);   
    // pfh_est_tgt.setInputNormals (norm_tgt); 
    // pfh_est_tgt.setInputCloud (keypoints_tgt); 
    // pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_tgt (new pcl::PointCloud<pcl::PFHSignature125>); 
    // PCL_INFO (" PFH - Compute Target\n"); 
    // pfh_est_tgt.compute (*pfh_tgt); 
    // PCL_INFO (" PFH - finished\n"); 
    
    

    //FPFH Source 
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est_src; 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_fpfh_src (new pcl::search::KdTree<pcl::PointXYZ>); 
    fpfh_est_src.setSearchSurface (cloud_src);//<-------------- Use All Points for analyzing  the local structure of the cloud 
    fpfh_est_src.setInputCloud (keypoints_src); //<------------- But only compute features at the key-points 
    fpfh_est_src.setInputNormals (norm_src); 
    fpfh_est_src.setRadiusSearch (0.05);  
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_src (new pcl::PointCloud<pcl::FPFHSignature33>); 
    PCL_INFO ("   FPFH - Compute Source\n"); 
    fpfh_est_src.compute (*fpfh_src); 

    //FPFH Target 
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est_tgt; 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_fpfh_tgt (new pcl::search::KdTree<pcl::PointXYZ>); 
    fpfh_est_src.setSearchSurface (cloud_tgt); 
    fpfh_est_tgt.setInputCloud (keypoints_tgt); 
    fpfh_est_tgt.setInputNormals (norm_tgt); 
    fpfh_est_tgt.setRadiusSearch (0.05); 
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_tgt (new pcl::PointCloud<pcl::FPFHSignature33>); 
    PCL_INFO ("   FPFH - Compute Target\n"); 
    fpfh_est_tgt.compute (*fpfh_tgt); 
    PCL_INFO ("   FPFH - finished\n"); 




    PCL_INFO ("Correspondence Estimation\n"); 
    pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corEst; 
    corEst.setInputCloud (fpfh_src); 
    corEst.setInputTarget (fpfh_tgt); 
    PCL_INFO (" Correspondence Estimation - Estimate C.\n"); 
    //pcl::Correspondences cor_all; 
    //Pointer erzeugen 
    boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences); 
    corEst.determineCorrespondences (*cor_all_ptr); 
    PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size());  



    Eigen::Matrix4f transformation; 
                
    PCL_INFO ("Correspondence Rejection Features\n"); 

    //SAC 
    double epsilon_sac = 0.1; // 10cm 
    int iter_sac = 10000; 
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> sac; 
    //pcl::registration::corres 
    sac.setInputCloud (cloud_src); 
    sac.setTargetCloud (cloud_tgt); 
    sac.setInlierThreshold (epsilon_sac); 
    sac.setMaxIterations (iter_sac); 
    sac.setInputCorrespondences (cor_all_ptr); 

    boost::shared_ptr<pcl::Correspondences> cor_inliers_ptr (new pcl::Correspondences); 
    sac.getCorrespondences (*cor_inliers_ptr); 
    //int sac_size = cor_inliers_ptr->size(); 
    PCL_INFO (" RANSAC: %d Correspondences Remaining\n", cor_inliers_ptr->size ()); 

    transformation = sac.getBestTransformation(); 

    //Punktwolke Transformieren 
    //pcl::transformPointCloud (*cloud_src, *cloud_tmp, transformation); 
    //MView->addPointCloud (cloud_tmp, green, "tmp", v2); 
    //MView->addPointCloud (cloud_tgt, red, "target_2", v2); 
    //MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "tmp"); 
    //MView->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_2"); 
    
    
// Warten bis Viewer geschlossen wird 
    while (!MView->wasStopped()) 
    { 
            MView->spinOnce(100); 
    } 

    

    return 0;
}



