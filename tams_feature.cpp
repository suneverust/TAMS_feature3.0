/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Nov 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/
#define PCL_NO_PRECOMPILE
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/console/print.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/feature.h>

#include "tams_feature_type.hpp"
#include "tams_feature.h"
#include "tams_feature.hpp"

using namespace pcl;
using namespace tams;

// Typedef Types
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef tams::TAMSFeatureType FeatureT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT;

int
main (int argc, char **argv)
{
  // Initiate Point clouds
  PointCloudT::Ptr object (new PointCloudT);
  PointCloudT::Ptr object_downsample (new PointCloudT);
  FeatureCloudT::Ptr object_features (new FeatureCloudT);

  // Get input object and scene
  if (argc != 2)
  {
    pcl::console::print_error ("Syntax is: %s object.pcd\n", argv[0]);
    return (1);
  }

  // Load object
  pcl::console::print_highlight ("Loading point clouds...\n");

  if (pcl::io::loadPCDFile<PointT> (argv[1], *object) < 0)
  {
    pcl::console::print_error ("Error loading object/scene file!\n");
    return (1);
  }

  // Remove the nan points
  std::vector<int> indices_object_nan;
  pcl::removeNaNFromPointCloud(*object, *object, indices_object_nan);

  // Downsample (Actually this part should be replaced by keypoints detection algorithm)
  pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointT> grid;
  grid.setLeafSize (1.0, 1.0, 1.0);
  grid.setInputCloud (object);
  grid.filter (*object_downsample);

  // Test Feature class
  pcl::console::print_highlight ("Estimating features...\n");
  tams::TAMSFeatureEstimation<PointT, FeatureT> fest;

  /*****The following snippet showing how to setIndices*********
    std::vector<int> indices;
    indices.push_back(100);
    indices.push_back(200);
    indices.push_back(2000);
    indices.push_back(1500);
    ...
    boost::shared_ptr<std::vector<int> > indicesptr (new std::vector<int> (indices));
    fest.setIndices (indicesptr);
  ***************************************************************/

  fest.setSEIAzimuthDim(TAMS_AZIMUTH);
  fest.setSEIPolarDim(TAMS_POLAR);
  fest.setInputCloud (object_downsample);
  fest.setSearchSurface(object);
  fest.setRadiusSearch(3.0);
  fest.computeFeature(*object_features);
  pcl::io::savePCDFileASCII ("test_object_feature.pcd", *object_features);

  return (0);
}

