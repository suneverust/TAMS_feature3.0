/*********************************************
 * Author: Bo Sun                            *
 * Afflication: TAMS, University of Hamburg  *
 * E-Mail: bosun@informatik.uni-hamburg.de   *
 *         user_mail@QQ.com                  *
 * Date: Oct 13, 2014                        *
 * Licensing: GNU GPL license.               *
 *********************************************/

#ifndef TAMS_FEATURE_H_
#define TAMS_FEATURE_H_

#include <cmath>
#include <vector>
#include <algorithm>

#include <eigen3/Eigen/Dense>
#include <pcl/common/utils.h>
#include <pcl/common/geometry.h>
#include <pcl/common/angles.h>
#include <pcl/features/shot_lrf.h>

#include "tams_feature_type.hpp"

namespace tams{
template <typename PointInT, typename PointOuT = tams::TAMSFeatureType>
class TAMSFeatureEstimation : public pcl::Feature<PointInT, PointOuT>
{
public:
    typedef boost::shared_ptr<TAMSFeatureEstimation<PointInT, PointOuT> > Ptr;
    typedef boost::shared_ptr<const TAMSFeatureEstimation<PointInT, PointOuT> > ConstPtr;

    typedef pcl::ReferenceFrame PointLRF;
    typedef pcl::PointCloud<PointLRF> PointCloudLRF;
    typedef pcl::SHOTLocalReferenceFrameEstimation<PointInT, PointLRF> LRFEstimation;

    using pcl::Feature<PointInT, PointOuT>::feature_name_;
    using pcl::Feature<PointInT, PointOuT>::getClassName;
    using pcl::Feature<PointInT, PointOuT>::input_;
    using pcl::Feature<PointInT, PointOuT>::surface_;
    using pcl::Feature<PointInT, PointOuT>::indices_;
    using pcl::Feature<PointInT, PointOuT>::k_;
    using pcl::Feature<PointInT, PointOuT>::search_parameter_;
    using pcl::Feature<PointInT, PointOuT>::fake_surface_;

    typedef typename pcl::Feature<PointInT, PointOuT>::PointCloudOut PointCloudOut;

    // Constructor function
    TAMSFeatureEstimation ( )
    {
        // defaul values and could be set by user
        tams_entropy_bin_ = 12;
        tams_sei_azimutch_dim_=12;
        tams_sei_polar_dim_= 12;
        tams_sei_azimutch_spa_ = 2*M_PI/tams_sei_azimutch_dim_;
        tams_sei_polar_spa_ = M_PI/tams_sei_polar_dim_;

        feature_name_ = "TAMSFEATUREstimation";
    }

    // I/O to private attributes
    inline void setEntropyBinDim (size_t tams_entropy_bin_);
    inline void setSEIAzimuthDim (size_t tams_sei_azimutch_dim_);
    inline void setSEIPolarDim (size_t tams_sei_polar_dim_);
    inline int getEntropyBinDim ();
    inline int getSEIAzimuchDim ();
    inline int getSEIPolarDim ();

    /** \breif computeFeature is a virtual function of Class Feature
      * which means you have to define a computeFeature function for the class
      * inherit from Class Feature
      * Usually, this is the critical function of FeatureEstimation Class
      */
    void computeFeature (PointCloudOut &output);

protected:
    /** \brief cart2sph(X,Y,Z,azimuth,polar) transforms Cartesian coordinates stored in
      * corresponding elements of X, Y, and Z into spherical coordinates.
      * azimuth and polar are angular displacements in radians.
      * azimuth(longitudinal) is the counterclockwise angle in the x-y plane measured from the positive x-axis.
      * polar(colatitudianl) is the polar angle measured from the positive z axis.
      * 0 < azimuth < 2*M_PI; 0 < polar < M_PI
      */
    void tams_cart2sph (float x, float y, float z,
                        float& azimuth, float& polar);

    /** \brief tams_vector_normalization normalize the input vector
      * Parameters:
      * \param[in]     tams_vector   the input vector
      * \param[out]    tams_vector   the normalized vector (values are in range [0,1])
      */
    void tams_vector_normalization (std::vector<float> &tams_vector);

    /** \brief tams_vector2entropy compute the entropy of a vector
      * Parameters:
      * \param[in]   tams_vector     the input vector
      * \param[in]   hist_bin        the size of histogram in entropy computation
      * \param[out]  entropy         the resultant entropy
      */
    void tams_vector2entropy(const std::vector<float> & tams_vector,
                             float &entropy);

private:
    size_t tams_sei_azimutch_dim_, tams_sei_polar_dim_, tams_entropy_bin_;
    float tams_sei_azimutch_spa_, tams_sei_polar_spa_;
}; /*end of class*/
} /*end of namespace*/
#endif /*TAMS_FEATURE_H_*/
