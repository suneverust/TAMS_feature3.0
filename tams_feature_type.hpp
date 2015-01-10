
/*************************************************
 * Add a new feature type which contains 
 * the rotation invariant frequency energy
 * of spherical functions.

 * Author: Bo Sun
 * Afflication: TAMS, University of Hamburg
 * E-Mail: bosun@informatik.uni-hamburg.de
 * Date: October  8, 2014
 ************************************************/

#ifndef TAMS_FEATURE_TYPE_
#define TAMS_FEATURE_TYPE_

#define PCL_NO_PRECOMPILE

// Equal to bandwidth of spherical harmonic transform
// I am REALLY SORRY using Global const variable
// Actually we want to set TAMSBANDWIDTH by the user
// but the POINT_CLOUD_REGISTER_POINT_STRUCT forbid doing that
// that is why in PCL we could see the SHOT352 and FPFT33 feature types,
// whose dimensions are set.(It is a litter weird, to be honest)
const int TAMS_AZIMUTH = 12;
const int TAMS_POLAR = 12;
const int FEATURE_SIZE = TAMS_AZIMUTH*TAMS_POLAR;

namespace tams{
/** \brief ADD a new feature type */
struct TAMSFeatureType
{
    float tams_sei[FEATURE_SIZE];
    static int descriptorSize() {return FEATURE_SIZE; }
    friend std::ostream& operator << (std::ostream& os, const TAMSFeatureType& p);
};

PCL_EXPORTS std::ostream&
operator << (std::ostream& os, const TAMSFeatureType& p)
{
    for (int i = 0; i < FEATURE_SIZE; ++i)
        os << (i == 0 ? "(" : "") << p.tams_sei[i] << (i < (FEATURE_SIZE-1) ? ", " : ")");
    return (os);
}
}

// Register New PointType
// (import to let other libraries in PCL know our new point type)
POINT_CLOUD_REGISTER_POINT_STRUCT ( tams::TAMSFeatureType,
                                    (float[FEATURE_SIZE], tams_sei, tams_sei))
#endif /*TAMS_FEATURE_TYPE_*/
