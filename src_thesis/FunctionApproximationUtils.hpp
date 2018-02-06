#ifndef __FA_UTILS_HPP__
#define __FA_UTILS_HPP__

#include <common/Constants.h>
#include <environment/ale_ram.hpp>
#include <set>

/**
 * FunctionApproximationUtils
 * Manages the obtention of vectors of features given
 * the type of the agent
 */
class FunctionApproximationUtils
{
private:
    // Maximum number of features of the feature vector (the size)
    int _numFeatures;
    
    // Feature vector
    int* _features;
    
    // Number of features which are not 0-valued
    int _numNonZeroFeatures;
    
    // Feature positions which should always be zero
    std::set<int> _nullFeatures;
    
public:
    // Constructor
    FunctionApproximationUtils(const std::string& agentType);
    
    // Destructor
    virtual ~FunctionApproximationUtils();

    // Getters
    int getNumFeatures() const;
    int* getFeatures() const;
    int getNumNonZeroFeatures() const;

    // Fills vector of features with binary features corresponding
    // to the bits extracted from the RAM
    void fillFeatureVectorFromRAM(const ALERAM& ram);
    
    // Features that should be ignored (i.e. not added to _features)
    void clearNullFeatures();
    void addNullFeature(int feature);
};

#endif /* __FA_UTILS_HPP__ */

