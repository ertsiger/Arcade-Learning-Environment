#include "FunctionApproximationUtils.hpp"

#define BYTE_LENGTH 8

FunctionApproximationUtils::FunctionApproximationUtils(const std::string& agentType)
: _numNonZeroFeatures(0)
{
    if (agentType == "ram_agent")
    {
        _numFeatures = RAM_LENGTH * BYTE_LENGTH + 1;
    }
    else
    {
        std::cerr << "Error: Agent type is not correctly specified for function approximation" << std::endl;
        exit(1);
    }

    _features = new int[_numFeatures];
}

FunctionApproximationUtils::~FunctionApproximationUtils()
{
    if (_features != NULL)
    {
        delete [] _features;
    }
}

int FunctionApproximationUtils::getNumFeatures() const
{
    return _numFeatures;
}

int* FunctionApproximationUtils::getFeatures() const
{
    return _features;
}

int FunctionApproximationUtils::getNumNonZeroFeatures() const
{
    return _numNonZeroFeatures;
}

void FunctionApproximationUtils::fillFeatureVectorFromRAM(const ALERAM& ram)
{
    _numNonZeroFeatures = 0;

    // Get RAM bytes and convert them to bits which will represent our
    // binary features (we will save the position in which they appear
    // to avoid big products in Sarsa algorithm)
    for (int i = 0; i < RAM_LENGTH; ++i)
    {
        byte_t ramByte = ram.get(i);
               
        for (int j = 0; j < BYTE_LENGTH; ++j)
        {
            byte_t mask = 1 << j;
            byte_t masked_n = ramByte & mask;
            byte_t bit = masked_n >> j;
                       
            if (bit == 1)
            {
                int pos = BYTE_LENGTH * i + (BYTE_LENGTH - 1) - j;
                
                if (_nullFeatures.find(pos) == _nullFeatures.end())
                {
                    _features[_numNonZeroFeatures] = pos;
                    ++_numNonZeroFeatures;
                }
            }
        }
    }
    
    // We always use a final 1 at the end of the feature vector
    // to avoid 0 products if all RAM contains 0-bit
    if (_nullFeatures.find(_numFeatures - 1) == _nullFeatures.end())
    {
        _features[_numNonZeroFeatures] = _numFeatures - 1;
        ++_numNonZeroFeatures;
    }
    
    // Action dependent part and final 1 to avoid all-zeros
    /*for (int i = 0; i < _numAvailableActions; ++i)
    {
        // Action dependent part
        // If action == 0 --> add non-zero position 1024
        // If action == 1 --> add non-zero position 1025
        // If action == 17 --> add non-zero positions 1041
        _currentFeatures[i][_numNonZeroFeatures[i]] = RAM_LENGTH_BITS + i;
        ++_numNonZeroFeatures[i];

        // Final 1 present at all feature vectors
        _currentFeatures[i][_numNonZeroFeatures[i]] = _numFeatures - 1;
        ++_numNonZeroFeatures[i];
    }*/
    
    // Debug
    /*
    std::cout << "FEATURES" << std::endl;
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        std::cout << "Action # " << i << " :: Num Nonzero Features: " << _numNonZeroFeatures[i] << std::endl;
        
        for (int j = 0; j < _numNonZeroFeatures[i]; ++j)
        {
            std::cout << "Feature #" << j << " :: RAM Bit Position: " << _currentFeatures[i][j] << std::endl;
        }
    }
    */
}

void FunctionApproximationUtils::clearNullFeatures()
{
    _nullFeatures.clear();
}

void FunctionApproximationUtils::addNullFeature(int feature)
{
    _nullFeatures.insert(feature);
}

