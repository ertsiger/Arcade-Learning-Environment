#include "LFAMethod.hpp"
#include "common/random_tools.h"

LFAMethod::LFAMethod(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions)
: _playerAgent(playerAgent), _numFeatures(numFeatures), _numAvailableActions(numAvailableActions)
{
    // Optimistic initialization parameters
    _useOptimisticInitialization = settings.getBool("lfa_optimistic_initialization", false);
    _firstNonZeroRewardSeen = false;
    _firstNonZeroReward = 0.0;

    // Normalize Q-values
    _normalizeStateActionValues = settings.getBool("lfa_normalize", true);
    
    // Do not update function parameters
    _policyFrozen = settings.getBool("lfa_policy_frozen", false);
    
    // Function exportation
    _exportFunction = settings.getBool("lfa_export_function", false);
    
    if (_exportFunction)
    {
        _exportRoute = settings.getString("lfa_export_route", true);
    }
    
    // Function importation
    _importFunction = settings.getBool("lfa_import_function", false);
    
    if (_importFunction)
    {
        _importRoute = settings.getString("lfa_import_route", true);
    }
}

LFAMethod::~LFAMethod()
{
    // empty
}

int LFAMethod::episodeStart(int* features, int numNonZeroFeatures)
{
    if (_useOptimisticInitialization)
    {
        _firstNonZeroRewardSeen = false;
        _firstNonZeroReward = 0.0;
    }
    
    return -1;
}

int LFAMethod::episodeStep(double reward, int* features, int numNonZeroFeatures)
{
    return -1;
}

void LFAMethod::episodeEnd(double reward, int* features, int numNonZeroFeatures)
{
	// empty
}

void LFAMethod::setFirstNonZeroReward(double reward)
{
    if (!_firstNonZeroRewardSeen && reward > 0.0)
    {
        _firstNonZeroRewardSeen = true;
        _firstNonZeroReward = std::abs(reward);
    }
}

double LFAMethod::getOptimisticReward(double gamma, double reward) const
{
    double optReward = 0.0;

    if (_firstNonZeroRewardSeen)
    {
        optReward = reward / _firstNonZeroReward + (gamma - 1.0);
    }
    else
    {
    	optReward = gamma - 1.0;
    }
    
    return optReward;
}

double LFAMethod::getOptimisticRewardEnd(double gamma, double reward, int timeDiff) const
{
    double optReward = getOptimisticReward(gamma, reward);
    optReward -= std::pow(gamma, timeDiff + 1) - 1.0;
    
    return optReward;
}

void LFAMethod::updateFunctionParams(double incr, int action, double** functionParams, int* features, int numNonZeroFeatures)
{
    double* actionFunctionParams = functionParams[action];
    
    // As eligibility traces are not used, the increment will be added
    // in all those positions in which the feature is non-zero
    for (int i = 0; i < numNonZeroFeatures; ++i)
    {
        int pos = features[i];
        actionFunctionParams[pos] += incr;
    }
}

void LFAMethod::clearFunctionParams(double** functionParams)
{
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        std::fill(functionParams[i], functionParams[i] + _numFeatures, 0.0);
    }
}

void LFAMethod::computeStateActionValues(double* actionValues, double** functionParams, int* features, int numNonZeroFeatures)
{
    for (int a = 0; a < _numAvailableActions; ++a)
    {
        computeStateActionValue(a, actionValues, functionParams, features, numNonZeroFeatures);
    }
}

void LFAMethod::computeStateActionValue(int action, double* actionValues, double** functionParams, int* features, int numNonZeroFeatures)
{
    actionValues[action] = 0.0;
    
    double* actionFunctionParams = functionParams[action];
    
    // In linear function approximation the Q-values are a dot product 
    // between the feature vector and the theta (function) vector
    for (int i = 0; i < numNonZeroFeatures; ++i)
    {
        int pos = features[i];
        actionValues[action] += actionFunctionParams[pos];
    }
    
    // Normalize so as not to overflow double type capacity in
    // next iterations
    if (_normalizeStateActionValues && numNonZeroFeatures != 0)
    {
        actionValues[action] /= numNonZeroFeatures;
    }
}

int LFAMethod::getEpsilonGreedyAction(double epsilon, double* actionValues) const
{
    if (drand48() <= epsilon) // with probability epsilon
    {
        return rand_range(0, _numAvailableActions - 1);
    }
    else // with probability 1 - epsilon
    {
        return getGreedyAction(actionValues);
    }
}

int LFAMethod::getGreedyAction(double* actionValues) const
{
    return getMaxElementIndex(_numAvailableActions, actionValues);
}

void LFAMethod::saveFeatures(int* toFeatures, int& toNumNonZeroFeatures, int* fromFeatures, int fromNumNonZeroFeatures)
{
    for (int i = 0; i < fromNumNonZeroFeatures; ++i)
    {
        toFeatures[i] = fromFeatures[i];
    }
    
    toNumNonZeroFeatures = fromNumNonZeroFeatures;
}

void LFAMethod::loadFunctionParams(double** functionParams)
{
    std::ifstream inputFile;
    inputFile.open(_importRoute.c_str());
    
    if (!inputFile.is_open())
    {
        std::cout << "Error: Could not open file \'" << _importRoute << "\'\n";
        exit(1);
    }
    
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        double* actionFunctionParams = functionParams[i];
        
        for (int j = 0; j < _numFeatures; ++j)
        {
            inputFile >> actionFunctionParams[j];
        }
    }
    
    inputFile.close();
}

void LFAMethod::saveFunctionParams(const std::string& fileName, double** functionParams)
{
    const std::string fullPath = _exportRoute + fileName;

    std::ofstream outputFile;
    outputFile.open(fullPath.c_str());
    
    if (!outputFile.is_open())
    {
        std::cout << "Error: Could not create file \'" << fileName << "\'\n";
        exit(1);
    }
    
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        double* actionFunctionParams = functionParams[i];
        
        for (int j = 0; j < _numFeatures; ++j)
        {
            outputFile << actionFunctionParams[j] << "\n";
        }
    }
    
    outputFile.close();
}

