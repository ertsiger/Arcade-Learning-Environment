#include <limits>
#include <sstream>

#include "common/random_tools.h"
#include "Sarsa.hpp"

Sarsa::Sarsa(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions)
: LFAMethod(playerAgent, settings, numFeatures, numAvailableActions), _lastFeatures(NULL)
{
    setAlgorithmSettings(settings);
    printAlgorithmSettings();
    
    if (_importFunction)
    {
        loadFunctionParams();
    }
}

Sarsa::~Sarsa()
{
    if (_actionValues != NULL)
    {
        delete [] _actionValues;
    }
    
    if (_functionParams != NULL)
    {
        for (int i = 0; i < _numAvailableActions; ++i)
        {
            delete [] _functionParams[i];
        }
        
        delete [] _functionParams;
    }
    
	if (_lastFeatures != NULL)
	{
	    delete [] _lastFeatures;
	}
}

int Sarsa::episodeStart(int* currentFeatures, int numNonZeroFeatures)
{
    LFAMethod::episodeStart(currentFeatures, numNonZeroFeatures);

    // Compute current Q(s,a) values
    computeStateActionValues(currentFeatures, numNonZeroFeatures);
    
    if (_policyFrozen)
    {
        _lastAction = getGreedyAction();
    }
    else
    {
        _lastAction = getEpsilonGreedyAction();
        saveFeatures(currentFeatures, numNonZeroFeatures);
    }
    
    return _lastAction;
}

int Sarsa::episodeStep(double reward, int* currentFeatures, int numNonZeroFeatures)
{
    // If the policy is frozen, the function parameters are not updated
    if (_policyFrozen)
    {
        computeStateActionValues(currentFeatures, numNonZeroFeatures);
        _lastAction = getGreedyAction();
    }
    else
    {
        double mreward = reward;
    
        if (_useOptimisticInitialization)
        {
            setFirstNonZeroReward(mreward);
            mreward = getOptimisticReward(_gammaFactor, mreward);
        }
    
        // From the previously computed action value, we compute delta (delta = reward - Q(s,a))
        double delta = mreward - _actionValues[_lastAction];

#ifdef __DEBUG
        // std::cout << "Before update: Q(s,a)=" << _actionValues[_lastAction] << ", r=" << reward << "\n";
#endif
        
        int currentAction = getNextAction(currentFeatures, numNonZeroFeatures);
        
        delta += _gammaFactor * _actionValues[currentAction];
        
        if (isnan(delta) || isinf(delta))
        {
            std::cout << "Error: Treating infinite numbers." << "\n";
            exit(1);
        }

#ifdef __DEBUG
        // std::cout << "Before update: Q(s',a')=" << _actionValues[currentAction] << ", delta=" << delta << "\n";
#endif
        
        updateFunctionParams(delta);

#ifdef __DEBUG
        // computeStateActionValues(_lastFeatures, _lastNumNonZeroFeatures);
        // std::cout << "After update: Q(s,a)=" << _actionValues[_lastAction] << "\n";
#endif
     
        saveFeatures(currentFeatures, numNonZeroFeatures);
        
        _lastAction = currentAction;
    }
    
    return _lastAction;
}

void Sarsa::episodeEnd(double reward, int* currentFeatures, int numNonZeroFeatures)
{
    if (!_policyFrozen)
    {
        double mreward = reward;
    
        if (_useOptimisticInitialization)
        {
            setFirstNonZeroReward(mreward);
            mreward = getOptimisticRewardEnd(_gammaFactor, mreward, _playerAgent->getMaxNumFramesPerEpisode() - _playerAgent->getCurrentEpisodeFrame());           
        }
    
        // Proceed as in agentStep but without getting a new action
        double delta = mreward - _actionValues[_lastAction];
        updateFunctionParams(delta);
    }
    
    if (_exportFunction)
    {
        std::stringstream ss;
        ss << "sarsa_" << _playerAgent->getCurrentEpisode() << ".txt";
        saveFunctionParams(ss.str());  
    }
}

void Sarsa::setAlgorithmSettings(const AgentSettings& settings)
{
    // Sarsa parameters
    _alphaFactor = settings.getFloat("sarsa_alpha", true);
    _epsilonFactor = settings.getFloat("sarsa_epsilon", true);
    _gammaFactor = settings.getFloat("sarsa_gamma", true);
    _lambdaFactor = settings.getFloat("sarsa_lambda", false);
    
    // Reserve memory for actionValues
    _actionValues = new double[_numAvailableActions];
    std::fill(_actionValues, _actionValues + _numAvailableActions, 0.0);
    
    // Vector of parameters
    _functionParams = new double*[_numAvailableActions];
    
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        _functionParams[i] = new double[_numFeatures];
        std::fill(_functionParams[i], _functionParams[i] + _numFeatures, 0.0);
    }
    
    // Last features
    _lastFeatures = new int[_numFeatures];
}

void Sarsa::printAlgorithmSettings() const
{
    std::cout << "Algorithm: Sarsa Lambda" << "\n";
    std::cout << "  Alpha: " << _alphaFactor << "\n";
    std::cout << "  Epsilon: " << _epsilonFactor << "\n";
    std::cout << "  Gamma: " << _gammaFactor << "\n";
    std::cout << "  Normalize State-Action Values: " << _normalizeStateActionValues << "\n";
    std::cout << "  Use Optimistic State-Action Values: " << _useOptimisticInitialization << "\n";
    std::cout << "  Policy frozen: " << _policyFrozen << "\n";
    
    std::cout << "  Export function: " << _exportFunction << "\n";
    
    if (_exportFunction)
    {
        std::cout << "  Export route: " << _exportRoute << "\n";
    }
    
    std::cout << "  Import function: " << _importFunction << "\n";
    
    if (_importFunction)
    {
        std::cout << "  Import route: " << _importRoute << "\n";
    }
}

int Sarsa::getNextAction(int* features, int numNonZeroFeatures)
{
    int currentAction = UNDEFINED;
    
    if (drand48() <= _epsilonFactor) // with probability _epsilonFactor
    {
        // Take a random action from the set of available actions
        // and update its Q-value
        currentAction = rand_range(0, _numAvailableActions - 1);
        computeStateActionValue(currentAction, features, numNonZeroFeatures);
    }
    else // with probability 1 - _epsilonFactor
    {
        // Update the Q-value for all actions and take the greedy one
        computeStateActionValues(features, numNonZeroFeatures);
        currentAction = getGreedyAction();
    }
    
    return currentAction;
}

void Sarsa::updateFunctionParams(double delta)
{
    double incr = _alphaFactor * delta;
    LFAMethod::updateFunctionParams(incr, _lastAction, _functionParams, _lastFeatures, _lastNumNonZeroFeatures);
}

void Sarsa::computeStateActionValues(int* features, int numNonZeroFeatures)
{
    LFAMethod::computeStateActionValues(_actionValues, _functionParams, features, numNonZeroFeatures);
}

void Sarsa::computeStateActionValue(int action, int* features, int numNonZeroFeatures)
{
    LFAMethod::computeStateActionValue(action, _actionValues, _functionParams, features, numNonZeroFeatures);
}

int Sarsa::getEpsilonGreedyAction() const
{
    return LFAMethod::getEpsilonGreedyAction(_epsilonFactor, _actionValues);
}

int Sarsa::getGreedyAction() const
{
    return LFAMethod::getGreedyAction(_actionValues);
}

void Sarsa::saveFeatures(int* features, int numNonZeroFeatures)
{
    LFAMethod::saveFeatures(_lastFeatures, _lastNumNonZeroFeatures, features, numNonZeroFeatures);
}

void Sarsa::loadFunctionParams()
{
    LFAMethod::loadFunctionParams(_functionParams);
}

void Sarsa::saveFunctionParams(const std::string& fileName)
{
    LFAMethod::saveFunctionParams(fileName, _functionParams);
}

