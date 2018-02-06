#include "DynaMemories.hpp"
#include "common/random_tools.h"

DynaMemories::DynaMemories(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions)
: LFAMethod(playerAgent, settings, numFeatures, numAvailableActions)
{
    setAlgorithmSettings(settings);
    
    if (_importFunction)
    {
        loadPermanentFunction();
    }
}

DynaMemories::~DynaMemories()
{
    if (_actionValuesPermanent != NULL)
    {
        delete [] _actionValuesPermanent;
    }
    
    if (_actionValuesTransient != NULL)
    {
        delete [] _actionValuesTransient;
    }
    
    if (_permanentFeatures != NULL)
    {
        delete [] _permanentFeatures;
    }
    
    if (_transientFeatures != NULL)
    {
        delete [] _transientFeatures;
    }
    
    if (_functionPermanent != NULL)
    {
        for (int i = 0; i < _numAvailableActions; ++i)
        {
            delete [] _functionPermanent[i];
        }
        
        delete [] _functionPermanent;
    }

    if (_functionTransient != NULL)
    {
        for (int i = 0; i < _numAvailableActions; ++i)
        {
            delete [] _functionTransient[i];
        }
        
        delete [] _functionTransient;
    }
}

int DynaMemories::episodeStart(int* features, int numNonZeroFeatures)
{
    LFAMethod::episodeStart(features, numNonZeroFeatures);

    // Compute Q(s,a) for next step
    computePermanentActionValues(features, numNonZeroFeatures);

    // Q'(s,a) was set in the previous SEARCH phase, so it is not necessary to compute them
    _lastPermanentAction = getEpsilonGreedyAction();
    savePermanentFeatures(features, numNonZeroFeatures);
    
    return _lastPermanentAction;
}

int DynaMemories::episodeStep(double reward, int* features, int numNonZeroFeatures)
{
    double mreward = reward;
    
    if (_useOptimisticInitialization)
    {
        setFirstNonZeroReward(mreward);
        mreward = getOptimisticReward(1.0, mreward);
    }

    double delta = mreward - _actionValuesPermanent[_lastPermanentAction];
    
    // int currentAction = getNextAction(features, numNonZeroFeatures);
    computePermanentActionValues(features, numNonZeroFeatures);
    int currentAction = LFAMethod::getEpsilonGreedyAction(_epsilonPermanent, _actionValuesPermanent);
    
    delta += _actionValuesPermanent[currentAction];
    
    updatePermanentMemoryFunction(delta);
    
    savePermanentFeatures(features, numNonZeroFeatures);
    
    _lastPermanentAction = currentAction;

    return _lastPermanentAction;
}

void DynaMemories::episodeEnd(double reward, int* features, int numNonZeroFeatures)
{
    double mreward = reward;
    
    if (_useOptimisticInitialization)
    {
        setFirstNonZeroReward(mreward);
        mreward = getOptimisticRewardEnd(1.0, mreward, _playerAgent->getMaxNumFramesPerEpisode() - _playerAgent->getCurrentEpisodeFrame()); 
    }

    double delta = mreward - _actionValuesPermanent[_lastPermanentAction];
    updatePermanentMemoryFunction(delta);
    
    if (_exportFunction)
    {
        std::stringstream ss;
        ss << "dyna_p_" << _playerAgent->getCurrentEpisode() << ".txt";
        savePermanentFunction(ss.str()); 
    }
}

void DynaMemories::startTransientMemory(int action, int* features, int numNonZeroFeatures)
{
    _lastTransientAction = action;
    computeTransientActionValues(features, numNonZeroFeatures);
    saveTransientFeatures(features, numNonZeroFeatures);
}

void DynaMemories::updateTransientMemory(int action, int* features, int numNonZeroFeatures, double reward)
{
    double delta = reward - _actionValuesTransient[_lastTransientAction];
    
    // Similar to getNextAction of permanent memory but we have already an action, 
    // so compute directly the values
    computeTransientActionValues(features, numNonZeroFeatures);
    
    delta += _actionValuesTransient[action];
    
    updateTransientMemoryFunction(delta);
    
    saveTransientFeatures(features, numNonZeroFeatures);
    
    _lastTransientAction = action;
}

void DynaMemories::clearPermanentMemory()
{
    clearFunctionParams(_functionPermanent);
}

void DynaMemories::clearTransientMemory()
{
    clearFunctionParams(_functionTransient);
}

void DynaMemories::setAlgorithmSettings(const AgentSettings& settings)
{
    // Dyna parameters
    _alphaPermanent = settings.getFloat("dyna_p_alpha", true);
    _alphaTransient = settings.getFloat("dyna_t_alpha", true);
    
    _epsilonPermanent = settings.getFloat("dyna_p_epsilon", true);
    _epsilonTransient = settings.getFloat("dyna_t_epsilon", true);

    _lambdaPermanent = settings.getFloat("dyna_p_lambda", false);
    _lambdaTransient = settings.getFloat("dyna_t_lambda", false);

    // Q(s,a) values
    _actionValuesPermanent = new double[_numAvailableActions];
    _actionValuesTransient = new double[_numAvailableActions];
    
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        _actionValuesPermanent[i] = 0.0;
        _actionValuesTransient[i] = 0.0;
    }
    
    // Features for each memory
    _permanentFeatures = new int[_numFeatures];
    _transientFeatures = new int[_numFeatures];
    
    // A function for each action in each of the memories
    _functionPermanent = new double*[_numAvailableActions];
    _functionTransient = new double*[_numAvailableActions];
    
    for (int i = 0; i < _numAvailableActions; ++i)
    {
        _functionPermanent[i] = new double[_numFeatures];
        _functionTransient[i] = new double[_numFeatures];
        
        for (int j = 0; j < _numFeatures; ++j)
        {
            _functionPermanent[i][j] = 0.0;
            _functionTransient[i][j] = 0.0;
        }
    }
}

int DynaMemories::getNextAction(int* features, int numNonZeroFeatures)
{
    int currentAction = UNDEFINED;
    
    if (drand48() <= _epsilonPermanent) // with probability epsilon
    {
        // Take a random action from the set of available actions
        // and update its permanent Q-value
        currentAction = rand_range(0, _numAvailableActions - 1);
        computePermanentActionValue(currentAction, features, numNonZeroFeatures);
    }
    else // with probability 1 - epsilon
    {
        // Update the Q-value for all actions and take the greedy one
        computePermanentActionValues(features, numNonZeroFeatures);
        currentAction = getGreedyAction();
    }
    
    return currentAction;
}

void DynaMemories::updatePermanentMemoryFunction(double delta)
{
    double incr = _alphaPermanent * delta;
    updateFunctionParams(incr, _lastPermanentAction, _functionPermanent, _permanentFeatures, _numNonZeroPermanentFeatures);
}

void DynaMemories::updateTransientMemoryFunction(double delta)
{
    double incr = _alphaTransient * delta;
    updateFunctionParams(incr, _lastTransientAction, _functionTransient, _transientFeatures, _numNonZeroTransientFeatures);
}

void DynaMemories::computePermanentActionValues(int* features, int numNonZeroFeatures)
{
    computeStateActionValues(_actionValuesPermanent, _functionPermanent, features, numNonZeroFeatures);
}

void DynaMemories::computePermanentActionValue(int action, int* features, int numNonZeroFeatures)
{
    computeStateActionValue(action, _actionValuesPermanent, _functionPermanent, features, numNonZeroFeatures);
}

void DynaMemories::computeTransientActionValues(int* features, int numNonZeroFeatures)
{
    for (int a = 0; a < _numAvailableActions; ++a)
    {
    	computeTransientActionValue(a, features, numNonZeroFeatures);
    }
}

void DynaMemories::computeTransientActionValue(int action, int* features, int numNonZeroFeatures)
{
    // Compute the transient Q-values
    computeStateActionValue(action, _actionValuesTransient, _functionTransient, features, numNonZeroFeatures);
    
    // Sum the permanent Q-value to the transient one
    _actionValuesTransient[action] += _actionValuesPermanent[action];
}

int DynaMemories::getEpsilonGreedyAction() const
{
    return LFAMethod::getEpsilonGreedyAction(_epsilonPermanent, _actionValuesTransient);
}

int DynaMemories::getGreedyAction() const
{
    return LFAMethod::getGreedyAction(_actionValuesTransient);
}

void DynaMemories::savePermanentFeatures(int* features, int numNonZeroFeatures)
{
    saveFeatures(_permanentFeatures, _numNonZeroPermanentFeatures, features, numNonZeroFeatures);
}

void DynaMemories::saveTransientFeatures(int* features, int numNonZeroFeatures)
{
    saveFeatures(_transientFeatures, _numNonZeroTransientFeatures, features, numNonZeroFeatures);
}

void DynaMemories::loadPermanentFunction()
{
    loadFunctionParams(_functionPermanent);
}

void DynaMemories::savePermanentFunction(const std::string& fileName)
{
    saveFunctionParams(fileName, _functionPermanent);
}

void DynaMemories::saveTransientFunction(const std::string& fileName)
{
    saveFunctionParams(fileName, _functionTransient);
}

