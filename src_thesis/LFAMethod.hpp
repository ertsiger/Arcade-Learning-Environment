#ifndef __LFA_METHOD_HPP__
#define __LFA_METHOD_HPP__

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"

/**
 * LFAMethod (Linear Function Approximation Method)
 * Generalization of all methods that use Linear Function
 * Approximation
 */
class LFAMethod
{
protected:
    // The agent using the method is saved to obtain some of
    // its features (e.g. current frame, current episode, ...)
    PlayerAgent* _playerAgent;
    
    // Number of features that will have our feature vectors
    int _numFeatures;
    
    // Number of applicable actions
    int _numAvailableActions;

    // Tells if using optimistic action values to encourage exploration
    // http://webdocs.cs.ualberta.ca/~machado/publications/optimistic.pdf
    bool _useOptimisticInitialization;

    // First non-zero reward used to perform optimistic initialization
    bool _firstNonZeroRewardSeen;
    double _firstNonZeroReward;

    // If true, state-action values will be normalized by dividing
    // their values by the number of non-zero features
    bool _normalizeStateActionValues;

    // Tells if the policy is frozen so as not to make updates on the
    // action values and the function parameters
    bool _policyFrozen;

    // Tells whether to export the functions at each episode
    bool _exportFunction;
    std::string _exportRoute;
    
    // Tells whether to import initial function parameters
    // for theta vector
    bool _importFunction;
    std::string _importRoute;

public:
    // Constructor
    LFAMethod(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions);
    
    // Destructor
    virtual ~LFAMethod();

    // Algorithm phases
    virtual int episodeStart(int* features, int numNonZeroFeatures);
    virtual int episodeStep(double reward, int* features, int numNonZeroFeatures);
    virtual void episodeEnd(double reward, int* features, int numNonZeroFeatures);

protected:
    // To select and action to return
    virtual int getNextAction(int* features, int numNonZeroFeatures) = 0;
    
    // Methods for managing the optimistic rewards
    void setFirstNonZeroReward(double reward);
    double getOptimisticReward(double gamma, double reward) const;
    double getOptimisticRewardEnd(double gamma, double reward, int timeDiff) const;

    // Updates the function for a given action
    void updateFunctionParams(double incr, int action, double** functionParams, int* features, int numNonZeroFeatures);

    // Sets all the function parameters to 0
    void clearFunctionParams(double** functionParams);

    // To compute the Q values for each action on the state 
    // specified by the features
    void computeStateActionValues(double* actionValues, double** functionParams, int* features, int numNonZeroFeatures);
    void computeStateActionValue(int action, double* actionValues, double** functionParams, int* features, int numNonZeroFeatures);
    
    // Returns the greedy action with probability 1 - epsilon and a
    // random action with probability epsilon
    int getEpsilonGreedyAction(double epsilon, double* actionValues) const;
    
    // Returns the action identifier for the action with the
    // highest Q-value
    int getGreedyAction(double* actionValues) const;
    
    // Save features
    void saveFeatures(int* toFeatures, int& toNumNonZeroFeatures, int* fromFeatures, int fromNumNonZeroFeatures);
    
    // To load/save the functions into/from file
    void loadFunctionParams(double** functionParams);
    void saveFunctionParams(const std::string& fileName, double** functionParams);
};

#endif /* __LFA_METHOD_HPP__ */

