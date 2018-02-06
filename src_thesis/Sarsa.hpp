#ifndef __SARSA_HPP__
#define __SARSA_HPP__

#include <common/Constants.h>
#include <emucore/OSystem.hxx>

#include "AgentSettings.hpp"
#include "LFAMethod.hpp"
#include "PlayerAgent.hpp"

/**
 * Sarsa
 * Sarsa RL algorithm with linear function approximation
 * with the difference that it has a function for each action
 * http://webdocs.cs.ualberta.ca/~sutton/book/sarsaFA2.pdf
 */
class Sarsa : public LFAMethod
{
private:
    // Step factor of parameter function updating
    double _alphaFactor;

    // Probability with which we choose a random action
    double _epsilonFactor;
    
    // Weight that we give to the value of the next state-action pair Q(st+1,at+1)
    double _gammaFactor;
    
    // Factor to update eligibility traces
    double _lambdaFactor;

    // Array of action values Q(s,a) used to update the parameter vector
    double* _actionValues;
    
    // Vector 'theta' of parameters (one per action)
    double** _functionParams;
    
    // Last applied action (needed between steps)
    int _lastAction;
    
    // Last features (needed between steps)
    int* _lastFeatures;
    int _lastNumNonZeroFeatures;

public:
    // Constructor
    Sarsa(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions);
    
    // Destructor
    virtual ~Sarsa();
    
    // Algorithm phases
    int episodeStart(int* currentFeatures, int numNonZeroFeatures);
    int episodeStep(double reward, int* currentFeatures, int numNonZeroFeatures);
    void episodeEnd(double reward, int* currentFeatures, int numNonZeroFeatures);

protected:
    // To set and print the parameters used by the algorithm
    void setAlgorithmSettings(const AgentSettings& settings);
    void printAlgorithmSettings() const;

    // Returns the next action to be applied based on
    // an epsilon-greedy policy
    int getNextAction(int* features, int numNonZeroFeatures);
    
    void updateFunctionParams(double incr);
    
    void computeStateActionValues(int* features, int numNonZeroFeatures);
    void computeStateActionValue(int action, int* features, int numNonZeroFeatures);
    
    int getEpsilonGreedyAction() const;
    int getGreedyAction() const;
    
    void saveFeatures(int* features, int numNonZeroFeatures);
    
    void loadFunctionParams();
    void saveFunctionParams(const std::string& fileName);
};

#endif /* __SARSA_HPP__ */

