#ifndef __DYNA_MEMORIES_HPP__
#define __DYNA_MEMORIES_HPP__

#include <common/Constants.h>

#include "AgentSettings.hpp"
#include "LFAMethod.hpp"

/**
 * DynaMemories
 * Class that manages the memories that characterize the Dyna-2
 * architecture (used by DynaAgent)
 * @see LFAMethod: Most of the methods in DynaMemories use the
 * ones at LFAMethod
 */
class DynaMemories : public LFAMethod
{
private:
    // Paramerers for each of the memories
    double _alphaPermanent, _alphaTransient;
    double _epsilonPermanent, _epsilonTransient;
    double _lambdaPermanent, _lambdaTransient;    

    // Last action applied for each memory (necessary to
    // perform the Sarsa updates)
    int _lastPermanentAction;
    int _lastTransientAction;
    
    // Q(s,a) values for each memory
    double* _actionValuesPermanent;
    double* _actionValuesTransient;
    
    // A function for each action in each memory
    double** _functionPermanent;
    double** _functionTransient;
    
    // Last read permanent features
    int _numNonZeroPermanentFeatures;
    int* _permanentFeatures;

    // Last read permanent features    
    int _numNonZeroTransientFeatures;
    int* _transientFeatures;

public:
    // Constructor
    DynaMemories(PlayerAgent* playerAgent, const AgentSettings& settings, int numFeatures, int numAvailableActions);
    
    // Destructor
    virtual ~DynaMemories();
    
    // Algorithm phases
    int episodeStart(int* features, int numNonZeroFeatures);
    int episodeStep(double reward, int* features, int numNonZeroFeatures);
    void episodeEnd(double reward, int* features, int numNonZeroFeatures);
    
    // Initialize the transient memory with the first action applied and
    // the first seen features
    void startTransientMemory(int action, int* features, int numNonZeroFeatures);
    
    // Updates the transient memory with the features observed in a new state,
    // the action selected from that state and the previously obtained reward
    void updateTransientMemory(int action, int* features, int numNonZeroFeatures, double reward);
    
    // Sets all memory parameters as 0
    void clearTransientMemory();
    void clearPermanentMemory();

protected:
    // To set and print the parameters used by the algorithm
    void setAlgorithmSettings(const AgentSettings& settings);

    // Select next action to return to DynaAgent
    int getNextAction(int* features, int numNonZeroFeatures);

    void updatePermanentMemoryFunction(double delta);
    void updateTransientMemoryFunction(double delta);

    void computePermanentActionValues(int* features, int numNonZeroFeatures);
    void computePermanentActionValue(int action, int* features, int numNonZeroFeatures);
    
    void computeTransientActionValues(int* features, int numNonZeroFeatures);
    void computeTransientActionValue(int action, int* features, int numNonZeroFeatures);

    int getEpsilonGreedyAction() const;
    int getGreedyAction() const;
    
    void savePermanentFeatures(int* features, int numNonZeroFeatures);
    void saveTransientFeatures(int* features, int numNonZeroFeatures);
    
    void loadPermanentFunction();
    void savePermanentFunction(const std::string& fileName);
    
public:
    // Writes the current transient function into a file
    // Used from DynaAgent for debugging purposes
    void saveTransientFunction(const std::string& fileName);
};

#endif /* __DYNA_MEMORIES_HPP__ */

