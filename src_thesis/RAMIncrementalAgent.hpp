#ifndef __RAM_INCREMENTAL_AGENT_HPP__
#define __RAM_INCREMENTAL_AGENT_HPP__

#include <set>
#include "RAMAgent.hpp"

/**
 * RAMIncrementalAgent
 * Reinforcement Learning agent that uses Sarsa algorithm to learn
 * by building binary feature vectors from the Atari 2600 RAM
 * Unlike RAMAgent, it incrementally adds new features (i.e. does
 * not use all of them at the beginning)
 */
class RAMIncrementalAgent : public RAMAgent
{
private:
    // Division of features to be added in groups
    int _numFeatureGroups;
    std::vector<int>* _featureGroups;
    std::set<int> _nullFeatureGroups;
    
    //
    int _numFeatureChangeEpisodes;

    int _elapsedEpisodes;

public:
    // Constructor
    RAMIncrementalAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~RAMIncrementalAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();

private:
    void createNullFeatureGroups();
    void createFeatureGroups();
    int getGroupForFeature(int numFeaturesPerGroup);
    void putFeatureToGroup(int feature, int group);
    void setNullFeatures();
    void removeRandomNullFeatureGroup();
};

#endif /* __RAM_INCREMENTAL_AGENT_HPP__ */

