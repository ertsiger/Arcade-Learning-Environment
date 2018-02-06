#ifndef __RAM_AGENT_HPP__
#define __RAM_AGENT_HPP__

#include "AgentSettings.hpp"
#include "FunctionApproximationUtils.hpp"
#include "PlayerAgent.hpp"
#include "Sarsa.hpp"

/**
 * RAMAgent
 * Reinforcement Learning agent that uses Sarsa algorithm to learn
 * by building binary feature vectors from the Atari 2600 RAM
 */
class RAMAgent : public PlayerAgent
{
protected:
    // To manage the obtention of feature vectors
    FunctionApproximationUtils* _faUtils;

    // Algorithm to be executed
    Sarsa* _sarsaAlgorithm;

public:
    // Constructor
    RAMAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~RAMAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();
};

#endif /* __RAM_AGENT_HPP__ */

