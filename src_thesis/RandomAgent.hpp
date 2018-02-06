#ifndef __RANDOM_AGENT_HPP__
#define __RANDOM_AGENT_HPP__

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"

/**
 * RandomAgent
 * Agent that executes a random available action at each step
 */
class RandomAgent : public PlayerAgent
{
public:
    // Constructor
    RandomAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~RandomAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();
};

#endif /* __RANDOM_AGENT_HPP__ */

