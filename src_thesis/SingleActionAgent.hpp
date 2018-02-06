#ifndef __SINGLE_ACTION_AGENT_HPP__
#define __SINGLE_ACTION_AGENT_HPP__

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"

/**
 * SingleActionAgent
 * Agent that executes an specific action with probability 1 - _epsilon or
 * a random action with probability _epsilon
 */
class SingleActionAgent : public PlayerAgent
{
private:
    // Probability with which a random action is selected
    double _epsilon;
    
    // Action to always take with probability 1 - _epsilon
    Action _agentAction;    
    
public:
    // Constructor
    SingleActionAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~SingleActionAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();
    
private:
    // Selects an specific action or a random one depending on _epsilon
    Action getNextAction() const;
};

#endif /* __SINGLE_ACTION_AGENT_HPP__ */

