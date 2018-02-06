#include "SingleActionAgent.hpp"
#include "common/random_tools.h"

SingleActionAgent::SingleActionAgent(const AgentSettings& settings)
: PlayerAgent(settings)
{
    _epsilon = settings.getFloat("agent_epsilon", true);
    _agentAction = (Action)settings.getInt("agent_action", true);
}

SingleActionAgent::~SingleActionAgent()
{
    // empty
}

void SingleActionAgent::agentStart()
{
    PlayerAgent::agentStart();
    
    Action a = getNextAction();  
    _lastReward = act(a);
}

void SingleActionAgent::agentStep()
{
    PlayerAgent::agentStep();
    
    Action a = getNextAction();
    _lastReward = act(a);
}

void SingleActionAgent::agentEnd()
{
    PlayerAgent::agentEnd();
}

Action SingleActionAgent::getNextAction() const
{
    Action a;
    
    if (drand48() <= _epsilon) // with probability _epsilon
    {
        ActionVect legalActionSet = _selectedGameInterface->getLegalActionSet();
        a = choice(&legalActionSet); 
    }
    else // with probability 1 - _epsilon
    {
        a = _agentAction;
    }
    
    return a;
}

