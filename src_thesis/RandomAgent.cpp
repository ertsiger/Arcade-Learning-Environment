#include "RandomAgent.hpp"
#include "common/random_tools.h"

RandomAgent::RandomAgent(const AgentSettings& settings)
: PlayerAgent(settings)
{
    // empty
}

RandomAgent::~RandomAgent()
{
    // empty
}

void RandomAgent::agentStart()
{
    PlayerAgent::agentStart();

    ActionVect legalActionSet = _selectedGameInterface->getLegalActionSet();
    Action a = choice(&legalActionSet);
    _lastReward = act(a);  
}

void RandomAgent::agentStep()
{
    PlayerAgent::agentStep();

    ActionVect legalActionSet = _selectedGameInterface->getLegalActionSet();
    Action a = choice(&legalActionSet);
    _lastReward = act(a);  
}

void RandomAgent::agentEnd()
{
    PlayerAgent::agentEnd();
}

