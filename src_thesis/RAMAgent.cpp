#include "RAMAgent.hpp"

RAMAgent::RAMAgent(const AgentSettings& settings)
: PlayerAgent(settings)
{
    _faUtils = new FunctionApproximationUtils("ram_agent");
    _sarsaAlgorithm = new Sarsa(this, settings, _faUtils->getNumFeatures(), _numAvailableActions);
}

RAMAgent::~RAMAgent()
{
    if (_faUtils != NULL)
    {
        delete _faUtils;
    }

    if (_sarsaAlgorithm != NULL)
    {
        delete _sarsaAlgorithm;
    }
}

void RAMAgent::agentStart()
{
    PlayerAgent::agentStart();
    
    _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());

    int numAction = _sarsaAlgorithm->episodeStart(_faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
    
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];
    _lastReward = act(action);
}

void RAMAgent::agentStep()
{
    PlayerAgent::agentStep();
    
    _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());
    
    int numAction = _sarsaAlgorithm->episodeStep(_lastReward, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
    
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];
    _lastReward = act(action);
}

void RAMAgent::agentEnd()
{
    PlayerAgent::agentEnd();
    
    _sarsaAlgorithm->episodeEnd(_lastReward, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
}

