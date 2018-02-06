#include "SearchAgent.hpp"
#include "common/random_tools.h"

SearchAgent::SearchAgent(const AgentSettings& settings)
: PlayerAgent(settings)
{
    _uctSearch = new UCTSearch(settings, this, _numAvailableActions);
    _exportFrameImages = settings.getBool("export_frame_images");
    
    if (_exportFrameImages)
    {
        _exportFrameImagesRoute = settings.getString("export_frame_images_route", true);
    }
}

SearchAgent::~SearchAgent()
{
    if (_uctSearch != NULL)
    {
        delete _uctSearch;
    }
}

void SearchAgent::agentStart()
{
    PlayerAgent::agentStart();

    // Get the current state and whether it is terminal to
    // initialize the search
    const ALEState& state = _selectedGameInterface->cloneState();
    bool isTerminal = _selectedGameInterface->game_over();
    
    // Initialize the tree search for the upcoming steps
    _uctSearch->initializeSearch(state, isTerminal);
    
    int numAction = _uctSearch->search();
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];

    // The state cloned before must be restored to apply the
    // selected action
    _selectedGameInterface->restoreState(state);
    
    _lastReward = act(action);
}

void SearchAgent::agentStep()
{
    PlayerAgent::agentStep();

    // Get the current stateto restored it after the simulation
    const ALEState& state = _selectedGameInterface->cloneState();
    bool isTerminal = _selectedGameInterface->game_over();
    
    if (!state.equals(_uctSearch->getRootState()))
    {
    	_uctSearch->initializeSearch(state, isTerminal);
    	assert(state.equals(_uctSearch->getRootState()));
    }

    int numAction = _uctSearch->search();
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];
    
    // Restore the previous saved state
    _selectedGameInterface->restoreState(state);
    
    _lastReward = act(action);
}

void SearchAgent::agentEnd()
{
    PlayerAgent::agentEnd();
}

