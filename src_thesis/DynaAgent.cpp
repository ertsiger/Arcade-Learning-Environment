#include "DynaAgent.hpp"

DynaAgent::DynaAgent(const AgentSettings& settings)
: PlayerAgent(settings)
{
    _faUtils = new FunctionApproximationUtils("ram_agent");
    _uctSearch = new UCTSearch(settings, this, _numAvailableActions);
    _dynaMemories = new DynaMemories(this, settings, _faUtils->getNumFeatures(), _numAvailableActions);
    _maxNumFramesSearch = settings.getInt("dyna_max_frames_search", true);
    _maxNumSearchIterations = settings.getInt("dyna_max_search_iterations", true);
}

DynaAgent::~DynaAgent()
{
    if (_faUtils != NULL)
    {
        delete _faUtils;
    }
    
    if (_uctSearch != NULL)
    {
        delete _uctSearch;
    }
    
    if (_dynaMemories != NULL)
    {
        delete _dynaMemories;
    }
}

void DynaAgent::agentStart()
{
    PlayerAgent::agentStart();
    
    _dynaMemories->clearTransientMemory();
    
    search();
    
    _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());
    
    int numAction = _dynaMemories->episodeStart(_faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];

    _lastReward = act(action);
}

void DynaAgent::agentStep()
{
    PlayerAgent::agentStep();
    
    search();
    
    _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());
    
    int numAction = _dynaMemories->episodeStep(_lastReward, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
    Action action = _selectedGameInterface->getLegalActionSet()[numAction];
    
    _lastReward = act(action);
}

void DynaAgent::agentEnd()
{
    PlayerAgent::agentEnd();
    
    _dynaMemories->episodeEnd(_lastReward, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());
}

void DynaAgent::search()
{
    // The initial state is got from the game to initialize the tree
    // and to restore it after the simulation
    const ALEState& initState = _selectedGameInterface->cloneState();
    bool isTerminal = _selectedGameInterface->game_over();
    
    // While time is available...
    for (int i = 0; i < _maxNumSearchIterations; ++i)
    {
        int initNumFrames = _selectedGameInterface->getFrameNumber();
        int diffNumFrames = 0;
    
        // Set the root of the tree
        _uctSearch->initializeSearch(initState, isTerminal);
        
        int numAction = _uctSearch->search();
        
        // Restore the state to apply the action in the following while loop
        _selectedGameInterface->restoreState(initState);
        
        // Get the features for the initial state and start the transient
        // memory with these features (needed to perform the Sarsa-like update)
        _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());
        _dynaMemories->startTransientMemory(numAction, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures());

        while (!_selectedGameInterface->game_over() && (diffNumFrames < _maxNumFramesSearch))
        {
            Action action = _selectedGameInterface->getLegalActionSet()[numAction];
            double reward = act(action);
            
#ifdef __DEBUG
            std::cout << "Action: " << action << " -- Reward: " << reward << "\n";
#endif

            // Avoid problems with terminal states when selecting max
            // child at UCT (the selected node might not have children
            // since it is terminal)
            if (!_selectedGameInterface->game_over())
            {
                const ALEState& prevState = _selectedGameInterface->cloneState();
                
                assert(prevState.equals(_uctSearch->getRootState()));

                numAction = _uctSearch->search();
                
                _selectedGameInterface->restoreState(prevState);
                
                // Update the transient memory with the features of the
                // new state and the last obtained reward
                _faUtils->fillFeatureVectorFromRAM(_selectedGameInterface->getRAM());
                _dynaMemories->updateTransientMemory(numAction, _faUtils->getFeatures(), _faUtils->getNumNonZeroFeatures(), reward);
                
                diffNumFrames = _selectedGameInterface->getFrameNumber() - initNumFrames;
            }
        }
    
        // Initial state must be restored in order to guarantee that next
        // actions are applied to the corresponding state
        _selectedGameInterface->restoreState(initState);
    }
    
#ifdef __DEBUG
    std::stringstream ss;
    ss << "dyna_t_" << _selectedGameInterface->getFrameNumber() << ".txt";
    _dynaMemories->saveTransientFunction(ss.str());
#endif
}

