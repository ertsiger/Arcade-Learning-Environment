#include "PlayerAgent.hpp"
#include "DynaAgent.hpp"
#include "RAMAgent.hpp"
#include "RAMIncrementalAgent.hpp"
#include "RandomAgent.hpp"
#include "SearchAgent.hpp"
#include "SingleActionAgent.hpp"
#include "common/random_tools.h"

PlayerAgent::PlayerAgent(const AgentSettings& settings)
: _lastReward(0), _isFirstSelectionDone(false)
{
    bool displayScreen = settings.getBool("display_screen");
    bool useEnvironmentDistribution = settings.getBool("use_environment_distribution");
    
    _maxNumFramesPerEpisode = settings.getInt("max_num_frames_per_episode");
    _numFramesPerAction = settings.getInt("frame_skip");
    _currentEpisodeFrame = 0;
    _currentEpisode = 0;

    _numGames = settings.getInt("num_games", true);
    
    if (_numGames < 1)
    {
        std::cout << "Error: At least one game must be defined." << "\n";
        exit(1);
    }
    
    _gameInterfaces = new ALEInterface*[_numGames];
    
    for (int i = 0; i < _numGames; ++i)
    {
    	std::stringstream romName;
    	romName << "rom_file_" << i;

        std::string rom = settings.getString(romName.str(), true);
        _gameInterfaces[i] = new ALEInterface(displayScreen/*, useEnvironmentDistribution*/);
        _gameInterfaces[i]->loadROM(rom);
    }
    
    _gameSelectionMethod = settings.getString("game_selection_method");
    _agentEndMethod = settings.getString("agent_end_method");
    
    // Assume all games have the same actions
    _numAvailableActions = _gameInterfaces[0]->getLegalActionSet().size();
    
    _useScaledRewards = settings.getBool("use_scaled_rewards");
    
    _exportFrameImages = settings.getBool("export_frame_images");
    
    if (_exportFrameImages)
    {
        _exportFrameImagesRoute = settings.getString("export_frame_images_route");
    }
}

PlayerAgent* PlayerAgent::createPlayerAgent(const AgentSettings& settings)
{
    const std::string& playerAgent = settings.getString("player_agent");
    
    if (playerAgent == "dyna_agent")
    {
        return new DynaAgent(settings);
    }
    else if (playerAgent == "ram_agent")
    {
        return new RAMAgent(settings);
    }
    else if (playerAgent == "ram_incremental_agent")
    {
        return new RAMIncrementalAgent(settings);
    }
    else if (playerAgent == "random_agent")
    {
        return new RandomAgent(settings);
    }
    else if (playerAgent == "search_agent")
    {
        return new SearchAgent(settings);
    }
    else if (playerAgent == "single_action_agent")
    {
        return new SingleActionAgent(settings);
    }
    else
    {
        std::cerr << "Error: Agent \'" << playerAgent << "\' does not exist." << std::endl;
        exit(1);
    }
    
    return NULL;
}

PlayerAgent::~PlayerAgent()
{
    if (_gameInterfaces != NULL)
    {
        for (int i = 0; i < _numGames; ++i)
        {
            delete _gameInterfaces[i];
        }
        
        delete [] _gameInterfaces;
    }
}

reward_t PlayerAgent::getLastReward() const
{
    return _lastReward;
}

int PlayerAgent::getMaxNumFramesPerEpisode() const
{
    return _maxNumFramesPerEpisode;
}

int PlayerAgent::getNumFramesPerAction() const
{
    return _numFramesPerAction;
}

int PlayerAgent::getCurrentEpisodeFrame() const
{
    return _currentEpisodeFrame;
}

int PlayerAgent::getCurrentEpisode() const
{
    return _currentEpisode;
}

reward_t PlayerAgent::act(Action a)
{
    reward_t reward = 0;

    for (int i = 0; i < _numFramesPerAction; ++i)
    {
        reward += _selectedGameInterface->act(a);
    }
    
    if (_useScaledRewards)
    {
        if (reward > 0)
        {
            reward = 1;
        }
        else if (reward < 0)
        {
            reward = -1;
        }
    }
    
    return reward;
}

void PlayerAgent::agentStart()
{
    selectGameInterface();
    _currentEpisodeFrame = 0;
    
    if (_exportFrameImages)
    {
        exportCurrentFrame();
    }
}

void PlayerAgent::agentStep()
{
    // Uncomment to perform action-steps
    //selectGameInterface();
    
    _currentEpisodeFrame += _numFramesPerAction;
    
    if (_exportFrameImages)
    {
        exportCurrentFrame();
    }
}

void PlayerAgent::agentEnd()
{
    ++_currentEpisode;

    if (_exportFrameImages)
    {
        exportCurrentFrame();
    }
}

void PlayerAgent::agentReset()
{
    for (int i = 0; i < _numGames; ++i)
    {
        _gameInterfaces[i]->reset_game();
    }
    
    // Uncomment to perform action-steps
    //_isFirstSelectionDone = false;
}

bool PlayerAgent::hasAgentEnded()
{
    if (_maxNumFramesPerEpisode > 0 && _currentEpisodeFrame >= _maxNumFramesPerEpisode)
    {
        return true;
    }
    else if (_agentEndMethod == "some_game")
    {
        return hasSomeGameEnded();
    }
    else if (_agentEndMethod == "all_games")
    {
        return haveAllGamesEnded();
    }

    // default
    return hasSomeGameEnded();
}

void PlayerAgent::oneStepSimulation(int action, const ALEState& oldState, ALEState& newState, bool& isTerminal, double& reward)
{
    // Restore the current state since we do not know if it is the current
    // node being simulated
    _selectedGameInterface->restoreState(oldState);

    // Perfom action over the restored state and get the reward
    Action a = _selectedGameInterface->getLegalActionSet()[action];

    reward = act(a);
    
    // Get other attributes
    isTerminal = _selectedGameInterface->game_over();
    newState = _selectedGameInterface->cloneState();
}

double PlayerAgent::fullRandomizedSimulation(const ALEState& originState, int simulatedFrames)
{
    // Set the origin state to start the random simulation
    _selectedGameInterface->restoreState(originState);
    
    int initNumFrames = _selectedGameInterface->getFrameNumber();
    int diffNumFrames = 0;
    
    double reward = 0.0;
    
    while (!_selectedGameInterface->game_over() && (diffNumFrames < simulatedFrames))
    {
        ActionVect legalActionSet = _selectedGameInterface->getLegalActionSet();
    	Action randomAction = choice(&legalActionSet);
    	
		reward += act(randomAction);
    	
        diffNumFrames = _selectedGameInterface->getFrameNumber() - initNumFrames;
    }
    
    return reward;
}

void PlayerAgent::selectGameInterface()
{
    if (_numGames == 1)
    {
        _lastGameInterfaceIndex = 0;
    }
    else if (_gameSelectionMethod == "random")
    {
        setRandomGameInterfaceIndex();
    }
    else if (_gameSelectionMethod == "ascending_order")
    {
        setNextAscendingGameInterfaceIndex();
    }
    else if (_gameSelectionMethod == "descending_order")
    {
        setNextDescendingGameInterfaceIndex();
    }
    else
    {
        setRandomGameInterfaceIndex();
    }
    
    _selectedGameInterface = _gameInterfaces[_lastGameInterfaceIndex];

    std::cout << "Game: " << _lastGameInterfaceIndex << std::endl;
}

void PlayerAgent::setRandomGameInterfaceIndex()
{
    if (_agentEndMethod == "all_games")
    {
        std::vector<int> notEndedGames;
    
        for (int i = 0; i < _numGames; ++i)
        {
            if (!_gameInterfaces[i]->game_over())
            {
                notEndedGames.push_back(i);
            }
        }
        
        _lastGameInterfaceIndex = choice(&notEndedGames);
    }
    else
    {
        _lastGameInterfaceIndex = rand_range(0, _numGames - 1);
    }
}

void PlayerAgent::setNextAscendingGameInterfaceIndex()
{
    if (_isFirstSelectionDone)
    {
        _lastGameInterfaceIndex = (_lastGameInterfaceIndex + 1) % _numGames;
        
        while (_gameInterfaces[_lastGameInterfaceIndex]->game_over())
        {
            _lastGameInterfaceIndex = (_lastGameInterfaceIndex + 1) % _numGames;
        }
    }
    else
    {
        _lastGameInterfaceIndex = 0;
        _isFirstSelectionDone = true;
    }
}

void PlayerAgent::setNextDescendingGameInterfaceIndex()
{
    if (_isFirstSelectionDone)
    {
        _lastGameInterfaceIndex = (_lastGameInterfaceIndex - 1 + _numGames) % _numGames;
        
        while (_gameInterfaces[_lastGameInterfaceIndex]->game_over())
        {
            _lastGameInterfaceIndex = (_lastGameInterfaceIndex - 1 + _numGames) % _numGames;
        }
    }
    else
    {
        _lastGameInterfaceIndex = _numGames - 1;
        _isFirstSelectionDone = true;
    }
}

bool PlayerAgent::hasSomeGameEnded() const
{
    for (int i = 0; i < _numGames; ++i)
    {
        if (_gameInterfaces[i]->game_over())
        {
            return true;
        }
    }
    
    return false;
}
    
bool PlayerAgent::haveAllGamesEnded() const
{
    for (int i = 0; i < _numGames; ++i)
    {
        if (!_gameInterfaces[i]->game_over())
        {
            return false;
        }
    }
    
    return true;
}

void PlayerAgent::exportCurrentFrame() const
{
    std::stringstream imgFile;
    imgFile << _exportFrameImagesRoute;
    imgFile << setfill('0') << setw(6);
    imgFile << _selectedGameInterface->getFrameNumber();
    imgFile << ".png";
    
   _selectedGameInterface->saveScreenPNG(imgFile.str());
}

