#ifndef __PLAYER_AGENT_HPP__
#define __PLAYER_AGENT_HPP__

#include <ale_interface.hpp>
#include <common/Constants.h>
#include "AgentSettings.hpp"

/**
 * PlayerAgent
 * Class representing main functionalities of an agent
 */
class PlayerAgent
{
protected:
    // Total number of games with which the agent interacts
    int _numGames;

    // An interface to interact with the environment of each game
    ALEInterface** _gameInterfaces;

    // Currently selected game interface (e.g. needed in 
    // multigame learning)
    ALEInterface* _selectedGameInterface;

    // Method used to select the next game interface
    std::string _gameSelectionMethod;
    
    // Determines when the user has to end (e.g. needed in
    // multigame learning)
    std::string _agentEndMethod;
    
    // Attributes for game selection
    bool _isFirstSelectionDone;
    int _lastGameInterfaceIndex;

    // Number of actions that can be applied between states
    int _numAvailableActions;

    // Last reward that the agent got for applying an action
    reward_t _lastReward;
    
    // Use rewards -1 for negative rewards and 1 for positive rewards
    bool _useScaledRewards;
    
    // Frames for episode attributes
    int _maxNumFramesPerEpisode;
    int _currentEpisodeFrame;
    
    // Frame-skip: Number of frames during which an action is applied
    int _numFramesPerAction;
    
    // Current number of episode
    int _currentEpisode;
    
    // Whether to export the current frame as an image
    bool _exportFrameImages;
    std::string _exportFrameImagesRoute;

public:
    // Constructor
    PlayerAgent(const AgentSettings& settings);
    
    // Returns an specific agent depending on the field specified on the
    // AgentSettings instance
    static PlayerAgent* createPlayerAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~PlayerAgent();
    
    // Getters
    reward_t getLastReward() const;
    int getMaxNumFramesPerEpisode() const;
    int getNumFramesPerAction() const;
    int getCurrentEpisodeFrame() const;
    int getCurrentEpisode() const;
    
    reward_t act(Action a);
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();
    
    // Resets the environment with which the agent interacts
    virtual void agentReset();
    
    // Tells whether if the interaction with the environment has ended
    virtual bool hasAgentEnded();

    // Applies one action over the environment and saves the results in the
    // parameters passed as references
    void oneStepSimulation(int action, const ALEState& oldState, ALEState& newState, bool& isTerminal, double& reward);
    
    // Performs a full randomized simulation from the specified origin state
    // and returns the overall scored obtained from there
    double fullRandomizedSimulation(const ALEState& originState, int numSimulatedFrames);

protected:
    // To set the next game interface to use
    void selectGameInterface();
    void setRandomGameInterfaceIndex();
    void setNextAscendingGameInterfaceIndex();
    void setNextDescendingGameInterfaceIndex();
    
    // Tells whether at least one game has ended
    bool hasSomeGameEnded() const;
    
    // Tells whether all games have ended
    bool haveAllGamesEnded() const;
    
    // Saves the current frame as a PNG image
    void exportCurrentFrame() const;
};

#endif /* __PLAYER_AGENT_HPP__ */

