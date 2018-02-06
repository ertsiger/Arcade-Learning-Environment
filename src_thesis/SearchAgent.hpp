#ifndef __SEARCH_AGENT_HPP__
#define __SEARCH_AGENT_HPP__

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"
#include "UCTSearch.hpp"

/**
 * SearchAgent
 * Agent that performs a search over the state space of the game
 */
class SearchAgent : public PlayerAgent
{
protected:   
    // Algorithm to execute
    UCTSearch* _uctSearch;

public:
    // Constructor
    SearchAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~SearchAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();
};

#endif /* __SEARCH_AGENT_HPP__ */

