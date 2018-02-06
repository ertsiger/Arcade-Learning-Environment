#ifndef __DYNA_HPP__
#define __DYNA_HPP__

#include "AgentSettings.hpp"
#include "DynaMemories.hpp"
#include "FunctionApproximationUtils.hpp"
#include "PlayerAgent.hpp"
#include "UCTSearch.hpp"

/**
 * DynaAgent
 * Agent that implements the Dyna-2 architecture
 * Paper: "Sample-Based Learning and Search with Permanent and Transient Memories"
 */
class DynaAgent : public PlayerAgent
{
private:
    // To manage the obtention of feature vectors
    FunctionApproximationUtils* _faUtils;
    
    // To perform the search phase of Dyna
    UCTSearch* _uctSearch;
    
    // To manage the function approximation attributes for the
    // transient (search) and the permanent memory (learning)
    DynaMemories* _dynaMemories;
    
    // Maximum number of frames during which the UCT search is performed
    int _maxNumFramesSearch;
    
    // Maximum number of iterations of the search phase
    int _maxNumSearchIterations;

public:
    // Constructor
    DynaAgent(const AgentSettings& settings);
    
    // Destructor
    virtual ~DynaAgent();
    
    // Agent phases
    virtual void agentStart();
    virtual void agentStep();
    virtual void agentEnd();

private:
    // Search phase of the algorithm during which
    // transient parameters are updated
    void search();
};

#endif /* __DYNA_HPP__ */

