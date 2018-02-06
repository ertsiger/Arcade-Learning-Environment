#ifndef __UCT_SEARCH_HPP__
#define __UCT_SEARCH_HPP__

#include <environment/ale_state.hpp>

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"
#include "UCTNode.hpp"

/**
 * UCTSearch
 * UCT (Upper Confidence Tree) algorithm used to get the best action
 * to execute at each step depending on the current state at the root
 */
class UCTSearch
{
private:
    // Exploration factor used in computing the nodes score in order to
    // select the best of them during the tree policy
    double _exploreMultiplier;
    
    // Number of available actions to apply
    int _numAvailableActions;
    
    // SearchAgent must be accessible in order to perform the simulations
    PlayerAgent* _searchAgent;
    
    // Root of the search tree (the action to its best child will be selected)
    UCTNode* _treeRoot;
    
    // How many times the search loop will be executed per each root node
    int _numSimulationsPerNode;
    
    // Which method does the 'best' node define (most score, most visited)
    std::string _selectionCriteria;
    
    // Number of frames simulated in the 'default policy' phase
    int _numSimulatedFrames;
    
    // Perform All Moves As First heuristic with the Rapid Value Function
    // Estimation (RAVE) variant
    bool _useAMAFSelection;
    int _raveParam;
    
    // Discount factor to reduce the reward in the backup phase for each
    // node until the root of the tree is reached
    bool _useDiscountFactor;
    double _discountFactor;

public:
    // Constructor
    UCTSearch(const AgentSettings& settings, PlayerAgent* searchAgent, int numAvailableActions);
    
    // Destructor
    virtual ~UCTSearch();
    
    const ALEState& getRootState() const;
    
    // Creates the tree root with which the search will start
    void initializeSearch(const ALEState& state, bool isTerminal);

    // Returns the best action that can be selected from the root of the tree
    int search();
    
private:
    // Returns the child of the root over which the simulation will be perfomed
    UCTNode* treePolicy();
    
    // Creates a new child for the node specified by parameter which is obtained
    // by applying a random action on the environment
    UCTNode* expand(UCTNode* node);
    
    void createChildrenForNode(UCTNode* node);
    
    // Applies a full simulation over the parameter node and returns the overall
    // score of the simulation
    double defaultPolicy(UCTNode* simulationNode);
    
    // Updates the nodes properties from the simulation node upwards in the treePolicy
    // until the root node
    void backup(UCTNode* simulationNode, double reward);
    
    // Selects the best child of the root according to the _selectionCriteria
    UCTNode* selectBestRootChild();
    
    UCTNode* selectMaxChildFromNode(UCTNode* node, double exploreMultiplier);
    
    // Substitutes and erases the root node for the selected child node
    void setChildNodeAsRoot(UCTNode* childNode);
};

#endif /* __UCT_SEARCH_HPP__ */

