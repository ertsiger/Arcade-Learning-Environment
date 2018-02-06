#ifndef __UCT_NODE_HPP__
#define __UCT_NODE_HPP__

#include <environment/ale_state.hpp>
#include <vector>

/**
 * UCTNode
 * Represents a node used by the UCT algorithm
 */
class UCTNode
{
private:
    // Action that brought us to this node
    int _action;

    // Action that brought us to a child node that we have to ignore when
    // deleting this node
    int _childActionNotDelete;

    // Current state of the Atari game
    ALEState _state;

    // Parent of this node (the one that executed the action that brought us to this node)
    UCTNode* _parent;

    // Tells if the state corresponding to this node is terminal
    bool _isTerminal;

    // Number of times that this node has been visited
    int _numVisits;
    int _numVisitsAMAF;

    // Average score that we have obtained during the diverse played simulations
    double _avgScore;
    double _avgScoreAMAF;

    // Children of the current node
    std::vector<UCTNode*> _children;

    // Actions not executed
    std::vector<int> _unappliedActions;

public:
    // Constructor
    UCTNode(int action, const ALEState& state, int numAvailableActions, UCTNode* parent, bool isTerminal);

    // Destructor
    virtual ~UCTNode();

    // Getters
    int getAction() const;
    int getChildActionNotDelete() const;
    UCTNode* getParent() const;
    int getNumVisits() const;
    int getNumVisitsAMAF() const;
    double getAvgScore() const;
    double getAvgScoreAMAF() const;
    const ALEState& getState() const; 
    bool isTerminal() const;
    bool areChildrenCreated() const;

    // Setters
    void setChildActionNotDelete(int action);
    void setParent(UCTNode* parent);

    // Returns whether the node is not terminal and not all of its actions
    // have been tried
    bool isExpandable() const;

    // Adds a new child to the child list
    void addChild(UCTNode* child);
    
    // Returns the child node with the highest score
    UCTNode* selectMaxChild(double exploreMultiplier) const;
    UCTNode* selectMaxChildAMAF(double exploreMultiplier, int raveParam) const;
    
    // Returns the most visited child node
    UCTNode* selectRobustChild() const;
    
    // Updates node's attributes
    void backup(double reward);
    void backupAMAF(double reward);
    void backupChildrenAMAF(double reward);
    
    // Select a random node that has not been tried yet
    UCTNode* selectRandomNodeWithUnappliedAction();

private:
    // Returns the selection ratio for the 'max child' selection
    double getSelectionRatio(UCTNode* child, double exploreMultiplier) const;
    double getSelectionRatioAMAF(UCTNode* child, double exploreMultiplier) const;
    
    // Returns the alpha parameter needed to balance between the
    // UCT counter and the AMAF counter
    double getAMAFAlpha(int raveParam) const;
};

#endif /* __UCT_NODE_HPP__ */

