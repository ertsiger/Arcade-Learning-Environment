#include "UCTNode.hpp"
#include "common/random_tools.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>

UCTNode::UCTNode(int action, const ALEState& state, int numAvailableActions, UCTNode* parent, bool isTerminal)
: _action(action), _childActionNotDelete(-1), _state(state), _parent(parent), _isTerminal(isTerminal), _numVisits(0), _numVisitsAMAF(0), _avgScore(0.0), _avgScoreAMAF(0.0)
{
    _unappliedActions.resize(numAvailableActions);

    for (int i = 0; i < numAvailableActions; ++i)
    {
        _unappliedActions[i] = i;
    }
}

UCTNode::~UCTNode()
{
    for (int i = 0; i < _children.size(); ++i)
    {
        if (_children[i]->getAction() != _childActionNotDelete)
        {
            delete _children[i];
        }
    }
}

int UCTNode::getAction() const
{
    return _action;
}

int UCTNode::getChildActionNotDelete() const
{
    return _childActionNotDelete;
}

int UCTNode::getNumVisits() const
{
    return _numVisits;
}

int UCTNode::getNumVisitsAMAF() const
{
    return _numVisitsAMAF;
}

UCTNode* UCTNode::getParent() const
{
    return _parent;
}

double UCTNode::getAvgScore() const
{
    return _avgScore;
}

double UCTNode::getAvgScoreAMAF() const
{
    return _avgScoreAMAF;
}

const ALEState& UCTNode::getState() const
{
    return _state;
}

bool UCTNode::isTerminal() const
{
    return _isTerminal;
}

bool UCTNode::areChildrenCreated() const
{
    return !_children.empty();
}

void UCTNode::setChildActionNotDelete(int action)
{
    _childActionNotDelete = action;
}

void UCTNode::setParent(UCTNode* parent)
{
    _parent = parent;
}

bool UCTNode::isExpandable() const
{
    return !_isTerminal && !_unappliedActions.empty();
}

void UCTNode::addChild(UCTNode* child)
{
    _children.push_back(child);
}

UCTNode* UCTNode::selectMaxChild(double exploreMultiplier) const
{
    assert(!_isTerminal);
    assert(!_children.empty());

    double childrenRatios[_children.size()];

    for (int i = 0; i < _children.size(); ++i)
    {
        UCTNode* child = _children[i];
        double ratio = getSelectionRatio(child, exploreMultiplier);
        childrenRatios[i] = ratio;
    }

    int maxChildrenIndex = getMaxElementIndex(_children.size(), childrenRatios);

    return _children[maxChildrenIndex];
}

UCTNode* UCTNode::selectMaxChildAMAF(double exploreMultiplier, int raveParam) const
{
    assert(!_isTerminal);
    assert(!_children.empty());

    double childrenRatios[_children.size()];

    for (int i = 0; i < _children.size(); ++i)
    {
        UCTNode* child = _children[i];
        
        double uctRatio = getSelectionRatio(child, exploreMultiplier);
        double amafRatio = getSelectionRatioAMAF(child, exploreMultiplier);
        double alpha = child->getAMAFAlpha(raveParam);
        
#ifdef __DEBUG
        // std::cout << "AMAF alpha: " << alpha << "\n";
#endif

        childrenRatios[i] = alpha * amafRatio + (1.0 - alpha) * uctRatio;
    }

    int maxChildrenIndex = getMaxElementIndex(_children.size(), childrenRatios);

    return _children[maxChildrenIndex];
}

UCTNode* UCTNode::selectRobustChild() const
{
    assert(!_isTerminal);
    assert(!_children.empty());

    int childrenVisits[_children.size()];

    for (int i = 0; i < _children.size(); ++i)
    {
        UCTNode* child = _children[i];
        childrenVisits[i] = child->getNumVisits();
    }

    int maxChildrenIndex = getMaxElementIndex(_children.size(), childrenVisits);

    return _children[maxChildrenIndex];
}

void UCTNode::backup(double reward)
{
    ++_numVisits;
    
    double incr = (reward - _avgScore) / (double)_numVisits;
    _avgScore += incr;
}

void UCTNode::backupAMAF(double reward)
{
    ++_numVisitsAMAF;
    
    double incr = (reward - _avgScoreAMAF) / (double)_numVisitsAMAF;
    _avgScoreAMAF += incr;
}

void UCTNode::backupChildrenAMAF(double reward)
{
    for (int i = 0; i < _children.size(); ++i)
    {
        _children[i]->backupAMAF(reward);
    }
}

UCTNode* UCTNode::selectRandomNodeWithUnappliedAction()
{
    int randPos = rand() % _unappliedActions.size();
    int action = _unappliedActions[randPos];
        
    std::vector<int>::iterator it = std::lower_bound(_unappliedActions.begin(), _unappliedActions.end(), action);
    assert(*it == action);
    _unappliedActions.erase(it);

    UCTNode* selectedNode = _children[action];
    assert(action == selectedNode->getAction());
    
    return selectedNode;
}

double UCTNode::getSelectionRatio(UCTNode* child, double exploreMultiplier) const
{
    double ratio = 0.0;
    ratio = child->getAvgScore();
    ratio += exploreMultiplier * sqrt(2.0 * log(_numVisits) / (double)child->getNumVisits());

    return ratio;
}

double UCTNode::getSelectionRatioAMAF(UCTNode* child, double exploreMultiplier) const
{
    double ratio = 0.0;
    ratio = child->getAvgScoreAMAF();
    ratio += exploreMultiplier * sqrt(2.0 * log(_numVisitsAMAF) / (double)child->getNumVisitsAMAF());

    return ratio;
}

double UCTNode::getAMAFAlpha(int raveParam) const
{
    double alpha = raveParam - _numVisitsAMAF;
    alpha /= (double)raveParam;
    
    return std::max(0.0, alpha);
}

