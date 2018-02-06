#include "UCTSearch.hpp"

#include <ctime>

UCTSearch::UCTSearch(const AgentSettings& settings, PlayerAgent* searchAgent, int numAvailableActions)
: _searchAgent(searchAgent), _numAvailableActions(numAvailableActions), _treeRoot(NULL)
{
    _exploreMultiplier = settings.getFloat("uct_explore_multiplier", true);
    _numSimulationsPerNode = settings.getInt("uct_simulations_per_node", true);
    _selectionCriteria = settings.getString("uct_best_child_selection_criteria", false);
    _numSimulatedFrames = settings.getInt("uct_num_simulated_frames", true);
    
    _useAMAFSelection = settings.getBool("uct_use_amaf_selection", false);
    
    if (_useAMAFSelection)
    {
        _raveParam = settings.getInt("uct_rave_param", true);
    }
    
    _useDiscountFactor = settings.getBool("uct_use_discount_factor", false);
    
    if (_useDiscountFactor)
    {
        _discountFactor = settings.getFloat("uct_discount_factor", true);
    }
}

UCTSearch::~UCTSearch()
{
    if (_treeRoot != NULL)
    {
        delete _treeRoot;
    }
}

const ALEState& UCTSearch::getRootState() const
{
    return _treeRoot->getState();
}

void UCTSearch::initializeSearch(const ALEState& state, bool isTerminal)
{
    // Create root node v0 with state s0
    if (_treeRoot != NULL)
    {
        delete _treeRoot;
    }
    
    _treeRoot = new UCTNode(-1, state, _numAvailableActions, NULL, isTerminal);
}

int UCTSearch::search()
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    for (int i = 0; i < _numSimulationsPerNode; ++i)
    {
        UCTNode* simulationNode = treePolicy();
        double reward = defaultPolicy(simulationNode);    
        backup(simulationNode, reward);
    }
    
    UCTNode* selectedNode = selectBestRootChild();
    setChildNodeAsRoot(selectedNode);
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    std::cout << "Frame: " << _treeRoot->getState().getFrameNumber() << " - Time: " << difftime(t2, t1) << "\n";
#endif

    return selectedNode->getAction();
}

UCTNode* UCTSearch::treePolicy()
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    UCTNode* currentNode = _treeRoot;
    
    while (!currentNode->isTerminal())
    {
        if (currentNode->isExpandable())
        {
            currentNode = expand(currentNode);
            break;
        }
        else
        {
            currentNode = selectMaxChildFromNode(currentNode, _exploreMultiplier);
        }
    }
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Tree Policy - Time: " << difftime(t2, t1) << "\n";
#endif
    
    return currentNode;
}

UCTNode* UCTSearch::expand(UCTNode* node)
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    if (!node->areChildrenCreated())
    {
        createChildrenForNode(node);
    }

    // Get an untried action from the current node
    UCTNode* childNode = node->selectRandomNodeWithUnappliedAction();
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Expand - Time: " << difftime(t2, t1) << "\n";
#endif
    
    return childNode;
}

void UCTSearch::createChildrenForNode(UCTNode* node)
{
    const ALEState& oldState = node->getState();

    for (int a = 0; a < _numAvailableActions; ++a)
    {
        // Parameters to be set at the agent since it is who has access
        // to the simulation engine
        ALEState newState;
        bool isTerminal = false;
        double reward = 0.0;
        _searchAgent->oneStepSimulation(a, oldState, newState, isTerminal, reward);
        
        UCTNode* childNode = new UCTNode(a, newState, _numAvailableActions, node, isTerminal);
        node->addChild(childNode);
    }
}

double UCTSearch::defaultPolicy(UCTNode* simulationNode)
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    double score = _searchAgent->fullRandomizedSimulation(simulationNode->getState(), _numSimulatedFrames);
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Default Policy - Time: " << difftime(t2, t1) << "\n";
#endif

    return score;
}

void UCTSearch::backup(UCTNode* simulationNode, double reward)
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    UCTNode* currentNode = simulationNode;
    
    double mult = 1.0;
    
    while (currentNode != NULL)
    {
        UCTNode* parent = currentNode->getParent();
    
        double newReward = reward;
    
        if (_useDiscountFactor)
        {
            newReward *= mult;
        }
    
        currentNode->backup(newReward);
        
        if (_useAMAFSelection)
        {
            currentNode->backupChildrenAMAF(newReward);
            
            if (parent == NULL)
            {
                currentNode->backupAMAF(newReward);
            }
        }
        
        if (_useDiscountFactor)
        {
            mult *= _discountFactor;
        }
        
        currentNode = parent;
    }
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Backup - Time: " << difftime(t2, t1) << "\n";
#endif
}

UCTNode* UCTSearch::selectBestRootChild()
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    UCTNode* bestChild = NULL;

    if (_selectionCriteria == "max_child")
    {
        bestChild = selectMaxChildFromNode(_treeRoot, 0.0);
    }
    else if (_selectionCriteria == "robust_child")
    {
        bestChild = _treeRoot->selectRobustChild();
    }
    else // default
    {
        bestChild = selectMaxChildFromNode(_treeRoot, 0.0);
    }
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Select Best Child - Time: " << difftime(t2, t1) << "\n"; 
#endif
    
    return bestChild;
}

UCTNode* UCTSearch::selectMaxChildFromNode(UCTNode* node, double exploreMultiplier)
{
    UCTNode* child = NULL;
    
    if (_useAMAFSelection)
    {
        child = node->selectMaxChildAMAF(exploreMultiplier, _raveParam);
    }
    else
    {
        child = node->selectMaxChild(exploreMultiplier);
    }
    
    return child;
}

void UCTSearch::setChildNodeAsRoot(UCTNode* childNode)
{
#ifdef __DEBUG
    std::time_t t1 = std::time(NULL);
#endif

    assert(!_treeRoot->getState().equals(childNode->getState()));

    _treeRoot->setChildActionNotDelete(childNode->getAction());
    
    delete _treeRoot;
    
    childNode->setParent(NULL);
    
    _treeRoot = childNode;
    
#ifdef __DEBUG
    std::time_t t2 = std::time(NULL);
    if (difftime(t2, t1) > 0)
        std::cout << "Set Child Node As Root - Time: " << difftime(t2, t1) << "\n";
#endif
}

