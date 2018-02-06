#include "RAMIncrementalAgent.hpp"
#include "common/random_tools.h"

RAMIncrementalAgent::RAMIncrementalAgent(const AgentSettings& settings)
: RAMAgent(settings), _elapsedEpisodes(0)
{
    _numFeatureGroups = settings.getInt("num_feature_groups");
    _featureGroups = new vector<int>[_numFeatureGroups];
    createFeatureGroups();
    createNullFeatureGroups();
    _numFeatureChangeEpisodes = settings.getInt("num_feature_change_episodes");
}

RAMIncrementalAgent::~RAMIncrementalAgent()
{
    if (_featureGroups != NULL)
    {
        delete _featureGroups;
    }
}

void RAMIncrementalAgent::agentStart()
{
    if ((_elapsedEpisodes == 0) || (_elapsedEpisodes == _numFeatureChangeEpisodes))
    {
        removeRandomNullFeatureGroup();
        setNullFeatures();
        _elapsedEpisodes = 0;
    }

    RAMAgent::agentStart();
}

void RAMIncrementalAgent::agentStep()
{
    RAMAgent::agentStep();
}

void RAMIncrementalAgent::agentEnd()
{
    RAMAgent::agentEnd();
    
    ++_elapsedEpisodes;
}

void RAMIncrementalAgent::createNullFeatureGroups()
{
    _nullFeatureGroups.clear();

    for (int i = 0; i < _numFeatureGroups; ++i)
    {
        _nullFeatureGroups.insert(i);
    }
}

void RAMIncrementalAgent::createFeatureGroups()
{
    for (int i = 0; i < _numFeatureGroups; ++i)
    {
        _featureGroups[i].clear();
    }

    int numFeatures = _faUtils->getNumFeatures();
    int numFeaturesPerGroup = numFeatures / _numFeatureGroups;
    
    for (int i = 0; i < numFeatures; ++i)
    {
        int group = getGroupForFeature(numFeaturesPerGroup);
        putFeatureToGroup(i, group);
    }
    
    /* -- DEBUG -- */
    /*
    for (int i = 0; i < _numFeatureGroups; ++i)
    {
        cout << "Feature group " << i << " is formed by (" << _featureGroups[i].size() << "): ";
        
        for (int j = 0; j < _featureGroups[i].size(); ++j)
        {
            cout << _featureGroups[i][j] << ", ";
        }
        
        cout << endl;
    }
    */
}

int RAMIncrementalAgent::getGroupForFeature(int numFeaturesPerGroup)
{
    vector<int> candidateGroups;

    for (int i = 0; i < _numFeatureGroups; ++i)
    {
        if (_featureGroups[i].size() < numFeaturesPerGroup)
        {
            candidateGroups.push_back(i);
        }
    }
    
    if (candidateGroups.empty())
    {
        // any group is ok
        // this case happens when it is impossible for
        // all groups to have the same number of features
        return rand_range(0, _numFeatureGroups - 1);
    }
    else
    {
        int index = rand_range(0, candidateGroups.size() - 1);
        return candidateGroups[index];
    }
}

void RAMIncrementalAgent::putFeatureToGroup(int feature, int group)
{
    _featureGroups[group].push_back(feature);
}

void RAMIncrementalAgent::setNullFeatures()
{
    _faUtils->clearNullFeatures();

    for (std::set<int>::iterator it = _nullFeatureGroups.begin(); it != _nullFeatureGroups.end(); ++it)
    {
        int groupId = *it;
        
        const std::vector<int>& group = _featureGroups[groupId];
        
        for (int i = 0; i < group.size(); ++i)
        {
            int feature = group[i];
            _faUtils->addNullFeature(feature);
        }
    }
}

void RAMIncrementalAgent::removeRandomNullFeatureGroup()
{
    if (!_nullFeatureGroups.empty())
    {
        int index = rand_range(0, _nullFeatureGroups.size() - 1);
        std::set<int>::iterator it = _nullFeatureGroups.begin();
        std::advance(it, index);
        _nullFeatureGroups.erase(it);
    }
}

