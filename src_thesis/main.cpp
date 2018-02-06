#include <iostream>
#include <fstream>
#include <sstream>

#include "AgentSettings.hpp"
#include "PlayerAgent.hpp"

#define PRINT_WIDTH 12

std::string generateExportHeaderLine()
{
    std::stringstream stream;
    stream << std::left << std::setw(PRINT_WIDTH) << "Episode";
    stream << std::left << std::setw(PRINT_WIDTH) << "Score";
    stream << std::left << std::setw(PRINT_WIDTH) << "Average" << '\n';
    
    return stream.str();
}

std::string generateExportContentLine(int episode, double score, double avgScore)
{
    std::stringstream stream;
    stream << std::left << std::setw(PRINT_WIDTH) << episode;
    stream << std::left << std::setw(PRINT_WIDTH) << score;
    stream << std::left << std::setw(PRINT_WIDTH) << avgScore << '\n';
    
    return stream.str();
}

void exportStringToFile(std::ofstream& exportFile, const std::string& str)
{
    exportFile << str;
    exportFile << std::flush;
}

double getAverageScore(int currentEpisode, double newScore, double currentScoreAverage)
{
    return currentScoreAverage + (1.0 / (double)currentEpisode) * (newScore - currentScoreAverage);
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Error: ./main config-file" << std::endl;
        exit(1);
    }

    /**
     * Set parameters
     */
    AgentSettings agentSettings(argv[1]);
    
    PlayerAgent* agent = PlayerAgent::createPlayerAgent(agentSettings);
    
    int numEpisodes = agentSettings.getInt("max_num_episodes", true);
    
    const std::string& header = generateExportHeaderLine();

    bool printScores = agentSettings.getBool("print_scores");
    
    if (printScores)
    {
        std::cout << header;
    }
    
    bool exportScores = agentSettings.getBool("export_scores");
    std::string exportRoute = "";
    std::ofstream exportFile;
    
    if (exportScores)
    {
        exportRoute = agentSettings.getString("export_route", true);
        exportFile.open(exportRoute.c_str());
        
        if (!exportFile.is_open())
        {
            std::cout << "Error: Could not create file \'" << exportRoute << "\'\n";
            exit(1);
        }
        else
        {
            exportStringToFile(exportFile, header);
        }
    }

    /**
     * Agent interaction
     */
    
    double avgScore = 0.0;
    
    for (int episode = 1; episode <= numEpisodes; ++episode)
    {
        double episodeScore = 0.0;
    
        agent->agentStart();
        
        episodeScore += agent->getLastReward();
        
        while (!agent->hasAgentEnded())
        {
            agent->agentStep();
            episodeScore += agent->getLastReward();
        }
        
        agent->agentEnd();
        agent->agentReset();
        
        avgScore = getAverageScore(episode, episodeScore, avgScore);
        
        const std::string& content = generateExportContentLine(episode, episodeScore, avgScore);
        
        if (printScores)
        {
            std::cout << content;
        }
        
        if (exportScores)
        {
            exportStringToFile(exportFile, content);
        }
    }
    
    if (exportFile.is_open())
    {
        exportFile.close();
    }

    delete agent;
    
    return 0;
}

