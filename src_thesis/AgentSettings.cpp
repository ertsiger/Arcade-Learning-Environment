#include "AgentSettings.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

const std::string AgentSettings::_defaultFilename = "agent_config.txt";
const std::string AgentSettings::_fieldDelimiter = "=";
const std::string AgentSettings::_emptyString = " ";
const char AgentSettings::_commentDelimiter = ';';

AgentSettings::AgentSettings()
{
    init(_defaultFilename);
}

AgentSettings::AgentSettings(const std::string& filename)
{
    init(filename);
}

bool AgentSettings::getBool(const std::string& attr, bool strict) const
{
    SettingsMapIt it = _settings.find(attr);

    if (it == _settings.end())
    {
        if (strict)
        {
            showAttributeNotFoundMsg(attr);
            exit(1);
        }
        
        return false;
    }

    const std::string& val = it->second;
    return (val == "1");
}

double AgentSettings::getFloat(const std::string& attr, bool strict) const
{
    SettingsMapIt it = _settings.find(attr);

    if (it == _settings.end())
    {
        if (strict)
        {
            showAttributeNotFoundMsg(attr);
            exit(1);
        }
        
        return -1.0;
    }

    const std::string& val = it->second;
    return atof(val.c_str());
}

int AgentSettings::getInt(const std::string& attr, bool strict) const
{
    SettingsMapIt it = _settings.find(attr);

    if (it == _settings.end())
    {
        if (strict)
        {
            showAttributeNotFoundMsg(attr);
            exit(1);
        }
        
        return -1;
    }

    const std::string& val = it->second;
    return atoi(val.c_str());
}

const std::string& AgentSettings::getString(const std::string& attr, bool strict) const
{
    SettingsMapIt it = _settings.find(attr);

    if (it == _settings.end())
    {
        if (strict)
        {
            showAttributeNotFoundMsg(attr);
            exit(1);
        }
        
        return _emptyString;
    }

    return it->second;
}

void AgentSettings::init(const std::string& filename)
{
    std::ifstream configFile;
    configFile.open(filename.c_str());
    
    if (!configFile.is_open())
    {
        std::cerr << "Error: Could not open file \'" << filename << "\'" << std::endl;
        exit(1);
    }
    
    std::string line = "";
    std::string attr = "";
    std::string val = "";
    int pos = 0;
    
    while (std::getline(configFile, line))
    {
        if (!line.empty())
        {
            line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
        
            if (line[0] != _commentDelimiter)
            {
                pos = line.find(_fieldDelimiter);
                
                if (pos == std::string::npos)
                {
                    std::cerr << "Error: Syntax is not correct in file \'" << filename << "\'" << std::endl;
                    exit(1);
                }
                
                attr = line.substr(0, pos);                
                val = line.substr(pos + 1, line.length());
                
                _settings[attr] = val;            
            }
        }
    }
}

void AgentSettings::showAttributeNotFoundMsg(const std::string& attr) const
{
    std::cerr << "Error: Undefined configuration parameter \'" << attr << "\'." << std::endl;
}

