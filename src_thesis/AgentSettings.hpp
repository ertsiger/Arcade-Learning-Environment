#ifndef __AGENT_SETTINGS_HPP__
#define __AGENT_SETTINGS_HPP__

#include <map>
#include <string>

typedef std::map<std::string, std::string> SettingsMap;
typedef std::map<std::string, std::string>::const_iterator SettingsMapIt;

/**
 * AgentSettings
 * Class that manages the data obtained from a filename
 * Imitates the behaviour of the OSystem Stella class
 */
class AgentSettings
{
private:
    // Default name of the file from where data is read
    static const std::string _defaultFilename;
    
    // What separes the data attribute from the value (=)
    static const std::string _fieldDelimiter;
    
    // Used to guarantee a better parsing
    static const std::string _emptyString;
    
    // Used to indicate what char begins a comment with
    static const char _commentDelimiter;
    
    // Where all mappings between attributes and values in the file are saved
    SettingsMap _settings;

public:
    // Constructor using _defaultFilename
    AgentSettings();
    
    // Constructor using filename parameter
    AgentSettings(const std::string& filename);
    
    /* To get specific types of data (strict indicates if the attribute
     * must appear in the SettingsMap)
     */
    // Returns false by default
    bool getBool(const std::string& attr, bool strict = false) const;
    
    // Returns -1.0 by default
    double getFloat(const std::string& attr, bool strict = false) const;
    
    // Returns -1 by default
    int getInt(const std::string& attr, bool strict = false) const;
    
    // Returns an empty string by default
    const std::string& getString(const std::string& attr, bool strict = false) const;

private:
    // Parses the file and sets the SettingsMap
    void init(const std::string& filename);
    
    // Message to be shown if the attribute is not found
    void showAttributeNotFoundMsg(const std::string& attr) const;
};

#endif /* __AGENT_SETTINGS_HPP__ */

