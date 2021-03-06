#include "Config.h"
#include <fstream>
#include "core/util/raii/RAIIS.h"

namespace core{
    namespace util{
        using namespace std;        
        using namespace rapidxml;
        using namespace core::util::raii;
        
        Config::Config() : Singleton<Config>(){
            rootNodeProcessing = true;
            init();            
        }
        
        Config::~Config(){
            
        }
        
        string Config::getPropertyValue(const string& key) throw(SDException&){
            unordered_map<string, string>::iterator iter = mappedValues.find(key);
            if (iter != mappedValues.end()){
                return iter->second;
            }
            else{ 
                SDException exc(SHADOW_NOT_FOUND_PROPERTY, "Config::getPropertyValue: " + key);
                throw exc;
            }
        }
        void Config::init() {            
            string path = CONFIG_FILE;
            ifstream inputFile;
            inputFile.open(path.c_str(), ifstream::in);
            if (inputFile && inputFile.is_open()) {
                ostringstream stream;
                string line;
                while (getline(inputFile, line)) {
                    stream << line;
                }
                inputFile.close();
                string xmlContent = stream.str();
                fillMap(xmlContent);
            } else {
                SDException exc(SHADOW_READ_UNABLE, "Config init");
                throw exc;
            }
        }
        void Config::fillMap(string xmlFileContent) {
            try {
                xml_document<> doc;
                const char* constContent = xmlFileContent.c_str();
                char* content = New char[strlen(constContent) + 1];
                VectorRaii<char> cntRaii(content);
                strcpy(content, constContent);
                doc.parse<0>(content);
                string currName = "";
                for (xml_node<>* root = doc.first_node(); root; root = root->next_sibling()) {                    
                    processNode(root, currName);
                }
            } catch (exception e) {
                SDException exc(SHADOW_INVALID_XML, "Config init");
                throw exc;
            }
        }

        string getNodeName(xml_node<>* node) {
            string retString = node->name();
            return retString;
        }        
        
        void Config::processNode(xml_node<>* node, string currName) {
            string refName = currName;
            if (refName != "") {
                refName += ".";
            }

            if (rootNodeProcessing) {
                rootNodeProcessing = false;
            } else {
                string nodeName = getNodeName(node);
                refName += nodeName;
            }            
            
            bool hasChildren = false;            
            for (xml_node<>* child = node->first_node(); child; child = child->next_sibling()) {
                hasChildren = true;                                
                processNode(child, refName);
            }

            if (hasChildren == false) {
                string nodeValStr = node->value();
                nodeValStr = trim(nodeValStr);
                refName = trim(refName);
                mappedValues[refName] = nodeValStr;                               
            }
        }
        
    }
}
