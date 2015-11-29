#include "ObjectFactory.h"
#include "core/util/prediction/IPrediction.h"
#include "core/util/prediction/libsvm/SvmPredict.h"
#include "core/util/Config.h"

namespace core {
    namespace util {
        namespace RTTI {

            using namespace core::tools::image;
            using namespace core::util::prediction;
            using namespace std;
            using namespace core::util;
            
            void* ObjectFactory::getSingleton(std::string classID){
                raii::MutexRaii autoLock(&mutex);
                std::unordered_map<std::string, void*>::iterator iter = mappedSingletons.find(classID);
                if (iter != mappedSingletons.end())
                    return iter->second;
                return 0;
            }
            
            IImageParameteres* ObjectFactory::createImageParameters() {
                string classID = Config::getInstancePtr()->getPropertyValue("general.Prediction.parametersClass");                
                IImageParameteres* parametersClass = createInstance<IImageParameteres>(classID);
                return parametersClass;
            }
            
            IPrediction* ObjectFactory::createPredictor(){
                Config* conf = Config::getInstancePtr();
                string type = conf->getPropertyValue("general.Prediction.predictionClass");
                return ObjectFactory::getInstancePtr()->createInstance<IPrediction>(type);
//                if (type == "SVM"){
//                    return ObjectFactory::getInstancePtr()->createInstance<IPrediction>("core::util::prediction::svm::SvmPredict");
//                }
//                else{
//                    return ObjectFactory::getInstancePtr()->createInstance<IPrediction>("core::util::prediction::regression::RegressionPredict");
//                }
            }
            
            int ObjectFactory::registerSingleton(std::string classID, void* singleton){
                raii::MutexRaii autoLock(&mutex);
                mappedSingletons[classID] = singleton;
                return 1;
            }
        }
    }
}
