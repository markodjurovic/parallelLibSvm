#include "PredictorFactory.h"
#include "core/util/prediction/libsvm/SvmPredict.h"
#include "core/util/Config.h"

using namespace core::util::prediction;
using namespace core::util::prediction::svm;
using namespace core::util;
using namespace std;

namespace core{
    namespace util{
        IPrediction* getPredictor(){
            Config* conf = Config::getInstancePtr();
            string type = conf->getPropertyValue("process.Prediction.predictionType");
            if (type == "SVM"){
                return SvmPredict::getInstancePtr();
            }
            return 0;
        }
    }
}
