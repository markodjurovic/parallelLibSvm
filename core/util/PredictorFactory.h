#ifndef __PREDICTOR_MANAGER_H__ 
#define __PREDICTOR_MANAGER_H__

#include "core/util/prediction/IPrediction.h"

namespace core{
    namespace util{
        core::util::prediction::IPrediction* getPredictor();
    }
}

#endif
