#include <iostream>
#include <iostream>
#include <memory>
#include "core/util/Config.h"
#include "core/util/Matrix.h"
#include "core/opencl/libsvm/OpenCLToolsTrain.h"
#include "core/opencl/OpenCLTools.h"
#include "core/process/IProcessor.h"
#include "core/util/rtti/ObjectFactory.h"
#include "core/util/rtti/ObjectFactory.h"
#include "core/svm/svm-train.h"

using namespace std;
using namespace core::util;
#ifdef _OPENCL
using namespace core::opencl;
using namespace core::opencl::libsvm;
#endif
using namespace core::svm;

int main(int argc, char **argv) {
    cout << "MAIN" << endl;    
    if (argc == 1){
        cout << "Call with -help for help" << endl;
#ifdef _OPENCL
        cout << "Call with -list for list of openCL capable platforms" << endl;
#endif
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "-help") == 0){
	cout << "*********************************************************" << endl;
        cout << "For training mode start with parameters: " << endl;
	cout << "1. -training" << endl;
	cout << "2. libSVM compatibile input training file" << endl;
	cout << "3. libSVM compatibile output model file" << endl;
        cout << "*********************************************************" << endl;
	cout << "To list opencl capable platforms and devices start with: " << endl;
	cout << "1. -list" << endl;
	cout << "*********************************************************" << endl;
	return 0;
    }

#ifdef _OPENCL
    if (argc == 2 && strcmp(argv[1], "-list") == 0){
        try{
            OpenclTools::getInstancePtr()->init(0, 0, true);
            OpenclTools::getInstancePtr()->cleanUp();
        } catch (SDException& exception) {
            cout << exception.handleException() << endl;
            exit(1);
        }
        return 0;
    }
#endif                 
        
    if (argc >= 2 && strcmp(argv[1], "-training") == 0) {
        if (argc < 4){
            cout << "training needs more parameters: input data file, output file" << endl;
            return 0;
        }
        try{
            Config* conf = Config::getInstancePtr();
#ifdef _OPENCL
            int platformId = 0;
            int deviceId = 0;
            string platformStr = conf->getPropertyValue("general.openCL.platformid");
            string deviceStr = conf->getPropertyValue("general.openCL.deviceid");
            int tmp = atoi(platformStr.c_str());
            if (tmp != 0)
                platformId = tmp;
            tmp = atoi(deviceStr.c_str());
            if (tmp != 0)
                deviceId = tmp;
            OpenCLToolsTrain::getInstancePtr()->init(platformId, deviceId, false);
#endif
            int val = train(argv[2], argv[3]);
            cout << val << endl;
        }
        catch (SDException& e){
            cout << e.handleException() << endl;
            exit(1);
        }
#ifdef _OPENCL
        OpenclTools::getInstancePtr()->cleanUp();
        OpenclTools::destroy();        
#endif
        Config::destroy();
        return 0;
    }
}
