//
//  Generator.cpp
//  test1126
//
//  Created by 许清嘉 on 11/26/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

//Generator(inputColumn, outputColumn, trainNum, testNum, fluctuation, func)
#include "Generator.hpp"
#include <random>
//row * n version
Generator::Generator(const int inputRow, const int outputRow, const int trainNum, const int testNum, const float fluctuation, const string& func){
    default_random_engine e;
    uniform_real_distribution<float> u(-1, 1);
    uniform_real_distribution<float> bias(1 - fluctuation, 1 + fluctuation);
    for (int i = 0; i < inputRow; i++){
        vector<float> oneInput;
        for (int j = 0; j < trainNum; j++)
            oneInput.push_back(u(e));
        trainInput.push_back(oneInput);
    }

    for (int i = 0; i < inputRow; i++){
        vector<float> oneInput;
        for (int j = 0; j < testNum; j++)
            oneInput.push_back(u(e));
        testInput.push_back(oneInput);
    }
    
    if (func == "prod-sin"){
        for (int i = 0; i < outputRow; i++){
            vector<float> oneOutput;
            for (int j = 0; j < trainNum; j++){
                float sum = 0.f;
                for (int k = 0; k < inputRow; k++){
                    sum += (i + 1) * trainInput.at(k).at(j);
                }
                sum = abs(sin(sum)) * (1 - fluctuation);
                sum *= bias(e);//fluctuation
                oneOutput.push_back(sum);
            }
            trainOutput.push_back(oneOutput);
        }
        
        for (int i = 0; i < outputRow; i++){
            vector<float> oneOutput;
            for (int j = 0; j < testNum; j++){
                float sum = 0.f;
                for (int k = 0; k < inputRow; k++){
                    sum += (i + 1) * testInput.at(k).at(j);
                }
                sum = abs(sin(sum)) * (1 - fluctuation);
                oneOutput.push_back(sum);
            }
            testOutput.push_back(oneOutput);
        }
    }

}

//n * row version
//Generator::Generator(const int inputColumn, const int outputColumn, const int trainNum, const int testNum, const float fluctuation, const string& func){
//    default_random_engine e;
//    uniform_real_distribution<float> u(-1, 1);
//    uniform_real_distribution<float> bias(1 - fluctuation, 1 + fluctuation);
//    if (func == "prod-sin"){
//        for (int i = 0; i < trainNum; i++){
//            vector<float> oneInput;
//            for (int j = 0; j < inputColumn; j++){
//                oneInput.push_back(u(e));
//            }
//            vector<float> oneOutput;
//            for (int k = 0; k < outputColumn; k++){
//                float sum = 0.f;
//                for (int j = 0; j < inputColumn; j++){
//                    sum += (k + 1) * oneInput.at(j);
//                }
//                sum = abs(sin(sum)) * (1 - fluctuation);
//                sum *= bias(e);//fluctuation
//                oneOutput.push_back(sum);
//            }
//            trainInput.push_back(oneInput);
//            trainOutput.push_back(oneOutput);
//        }
//
//        for (int i = 0; i < testNum; i++){
//            vector<float> oneInput;
//            for (int j = 0; j < inputColumn; j++){
//                oneInput.push_back(u(e));
//            }
//            vector<float> oneOutput;
//            for (int k = 0; k < outputColumn; k++){
//                float sum = 0.f;
//                for (int j = 0; j < inputColumn; j++){
//                    sum += (k + 1) * oneInput.at(j);
//                }
//                sum = abs(sin(sum)) * (1 - fluctuation);
//                oneOutput.push_back(sum);
//            }
//            testInput.push_back(oneInput);
//            testOutput.push_back(oneOutput);
//        }
//    }
//}
