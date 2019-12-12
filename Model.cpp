//
//  Model.cpp
//  test1126
//
//  Created by 许清嘉 on 11/27/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Model.hpp"
#include <random>
#include <cmath>
#include <assert.h>
#include <algorithm>
#include <numeric>
#define LAMBDA_SELU 1.0507009873554804934193349852946
#define ALPHA_SELU 1.6732632423543772848170429916717

const list<string> activationList = {"sigmoid", "tanh", "ReLU", "LeakyReLU", "SELU"};
const list<string> lossList = {"squared", "L1", "L2", "Huber", "Log-Cosh"};
const list<string> modeList = {"SGD", "BGD", "MBGD"};//Stochastic Gradient Descent, Batch Gradient Descent, Mini-Batch Gradient Descent

//weight coef net

void print(float ele){
    cout << " " << ele;
}

Model::Model(const list<int>& layerNodeNum, const vector<vector<float>>& trainIn, const vector<vector<float>>& trainOut, const vector<vector<float>>& testIn, const vector<vector<float>>& testOut){

    currentIteration = 0;
    currentEpoch = 0;
    stopTraining = false;
    
    net.clear();
    for (auto iter = layerNodeNum.begin(); iter != layerNodeNum.end(); iter++){
        Layer layer = Layer(*iter);
        net.push_back(layer);
    }
    net.front().isInputLayer = true;
    net.front().setOut(trainIn);//to-do testIn
    net.back().isOutputLayer = true;
    
    assert(trainIn.size() == layerNodeNum.front());
    assert(testIn.size() == layerNodeNum.front());
    assert(trainOut.size() == layerNodeNum.back());
    assert(trainOut.size() == layerNodeNum.back());
    
    for (int i = 0; i < trainIn.size(); i++){
        for (int j = 0; j < trainOut.size(); j++){
            assert(trainIn.at(i).size() == trainOut.at(j).size());
            assert(testIn.at(i).size() == testOut.at(j).size());
        }
    }
    
    trainInput = trainIn;
    trainOutput = trainOut;
    testInput = testIn;
    testOutput = testOut;
}

void Model::initializeNet(const float fixed, float bias){
    if (fixed == DEFAULT_VALUE){
        //no given weight, weight is assigned by an uniform distribution
        default_random_engine e;
        for (auto iter = net.begin(); iter != net.end() - 1; iter++){
            iter->weight.clear();
            int row = iter->getNodeNum();
            int column = (iter + 1)->getNodeNum() - 1;
            for (int i = 0; i < row; i++){
                vector<float> oneWeight;
                normal_distribution<float> u(0, 1.0 / column);
//                uniform_real_distribution<float> u(0, 1.0 / column);
                for (int j = 0; j < column; j++){
                    oneWeight.push_back(u(e));
                }
                iter->weight.push_back(oneWeight);
            }
        }
    }
    else {
        //assign user-specified value for all weights
        if (bias == 0)
            //bias is not defined
            bias = fixed;
        
        for (auto iter = net.begin(); iter != net.end() - 1; iter++){
            iter->weight.clear();
            int row = iter->getNodeNum();
            int column = (iter + 1)->getNodeNum() - 1;
            for (int i = 0; i < row - 1; i++){
                vector<float> oneWeight;
                for (int j = 0; j < column; j++){
                    oneWeight.push_back(fixed);
                }
                iter->weight.push_back(oneWeight);
            }
            vector<float> biasWeight;
            for (int j = 0; j < column; j++){
                biasWeight.push_back(bias);
            }
            iter->weight.push_back(biasWeight);
        }
    }
}
  
void Model::forwardPropagation(const int layerNo, const int batchNo){
    if (layerNo > net.size() - 2)
        return;
    Layer layer = net.at(layerNo);
    Layer renewLayer = net.at(layerNo + 1);
    int row = layer.getNodeNum();
    int column = renewLayer.getNodeNum() - 1;
    int times = 0;
    int startIndex = 0;
    if (phase == "train"){
        if (layer.isInputLayer)
            startIndex = batchNo * batchSize;
        times = min(batchSize, (int)trainInput.at(0).size() - batchNo * batchSize);
    }
    else if (phase == "test")//to-do update first layer's out
        times = (int)testInput.at(0).size();
    
    vector<vector<float>> outs;
    for (int j = 0; j < column; j++){
        vector<float> oneOut;
        for (int t = 0; t < times; t++){
            float in = 0.f, out;
//            for (int i = 0; i < row; i++){
//                in += layer.weight.at(i).at(j) * layer.getOut().at(i).at(t + startIndex);
//            }
            in = inner_product(begin(layer.weight), begin(layer.weight) + row, begin(layer.getOut()), in, plus<double>(), [j, t, startIndex](const vector<float>& f1, const vector<float>& f2) {return f1.at(j) * f2.at(t + startIndex);});
            if (activationFunction == "sigmoid"){
                out = 1.f / (exp(-in) + 1);
            }
            else if (activationFunction == "tanh"){
                out = tanh(in);
            }
            else if (activationFunction == "ReLU"){//Rectified Linear Unit
                if (in < 0)
                    out = 0;
                else out = in;
            }
            else if (activationFunction == "LeakyReLU"){
                if (in < 0)
                    out = 0.01 * in;
                else out = in;
            }
            else if (activationFunction == "SELU"){
                if (in < 0){
                    out = LAMBDA_SELU * ALPHA_SELU * (exp(in) - 1);
                }
                else
                    out = LAMBDA_SELU * in;
            }
            oneOut.push_back(out);
        }
        outs.push_back(oneOut);
    }
    net[layerNo + 1].setOut(outs);
}

void Model::lossComputation(const int batchNo){
    vector<vector<float>> layerOut = net.back().getOut();
    float loss = 0.f;
    float lossSum = 0.f;
    int times = 0;
    int startIndex = 0;
    
    if (phase == "train"){
        startIndex = batchNo * batchSize;
        times = min(batchSize, (int)trainInput.at(0).size() - startIndex);
        if (lossFunction == "squared"){
            for (int i = 0; i < trainOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += pow(layerOut.at(i).at(t) - trainOutput.at(i).at(startIndex + t), 2);
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(trainOutput.at(i)) + startIndex, loss, plus<double>(), [] (float x, float y) { return pow(x - y, 2); });
            }
            loss /= 2;
        }
        else if (lossFunction == "L1"){
            for (int i = 0; i < trainOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += abs(layerOut.at(i).at(t) - trainOutput.at(i).at(startIndex + t));
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(trainOutput.at(i)) + startIndex, loss, plus<float>(), [] (float x, float y) { return abs(x - y); });
            }
            loss /= times;
        }
        else if (lossFunction == "L2"){
            for (int i = 0; i < trainOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += pow(layerOut.at(i).at(t) - trainOutput.at(i).at(startIndex + t), 2);
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(trainOutput.at(i)) + startIndex, loss, plus<double>(), [] (float x, float y) { return pow(x - y, 2); });
            }
            loss /= 2 * times;
        }
        else if (lossFunction == "Huber"){
            for (int i = 0; i < trainOutput.size(); i++){
                for (int t = 0; t < times; t++){
                    if (abs(trainOutput.at(i).at(startIndex + t) - layerOut.at(i).at(t)) <= delta){
                        loss += pow(layerOut.at(i).at(t) - trainOutput.at(i).at(startIndex + t), 2) / 2;
                    }
                    else {
                        loss += delta * abs(trainOutput.at(i).at(startIndex + t) - layerOut.at(i).at(t)) - pow(delta, 2) / 2;
                    }
                }
            }
            loss /= times;
        }
        else if (lossFunction == "Log-Cosh"){
            for (int i = 0; i < trainOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += log(cosh(layerOut.at(i).at(t) - trainOutput.at(i).at(startIndex + t)));
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(trainOutput.at(i)) + startIndex, loss, plus<double>(), [] (float x, float y) { return log(cosh(x - y)); });
            }
        }
        lossIteration[batchNo].push_back(loss);
        if (batchNo == batchNum - 1){
//            for (int i = 0; i < batchNum; i++)
//                lossSum += lossIteration.at(i).at(currentEpoch);
            lossSum = accumulate(begin(lossIteration), begin(lossIteration) + batchNum, lossSum, [this](const float sum, const vector<float>& f) {return sum + f.at(currentEpoch);});
            lossSum /= batchNum;
            cout << "\tEpoch(" << currentEpoch + 1 << "/" << maxEpoch << "), loss = " << lossSum << endl;
            if (adjustmentMethod == "diy"){
                if ((lastTrainLoss != -1) && ((lossSum - lastTrainLoss > 0) || abs(lossSum - lastTrainLoss) <= 1E-8)){
                    float ratio = 1.f / (exp(-(float)currentEpoch / maxEpoch) + 1);
                    ratio = 2 * ratio - 0.5;
                    learningRate *= ratio;
                    if (learningRate < 1E-4)
                        stopTraining = true;
                    cout << "[Reduce Learning rate]: " << learningRate << ", ratio = " << ratio << "(@Epoch=" << currentEpoch + 1 << ")" << endl;
                }
            }
            lastTrainLoss = lossSum;
        }
    }
    else if (phase == "test"){
        times = (int)testInput.at(0).size();
        if (lossFunction == "squared"){
            for (int i = 0; i < testOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += pow(layerOut.at(i).at(t) - testOutput.at(i).at(t), 2);
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(testOutput.at(i)), loss, plus<double>(), [] (float x, float y) { return pow(x - y, 2); });
            }
            loss /= 2;
        }
        else if (lossFunction == "L1"){
            for (int i = 0; i < testOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += abs(layerOut.at(i).at(t) - testOutput.at(i).at(t));
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(testOutput.at(i)), loss, plus<float>(), [] (float x, float y) { return abs(x - y); });
            }
            loss /= times;
        }
        else if (lossFunction == "L2"){
            for (int i = 0; i < testOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += pow(layerOut.at(i).at(t) - testOutput.at(i).at(t), 2);
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(testOutput.at(i)), loss, plus<double>(), [] (float x, float y) { return pow(x - y, 2); });
            }
            loss /= 2 * times;
        }
        else if (lossFunction == "Huber"){
            for (int i = 0; i < testOutput.size(); i++){
                for (int t = 0; t < times; t++){
                    if (abs(testOutput.at(i).at(t) - layerOut.at(i).at(t)) <= delta){
                        loss += pow(layerOut.at(i).at(t) - testOutput.at(i).at(t), 2) / 2;
                    }
                    else {
                        loss += delta * abs(testOutput.at(i).at(t) - layerOut.at(i).at(t)) - pow(delta, 2) / 2;
                    }
                }
            }
            loss /= times;
        }
        else if (lossFunction == "Log-Cosh"){
            for (int i = 0; i < testOutput.size(); i++){
//                for (int t = 0; t < times; t++){
//                    loss += log(cosh(layerOut.at(i).at(t) - testOutput.at(i).at(t)));
//                }
                loss = inner_product(begin(layerOut.at(i)), begin(layerOut.at(i)) + times, begin(testOutput.at(i)), loss, plus<double>(), [] (float x, float y) { return log(cosh(x - y)); });
            }
        }
        for (int i = 0; i < testOutput.size(); i++){
            cout << "Out " << i << ": ";
//            for (int t = 0; t < times; t++){
//                cout << " " << layerOut.at(i).at(t);
//            }
            for_each(begin(layerOut.at(i)), end(layerOut.at(i)), [](float ele) {cout << " " << ele;});
            cout << endl;
            cout << "Excepted " << i << ": ";
//            for (int t = 0; t < times; t++){
//                cout << " " << testOutput.at(i).at(t);
//            }
            for_each(begin(testOutput.at(i)), end(testOutput.at(i)), [](float ele) {cout << " " << ele;});
            cout << endl;
        }
        cout << "\tEpoch(1/1), loss = " << loss << endl;
    }
    
    
//    cout << "\tIteration(" << currentIteration + 1 << "/" << maxIteration << "), Epoch(" << currentEpoch + 1 << "/" << maxEpoch << "), loss = " << loss << endl;
}

void Model::backwardPropagation(const int layerNo, const int batchNo){
    if (layerNo < 1)
        return;
    Layer layer = net.at(layerNo);
    Layer renewLayer = net.at(layerNo - 1);
    int row = renewLayer.getNodeNum() - 1;
    int column = layer.getNodeNum() - 1;
    int startIndex = batchNo * batchSize;
    int times = min(batchSize, (int)trainInput.at(0).size() - startIndex);
    layer.coef.clear();
    if (layer.isOutputLayer == true){
        for (int j = 0; j < column; j++){
            vector<float> subCoef;
            if (lossFunction == "squared"){
                for (int t = 0; t < times; t++){
                    subCoef.push_back(layer.getOut().at(j).at(t) - trainOutput.at(j).at(startIndex + t));
                }
            }
            else if (lossFunction == "L2"){
                for (int t = 0; t < times; t++){
                    subCoef.push_back((layer.getOut().at(j).at(t) - trainOutput.at(j).at(startIndex + t)) / times);
                }
            }
            else if (lossFunction == "L1"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) > trainOutput.at(j).at(startIndex + t))
                        subCoef.push_back(1.0 / times);
                    else if (layer.getOut().at(j).at(t) < trainOutput.at(j).at(startIndex + t))
                        subCoef.push_back(-1.0 / times);
                    else
                        subCoef.push_back(0);
                }
            }
            else if (lossFunction == "Huber"){
                for (int t = 0; t < times; t++){
                    if (abs(trainOutput.at(j).at(startIndex + t) - layer.getOut().at(j).at(t)) <= delta)
                        subCoef.push_back((layer.getOut().at(j).at(t) - trainOutput.at(j).at(startIndex + t)) / times);
                    else if (layer.getOut().at(j).at(t) - trainOutput.at(j).at(startIndex + t) > 0)
                        subCoef.push_back(delta);
                    else
                        subCoef.push_back(-delta);
                }
            }
            else if (lossFunction == "Log-Cosh"){
                for (int t = 0; t < times; t++){
                    subCoef.push_back(tanh(layer.getOut().at(j).at(t) - trainOutput.at(j).at(startIndex + t)) * layer.getOut().at(j).at(t));
                }
            }
            
            if (activationFunction == "sigmoid"){
                for (int t = 0; t < times; t++){
                    subCoef[t] *= layer.getOut().at(j).at(t) * (1 - layer.getOut().at(j).at(t));
                }
            }
            else if (activationFunction == "tanh"){
                for (int t = 0; t < times; t++){
                    subCoef[t] *= 1 - pow(tanh(layer.getOut().at(j).at(t)), 2);
                }
            }
            else if (activationFunction == "ReLU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] = 0;
                }
            }
            else if (activationFunction == "LeakyReLU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] *= 0.01;
                }
            }
            else if (activationFunction == "SELU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] *= LAMBDA_SELU * ALPHA_SELU * exp(layer.getOut().at(j).at(t));
                    else
                        subCoef[t] *= LAMBDA_SELU;
                }
            }
            
//            float oneCoef = 0;
//            for (int t = 0; t < times; t++){
//                oneCoef += subCoef.at(t);
//            }
            
            float oneCoef = accumulate(subCoef.begin(), subCoef.end(), 0.f);
            
            layer.coef.push_back(subCoef);
            //renew b
            renewLayer.weight[row][j] -= learningRate * oneCoef;
//            cout << "oneCoef = " << oneCoef << endl;
            
            for (int i = 0; i < row; i++){
                //renew weights
                float postCoef = 0.f;
//                for (int t = 0; t < times; t++){
//                    postCoef += subCoef.at(t) * renewLayer.getOut().at(i).at(t);
//                }
                postCoef = inner_product(begin(subCoef), end(subCoef), begin(renewLayer.getOut().at(i)), postCoef, plus<double>(), multiplies<float>());
                renewLayer.weight[i][j] -= learningRate * postCoef;
            }
        }
    }
    else {
        Layer nextLayer = net.at(layerNo + 1);
        int nextColumn = nextLayer.getNodeNum() - 1;
        for (int j = 0; j < column; j++){
            vector<float> subCoef;
            for (int t = 0; t < times; t++){
                float preCoef = 0.f;
                for (int k = 0; k < nextColumn; k++){
                    preCoef += nextLayer.coef.at(k).at(t) * layer.weight.at(j).at(k);
                }
                subCoef.push_back(preCoef);
            }
            if (activationFunction == "sigmoid"){
                for (int t = 0; t < times; t++){
                    subCoef[t] *= layer.getOut().at(j).at(t) * (1 - layer.getOut().at(j).at(t));
                }
            }
            else if (activationFunction == "tanh"){
                for (int t = 0; t < times; t++){
                    subCoef[t] *= 1 - pow(tanh(layer.getOut().at(j).at(t)), 2);
                }
            }
            else if (activationFunction == "ReLU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] = 0;
                }
            }
            else if (activationFunction == "LeakyReLU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] *= 0.01;
                }
            }
            else if (activationFunction == "SELU"){
                for (int t = 0; t < times; t++){
                    if (layer.getOut().at(j).at(t) <= 0)
                        subCoef[t] *= LAMBDA_SELU * ALPHA_SELU * exp(layer.getOut().at(j).at(t));
                    else
                        subCoef[t] *= LAMBDA_SELU;
                }
            }
//            float oneCoef = 0;
//            for (int t = 0; t < times; t++){
//                oneCoef += subCoef.at(t);
//            }
            
            float oneCoef = accumulate(subCoef.begin(), subCoef.end(), 0.f);
            
            layer.coef.push_back(subCoef);
            //renew b
            renewLayer.weight[row][j] -= learningRate * oneCoef;
//            cout << "Iteration: " << currentIteration << ", gradient: " << learningRate * oneCoef << endl;
            
            for (int i = 0; i < row; i++){
                //renew weights
                float postCoef = 0.f;
//                for (int t = 0; t < times; t++){
//                    postCoef += subCoef.at(t) * renewLayer.getOut().at(i).at(t);
//                }
                postCoef = inner_product(begin(subCoef), end(subCoef), begin(renewLayer.getOut().at(i)), postCoef, plus<double>(), multiplies<float>());
                renewLayer.weight[i][j] -= learningRate * postCoef;
            }
        }
    }
    net[layerNo].coef = layer.coef;
    net[layerNo - 1].weight = renewLayer.weight;
}

void Model::train(const float learning, const int max, const string& activation, const string& loss, const string& m, const int bNum, const float d, const string& adjustment){
    assert(find(activationList.begin(), activationList.end(), activation) != activationList.end());
    assert(find(lossList.begin(), lossList.end(), loss) != lossList.end());
    assert(find(modeList.begin(), modeList.end(), m) != modeList.end());
    assert(bNum <= (int)trainInput.at(0).size());
    assert(bNum >= 1);
    assert(d > 0 && d < 1);
    
    phase = "train";
    
    learningRate = learning;
    maxEpoch = max;
    activationFunction = activation;
    lossFunction = loss;
    adjustmentMethod = adjustment;
    
    if (lossFunction == "squared")
        cout << "Warning: The use of squared is deprecated, use L2 instead." << endl;
    
    if (m == "SGD")
        batchNum = (int)trainInput.at(0).size();
    else if (m == "BGD")
        batchNum = 1;
    else if (m == "MBGD")
        batchNum = bNum;
    
    lossIteration.resize(batchNum);
    
    maxIteration = batchNum * maxEpoch;
    delta = d;
    
    batchSize = ceil((float)trainInput.at(0).size() / batchNum);
    
    cout << "=*=*=*=*=*=*[Train Overview]*=*=*=*=*=*=" << endl;
    cout << "activation Function: " << activationFunction << endl;
    cout << "Loss Function: " << lossFunction << endl;
    cout << "Train Mode: " << m << endl;
    cout << "Batch Num: " << batchNum << endl;
    cout << "Learning Rate: " << learningRate << endl;
    
    cout << endl << "<============Training Starts============>" << endl;
    
    while(currentEpoch < maxEpoch){
        for (int j = 0; j < batchNum; j++){
            for (int i = 0; i < net.size() - 1; i++){
                forwardPropagation(i, j);
            }
            lossComputation(j);
            currentIteration++;
            for (int i = (int)net.size() - 1; i > 0; i--){
                backwardPropagation(i, j);
            }
        }
        if (stopTraining){
            cout << ">============Training Stops(in advance)============<" << endl;
            break;
        }
        currentEpoch++;
    }
    if (!stopTraining){
        cout << ">=============Training Stops=============<" << endl;
    }
}


void Model::predict(){
    phase = "test";
    net.front().setOut(testInput);
    
    cout << "=*=*=*=*=*=*[Test Overview]*=*=*=*=*=*=" << endl;
    cout << "activation Function: " << activationFunction << endl;
    cout << "Loss Function: " << lossFunction << endl;
    
    cout << endl << "<============Test Starts============>" << endl;
    
    for (int i = 0; i < net.size() - 1; i++){
        forwardPropagation(i, 0);
    }
    lossComputation(0);
    
    cout << ">=============Test Stops=============<" << endl;
}
