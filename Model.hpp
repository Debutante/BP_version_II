//
//  Model.hpp
//  test1126
//
//  Created by 许清嘉 on 11/27/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Model_hpp
#define Model_hpp

#include "Layer.hpp"
#include <iostream>
#include <vector>
#include <list>

#define DEFAULT_VALUE 999

class Model{
private:
    vector<Layer> net;

    vector<vector<float>> trainInput;
    vector<vector<float>> trainOutput;

    string activationFunction;
    string lossFunction;
    vector<vector<float>> lossIteration;//detailed loss
    float lastTrainLoss = -1;//for adjusting learning rate

    
    string phase;
    int batchNum;
    int batchSize;
    int currentIteration;//updating times using a batch
    int maxIteration;
    int currentEpoch;//updating times using a dataset
    int maxEpoch;
    float delta;//for adjusting Huber loss Func
    float learningRate;
    
    bool stopTraining;
    
    string adjustmentMethod;
    
    vector<vector<float>> testInput;
    vector<vector<float>> testOutput;

    void forwardPropagation(const int, const int);
    void lossComputation(const int);
    void backwardPropagation(const int, const int);

public:
    Model(const list<int>&, const vector<vector<float>>&, const vector<vector<float>>&, const vector<vector<float>>&, const vector<vector<float>>&);
    void initializeNet(const float fixed = DEFAULT_VALUE, float bias = 0);
    void train(const float learning = 0.5, const int max = 5000, const string& activation = "sigmoid", const string& loss = "L2", const string& m = "BGD", const int bNum = 1, const float d = 0.5, const string& adjustment = "diy");//d is used to adjust Huber loss Func
    void predict();
    
};

#endif /* Model_hpp */
