//
//  Layer.cpp
//  test1126
//
//  Created by 许清嘉 on 11/26/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Layer.hpp"
Layer::Layer(const int nodeNum){
    this->nodeNum = nodeNum + 1;
    isInputLayer = false;
    isOutputLayer = false;
}

void Layer::setOut(const vector<vector<float>>& newOut){
    out = newOut;
    if (!isOutputLayer){
        vector<float> oneOut;
        for (int i = 0; i < out.at(0).size(); i++){
            oneOut.push_back(1);
        }
        out.push_back(oneOut);
    }
}

vector<vector<float>> Layer::getOut() const{
    return out;
}

int Layer::getNodeNum() const{
    return nodeNum;
}
