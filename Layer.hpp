//
//  Layer.hpp
//  test1126
//
//  Created by 许清嘉 on 11/26/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Layer_hpp
#define Layer_hpp

#include <iostream>
#include <vector>

using namespace std;

class Layer{
private:
    vector<vector<float>> out;
    int nodeNum;
    
public:
    vector<vector<float>> weight;
    vector<vector<float>> coef;//common product item used to update weight
    bool isInputLayer;
    bool isOutputLayer;
    
    
    Layer(const int);
    void setOut(const vector<vector<float>>&);
    vector<vector<float>> getOut() const;
    
    int getNodeNum() const;
};

#endif /* Layer_hpp */
