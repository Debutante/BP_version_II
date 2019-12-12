//
//  Generator.hpp
//  test1126
//
//  Created by 许清嘉 on 11/26/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#ifndef Generator_hpp
#define Generator_hpp

#include <iostream>
#include <vector>
using namespace std;

class Generator{
public:
//    int inputColumn;//the num in a group
//    int outputColumn;
//
//    int trainNum;//the group num

    vector<vector<float>> trainInput;
    vector<vector<float>> trainOutput;
    
//    int testNum;

    vector<vector<float>> testInput;
    vector<vector<float>> testOutput;
    
    Generator(const int, const int, const int, const int, const float fluctuation = 0.05, const string& func = "prod-sin");
};

#endif /* Generator_hpp */
