//
//  main.cpp
//  test1126
//
//  Created by 许清嘉 on 11/26/19.
//  Copyright © 2019 许清嘉. All rights reserved.
//

#include "Generator.hpp"
#include "Model.hpp"
#include <cmath>
#include <ctime>

int main(int argc, const char * argv[]) {
    // insert code here...

    //Generator(inputColumn, outputColumn, trainNum, testNum, fluctuation, func)
    clock_t start_generate, end_generate, start_train, end_train, start_test, end_test;
    start_generate = clock();
    Generator g(2, 2, 10, 3);
    end_generate = clock();
    cout << "···Generating data consumes " << (double)(end_generate - start_generate) / CLOCKS_PER_SEC << "s.···" << endl;
    Model m({2, 3, 2}, g.trainInput, g.trainOutput, g.testInput, g.testOutput);
    
    m.initializeNet();
    //Params: train(learningRate, maxEpoch, activationFunction, lossFunction, trainMode, batchNum, delta)
    start_train = clock();
    m.train(10, 150, "sigmoid", "L2", "BGD");
    end_train = clock();
    cout << "···Training consumes " << (double)(end_train - start_train) / CLOCKS_PER_SEC << "s.···" << endl << endl;
    
    start_test = clock();
    m.predict();
    end_test = clock();
    cout << "···Test consumes " << (double)(end_test - start_test) / CLOCKS_PER_SEC << "s.···" << endl << endl;
    
    cout << "······Total process consumes " << (double)(end_test - start_generate) / CLOCKS_PER_SEC << "s.······" << endl << endl;
    
    return 0;
}
