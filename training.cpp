/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: training.cpp
    Last modifed:   22.12.2016 by Leonardo Citraro
    Description:    Example machine learning training process using OpenCV

    =========================================================================
    Things to keep in mind when coding with OpenCV:
        - make sure you always use the same type for matrices, vectors and
          so on. OpenCV doesn't properly convert variables of different 
          types! it's a pain to find out the bugs.
        - Do not fill a cv::Mat using Mat::ptr<>() !!! Use Mat::at<>() instead
        - if you work with floating point variable is good practice to test
          these value with std::isnan and std::isinf
        - Make sure you adjust the setTermCriteria correctly
    =========================================================================
*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iomanip>
#include <math.h>

using TYPE = float;
static int MatTYPE = CV_32FC1;

TYPE compute_mean(std::vector<TYPE> v) {
    return std::accumulate(std::begin(v), std::end(v), 0.0f)/v.size();
}

void feature_mean_variance(const cv::Mat& data, std::vector<TYPE>& mean, std::vector<TYPE>& var) {
    mean.resize(data.cols);
    var.resize(data.cols);
    
    for(size_t col=0; col<data.cols; ++col) {
        std::vector<TYPE> feature(data.rows);
        for(size_t i = 0; i < data.rows; ++i) {
            const TYPE* ptr_row = data.ptr<TYPE>(i);
            feature[i] = ptr_row[col];
        }
        TYPE m = std::accumulate(std::begin(feature), std::end(feature), 0.0)/feature.size();
        mean[col] = m;
        std::vector<TYPE> diff(data.rows);
        std::transform(std::begin(feature), std::end(feature), std::begin(diff), std::bind2nd(std::minus<TYPE>(), m));
        TYPE v = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0)/feature.size();
        var[col] = v;
    }
}

template<class T>
void save_vector( const std::string& filename, const std::vector<T>& v ) {
    std::ofstream f(filename, std::ios::out | std::ofstream::binary);
    unsigned int len = v.size();
    f.write( (char*)&len, sizeof(len) );
    f.write( (const char*)&v[0], len * sizeof(T) );
}

template<class T>
auto load_vector( const std::string& filename ) {
    std::vector<T> v;
    std::ifstream f(filename, std::ios::in | std::ofstream::binary);
    unsigned int len = 0;
    f.read( (char*)&len, sizeof(len) );
    v.resize(len);
    if( len > 0 ) 
        f.read( (char*)&v[0], len * sizeof(T) );
    return v;
}

int main(int argc, char* argv[]) {
    // ----------------------------------------------------------------------------
    // Data creation
    // ----------------------------------------------------------------------------
    std::vector<std::vector<TYPE>> data;
    std::vector<int> labels;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> gauss1(15,5);
    std::normal_distribution<> gauss2(5,2);
    
    const int N = 1000;
    
    for(int y=0; y<N; ++y){
        std::vector<TYPE> row = {static_cast<TYPE>(gauss1(gen)), static_cast<TYPE>(gauss1(gen))};
        data.push_back(row);
        labels.push_back(0);
    }
    for(int y=0; y<N; ++y){
        std::vector<TYPE> row = {static_cast<TYPE>(gauss2(gen)), static_cast<TYPE>(gauss2(gen))};
        data.push_back(row);
        labels.push_back(1);
    }
    
    // plot the data
    const int alpha = 10;
    cv::Mat graph = cv::Mat::zeros(40*alpha,40*alpha, CV_32FC3);
    for(int y=0; y<N; ++y){
        cv::circle(graph, cv::Point(data[y][0]*alpha, data[y][1]*alpha), 1, cv::Scalar(255,0,0), 1);
    }
    for(int y=N; y<2*N; ++y){
        cv::circle(graph, cv::Point(data[y][0]*alpha, data[y][1]*alpha), 1, cv::Scalar(0,255,0), 1);
    }
    cv::imshow("data",graph);
    
    std::cout << "data=[" << data.size() << " x " << data[0].size() << "]\n";
    std::cout << "labels=[" << labels.size() << " x " << 1 << "]\n";
    
    std::cout << "Creation of the dataset done!\n";

    // ----------------------------------------------------------------------------
    // Conversion to cv::Mat
    // ----------------------------------------------------------------------------
    cv::Mat mat_labels(labels,false);
    cv::Mat mat_data(data.size(), data[0].size(), MatTYPE);
    for(size_t i = 0; i < mat_data.rows; ++i) {
        //double* ptr_row = mat_data.ptr<double>(i);
        for (size_t j = 0; j < mat_data.cols; ++j) {
            TYPE val = data[i][j];
            if(std::isnan(val) || std::isinf(val))
                std::cerr << "val is inf or nan!\n";
            mat_data.at<TYPE>(i,j) = val;
            //ptr_row[j] = val;
        }
    }
    
    std::cout << "mat_data=" << mat_data.size() << "\n";
    std::cout << "mat_labels=" << mat_labels.size() << "\n";
    
    std::cout << "Conversion std::vector -> cv::Mat done!\n";
    
    // ----------------------------------------------------------------------------
    // Get mean and variance of all features
    // ----------------------------------------------------------------------------
    std::vector<TYPE> mean, var;
    feature_mean_variance(mat_data, mean, var);
    save_vector("mean.ext", mean);
    save_vector("var.ext", var);
    //auto temp = load_vector<TYPE>("mean.ext");
    
    std::cout << "mean0=" << mean[0] << " mean1=" << mean[1] << "\n";
    
    std::cout << "Get mean and variance of the features done!\n";
    
    // ----------------------------------------------------------------------------
    // Normalization mean and variance
    // ----------------------------------------------------------------------------
    
    for(size_t i = 0; i < mat_data.rows; ++i) {
        TYPE* ptr_row = mat_data.ptr<TYPE>(i);
        for(size_t j = 0; j < mat_data.cols; ++j) {
            ptr_row[j] -= mean[j];
            ptr_row[j] /= var[j];
            if(std::isnan(ptr_row[j]) || std::isinf(ptr_row[j]))
                std::cerr << "ptr_row[j] is inf or nan!\n";
        }
    }
    
    // ----------------------------------------------------------------------------
    // Storage
    // ----------------------------------------------------------------------------
    cv::FileStorage file_data("mat_data.ext", cv::FileStorage::WRITE);
    file_data << "mat_data" << mat_data;
    cv::FileStorage file_labels("mat_labels.ext", cv::FileStorage::WRITE);
    file_labels << "mat_labels" << mat_labels;
    
    //cv::FileStorage file_data("mat_data.ext", FileStorage::READ);
    //file_data["mat_data"] >> mat_data;
    
    std::cout << "Data saved!\n";
    
    // ----------------------------------------------------------------------------
    // Learning machine
    // ----------------------------------------------------------------------------

    cv::Ptr<cv::ml::SVM> clf = cv::ml::SVM::create();
    clf->setType(cv::ml::SVM::C_SVC);
    clf->setKernel(cv::ml::SVM::LINEAR);
    //clf->setDegree(2);
    //clf->setNu(0.5);
    //clf->setC(1);
    //clf->setGamma(1e-8);
    clf->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-6));

/*
    cv::Ptr<cv::ml::KNearest> clf = cv::ml::KNearest::create();
    clf->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);
    clf->setDefaultK(10);
*/
    
    std::cout << "Learning machine prepaired!\n";
    
    // ----------------------------------------------------------------------------
    // Cross-validation (10x random split 70%)
    // ----------------------------------------------------------------------------
    cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::create(mat_data, cv::ml::SampleTypes::ROW_SAMPLE, mat_labels);
    
    //clf->trainAuto(dataset, 10);
    
    std::cout   << "\n-------------------------------------------\n";
    
    std::vector<TYPE> accuracies;
    std::vector<TYPE> sensitivities;
    std::vector<TYPE> specificities;
    std::vector<TYPE> false_positive_rates;
    for(int k=0; k<10; k++) {
        dataset->setTrainTestSplitRatio(0.7, true); // random split of the data
        /*
        cv::Mat train_idx = dataset->getTrainSampleIdx();
        std::cout << train_idx.size() << "\n";
        for(int i=0; i<train_idx.cols; ++i){
            std::cout << train_idx.at<int>(i) << ", ";
        }
        std::cout << "\n";
        cv::Mat test_idx = dataset->getTestSampleIdx();
        
        for(int i=0; i<test_idx.cols; ++i){
            std::cout << test_idx.at<int>(i) << ", ";
        }
        std::cout << "\n";
        */
        // training & test samples
        cv::Mat train_idx = dataset->getTrainSampleIdx();
        cv::Mat train_data = cv::ml::TrainData::getSubVector(mat_data, train_idx);//dataset->getTrainSamples();
        cv::Mat train_labels = cv::ml::TrainData::getSubVector(mat_labels, train_idx);//dataset->getTrainResponses();
        
        cv::Mat test_idx = dataset->getTestSampleIdx();
        cv::Mat test_data = cv::ml::TrainData::getSubVector(mat_data, test_idx);//dataset->getTestSamples();
        cv::Mat test_labels = cv::ml::TrainData::getSubVector(mat_labels, test_idx);//;//dataset->getTestResponses();
        /*
        std::cout << train_data.size() << "\n";
        for(int i=0; i<train_data.rows; ++i){
            std::cout << train_data.at<TYPE>(i,0) << ", " << train_data.at<TYPE>(i,1) << " label=" << train_labels.at<int>(i) << "\n";
        }
        std::cout << "\n";
        for(int i=0; i<test_data.rows; ++i){
            std::cout << test_data.at<TYPE>(i,0) << ", " << test_data.at<TYPE>(i,1) << " label=" << test_labels.at<int>(i) << "\n";
        }
        std::cout << "\n";
        */
        clf->train(train_data, cv::ml::SampleTypes::ROW_SAMPLE, train_labels);
        
        // validation
        TYPE TP = 0; // true positive
        TYPE TN = 0; // true negative
        TYPE FP = 0; // false positive
        TYPE FN = 0; // false negative
        for(size_t i = 0; i < test_data.rows; ++i) {
            cv::Mat row(1, test_data.cols, MatTYPE, test_data.ptr<TYPE>(i));
            int prediction = clf->predict(row);
            int label = test_labels.at<int>(i);
            if(prediction == label) {
                if(label == 1) ++TP;
                else ++TN;
            }
            if(prediction != label) {
                if(label == 1) ++FN;
                else ++FP;
            }
        }
        TYPE accuracy = (TP+TN)/(TP+TN+FP+FN);
        TYPE sensitivity = (TP)/(TP+FN);
        TYPE specificity = (TN)/(TN+FP);
        TYPE false_positive_rate = (FP)/(FP+TN);
        
        std::cout   << "round=" << k
                    << " accuracy=" << std::setprecision(4) << std::setw(7) << accuracy
                    << " sensitivity=" << std::setprecision(4) << std::setw(7)  << sensitivity
                    << " specificity=" << std::setprecision(4) << std::setw(7)  << specificity
                    << " false_p_r=" << std::setprecision(4) << std::setw(7)  << false_positive_rate << "\n";
        
        accuracies.push_back(accuracy);
        sensitivities.push_back(sensitivity);
        specificities.push_back(specificity);
        false_positive_rates.push_back(false_positive_rate);
    }
    std::cout   << "\n------------- final ------------------------\n";
    std::cout   << "accuracy=" << std::setprecision(4) << std::setw(7)  << compute_mean(accuracies)
                << " sensitivity=" << std::setprecision(4) << std::setw(7)  << compute_mean(sensitivities)
                << " precision=" << std::setprecision(4) << std::setw(7)  << compute_mean(specificities)
                << " false_p_r=" << std::setprecision(4) << std::setw(7)  << compute_mean(false_positive_rates) << "\n\n";
                
    std::cout << "Validation done!\n";
    
    // ----------------------------------------------------------------------------
    // Final training with the whole dataset
    // ----------------------------------------------------------------------------
    clf->train(dataset->getSamples(), cv::ml::SampleTypes::ROW_SAMPLE, dataset->getResponses());
    
    // plot the boundary
    for(int y=0; y<40; ++y){
        for(int x=0; x<40; ++x){
            TYPE d[2] = {(static_cast<TYPE>(x)-mean[0])/var[0],(static_cast<TYPE>(y)-mean[0])/var[0]};
            cv::Mat row(1, mat_data.cols, MatTYPE, &d);
            int prediction = clf->predict(row);
            if(prediction == 1)
                cv::circle(graph, cv::Point(x*alpha, y*alpha), 1, cv::Scalar(10,10,10), 1);
        }
    }
    cv::imshow("data",graph);
    
    clf->save("clf.ext");
    
    std::cout << "Final training + save classifier done!\n";
    
    cv::waitKey();
    
    return 0;
}
