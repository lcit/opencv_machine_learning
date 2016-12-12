C++ - Simple CPU implementation of the HOG (Histogram of Oriented Grandients) based on OpenCV's utility functions.

The reference article: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
'''
int main(int argc, char* argv[]){

    // open and display an image
    cv::Mat image = cv::imread(argv[1], CV_8U);
    cv::imshow("original", image);
    
    // Retrieve the HOG from the image
    size_t blocksize = 64;
    size_t cellsize = 32;
    size_t stride = 32;
    size_t binning = 9;
    HOG hog(blocksize, cellsize, stride, binning);
    auto hist = hog.convert(image);

	// print resulting histogram
    std::cout << "Histogram size: " << hist.size() << "\n";
    for(auto h:hist)
        std::cout << h << ",";
    std::cout << "\n";
}
'''
![alt tag](https://raw.githubusercontent.com/lcit/HOG/master/img/HOG.png)
