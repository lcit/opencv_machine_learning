/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: main.cpp
	Last modifed:   11.12.2016 by Leonardo Citraro
	Description:    Test of the HOG feature

	=========================================================================
    https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
	=========================================================================
*/
#include "opencv_io.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <math.h>

class HOG {     
    public:
        using TType = float;
        using THist = std::vector<TType>;
        
        static constexpr TType epsilon = 1e-6;
        static constexpr TType epsilon_squared = epsilon*epsilon;
        
        // see: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#Block_normalization
        static const inline void L2norm(THist& v) {
            THist temp = v; 
            std::transform(std::begin(v), std::end(v), std::begin(temp), [](const TType& x){ return x*x; });
            TType den = std::accumulate(std::begin(temp), std::end(temp), 0);
            den = std::sqrt(den + epsilon_squared);
            if(den != 0)
                std::transform(std::begin(v), std::end(v), std::begin(v), [den](TType nom){ return nom/den; });
        }
        
        
    private:    
        const size_t _blocksize;
        const size_t _cellsize;
        const size_t _stride;
        const size_t _binning; ///< the number of bins for each cell-histogram
        const size_t _bin_width; ///< size of one bin in degree
        const std::function<void(THist&)> _block_norm; ///< function that normalize the block histogram
        const cv::Mat _kernelx = (cv::Mat_<char>(1,3) << -1,0,1); ///< derivive kernel
        const cv::Mat _kernely = (cv::Mat_<char>(3,1) << -1,0,1); ///< derivive kernel
        
        cv::Mat mag, ori, norm;
        THist img_hist;
        std::vector<THist> _all_hists;
        
    public:
        HOG(const size_t blocksize, 
            std::function<void(THist&)> block_norm = L2norm) 
            : _blocksize(blocksize), _cellsize(blocksize/2), _stride(blocksize/2), _binning(9),
            _bin_width(360 / _binning), _block_norm(block_norm) {}
        HOG(const size_t blocksize, size_t cellsize, 
            std::function<void(THist&)> block_norm = L2norm) 
            : _blocksize(blocksize), _cellsize(cellsize), _stride(blocksize/2), _binning(9),
            _bin_width(360 / _binning), _block_norm(block_norm) {}
        HOG(const size_t blocksize, size_t cellsize, size_t stride, 
            std::function<void(THist&)> block_norm = L2norm) 
            : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(9),
            _bin_width(360 / _binning), _block_norm(block_norm) {}
        HOG(const size_t blocksize, size_t cellsize, size_t stride, size_t binning = 9, 
            std::function<void(THist&)> block_norm = L2norm) 
            : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(binning),
            _bin_width(360 / _binning), _block_norm(block_norm) {}
        ~HOG() {}
        
        /// Retrieves the HOG from an image
        ///
        /// @param img: source image (any size)
        /// @return the HOG histogram as std::vector
        THist convert(const cv::Mat& img) {
            // makes sure the image is normalized
            cv::normalize(img, norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);
            
            // extracts the magnitude and orientations images
            magnitude_and_orientation(img, mag, ori);
            
            // iterates over all blocks and cells
            for(size_t y=0; y<=mag.rows-_blocksize; y+=_stride) {
                for(size_t x=0; x<=mag.cols-_blocksize; x+=_stride) {
                    cv::Rect block_rect = cv::Rect(x, y, _blocksize, _blocksize);
                    THist block_hist = process_block(cv::Mat(mag, block_rect), cv::Mat(ori, block_rect));
                    
                    // concatenate all the blocks histograms
                    img_hist.insert(std::end(img_hist), std::begin(block_hist), std::end(block_hist));
                }
            }
            return img_hist;
        }
    
    private:
        /// Retrieves magnitude and orientation form an image
        ///
        /// @param img: source image (any size)
        /// @param mag: ref. to the magnitude matrix where to store the result
        /// @param pri: ref. to the orientation matrix where to store the result
        /// @return none
        void magnitude_and_orientation(const cv::Mat& img, cv::Mat& mag, cv::Mat& ori) {
            cv::Mat Dx, Dy;
            cv::filter2D(img, Dx, CV_32F, _kernelx);
            cv::filter2D(img, Dy, CV_32F, _kernely);
            cv::magnitude(Dx, Dy, mag);
            cv::phase(Dx, Dy, ori, true);
        }
        
        /// Iterates over a block and concatenates the cell histograms
        ///
        /// @param block_mag: a portion (block) of the magnitude matrix
        /// @param block_ori: a portion (block) of the orientation matrix
        /// @return the block histogram as std::vector
        THist process_block(const cv::Mat& block_mag, const cv::Mat& block_ori) {
            THist block_hist_concat;
            for(size_t y=0; y<block_mag.rows; y+=_cellsize) {
                for(size_t x=0; x<block_mag.cols; x+=_cellsize) {
                    //std::cout << "Cell x:" << x << ", y:" << y << "\n";
                    cv::Rect cell_rect = cv::Rect(x, y, _cellsize, _cellsize);
                    cv::Mat cell_mag = cv::Mat(block_mag, cell_rect);
                    cv::Mat cell_ori = cv::Mat(block_ori, cell_rect);
                    THist cell_hist = process_cell(cell_mag, cell_ori);
                    block_hist_concat.insert(std::end(block_hist_concat), std::begin(cell_hist), std::end(cell_hist));
                }
            }
            _block_norm(block_hist_concat); // inplace normalization
            _all_hists.push_back(block_hist_concat);
            return block_hist_concat;
        }
        
        /// Iterates over a cell to create the cell histogram
        ///
        /// @param cell_mag: a portion of a block (cell) of the magnitude matrix
        /// @param cell_ori: a portion of a block (cell) of the orientation matrix
        /// @return the cell histogram as std::vector
        THist process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) {
            THist cell_hist(_binning);
            for(size_t i=0; i<cell_mag.rows; ++i) {
                for(size_t j=0; j<cell_mag.cols; ++j) {
                    const TType* ptr_row_mag = cell_mag.ptr<TType>(i);
                    const TType* ptr_row_ori = cell_ori.ptr<TType>(i);
                    cell_hist[static_cast<int>(ptr_row_ori[j]/_bin_width)] += ptr_row_mag[j];
                }
            }
            return cell_hist;
        }
        
    public:
        /// Utility funtion to retreve the magnitude matrix
        ///
        /// @return the magnitude matrix CV_32F
        cv::Mat get_magnitudes() { 
            return mag;
        }
        
        /// Utility funtion to retreve the orientation matrix
        ///
        /// @return the orientation matrix CV_32F
        cv::Mat get_orientations() { 
            return ori; 
        }
        
        /// Utility funtion to retreve a mask of vectors
        ///
        /// @return the vector matrix CV_32F
        cv::Mat get_vector_mask() {
            cv::Mat vector_mask = cv::Mat::zeros(norm.size(), CV_8U);
            
            // retrieve the max value of the final HOG histogram
            //TType hist_max = *std::max_element(std::begin(img_hist), std::end(img_hist));
            TType hist_mean = std::accumulate(std::begin(img_hist), std::end(img_hist), 0.0)/img_hist.size();
            
            // iterates over all blcoks and cells
            size_t idx_blocks = 0;
            size_t y,x,i,j;
            for(y=0; y<=norm.rows-_blocksize; y+=_stride) {
                for(x=0; x<=norm.cols-_blocksize; x+=_stride) {
                    size_t idx_cells = 0;
                    for(i=0; i<_blocksize; i+=_cellsize) {
                        for(j=0; j<_blocksize; j+=_cellsize) {
                            // retrieves the cell histogram
                            THist block_hist = _all_hists[idx_blocks];
                            THist cell_hist(_binning);
                            std::copy(  std::begin(block_hist) + idx_cells*_binning, 
                                        std::begin(block_hist) + (idx_cells+1)*_binning, 
                                        std::begin(cell_hist) );
                            
                            // retrieve the max value of the this cell histogram
                            TType max = *std::max_element(std::begin(cell_hist), std::end(cell_hist));
                            //std::cout << "max=" << max << "\n";
                            TType mean = std::accumulate(std::begin(cell_hist), std::end(cell_hist), 0.0)/cell_hist.size();
                            //std::cout << "mean=" << mean << "\n";
                            
                            int color_magnitude = static_cast<int>(mean/hist_mean*255.0);
                            //std::cout << "color_magnitude=" << color_magnitude << "\n";
                            
                            // iterates over the cell histogram
                            for(size_t k=0; k<cell_hist.size(); ++k) {
                                // fixed line thinkness
                                int thickness = 2; 
                                
                                // length of the "arrows"
                                int length = static_cast<int>((cell_hist[k]/max)*_cellsize/2);
                                if(length > 0 && !isinf(length)){
                                    // draw "arrows" of varing length
                                    cv::line(vector_mask, cv::Point(x+j+_cellsize/2, y+i+_cellsize/2), 
                                        cv::Point(  x+j+_cellsize/2+cos((k*_bin_width)*3.1415/180)*length,
                                                    y+i+_cellsize/2+sin((k*_bin_width)*3.1415/180)*length), 
                                        cv::Scalar(color_magnitude,color_magnitude,color_magnitude), thickness);
                                }
                            }
                            // draw cell delimiters
                            cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i+norm.rows-1,y+j-1), cv::Scalar(255,255,255), 1);
                            cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i-1, y+j+norm.rows-1), cv::Scalar(255,255,255), 1);
                            idx_cells++;
                        }
                        // draw cell delimiters
                        cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i+norm.rows-1,y+j-1), cv::Scalar(255,255,255), 1);
                        cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i-1, y+j+norm.rows-1), cv::Scalar(255,255,255), 1);
                    }
                    // draw cell delimiters
                    cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i+norm.rows-1,y+j-1), cv::Scalar(255,255,255), 1);
                    cv::line(vector_mask, cv::Point(x+i-1, y+j-1), cv::Point(x+i-1, y+j+norm.rows-1), cv::Scalar(255,255,255), 1);
                    idx_blocks++;
                }
                // draw cell delimiters
                cv::line(vector_mask, cv::Point(x-1, y-1), cv::Point(x+norm.rows-1,y-1), cv::Scalar(255,255,255), 1);
                cv::line(vector_mask, cv::Point(x-1, y-1), cv::Point(x-1, y+norm.rows-1), cv::Scalar(255,255,255), 1);
            }
            return vector_mask;
        }
};

void display_superimposed(const cv::Mat& A, const cv::Mat& B, const std::string& name) {
    cv::Mat superimposed;
    cv::addWeighted(A, 0.5, B, 0.5, 0.0, superimposed);
    imshow("superimposed_"+name, superimposed );
}

cv::Mat custom_normalization(const cv::Mat& src) {
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    
    cv::Mat dst = src*200/(max-min)+128;
    dst.convertTo(dst,CV_8U);
    return dst;
}

int main(int argc, char* argv[]){
    
    std::vector<float> v = {1,6,7,2,2,2,3};
    HOG::L2norm(v);
    for(auto vv:v)
        std::cout << vv << ",";

    // open and display an image
    cv::Mat image = cv::imread(argv[1], CV_8U);
    cv::imshow("original", image);
    
    // Retrieve the HOG from the image
    size_t blocksize = atoi(argv[2]);
    size_t cellsize = atoi(argv[3]);
    size_t stride = atoi(argv[4]);
    size_t binning = atoi(argv[5]);
    HOG hog(blocksize, cellsize, stride, binning);
    auto hist = hog.convert(image);
    
    // print the resulting histogram
    std::cout << "Histogram size: " << hist.size() << "\n";
    for(auto h:hist)
        std::cout << h << ",";
    std::cout << "\n";

    // display some usefull images
    display_superimposed(image, hog.get_vector_mask(), "vector_mask");
    display_superimposed(custom_normalization(hog.get_magnitudes()), hog.get_vector_mask(), "magnitude");
    display_superimposed(custom_normalization(hog.get_orientations()), hog.get_vector_mask(), "orientation");
    
    cv::waitKey();
    
    return 0;

}
