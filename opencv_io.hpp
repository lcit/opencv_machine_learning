/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: opencv_io.hpp
	Last modifed:   08.12.2016 by Leonardo Citraro
	Description:    OpenCV utilities classes for input/output streams.

	=========================================================================

	=========================================================================
*/

#ifndef __OPENCV_IO_HPP__
#define __OPENCV_IO_HPP__

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>
#include <map>
#include <memory>

class InputStream {
    private:
        const std::string _input;
        cv::VideoCapture cap;
        
    public:
        InputStream(const char* input) : _input(input) {}
        virtual ~InputStream();
        
        virtual int open();
        
        virtual void close();
        
        virtual const int n_frames();
        
        virtual const int height();
        
        virtual const int width();
        
        virtual double fps();
        
        virtual int frame_number();
        
        virtual bool next_frame(cv::Mat& frame);
        
        virtual double timestamp();
};

class RecordStream {
    private:
        const std::string _filename;
        const int _fourcc;
        const int _fps;
        const int _width;
        const int _height;
        cv::VideoWriter out;
        
    public:
        RecordStream(   const std::string& filename, const double fps = 25, 
                        const int width = 1920, const int height = 1080,
                        const int fourcc = CV_FOURCC('m','p', '4', 'v')) 
                        : _filename(filename), _fourcc(fourcc), _fps(fps),
                        _width(width), _height(height) {}
        
        virtual ~RecordStream();
        
        virtual bool open();
        
        virtual void close();
        
        virtual const int height();
        
        virtual const int width();
        
        virtual double fps();
        
        virtual int fourcc();
        
        virtual void next_frame(cv::Mat& frame);
};

class TrackBar {
    private:
        const std::string& _name;
        const std::string& _windowname;
        int* _value;
        const int _max_value;
        const int _min_value;
        const cv::TrackbarCallback _onChange;
        void* _userdata;
    public:
        TrackBar(const std::string& name, const std::string& windowname, const int min_value, const int max_value);
        ~TrackBar();
        
        virtual int get_value();
        
        virtual void set_value(int value);
};

class DisplayStream {
    private:
        const std::string _windowname;
        const int _width;
        const int _height;
        const int _x;
        const int _y;
        const int _flags;
        std::map<std::string,std::unique_ptr<TrackBar>> trackbars;
        
    public:
        DisplayStream(  const std::string& windowname, 
                        const int flags = CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED) 
                        : _windowname(windowname), _flags(flags), _width(-1), _height(-1),
                        _x(-1), _y(-1) {}
        DisplayStream(  const std::string& windowname, const int width, const int height,
                        const int flags = CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED) 
                        : _windowname(windowname), _flags(flags), _width(width), _height(height),
                        _x(-1), _y(-1) {}
        DisplayStream(  const std::string& windowname, const int width, const int height,
                        const int x, const int y,
                        const int flags = CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED) 
                        : _windowname(windowname), _flags(flags), _width(width), _height(height), 
                        _x(x), _y(y) {}
        
        virtual ~DisplayStream() {}
        
        virtual void open();
        
        virtual void close();
        
        virtual const int height();
        
        virtual const int width();
        
        virtual void next_frame(cv::Mat& frame);
        
        virtual void add_trackbar(const std::string& trackbarname, const int min_value, const int max_value);
        
        virtual int get_trackbar_value(const std::string& trackbarname);
        
        virtual void set_trackbar_value(const std::string& trackbarname, int value);
};

#endif
