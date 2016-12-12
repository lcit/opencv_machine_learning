/*	=========================================================================
	Author: Leonardo Citraro
	Company: 
	Filename: opencv_io.cpp
	Last modifed:   08.12.2016 by Leonardo Citraro
	Description:    OpenCV utilities classes for input/output streams.

	=========================================================================

	=========================================================================
*/
#include "opencv_io.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdexcept>
#include <memory>

InputStream::~InputStream() {
    cap.release();
}

int InputStream::open() {
    cap = cv::VideoCapture(_input);
    return cap.isOpened();
}

void InputStream::close() {
    cap.release();
}

const int InputStream::n_frames() {
    return static_cast<int>(cap.get(CV_CAP_PROP_FRAME_COUNT ));
}

const int InputStream::height() {
    return static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
}

const int InputStream::width() {
    return static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
}

double InputStream::fps() {
    return cap.get(CV_CAP_PROP_FPS);
}

int InputStream::frame_number() {
    return static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
}

bool InputStream::next_frame(cv::Mat& frame){
    if(!cap.read(frame))
        return false;
    return true;
}

double InputStream::timestamp() {
    return cap.get(CV_CAP_PROP_POS_MSEC);
}






        
RecordStream::~RecordStream() {
    out.release();
}

bool RecordStream::open() {
    out = cv::VideoWriter(_filename, _fourcc, _fps, cv::Size(width(),height()));
    return out.isOpened();
}

void RecordStream::close() {
    out.release();
}

const int RecordStream::height() {
    return _height;
}

const int RecordStream::width() {
    return _width;
}

double RecordStream::fps() {
    return _fps;
}

int RecordStream::fourcc() {
    return _fourcc;
}

void RecordStream::next_frame(cv::Mat& frame){
    //out.write(frame);
    out << frame;
}







TrackBar::TrackBar(const std::string& name, const std::string& windowname, const int min_value, const int max_value) 
    : _name(name), _windowname(windowname), _value(new int(0)), _min_value(min_value),
    _max_value(max_value), _onChange(0), _userdata(0) {
    cv::createTrackbar(_name, _windowname, _value, _max_value, _onChange, _userdata);
}
TrackBar::~TrackBar() {
    delete _value;
}

int TrackBar::get_value() {
    //return *_value;
    int value = cv::getTrackbarPos(_name, _windowname);
    if(value < _min_value) {
        cv::setTrackbarPos(_name, _windowname, _min_value);
        return _min_value;
    }
    return value;
}

void TrackBar::set_value(int value) {
    //*_value = value;
    if(value < _min_value)
        cv::setTrackbarPos(_name, _windowname, _min_value);
    else
        cv::setTrackbarPos(_name, _windowname, value);
}



        
void DisplayStream::open() {
    cv::namedWindow(_windowname, _flags);
    if(_width != -1 && _height != -1){
        cv::resizeWindow(_windowname, _width, _height);
    }
    if(_x != -1 && _y != -1){
        cv::moveWindow(_windowname, _x, _y);
    }
}

void DisplayStream::close() {
    cv::destroyWindow(_windowname);
}

const int DisplayStream::height() {
    return _height;
}

const int DisplayStream::width() {
    return _width;
}

void DisplayStream::next_frame(cv::Mat& frame){
    cv::imshow(_windowname, frame);
}

void DisplayStream::add_trackbar(const std::string& trackbarname, const int min_value, const int max_value) {
    trackbars[trackbarname] = std::make_unique<TrackBar>(trackbarname, _windowname, min_value, max_value);
    //auto ptr = std::make_unique<int>(1);
    //trackbars.insert( std::pair<std::string,TrackBar>(trackbarname,TrackBar(trackbarname, _windowname, max_value)) );
}

int DisplayStream::get_trackbar_value(const std::string& trackbarname) {
    return trackbars[trackbarname]->get_value();
}

void DisplayStream::set_trackbar_value(const std::string& trackbarname, int value) {
    trackbars[trackbarname]->set_value(value);
}


