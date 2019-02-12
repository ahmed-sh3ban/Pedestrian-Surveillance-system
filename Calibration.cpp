#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <stdlib.h>

#define timeGap 3000000000U
#define SquareSize 2.5
using namespace cv;
using namespace std;

static void help() {
    cout<<"/******** HELP *******/\n";
    cout << "\nThis program helps you to calibrate the stereo cameras.\n This program generates intrinsics.yml and extrinsics.yml which can be used in Stereo Matching Algorithms.\n";
    cout<<"It also displays the rectified image\n";
    cout<<"\nKeyboard Shortcuts for real time (ie clicking stereo image at run time):\n";
    cout<<"1. Default Mode: Detecting (Which detects chessboard corners in real time)\n";
    cout<<"2. 'c': Starts capturing stereo images (With 2 Sec gap, This can be changed by changing 'timeGap' macro)\n";
    cout<<"3. 'p': Process and Calibrate (Once all the images are clicked you can press 'p' to calibrate)";
    cout<<"\nType ./stereo_calib --help for more details.\n";
    cout<<"\n/******* HELP ENDS *********/\n\n";
}
// Public variables

enum Modes { DETECTING, CAPTURING, CALIBRATING};
Modes mode = DETECTING;
int noOfStereoPairs;
int stereoPairIndex = 0, cornerImageIndex=0;
int goIn = 1;
Mat _leftOri, _rightOri;
int64 prevTickCount;
vector<Point2f> cornersLeft, cornersRight;
vector<vector<Point2f> > cameraImagePoints[2];
Size boardSize;


string prefixLeft;
string prefixRight;
string postfix;
string dir;

int calibType;

Mat displayCapturedImageIndex(Mat);
Mat displayMode(Mat);
bool findChessboardCornersAndDraw(Mat, Mat);
void displayImages();
void saveImages(Mat, Mat, int);
void calibrateStereoCamera(Size);
void calibrateInRealTime(int, int);
void calibrateFromSavedImages(string, string, string, string);

Mat displayCapturedImageIndex(Mat img) {
    std::ostringstream imageIndex;
    imageIndex<<stereoPairIndex<<"/"<<noOfStereoPairs;
    putText(img, imageIndex.str().c_str(), Point(50, 70), FONT_HERSHEY_PLAIN, 0.9, Scalar(0,0,255), 2);
    return img;
}

Mat displayMode(Mat img) {
    String modeString = "DETECTING";
    if (mode == CAPTURING) {
        modeString="CAPTURING";
    }
    else if (mode == CALIBRATING) {
        modeString="CALIBRATED";
    }
    putText(img, modeString, Point(50,50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,255,0), 2);
    if (mode == CAPTURING) {
        img = displayCapturedImageIndex(img);
    }
    return img;
}

bool findChessboardCornersAndDraw(Mat inputLeft, Mat inputRight) {
    _leftOri = inputLeft;
    _rightOri = inputRight;
    bool foundLeft = false, foundRight = false;
    // Change the image to gray scale (Left image and right image)
    cvtColor(inputLeft, inputLeft, COLOR_BGR2GRAY);
    cvtColor(inputRight, inputRight, COLOR_BGR2GRAY);

    //check where is the corner is found or not yet in both image
    foundLeft = findChessboardCorners(inputLeft, boardSize, cornersLeft, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    foundRight = findChessboardCorners(inputRight, boardSize, cornersRight, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    // draw the corner in the left and right images
    drawChessboardCorners(_leftOri, boardSize, cornersLeft, foundLeft);
    drawChessboardCorners(_rightOri, boardSize, cornersRight, foundRight);
    
    // DIsplay corners in the image.
    _leftOri = displayMode(_leftOri);
    _rightOri = displayMode(_rightOri);
    // if both corner in both image found return true else return false
    if (foundLeft && foundRight) {
        return true;
    }
    else {
        return false;
    }
}
//
void displayImages() {
    imshow("Left Image", _leftOri);
    imshow("Right Image", _rightOri);
}
// Saving image.
void saveImages(Mat leftImage, Mat rightImage, int pairIndex) {

    cameraImagePoints[0].push_back(cornersLeft);
    cameraImagePoints[1].push_back(cornersRight);

    if (calibType == 1) {
        cvtColor(leftImage, leftImage, COLOR_BGR2GRAY);
        cvtColor(rightImage, rightImage, COLOR_BGR2GRAY);
        std::ostringstream leftString, rightString;
        //Set name for the saved image.
        leftString<<dir<<"/"<<prefixLeft<<pairIndex<<"."<<postfix;
        rightString<<dir<<"/"<<prefixRight<<pairIndex<<"."<<postfix;
        //Save image
        imwrite(leftString.str().c_str(), leftImage);
        imwrite(rightString.str().c_str(), rightImage);
    }
}

void calibrateStereoCamera(Size imageSize) {
    vector<vector<Point3f> > objectPoints;
    objectPoints.resize(noOfStereoPairs);
    for (int i=0; i<noOfStereoPairs; i++) {
        for (int j=0; j<boardSize.height; j++) {
            for (int k=0; k<boardSize.width; k++) {
                objectPoints[i].push_back(Point3f(float(j * SquareSize),float(k * SquareSize),0.0));
            }
        }
    }
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
    // find intersic and extersic paramter
    double rms = stereoCalibrate(objectPoints, cameraImagePoints[0], cameraImagePoints[1],
                                 cameraMatrix[0], distCoeffs[0],
                                 cameraMatrix[1], distCoeffs[1],
                                 imageSize, R, T, E, F,
                                 CALIB_FIX_ASPECT_RATIO +
                                 CALIB_ZERO_TANGENT_DIST +
                                 CALIB_SAME_FOCAL_LENGTH +
                                 CALIB_RATIONAL_MODEL +
                                 CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                                 TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout<<"RMS Error: "<<rms<<"\n"; // re-projection error

    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(int i = 0; i < noOfStereoPairs; i++ )
    {
        // Go throught number of images.

        // Get the size of captured image
        int npt = (int)cameraImagePoints[0][i].size();

        Mat imgpt[2];
        for(int k = 0; k < 2; k++ )
        {
            // Get the points from image [k][i] where k is index of camera and i go through the points in the image
            imgpt[k] = Mat(cameraImagePoints[k][i]);

            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]); // the first imgpt is the input point and the second imgpt is the output point with undisort

            // construct the correspond Epil the output is  epoplar lines where k is the index of the image (1 or 2) that contains the points
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(int j = 0; j < npt; j++ )
        {

            // Camera Image points where [0] is the index of the camera, i is the index image and [j] is the index of points.
            // Lines [1] index of the camera [j] is index of the lines ([0], [1]) index of the points of the line.
            double errij = fabs(cameraImagePoints[0][i][j].x*lines[1][j][0] +
                                cameraImagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
            fabs(cameraImagePoints[1][i][j].x*lines[0][j][0] +
                 cameraImagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "Average Reprojection Error: " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs("intrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
        "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout<<"Error: Could not open intrinsics file.";
    // R1-> rotation of the first camera. R2-> rotation of the second camera.
    Mat R1, R2, P1, P2, Q;
    Rect validROI[2];

    // Extrinsic paramter and   
    stereoRectify(cameraMatrix[0], distCoeffs[0], cameraMatrix[1], distCoeffs[1], imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validROI[0], &validROI[1]);
    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened()) {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout<<"Error: Could not open extrinsics file";
    // OpenCV can handle left-right
    // or up-down camera arrangements

    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));
    Mat rmap[2][2];
    // computes undistortion and rectification transformation map.
    //R – Optional rectification transformation in the object space (3x3 matrix). R1 or R2 , computed by stereoRectify()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    Mat canvas;
    double sf;
    int w, h;

    if (!isVerticalStereo) {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }
    String file;
    namedWindow("rectified");
    for (int i=0; i < noOfStereoPairs; i++) { // go through all the images
        for (int j=0; j < 2; j++) {  // j < 2 because we have 2 images
            if (j==0) {
                file = prefixLeft;
            }
            else if (j==1) {
                file = prefixRight;
            }
            ostringstream st;
            st<<dir<<"/"<<file<<i+1<<"."<<postfix;
            Mat img = imread(st.str().c_str()), rimg, cimg;
            remap(img, rimg, rmap[j][0], rmap[j][1], INTER_LINEAR);
            cimg=rimg;
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*j, 0, w, h)) : canvas(Rect(0, h*j, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            Rect vroi(cvRound(validROI[j].x*sf), cvRound(validROI[j].y*sf),
                      cvRound(validROI[j].width*sf), cvRound(validROI[j].height*sf));
            rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
        }
        if( !isVerticalStereo )
            for(int j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for(int j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
    }
}

// if we want to calibrate in realtime so we should use the 2 cameras
void calibrateInRealTime(int cam1, int cam2) {

    VideoCapture camLeft(cam1), camRight(cam2);
    // if left camera and right camera is closed exit (error)
    if (!camLeft.isOpened() || !camRight.isOpened()) {
        cout<<"Error: Stereo Cameras not found or there is some problem connecting them. Please check your cameras.\n";
        exit(-1);
    }
    Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
    bool foundCornersInBothImage = false; // default is false we still didnt find any corners
    for( ; ; ) {
        camLeft>>inputLeft;    // get frames of left camera
        camRight>>inputRight; // get frames from the right camera
        // to be able to do calibration the number of frames captured from right camera should be qeual to the number of the frames captured
        // in left camera if not exit.
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout<<"Error: Images from both cameras are not of some size. Please check the size of each camera.\n";
            exit(-1);
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        
        // find coreners in both left and right image (camera captured)
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        // if the mode is capturing and number of images is less than the need number of the calibration
        if (foundCornersInBothImage && mode == CAPTURING && stereoPairIndex < noOfStereoPairs) {
            // set time
            int64 thisTick = getTickCount();
            int64 diff = thisTick - prevTickCount;
            if (goIn==1 || diff >= timeGap) {
                goIn=0;
                // Save image
                saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
                prevTickCount = getTickCount();
            }
        }

        displayImages();
        if (mode == CALIBRATING) {
            calibrateStereoCamera(inputLeft.size());
            waitKey();
        }
        char keyBoardInput = (char)waitKey(50);
        if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            exit(-1);
        }
        else if(keyBoardInput == 'c' || keyBoardInput == 'C') {
            mode = CAPTURING;
        }
        else if (keyBoardInput == 'p' || keyBoardInput == 'P') {
            mode = CALIBRATING;
        }
    }
}
// if we want to calibrate offline and we have save images
void calibrateFromSavedImages(string dr, string prel, string prer, string post) {
    Size imageSize;
    for (int i=0; i<noOfStereoPairs; i++) {
        Mat inputLeft, inputRight, copyImageLeft, copyImageRight;
        ostringstream imgIndex;
        imgIndex << i+1;
        bool foundCornersInBothImage = false;
        string sourceLeftImagePath, sourceRightImagePath;
        sourceLeftImagePath = dr+"/"+prel+imgIndex.str()+"."+post;
        sourceRightImagePath = dr+"/"+prer+imgIndex.str()+"."+post;
        inputLeft = imread(sourceLeftImagePath);
        inputRight = imread(sourceRightImagePath);
        imageSize = inputLeft.size();
        if (inputLeft.empty() || inputRight.empty()) {
            cout<<"\nCould no find image: "<<sourceLeftImagePath<<" or "<<sourceRightImagePath<<". Skipping images.\n";
            continue;
        }
        if ((inputLeft.rows != inputRight.rows) || (inputLeft.cols != inputRight.cols)) {
            cout<<"\nError: Left and Right images are not of some size. Please check the size of the images. Skipping Images.\n";
            continue;
        }
        inputLeft.copyTo(copyImageLeft);
        inputRight.copyTo(copyImageRight);
        foundCornersInBothImage = findChessboardCornersAndDraw(inputLeft, inputRight);
        if (foundCornersInBothImage && stereoPairIndex<noOfStereoPairs) {
            saveImages(copyImageLeft, copyImageRight, ++stereoPairIndex);
        }
        displayImages();
    }
    if(stereoPairIndex > 2) {
        calibrateStereoCamera(imageSize);
        waitKey();
    }
    else {
        cout<<"\nInsufficient stereo images to calibrate.\n";
    }
}

int main(int argc, char** argv) {
    help();
    const String keys =
    "{help| |Prints this}"
    "{h height|6|Height of the board}"
    "{w width|9|Width of the board}"
    "{rt realtime|1|Clicks stereo images before calibration. Use if you do not have stereo pair images saved}"
    "{n images|20|No of stereo pair images}"
    "{dr folder|.|Directory of images}"
    "{prel prefixleft|image_left_|Left image name prefix. Ex: image_left_}"
    "{prer prefixright|image_right_|Right image name postfix. Ex: image_right_}"
    "{post postfix|jpg|Image extension. Ex: jpg,png etc}"
    "{cam1|1|Camera 1 Index}"
    "{cam2|2|Camera 2 Index}";
    CommandLineParser parser(argc, argv, keys);
    if(parser.has("help")) {
        parser.printMessage();
        exit(-1);
    }
    // board size (chessboard) 9x6
    boardSize = Size(parser.get<int>("w"), parser.get<int>("h"));
    // number of images including left and right images thats been taking from left, right camera
    noOfStereoPairs = parser.get<int>("n");
    prefixLeft = parser.get<string>("prel");
    prefixRight = parser.get<string>("prer");
    postfix = parser.get<string>("post");
    // directort of image
    dir =parser.get<string>("dr");
    // calib in realtime or capturing
    calibType = parser.get<int>("rt");

    // window to display content of for Left camera
    namedWindow("Left Image");

   // window to display content of for Right camera
    namedWindow("Right Image");
    //choose whether we need calibrating online or offline
    switch (calibType) {
        case 0:
            calibrateFromSavedImages(dir, prefixLeft, prefixRight, postfix);
            break;
        case 1:
            calibrateInRealTime(parser.get<int>("cam1"), parser.get<int>("cam2"));
            break;
        default:
            cout<<"-rt should be 0 or 1. \n";
            break;
    }
    return 0;
}
