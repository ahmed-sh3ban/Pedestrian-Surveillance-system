#include <iostream>
#include <stdlib.h>
#include <stdexcept>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/core/utility.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;



// Some stativ values:
static std::string intrinsic_filename = "intrinsics.yml";
static std::string extrinsic_filename = "extrinsics.yml";
static std::string disparity_filename = "";
static std::string point_cloud_filename = "";
static std::string _alg = "bm";
static int TotalDetectedPedstrain = 0;
static int SADWindowSize = 9, numberOfDisparities = 16, scale = 1;


void Matching( Mat Left,Mat Right)
{

    // create bm algorithm (block match)
    Ptr<StereoBM> bm = StereoBM::create(16,9);   // SDA Size of the compared windows in the left and in the right image, where the sums of absolute differences are calculated to find corresponding pixels.
    Mat img1, img2;
    // Convert the input image to gray.
     cvtColor(Right,img1, COLOR_BGR2GRAY);
     cvtColor(Left,img2,COLOR_BGR2GRAY);


    if (scale != 1.f)
    {
        Mat temp1, temp2;
        int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), scale, scale, method);
        img2 = temp2;
    }
    // make rectification steps again.

    Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;
    // check whether there are intrisic data or no
    if( !intrinsic_filename.empty() )
    {
        // get intrisic data and extrisic data then rectification.
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", intrinsic_filename.c_str());
            return ;
        }
        // Copy data from file to variables.
        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;

        M1 *= scale;
        M2 *= scale;

        fs.open(extrinsic_filename, FileStorage::READ);
        if(!fs.isOpened())
        {
            printf("Failed to open file %s\n", extrinsic_filename.c_str());
            return;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;

        stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }


    //Set bm data.
    bm->setROI1(roi1);//Calculate the disparities only in these regions
    bm->setROI2(roi2);
    bm->setPreFilterCap(31); // value for the prefiltered image pixels.
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0); // Minimum possible disparity value.
    bm->setNumDisparities(numberOfDisparities); //  Maximum disparity minus minimum disparity. This parameter must be divisible by 16.
    bm->setTextureThreshold(10);// Calculate the disparity only at locations, where the texture is larger than (or at least equal to
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);// disparity variation window
    bm->setSpeckleRange(32);//acceptable range of variation in window
    bm->setDisp12MaxDiff(1); // Maximum allowed difference (in integer pixel units) in the left-right disparity check.
 
   

    Mat disp, disp8;

    // calculate time
    int64 t = getTickCount();
    //compute disparity
    bm->compute(img1, img2, disp);
   
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
   
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
   
        //Display
        namedWindow("left", 1);
        imshow("left", img1);
        namedWindow("right", 1);
        imshow("right", img2);
        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        printf("press any key to continue...");
        fflush(stdout);
        printf("\n");
    

}

static void detectAndDraw(const HOGDescriptor &hog, Mat &img)
{
    vector<Rect> found, found_filtered;
    double t = (double) getTickCount();
    // Run the detector with default parameters. to get a higher hit-rate
    // (and more false alarms, respectively), decrease the hitThreshold and
    // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
    hog.detectMultiScale(img, found, 0, Size(8,8), Size(8,8), 1.05, 2);
    t = (double) getTickCount() - t;
    cout << "detection time = " << (t*1000./cv::getTickFrequency()) << " ms \t" << found.size() << endl;
    TotalDetectedPedstrain += found.size();
    for(size_t i = 0; i < found.size(); i++ )
    {
        Rect r = found[i];

        size_t j;
        // Do not add small detections inside a bigger detection.
        for ( j = 0; j < found.size(); j++ )
            if ( j != i && (r & found[j]) == r )
                break;

        if ( j == found.size() )
            {
             found_filtered.push_back(r);
            }

    }

    for (size_t i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        int x = (rand()+10) %256;
        int y = (rand()+10) %256;
        int z = (rand()+10) %256;
        // The HOG detector returns slightly larger rectangles than the real objects,
        // so we slightly shrink the rectangles to get a nicer output.
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
       
     
        rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
     
    }
    
  
    
}

 void Camera(const HOGDescriptor &hog)
  {
         VideoCapture stream(1);   //0 is the id of video device.
 
         if (!stream.isOpened()) { //check if video device has been initialised
          cout << "cannot open camera";
          }
 
         //unconditional loop..infinty loop
         while (true) 
         {
            Mat cameraFrame;
            
            stream >> cameraFrame;
            detectAndDraw(hog, cameraFrame); 
            imshow("People Detection", cameraFrame);

            int c = waitKey( stream.isOpened() ? 30 : 0 ) & 255;
            if ( c == 'q' || c == 'Q' || c == 27)
            break;
         }
        
  }
   void Disparity(const HOGDescriptor &hog)
  {
      // get the Left and Right image.
      VideoCapture RightCam(1), LeftCam(2);
         
      
        
        
      //check if video device has been initialised
      if (!RightCam.isOpened() || !LeftCam.isOpened()) 
      { 
          
        printf("cannot open camera");
          return;
      }

      while(true)
      {

            Mat Left, Right;
            RightCam >> Right;
            LeftCam >> Left;
            // do Stereo correspondence
            Matching(Left, Right);
            
            char keyBoardInput = (char)waitKey(50);
            if (keyBoardInput == 'q' || keyBoardInput == 'Q') {
            exit(-1);
        }
     }
  }

 void Video(const HOGDescriptor &hog, String path)
 {
    string filename = path;
    VideoCapture capture(filename);

    Mat frame ;
    vector< vector <Mat> > img1;

    // if the path was wrong or have any error break
    if( !capture.isOpened() )
        {
          printf("File path is wrong");
          return;
        }


while(true)
    {
       
        capture >> frame;
        if(frame.empty())
            break;

          // DO the detection process
          detectAndDraw(hog, frame);        
          imshow("People Detection", frame);
         int c = waitKey( capture.isOpened() ? 30 : 0 ) & 255;
         if ( c == 'q' || c == 'Q' || c == 27)
         break;
    }
 }

int main(int argc, char** argv)
{
    // value thats determine which part of the system should work.
    char Option;

    // Default video path
    string DefaultorNo = "skip";
    string Path = "vtest.avi";
  
     if(argc == 2)
      {
          Option = argv[1][0];
      }
    else if(argc > 2)
      {
          Path = argv[2] ;
      }

     //instantiat descripto
     HOGDescriptor hog;
     //Set the SVM Classifier and make the detection detect people
     hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
     switch (Option)
     {
       case 'c':
        Camera(hog);
         break;

       case 'v':     
        printf ("Please enter the path of the video or if you want to use the default write 'skip'\n");
        
        // Get the path of the video
        cin >> DefaultorNo;
        
        if(DefaultorNo != "skip")
        {
          Path = DefaultorNo;
        }
        
        Video(hog,Path);
        break;

       default:
        Disparity(hog);
    }

    printf("Total number of detected people:  %d\n",TotalDetectedPedstrain);


    return 0;
}
