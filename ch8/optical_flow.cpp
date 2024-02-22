#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "iostream"
using namespace std;
using namespace cv;

string file_1 = "../LK1.png";  // first image
string file_2 = "../LK2.png";  // second image

/// Optical flow tracker and interface
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1_,
        const Mat &img2_,
        const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);//定义calculateOpticalFlow（计算光流）函数
    //Range中有两个关键的变量start和end  Range可以用来表示矩阵的多个连续的行或列
    //Range表示范围从start到end，包含start，但不包含end


private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;//true if a keypoint is tracked successfully 关键点跟踪是正确的
    bool inverse = true;//bool型变量 判断是否采用反向光流
    bool has_initial = false;
};

/**
 * single level optical flow单层光流
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default多层光流
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

//双线性插值求灰度值
//inline表示内联函数，它是为了解决一些频繁调用的小函数大量消耗栈空间的问题而引入的
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;
    
    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv) {

    // images, note they are CV_8UC1, not CV_8UC3
    Mat img1 = imread(file_1, 0);
    Mat img2 = imread(file_2, 0);

    // key points, using GFTT here.
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    //GFTTDetector描述子
    //三个参数从左到右依次为
    //maxCorners表示最大角点数目。在此处为500。
    //qualityLevel表示角点可以接受的最小特征值，一般0.1或者0.01，不超过1。在此处为0.01。
    //minDistance表示角点之间的最小距离。在此处为20。
    detector->detect(img1, kp1);

    // now lets track these key points in the second image
    // first use single level LK in the validation picture
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // then test multi-level LK
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi, true);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by gauss-newton: " << time_used.count() << endl;

    // use opencv's flow for validation
    vector<Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    // plot the differences of those functions
    Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if (success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multi level", img2_multi);
    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse, bool has_initial) 
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    //定义了一个OpticalFlowTracker类型的变量tracker，并进行了初始化
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
    //parallel_for_()实现并行调用OpticalFlowTracker::calculateOpticalFlow()的功能
    //代码在 calculateOpticalFlow 函数中实现了单层光流函数，
    //其中调用了 cv::parallel_for_ 并行调用 OpticalFlowTracker::calculateOpticalFlow 该函数计算指定范围内特征点的光流。
    //这个并行 for 循环内部是 Intel tbb 库实现的，我们只需按照其接口，将函数本体定义出来，然后将函数作为 std::function 对象传递给它。
}
//使用高斯牛顿法求解图像2中相应的角点坐标
void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;//最大迭代次数
    for (size_t i = range.start; i < range.end; i++)//对图像1中的每个GFTT角点进行高斯牛顿优化
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated 优化变量
        if (has_initial)//如果kp2进行了初始化，则执行
        {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }
        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded
        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian 将H初始化为0
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias 将b初始化为0
        Eigen::Vector2d J;  // jacobian 雅克比矩阵J
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) 
            {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } 
            
            else 
            {
                // only reset b 只重置矩阵b。在反向光流中，海塞矩阵H在整个高斯牛顿迭代过程中均保持不变
                b = Eigen::Vector2d::Zero();
            }
            cost = 0;//代价初始化为0 
            // compute cost and jacobian 计算代价和雅克比矩阵
            //这个循环是在做什么，在patch里遍历？
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++)  //x,y是patch内遍历
                {
                    //(u, v)表示图像中的角点u表示x坐标，v表示y坐标
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    //误差 eij = I1(u+i,v+j)-I2(U+I+Δu,v+j+Δv)
                    //i  -> kp.pt.x
                    //j  -> kp.pt.y
                    //u  -> x
                    //v  -> y              
                    //Δu -> dx
                    //Δv -> dy
                    // Jacobian
                    
                    if (inverse == false) 
                    {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );//dx,dy是优化变量 即（Δu，Δv） 计算雅克比矩阵
                    //相当于 J = - [ {I2( u + i + Δu + 1,v + j + Δv)-I2(u + i + Δu - 1,v + j + Δv)}/2,I2( u + i + Δu ,v + j + Δv + 1)-I2( u + i + Δu,v + j + Δv - 1)}/2]T T表示转置
                    //I2 -> 图像2的灰度信息
                    //u  -> x
                    //v  -> y
                    //Δu -> dx
                    //Δv -> dy
                    //i  -> kp.pt.x
                    //j  -> kp.pt.y
                    } else if (iter == 0) //采用反向光流时
                    {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );//dx,dy是优化变量 即（Δu，Δv） 计算雅克比矩阵
                    //相当于 J = - [ {I1( u + i + 1,v + j )-I1(u + i - 1,v + j )}/2,I1( u + i,v + j + 1)-I1( u + i ,v + j - 1)}/2]T T表示转置
                    //I2 -> 图像2的灰度信息
                    //i -> x
                    //j -> y
                    //u  -> kp.pt.x
                    //v  -> kp.pt.y
                    }
                    // compute H, b and set cost;
                    b += -error * J;//b = -Jij * eij(累加和)
                    cost += error * error;//cost = || eij ||2 2范数
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();//H = Jij Jij(T)(累加和)
                    }
                }
            // compute update
            //求解增量方程，计算更新量
            Eigen::Vector2d update = H.ldlt().solve(b);
            if (std::isnan(update[0]))//计算出来的更新量是非数字，光流跟踪失败，退出GN迭代
            {
                // sometimes occurred when we have a black or white patch and H is irreversible
                std::cout << "update is nan" << endl;
                succ = false;
                break;
            }
            if (iter > 0 && cost > lastCost) //代价不再减小，退出GN迭代
            {
                break;
            }
            // update dx, dy 更新优化变量和lastCost
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;
            if (update.norm() < 1e-2) //更新量的模小于1e-2，退出GN迭代
            {
                // converge
                break;
            }
        }//GN法进行完一次迭代
        success[i] = succ;
        // set kp2
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}//对图像1中的所有角点都完成了光流跟踪
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse) {
    // parameters
    int pyramids = 4;//金字塔层数为4
    double pyramid_scale = 0.5;//每层之间的缩放因子设为0.5
    double scales[] = {1.0, 0.5, 0.25, 0.125};
    // create pyramids 创建图像金字塔
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
    vector<Mat> pyr1, pyr2; // image pyramids pyr1 -> 图像1的金字塔 pyr2 -> 图像2的金字塔
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) 
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } 
        else 
        {
            Mat img1_pyr, img2_pyr;
            //将图像pyr1[i-1]的宽和高各缩放0.5倍得到图像img1_pyr
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            //将图像pyr2[i-1]的宽和高各缩放0.5倍得到图像img2_pyr
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
    cout << "build pyramid time: " << time_used.count() << endl;//输出构建图像金字塔的耗时
    // coarse-to-fine LK tracking in pyramids 由粗至精的光流跟踪
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp:kp1) 
    {
        auto kp_top = kp;//这里意思大概是视觉slam十四讲p215的把上一层的追踪结果作为下一层光流的初始值
        kp_top.pt *= scales[pyramids - 1];//
        kp1_pyr.push_back(kp_top);//最顶层图像1的角点坐标
        kp2_pyr.push_back(kp_top);//最顶层图像2的角点坐标：用图像1的初始化图像2的
    }
    for (int level = pyramids - 1; level >= 0; level--)//从最顶层开始进行光流追踪
    {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();//开始计时
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        //has_initial设置为true，表示图像2中的角点kp2_pyr进行了初始化
        t2 = chrono::steady_clock::now();//计时结束
        auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
        cout << "track pyr " << level << " cost time: " << time_used.count() << endl;//输出光流跟踪耗时
        if (level > 0) 
        {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;//pyramidScale等于0.5，相当于乘了2
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;//pyramidScale等于0.5，相当于乘了2
        }
    }
    for (auto &kp: kp2_pyr)
        kp2.push_back(kp);//存输出kp2
}
