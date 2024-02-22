#include <iostream>
#include <opencv2/core/core.hpp>             //opencv核心模块
#include <opencv2/features2d/features2d.hpp> //opencv特征点
#include <opencv2/highgui/highgui.hpp>       //opencv gui
#include <opencv2/calib3d/calib3d.hpp>       //求解器头文件
#include <Eigen/Core>                        //eigen核心模块
#include <g2o/core/base_vertex.h>            //g2o顶点（Vertex）头文件 视觉slam十四讲p141用顶点表示优化变量，用边表示误差项
#include <g2o/core/base_unary_edge.h>        //g2o边（edge）头文件
#include <g2o/core/sparse_optimizer.h>       //稠密矩阵求解
#include <g2o/core/block_solver.h>           //求解器头文件
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h> //高斯牛顿算法头文件
#include <g2o/solvers/dense/linear_solver_dense.h>        //线性求解
#include <sophus/se3.hpp>                                 //李群李代数se3
#include <chrono>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;
using namespace cv;
double fx = 520.9;
double fy = 521.0;
double cx = 325.1;
double cy = 249.7;
double camera[6] = {1,1,1,1,1,1};  //初始值
// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;


struct SnavelyReprojectionError{
    SnavelyReprojectionError(double observed_x, double observed_y, Point3f point)
        : observed_x(observed_x), observed_y(observed_y), point(point){}

    template<typename T>
    bool operator()(const T* const camera,//camera:待优化变量
                    T* residuals)const{
        //计算投影点
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        T point_w[3] = {T(point.x), T(point.y), T(point.z)};
        
        ceres::AngleAxisRotatePoint(camera, point_w, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        /*
        u_x=f_x*X/Z+c_x
        u_y=f_y*Y/Z+c_y
        相机投影模型
        */
        T predicted_x = fx * xp + cx;
        T predicted_y = fy * yp + cy;

        residuals[0] = observed_x - predicted_x;
        residuals[1] = observed_y - predicted_y;

        return true;
    }

    double observed_x;
    double observed_y;
    Point3f point;
};

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches);

Point2d pixel2cam(const Point2d &p, const Mat &K);

void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);


void GN_ceres(vector<Point2f> pts_2d, vector<Point3f> pts_3d, double camera[6]);

int main(int argc, char **argv)
{
    Mat img_1 = imread("../1.png", cv::IMREAD_COLOR);
    Mat img_2 = imread("../2.png", cv::IMREAD_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches); // 以后这个函数直接用
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat d1 = imread("../1_depth.png", cv::IMREAD_UNCHANGED);
    // Mat d2 = imread("../2_depth.png", cv::IMREAD_UNCHANGED);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;

    for (DMatch m : matches) // 用于遍历matches中每一个DMATCH对象
    {
        // image.ptr<uchar>(row)[col]访问(row,col)处的像素
        // ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];

        // 能否改写--可以，但耗时是第一种方法的两倍左右
        int col = keypoints_1[m.queryIdx].pt.x;
        int row = keypoints_1[m.queryIdx].pt.y;

        ushort d = d1.at<unsigned short>(row, col);

        if (d == 0) // bad depth
            continue;
        float dd = d / 5000.0; // 这个数有什么根据么？
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints_2[m.trainIdx].pt);
    }

    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

    cout << "R=" << endl
         << R << endl;
    cout << "t=" << endl
         << t << endl;

    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i)
    {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "calling bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

    cout << "calling BA by g2o" << endl;
    Sophus::SE3d pose_g2o;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by g2o cost time: " << time_used.count() << " seconds." << endl;

    cout << "calling BA by ceres" << endl;
    Sophus::SE3d pose_ceres;
    t1 = chrono::steady_clock::now();
    GN_ceres(pts_2d, pts_3d, camera);
    t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve pnp by ceres cost time: " << time_used.count() << " seconds." << endl;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          vector<KeyPoint> &keypoints_1, vector<KeyPoint> &keypoints_2,
                          vector<DMatch> &matches)
{
    // 描述子作形参，因为它在主函数后面用不到
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K)
{
    return Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}


void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose)
    
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations;iter++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();
        cost = 0;
        for (int i = 0; i < points_3d.size(); i++)//计算H,b
        {
            Eigen::Vector3d pc = pose * points_3d[i];
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }
        
        Vector6d dx;
        dx = H.ldlt().solve(b);
        
        if (isnan(dx[0]))
        {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost)
        {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << std::setprecision(12) << cost << endl;
        if (dx.norm() < 1e-6)
        {
            // converge
            break;
        }

    }
  cout << "pose by g-n: \n"
       << pose.matrix() << endl;

}


/*------------g2o-------------*/

// 曲线模型的顶点，模板参数：优化变量维度和数据类型
// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d>//:表示继承，public表示公有继承；CurveFittingVertex是派生类，:BaseVertex<6, Sophus::SE3d>是基类
 {
public://以下定义的成员变量和成员函数都是公有的
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题
  // 重置
  virtual void setToOriginImpl() override //virtual表示该函数为虚函数，override保留字表示当前函数重写了基类的虚函数
  {
    _estimate = Sophus::SE3d();
  }
  // left multiplication on SE3
  // 更新
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }
  // 存盘和读盘：留空
  virtual bool read(istream &in) override {}//istream类是c++标准输入流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
  virtual bool write(ostream &out) const override {}//ostream类是c++标准输出流的一个基类
  //可参照C++ Primer Plus第六版的6.8节
};
// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;//解决Eigen库数据结构内存对齐问题
  EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}//使用列表赋初值
  virtual void computeError() override//virtual表示虚函数，保留字override表示当前函数重写了基类的虚函数
   {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);//创建指针v
    Sophus::SE3d T = v->estimate();//将estimate()值赋给T
    Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
    pos_pixel /= pos_pixel[2];
    _error = _measurement - pos_pixel.head<2>();
  }
  virtual void linearizeOplus() override {
    const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
    Sophus::SE3d T = v->estimate();
    Eigen::Vector3d pos_cam = T * _pos3d;
    double fx = _K(0, 0);
    double fy = _K(1, 1);
    double cx = _K(0, 2);
    double cy = _K(1, 2);
    double X = pos_cam[0];
    double Y = pos_cam[1];
    double Z = pos_cam[2];
    double Z2 = Z * Z;
    _jacobianOplusXi
      << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
      0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
  } //雅克比矩阵表达式见 视觉slam十四讲p186式7.46
  virtual bool read(istream &in) override {}
  virtual bool write(ostream &out) const override {}
private:
  Eigen::Vector3d _pos3d;
  Eigen::Matrix3d _K;
};


void bundleAdjustmentG2O(
  const VecVector3d &points_3d,
  const VecVector2d &points_2d,
  const Mat &K,
  Sophus::SE3d &pose) 
{
    // 构建图优化，先设定g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));//c++中的make_unique表示智能指针类型
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出
    // vertex
    // 往图中增加顶点
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);//对顶点进行编号，里面的0你可以写成任意的正整数，但是后面设置edge连接顶点时，必须要和这个一致
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);//添加顶点
    // K 相机内参矩阵
    Eigen::Matrix3d K_eigen;
    K_eigen <<
            K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);
    // edges
    // 往图中增加边
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i)//遍历2d点
    {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);//对顶点进行编号，里面的0你可以写成任意的正整数，但是后面设置edge连接顶点时，必须要和这个一致
        edge->setVertex(0, vertex_pose);  // 设置连接的顶点
        edge->setMeasurement(p2d);// 观测数值
        edge->setInformation(Eigen::Matrix2d::Identity());// 信息矩阵：协方差矩阵之逆
        optimizer.addEdge(edge);//添加边
        index++;
    }
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//开始计时
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);//迭代次数10
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时结束
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);//计算耗时
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}





void GN_ceres(vector<Point2f> pts_2d, vector<Point3f> pts_3d, double camera[6]){
    // ceres优化
    ceres::Problem problem;
    for (int i = 0; i < pts_2d.size(); i++){
        double observed_x = pts_2d[i].x;
        double observed_y = pts_2d[i].y;

        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 6>(
                new SnavelyReprojectionError(observed_x, observed_y, pts_3d[i])),
            NULL,
            camera
        );
    }

    //配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.FullReport() << endl;

    Eigen::Vector3d rotation_vector(camera[0],camera[1],camera[2]);
    Eigen::Vector3d translation(camera[3],camera[4],camera[5]);
    // 将轴-角旋转向量转换为旋转矩阵
    Eigen::Matrix3d R = Sophus::SO3d::exp(rotation_vector).matrix();
    // 使用旋转矩阵和平移向量构建SE3变换矩阵
    Sophus::SE3d SE3_transform(R, translation);
    // 输出变换矩阵
    std::cout << "SE3 transform matrix: \n" << SE3_transform.matrix() << std::endl;
}