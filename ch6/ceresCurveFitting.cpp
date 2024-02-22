#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

struct CostFunctor
{
    CostFunctor(double x,double y):_x(x),_y(y){}
    /* data */
    template <typename T>
    bool operator()(const T* const abc,T* residual) const
    {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const double _x,_y;
};



int main(int argc,char**argv)
{
    /*生成数据点*/
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;      // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    double abc[3] = {ae, be, ce};

    //问题构建
    ceres::Problem problem;
    for(int i=0;i<N;i++)
    {
        ceres::CostFunction* costfunction= new ceres::AutoDiffCostFunction<CostFunctor,1,3>(
            new CostFunctor(x_data[i],y_data[i]));
        
        problem.AddResidualBlock(costfunction,nullptr,abc);
        //两种写法都一样
        // problem.AddResidualBlock(
        //     // 使用自动求导，模板参数：误差类型，输出维度residual，输入维度adc,...，维数要与前面struct中一致
        //     new ceres::AutoDiffCostFunction<CostFunctor,1,3>(
        //         new CostFunctor(x_data[i],y_data[i])
        //     ),
        //     nullptr,// 核函数，这里不使用，为空
        //     abc//待估计参数
        // );
    }

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a:abc) cout << a << " ";
    cout << endl;

    return 0;

}