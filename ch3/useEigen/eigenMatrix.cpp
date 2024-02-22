#include<iostream>
#include<ctime>

#include<Eigen/Core>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int argc,char** argv)
{
    Matrix<float,2,3>matrix_23;
    Vector3d v_3d;
    Matrix3d matrix_33=Matrix3d::Zero();
    matrix_23 << 1, 2, 3, 4, 5, 6;
    // 输出
    cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;
  // 用()访问矩阵中的元素
    cout << "print matrix 2x3: " << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
        cout << endl;
    }    
    matrix_33 = Matrix3d::Random();      // 随机数矩阵
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;      // 转置
    cout << "sum: " << matrix_33.sum() << endl;            // 各元素和
    cout << "trace: " << matrix_33.trace() << endl;          // 迹
    cout << "times 10: \n" << 10 * matrix_33 << endl;               // 数乘
    cout << "inverse: \n" << matrix_33.inverse() << endl;        // 逆
    cout << "det: " << matrix_33.determinant() << endl;    // 行列式

    //这里特征值为什么要这样算
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(matrix_33.transpose()*matrix_33);//MatrixXd自动计算维数
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;
    // SelfAdjointEigenSolver<MatrixXd> eigen_solver(matrix_33);//MatrixXd自动计算维数
    // cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    // cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

    
    // 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程
    // N的大小在前边的宏里定义，它由随机数生成
    // 直接求逆自然是最直接的，但是求逆运算量大
    Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
        = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose();  // 保证半正定
    Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock(); // 计时
    // 直接求逆
    Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    cout << "time of normal inverse is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // 通常用矩阵分解来求，例如QR分解，速度会快很多
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "time of Qr decomposition is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    // 对于正定矩阵，还可以用cholesky分解来解方程
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "time of ldlt decomposition is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x = " << x.transpose() << endl;

    return 0;
}