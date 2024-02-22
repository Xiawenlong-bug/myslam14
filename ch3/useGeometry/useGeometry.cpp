#include<iostream>
#include<cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

int main(int argc,char** argv)
{
    Matrix3d rotation_matrix=Matrix3d::Identity();
    AngleAxisd rotation_vector(M_PI/4,Vector3d(0,0,1));//构造函数，参数列表为转角和转轴
    cout.precision(3);//cout.precision(val)：在输出的时候，设定输出值以新的浮点数精度值显示，即小数点后保留3位。
    cout << "rotation matrix =\n" << rotation_vector.matrix() << endl;   //用matrix()转换成矩阵  
    
    // 也可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();
    // 用 AngleAxis 可以进行坐标变换
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;    

    return 0;
}