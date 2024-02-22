#include <pangolin/pangolin.h>//pangolin是一个基于OPENGL的绘图库，在OPENGL的绘图操作基础上提供了一些GUI的功能
#include <Eigen/Core>
#include <unistd.h>

// 本例演示了如何画出一个预先存储的轨迹

using namespace std;
using namespace Eigen;

// path to trajectory file

string trajectory_file = "../../plotTrack/trajectory.txt";
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) 
{
    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
    ifstream fin(trajectory_file);
    if (!fin) {
        cout << "cannot find trajectory file at " << trajectory_file << endl;
        return 1;
    }
/*
    使用ifstream流来读取文件
    说明：
    1.ifstream类的对象创建成功的时候会返回非空值，借此判断是否创建文件对象成功
    2.ifstream有个函数eof()用来判断文件是否读到尾部,没读到尾部返回false，否则返回true。
    若尾部有回车，那么最后一条记录会读取两次。
    若尾部没有回车，那么最后一条记录只会读取一次
    3.iftream的对象假设为fin，fin在读取数据的时候会根据你的输出对象来选择输出的方式。
*/
    while (!fin.eof()) {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));//欧式变换矩阵Isometry（虽然称为3d，实质上是4＊4的矩阵）
        Twr.pretranslate(Vector3d(tx, ty, tz));///设置平移向量，理解是加入这个平移向量
        poses.push_back(Twr);//函数将一个新的元素加到vector的最后面，位置为当前最后一个元素的下一个元素 push_back() 在Vector最后添加一个元素
    }
    cout << "read total " << poses.size() << " pose entries" << endl;

    // draw trajectory in pangolin
    DrawTrajectory(poses);
    return 0;

}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses)
{
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);//创建一个名叫"Trajectory Viewer"的GUI窗口用于显示，窗口的大小是640x480像素。
    glEnable(GL_DEPTH_TEST);//启动深度测试，开启这个功能之后，窗口中只会绘制面朝相机的那一面像素。一般如果你使用的3D可视化，就要打开这个功能。
    glEnable(GL_BLEND);//打开颜色混合，把某一像素位置原来的颜色和将要画上去的颜色，通过某种方式混在一起，从而实现特殊的效果。这个有点儿类似你透过红色玻璃看绿色物体的效果。
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);//使用glEnable(GL_BLEND);之后，后面紧跟着这行代码，表示两种颜色以怎么样的方式进行混合。

    //创建一个相机的观察视图，相当于是模拟一个真实的相机去观测虚拟的三维世界,最终在GUI中呈现的图像就是通过这个设置的相机内外参得到的
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),//相机内参:参数依次为相机的图像高度、宽度、4个内参以及最近和最远视距
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)//前三个参数依次为相机所在的位置，第四到第六个参数相机所看的视点位置(一般会设置在原点)，最后是相机轴的方向
    //可以用自己的脑袋当做例子，前三个参数告诉你脑袋在哪里，然后再告诉你看的东西在哪里，最后告诉你的头顶朝着哪里
    );
    /*进行显示设置。SetBounds函数前四个参数依次表示视图在视窗中的范围（下、上、左、右），最后一个参数是显示的长宽比。（0.0, 1.0, 0.0, 1.0）
    第一个参数0.0表示显示的拍摄窗口的下边在整个GUI中最下面，第二个参数1.0表示上边在GUI的最上面，以此类推。如果在中间就用0.5表示
    */
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    //检测是否关闭OpenGL窗口
    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//清空颜色和深度缓存,每次都会刷新显示，不至于前后帧的颜信息相互干扰。
        d_cam.Activate(s_cam);//激活显示并设置状态矩阵,以下代码到Finish是显示内容
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        for (size_t i = 0; i < poses.size(); i++) {
        // 画每个位姿的三个坐标轴
        Vector3d Ow = poses[i].translation();
        Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
        Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
        Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Xw[0], Xw[1], Xw[2]);
        glColor3f(0.0, 1.0, 0.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Yw[0], Yw[1], Yw[2]);
        glColor3f(0.0, 0.0, 1.0);
        glVertex3d(Ow[0], Ow[1], Ow[2]);
        glVertex3d(Zw[0], Zw[1], Zw[2]);
        glEnd();//结束
        }
        // 画出连线
        for (size_t i = 0; i < poses.size(); i++) {
        glColor3f(0.0, 0.0, 0.0);
        glBegin(GL_LINES);//绘制直线
        auto p1 = poses[i], p2 = poses[i + 1];
        glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
        glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
        glEnd();
        }
        pangolin::FinishFrame();//执行后期渲染，事件处理和帧交换，进行最终的显示
        usleep(5000);   // sleep 5 ms
    }
}