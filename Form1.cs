using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OpenCv_Demo
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        private string curtimgFile;


        private void button1_Click(object sender, EventArgs e)
        {
            //bitmap和mat互相转换
            //BitmapConverter.ToBitmap();

            Mat mat1 = new Mat(this.curtimgFile);

            Mat mat2 = Cv2.ImRead(this.curtimgFile);


            //Cv2.ImShow("1",mat1);
            //Cv2.ImShow("2", mat2);

            //图片尺寸缩放
            Mat dstImage1 = new Mat();
            Cv2.Resize(mat1, dstImage1, new OpenCvSharp.Size(mat1.Cols / 19, mat1.Rows / 19), (double)InterpolationFlags.Linear);

            Cv2.ImShow("1", dstImage1);



            //图像透视




        }

        private void button2_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            //dialog.Multiselect = true;//该值确定是否可以选择多个文件
            dialog.Title = "请选择文件";
            dialog.Filter = "图像文件(*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff)|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff";
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                this.curtimgFile = dialog.FileName;
                pictureBox1.Image = new Bitmap(this.curtimgFile);
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {

            Mat img = new Mat(this.curtimgFile, ImreadModes.Grayscale);
            Point2f center = new Point2f(img.Cols / 2f, img.Rows / 2f);//获取图像的中心点，图像宽度，高度 / 2f
            //使用了Cv2.GetRotationMatrix2D()函数构建旋转矩阵，然后使用Cv2.WarpAffine()函数进行仿射变换。
            Mat matrix = Cv2.GetRotationMatrix2D(center, 90, 0.6);//旋转矩阵，绕中心点顺时针90°，并缩小为原来的60
            

            Mat xi = new Mat();
            Cv2.Rotate(img, xi, RotateFlags.Rotate90CounterClockwise);
            Cv2.WarpAffine(img, img, matrix, img.Size());//
            Cv2.ImShow("img", img);
            Cv2.WaitKey(0);//等待用户按键
            Cv2.DestroyAllWindows();//关闭所有图片窗口

        }
    }
}
