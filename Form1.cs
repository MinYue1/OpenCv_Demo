using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
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
        private string targetImg;


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
        /// <summary>
        /// 旋转
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button3_Click(object sender, EventArgs e)
        {

            Mat img = new Mat(this.curtimgFile, ImreadModes.Grayscale);
            Point2f center = new Point2f(img.Cols / 2f, img.Rows / 2f);//获取图像的中心点，图像宽度，高度 / 2f
            //使用了Cv2.GetRotationMatrix2D()函数构建旋转矩阵，然后使用Cv2.WarpAffine()函数进行仿射变换。
            Mat matrix = Cv2.GetRotationMatrix2D(center, 60, 0.6);//构建旋转矩阵，绕中心点顺时针90°，并缩小为原来的60
            
            Cv2.WarpAffine(img, img, matrix, img.Size());//执行旋转、平移、缩放等操作
            Cv2.ImShow("img", img);
            Cv2.WaitKey(0);//等待用户按键
            Cv2.DestroyAllWindows();//关闭所有图片窗口

        }
        /// <summary>
        /// 轮廓检测
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button4_Click(object sender, EventArgs e)
        {
            Mat m1 = Cv2.ImRead(this.curtimgFile);//尝试读取为灰度图
            Mat image = Cv2.ImRead(this.curtimgFile, ImreadModes.Color);

            //转换为灰度图
            Mat gray = new Mat();
            Cv2.CvtColor(m1, gray, ColorConversionCodes.BGR2GRAY);

            //二值化
            Mat m2 = new Mat();
            Cv2.Threshold(gray, m2, 150, 255, ThresholdTypes.Binary); 

            this.pictureBox1.Image = BitmapConverter.ToBitmap(m2);

            //
            OpenCvSharp.Point[][] contours;
            HierarchyIndex[] hierarchy;

            Cv2.FindContours(m2, out contours, out hierarchy, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            Scalar color = new Scalar(0, 0, 255); // 轮廓颜色为绿色
            int thickness = 2; // 轮廓线粗细为2

            for (int i = 0; i < contours.Length; i++)
            {
                Cv2.DrawContours(image, contours, i, color, thickness); // 绘制轮廓
            }


            this.pictureBox2.Image = BitmapConverter.ToBitmap(image);

            Cv2.Resize(image,image,new OpenCvSharp.Size(image.Width / 3, image.Height / 3));

            Cv2.ImShow("image", image);
            Cv2.WaitKey(0);

            
        }
        /// <summary>
        /// 边缘检测
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button6_Click(object sender, EventArgs e)
        {
            Mat m1 = Cv2.ImRead(this.curtimgFile, ImreadModes.Grayscale);//尝试加载为灰度图


            //高斯模糊
            Mat dst = new Mat();
            Cv2.GaussianBlur(m1, dst, new OpenCvSharp.Size(7, 7), 2, 2);

            this.pictureBox2.Image = BitmapConverter.ToBitmap(dst);
            MessageBox.Show("");

            //边缘检测
            Mat edges = new Mat();
            Cv2.Canny(m1, edges, 100, 200);



            //显示原有图像和边缘图像
            this.pictureBox2.Image = BitmapConverter.ToBitmap(edges);
            //Cv2.ImShow("原图", m1);
            //Cv2.ImShow("结果", edges);
            //Cv2.WaitKey(0);

            //释放内存
            Cv2.DestroyAllWindows();
            m1.Dispose();
            edges.Dispose();
        }

        /// <summary>
        /// 斑点检测
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button5_Click(object sender, EventArgs e)
        {
            
        }

        /// <summary>
        /// 缩放
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button7_Click(object sender, EventArgs e)
        {

            Mat m1 = new Mat(this.curtimgFile);

            Mat dst1 = new Mat();
            Cv2.Resize(m1, dst1, new OpenCvSharp.Size(m1.Width / 3, m1.Height / 3), interpolation: InterpolationFlags.Area);


            this.pictureBox1.Image = BitmapConverter.ToBitmap(dst1);


        }

        /// <summary>
        /// 模板匹配
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button8_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(this.curtimgFile) || string.IsNullOrEmpty(this.targetImg)) return;

            //加载图片
            Mat temp  = new Mat(this.targetImg,ImreadModes.AnyColor); //模板图
            Mat main = new Mat(this.curtimgFile,ImreadModes.AnyColor); //被匹配的图
            Mat result = new Mat(this.curtimgFile,ImreadModes.AnyColor);//结果图


            //缩小图片，减少计算量

            //模板匹配
            Cv2.MatchTemplate(main, temp, result, TemplateMatchModes.CCoeffNormed);


        }

        /// <summary>
        /// 添加目标图片
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button9_Click(object sender, EventArgs e)
        {
            OpenFileDialog dialog = new OpenFileDialog();
            //dialog.Multiselect = true;//该值确定是否可以选择多个文件
            dialog.Title = "请选择文件";
            dialog.Filter = "图像文件(*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff)|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.tiff";
            if (dialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                this.targetImg = dialog.FileName;
                Mat mat = new Mat(this.targetImg);
                using(new Window("目标图片", mat))
                {
                    Cv2.WaitKey();
                }
            }
        }
    }
}
