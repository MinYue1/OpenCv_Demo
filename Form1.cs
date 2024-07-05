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
using System.Windows;
using System.Windows.Forms;
using System.Windows.Media;
using Point = OpenCvSharp.Point;
using Window = OpenCvSharp.Window;

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

            Scalar color = new Scalar(0, 255, 0); // 轮廓颜色为绿色
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
            //MessageBox.Show("");

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
        /// 膨胀
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button5_Click(object sender, EventArgs e)
        {

            Mat img = Cv2.ImRead(this.curtimgFile);    // 读取图像
            //转为灰度图
            Mat grayImg = new Mat();
            Cv2.CvtColor(img, grayImg, ColorConversionCodes.BGR2GRAY);
            //高斯模糊
            Mat blurImg = new Mat();
            Cv2.GaussianBlur(grayImg, blurImg, new OpenCvSharp.Size(7, 7), 0);
            //Canny边缘检测
            Mat cannyImg = new Mat();
            Cv2.Canny(blurImg, cannyImg, 150, 200);
            //膨胀
            Mat dialationImg = new Mat();
            Mat kernel = new Mat(5, 5, MatType.CV_8UC1); //膨胀核为矩形
            Cv2.Dilate(blurImg, dialationImg, kernel);
            //腐蚀
            Mat erodeImg = new Mat();
            Cv2.Erode(dialationImg, erodeImg, kernel);


            Cv2.ImShow("Image", img);   // 显示原图像
            Cv2.ImShow("Gray Image", grayImg);  // 显示灰度图像
            Cv2.ImShow("Blur Image", blurImg);  // 显示高斯模糊图像
            Cv2.ImShow("Canny Image", cannyImg);    // 显示Canny边缘检测图像
            Cv2.ImShow("Dialation Image", dialationImg);    // 膨胀图
            Cv2.ImShow("Erode Image", erodeImg);    // 腐蚀图


            Cv2.WaitKey(0);

            Cv2.DestroyAllWindows();

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
        /// 模板匹配，参考：https://blog.csdn.net/qq_42807924/article/details/104025672
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button8_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(this.curtimgFile) || string.IsNullOrEmpty(this.targetImg)) return;

            //加载图片
            Mat temp  = new Mat(this.targetImg,ImreadModes.AnyColor); //模板图
            Mat main = new Mat(this.curtimgFile,ImreadModes.AnyColor); //被匹配的图
            Mat result = new Mat();//结果图

            //缩小图片，减少计算量
            //Cv2.Resize(main, main, new OpenCvSharp.Size(main.Width / 3, main.Height / 3), interpolation: InterpolationFlags.Area);
            //模板匹配
            Cv2.MatchTemplate(main, temp, result, TemplateMatchModes.CCoeffNormed);

            //double minVal, maxVal;
            //Point minLoc, maxLoc;
            //Cv2.MinMaxLoc(result, out minVal, out maxVal, out minLoc, out maxLoc);

            //数组位置下x,y
            Point minLoc = new Point(0, 0);
            Point maxLoc = new Point(0, 0);
            Point matchLoc = new Point(0, 0);
            Cv2.MinMaxLoc(result, out minLoc, out maxLoc);
            matchLoc = maxLoc;
            Mat mask = main.Clone();
            //画框显示
            Cv2.Rectangle(mask, matchLoc, new Point(matchLoc.X + temp.Cols, matchLoc.Y + temp.Rows), Scalar.Green, 2);
            this.pictureBox1.Image = BitmapConverter.ToBitmap(mask);
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

                Cv2.ImShow("目标图片", mat);
                //using(new OpenCvSharp.Window("目标图片", mat))
                //{
                //    Cv2.WaitKey(0);
                //}
            }
        }

        //旋转图像
        private Mat ImageRotate(Mat image, double angle)
        {
            Mat newImg = new Mat();
            Point2f center = new Point2f(image.Cols / 2f, image.Rows / 2f);//获取图像的中心点，图像宽度，高度 / 2f
            //使用了Cv2.GetRotationMatrix2D()函数构建旋转矩阵，然后使用Cv2.WarpAffine()函数进行仿射变换。
            Mat matrix = Cv2.GetRotationMatrix2D(center, angle, 0.6);//构建旋转矩阵，绕中心点顺时针90°，并缩小为原来的60
            Cv2.WarpAffine(image, newImg, matrix, image.Size());//执行旋转、平移、缩放等操作
            return newImg;
        }

        /// <summary>
        /// 形状检测: 先获取所有轮廓，判断形状
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button10_Click(object sender, EventArgs e)
        {
            //1. 图片预处理
            Mat m1 = new Mat(this.curtimgFile);

            //缩小
            Cv2.Resize(m1, m1, new OpenCvSharp.Size(m1.Width / 2, m1.Height / 2));

            //转成灰度图
            Mat gray = new Mat();
            Cv2.CvtColor(m1, gray, ColorConversionCodes.RGB2GRAY);

            Cv2.Blur(gray, gray, new OpenCvSharp.Size(3, 3), new Point(-1, -1), BorderTypes.Default);//模糊处理（降低噪点）

            //使用Canny边缘检测（高斯->灰度->梯度）
            Cv2.Canny(gray,gray,49,30); //高低阈值


            //2. 获取轮廓
            Point[][] contours;
            HierarchyIndex[] hierarchy;
            Cv2.FindContours(gray, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);


            //3. 判断形状
            foreach (Point[] contour in contours)
            {
                double area = Cv2.ContourArea(contour);
                if (area < 100) continue;

                Point[] approx = Cv2.ApproxPolyDP(contour, 0.04 * Cv2.ArcLength(contour, true), true);

                // 根据 approx.Length 判断形状
                int vertices = approx.Length;
                string shapeType = "";
                if (vertices == 3) shapeType = "Triangle";
                else if (vertices == 4)
                {
                    OpenCvSharp.Rect rect = Cv2.BoundingRect(approx);
                    double aspectRatio = (double)rect.Width / rect.Height;
                    shapeType = (aspectRatio >= 0.95 && aspectRatio <= 1.05) ? "Square" : "Rectangle";
                }
                else if (vertices == 5) shapeType = "Pentagon";
                else if (vertices > 10) shapeType = "Circle";

                // 在图像上绘制轮廓和形状名称
                Cv2.DrawContours(m1, new[] { contour }, 0, Scalar.Red, 2);
                Moments M = Cv2.Moments(contour);
                int cx = (int)(M.M10 / M.M00);
                int cy = (int)(M.M01 / M.M00);
                //Cv2.PutText(m1, shapeType, new Point(cx, cy), HersheyFonts.HersheySimplex, 0.5, Scalar.Red, 2);


            }

            Cv2.ImShow("结果", gray);
            Cv2.ImShow("原图", m1);
            Cv2.WaitKey();
            Cv2.DestroyAllWindows();

        }

        /// <summary>
        /// 匹配多个模板
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button11_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(this.curtimgFile) || string.IsNullOrEmpty(this.targetImg)) return;

            //加载图片
            Mat temp = new Mat(this.targetImg, ImreadModes.AnyColor); //模板图
            Mat main = new Mat(this.curtimgFile, ImreadModes.AnyColor); //被匹配的图
            Mat result = new Mat();//结果图

            //模板匹配
            Cv2.MatchTemplate(main, temp, result, TemplateMatchModes.CCoeffNormed);

            Double minVul;
            Double maxVul;
            Point minLoc = new Point(0, 0);
            Point maxLoc = new Point(0, 0);
            Point matchLoc = new Point(0, 0);
            Cv2.Normalize(result, result, 0, 1, NormTypes.MinMax, -1);//归一化
            Cv2.MinMaxLoc(result, out minVul, out maxVul, out minLoc, out maxLoc);//查找极值
            matchLoc = maxLoc;//最大值坐标
            //result.Set(matchLoc.Y, matchLoc.X, 0);//改变最大值为最小值  
            Mat mask = main.Clone();//复制整个矩阵
            //画框显示
            Cv2.Rectangle(mask, matchLoc, new Point(matchLoc.X + temp.Cols, matchLoc.Y + temp.Rows), Scalar.Green, 2);


            Console.WriteLine("最大值：{0}，X:{1}，Y:{2}", maxVul, matchLoc.Y, matchLoc.X);
            Console.WriteLine("At获取最大值(Y,X)：{0}", result.At<float>(matchLoc.Y, matchLoc.X));
            Console.WriteLine("result的类型：{0}", result.GetType());

            //循环查找画框显示
            Double threshold = 0.91;
            Mat maskMulti = main.Clone();//复制整个矩阵

            for (int i = 1; i < result.Rows - temp.Rows; i += temp.Rows)//行遍历
            {

                for (int j = 1; j < result.Cols - temp.Cols; j += temp.Cols)//列遍历
                {
                    OpenCvSharp.Rect roi = new OpenCvSharp.Rect(j, i, temp.Cols, temp.Rows);        //建立感兴趣
                    Mat RoiResult = new Mat(result, roi);
                    Cv2.MinMaxLoc(RoiResult, out minVul, out maxVul, out minLoc, out maxLoc);//查找极值
                    matchLoc = maxLoc;//最大值坐标
                    if (maxVul > threshold)
                    {

                        //画框显示
                        Cv2.Rectangle(maskMulti, new Point(j + maxLoc.X, i + maxLoc.Y), new Point(j + maxLoc.X + temp.Cols, i + maxLoc.Y + temp.Rows), Scalar.Green, 2);
                        string axis = '(' + Convert.ToString(i + maxLoc.Y) + ',' + Convert.ToString(j + maxLoc.X) + ')';
                        Cv2.PutText(maskMulti, axis, new Point(j + maxLoc.X, i + maxLoc.Y), HersheyFonts.HersheyPlain, 1, Scalar.Red, 1, LineTypes.Link4);

                    }

                }
            }


            Cv2.ImShow("结果", maskMulti);


            this.pictureBox1.Image = BitmapConverter.ToBitmap(maskMulti);

        }

        private void button12_Click(object sender, EventArgs e)
        {
            Mat m1 = new Mat(this.curtimgFile);

            //转换成灰度
            Mat gray = new Mat();
            Cv2.CvtColor(m1, gray, ColorConversionCodes.RGB2GRAY);

            //高斯模糊
            Mat gs = new Mat();
            Cv2.GaussianBlur(gray, gs, new OpenCvSharp.Size(7, 7), 0);

            //边缘检测
            Mat by = new Mat();
            Cv2.Canny(gs, by, 100, 200);

            //膨胀
            Mat pz = new Mat(); 
            Mat kn = new Mat(5,5,MatType.CV_8UC1);
            Cv2.Dilate(by, pz, kn);//原图，结果图，矩形核

            //腐蚀
            Mat fs = new Mat();
            Cv2.Erode(by, fs, kn);

            //
            using(new Window("边缘检测",by))
            using(new Window("灰度图",gray))
            using (new Window("高斯模糊",gs))
            using(new Window("膨胀",pz))
            using(new Window("腐蚀", fs))
            {
                Cv2.WaitKey(0);
            }


        }

        /// <summary>
        /// matchShapes 形状匹配
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void button13_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(this.curtimgFile) || string.IsNullOrEmpty(this.targetImg)) return;

            // loading 原始图和模板图
            Mat all = new Mat(this.curtimgFile, ImreadModes.AnyColor);
            Mat model_1 = new Mat(this.targetImg, ImreadModes.AnyColor);

            //预处理 
            Mat main = all.Clone();
            Mat main1 = all.Clone();
            Mat switch1 = model_1.Clone();
            Mat switch2 = model_1.Clone();
            Cv2.CvtColor(main, main, ColorConversionCodes.RGB2GRAY);
            //Cv2.CvtColor(main, main, ColorConversionCodes.RGB2GRAY);
            Cv2.CvtColor(model_1,model_1, ColorConversionCodes.RGB2GRAY);

            Cv2.Threshold(main, main, 0, 255, ThresholdTypes.Otsu);
            Cv2.Threshold(model_1, model_1, 0, 255, ThresholdTypes.Otsu);

            Cv2.ImShow("x1", main);
            Cv2.ImShow("x2", model_1);

            //=>边缘检测
            //高斯模糊
            Cv2.GaussianBlur(main, main, new OpenCvSharp.Size(7, 7), 2, 2);
            Cv2.GaussianBlur(model_1, model_1, new OpenCvSharp.Size(7, 7), 2, 2);
            //边缘检测
            Cv2.Canny(main, main, 100, 200);
            Cv2.Canny(model_1, model_1, 100, 200);


            //查找轮廓，绘制
            Point[][] contours_main;
            HierarchyIndex[] hierarchy_main;

            Cv2.FindContours(main, out contours_main, out hierarchy_main, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            Scalar color = new Scalar(0, 255, 0); // 轮廓颜色为绿色
            int thickness = 2; // 轮廓线粗细为2
            for (int i = 0; i < contours_main.Length; i++)
            {
                Cv2.DrawContours(all, contours_main, i, color, thickness); // 绘制轮廓
            }

            Cv2.ImShow("原图", all);
            

            Point[][] contours_model1; //轮廓点集
            HierarchyIndex[] hierarchy_model1;// 轮廓层次结构
            Cv2.FindContours(model_1, out contours_model1, out hierarchy_model1, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            //输出轮廓点集,手动绘制轮廓
            //for (int i =0;i < contours_model1.Length; i++)
            //{
            //    Console.WriteLine(contours_model1[i].Length);

            //    for(int j = 0;j < contours_model1[i].Length; j++)
            //    {
            //        Console.Write($"[{contours_model1[i][j].X},{contours_model1[i][j].Y}]");


            //        switch1.Set <Vec3b> (contours_model1[i][j].Y, contours_model1[i][j].X, new Vec3b(0,255,0));
            //    }
            //    Console.WriteLine();
            //}

            //Cv2.Resize(switch1, switch1,new OpenCvSharp.Size(switch1.Width * 3f, switch1.Height * 3f), (double)InterpolationFlags.Cubic);

            //Cv2.ImShow("test", switch1);

            for (int i = 0; i < contours_model1.Length; i++)
            {
                Cv2.DrawContours(switch2, contours_model1, i, color, thickness); // 绘制轮廓
            }
            //模型图
            Cv2.ImShow("模型图", switch2);
            Cv2.WaitKey(0);

            //形状匹配
            double min = 0;
            int minIndex = 0;
            for(int i =  0; i < contours_main.Length; i++)
            {
                double temp = Cv2.MatchShapes(contours_main[i], contours_model1[0], ShapeMatchModes.I1);
                if (temp < min)
                {
                    min = temp;
                    minIndex = i;
                }

                Console.WriteLine($"i : {i} 比较模板：{temp}");
            }


            for(int i = 0; i < contours_model1.Length;i++) Console.WriteLine($"i : {i} 模板值：{contours_model1[i]}");
            //绘制最匹配的轮廓
            //Scalar color = new Scalar(0, 255, 0); // 轮廓颜色为绿色
            //int thickness = 2; // 轮廓线粗细为2
            Cv2.DrawContours(main1, contours_main, minIndex, color, thickness);
            
            //Cv2.MatchShapes(contours_main, contours_model1, ShapeMatchModes.I1);

            Cv2.ImShow("结果", main1);
            Cv2.WaitKey(0);

            Cv2.DestroyAllWindows();

        }
    }
}
