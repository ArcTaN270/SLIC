#include<iostream>  
#include<math.h>
#include<string.h>
#include<stdlib.h>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define SPN 64 // Superpixels for each edge
#define IMG 512 // The edge length of the image

int label[IMG][IMG]; // Sign which superpixel belong to
int dis[IMG][IMG]; //Save the distance (in [L,A,B,X,Y])

struct cluster {
	int row;
	int col;
	int l;
	int a;
	int b;
};
cluster clusters[SPN * SPN];

int GetDistance(const Mat LAB, int clusters_index, int i, int j, int m)
{
	int Dl = clusters[clusters_index].l - LAB.at<Vec3b>(i, j)[0]; //Calculation of distance in L
	int Da = clusters[clusters_index].a - LAB.at<Vec3b>(i, j)[1]; //Calculation of distance in A
	int Db = clusters[clusters_index].b - LAB.at<Vec3b>(i, j)[2]; //Calculation of distance in B
	int Dx = clusters[clusters_index].row - i; //Calculation of distance in X
	int Dy = clusters[clusters_index].col - j; //Calculation of distance in Y

	int Dc = sqrt(Dl * Dl + Da * Da + Db * Db);
	int Ds = sqrt(Dx * Dx + Dy * Dy);

	return Dc + Ds * m;
}

void UpdatePixel(const Mat LAB, int s,int m)
{
	for (int i = 0; i < SPN * SPN; i++)
	{
		int clusters_x = clusters[i].row;
		int clusters_y = clusters[i].col;
		for (int x = -s; x <= s; x++)
		{
			for (int y = -s; y <= s; y++)
			{
				int now_x = clusters_x + x;
				int now_y = clusters_y + y;
				if (now_x < 0 || now_x >= IMG || now_y < 0 || now_y >= IMG)
				{
					continue;
				}
				//Calculate the distance in[L, A, B, X, Y] SPACE
				int new_dis = GetDistance(LAB, i, now_x, now_y, m);

				if (dis[now_x][now_y] > new_dis || dis[now_x][now_y] == -1)
				{
					dis[now_x][now_y] = new_dis;
					label[now_x][now_y] = i;
				}
			}
		}
	}
}

void CenterClusters(const Mat lab)
{
	// After adjusting the clustering, the superpixel of each clustering region is taken as the center
	int* sum_count = new int[SPN * SPN]();
	int* sum_l = new int[SPN * SPN]();
	int* sum_a = new int[SPN * SPN]();
	int* sum_b = new int[SPN * SPN]();
	int* sum_i = new int[SPN * SPN]();
	int* sum_j = new int[SPN * SPN]();
	for (int i = 0; i < IMG; i++)
	{
		for (int j = 0; j < IMG; j++)
		{
			sum_count[label[i][j]]++;
			sum_i[label[i][j]] += i;
			sum_j[label[i][j]] += j;
			sum_l[label[i][j]] += lab.at<Vec3b>(i, j)[0];
			sum_a[label[i][j]] += lab.at<Vec3b>(i, j)[1];
			sum_b[label[i][j]] += lab.at<Vec3b>(i, j)[2];
		}
	}
	// Update the cluster center
	for (int i = 0; i < SPN * SPN; i++)
	{
		if (sum_count[i] == 0)
		{
			sum_count[i]=1;
		}
		clusters[i].l = round(sum_l[i] / sum_count[i]);
		clusters[i].a = round(sum_a[i] / sum_count[i]);
		clusters[i].b = round(sum_b[i] / sum_count[i]);
		clusters[i].row = round(sum_i[i] / sum_count[i]);
		clusters[i].col = round(sum_j[i] / sum_count[i]);
	}

	delete[] sum_count;
	delete[] sum_i;
	delete[] sum_j;
	delete[] sum_l;
	delete[] sum_a;
	delete[] sum_b;
}

void DrawSuperpixels(const Mat src)
{
	Mat img = src.clone();
	// Draw super pixels
	for (int pixel = 0; pixel < SPN * SPN; pixel++)
	{
		Point p(clusters[pixel].row, clusters[pixel].col);
		circle(img, p, 0.5, Scalar(0, 0, 255), 0.5); 
	}
	imshow("Draw the pixel for cluster", img);
}

void DrawFinalimg(const Mat lab)
{
	Mat img = lab.clone();
	// Each pixel is recoloured by the label to display the clustered image
	for (int i = 0; i < IMG; i++)
	{
		for (int j = 0; j < IMG; j++)
		{
			int pixel = label[i][j];
			img.at<Vec3b>(i, j)[0] = lab.at<Vec3b>(clusters[pixel].row, clusters[pixel].col)[0];
			img.at<Vec3b>(i, j)[1] = lab.at<Vec3b>(clusters[pixel].row, clusters[pixel].col)[1];
			img.at<Vec3b>(i, j)[2] = lab.at<Vec3b>(clusters[pixel].row, clusters[pixel].col)[2];
		}
	}
	cvtColor(img, img, CV_Lab2BGR);

	imshow("The segmentation of the image", img);
}

void DrawEdge(const Mat LAB) 
{
	Mat img = LAB.clone();
	// Each pixel is recoloured by the label to display the clustered image
	for (int i = 0; i < IMG; i++) 
	{
		for (int j = 0; j < IMG; j++) 
		{
			int index = label[i][j];
			img.at<Vec3b>(i, j)[0] = LAB.at<Vec3b>(clusters[index].row, clusters[index].col)[0];
			img.at<Vec3b>(i, j)[1] = LAB.at<Vec3b>(clusters[index].row, clusters[index].col)[1];
			img.at<Vec3b>(i, j)[2] = LAB.at<Vec3b>(clusters[index].row, clusters[index].col)[2];
		}
	}

	static int X[4] = { 0,0,-1,1 };
	static int Y[4] = { 1,-1,0,0 };
	// Itraverse the pixels in the four directions, drawing a black boundary if the markings are different
	cvtColor(img, img, CV_Lab2BGR);
	for (int i = 0; i < IMG; i++) 
	{
		for (int j = 0; j < IMG; j++) 
		{
			for (int k = 0; k < 4; k++) 
			{
				if (label[i][j] != label[i + X[k]][j + X[k]]) // Draw the black edge
				{
					img.at<Vec3b>(i, j)[0] = img.at<Vec3b>(i, j)[1] = img.at<Vec3b>(i, j)[2] = 0;
				}
			}
		}
	}

	imshow("Draw the edge of the FinalImage", img);
}

int main()
{
	Mat src = imread(".\\feixieer.jpg");
	Mat lab;

	resize(src, src, Size(IMG, IMG));
	imshow("Original Image", src);

	cvtColor(src, lab, CV_BGR2Lab);
	//imshow("LabImage", lab);

	int N = IMG * IMG;// Total number of image pixels
	int K = SPN * SPN;// Total number of superpixels
	int S = sqrt(N / K); //Distance between per superpixel
	int M = 10; // Weighing color similarity against spatial similarity
	int cluster_time = 10; // The number of iterations

	// Initialize each superpixel
	int row;
	for (int i = 0; i < SPN; i++)
	{
		for (int j = 0; j < SPN; j++) // Record the coordinates of each superpixel
		{
			clusters[i * SPN + j].row = S / 2 + i * S;
			clusters[i * SPN + j].col = S / 2 + j * S;
		}
	}

	for (int i = 0; i < IMG; i++) // Initializes which superpixel each pixel belongs to
	{
		row = i / S;
		for (int j = 0; j < IMG; j++)
		{
			label[i][j] = row * SPN + j / S;
		}
	}

	memset(dis, -1, sizeof(dis)); //Initialize each distance to superpixel

	// Start the iteration
	for (int i = 0; i < cluster_time; i++)
	{

		UpdatePixel(lab, 2 * S, M);

		CenterClusters(lab);

		DrawSuperpixels(src);

		DrawEdge(lab);

		DrawFinalimg(lab);

		waitKey(10);
	}

	waitKey(0);
}