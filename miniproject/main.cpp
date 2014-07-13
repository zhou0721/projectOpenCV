#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string>

using namespace std;
using namespace cv;

/*!
    \brief Calcule les histogrammes locaux
    \param yx le centre du point de fixation
    \param r le rayon de la zone net
    \param l largeur de l'image
    \param h hauteur de l'image
    \param bgr_planes trois plans BGR
    \param hls_planes trois plans HLS
    \param out valeur de retour ; histrogramme sur les plan BGRHLS
*/
void HistogrammeLocaux(const Point &xy, int r, int l, int h, vector<Mat> *bgr_planes, vector<Mat> *hls_planes, vector<Mat> *out);

/*!
    \brief Calcule les histogrammes en HS
    \param yx le centre du point de fixation
    \param r le rayon de la zone net
    \param w largeur de l'image
    \param h hauteur de l'image
    \param src_hsv
    \param hist valeur de retour ; histrogramme sur les plan HS
*/
void HistogrammeLocaux3(const Point &xy,int r, int w, int h, Mat &src_hsv, MatND &hist);

MatND HistogrammeLocaux2(Mat &src);

/*!
    \brief Dessine les histogrammes locaux en un point
    \param hist histrogramme sur les plan BGRHLS
*/
void DrawHistogramme(const vector<Mat> &hist);

/*!
    \brief Extrait les points de fixations
    \param dir repertoir du fichier
    \param file ficher à lire
    \param out valeur de retour ; vecteur de points de fixation
*/

void LireXYs(const char * dir, const char *file, vector<Point> & out);

/*!
    \brief Changement de reper, passe d'un reper oculaire (-20,20;15,-15) à un repere image
    \param yx En entrée : les points de fixations dans un reper oculaire. En sourtie les points de fixations dans un image.
    \param l largeur de l'image
    \param h hauteur de l'image
*/
void XYtoPixel(vector<Point> & xy, int l, int h);

/*!
    \param Extrait les ficher ".txt" d'un dossier
    \param dir le dossier à lire
    \param out la liste de fichiers textes
*/
void FilesObservations(const char * dir, vector<char *> &out);

/*!
    cluster des points
*/
Mat clustering(Mat src, vector<Point> & xy,int clusterNum);

/*!
    \brief Delete les matrices et vide le vecteur
    \param vect vecteur a deleter
*/
void ReleaseVectorMat(vector<Mat> &vect);

typedef struct _params{
    char * img; /**< chemain de l'image */
    char * din; /**< chemin des observtions */
    char * dout; /**< chemin des histogrammes */
    int rayon; /**< rayon de la zone net */
}Params ;

typedef struct _minpos{
    int pos;
    double dis;
}min_pos;

//obtenir des sous-images correspondant des points.
void getSubImage(Mat src, vector<Point> & xy, int rayon, vector<Mat> &result);

//K-means-clustering fonction pour obtenir des zones centrées.
void kmeansclustering(Mat src,vector<Mat> &subImage,int nbcluster,int rayon, vector<min_pos> &cluster_num, vector<Mat> &label);

//
void re_centre(vector<Mat> &subImage,int nbcluster,int rayon,vector<min_pos> &cluster_num,vector<Mat> &label);
void cluster_pos(vector<Mat> &subImage,int nbcluster,vector<min_pos> &cluster_num,vector<Mat> &label);
//void inti_Cluster_Center2(Mat src, int nbcluster,int rayon, vector<Mat> &label);

//initialiser les zones centrées
void inti_Cluster_Center(int nbcluster,int rayon, vector<Mat> &label);

/*!
    \struct _paramscluster_pos
    \brief contien les parametres de l'application
*/

void cluster2(vector<Point> & xy);

void Arguments(int argc, char** argv , Params & out);
/**
 * @function main
 */
int main( int argc, char** argv )
{
    Params parametres;
    parametres.img = NULL;
    parametres.din = NULL;
    parametres.dout= NULL;
    parametres.rayon = -1;

    Arguments(argc, argv, parametres);

    printf("Execution ...");

    Mat src_rgb; //image source

    /// Load image
    src_rgb = imread( parametres.img, 1 );
    Mat src_rgb2 = imread( parametres.img, 1 );

    if( !src_rgb.data )
        { printf("Image introuvabe \n"); return -1; } //exite si l'iamge n'est pas conforme

    //namedWindow("Image", 1);
    //imshow("Image", src_rgb); //afficage de l'iamge

    /// convert image to HLS color space.
    /// The output image will be allocated automatically
    Mat src_hls;
    cvtColor(src_rgb, src_hls, CV_BGR2HLS); //image HLS

    /// Separate the BGR image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src_rgb, bgr_planes ); //separation des plans BGR

    /// Separate the HLS image in 3 places ( H, L and S )
    vector<Mat> hls_planes;
    split( src_hls, hls_planes ); //separation des plans HLS

    vector<Mat> hist; //Valeur de l'histogramme pour les plans BGRHLS

    vector<char *> files;
    FilesObservations(parametres.din, files); //chargement des observations

    if(parametres.rayon == -1 )
        parametres.rayon = src_rgb.size().width/40*2; //calcule du rayon standard

    char bufferTIME[256];
    time_t timestamp = time(NULL);

    Mat src_hsv;
    cvtColor(src_rgb, src_hsv, CV_BGR2HSV);

    vector<vector<Point> > points;

    //le totale de points
    vector<Point> myPoints;

    vector<MatND> myhist;

    for(int k = 0; k < (int)files.size() -1; ++k){ //parcour des observations

        char * adresse = (char *)malloc((strlen(parametres.dout) + strlen(files.at(k)) + 1)*sizeof(char));
        if(adresse == NULL){
            printf("erreur d'allocation : adresse");
            exit(-1);
        }

        strcpy(adresse, parametres.dout);
        strcpy(adresse + strlen(parametres.dout), files.at(k));

        FILE * pFile = fopen(adresse, "w");
        if(pFile == NULL){
            mkdir(parametres.dout, 0775);
            pFile = fopen(adresse, "w");
        }
        if(pFile != NULL){

            for( int i=0; i < 6; ++i) ///init hist
                hist.push_back(Mat( src_rgb.size().height, src_rgb.size().width, CV_8UC1, Scalar( 0,0,0) )); //initialisation des matrices d'histogramme BGRHLS

            vector<Point> points_fixation_xy;
            LireXYs(parametres.din, files.at(k), points_fixation_xy); //recuperation des points de fixations
            XYtoPixel(points_fixation_xy, src_rgb.size().width, src_rgb.size().height); //changement de repere

            points.push_back(points_fixation_xy);

            copy(points_fixation_xy.begin(),points_fixation_xy.end(),back_inserter(myPoints));

            strftime(bufferTIME, sizeof(bufferTIME), "%A %d %B %Y - %X.", localtime(&timestamp));
            fprintf(pFile, "Date de création : %s\n", bufferTIME);

            fprintf(pFile, "Nome de l'observation : %s\n\n", files.at(k));

            for(int i =0; i < (int)points_fixation_xy.size(); ++i){
                HistogrammeLocaux( points_fixation_xy.at(i), parametres.rayon, src_rgb.size().width, src_rgb.size().height, &bgr_planes, &hls_planes, &hist ); //calcule des histogrammes
                //DrawHistogramme(hist);
                MatND hist1;
                HistogrammeLocaux3( points_fixation_xy.at(i), parametres.rayon,  src_rgb.size().width, src_rgb.size().height, src_hsv, hist1);
                myhist.push_back(hist1);
                fprintf(pFile, "Point de fixation n°%d\n",i);
                fprintf(pFile, "Position en pixel (%d; %d); rayon %d\n \n", points_fixation_xy.at(i).x, points_fixation_xy.at(i).y, parametres.rayon);
                fprintf(pFile, "Valeur\tRouge\tVert\tBleu\tHue\tLightness\tSaturation \n");

                for(int j = 0; j < 256; ++j)
                    fprintf(pFile, "%d\t%f\t%f\t%f\t%f\t%f\t%f\n",j,hist[2].at<float>(j),hist[1].at<float>(j),hist[0].at<float>(j),hist[3].at<float>(j),hist[4].at<float>(j),hist[5].at<float>(j));
            }

            ReleaseVectorMat(hist); //free hist
            fprintf(pFile, "\n");
            fclose(pFile);
        }
        else
            printf("Ouvertur du fichier : %s IMPOSSIBLE", adresse);
        free(adresse);
    }


    /** ____ Désalucation finanle _____ **/
    while(!files.empty()){
        char * tmp = files.back();
        free(tmp);
        files.pop_back();
    }


    vector<Mat> subImage;
    getSubImage(src_rgb, myPoints, parametres.rayon, subImage);
    //cout<<"subimage:"<<subImage.size()<<endl;
/*
    vector<MatND> subHist;
    for(size_t i=0;i<subImage.size();i++)
    {
        subHist.push_back(HistogrammeLocaux2(subImage.at(i)));
    }

    cvWaitKey();
    double distance;
    distance = compareHist(subHist.at(0),subHist.at(1), CV_COMP_CORREL);
    cout<<"distance: "<<distance<<endl;
*/
/*
   vector<Mat> label;
   vector<min_pos> cluster_num;
   kmeansclustering(src_rgb, subImage,10, parametres.rayon, cluster_num, label);
   for(size_t i=0;i<label.size();i++)
   {
        imshow("Display window", label.at(i));
        imwrite("/tmp/Gray_Image.jpg", label.at(i) );
        cvWaitKey();
   }
*/
   /* cout<<"myout:"<<myPoints.size()<<endl;
    Mat cluster(myPoints.size(),1, CV_32F);
    cluster=clustering(src_rgb, myPoints,6);
    for(size_t i=0;i<10;i++)
    {
        cout<<i<<" "<<cluster.at<float>(i, 0)<<endl;
    }*/

    // calculer les points sur pixel.
     Mat m=clustering(src_rgb, myPoints,10);

    free(parametres.din);
    free(parametres.dout);
    free(parametres.img);

    src_rgb.release();
    src_hls.release();
    ReleaseVectorMat(bgr_planes);
    ReleaseVectorMat(hls_planes);
    ReleaseVectorMat(hist);

    destroyAllWindows();

    printf("DONE\n");
    return 0;
}

void inti_Cluster_Center(int nbcluster,int rayon, vector<Mat> &label)
{

    for(int i=0;i<nbcluster;i++)
    {
        Mat m(2*rayon, 2*rayon, CV_8UC3, Scalar(0, 0, 0));
        for(int j=0;j<rayon*2;j++)
        {
            for(int k=0;k<rayon*2;k++)
            {
                if (pow(double(k-rayon), 2) + pow(double(j-rayon), 2) - rayon*rayon < 0.00000000001)
                    m.at<Vec3b>(k, j) = Vec3b(rand()%256, rand()%256, rand()%256);
            }
        }
        label.push_back(m);
    }
}
/*
void inti_Cluster_Center2(Mat src,int nbcluster,int rayon, vector<Mat> &label)
{
    vector<Rect> rects;
    for (int i = 0; i < nbcluster; i++)
    {
        rects.push_back(Rect(i*10, i*10,2*rayon, 2*rayon));
    }

    for(int i = 0; i < nbcluster; i++)
    {
        Mat tempImg;
        src(rects[i]).copyTo(tempImg);
        label.push_back(tempImg);
    }
}
*/
void cluster_pos(vector<Mat> &subImage,int nbcluster,vector<min_pos> &cluster_num,vector<Mat> &label)
{
    double *dis=new double[nbcluster*subImage.size()]; //enregister
    vector<MatND> result1;
    vector<MatND> result2;
    for(size_t i=0; i<subImage.size(); i++)
    {
        result1.push_back(HistogrammeLocaux2(subImage.at(i)));
    }

    for(size_t i=0; i<label.size(); i++)
    {
        result2.push_back(HistogrammeLocaux2(label.at(i)));
    }

    for (int k=0;k<nbcluster;k++)      //calcule la distance entre chaque zone et chaque center.
    {
        for (size_t i=0;i<subImage.size();i++)
        {
            dis[i*nbcluster+k]=compareHist(result1.at(i),result2.at(k), CV_COMP_CORREL);
        }
    }

    min_pos mpos;
    for (size_t i=0;i<subImage.size();i++)   //classifier les zone
    {
        double dis_Int=dis[i*nbcluster+0];
        for (int k=0;k<nbcluster;k++)
        {
            if (dis[i*nbcluster+k]>=dis_Int)
            {
                dis_Int=dis[i*nbcluster+k];
                mpos.dis=dis[i*nbcluster+k];
                mpos.pos=k;
            }
        }
        cluster_num.push_back(mpos);
    }
    delete [] dis;
}


void re_centre(vector<Mat> &subImage,int nbcluster,int rayon,vector<min_pos> &cluster_num,vector<Mat> &label)
{
    for (int i=0;i<nbcluster;i++)
    {
        int count=0;
        int temp[rayon*2][rayon*2][3];
        for(int j=0;j<rayon*2;j++)
        {
            for(int k=0;k<rayon*2;k++)
            {
                temp[j][k][0]=0;
                temp[j][k][1]=0;
                temp[j][k][2]=0;
            }
        }
        for (size_t m=0;m<subImage.size();m++)
        {
            if (cluster_num.at(m).pos==i)
            {
                for(int j=0;j<rayon*2;j++)
                {
                    for(int k=0;k<rayon*2;k++)
                    {
                        temp[j][k][0]=temp[j][k][0]+int(subImage.at(m).at<Vec3b>(j, k)[0]);
                        temp[j][k][1]=temp[j][k][1]+int(subImage.at(m).at<Vec3b>(j, k)[1]);
                        temp[j][k][2]=temp[j][k][2]+int(subImage.at(m).at<Vec3b>(j, k)[2]);
                    }
                }
                count++;
            }
        }
        if(count!=0)
        {
            cout<<"  i: "<<i<<"count: "<<count<<endl;
            for(int j=0;j<rayon*2;j++)
                for(int k=0;k<rayon*2;k++)
                {
                    label.at(i).at<Vec3b>(j, k)[0]=temp[j][k][0]/count;
                    label.at(i).at<Vec3b>(j, k)[1]=temp[j][k][1]/count;
                    label.at(i).at<Vec3b>(j, k)[2]=temp[j][k][2]/count;
                }
        }
    }
}

void kmeansclustering(Mat src,vector<Mat> &subImage,int nbcluster,int rayon,vector<min_pos> &cluster_num, vector<Mat> &label)
{
    //vector<min_pos> cluster_num;
    inti_Cluster_Center(nbcluster, rayon, label);
    for(int iter=0;iter<nbcluster;++iter)
    {
        cluster_num.erase(cluster_num.begin(),cluster_num.end());
        cluster_pos(subImage,nbcluster,cluster_num,label);
        re_centre(subImage, nbcluster, rayon, cluster_num, label);
    }
}
/*
void kmeansclustering(vector<Mat> &subImage,int nbcluster,int rayon,vector<min_pos> &cluster_num, vector<Mat> &label)
{
    //vector<min_pos> cluster_num;
    inti_Cluster_Center(nbcluster, rayon, label);
    for(int iter=0;iter<nbcluster;++iter)
    {
        cluster_num.erase(cluster_num.begin(),cluster_num.end());
        cluster_pos(subImage,nbcluster,cluster_num,label);
        re_centre(subImage, nbcluster, rayon, cluster_num, label);
    }
}
*/
void getSubImage(Mat src, vector<Point> & xy, int rayon, vector<Mat> &result)
{

    vector<Rect> rects;
    for (size_t i = 0; i < xy.size(); i++)
    {
        if(xy.at(i).x-rayon>0 && xy.at(i).y-rayon>0 && xy.at(i).x+rayon<src.cols && xy.at(i).y+rayon<src.rows)
        {
            rects.push_back(Rect(xy.at(i).x-rayon, xy.at(i).y-rayon,2*rayon, 2*rayon));
        }
    }

    for(size_t i = 0; i < rects.size(); i++)
    {
        Mat tempImg;
        src(rects[i]).copyTo(tempImg);
        for ( int i = 0; i < 2*rayon; i++)
        {
            for(int j = 0; j < 2*rayon ; j++)
            {
                if (pow(double(i-rayon), 2) + pow(double(j-rayon), 2) - rayon*rayon > 0.00000000001)
                    tempImg.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
            }
        }

        result.push_back(tempImg);
    }
    //imshow("roi", roiImage);
}

/*
//on ne peut que obtenir l
void cluster2(vector<Point> & xy)
{

          Mat src = imread("/tmp/01_K949.bmp",1); //Load a color image
          Mat samples(xy.size(),3, CV_32F);//sample

          //儲存所有像素到sample裡面
            for( size_t i= 0; i<xy.size(); i++ )
            {
                samples.at<float>(i, 0) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[0];
                samples.at<float>(i, 1) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[1];
                samples.at<float>(i, 2) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[2];
            }

        int k=10;
        Mat cluster;
        int attempts = 4;
        Mat centers;
          //使用k means分群
        kmeans(samples, k, cluster,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 100), attempts,
        KMEANS_PP_CENTERS,centers );

        Mat m(src.rows, src.cols, CV_8UC3, Scalar(0, 0, 0));
        //Mat resut_image( src.size(), src.type() );
        for( size_t i= 0; i<xy.size(); i++ )
        {
            //samples.at<float>(i, 0) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[0];
            int index = cluster.at<int>(i,0);
            cout<<i<<" "<<index<<" ";
            cout<<centers.at<float>(index, 0)<<"  "<<centers.at<float>(index, 1)<<" "<<centers.at<float>(index, 2)<<endl;
            //cout<<resut_image.at<Vec3b>(xy.at(i).x, xy.at(i).y)[0]<<endl;
            m.at<Vec3b>(xy.at(i).x, xy.at(i).y)[0] = centers.at<float>(index, 0);
            m.at<Vec3b>(xy.at(i).x, xy.at(i).y)[1] = centers.at<float>(index, 1);
            m.at<Vec3b>(xy.at(i).x, xy.at(i).y)[2] = centers.at<float>(index, 2);
        }

        imshow( "result",  m );
        waitKey(0);

}
*/
Mat clustering(Mat src, vector<Point> & xy,int clusterNum){
    Mat samples(xy.size(),3, CV_32F);
    unsigned int i;
    //ajouter toutes les informations des points dans samples.
    for(  i= 0; i<xy.size(); i++ )
    {
            samples.at<float>(i, 0) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[0];
            samples.at<float>(i, 1) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[1];
            samples.at<float>(i, 2) = src.at<Vec3b>(xy.at(i).x, xy.at(i).y)[2];
    }
    //cluster enregister les informations après la classification
    //centers: les centres de gravité des points
    Mat cluster(xy.size(),1, CV_32F);
    //int [] cluster;

    Mat centers(clusterNum,3, CV_32F);
    kmeans(samples, clusterNum, cluster,TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10, 100), 4,
          KMEANS_PP_CENTERS,centers);
    for( i= 0; i<xy.size(); i++)
    {
        cout<<"le "<<i<<"ième point appartient à classe "<<cluster.at<int>(i, 0)<<endl;
    }
    cout<<"les RGB de centres sont"<<endl;
    for(int i=0;i<clusterNum;i++)
    {
        cout<<i<<" "<<centers.at<float>(i, 0)<<" "<<centers.at<float>(i, 1)<<" "<<centers.at<float>(i, 2)<<endl;
    }
    return cluster;
}

void ReleaseVectorMat(vector<Mat> &vect){
    while(!vect.empty()){
         vect.back().release();
         vect.pop_back();
    }
}

void HistogrammeLocaux3(const Point &xy,int r, int w, int h, Mat &src_hsv, MatND &hist)
{
	int hbins = 256, sbins = 256;
	int histSize[] = {hbins, sbins};

	float hranges[] = { 0, 256 };
	float sranges[] = { 0, 256 };
	int channels[] = {0, 1};
	//mask
	Scalar un(255,255,255);
	Scalar zero(0,0,0);
	Mat mask( h, w, CV_8UC1, zero);
	circle(mask, xy, r, un, -1);

	const float* ranges[] = { hranges, sranges };
	calcHist(&src_hsv, 1, channels, mask,
             hist, 2, histSize, ranges);
    normalize(hist,hist, 0, 255, NORM_MINMAX, -1, Mat() );
    mask.release();
}

MatND HistogrammeLocaux2(Mat &src)
{
    int hbins = 256, sbins = 256;
    int histSize[] = {hbins, sbins};

    float hranges[] = { 0, 256 };
    float sranges[] = { 0, 256 };
    int channels[] = {0, 1};
    const float* ranges[] = { hranges, sranges };

    Scalar un(255,255,255);
	Scalar zero(0,0,0);
	Point x=Point(src.size().width/2,src.size().height/2);
	Mat mask( src.size().width, src.size().height, CV_8UC1, zero);
	circle(mask, x, src.size().width/2, un, -1);

    MatND hist;
    Mat src_hsv;
    cvtColor(src, src_hsv, CV_BGR2HSV);
    calcHist(&src_hsv, 1, channels, mask,
        hist, 2, histSize, ranges);
        normalize(hist,hist, 0, 255, NORM_MINMAX, -1, Mat() );
   // mask.release();
    return hist;
}

void HistogrammeLocaux(const Point &xy, int r, int w, int h, vector<Mat> *bgr_planes, vector<Mat> *hls_planes, vector<Mat> *out){


  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  ///Mask
  Scalar un(255,255,255);
  Scalar zero(0,0,0);
  Mat mask( h, w, CV_8UC1, zero);
  circle(mask, xy, r, un, -1);

  /// Compute the histograms:
  calcHist( &(bgr_planes->at(0)), 1, 0, mask, out->at(0), 1, &histSize, &histRange, uniform, accumulate ); //B
  calcHist( &(bgr_planes->at(1)), 1, 0, mask, out->at(1), 1, &histSize, &histRange, uniform, accumulate ); //G
  calcHist( &(bgr_planes->at(2)), 1, 0, mask, out->at(2), 1, &histSize, &histRange, uniform, accumulate ); //R

  calcHist( &(hls_planes->at(0)), 1, 0, mask, out->at(3), 1, &histSize, &histRange, uniform, accumulate ); //H
  calcHist( &(hls_planes->at(1)), 1, 0, mask, out->at(4), 1, &histSize, &histRange, uniform, accumulate ); //L
  calcHist( &(hls_planes->at(2)), 1, 0, mask, out->at(5), 1, &histSize, &histRange, uniform, accumulate ); //S

  /// Normalize the result to [ 0, 255]
  normalize(out->at(0), out->at(0), 0, 255, NORM_MINMAX, -1, Mat() );
  normalize(out->at(1), out->at(1), 0, 255, NORM_MINMAX, -1, Mat() );
  normalize(out->at(2), out->at(2), 0, 255, NORM_MINMAX, -1, Mat() );
  normalize(out->at(3), out->at(3), 0, 255, NORM_MINMAX, -1, Mat() );
  normalize(out->at(4), out->at(4), 0, 255, NORM_MINMAX, -1, Mat() );
  normalize(out->at(5), out->at(5), 0, 255, NORM_MINMAX, -1, Mat() );

  mask.release();
}

void DrawHistogramme(const vector<Mat> &hist){
    /// Establish the number of bins
    int histSize = 256;

    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImageRGB( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    Mat histImageHLS( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImageRGB, Point( bin_w*(i-1), hist_h - cvRound(hist[0].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[0].at<float>(i)) ),
            Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImageRGB, Point( bin_w*(i-1), hist_h - cvRound(hist[1].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[1].at<float>(i)) ),
            Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImageRGB, Point( bin_w*(i-1), hist_h - cvRound(hist[2].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[2].at<float>(i)) ),
            Scalar( 0, 0, 255), 2, 8, 0  );

        line( histImageHLS, Point( bin_w*(i-1), hist_h - cvRound(hist[3].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[3].at<float>(i)) ),
            Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImageHLS, Point( bin_w*(i-1), hist_h - cvRound(hist[4].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[4].at<float>(i)) ),
            Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImageHLS, Point( bin_w*(i-1), hist_h - cvRound(hist[5].at<float>(i-1)) ) ,
            Point( bin_w*(i), hist_h - cvRound(hist[5].at<float>(i)) ),
            Scalar( 0, 0, 255), 2, 8, 0  );
    }

    /// Display
    namedWindow("calcHist BGR", CV_WINDOW_AUTOSIZE );
    namedWindow("calcHist HLS", CV_WINDOW_AUTOSIZE );
    imshow("calcHist BGR", histImageRGB );
    imshow("calcHist HLS", histImageHLS );

    waitKey(0);

    histImageRGB.release();
    histImageHLS.release();
}

void LireXYs(const char * dir, const char *file, vector<Point> & out)
{

    int i;
    float x,y,t,dt;
    char *pos;

    char *strTmp = (char *)malloc((strlen(dir) + strlen(file) + 100)*sizeof(char));
    if(strTmp == NULL){
        printf("erreur d'allocation : strTmp\n");
        exit(-1); //teste si l'allocation est prete
    }
    memset(strTmp, '0', (strlen(dir) + strlen(file) +100)*sizeof(char)); //initialisation

    strcpy(strTmp, dir);
    strcpy(strTmp + strlen(dir), file);

    ifstream ifs ( strTmp, ifstream::in ); //ouverture du fichier

    if(ifs.is_open()){ //si le fichier est ouvre
        memset(strTmp, '0',  (strlen(dir) + strlen(file) +100)*sizeof(char)); //reinitialisation

        while (ifs.getline(strTmp,100)){
            while( (pos = strchr(strTmp, ',')) != NULL) *pos = '.'; //converti les , en . pour la convertion en float
            if(sscanf(strTmp, "%d\t%f\t%f\t%f\t%f",&i,&x,&y,&t,&dt) == 5) //recherche la srtucture
                out.push_back(Point((int)(x*100), (int)(y*100))); //stock les coordonnées
        }
    }
    else{
        printf("Impossible d'ouvrir : %s\n",strTmp);
        exit(-1);
    }

    ifs.close();
    free(strTmp);
}

void XYtoPixel(vector<Point> & xy, int l, int h){
    float unit_x = l/40;
    float unit_y = h/30;
    float centre_x = l/2;
    float centre_y = h/2;

    for(size_t i = 0; i < xy.size(); ++i){
        Point point = xy.at(i); //lecture
        point.x = centre_x + (point.x*unit_x)/100; //conversion
        point.y = centre_y + (point.y*unit_y)/100; //conversion

        xy.erase(xy.begin()+i); //supression
        xy.insert(xy.begin()+i, point); //ajout du nouveau point avec les nouvelles valeurs
    }
}

void FilesObservations(const char * dir, vector<char *> &out){
    DIR *rep;
    struct dirent *lecture;

    rep = opendir(dir); //ouverture du dossier
    if(rep == NULL){
        printf("Impossible d'ouvrir le dossier : %s\n", dir);
        exit(-1);
    }

    while ((lecture = readdir(rep))) {//lit tous les fichiers
        size_t len = strlen(lecture->d_name); //taille du nom
        if(len > 4 && strcmp(lecture->d_name + len - 4, ".txt") == 0){ //si le fichier a une extansion et que c'est du ".txt"

            char * file = (char *)malloc((len +1)*sizeof(char));
            if(file == NULL){
                printf("erreur d'allocation : file\n");
                exit(-1);
            }
            strcpy(file,lecture->d_name); //ajout le fichier
            out.push_back(file); //ajout au vecteur de sortie
        }
    }
    free(rep);
}

void Arguments(int argc, char** argv , Params & out){
    size_t posStart;
    size_t posEnd;
    int minOptions = 0;

    if(( argc >= 3)&&(argc <= 4)){
        for(int i = 0 ; i < argc; ++i)
        {
            string arg = argv[i];

            if((posStart = arg.find("--image-in=")) != string::npos){
                posStart += strlen("--image-in=");

                out.img = (char *)malloc((arg.length() - posStart +1)*sizeof(char));
                if(out.img == NULL){
                    printf("erreur d'allocation : out.img");
                    exit(-1);
                }

                strncpy(out.img, arg.c_str() + posStart, arg.length() - posStart +1);
                printf("Image : %s\n", out.img);
                minOptions++;
            }
            else if((posStart = arg.find("--dossier-observations=")) != string::npos){
                posStart += strlen("--dossier-observations=");

                out.din = (char *)malloc((arg.length() - posStart +1)*sizeof(char));
                if(out.din == NULL){
                    printf("erreur d'allocation : out.din");
                    exit(-1);
                }

                strncpy(out.din, arg.c_str() + posStart, arg.length() - posStart + 1);
                printf("Observations file : %s\n", out.din);
                minOptions++;
            }
            else if((posStart = arg.find("--dossier-histogrammes=")) != string::npos){
                posStart += strlen("--dossier-histogrammes=");

                out.dout = (char *)malloc((arg.length() - posStart +1)*sizeof(char));
                if(out.dout == NULL){
                    printf("erreur d'allocation : out.dout");
                    exit(-1);
                }

                strncpy(out.dout, arg.c_str() + posStart, arg.length() - posStart + 1);
                printf("Histogrammes file : %s\n", out.dout);
                minOptions++;
            }
            else if((posStart = arg.find("--rayon=")) != string::npos){
                posEnd = arg.find_last_of("--", posStart);
                if(posEnd == string::npos) posEnd = arg.length();

                out.rayon = std::atoi(string(arg, posStart, posEnd-posStart).c_str());
                if(out.rayon < 0) out.rayon = -out.rayon;

                printf("Rayon : %d\n", out.rayon);
            }
        }
    }

    if(minOptions != 3){
        printf("\n******** CartoCulaireHistogramme ********\n");
        printf("NAME \n\t CartoCulaireHistogramme - Programme d'extraction d'histogrammes locaux.\n");
        printf("SYNOPSIS\n\t CartoCulaireHistogramme --image-in=#IMAGE# --dossier-observations=#DOSSIER_OBSERRVATION# --dossier-histogrammes=#DOSSIER_HISTOGRAMME#  [--rayon=#RAYON#]\n");
        printf("DESCRIPTION\n\t Extrait les histogrammes locaux des points de fixations contenuent dans les observations.\n");
        printf("OPTIONS\n\t --image-in= : chemin de la carte au format BMP.\n");
        printf("\t --dossier-observations= : chemin du dossier où se trouvent les observations.\n");
        printf("\t --dossier-histogrammes= : chemin du dossier où seront enregistrés les histogrammes.\n");
        printf("\t --rayon= : OPTIONNEL, définit le rayon de la zone net. Il doit être différent de zéro. Par défaut elle est de 2U soit 50 pixels pour une image 1024x768.\n");
        exit(-1);
    }
}
