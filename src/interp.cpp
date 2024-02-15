#include "math.h"
#include "interp.h"
#include "ftk/ndarray.hh"
void interp2d(const double p[2], double v[2],const ftk::ndarray<double> &grad){
  double X[3][2];
  double V[3][2];
  int x0 = floor(p[0]);
  int y0 = floor(p[1]);
  float x_ex = p[0] - x0;
  float y_ex = p[1] - y0;
  int upper =1;
  if (y_ex > x_ex){
    upper = 1;
  }
  else{
    upper = 0;
  }
  if (upper == 1){
      X[0][0] = x0;
      X[0][1] = y0;
      X[1][0] = x0;
      X[1][1] = y0+1;
      X[2][0] = x0+1;
      X[2][1] = y0+1;
      for (int i =0;i <2;i++){
        V[0][i] = grad(i, x0, y0);
        V[1][i] = grad(i, x0, y0+1);
        V[2][i] = grad(i, x0+1, y0+1);
      }
    }
  else{
    X[0][0] = x0;
    X[0][1] = y0;
    X[1][0] = x0+1;
    X[1][1] = y0;
    X[2][0] = x0+1;
    X[2][1] = y0+1;
    for (int i =0;i <2;i++){
      V[0][i] = grad(i, x0, y0);
      V[1][i] = grad(i, x0+1, y0);
      V[2][i] = grad(i, x0+1, y0+1);
    }
  }
  double lambda[3];
  barycent2d(X[0], X[1], X[2], p, lambda);
  v[0] = lambda[0]*V[0][0] + lambda[1]*V[1][0] + lambda[2]*V[2][0];
  v[1] = lambda[0]*V[0][1] + lambda[1]*V[1][1] + lambda[2]*V[2][1];

}

double triarea(double a, double b, double c)

{

    double s = (a + b + c)/2.0;

    double area=sqrt(fabs(s*(s-a)*(s-b)*(s-c)));

    return area;     

}

double dist(double x0, double y0, double z0, double x1, double y1, double z1)

{

    double a = x1 - x0;	  

    double b = y1 - y0;

    double c = z1 - z0;

    return sqrt(a*a + b*b + c*c);

}

void barycent2d(double *p0, double *p1, double *p2, const double *v, double *lambda )
{

	double x0 = p0[0], y0 = p0[1], z0 = 0;
	double x1 = p1[0], y1 = p1[1], z1 = 0;
	double x2 = p2[0], y2 = p2[1], z2 = 0;
	double vx = v[0], vy = v[1], vz = 0;

    // compute the area of the big triangle

    double a = dist(x0, y0, z0, x1, y1, z1);
    double b = dist(x1, y1, z1, x2, y2, z2);
    double c = dist(x2, y2, z2, x0, y0, z0);

    double totalarea = triarea(a, b, c);

	

    // compute the distances from the outer vertices to the inner vertex

    double length0 = dist(x0, y0, z0, vx, vy, vz);	  

    double length1 = dist(x1, y1, z1, vx, vy, vz);	  

    double length2 = dist(x2, y2, z2, vx, vy, vz);	  

    

    // divide the area of each small triangle by the area of the big triangle

    lambda[0] = triarea(b, length1, length2)/totalarea;

    lambda[1] = triarea(c, length0, length2)/totalarea;

    lambda[2] = triarea(a, length0, length1)/totalarea;	  

}

