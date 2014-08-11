#include <math.h>
#include <stdio.h>
#include <omp.h>
#include "helperwham.h"

double maxarray(double* inparray, int N)
{
	int i;
	double j;
	j = inparray[0];

	for (i = 1; i < N; i++ )
	{
		if (inparray[i] > j){
			j = inparray[i];
		}
	}
	return j;
}

void c_compute_logsum(double* inparray, int N,double* logsum)
{
	int i;
	*logsum = 0;
	double ArrayMax;

	ArrayMax = maxarray(inparray,N);
	for (i = 0; i < N; i++)
	{
		*logsum += exp( inparray[i]- ArrayMax );
	}
	if (*logsum != 0)
	{
	    *logsum = log(*logsum) + ArrayMax;
	}
	else
	{
	    *logsum = ArrayMax;
	}
}

int c_update_F_k2d(long N, long nmidpx, long nmidpy,double* F_k,
		long* hist_pointer, double* U_bij_pointer,
		double beta, long* N_k,  double* g_k,double* F_knew, long windowZero)
{
	int i,j,k;
	double logbf[N];
	double denom,feZero;
	//recasting for easy array manipulations.
	double (*U_bij)[nmidpy][N] = (double (*)[nmidpy][N] ) U_bij_pointer;
    long (*hist)[nmidpy] = (long (*)[nmidpy]) hist_pointer;

	for (k=0; k<N; k++)
	{
		F_knew[k] = 0.0 ;
	}

	for (i=0; i<nmidpx; i++)
	{
		for (j=0; j<nmidpy; j++)
		{
            #pragma omp parallel for default(shared) private(k)
			for (k=0; k<N ;k++)
			{
				if (N_k[k] != 0)
				{
					logbf[k] = beta * (F_k[k] - U_bij[i][j][k]) + log(N_k[k]/g_k[k]) ;
				}
				else
				{
					logbf[k] = 0;
				}
			}

			c_compute_logsum(logbf, N,&denom);
			if (hist[i][j] != 0)
			{
				#pragma omp parallel for default(shared) private(k)
				for (k=0; k<N ;k++)
				{
					logbf[k] = (-1. * beta * U_bij[i][j][k] ) + log(hist[i][j]/g_k[k]) - denom ;
					F_knew[k] += exp(logbf[k]);
				}
			}

		}
	}

	if (F_knew[windowZero] != 0)
	{
		feZero = -log(F_knew[windowZero])/beta ;
	}
	else
	{
		feZero = 0;
	}

	for (k=0; k<N ;k++)
	{
		if (F_knew[k] != 0)
		{
			F_knew[k] = -log(F_knew[k])/beta ;
			F_knew[k] -= feZero ;
		}
	}



	return 1;
}

int c_update_F_k1d(long N, long nmidp,double* F_k,
		long* hist, double* U_bi,
		double beta, long* N_k,  double* F_knew, long windowZero)

{
	int i,k;
	double logbf[N];
	double denom,feZero;
	double (*U_b)[N] = (double (*)[N]) U_bi;

	for (k=0; k<N; k++)
	{
		F_knew[k] = 0.0 ;
	}

	for (i=0; i<nmidp; i++)
	{
		#pragma omp parallel for default(shared) private(k)
		for (k=0; k<N ;k++)
		{
			if (N_k[k] != 0)
			{
				logbf[k] = beta * (F_k[k] - U_b[i][k]) + log(N_k[k]) ;
			}
			else
			{
				logbf[k] = 0;
			}
		}

		c_compute_logsum(logbf, N,&denom);
		if (hist[i] != 0)
		{
			#pragma omp parallel for default(shared) private(k)
			for (k=0; k<N ;k++)
			{
				logbf[k] = (-1. * beta * U_b[i][k] ) + log(hist[i]) - denom ;
				F_knew[k] += exp(logbf[k]);
			}
		}
	}

	if (F_knew[windowZero] != 0)
	{
		feZero = -log(F_knew[windowZero])/beta ;
	}
	else
	{
		feZero = 0;
	}


	for (k=0; k<N ;k++)
	{
		if (F_knew[k] != 0)
		{
			F_knew[k] = -log(F_knew[k])/beta ;
			F_knew[k] -= feZero ;
		}
	}
	return 1;
}

int c_minimize2d(long N, long nmidpx, long nmidpy,
        double* F_k, long* hist, double* U_bij, double beta,
        long* N_k, double* g_k,  double* F_knew, long chkdur,long windowZero)

{
	int i = 0;
	int k;
	int ret = -1;
	double diff=1.0;
	double DBL_EPSILON = 1e-15;

	/* Make n-1 nterations and set F_k to F_knew
	 * for nth iteration do not set F_k to F_knew
	 */
	for ( i=0; i < (chkdur-1) || diff <= DBL_EPSILON; i++)
	{
		ret = c_update_F_k2d(N,nmidpx,nmidpy, F_k,
				hist, U_bij, beta, N_k, g_k, F_knew, windowZero);
		diff = 0.0;
		for (k=0; k<N; k++)
		{
			diff += fabs(F_knew[k] - F_k[k]);
			F_k[k] = F_knew[k];
			//printf("%f %f %d\n",F_knew[k],F_k[k],k);

		}

	}
	ret = c_update_F_k2d(N,nmidpx,nmidpy, F_k,
			hist, U_bij, beta, N_k, g_k, F_knew,windowZero);

	return ret;
}

int c_minimize1d(long N, long nmidp,double* F_k, long* hist,
	  double* U_bi, double beta, long* N_k, double* g_k,
      double* F_knew, long chkdur, long windowZero )

{
	int i = 0;
	int k;
	int ret = -1;
	double diff=1.0;
	double DBL_EPSILON = 1e-15;

	/* Make n-1 nterations and set F_k to F_knew
	 * for nth iteration do not set F_k to F_knew
	 */
	for ( i=0; i < (chkdur-1) || diff <= DBL_EPSILON; i++)
	{
		ret = c_update_F_k1d( N, nmidp, F_k,
				hist, U_bi, beta, N_k, F_knew, windowZero);
		diff = 0.0;
		for (k=0; k<N; k++)
		{
			diff += fabs(F_knew[k] - F_k[k]);
			F_k[k] = F_knew[k];
		}
	}
	ret = c_update_F_k1d( N, nmidp, F_k,
			hist, U_bi, beta, N_k, F_knew,windowZero);

	return ret;
}
