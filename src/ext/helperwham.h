/*
 * function for speeding up wham code. Note that returning value
 * does not always work with cython.
 */
/*
 * Return max of a single dimension array
 */
double maxarray(double* inparray, int N);

/*
 * Compute log of sum of exponentials of a given array.
 */
void c_compute_logsum(double* inparray, int N, double* logsum);

/*
 * Compute new free energies based on old free energy and wham equations.
 * For 1D dataset.
 */
int c_update_F_k1d(long N, long nmidp,double* F_k,
		long* hist, double* U_bi,
		double beta, long* N_k,  double* F_knew, long windowZero);

/*
 * Compute new free energies based on old free energy and wham equations.
 * For 2D dataset
 */
int c_update_F_k2d(long N, long nmidpx, long nmidpy,double* F_k,
		long* hist, double* U_bij, double beta, long* N_k,
		double* g_k,  double* F_knew, long windowZero);

/*
 * minimize the free energy estimation error
 * For 2D dataset
 */

int c_minimize2d(long N, long nmidpx, long nmidpy,
        double* F_k, long* hist, double* U_bij, double beta,
        long* N_k, double* g_k,  double* F_knew, long chkdur, long windowZero);

int c_minimize1d(long N, long nmidp,double* F_k, long* hist,
	  double* U_bi, double beta, long* N_k, double* g_k,
      double* F_knew, long chkdur, long windowZero);


