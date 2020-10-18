//Import libraries
#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <climits>
#include <vector>
#include <map>

using namespace std::chrono;
using namespace std;

class XC{
	private:
        //Declare all variables.
        vector<double> Omega_b; vector<double> h; vector<double> Omega_m; vector<double> ns; vector<double> Omega_DE;
        vector<double> w0; vector<double> wa; vector<double> sigma_8; vector<double> gamma;
        vector<double> C_IA; vector<double> A_IA; vector<double> n_IA; vector<double> B_IA;
        double eps_Omega_b, eps_h, eps_Omega_m, eps_ns, eps_Omega_DE, eps_w0, eps_wa, eps_sigma_8, eps_gamma, eps_C_IA, eps_A_IA, eps_n_IA, eps_B_IA, eps_b;
        vector<vector<double>> BX;
        double c, fsky, zrange_0, ng, cb, zb, sig_b, c0, z0, sig_0, f_out, sig_epsilon, alpha;
        int NNN, MMM, WWW, stencil;
        int red_B; int der_B; int paramo;
    	vector<double> zrange, z_ip, z_im, z_alot; double zmax; double zmin;
    	vector<string> fold_path; vector<string> pre_CC_path; vector<string> CC_path; string Pk_Path;
    	vector<double> l_edge, l; double l_max_WL, l_max_GC, delta_l;
    	vector<double> zpm; vector<double> zsec; vector<double> zbkgnd; vector<double> z_win_x_y;
    	vector<double> z_win_3_1; vector<double> z_win_1_3; vector<double> z_win_1_2; vector<double> z_win;
    	vector<double> z500; vector<double> z_pk; double delta_zpm, delta_zsec, delta_z500;
    	vector<double> E_tab, E_tab_up, E_tab_dw, DG_tab, DG_tab_up, DG_tab_dw;
    	vector<double> R_tab, R_tab_up, R_tab_dw, RT_tab, RT_tab_up, RT_tab_dw;
    	vector<vector<double>> Photoz_tab;
    	vector<vector<double>> W_tab, W_tab_up, W_tab_dw, WG_tab, WG_tab_up, WG_tab_dw, WIA_tab, WIA_tab_up, WIA_tab_dw;
        vector<vector<vector<double>>> P_dd_C, P_dd_C_up, P_dd_C_dw; vector<string> Pk_files;
        vector<vector<double>> C_ij_GG, C_ij_LL, C_ij_GL, C_ij_GG_up, C_ij_LL_up, C_ij_GL_up, C_ij_GG_dw, C_ij_LL_dw, C_ij_GL_dw;
        vector<double> z_lum, l_lum, lum_mean;

        vector<string> param_chain_Cl, param_chain_A; vector<double> fid_Cl, fid_A, steps_Cl, steps_A;
        int lsize, lsize2, Dim_x, Dim_y;

        string curvature, gamma_MG, zcut;

        double progress; int NP;

    public:
        //Default constructor
    	XC(int, int, int, double, double, double, int, int, int, int, string, string, string); 
        //Methods
        void Initializers_G(int);
        void Initializers_Pk();
    	void background();
    	void photoz();
    	void photoz_load();
        void C_l_computing();
        void Fisher(string, string, int, int, int, int, int);
    	double nz(double);
    	double P_phot(double, double);
    	void windows();
    	double P_shot_WL(double, double);
    	double P_shot_GC(double, double);

        double barometer(double, int);
        void nz_integ(int);

    	~XC();
};