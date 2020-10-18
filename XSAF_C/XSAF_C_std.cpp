#include "XSAF_C.h"

using namespace std;

//FUNCTIONS

//Compute the b(z) in the H(z) equation.
double b_aux(double z, double w0_p, double wa_p){
	return 1./(1+z)*(1+w0_p+wa_p*z/(1+z));
}

//Compute the integral of b(z) in the H(z) equation.
double b_new(double z, double w0_p, double wa_p){

	double step_integ = 0.1*z;
	int N_step = 11;
	vector<double> zzz;
	for(int k=0; k<=N_step; k++){
		zzz.push_back(z*k/(N_step-1));
	}

	double integrale_E = 0;
	for(int k=1; k<N_step; k++){
		integrale_E += 1./90*step_integ*(7.*(b_aux(zzz[k], w0_p, wa_p)+(b_aux(zzz[k-1], w0_p, wa_p))) + 32.*(b_aux(1./4*(3.*zzz[k]+zzz[k-1]), w0_p, wa_p) + b_aux(1./4*(zzz[k]+3*zzz[k-1]), w0_p, wa_p)) + 12*(b_aux(1./2*(zzz[k]+zzz[k-1]), w0_p, wa_p)));
	}
	return integrale_E;
}

//Compute E(z).
double E(double z, double h_p, double wm_p, double wDE_p, double w0_p, double wa_p, string curvature){
	if(curvature == "F"){
		return pow(wm_p*pow((1+z),3) + (1.-wm_p)*exp(3.*b_new(z, w0_p, wa_p)), 0.5);
	}
	else if(curvature == "NF"){
		return pow(wm_p*pow((1+z),3) + wDE_p*exp(3.*b_new(z, w0_p, wa_p)) + (1.-wm_p-wDE_p)*pow(1+z,2), 0.5);
	}
	else{
		return 0;
	}
}

//Compute R(z).
double chi_aux_R(double z, double h_p, double wm_p, double wDE_p, double w0_p, double wa_p, double c, string curvature){
    return c/(100.*h_p*(E(z, h_p, wm_p, wDE_p, w0_p, wa_p, curvature)));
}
   
//Compute integral of E(z). 
double chi_R(double z, double h_p, double wm_p, double wDE_p, double w0_p, double wa_p, double c, string curvature){ 
    double step_integ_R = 0.1*z;
	int N_step_R = 11;
	vector<double> zzz_R;
	for(int k=0; k<=N_step_R; k++){
		zzz_R.push_back(z*k/(N_step_R-1));
	}

    double integrale_R = 0;
    for(int j=1; j<N_step_R; j++){
		integrale_R += 1./90*step_integ_R*(7.*(chi_aux_R(zzz_R[j], h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature)+(chi_aux_R(zzz_R[j-1], h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature))) + 32.*(chi_aux_R(1./4*(3.*zzz_R[j]+zzz_R[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature) + chi_aux_R(1./4*(zzz_R[j]+3*zzz_R[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature)) + 12*(chi_aux_R(1./2*(zzz_R[j]+zzz_R[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature)));
	}
	return integrale_R;
}

//Compute Omega_m(z).
double OM_M(double z, double h_p, double wm_p, double wDE_p, double w0_p, double wa_p, double gamma_p, string curvature){
    return pow(wm_p*pow(1.+z,3)/(pow(E(z, h_p, wm_p, wDE_p, w0_p, wa_p, curvature),2)),gamma_p)/(1.+z);
}

//Compute G(z).
double DGrowth(double z, double h_p, double wm_p, double wDE_p, double w0_p, double wa_p, double gamma_p, string curvature){
    double step_integ_G = 0.1*z;
	int N_step_G = 11;
	vector<double> zzz_G;
	for(int k=0; k<=N_step_G; k++){
		zzz_G.push_back(z*k/(N_step_G-1));
	}

    double integrale_G = 0;
    for(int j=1; j<N_step_G; j++){
		integrale_G += 1./90*step_integ_G*(7.*(OM_M(zzz_G[j], h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature)+(OM_M(zzz_G[j-1], h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature))) + 32.*(OM_M(1./4*(3.*zzz_G[j]+zzz_G[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature) + OM_M(1./4*(zzz_G[j]+3*zzz_G[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature)) + 12*(OM_M(1./2*(zzz_G[j]+zzz_G[j-1]), h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature)));
	}
	return exp(-integrale_G);
}

//Inverse the covariance matrix.
vector<vector<double>> matrix_inverse(vector<vector<double>> F_matrix_A, vector<vector<double>> F_previous){
	cout<<"Begin the LT matrix inversion"<<endl;
    vector<double> F_temp(F_matrix_A.size()); vector<double> temp_prev(F_matrix_A.size());
    for(int i=0; i < F_matrix_A.size(); i++){
        for(int j=0; j < F_matrix_A.size(); j++){
            if(i == j){
                for(int k=0; k < F_matrix_A.size(); k++) {
                    if(F_previous[j][k] != 0){
                        F_previous[j][k] = F_previous[j][k]/F_matrix_A[j][i];
                        F_matrix_A[j][k] = F_matrix_A[j][k]/F_matrix_A[j][i];
                    }
                }
            }
            else{
                if(F_matrix_A[j][i] != 0){
                    for(int k=0; k < F_matrix_A.size(); k++) {
                    	if(F_previous[i][k] != 0){
                        	F_previous[j][k] = F_previous[j][k] - F_matrix_A[j][i]*F_previous[i][k];
                        	F_matrix_A[j][k] = F_matrix_A[j][k] - F_matrix_A[j][i]*F_matrix_A[i][k];
                        }
                    }
                }
            }
        }
    }

    vector<vector<double>> Fisher_new(F_matrix_A.size(), vector<double>(F_matrix_A.size(), 0));
    vector<vector<double>> F_previous_T(F_matrix_A.size(), vector<double>(F_matrix_A.size(), 0));

    #pragma omp parallel for schedule(dynamic,8)
    for(int i=0; i<F_matrix_A.size(); i++){
        for(int j=0; j<F_matrix_A.size(); j++){
            F_previous_T[i][j] = F_previous[j][i];
        }
    }

    #pragma omp parallel for schedule(dynamic,8)
    for(int i=0; i<F_matrix_A.size(); i++){
    	for(int k=0; k<F_matrix_A.size(); k++){
            if(F_previous_T[i][k] != 0){
        	    for(int j=0; j<=i; j++){
                    if(F_previous[k][j] !=0){
                	   Fisher_new[i][j] += F_previous_T[i][k]*F_previous[k][j];
                    }
                }
            }
        }
    }

    cout<<"Building the inverse covariance matrix"<<endl;
    #pragma omp parallel for schedule(dynamic,8)
    for(int i=0; i<F_matrix_A.size(); i++){
    	for(int j=i; j<F_matrix_A.size(); j++){
    		Fisher_new[i][j] = Fisher_new[j][i];
    	}
    }
    return Fisher_new;
}

//CLASS METHODS
//Default constructor. Initialize all the needed values to compute the Cl.
XC::XC(int D1, int D2, int D_l, double l_min, double lmaxWL, double lmaxGC, int NB_WinF, int NB_Window, int der_pts, int para_all, string curv_F_NF, string g_MG, string zcut_pess){
	cout<<endl;

    vector<string> elements_0; vector<double> elements_1;

    int ct=0;
    ifstream ifile("../QTLauncher/Parameters_W.txt");
    while(!ifile.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile>>elements_0[ct];
        ifile>>elements_1[ct];
        ct++;
    }
    ifile.close();
    ifstream ifile2("../QTLauncher/XSAF_W.txt");
    while(!ifile2.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile2>>elements_0[ct];
        ifile2>>elements_1[ct];
        ct++;
    }
    ifile2.close();

    map<string, double> XSAF_elts;
    for(int i=0; i<elements_0.size(); i++){
        XSAF_elts[elements_0[i]] = double(elements_1[i]);
    }

	curvature = curv_F_NF; gamma_MG = g_MG; zcut = zcut_pess;
	red_B = D1; der_B = D2; MMM = NB_WinF; WWW = NB_Window, stencil = der_pts;
    zmin = XSAF_elts["Vzmin"]; zmax = XSAF_elts["Vzmax"];
    l_max_WL = lmaxWL; l_max_GC = lmaxGC;
    c = 299792.458; fsky = XSAF_elts["VSurfArea"]/41253.0; zrange_0 = XSAF_elts["Vzrange_0"]; alpha=XSAF_elts["Valpha"]; ng = XSAF_elts["VGdensity"]/(8.461594994079999e-08); 
    cb = XSAF_elts["Vphotoz_cb"]; zb = XSAF_elts["Vphotoz_zb"]; sig_b = XSAF_elts["Vphotoz_sigb"]; c0 = XSAF_elts["Vphotoz_c0"]; z0 = XSAF_elts["Vphotoz_z0"]; sig_0 = XSAF_elts["Vphotoz_sig0"]; f_out = XSAF_elts["Vphotoz_f_out"]; sig_epsilon = XSAF_elts["Vsig_epsilon"];
    
    vector<double> Z_NB(10000000, 0); double integrale = 0; double integrale_bis = 0;
    for(int i=0; i<Z_NB.size(); i++){
        Z_NB[i] = zmin + (zmax-zmin)*i/(Z_NB.size()-1);
    }
    for(int i=1; i<Z_NB.size(); i++){
        integrale += 1./90*(Z_NB[i]-Z_NB[i-1])*(7*(nz(Z_NB[i]) + nz(Z_NB[i-1])) + 32*(nz(0.25*(3*Z_NB[i]+Z_NB[i-1])) + nz(0.25*(Z_NB[i]+3*Z_NB[i-1]))) + 12*nz(0.5*(Z_NB[i]+Z_NB[i-1])));
    }

    float bin_N=0./red_B;
    for(int i=1; i<Z_NB.size(); i++){
        integrale_bis += 0.5*(Z_NB[1]-Z_NB[0])*(nz(Z_NB[i]) + nz(Z_NB[i-1]));
        if(integrale_bis >= bin_N*integrale){
            z_im.push_back(Z_NB[i-1]);
            bin_N+=1./red_B;
        }
    }

    for(int i=1; i<red_B; i++){
        z_ip.push_back(z_im[i]);

    }
    z_ip.push_back(zmax);

    for(int i=0; i<red_B; i++){
        zrange.push_back((z_ip[i]+z_im[i])/2);
    }

    for(int i=0; i < D1; i++){
    	BX.push_back(vector<double>());
    	for(int j=0; j < D2; j++){
    		BX[i].push_back(pow(1+zrange[i], 0.5));
    		if(zcut == "Y"){
            	if(i > 4){
                	BX[i][j] = 0.;
                }
            }
    	}
    }

    for(int i=0; i<D2; i++){
    	Omega_b.push_back(XSAF_elts["VFidOmegab"]); h.push_back(XSAF_elts["VFidh"]); Omega_m.push_back(XSAF_elts["VFidOmegam"]); ns.push_back(XSAF_elts["VFidns"]); Omega_DE.push_back(XSAF_elts["VFidOmegaDE"]);
    	w0.push_back(XSAF_elts["VFidw0"]); wa.push_back(XSAF_elts["VFidwa"]); sigma_8.push_back(XSAF_elts["VFidsigma8"]); gamma.push_back(XSAF_elts["VFidgamma"]);
    	C_IA.push_back(XSAF_elts["VFidWLCia"]); A_IA.push_back(XSAF_elts["VFidWLAia"]); n_IA.push_back(XSAF_elts["VFidWLnia"]); B_IA.push_back(XSAF_elts["VFidWLBia"]);
    }
    eps_Omega_b = XSAF_elts["VStepOmegab"], eps_h = XSAF_elts["VSteph"], eps_Omega_m = XSAF_elts["VStepOmegam"], eps_ns = XSAF_elts["VStepns"], eps_Omega_DE = XSAF_elts["VStepOmegaDE"], eps_w0 = XSAF_elts["VStepw0"], eps_wa = XSAF_elts["VStepwa"];
    eps_sigma_8 = XSAF_elts["VStepsigma8"], eps_gamma = XSAF_elts["VStepgamma"], eps_A_IA = XSAF_elts["VStepWLAia"], eps_n_IA = XSAF_elts["VStepWLnia"], eps_B_IA = XSAF_elts["VStepWLBia"], eps_b = XSAF_elts["VStepGCphotbias"];

    fid_A = {Omega_m[2], Omega_DE[2], Omega_b[2], w0[2], wa[2], h[2], ns[2], sigma_8[2], gamma[2], A_IA[2], n_IA[2], B_IA[2]};
    steps_A = {eps_Omega_m, eps_Omega_DE, eps_Omega_b, eps_w0, eps_wa, eps_h, eps_ns, eps_sigma_8, eps_gamma, eps_A_IA, eps_n_IA, eps_B_IA};
    for(int i=0; i<zrange.size(); i++){
        fid_A.push_back(BX[i][2]);
        steps_A.push_back(eps_b);
    }

    for(int i=0; i < MMM; i++){
    	z500.push_back(zmin+(zmax-zmin)*i/(MMM-1));
    }
    for(int i=0; i < WWW; i++){
    	z_win_x_y.push_back(zmin+(zmax-zmin)*i/(WWW-1));
    	if(i > 0){
    		z_win_3_1.push_back(1./4*(3*z_win_x_y[i]+z_win_x_y[i-1]));
    		z_win_1_3.push_back(1./4*(z_win_x_y[i]+3*z_win_x_y[i-1]));
    		z_win_1_2.push_back(1./2*(z_win_x_y[i]+z_win_x_y[i-1]));
    	}
   	}
   	delta_z500 = z500[1]-z500[0];

   	int j=0; int i=0; int z_win_all_len = z_win_x_y.size()+z_win_3_1.size()+z_win_1_3.size()+z_win_1_2.size();
   	while (i < z_win_all_len){
   		if(i == 0){
        	z_win.push_back(z_win_x_y[j]);
        	i=i+1;
        	j=j+1;
        }
    	else if(i > 0 and i < z_win_all_len){
    		z_win.push_back(z_win_1_3[j-1]);
        	z_win.push_back(z_win_1_2[j-1]);
        	z_win.push_back(z_win_3_1[j-1]);
        	z_win.push_back(z_win_x_y[j]);
        	i=i+4;
        	j=j+1;
        }
   	}

    Pk_Path = "WP_Pk/"; fold_path = {Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid"};
    pre_CC_path = {"Cl_GG/", "Cl_LL/", "Cl_GL/"}; CC_path = {"C_wb_up", "C_wb_up2", "C_fid", "C_wb_dw", "C_wb_dw2"};
    param_chain_A = {"wm", "wde", "wb", "w0", "wa", "h", "ns", "s8", "gamma", "A_IA", "n_IA", "B_IA"};
    for(int i=0; i<zrange.size(); i++){
        param_chain_A.push_back("b"+to_string(i+1));
    }
	param_chain_Cl = {};

    string lines_L; int count=0;
    ifstream ifile_lum("scaledmeanlum-E2Sa.dat");
	while (!ifile_lum.eof()){
		z_lum.push_back(0); l_lum.push_back(0);
		ifile_lum>>z_lum[count]; ifile_lum>>l_lum[count];
		getline(ifile_lum,lines_L); count++;
	}
	count = 0;
	ifile_lum.close();

	for(int i=0; i < z_win.size(); i++){
		for(int j=0; j < z_lum.size()-1; j++){
			if(z_win[i] > z_lum[j] && z_win[i] <= z_lum[j+1]){
				lum_mean.push_back(l_lum[j] + (l_lum[j+1]-l_lum[j])*(z_win[i]-z_lum[j])/(z_lum[j+1]-z_lum[j]));
			}
		}
	}


    for(int i=0; i <= D_l; i++){
        if(l_max_WL>=l_max_GC){
        	l_edge.push_back(l_min+(l_max_WL-l_min)*i/(D_l));
        	if(i>0){
        		l.push_back((l_edge[i] + l_edge[i-1])/2);
        	}
        }
        if(l_max_WL<l_max_GC){
            l_edge.push_back(l_min+(l_max_GC-l_min)*i/(D_l));
            if(i>0){
                l.push_back((l_edge[i] + l_edge[i-1])/2);
            }
        }
   	}
   	delta_l = l[1]-l[0];

   	for(int i=0; i<z_win.size(); i++){
   		E_tab.push_back(0); E_tab_up.push_back(0); E_tab_dw.push_back(0);
		R_tab.push_back(0); R_tab_up.push_back(0); R_tab_dw.push_back(0);
		RT_tab.push_back(0); RT_tab_up.push_back(0); RT_tab_dw.push_back(0);
		DG_tab.push_back(0); DG_tab_up.push_back(0); DG_tab_dw.push_back(0);
	}

   	for(int i=0; i<z_win.size(); i++){
		Photoz_tab.push_back(vector<double>());
		for(int j=0; j<zrange.size(); j++){
			Photoz_tab[i].push_back(0);
		}
	}

	for(int i=0; i<z_win.size(); i++){
		WG_tab.push_back(vector<double>()); WG_tab_up.push_back(vector<double>()); WG_tab_dw.push_back(vector<double>());
		W_tab.push_back(vector<double>()); W_tab_up.push_back(vector<double>()); W_tab_dw.push_back(vector<double>());
		WIA_tab.push_back(vector<double>()); WIA_tab_up.push_back(vector<double>()); WIA_tab_dw.push_back(vector<double>());
		for(int j=0; j<zrange.size(); j++){
			WG_tab[i].push_back(0); WG_tab_up[i].push_back(0); WG_tab_dw[i].push_back(0);
			W_tab[i].push_back(0); W_tab_up[i].push_back(0); W_tab_dw[i].push_back(0);
			WIA_tab[i].push_back(0); WIA_tab_up[i].push_back(0); WIA_tab_dw[i].push_back(0);
		}
	}

	for(int i=0; i<zrange.size(); i++){
		C_ij_GG.push_back(vector<double>()); C_ij_GG_up.push_back(vector<double>()); C_ij_GG_dw.push_back(vector<double>());
		C_ij_LL.push_back(vector<double>()); C_ij_LL_up.push_back(vector<double>()); C_ij_LL_dw.push_back(vector<double>());
		C_ij_GL.push_back(vector<double>()); C_ij_GL_up.push_back(vector<double>()); C_ij_GL_dw.push_back(vector<double>());
		for(int j=0; j<zrange.size(); j++){
			C_ij_GG[i].push_back(0); C_ij_GG_up[i].push_back(0); C_ij_GG_dw[i].push_back(0);
			C_ij_LL[i].push_back(0); C_ij_LL_up[i].push_back(0); C_ij_LL_dw[i].push_back(0);
			C_ij_GL[i].push_back(0); C_ij_GL_up[i].push_back(0); C_ij_GL_dw[i].push_back(0);
		}
	}

	lsize=0;
	for(int i=0; i<l.size(); i++){
		if(l[i]<=min(l_max_WL,l_max_GC)){
			lsize++;
		}
	}

	lsize2 = l.size()-lsize;

	Dim_x = 0;
	for(int i=1; i<zrange.size()+1; i++){
		Dim_x += i;
	}
	Dim_y = pow(zrange.size(),2);

	NP = para_all*5; progress = 1./NP; 
	cout<<"Computing Cl's"<<endl;
	barometer(progress, NP);
}

//Reinitialize all parameters values as fiducials. Initialize all the power spectrums and Cl paths.
void XC::Initializers_G(int N_paramo){

	paramo = N_paramo;
	param_chain_Cl.push_back(param_chain_A[paramo]);
	fid_Cl.push_back(fid_A[paramo]); steps_Cl.push_back(steps_A[paramo]);

	for(int i=0; i<BX[0].size(); i++){
    	Omega_b[i]=Omega_b[2]; h[i]=h[2]; Omega_m[i]=Omega_m[2]; ns[i]=ns[2]; Omega_DE[i]=Omega_DE[2];
    	w0[i]=w0[2]; wa[i]=wa[2]; sigma_8[i]=sigma_8[2]; gamma[i]=gamma[2];
    	A_IA[i]=A_IA[2]; n_IA[i]=n_IA[2]; B_IA[i]=B_IA[2];
    	for(int j=0; j<BX.size(); j++){
    		BX[j][i]=BX[j][2];
    	}
    }

    fold_path = {Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid", Pk_Path+"fid"};

    if(paramo == 0){
    	Omega_m[0] *= (1.+eps_Omega_m); Omega_m[1] *= (1.+2*eps_Omega_m); Omega_m[3] *= (1.-eps_Omega_m); Omega_m[4] *= (1.-2*eps_Omega_m);
    	fold_path = {Pk_Path+"wm_up", Pk_Path+"wm_up2", Pk_Path+"fid", Pk_Path+"wm_dw", Pk_Path+"wm_dw2"};
    	CC_path = {"C_wm_up", "C_wm_up2", "C_fid", "C_wm_dw", "C_wm_dw2"};
	}
	else if(paramo == 1){
    	Omega_DE[0] *= (1.+eps_Omega_DE); Omega_DE[1] *= (1.+2*eps_Omega_DE); Omega_DE[3] *= (1.-eps_Omega_DE); Omega_DE[4]*= (1.-2*eps_Omega_DE);  
    	fold_path = {Pk_Path+"wde_up", Pk_Path+"wde_up2", Pk_Path+"fid", Pk_Path+"wde_dw", Pk_Path+"wde_dw2"};
    	CC_path = {"C_wde_up", "C_wde_up2", "C_fid", "C_wde_dw", "C_wde_dw2"};
	}
	else if(paramo == 2){
    	fold_path = {Pk_Path+"wb_up", Pk_Path+"wb_up2", Pk_Path+"fid", Pk_Path+"wb_dw", Pk_Path+"wb_dw2"};
    	CC_path = {"C_wb_up", "C_wb_up2", "C_fid", "C_wb_dw", "C_wb_dw2"};
    }
    else if(paramo == 3){
	    w0[0] *= (1.+eps_w0); w0[1] *= (1.+2*eps_w0); w0[3] *= (1.-eps_w0), w0[4] *= (1.-2*eps_w0);
	    fold_path = {Pk_Path+"w0_up", Pk_Path+"w0_up2", Pk_Path+"fid", Pk_Path+"w0_dw", Pk_Path+"w0_dw2"};
	    CC_path = {"C_w0_up", "C_w0_up2", "C_fid", "C_w0_dw", "C_w0_dw2"};
	}
	else if(paramo == 4){
	    wa[0] += eps_wa; wa[1] += 2*eps_wa; wa[3] -= eps_wa, wa[4] -= 2*eps_wa;
	    fold_path = {Pk_Path+"wa_up", Pk_Path+"wa_up2", Pk_Path+"fid", Pk_Path+"wa_dw", Pk_Path+"wa_dw2"};
	    CC_path = {"C_wa_up", "C_wa_up2", "C_fid", "C_wa_dw", "C_wa_dw2"};
	}
	else if(paramo == 5){
    	h[0] *= (1.+eps_h); h[1] *= (1.+2*eps_h); h[3] *= (1.-eps_h), h[4] *= (1.-2*eps_h);
    	fold_path = {Pk_Path+"h_up", Pk_Path+"h_up2", Pk_Path+"fid", Pk_Path+"h_dw", Pk_Path+"h_dw2"};
    	CC_path = {"C_h_up", "C_h_up2", "C_fid", "C_h_dw", "C_h_dw2"};
	}
	else if(paramo == 6){
    	fold_path = {Pk_Path+"ns_up", Pk_Path+"ns_up2", Pk_Path+"fid", Pk_Path+"ns_dw", Pk_Path+"ns_dw2"};
    	CC_path = {"C_ns_up", "C_ns_up2", "C_fid", "C_ns_dw", "C_ns_dw2"}; 
	}
	else if(paramo == 7){
	    fold_path = {Pk_Path+"s8_up", Pk_Path+"s8_up2", Pk_Path+"fid", Pk_Path+"s8_dw", Pk_Path+"s8_dw2"};
	    CC_path = {"C_s8_up", "C_s8_up2", "C_fid", "C_s8_dw", "C_s8_dw2"};
	}
	else if(paramo == 8){
	    gamma[0] *= (1.+eps_gamma); gamma[1] *= (1.+2*eps_gamma); gamma[3] *= (1.-eps_gamma); gamma[4] *= (1.-2*eps_gamma);
	    CC_path = {"C_gamma_up", "C_gamma_up2", "C_fid", "C_gamma_dw", "C_gamma_dw2"};
	}
	else if(paramo == 9){
	    A_IA[0] *= (1.+eps_A_IA); A_IA[1] *= (1.+2*eps_A_IA); A_IA[3] *= (1.-eps_A_IA); A_IA[4]*=(1.-2*eps_A_IA);
	    CC_path = {"C_A_IA_up", "C_A_IA_up2", "C_fid", "C_A_IA_dw", "C_A_IA_dw2"};
	}
	else if(paramo == 10){
	    n_IA[0] *= (1.+eps_n_IA); n_IA[1] *= (1.+2*eps_n_IA), n_IA[3] *= (1.-eps_n_IA); n_IA[4] *= (1.-2*eps_n_IA);
	    CC_path = {"C_n_IA_up", "C_n_IA_up2", "C_fid", "C_n_IA_dw", "C_n_IA_dw2"};
	}
	else if(paramo == 11){
	    B_IA[0] *= (1.+eps_B_IA); B_IA[1] *= (1.+2*eps_B_IA); B_IA[3] *= (1.-eps_B_IA); B_IA[4] *= (1.-2*eps_B_IA);
	    CC_path = {"C_B_IA_up", "C_B_IA_up2", "C_fid", "C_B_IA_dw", "C_B_IA_dw2"};
	}
	else if(paramo > 11){
    	CC_path = {"C_b"+to_string(paramo-11)+"_up", "C_b"+to_string(paramo-11)+"_up2", "C_fid", "C_b"+to_string(paramo-11)+"_dw", "C_b"+to_string(paramo-11)+"_dw2"};
    	BX[paramo-12][0] *= (1.+eps_b); BX[paramo-12][1] *= (1.+2*eps_b); BX[paramo-12][3] *= (1.-eps_b); BX[paramo-12][4] *= (1.-2*eps_b);
    }

    Pk_files.clear();
    for(int i=0; i < der_B; i++){
    	Pk_files.push_back(fold_path[i]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat");
    }
}

//Declare and initialize all the power spectrums.
void XC::Initializers_Pk(){
	string lines_N; int count=0;
	P_dd_C.clear(), P_dd_C_up.clear(), P_dd_C_dw.clear();

    if(paramo == 0){
        vector<vector<double>> P_dd_CT;
        ifstream ifile(Pk_files[2]);
        while(!ifile.fail()){
            if(count>2){
                P_dd_CT.push_back(vector<double>());
                for(int j=0;j<2;j++){
                    P_dd_CT[count-3].push_back(0);
                    ifile>>P_dd_CT[count-3][j];
                    if(j==0){
                        P_dd_CT[count-3][j]*=h[2];
                    }
                    else if(j==1){
                        P_dd_CT[count-3][j]/=pow(h[2],3);
                    }
                }
            }
            getline(ifile,lines_N); count++;
        }
        ifile.close(); count=0;

        int len_k = (P_dd_CT.size()-1)/z_win.size();

        int xxx = 0;
        for(int i=0; i<z_win.size(); i++){
            P_dd_C.push_back(vector<vector<double>>());
            for(int j=0; j<len_k; j++){
                P_dd_C[i].push_back(vector<double>());
                for(int k=0; k<2; k++){
                     P_dd_C[i][j].push_back(0);
                     P_dd_C[i][j][k] = P_dd_CT[xxx][k];
                }
                xxx++;
            }
        }
    }

    vector<vector<double>> P_dd_CT_up;
    ifstream ifile_up(Pk_files[0]);
    while(!ifile_up.fail()){
        if(count>2){
            P_dd_CT_up.push_back(vector<double>());
            for(int j=0;j<2;j++){
                P_dd_CT_up[count-3].push_back(0);
                ifile_up>>P_dd_CT_up[count-3][j];
                if(j==0){
                    P_dd_CT_up[count-3][j]*=h[0];
                }
                else if(j==1){
                    P_dd_CT_up[count-3][j]/=pow(h[0],3);
                }
            }
        }
        getline(ifile_up,lines_N); count++;
    }
    ifile_up.close(); count=0;

    int len_k = (P_dd_CT_up.size()-1)/z_win.size();

    int xxx = 0;
    for(int i=0; i<z_win.size(); i++){
        P_dd_C_up.push_back(vector<vector<double>>());
        for(int j=0; j<len_k; j++){
            P_dd_C_up[i].push_back(vector<double>());
            for(int k=0; k<2; k++){
                 P_dd_C_up[i][j].push_back(0);
                 P_dd_C_up[i][j][k] = P_dd_CT_up[xxx][k];
            }
            xxx++;
        }
    }

    vector<vector<double>> P_dd_CT_dw;
    ifstream ifile_dw(Pk_files[3]);
    while(!ifile_dw.fail()){
        if(count>2){
            P_dd_CT_dw.push_back(vector<double>());
            for(int j=0;j<2;j++){
                P_dd_CT_dw[count-3].push_back(0);
                ifile_dw>>P_dd_CT_dw[count-3][j];
                if(j==0){
                    P_dd_CT_dw[count-3][j]*=h[3];
                }
                else if(j==1){
                    P_dd_CT_dw[count-3][j]/=pow(h[3],3);
                }
            }
        }
        getline(ifile_dw,lines_N); count++;
    }
    ifile_dw.close(); count=0;

    len_k = (P_dd_CT_dw.size()-1)/z_win.size();

    xxx = 0;
    for(int i=0; i<z_win.size(); i++){
        P_dd_C_dw.push_back(vector<vector<double>>());
        for(int j=0; j<len_k; j++){
            P_dd_C_dw[i].push_back(vector<double>());
            for(int k=0; k<2; k++){
                 P_dd_C_dw[i][j].push_back(0);
                 P_dd_C_dw[i][j][k] = P_dd_CT_dw[xxx][k];
            }
            xxx++;
        }
    }
    
	progress = barometer(progress, NP);
}

//Compute the background quantities.
void XC::background(){
	for(int i=0; i<z_win.size(); i++){
		if(paramo==0){
			E_tab[i] = E(z_win[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], curvature);
			R_tab[i] = chi_R(z_win[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], c, curvature);
			DG_tab[i] = DGrowth(z_win[i], h[2], Omega_m[2], Omega_DE[2], w0[2], wa[2], gamma[2], curvature);
			RT_tab[i] = R_tab[i]/(c/(100*h[2]));
		}
		E_tab_up[i] = E(z_win[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], curvature);
		E_tab_dw[i] = E(z_win[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], curvature);
		R_tab_up[i] = chi_R(z_win[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], c, curvature);
		R_tab_dw[i] = chi_R(z_win[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], c, curvature);
		RT_tab_up[i] = R_tab_up[i]/(c/(100*h[0]));
		RT_tab_dw[i] = R_tab_dw[i]/(c/(100*h[3]));
		DG_tab_up[i] = DGrowth(z_win[i], h[0], Omega_m[0], Omega_DE[0], w0[0], wa[0], gamma[0], curvature);
		DG_tab_dw[i] = DGrowth(z_win[i], h[3], Omega_m[3], Omega_DE[3], w0[3], wa[3], gamma[3], curvature);
    }
	progress = barometer(progress, NP);
}

//Compute the galaxy density distribution.
double XC::nz(double z){
    return pow(z/zrange_0,2)*exp(-pow(z/zrange_0,alpha));
}

//Photo-z pdf.
double XC::P_phot(double z, double zp){
    return (1. -  f_out)/(pow(2.*3.1415926535,0.5)*sig_b*(1+z))*exp(-0.5*pow((z - cb*zp - zb)/(sig_b*(1.+z)),2)) + f_out/(pow(2.*3.1415926535,0.5)*sig_0*(1+z))*exp(-0.5*pow((z - c0*zp - z0)/(sig_0*(1.+z)),2));
}

//Compute the photo-z.
void XC::photoz(){
    cout<<"Computing the Photo-z (1st iteration only)..."<<endl;
	ofstream myfile;
	double z_p, z_m; vector<vector<double>> zp; int I_prec = 500;
	double Trap; double Trap_Z;
	vector<double> delta_zp; vector<vector<double>> TZ; vector<double> TTZ;
	myfile.open("Pz_Table_10000.txt");
	for(int i=0; i<z_win.size(); i++){
		cout<<i+1<<"/"<<z_win.size()<<endl;
		for(int j=0; j<zrange.size(); j++){
			Trap = 0.; Trap_Z = 0.; TTZ.clear(); TZ.clear();
			z_p = z_ip[j]; z_m = z_im[j]; 
			if(i==0){
				zp.push_back(vector<double>());
				for(int k=0; k<I_prec; k++){
					zp[j].push_back(z_m + (z_p-z_m)*k/(I_prec-1));
				}
				delta_zp.push_back(zp[j][1]-zp[j][0]);
			}

			for(int k=1; k<I_prec; k++){
				Trap += (7.*(P_phot(z_win[i], zp[j][k])+(P_phot(z_win[i], zp[j][k-1]))) + 32.*(P_phot(z_win[i], 1./4*(3.*zp[j][k]+zp[j][k-1])) + P_phot(z_win[i], 1./4*(zp[j][k]+3.*zp[j][k-1]))) + 12*(P_phot(z_win[i], 1./2*(zp[j][k]+zp[j][k-1]))));
			}
			Trap *= delta_zp[j]/90*nz(z_win[i]);


			for(int m=0; m<MMM; m++){
				TZ.push_back(vector<double>());
				for(int k=0; k<I_prec; k++){
					TZ[m].push_back(P_phot(z500[m], zp[j][k])*nz(z500[m]));
				}
			}

			for(int k=0; k<I_prec; k++){
				TTZ.push_back(0);
				for(int m=1; m<MMM; m++){
					TTZ[k] += (7*(TZ[m][k] + TZ[m-1][k]) + 32*((TZ[m-1][k] + (1./4*(3*z500[m]+z500[m-1]) - z500[m-1])*(TZ[m][k]-TZ[m-1][k])/(z500[m]-z500[m-1])) + (TZ[m-1][k] + (1./4*(z500[m]+3*z500[m-1]) - z500[m-1])*(TZ[m][k]-TZ[m-1][k])/(z500[m]-z500[m-1]))) + 12*(TZ[m-1][k] + (1./2*(z500[m]+z500[m-1]) - z500[m-1])*(TZ[m][k]-TZ[m-1][k])/(z500[m]-z500[m-1])) );
				}
				TTZ[k] *= delta_zp[j]/90;
			}

			for(int k=1; k<I_prec; k++){
				Trap_Z += (7*(TTZ[k] + TTZ[k-1]) + 32*( ((TTZ[k-1] + (1./4*(3*zp[j][k]+zp[j][k-1]) - zp[j][k-1])*(TTZ[k]-TTZ[k-1])/(zp[j][k]-zp[j][k-1]))) + ((TTZ[k-1] + (1./4*(zp[j][k]+3*zp[j][k-1]) - zp[j][k-1])*(TTZ[k]-TTZ[k-1])/(zp[j][k]-zp[j][k-1])))) + 12*(TTZ[k-1] + (1./2*(zp[j][k]+zp[j][k-1]) - zp[j][k-1])*(TTZ[k]-TTZ[k-1])/(zp[j][k]-zp[j][k-1])));
			}

			Trap_Z *= delta_z500/90;
			Photoz_tab[i][j] = Trap/Trap_Z;
			myfile<<setprecision(12)<<scientific<<Photoz_tab[i][j]<<" ";
		}
		myfile<<endl;
	}
	myfile.close();
}

//Load the photo-z.
void XC::photoz_load(){
	ifstream ifile("Pz_Table_10000.txt");
	while (!ifile.eof()){
		for(int i=0;i<z_win.size();i++){
			for(int j=0;j<zrange.size();j++){
				ifile>>Photoz_tab[i][j];
			}
		}
	}
	ifile.close();
	progress = barometer(progress, NP);
}

//Compute the windows functions.
void XC::windows(){
	int i_s = 0; double delta_zwin = z_win[4]-z_win[0];
	for(int i=0; i<z_win.size(); i++){
		for(int k=0; k<z_ip.size(); k++){
			if (z_win[i] <= z_ip[k] && z_win[i] >= z_im[k]){
				i_s = k;
			}
		}
		for(int j=0; j<zrange.size(); j++){
			if(paramo==0){
				WG_tab[i][j] = 100.*h[2]/c*Photoz_tab[i][j]*E_tab[i]*BX[i_s][2];
				WIA_tab[i][j] = 100.*h[2]/c*Photoz_tab[i][j]*E_tab[i];
			}

			WIA_tab_up[i][j] = 100.*h[0]/c*Photoz_tab[i][j]*E_tab_up[i];
			WIA_tab_dw[i][j] = 100.*h[3]/c*Photoz_tab[i][j]*E_tab_dw[i];
			WG_tab_up[i][j] = 100.*h[0]/c*Photoz_tab[i][j]*E_tab_up[i]*BX[i_s][0];
			WG_tab_dw[i][j] = 100.*h[3]/c*Photoz_tab[i][j]*E_tab_dw[i]*BX[i_s][3];

			for(int k=i+1; k<z_win.size()-4; k+=4){
				if(paramo==0){
					W_tab[i][j] += 7.*(Photoz_tab[k][j]*(1. - RT_tab[i]/RT_tab[k]) + Photoz_tab[k+4][j]*(1. - RT_tab[i]/RT_tab[k+4])) + 32.*(Photoz_tab[k+3][j]*(1. - RT_tab[i]/RT_tab[k+3]) + Photoz_tab[k+1][j]*(1. - RT_tab[i]/RT_tab[k+1])) + 12.*(Photoz_tab[k+2][j]*(1. - RT_tab[i]/RT_tab[k+2]));
				}
				W_tab_up[i][j] += 7.*(Photoz_tab[k][j]*(1. - RT_tab_up[i]/RT_tab_up[k]) + Photoz_tab[k+4][j]*(1. - RT_tab_up[i]/RT_tab_up[k+4])) + 32.*(Photoz_tab[k+3][j]*(1. - RT_tab_up[i]/RT_tab_up[k+3]) + Photoz_tab[k+1][j]*(1. - RT_tab_up[i]/RT_tab_up[k+1])) + 12.*(Photoz_tab[k+2][j]*(1. - RT_tab_up[i]/RT_tab_up[k+2]));
				W_tab_dw[i][j] += 7.*(Photoz_tab[k][j]*(1. - RT_tab_dw[i]/RT_tab_dw[k]) + Photoz_tab[k+4][j]*(1. - RT_tab_dw[i]/RT_tab_dw[k+4])) + 32.*(Photoz_tab[k+3][j]*(1. - RT_tab_dw[i]/RT_tab_dw[k+3]) + Photoz_tab[k+1][j]*(1. - RT_tab_dw[i]/RT_tab_dw[k+1])) + 12.*(Photoz_tab[k+2][j]*(1. - RT_tab_dw[i]/RT_tab_dw[k+2]));
			}

			if(paramo==0){
				W_tab[i][j] *= 1.5*100.*h[2]*Omega_m[2]*(1.+z_win[i])/c*RT_tab[i]/90*delta_zwin;
			}
			W_tab_up[i][j] *= 1.5*100.*h[0]*Omega_m[0]*(1.+z_win[i])/c*RT_tab_up[i]/90*delta_zwin;
			W_tab_dw[i][j] *= 1.5*100.*h[3]*Omega_m[3]*(1.+z_win[i])/c*RT_tab_dw[i]/90*delta_zwin;
        }
	}
	progress = barometer(progress, NP);
}

//Weak lensing shot noise.
double XC::P_shot_WL(double zpi, double zpj){
    if(zpi == zpj){
        return pow(sig_epsilon,2)/(ng/zrange.size());
    }
    else{
        return 0;
    }
}

//Photometric galaxy clustering shot noise.
double XC::P_shot_GC(double zpi, double zpj){
    if(zpi == zpj){
        return 1./(ng/zrange.size());
    }
    else{
        return 0;
    }
}

//Power spectrums interpolations and Cl computing.
void XC::C_l_computing(){
	vector<double> Pk_conv, Pk_conv_up, Pk_conv_dw; double k_conv;
	delta_zpm = z_win[4]-z_win[0];

	for(int i=0; i<l.size(); i++){

        ofstream outGG;
        ofstream outLL;
        ofstream outGL;
        if(paramo == 0){
            outGG.open(pre_CC_path[0]+CC_path[2]+"/COVAR_fid_"+to_string(l[i]));
            outLL.open(pre_CC_path[1]+CC_path[2]+"/COVAR_fid_"+to_string(l[i]));
            outGL.open(pre_CC_path[2]+CC_path[2]+"/COVAR_fid_"+to_string(l[i]));
        }

        ofstream outGGU; outGGU.open(pre_CC_path[0]+CC_path[0]+"/COVAR_up_"+to_string(l[i]));
        ofstream outGGD; outGGD.open(pre_CC_path[0]+CC_path[3]+"/COVAR_dw_"+to_string(l[i]));
        ofstream outLLU; outLLU.open(pre_CC_path[1]+CC_path[0]+"/COVAR_up_"+to_string(l[i]));
        ofstream outLLD; outLLD.open(pre_CC_path[1]+CC_path[3]+"/COVAR_dw_"+to_string(l[i]));
        ofstream outGLU; outGLU.open(pre_CC_path[2]+CC_path[0]+"/COVAR_up_"+to_string(l[i]));
        ofstream outGLD; outGLD.open(pre_CC_path[2]+CC_path[3]+"/COVAR_dw_"+to_string(l[i]));

		for(int m=0; m < z_win.size(); m++){
			k_conv = (l[i]+0.5)/R_tab_dw[m];
			if(k_conv < 35 && k_conv >  0.0005){
				for(int n=0; n < P_dd_C_dw[m].size()-1; n++){
					if(k_conv > P_dd_C_dw[m][n][0] && k_conv <= P_dd_C_dw[m][n+1][0]){
						if(paramo==8){
							Pk_conv_dw.push_back((P_dd_C_dw[m][n][1] + (P_dd_C_dw[m][n+1][1]-P_dd_C_dw[m][n][1])*(k_conv-P_dd_C_dw[m][n][0])/(P_dd_C_dw[m][n+1][0]-P_dd_C_dw[m][n][0]))*pow(DG_tab_dw[m]/DG_tab[m],2));
						}
						else{
							Pk_conv_dw.push_back(P_dd_C_dw[m][n][1] + (P_dd_C_dw[m][n+1][1]-P_dd_C_dw[m][n][1])*(k_conv-P_dd_C_dw[m][n][0])/(P_dd_C_dw[m][n+1][0]-P_dd_C_dw[m][n][0]));
						}
					}
				}
			}
			else{
				Pk_conv_dw.push_back(0);
			}
			if(paramo == 0){
				k_conv = (l[i]+0.5)/R_tab[m];
				if(k_conv < 35 && k_conv > 0.0005){
					for(int n=0; n < P_dd_C[m].size()-1; n++){
						if(k_conv > P_dd_C[m][n][0] && k_conv <= P_dd_C[m][n+1][0]){
							Pk_conv.push_back(P_dd_C[m][n][1] + (P_dd_C[m][n+1][1]-P_dd_C[m][n][1])*(k_conv-P_dd_C[m][n][0])/(P_dd_C[m][n+1][0]-P_dd_C[m][n][0]));
						}
					}
				}
				else{
					Pk_conv.push_back(0);
				}
			}
			k_conv = (l[i]+0.5)/R_tab_up[m];
			if(k_conv < 35 && k_conv >  0.0005){
				for(int n=0; n < P_dd_C_up[m].size()-1; n++){
					if(k_conv > P_dd_C_up[m][n][0] && k_conv <= P_dd_C_up[m][n+1][0]){
						if(paramo==8){
							Pk_conv_up.push_back((P_dd_C_up[m][n][1] + (P_dd_C_up[m][n+1][1]-P_dd_C_up[m][n][1])*(k_conv-P_dd_C_up[m][n][0])/(P_dd_C_up[m][n+1][0]-P_dd_C_up[m][n][0]))*pow(DG_tab_up[m]/DG_tab[m],2));
						}
						else{
							Pk_conv_up.push_back(P_dd_C_up[m][n][1] + (P_dd_C_up[m][n+1][1]-P_dd_C_up[m][n][1])*(k_conv-P_dd_C_up[m][n][0])/(P_dd_C_up[m][n+1][0]-P_dd_C_up[m][n][0]));
						}
					}
				}
			}
			else{
				Pk_conv_up.push_back(0);
			}
		}

		for(int aa=0; aa<zrange.size(); aa++){
			for(int bb=aa; bb<zrange.size(); bb++){
				if(paramo != 9 || paramo != 10 || paramo != 11){
					if(paramo == 0){
						for(int k=0; k<z_win.size()-4; k+=4){ 
							C_ij_GG[aa][bb] += c/(100.*h[2])*delta_zpm/90*(7.*(WG_tab[k][aa]*WG_tab[k][bb]/(E_tab[k]*pow(R_tab[k],2))*Pk_conv[k] + WG_tab[k+4][aa]*WG_tab[k+4][bb]/(E_tab[k+4]*pow(R_tab[k+4],2))*Pk_conv[k+4]) + 32.*(WG_tab[k+1][aa]*WG_tab[k+1][bb]/(E_tab[k+1]*pow(R_tab[k+1],2))*Pk_conv[k+1] + WG_tab[k+3][aa]*WG_tab[k+3][bb]/(E_tab[k+3]*pow(R_tab[k+3],2))*Pk_conv[k+3]) + 12.*(WG_tab[k+2][aa]*WG_tab[k+2][bb]/(E_tab[k+2]*pow(R_tab[k+2],2))*Pk_conv[k+2]));
						}
						C_ij_GG[aa][bb] += P_shot_GC(zrange[aa], zrange[bb]);
						C_ij_GG[bb][aa] = C_ij_GG[aa][bb];
					}
					for(int k=0; k<z_win.size()-4; k+=4){ 
						C_ij_GG_up[aa][bb] += c/(100.*h[0])*delta_zpm/90*(7.*(WG_tab_up[k][aa]*WG_tab_up[k][bb]/(E_tab_up[k]*pow(R_tab_up[k],2))*Pk_conv_up[k] + WG_tab_up[k+4][aa]*WG_tab_up[k+4][bb]/(E_tab_up[k+4]*pow(R_tab_up[k+4],2))*Pk_conv_up[k+4]) + 32.*(WG_tab_up[k+1][aa]*WG_tab_up[k+1][bb]/(E_tab_up[k+1]*pow(R_tab_up[k+1],2))*Pk_conv_up[k+1] + WG_tab_up[k+3][aa]*WG_tab_up[k+3][bb]/(E_tab_up[k+3]*pow(R_tab_up[k+3],2))*Pk_conv_up[k+3]) + 12.*(WG_tab_up[k+2][aa]*WG_tab_up[k+2][bb]/(E_tab_up[k+2]*pow(R_tab_up[k+2],2))*Pk_conv_up[k+2]));
						C_ij_GG_dw[aa][bb] += c/(100.*h[3])*delta_zpm/90*(7.*(WG_tab_dw[k][aa]*WG_tab_dw[k][bb]/(E_tab_dw[k]*pow(R_tab_dw[k],2))*Pk_conv_dw[k] + WG_tab_dw[k+4][aa]*WG_tab_dw[k+4][bb]/(E_tab_dw[k+4]*pow(R_tab_dw[k+4],2))*Pk_conv_dw[k+4]) + 32.*(WG_tab_dw[k+1][aa]*WG_tab_dw[k+1][bb]/(E_tab_dw[k+1]*pow(R_tab_dw[k+1],2))*Pk_conv_dw[k+1] + WG_tab_dw[k+3][aa]*WG_tab_dw[k+3][bb]/(E_tab_dw[k+3]*pow(R_tab_dw[k+3],2))*Pk_conv_dw[k+3]) + 12.*(WG_tab_dw[k+2][aa]*WG_tab_dw[k+2][bb]/(E_tab_dw[k+2]*pow(R_tab_dw[k+2],2))*Pk_conv_dw[k+2]));
					}
					C_ij_GG_up[aa][bb] += P_shot_GC(zrange[aa], zrange[bb]);
					C_ij_GG_dw[aa][bb] += P_shot_GC(zrange[aa], zrange[bb]);
					if (aa != bb){
						C_ij_GG_up[bb][aa] = C_ij_GG_up[aa][bb];
						C_ij_GG_dw[bb][aa] = C_ij_GG_dw[aa][bb];
					}
				}
				else{
					C_ij_GG[aa][bb] = 0; C_ij_GG_up[aa][bb] = 0; C_ij_GG_dw[aa][bb] = 0;
					C_ij_GG[bb][aa] = 0; C_ij_GG_up[bb][aa] = 0; C_ij_GG_dw[bb][aa] = 0;
				}
				if(paramo < 12){
					if(paramo == 0){
						for(int k=0; k<z_win.size()-4; k+=4){ 
							C_ij_LL[aa][bb] += c/(100.*h[2])*delta_zpm/90*(7.*(W_tab[k][aa]*W_tab[k][bb]/(E_tab[k]*pow(R_tab[k],2))*Pk_conv[k] + W_tab[k+4][aa]*W_tab[k+4][bb]/(E_tab[k+4]*pow(R_tab[k+4],2))*Pk_conv[k+4]) + 32.*(W_tab[k+1][aa]*W_tab[k+1][bb]/(E_tab[k+1]*pow(R_tab[k+1],2))*Pk_conv[k+1] + W_tab[k+3][aa]*W_tab[k+3][bb]/(E_tab[k+3]*pow(R_tab[k+3],2))*Pk_conv[k+3]) + 12.*(W_tab[k+2][aa]*W_tab[k+2][bb]/(E_tab[k+2]*pow(R_tab[k+2],2))*Pk_conv[k+2])
                            + 7.*((W_tab[k][aa]*WIA_tab[k][bb] + WIA_tab[k][aa]*W_tab[k][bb])/(E_tab[k]*pow(R_tab[k],2))*(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k]*pow(1.+z_win[k],n_IA[2])*pow(lum_mean[k],B_IA[2])*Pk_conv[k])
							+ (W_tab[k+4][aa]*WIA_tab[k+4][bb] + WIA_tab[k+4][aa]*W_tab[k+4][bb])/(E_tab[k+4]*pow(R_tab[k+4],2))*(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+4]*pow(1.+z_win[k+4],n_IA[2])*pow(lum_mean[k+4],B_IA[2])*Pk_conv[k+4]))
							+ 32.*((W_tab[k+1][aa]*WIA_tab[k+1][bb] + WIA_tab[k+1][aa]*W_tab[k+1][bb])/(E_tab[k+1]*pow(R_tab[k+1],2))*(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+1]*pow(1.+z_win[k+1],n_IA[2])*pow(lum_mean[k+1],B_IA[2])*Pk_conv[k+1])
							+ (W_tab[k+3][aa]*WIA_tab[k+3][bb] + WIA_tab[k+3][aa]*W_tab[k+3][bb])/(E_tab[k+3]*pow(R_tab[k+3],2))*(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+3]*pow(1.+z_win[k+3],n_IA[2])*pow(lum_mean[k+3],B_IA[2])*Pk_conv[k+3]))
							+ 12.*((W_tab[k+2][aa]*WIA_tab[k+2][bb] + WIA_tab[k+2][aa]*W_tab[k+2][bb])/(E_tab[k+2]*pow(R_tab[k+2],2))*(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+2]*pow(1.+z_win[k+2],n_IA[2])*pow(lum_mean[k+2],B_IA[2])*Pk_conv[k+2]))
							+ 7.*(WIA_tab[k][aa]*WIA_tab[k][bb]/(E_tab[k]*pow(R_tab[k],2))*pow(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k]*pow(1.+z_win[k],n_IA[2])*pow(lum_mean[k],B_IA[2]),2)*Pk_conv[k]
							+ WIA_tab[k+4][aa]*WIA_tab[k+4][bb]/(E_tab[k+4]*pow(R_tab[k+4],2))*pow(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+4]*pow(1.+z_win[k+4],n_IA[2])*pow(lum_mean[k+4],B_IA[2]),2)*Pk_conv[k+4])
							+ 32.*(WIA_tab[k+1][aa]*WIA_tab[k+1][bb]/(E_tab[k+1]*pow(R_tab[k+1],2))*pow(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+1]*pow(1.+z_win[k+1],n_IA[2])*pow(lum_mean[k+1],B_IA[2]),2)*Pk_conv[k+1]
							+ WIA_tab[k+3][aa]*WIA_tab[k+3][bb]/(E_tab[k+3]*pow(R_tab[k+3],2))*pow(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+3]*pow(1.+z_win[k+3],n_IA[2])*pow(lum_mean[k+3],B_IA[2]),2)*Pk_conv[k+3])
							+ 12.*(WIA_tab[k+2][aa]*WIA_tab[k+2][bb]/(E_tab[k+2]*pow(R_tab[k+2],2))*pow(-A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+2]*pow(1.+z_win[k+2],n_IA[2])*pow(lum_mean[k+2],B_IA[2]),2)*Pk_conv[k+2]));
						}
						C_ij_LL[aa][bb] += P_shot_WL(zrange[aa], zrange[bb]);
						C_ij_LL[bb][aa] = C_ij_LL[aa][bb];
					}
					for(int k=0; k<z_win.size()-4; k+=4){
						C_ij_LL_up[aa][bb] += c/(100.*h[0])*delta_zpm/90*(7.*(W_tab_up[k][aa]*W_tab_up[k][bb]/(E_tab_up[k]*pow(R_tab_up[k],2))*Pk_conv_up[k] + W_tab_up[k+4][aa]*W_tab_up[k+4][bb]/(E_tab_up[k+4]*pow(R_tab_up[k+4],2))*Pk_conv_up[k+4]) + 32.*(W_tab_up[k+1][aa]*W_tab_up[k+1][bb]/(E_tab_up[k+1]*pow(R_tab_up[k+1],2))*Pk_conv_up[k+1] + W_tab_up[k+3][aa]*W_tab_up[k+3][bb]/(E_tab_up[k+3]*pow(R_tab_up[k+3],2))*Pk_conv_up[k+3]) + 12.*(W_tab_up[k+2][aa]*W_tab_up[k+2][bb]/(E_tab_up[k+2]*pow(R_tab_up[k+2],2))*Pk_conv_up[k+2])
							+ 7.*((W_tab_up[k][aa]*WIA_tab_up[k][bb] + WIA_tab_up[k][aa]*W_tab_up[k][bb])/(E_tab_up[k]*pow(R_tab_up[k],2))*(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k]*pow(1.+z_win[k],n_IA[0])*pow(lum_mean[k],B_IA[0])*Pk_conv_up[k])
							+ (W_tab_up[k+4][aa]*WIA_tab_up[k+4][bb] + WIA_tab_up[k+4][aa]*W_tab_up[k+4][bb])/(E_tab_up[k+4]*pow(R_tab_up[k+4],2))*(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+4]*pow(1.+z_win[k+4],n_IA[0])*pow(lum_mean[k+4],B_IA[0])*Pk_conv_up[k+4]))
							+ 32.*((W_tab_up[k+1][aa]*WIA_tab_up[k+1][bb] + WIA_tab_up[k+1][aa]*W_tab_up[k+1][bb])/(E_tab_up[k+1]*pow(R_tab_up[k+1],2))*(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+1]*pow(1.+z_win[k+1],n_IA[0])*pow(lum_mean[k+1],B_IA[0])*Pk_conv_up[k+1])
							+ (W_tab_up[k+3][aa]*WIA_tab_up[k+3][bb] + WIA_tab_up[k+3][aa]*W_tab_up[k+3][bb])/(E_tab_up[k+3]*pow(R_tab_up[k+3],2))*(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+3]*pow(1.+z_win[k+3],n_IA[0])*pow(lum_mean[k+3],B_IA[0])*Pk_conv_up[k+3]))
							+ 12.*((W_tab_up[k+2][aa]*WIA_tab_up[k+2][bb] + WIA_tab_up[k+2][aa]*W_tab_up[k+2][bb])/(E_tab_up[k+2]*pow(R_tab_up[k+2],2))*(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+2]*pow(1.+z_win[k+2],n_IA[0])*pow(lum_mean[k+2],B_IA[0])*Pk_conv_up[k+2]))
							+ 7.*(WIA_tab_up[k][aa]*WIA_tab_up[k][bb]/(E_tab_up[k]*pow(R_tab_up[k],2))*pow(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k]*pow(1.+z_win[k],n_IA[0])*pow(lum_mean[k],B_IA[0]),2)*Pk_conv_up[k]
							+ WIA_tab_up[k+4][aa]*WIA_tab_up[k+4][bb]/(E_tab_up[k+4]*pow(R_tab_up[k+4],2))*pow(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+4]*pow(1.+z_win[k+4],n_IA[0])*pow(lum_mean[k+4],B_IA[0]),2)*Pk_conv_up[k+4])
							+ 32.*(WIA_tab_up[k+1][aa]*WIA_tab_up[k+1][bb]/(E_tab_up[k+1]*pow(R_tab_up[k+1],2))*pow(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+1]*pow(1.+z_win[k+1],n_IA[0])*pow(lum_mean[k+1],B_IA[0]),2)*Pk_conv_up[k+1]
							+ WIA_tab_up[k+3][aa]*WIA_tab_up[k+3][bb]/(E_tab_up[k+3]*pow(R_tab_up[k+3],2))*pow(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+3]*pow(1.+z_win[k+3],n_IA[0])*pow(lum_mean[k+3],B_IA[0]),2)*Pk_conv_up[k+3])
							+ 12.*(WIA_tab_up[k+2][aa]*WIA_tab_up[k+2][bb]/(E_tab_up[k+2]*pow(R_tab_up[k+2],2))*pow(-A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+2]*pow(1.+z_win[k+2],n_IA[0])*pow(lum_mean[k+2],B_IA[0]),2)*Pk_conv_up[k+2]));

						C_ij_LL_dw[aa][bb] += c/(100.*h[3])*delta_zpm/90*(7.*(W_tab_dw[k][aa]*W_tab_dw[k][bb]/(E_tab_dw[k]*pow(R_tab_dw[k],2))*Pk_conv_dw[k] + W_tab_dw[k+4][aa]*W_tab_dw[k+4][bb]/(E_tab_dw[k+4]*pow(R_tab_dw[k+4],2))*Pk_conv_dw[k+4]) + 32.*(W_tab_dw[k+1][aa]*W_tab_dw[k+1][bb]/(E_tab_dw[k+1]*pow(R_tab_dw[k+1],2))*Pk_conv_dw[k+1] + W_tab_dw[k+3][aa]*W_tab_dw[k+3][bb]/(E_tab_dw[k+3]*pow(R_tab_dw[k+3],2))*Pk_conv_dw[k+3]) + 12.*(W_tab_dw[k+2][aa]*W_tab_dw[k+2][bb]/(E_tab_dw[k+2]*pow(R_tab_dw[k+2],2))*Pk_conv_dw[k+2])
							+ 7.*((W_tab_dw[k][aa]*WIA_tab_dw[k][bb] + WIA_tab_dw[k][aa]*W_tab_dw[k][bb])/(E_tab_dw[k]*pow(R_tab_dw[k],2))*(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k]*pow(1.+z_win[k],n_IA[3])*pow(lum_mean[k],B_IA[3])*Pk_conv_dw[k])
							+ (W_tab_dw[k+4][aa]*WIA_tab_dw[k+4][bb] + WIA_tab_dw[k+4][aa]*W_tab_dw[k+4][bb])/(E_tab_dw[k+4]*pow(R_tab_dw[k+4],2))*(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+4]*pow(1.+z_win[k+4],n_IA[3])*pow(lum_mean[k+4],B_IA[3])*Pk_conv_dw[k+4]))
							+ 32.*((W_tab_dw[k+1][aa]*WIA_tab_dw[k+1][bb] + WIA_tab_dw[k+1][aa]*W_tab_dw[k+1][bb])/(E_tab_dw[k+1]*pow(R_tab_dw[k+1],2))*(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+1]*pow(1.+z_win[k+1],n_IA[3])*pow(lum_mean[k+1],B_IA[3])*Pk_conv_dw[k+1])
							+ (W_tab_dw[k+3][aa]*WIA_tab_dw[k+3][bb] + WIA_tab_dw[k+3][aa]*W_tab_dw[k+3][bb])/(E_tab_dw[k+3]*pow(R_tab_dw[k+3],2))*(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+3]*pow(1.+z_win[k+3],n_IA[3])*pow(lum_mean[k+3],B_IA[3])*Pk_conv_dw[k+3]))
							+ 12.*((W_tab_dw[k+2][aa]*WIA_tab_dw[k+2][bb] + WIA_tab_dw[k+2][aa]*W_tab_dw[k+2][bb])/(E_tab_dw[k+2]*pow(R_tab_dw[k+2],2))*(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+2]*pow(1.+z_win[k+2],n_IA[3])*pow(lum_mean[k+2],B_IA[3])*Pk_conv_dw[k+2]))
							+ 7.*(WIA_tab_dw[k][aa]*WIA_tab_dw[k][bb]/(E_tab_dw[k]*pow(R_tab_dw[k],2))*pow(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k]*pow(1.+z_win[k],n_IA[3])*pow(lum_mean[k],B_IA[3]),2)*Pk_conv_dw[k]
							+ WIA_tab_dw[k+4][aa]*WIA_tab_dw[k+4][bb]/(E_tab_dw[k+4]*pow(R_tab_dw[k+4],2))*pow(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+4]*pow(1.+z_win[k+4],n_IA[3])*pow(lum_mean[k+4],B_IA[3]),2)*Pk_conv_dw[k+4])
							+ 32.*(WIA_tab_dw[k+1][aa]*WIA_tab_dw[k+1][bb]/(E_tab_dw[k+1]*pow(R_tab_dw[k+1],2))*pow(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+1]*pow(1.+z_win[k+1],n_IA[3])*pow(lum_mean[k+1],B_IA[3]),2)*Pk_conv_dw[k+1]
							+ WIA_tab_dw[k+3][aa]*WIA_tab_dw[k+3][bb]/(E_tab_dw[k+3]*pow(R_tab_dw[k+3],2))*pow(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+3]*pow(1.+z_win[k+3],n_IA[3])*pow(lum_mean[k+3],B_IA[3]),2)*Pk_conv_dw[k+3])
							+ 12.*(WIA_tab_dw[k+2][aa]*WIA_tab_dw[k+2][bb]/(E_tab_dw[k+2]*pow(R_tab_dw[k+2],2))*pow(-A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+2]*pow(1.+z_win[k+2],n_IA[3])*pow(lum_mean[k+2],B_IA[3]),2)*Pk_conv_dw[k+2]));
					}
					C_ij_LL_up[aa][bb] += P_shot_WL(zrange[aa], zrange[bb]);
					C_ij_LL_up[bb][aa] = C_ij_LL_up[aa][bb];
					C_ij_LL_dw[aa][bb] += P_shot_WL(zrange[aa], zrange[bb]);
					C_ij_LL_dw[bb][aa] = C_ij_LL_dw[aa][bb];
				}	
				else{
					C_ij_LL[aa][bb] = 0; C_ij_LL_up[aa][bb] = 0; C_ij_LL_dw[aa][bb] = 0;
					C_ij_LL[bb][aa] = 0; C_ij_LL_up[bb][aa] = 0; C_ij_LL_dw[bb][aa] = 0;
				}
			}
		}

		#pragma omp parallel for schedule(dynamic,8)
		for(int aa=0; aa<zrange.size(); aa++){
			for(int bb=0; bb<zrange.size(); bb++){
				for(int k=0; k<z_win.size()-4; k+=4){ 
					if(paramo == 0){
						C_ij_GL[aa][bb] += c/(100.*h[2])*delta_zpm/90*(7.*(WG_tab[k][aa]*(W_tab[k][bb]-WIA_tab[k][bb]*A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k]*pow(1.+z_win[k],n_IA[2])*pow(lum_mean[k],B_IA[2]))*Pk_conv[k]/(E_tab[k]*pow(R_tab[k],2))
					 	+ WG_tab[k+4][aa]*(W_tab[k+4][bb]-WIA_tab[k+4][bb]*A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+4]*pow(1.+z_win[k+4],n_IA[2])*pow(lum_mean[k+4],B_IA[2]))*Pk_conv[k+4]/(E_tab[k+4]*pow(R_tab[k+4],2))) 
					 	+ 32.*(WG_tab[k+1][aa]*(W_tab[k+1][bb]-WIA_tab[k+1][bb]*A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+1]*pow(1.+z_win[k+1],n_IA[2])*pow(lum_mean[k+1],B_IA[2]))*Pk_conv[k+1]/(E_tab[k+1]*pow(R_tab[k+1],2)) 
					 	+ WG_tab[k+3][aa]*(W_tab[k+3][bb]-WIA_tab[k+3][bb]*A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+3]*pow(1.+z_win[k+3],n_IA[2])*pow(lum_mean[k+3],B_IA[2]))*Pk_conv[k+3]/(E_tab[k+3]*pow(R_tab[k+3],2)))
					 	+ 12.*WG_tab[k+2][aa]*(W_tab[k+2][bb]-WIA_tab[k+2][bb]*A_IA[2]*C_IA[2]*Omega_m[2]/DG_tab[k+2]*pow(1.+z_win[k+2],n_IA[2])*pow(lum_mean[k+2],B_IA[2]))*Pk_conv[k+2]/(E_tab[k+2]*pow(R_tab[k+2],2)));
					 }

					C_ij_GL_up[aa][bb] += c/(100.*h[0])*delta_zpm/90*(7.*(WG_tab_up[k][aa]*(W_tab_up[k][bb]-WIA_tab_up[k][bb]*A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k]*pow(1.+z_win[k],n_IA[0])*pow(lum_mean[k],B_IA[0]))*Pk_conv_up[k]/(E_tab_up[k]*pow(R_tab_up[k],2))
					 + WG_tab_up[k+4][aa]*(W_tab_up[k+4][bb]-WIA_tab_up[k+4][bb]*A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+4]*pow(1.+z_win[k+4],n_IA[0])*pow(lum_mean[k+4],B_IA[0]))*Pk_conv_up[k+4]/(E_tab_up[k+4]*pow(R_tab_up[k+4],2))) 
					 + 32.*(WG_tab_up[k+1][aa]*(W_tab_up[k+1][bb]-WIA_tab_up[k+1][bb]*A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+1]*pow(1.+z_win[k+1],n_IA[0])*pow(lum_mean[k+1],B_IA[0]))*Pk_conv_up[k+1]/(E_tab_up[k+1]*pow(R_tab_up[k+1],2)) 
					 + WG_tab_up[k+3][aa]*(W_tab_up[k+3][bb]-WIA_tab_up[k+3][bb]*A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+3]*pow(1.+z_win[k+3],n_IA[0])*pow(lum_mean[k+3],B_IA[0]))*Pk_conv_up[k+3]/(E_tab_up[k+3]*pow(R_tab_up[k+3],2)))
					 + 12.*WG_tab_up[k+2][aa]*(W_tab_up[k+2][bb]-WIA_tab_up[k+2][bb]*A_IA[0]*C_IA[0]*Omega_m[0]/DG_tab_up[k+2]*pow(1.+z_win[k+2],n_IA[0])*pow(lum_mean[k+2],B_IA[0]))*Pk_conv_up[k+2]/(E_tab_up[k+2]*pow(R_tab_up[k+2],2)));
					
					C_ij_GL_dw[aa][bb] += c/(100.*h[3])*delta_zpm/90*(7.*(WG_tab_dw[k][aa]*(W_tab_dw[k][bb]-WIA_tab_dw[k][bb]*A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k]*pow(1.+z_win[k],n_IA[3])*pow(lum_mean[k],B_IA[3]))*Pk_conv_dw[k]/(E_tab_dw[k]*pow(R_tab_dw[k],2))
					 + WG_tab_dw[k+4][aa]*(W_tab_dw[k+4][bb]-WIA_tab_dw[k+4][bb]*A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+4]*pow(1.+z_win[k+4],n_IA[3])*pow(lum_mean[k+4],B_IA[3]))*Pk_conv_dw[k+4]/(E_tab_dw[k+4]*pow(R_tab_dw[k+4],2))) 
					 + 32.*(WG_tab_dw[k+1][aa]*(W_tab_dw[k+1][bb]-WIA_tab_dw[k+1][bb]*A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+1]*pow(1.+z_win[k+1],n_IA[3])*pow(lum_mean[k+1],B_IA[3]))*Pk_conv_dw[k+1]/(E_tab_dw[k+1]*pow(R_tab_dw[k+1],2)) 
					 + WG_tab_dw[k+3][aa]*(W_tab_dw[k+3][bb]-WIA_tab_dw[k+3][bb]*A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+3]*pow(1.+z_win[k+3],n_IA[3])*pow(lum_mean[k+3],B_IA[3]))*Pk_conv_dw[k+3]/(E_tab_dw[k+3]*pow(R_tab_dw[k+3],2)))
					 + 12.*WG_tab_dw[k+2][aa]*(W_tab_dw[k+2][bb]-WIA_tab_dw[k+2][bb]*A_IA[3]*C_IA[3]*Omega_m[3]/DG_tab_dw[k+2]*pow(1.+z_win[k+2],n_IA[3])*pow(lum_mean[k+2],B_IA[3]))*Pk_conv_dw[k+2]/(E_tab_dw[k+2]*pow(R_tab_dw[k+2],2)));
				}
			}
		}

		Pk_conv.clear(); Pk_conv_up.clear(); Pk_conv_dw.clear();

		if(paramo == 0){
			for(int aa=0; aa<zrange.size(); aa++){
				for(int bb=0; bb<zrange.size(); bb++){
					outGG<<setprecision(12)<<scientific<<C_ij_GG[aa][bb]<<" "; outLL<<setprecision(12)<<scientific<<C_ij_LL[aa][bb]<<" "; outGL<<setprecision(12)<<scientific<<C_ij_GL[aa][bb]<<" ";
					C_ij_GG[aa][bb] = 0; C_ij_LL[aa][bb] = 0; C_ij_GL[aa][bb] = 0;
				}
				outGG<<endl; outLL<<endl; outGL<<endl;
			}
		}
        
		for(int aa=0; aa<zrange.size(); aa++){
			for(int bb=0; bb<zrange.size(); bb++){
				outGGU<<setprecision(12)<<scientific<<C_ij_GG_up[aa][bb]<<" "; outGGD<<setprecision(12)<<scientific<<C_ij_GG_dw[aa][bb]<<" ";
				outLLU<<setprecision(12)<<scientific<<C_ij_LL_up[aa][bb]<<" "; outLLD<<setprecision(12)<<scientific<<C_ij_LL_dw[aa][bb]<<" ";
				outGLU<<setprecision(12)<<scientific<<C_ij_GL_up[aa][bb]<<" "; outGLD<<setprecision(12)<<scientific<<C_ij_GL_dw[aa][bb]<<" ";
				C_ij_GG_up[aa][bb] = 0; C_ij_GG_dw[aa][bb] = 0;
				C_ij_LL_up[aa][bb] = 0; C_ij_LL_dw[aa][bb] = 0;
				C_ij_GL_up[aa][bb] = 0; C_ij_GL_dw[aa][bb] = 0;
			}
            outGGU<<endl; outGGD<<endl; outLLU<<endl; outLLD<<endl; outGLU<<endl; outGLD<<endl;
		}
        if(paramo == 0){
            outGG.close(); outLL.close(); outGL.close();
        }
        outGGU.close(); outGGD.close(); outLLU.close(); outLLD.close(); outGLU.close(); outGLD.close();
	}
	progress = barometer(progress, NP);
}

//COmpute the covariance matrix, invert it and build the Fisher matrix.
void XC::Fisher(string probe, string Fisher_matrix_name, int lcut_mn, int lcut_pl, int run_index, int Mat_N, int Mat_NTOT){
	
    vector<double> l_new; 
    for(int i=lcut_mn; i<lcut_pl; i++){
        l_new.push_back(l[i]);
    }
    lsize=0;
    for(int i=0; i<l_new.size(); i++){
        if(l_new[i]<=min(l_max_WL,l_max_GC)){
            lsize++;
        }
    }

    lsize2 = l_new.size()-lsize;

    cout<<endl; cout<<endl;
	cout<<"Begin to compute the "+probe+" Fisher matrix"<<endl; 
	cout<<"Computing low l's covariance matrix"<<endl;

	string F_mat_name = Fisher_matrix_name;

	vector<string> param_chain = param_chain_Cl;
	vector<double> fid_all = fid_Cl;
	vector<double> steps_all = steps_Cl;

	if(zcut == "Y"){
		int i=0;
		while(i<param_chain.size()){
			for (int j=5; j<BX.size(); j++){
				string prefix = "b"+to_string(j+1);
				if(param_chain[i].compare(prefix) == 0){
    				param_chain.erase(param_chain.begin() + i);
    				fid_all.erase(fid_all.begin() + i);
    				steps_all.erase(steps_all.begin() + i);
    				i--;
    			}
			}
			i++;
		}
	}

	if(probe == "GCp"){
		int i=0;
		while(i<param_chain.size()){
			if(param_chain[i] == "A_IA" || param_chain[i] == "n_IA" || param_chain[i] == "B_IA"){
    			param_chain.erase(param_chain.begin() + i);
    			fid_all.erase(fid_all.begin() + i);
    			steps_all.erase(steps_all.begin() + i);
    			i--;
			}
			i++;
		}
	}
	if(probe == "WL"){
		int i=0;
		while(i<param_chain.size()){
			for (int j=0; j<BX.size(); j++){
				string prefix = "b"+to_string(j+1);
				if(param_chain[i].compare(prefix) == 0){
    				param_chain.erase(param_chain.begin() + i);
    				fid_all.erase(fid_all.begin() + i);
    				steps_all.erase(steps_all.begin() + i);
    				i--;
    			}
			}
			i++;
		}
	}

	if(curvature == "F"){
		int i=0;
		while(i<param_chain.size()){
			if(param_chain[i] == "wde"){
				param_chain.erase(param_chain.begin() + i);
				fid_all.erase(fid_all.begin() + i);
	    		steps_all.erase(steps_all.begin() + i);
	    		i--;
			}
			i++;
		}
	}

	int lll=0; int FX=0; int FY=0; int k_vec=0; int U1=0; int U2=0;
	int I_55, I_100, I_rect_I1, I_rect_I2, I_rect_I3, I_rect_I4;

	vector<vector<string>> C_folder = {{"Cl_GG", "Cl_GL"},{"Cl_GL", "Cl_LL"}};

	vector<vector<double>> Fisher_M(param_chain.size(), vector<double>(param_chain.size(), 0));
	int relat_index = param_chain.size()+1;
	int i=0; int yn=0;
	while(i<param_chain.size()){
		if(param_chain[i] == "wa"){
			relat_index = i;
			yn++;
		}
		i++;
	}

	vector<vector<double>> C_ij_ABCD_GG(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LL(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GL(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LG(zrange.size(), vector<double>(zrange.size(), 0));

	vector<vector<double>> C_ij_ABCD_GG_up_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LL_up_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GL_up_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LG_up_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GG_dw_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LL_dw_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GL_dw_PX(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LG_dw_PX(zrange.size(), vector<double>(zrange.size(), 0));

	vector<vector<double>> C_ij_ABCD_GG_up_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LL_up_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GL_up_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LG_up_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GG_dw_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LL_dw_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_GL_dw_PY(zrange.size(), vector<double>(zrange.size(), 0));
	vector<vector<double>> C_ij_ABCD_LG_dw_PY(zrange.size(), vector<double>(zrange.size(), 0));

	vector<vector<double>> CC_GGGG_R(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_LLLL_R(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_GLGL_R(Dim_y, vector<double>(Dim_y, 0));
	vector<vector<double>> CC_GGGL_R(Dim_x, vector<double>(Dim_y, 0));
	vector<vector<double>> CC_GGLL_R(Dim_x, vector<double>(Dim_x, 0)); 
	vector<vector<double>> CC_GLGG_R(Dim_y, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_LLGG_R(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_GLLL_R(Dim_y, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_LLGL_R(Dim_x, vector<double>(Dim_y, 0));
	
	vector<vector<double>> CC_GGGG_DR(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_LLLL_DR(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_GLGL_DR(Dim_y, vector<double>(Dim_y, 0));
	vector<vector<double>> CC_GGGL_DR(Dim_x, vector<double>(Dim_y, 0));
	vector<vector<double>> CC_GGLL_DR(Dim_x, vector<double>(Dim_x, 0)); 
	vector<vector<double>> CC_GLGG_DR(Dim_y, vector<double>(Dim_x, 0)); 
	vector<vector<double>> CC_LLGG_DR(Dim_x, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_GLLL_DR(Dim_y, vector<double>(Dim_x, 0));
	vector<vector<double>> CC_LLGL_DR(Dim_x, vector<double>(Dim_y, 0));
	
	vector<double> CC_GGGG(pow(Dim_x,2), 0); vector<double> CC_GGGG_D(pow(Dim_x,2), 0);
	vector<double> CC_LLLL(pow(Dim_x,2), 0); vector<double> CC_LLLL_D(pow(Dim_x,2), 0);
	vector<double> CC_GLGL(pow(Dim_y,2), 0); vector<double> CC_GLGL_D(pow(Dim_y,2), 0);
	vector<double> CC_GGGL(Dim_x*Dim_y, 0); vector<double> CC_GGGL_D(Dim_x*Dim_y, 0);
	vector<double> CC_GGLL(pow(Dim_x,2), 0); vector<double> CC_GGLL_D(pow(Dim_x,2), 0);
	vector<double> CC_GLGG(Dim_x*Dim_y, 0); vector<double> CC_GLGG_D(Dim_x*Dim_y, 0);
	vector<double> CC_LLGG(pow(Dim_x,2), 0); vector<double> CC_LLGG_D(pow(Dim_x,2), 0);
	vector<double> CC_GLLL(Dim_x*Dim_y, 0); vector<double> CC_GLLL_D(Dim_x*Dim_y, 0);
	vector<double> CC_LLGL(Dim_x*Dim_y, 0); vector<double> CC_LLGL_D(Dim_x*Dim_y, 0);
	
	vector<vector<double>> CO_CL(lsize*(2*Dim_x+Dim_y), vector<double>(lsize*(2*Dim_x+Dim_y), 0));
	vector<vector<double>> CO_CL_AB(lsize*(2*Dim_x+Dim_y), vector<double>(lsize*(2*Dim_x+Dim_y), 0));
	vector<vector<double>> CO_I(lsize*(2*Dim_x+Dim_y), vector<double>(lsize*(2*Dim_x+Dim_y), 0));
	vector<vector<double>> CO_CL_D(lsize*(2*Dim_x+Dim_y), vector<double>(lsize*(2*Dim_x+Dim_y), 0));
	vector<vector<double>> CO_CL_WL(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
	vector<vector<double>> CO_CL_WL_AB(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
	vector<vector<double>> CO_WL_I(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
	vector<vector<double>> CO_CL_WL_D(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
	vector<vector<double>> CO_CL_ref(lsize*(2*Dim_x+Dim_y), vector<double>(lsize*(2*Dim_x+Dim_y), 0));
	vector<vector<double>> CO_CL_WL_ref(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));

	for(int i=0; i<CO_I.size(); i++){
		CO_I[i][i] = 1.;
	}
	for(int i=0; i<CO_WL_I.size(); i++){
		CO_WL_I[i][i] = 1.;
	}

	for(int FX=0; FX<Fisher_M.size(); FX++){
		for(int FY=FX; FY<Fisher_M.size(); FY++){
            for(int lll=0; lll<lsize; lll++){

                ifstream ifile_ABCD_GG_up_PX(C_folder[0][0]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GG_dw_PX(C_folder[0][0]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_LL_up_PX(C_folder[1][1]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_LL_dw_PX(C_folder[1][1]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GL_up_PX(C_folder[0][1]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GL_dw_PX(C_folder[0][1]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));

                ifstream ifile_ABCD_GG_up_PY(C_folder[0][0]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GG_dw_PY(C_folder[0][0]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_LL_up_PY(C_folder[1][1]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_LL_dw_PY(C_folder[1][1]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GL_up_PY(C_folder[0][1]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                ifstream ifile_ABCD_GL_dw_PY(C_folder[0][1]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));

				if(FX == 0 && FY == 0){
                    ifstream ifile_ABCD_GG(C_folder[0][0]+"/C_fid"+"/COVAR_fid_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_LL(C_folder[1][1]+"/C_fid"+"/COVAR_fid_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_GL(C_folder[0][1]+"/C_fid"+"/COVAR_fid_"+to_string(l_new[lll]));
    				for(int i=0;i<zrange.size();i++){
						for(int j=0;j<zrange.size();j++){
							ifile_ABCD_GG>>C_ij_ABCD_GG[i][j]; ifile_ABCD_LL>>C_ij_ABCD_LL[i][j]; ifile_ABCD_GL>>C_ij_ABCD_GL[i][j];
						}
					}
					for(int i=0;i<zrange.size();i++){
						for(int j=0;j<zrange.size();j++){
							C_ij_ABCD_LG[i][j] = C_ij_ABCD_GL[j][i];
						}
					}
                    ifile_ABCD_GG.close(); ifile_ABCD_LL.close(); ifile_ABCD_GL.close();
    			}
    			
				for(int i=0;i<zrange.size();i++){
					for(int j=0;j<zrange.size();j++){
						ifile_ABCD_GG_up_PX>>C_ij_ABCD_GG_up_PX[i][j]; ifile_ABCD_LL_up_PX>>C_ij_ABCD_LL_up_PX[i][j]; ifile_ABCD_GL_up_PX>>C_ij_ABCD_GL_up_PX[i][j];
						ifile_ABCD_GG_dw_PX>>C_ij_ABCD_GG_dw_PX[i][j]; ifile_ABCD_LL_dw_PX>>C_ij_ABCD_LL_dw_PX[i][j]; ifile_ABCD_GL_dw_PX>>C_ij_ABCD_GL_dw_PX[i][j];
						ifile_ABCD_GG_up_PY>>C_ij_ABCD_GG_up_PY[i][j]; ifile_ABCD_LL_up_PY>>C_ij_ABCD_LL_up_PY[i][j]; ifile_ABCD_GL_up_PY>>C_ij_ABCD_GL_up_PY[i][j];
						ifile_ABCD_GG_dw_PY>>C_ij_ABCD_GG_dw_PY[i][j]; ifile_ABCD_LL_dw_PY>>C_ij_ABCD_LL_dw_PY[i][j]; ifile_ABCD_GL_dw_PY>>C_ij_ABCD_GL_dw_PY[i][j];
					}	
				}

				for(int i=0;i<zrange.size();i++){
					for(int j=0;j<zrange.size();j++){
						C_ij_ABCD_LG_up_PX[i][j] = C_ij_ABCD_GL_up_PX[j][i]; C_ij_ABCD_LG_dw_PX[i][j] = C_ij_ABCD_GL_dw_PX[j][i]; 
						C_ij_ABCD_LG_up_PY[i][j] = C_ij_ABCD_GL_up_PY[j][i]; C_ij_ABCD_LG_dw_PY[i][j] = C_ij_ABCD_GL_dw_PY[j][i]; 
					}	
				}

                ifile_ABCD_GG_up_PX.close(); ifile_ABCD_GG_dw_PX.close(); ifile_ABCD_GG_up_PY.close(); ifile_ABCD_GG_dw_PY.close();
                ifile_ABCD_LL_up_PX.close(); ifile_ABCD_LL_dw_PX.close(); ifile_ABCD_LL_up_PY.close(); ifile_ABCD_LL_dw_PY.close();
                ifile_ABCD_GL_up_PX.close(); ifile_ABCD_GL_dw_PX.close(); ifile_ABCD_GL_up_PY.close(); ifile_ABCD_GL_dw_PY.close();

				I_55=0; I_100=0; I_rect_I1=0; I_rect_I2=0; I_rect_I3=0; I_rect_I4=0;
				
                for(int I1=0; I1 < zrange.size(); I1++){
			        for(int I2=0; I2 < zrange.size(); I2++){
			            for(int I3=0; I3 < zrange.size(); I3++){
			                for(int I4=0; I4 < zrange.size(); I4++){

			                    if(I2 <= I1 && I4 <= I3){
			                    	if(FX == 0 && FY == 0){
			                        	CC_GGGG[I_55] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_GG[I2][I4] + C_ij_ABCD_GG[I1][I4]*C_ij_ABCD_GG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GGGG
                        				CC_LLLL[I_55] = (C_ij_ABCD_LL[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_LL[I1][I4]*C_ij_ABCD_LL[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //LLLL
                        				CC_GGLL[I_55] = (C_ij_ABCD_GL[I1][I3]*C_ij_ABCD_GL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_GL[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GGLL
                        				CC_LLGG[I_55] = (C_ij_ABCD_LG[I1][I3]*C_ij_ABCD_LG[I2][I4] + C_ij_ABCD_LG[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //LLGG
			                        }
			                        if(FX == relat_index && FY == relat_index){
			                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
			                        }
			                        else if(FX == relat_index && FY != relat_index){
			                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
			                        }
			                        else if(FX != relat_index && FY == relat_index){
			                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
                            			CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
			                        }
			                        else{
			                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                            			CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
			                        }
			                        I_55++;
			                    }

			                    if(I2 <= I1){
			                    	if(FX == 0 && FY == 0){
			                    		CC_GGGL[I_rect_I3] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_GL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_GG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GGGL
                        				CC_GLGG[I_rect_I3] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_LG[I2][I4] + C_ij_ABCD_GG[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GLGG
			                    	}
			                    	if(FX == relat_index && FY == relat_index){
                            			CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]));
                            			CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*steps_all[FX]));
                        			}
                        			else if(FX == relat_index && FY != relat_index){
                            			CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]*fid_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]));
                            			CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]*fid_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*steps_all[FX]));
                        			}
                        			else if(FX != relat_index && FY == relat_index){
                            			CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                            			CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                        			}
                        			else{
                            			CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                            			CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
			                        }
			                        I_rect_I3++;
			                    }

			                    if(I2 <= I1){
			                    	if(FX == 0 && FY == 0){
			                        	CC_LLGL[I_rect_I1] = (C_ij_ABCD_LG[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_LL[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //LLGL
                        				CC_GLLL[I_rect_I1] = (C_ij_ABCD_GL[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_LL[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GLLL
			                        }
			                        if(FX == relat_index && FY == relat_index){
                            			CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]));
                            			CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*steps_all[FX]));
                        			}
                        			else if(FX == relat_index && FY != relat_index){
                            			CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]));
                            			CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*steps_all[FX]));
                        			}
                        			else if(FX != relat_index && FY == relat_index){
                            			CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                            			CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                        			}
                        			else{
                            			CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
                            			CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]));
			                        }
			                        I_rect_I1++;
			                    }
			                    if(FX == 0 && FY == 0){
			                    	CC_GLGL[I_100] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GLGL
			                    }
			                    if(FX == relat_index && FY == relat_index){
                        			CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]);
			                    }
                    			else if(FX == relat_index && FY != relat_index){
                        			CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                    			}
                    			else if(FX != relat_index && FY == relat_index){
                        			CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]);
                    			}
                    			else{
                        			CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                    			}
			                    I_100++;
			                }
			            }
			        }
			    }

			    k_vec=0;
			    for(int i=0; i<Dim_x; i++){
			    	for(int j=0; j<Dim_x; j++){
			    		if(FX == 0 && FY == 0){
			    			CC_GGGG_R[i][j] = CC_GGGG[k_vec]; CC_LLLL_R[i][j] = CC_LLLL[k_vec];
			    			CC_GGLL_R[i][j] = CC_GGLL[k_vec]; CC_LLGG_R[i][j] = CC_LLGG[k_vec];
			    		}
			    		CC_GGGG_DR[i][j] = CC_GGGG_D[k_vec]; CC_LLLL_DR[i][j] = CC_LLLL_D[k_vec];
			    		CC_GGLL_DR[i][j] = CC_GGLL_D[k_vec]; CC_LLGG_DR[i][j] = CC_LLGG_D[k_vec];
			    		k_vec++;
			    	}
			    }

			    k_vec=0;
			    for(int i=0; i<Dim_y; i++){
			    	for(int j=0; j<Dim_y; j++){
			    		if(FX == 0 && FY == 0){
			    			CC_GLGL_R[i][j] = CC_GLGL[k_vec];
			    		}
			    		CC_GLGL_DR[i][j] = CC_GLGL_D[k_vec];
			    		k_vec++;
			    	}
			    }
			    k_vec=0;
			    for(int i=0; i<Dim_x; i++){
			    	for(int j=0; j<Dim_y; j++){
			    		if(FX == 0 && FY == 0){
			    			CC_GGGL_R[i][j] = CC_GGGL[k_vec]; CC_LLGL_R[i][j] = CC_LLGL[k_vec];
			    		}
			    		CC_GGGL_DR[i][j] = CC_GGGL_D[k_vec]; CC_LLGL_DR[i][j] = CC_LLGL_D[k_vec];
			    		k_vec++;
			    	}
			    }
			    k_vec=0;

			    int U1=0; int U2=0;
			    for(int i=0; i<Dim_y; i++){
			    	for(int j=0; j<Dim_x; j++){
			    		if(FX == 0 && FY == 0){
			    			CC_GLGG_R[i][j] = CC_GGGL_R[U2][U1]; CC_GLLL_R[i][j] = CC_LLGL_R[U2][U1];
			    		}
			    		CC_GLGG_DR[i][j] = CC_GGGL_DR[U2][U1]; CC_GLLL_DR[i][j] = CC_LLGL_DR[U2][U1];
			    		U2++;
			    	}
			    	U2=0;
			    	U1++;
			    }

			    for(int z1=0; z1<Dim_x; z1++){
			    	for(int z2=0; z2<Dim_x; z2++){
			    		if(FX == 0 && FY == 0){
			            	CO_CL[z1*lsize+lll][z2*lsize+lll] = CC_GGGG_R[z1][z2];
			            	CO_CL[lsize*Dim_x+z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_LLLL_R[z1][z2];
			            	CO_CL[z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_GGLL_R[z1][z2];
			            	CO_CL[lsize*Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_LLGG_R[z1][z2];
			            }
			            CO_CL_D[z1*lsize+lll][z2*lsize+lll] = CC_GGGG_DR[z1][z2];
			            CO_CL_D[lsize*Dim_x+z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_LLLL_DR[z1][z2];
			            CO_CL_D[z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_GGLL_DR[z1][z2];
			            CO_CL_D[lsize*Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_LLGG_DR[z1][z2];
			        }
			    }

			    for(int z1=0; z1<Dim_y; z1++){
			        for(int z2=0; z2<Dim_y; z2++){
			        	if(FX == 0 && FY == 0){
			            	CO_CL[2*lsize*Dim_x+z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_GLGL_R[z1][z2];
			        	}
			        	CO_CL_D[2*lsize*Dim_x+z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_GLGL_DR[z1][z2];
			        }
			    }

			    for(int z1=0; z1<Dim_x; z1++){
			        for(int z2=0; z2<Dim_y; z2++){
			        	if(FX == 0 && FY == 0){
			           		CO_CL[z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_GGGL_R[z1][z2];
			            	CO_CL[lsize*Dim_x+z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_LLGL_R[z1][z2];
			            }
			            CO_CL_D[z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_GGGL_DR[z1][z2];
			            CO_CL_D[lsize*Dim_x+z1*lsize+lll][2*lsize*Dim_x+z2*lsize+lll] = CC_LLGL_DR[z1][z2];
			        }
			    }

			    for(int z1=0; z1<Dim_y; z1++){
			        for(int z2=0; z2<Dim_x; z2++){
			        	if(FX == 0 && FY == 0){
			        		CO_CL[2*lsize*Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_GLGG_R[z1][z2];
            				CO_CL[2*lsize*Dim_x+z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_GLLL_R[z1][z2];
            			}
            			CO_CL_D[2*lsize*Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_GLGG_DR[z1][z2];
            			CO_CL_D[2*lsize*Dim_x+z1*lsize+lll][lsize*Dim_x+z2*lsize+lll] = CC_GLLL_DR[z1][z2];
			        }
			    }
			}

			if(probe == "GCp"){
				if(FX == 0 && FY == 0){
					vector<vector<double>> CO_CL_temp(lsize*Dim_x, vector<double>(lsize*Dim_x, 0));
					CO_CL_AB = CO_CL_temp; CO_I = CO_CL_temp;
					for(int i=0; i<lsize*Dim_x; i++){
						CO_I[i][i] = 1.;
						for(int j=0; j<lsize*Dim_x; j++){
							CO_CL_temp[i][j] = CO_CL[i][j];
						}
					}
    				CO_CL = CO_CL_temp;
    			}
    			vector<vector<double>> CO_CL_temp_D(lsize*Dim_x, vector<double>(lsize*Dim_x, 0));
    			for(int i=0; i<lsize*Dim_x; i++){
					for(int j=0; j<lsize*Dim_x; j++){
						CO_CL_temp_D[i][j] = CO_CL_D[i][j];
					}
				}
    			CO_CL_D = CO_CL_temp_D;
			}

			if(probe == "WL"){
				if(FX == 0 && FY == 0){
					vector<vector<double>> CO_CL_temp(lsize*Dim_x, vector<double>(lsize*Dim_x, 0));
					CO_CL_AB = CO_CL_temp; CO_I = CO_CL_temp;
					for(int i=lsize*Dim_x; i<2*lsize*Dim_x; i++){
						CO_I[i-lsize*Dim_x][i-lsize*Dim_x] = 1.;
						for(int j=lsize*Dim_x; j<2*lsize*Dim_x; j++){
							CO_CL_temp[i-lsize*Dim_x][j-lsize*Dim_x] = CO_CL[i][j];
						}
					}
    				CO_CL = CO_CL_temp;
    			}
    			vector<vector<double>> CO_CL_temp_D(lsize*Dim_x, vector<double>(lsize*Dim_x, 0));
    			for(int i=lsize*Dim_x; i<2*lsize*Dim_x; i++){
					for(int j=lsize*Dim_x; j<2*lsize*Dim_x; j++){
						CO_CL_temp_D[i-lsize*Dim_x][j-lsize*Dim_x] = CO_CL_D[i][j];
					}
				}
    			CO_CL_D = CO_CL_temp_D;
    		}
			
			if(FX==0 && FY == 0){
				cout<<"Begin Cholesky decomposition"<<endl;
                for(int i=0; i<CO_CL.size(); i++){
                    for(int j=0; j<=i; j++){
                        if((CO_CL[i][0] != 0) || (CO_CL[i][j] != 0) || (j>0 && CO_CL_AB[i][j-1] != 0)){
                            double sum=0;
                            if(i==j){
                                for(int k=0; k<j; k++){
                                    if(CO_CL_AB[j][k] != 0){
                                        sum += pow(CO_CL_AB[j][k], 2);
                                    }
                                }
                                CO_CL_AB[i][j] = pow(CO_CL[j][j]-sum,0.5);
                            }
                            else{
                                for(int k=0; k<j; k++){
                                    if(CO_CL_AB[i][k] != 0 || CO_CL_AB[j][k] != 0){
                                        sum += CO_CL_AB[i][k]*CO_CL_AB[j][k];
                                    }
                                }
                                CO_CL_AB[i][j] = (CO_CL[i][j] - sum)/CO_CL_AB[j][j];
                            }
                        }
                    }
                }

    			CO_CL = matrix_inverse(CO_CL_AB, CO_I);
			}

			for(int i=0; i<CO_CL.size(); i++){
				for(int j=0; j<CO_CL.size(); j++){
                    if( CO_CL[i][j] != 0 || CO_CL_D[i][j] != 0){
					   Fisher_M[FX][FY] += CO_CL[i][j]*CO_CL_D[i][j];
                    }
				}
			}
			Fisher_M[FY][FX] = Fisher_M[FX][FY];
			CO_CL_D = CO_CL_ref;

            if(l_max_WL > l_max_GC && probe != "GCp"){
				for(int lll=lsize; lll<l_new.size(); lll++){

                    ifstream ifile_ABCD_LL_up_PX(C_folder[1][1]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_LL_dw_PX(C_folder[1][1]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_LL_up_PY(C_folder[1][1]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_LL_dw_PY(C_folder[1][1]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));

					if(FX == 0 && FY == 0){
                        ifstream ifile_ABCD_LL(C_folder[1][1]+"/C_fid"+"/COVAR_fid_"+to_string(l_new[lll]));
						if(lll==lsize){
							cout<<"Computing high l's covariance matrix"<<endl;
						}
	    				for(int i=0;i<zrange.size();i++){
							for(int j=0;j<zrange.size();j++){
								ifile_ABCD_LL>>C_ij_ABCD_LL[i][j];
							}
						}
                        ifile_ABCD_LL.close();
	    			}

					for(int i=0;i<zrange.size();i++){
						for(int j=0;j<zrange.size();j++){
							ifile_ABCD_LL_up_PX>>C_ij_ABCD_LL_up_PX[i][j]; ifile_ABCD_LL_dw_PX>>C_ij_ABCD_LL_dw_PX[i][j];
							ifile_ABCD_LL_up_PY>>C_ij_ABCD_LL_up_PY[i][j]; ifile_ABCD_LL_dw_PY>>C_ij_ABCD_LL_dw_PY[i][j];
						}	
					}

                    ifile_ABCD_LL_up_PX.close(); ifile_ABCD_LL_dw_PX.close(); ifile_ABCD_LL_up_PY.close(); ifile_ABCD_LL_dw_PY.close();
					
					I_55=0;
					for(int I1=0; I1 < zrange.size(); I1++){
				        for(int I2=0; I2 < zrange.size(); I2++){
				            for(int I3=0; I3 < zrange.size(); I3++){
				                for(int I4=0; I4 < zrange.size(); I4++){

				                    if(I2 <= I1 && I4 <= I3){
				                    	if(FX == 0 && FY == 0){
	                        				CC_LLLL[I_55] = (C_ij_ABCD_LL[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_LL[I1][I4]*C_ij_ABCD_LL[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //LLLL
				                        }
				                        if(FX == relat_index && FY == relat_index){
	                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
				                        }
				                        else if(FX == relat_index && FY != relat_index){
	                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
				                        }
				                        else if(FX != relat_index && FY == relat_index){
	                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]);
				                        }
				                        else{
	                            			CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
				                        }
				                        I_55++;
				                    }
				                }
				            }
				        }
				    }

				    k_vec=0;
				    for(int i=0; i<Dim_x; i++){
				    	for(int j=0; j<Dim_x; j++){
				    		if(FX == 0 && FY == 0){
				    			CC_LLLL_R[i][j] = CC_LLLL[k_vec];
				    		}
				    		CC_LLLL_DR[i][j] = CC_LLLL_D[k_vec];
				    		k_vec++;
				    	}
				    }

				    for(int z1=0; z1<Dim_x; z1++){
				    	for(int z2=0; z2<Dim_x; z2++){
				    		if(FX == 0 && FY == 0){
				            	CO_CL_WL[z1*lsize2+lll-lsize][z2*lsize2+lll-lsize] = CC_LLLL_R[z1][z2];
				            }
				            CO_CL_WL_D[z1*lsize2+lll-lsize][z2*lsize2+lll-lsize] = CC_LLLL_DR[z1][z2];
				        }
				    }
				}
			
				if(probe == "WL"){
					if(FX == 0 && FY == 0){
						vector<vector<double>> CO_CL_temp(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
						CO_CL_WL_AB = CO_CL_temp; CO_WL_I = CO_CL_temp;
						for(int i=0; i<lsize2*Dim_x; i++){
							CO_WL_I[i][i] = 1.;
							for(int j=0; j<lsize2*Dim_x; j++){
								CO_CL_temp[i][j] = CO_CL_WL[i][j];
							}
						}
	    				CO_CL_WL = CO_CL_temp;
	    			}
	    			vector<vector<double>> CO_CL_temp_D(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
	    			for(int i=0; i<lsize2*Dim_x; i++){
						for(int j=0; j<lsize2*Dim_x; j++){
							CO_CL_temp_D[i][j] = CO_CL_WL_D[i][j];
						}
					}
	    			CO_CL_WL_D = CO_CL_temp_D;
	    		}

				if (FX==0 && FY == 0){
					cout<<"Begin Cholesky decomposition"<<endl;
					for(int i=0; i<CO_CL_WL.size(); i++){
	        			for(int j=0; j<=i; j++){
                            if((CO_CL_WL[i][0] != 0) || (CO_CL_WL[i][j] != 0) || (j>0 && CO_CL_WL_AB[i][j-1] != 0)){
                    			double sum=0;
                    			if(i==j){
                        			for(int k=0; k<j; k++){
                                        if(CO_CL_WL_AB[j][k] != 0){
                            			    sum += pow(CO_CL_WL_AB[j][k], 2);
                                        }
                        			}
                        			CO_CL_WL_AB[i][j] = pow(CO_CL_WL[j][j]-sum,0.5);
                    			}
                    			else{
                        			for(int k=0; k<j; k++){
                                        if(CO_CL_WL_AB[i][k] != 0 && CO_CL_WL_AB[j][k] != 0){
                            			    sum += CO_CL_WL_AB[i][k]*CO_CL_WL_AB[j][k];
                                        }
                        			}
                    				CO_CL_WL_AB[i][j] = (CO_CL_WL[i][j] - sum)/CO_CL_WL_AB[j][j];
        	        			}
                            }
                        }
	    			}
	    			CO_CL_WL = matrix_inverse(CO_CL_WL_AB, CO_WL_I);
	    			cout<<"Computing the Fisher matrix elements"<<endl;
				}

				for(int i=0; i<CO_CL_WL.size(); i++){
					for(int j=0; j<CO_CL_WL.size(); j++){
                        if( CO_CL_WL[i][j] != 0 || CO_CL_WL_D[i][j] != 0){
						  Fisher_M[FX][FY] += CO_CL_WL[i][j]*CO_CL_WL_D[i][j];
                        }
					}
				}
				Fisher_M[FY][FX] = Fisher_M[FX][FY];
			}

            if(l_max_WL < l_max_GC && probe != "WL" ){
                for(int lll=lsize; lll<l_new.size(); lll++){

                    ifstream ifile_ABCD_GG_up_PX(C_folder[0][0]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_GG_dw_PX(C_folder[0][0]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_GG_up_PY(C_folder[0][0]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+to_string(l_new[lll]));
                    ifstream ifile_ABCD_GG_dw_PY(C_folder[0][0]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+to_string(l_new[lll]));

                    if(FX == 0 && FY == 0){
                        ifstream ifile_ABCD_GG(C_folder[0][0]+"/C_fid"+"/COVAR_fid_"+to_string(l_new[lll]));
                        if(lll==lsize){
                            cout<<"Computing high l's covariance matrix"<<endl;
                        }
                        for(int i=0;i<zrange.size();i++){
                            for(int j=0;j<zrange.size();j++){
                                ifile_ABCD_GG>>C_ij_ABCD_GG[i][j];
                            }
                        }
                        ifile_ABCD_GG.close();
                    }

                    for(int i=0;i<zrange.size();i++){
                        for(int j=0;j<zrange.size();j++){
                            ifile_ABCD_GG_up_PX>>C_ij_ABCD_GG_up_PX[i][j]; ifile_ABCD_GG_dw_PX>>C_ij_ABCD_GG_dw_PX[i][j];
                            ifile_ABCD_GG_up_PY>>C_ij_ABCD_GG_up_PY[i][j]; ifile_ABCD_GG_dw_PY>>C_ij_ABCD_GG_dw_PY[i][j];
                        }   
                    }

                    ifile_ABCD_GG_up_PX.close(); ifile_ABCD_GG_dw_PX.close(); ifile_ABCD_GG_up_PY.close(); ifile_ABCD_GG_dw_PY.close();
                    
                    I_55=0;
                    for(int I1=0; I1 < zrange.size(); I1++){
                        for(int I2=0; I2 < zrange.size(); I2++){
                            for(int I3=0; I3 < zrange.size(); I3++){
                                for(int I4=0; I4 < zrange.size(); I4++){

                                    if(I2 <= I1 && I4 <= I3){
                                        if(FX == 0 && FY == 0){
                                            CC_GGGG[I_55] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_GG[I2][I4] + C_ij_ABCD_GG[I1][I4]*C_ij_ABCD_GG[I2][I3])/((2*l_new[lll]+1)*fsky*delta_l); //GGGG
                                        }
                                        if(FX == relat_index && FY == relat_index){
                                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
                                        }
                                        else if(FX == relat_index && FY != relat_index){
                                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                                        }
                                        else if(FX != relat_index && FY == relat_index){
                                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]);
                                        }
                                        else{
                                            CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]);
                                        }
                                        I_55++;
                                    }
                                }
                            }
                        }
                    }

                    k_vec=0;
                    for(int i=0; i<Dim_x; i++){
                        for(int j=0; j<Dim_x; j++){
                            if(FX == 0 && FY == 0){
                                CC_GGGG_R[i][j] = CC_GGGG[k_vec];
                            }
                            CC_GGGG_DR[i][j] = CC_GGGG_D[k_vec];
                            k_vec++;
                        }
                    }

                    for(int z1=0; z1<Dim_x; z1++){
                        for(int z2=0; z2<Dim_x; z2++){
                            if(FX == 0 && FY == 0){
                                CO_CL_WL[z1*lsize2+lll-lsize][z2*lsize2+lll-lsize] = CC_GGGG_R[z1][z2];
                            }
                            CO_CL_WL_D[z1*lsize2+lll-lsize][z2*lsize2+lll-lsize] = CC_GGGG_DR[z1][z2];
                        }
                    }
                }
            
                if(probe == "GCp"){
                    if(FX == 0 && FY == 0){
                        vector<vector<double>> CO_CL_temp(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
                        CO_CL_WL_AB = CO_CL_temp; CO_WL_I = CO_CL_temp;
                        for(int i=0; i<lsize2*Dim_x; i++){
                            CO_WL_I[i][i] = 1.;
                            for(int j=0; j<lsize2*Dim_x; j++){
                                CO_CL_temp[i][j] = CO_CL_WL[i][j];
                            }
                        }
                        CO_CL_WL = CO_CL_temp;
                    }
                    vector<vector<double>> CO_CL_temp_D(lsize2*Dim_x, vector<double>(lsize2*Dim_x, 0));
                    for(int i=0; i<lsize2*Dim_x; i++){
                        for(int j=0; j<lsize2*Dim_x; j++){
                            CO_CL_temp_D[i][j] = CO_CL_WL_D[i][j];
                        }
                    }
                    CO_CL_WL_D = CO_CL_temp_D;
                }

                if (FX==0 && FY == 0){
                    cout<<"Begin Cholesky decomposition"<<endl;
                    for(int i=0; i<CO_CL_WL.size(); i++){
                        for(int j=0; j<=i; j++){
                            if((CO_CL_WL[i][0] != 0) || (CO_CL_WL[i][j] != 0) || (j>0 && CO_CL_WL_AB[i][j-1] != 0)){
                                double sum=0;
                                if(i==j){
                                    for(int k=0; k<j; k++){
                                        if(CO_CL_WL_AB[j][k] != 0){
                                            sum += pow(CO_CL_WL_AB[j][k], 2);
                                        }
                                    }
                                    CO_CL_WL_AB[i][j] = pow(CO_CL_WL[j][j]-sum,0.5);
                                }
                                else{
                                    for(int k=0; k<j; k++){
                                        if(CO_CL_WL_AB[i][k] != 0 && CO_CL_WL_AB[j][k] != 0){
                                            sum += CO_CL_WL_AB[i][k]*CO_CL_WL_AB[j][k];
                                        }
                                    }
                                    CO_CL_WL_AB[i][j] = (CO_CL_WL[i][j] - sum)/CO_CL_WL_AB[j][j];
                                }
                            }
                        }
                    }
                    CO_CL_WL = matrix_inverse(CO_CL_WL_AB, CO_WL_I);
                    cout<<"Computing the Fisher matrix elements"<<endl;
                }

                for(int i=0; i<CO_CL_WL.size(); i++){
                    for(int j=0; j<CO_CL_WL.size(); j++){
                        if( CO_CL_WL[i][j] != 0 || CO_CL_WL_D[i][j] != 0){
                            Fisher_M[FX][FY] += CO_CL_WL[i][j]*CO_CL_WL_D[i][j];
                        }
                    }
                }
                Fisher_M[FY][FX] = Fisher_M[FX][FY];
            }

			CO_CL_WL_D = CO_CL_WL_ref;
		}
	}

	ofstream outF; outF.open("output/"+F_mat_name+"_"+to_string(Mat_N));	
	for(int FX=0; FX<Fisher_M.size(); FX++){
		for(int FY=0; FY<Fisher_M.size(); FY++){
			outF<<setprecision(12)<<scientific<<Fisher_M[FX][FY]<<" ";
		}
		outF<<endl;
	}
	outF.close();

    if(run_index == 1){
        vector<vector<double>> Fisher_Final(param_chain.size(), vector<double>(param_chain.size(), 0));
        vector<vector<double>> Fisher_T(param_chain.size(), vector<double>(param_chain.size(), 0));

        for(int k=0; k<Mat_NTOT; k++){
            ifstream ifile("output/"+F_mat_name+"_"+to_string(k));
            for(int i=0;i<Fisher_Final.size();i++){
                for(int j=0;j<Fisher_Final.size();j++){
                    ifile>>Fisher_T[i][j];
                }
            }
            ifile.close();
            for(int i=0;i<Fisher_Final.size();i++){
                for(int j=0;j<Fisher_Final.size();j++){
                    Fisher_Final[i][j]+=Fisher_T[i][j];
                }
            }
        }

        ofstream outG; outG.open("output/"+F_mat_name); 
        for(int FX=0; FX<Fisher_Final.size(); FX++){
            for(int FY=0; FY<Fisher_Final.size(); FY++){
                outG<<setprecision(12)<<scientific<<Fisher_Final[FX][FY]<<" ";
            }
            outG<<endl;
        }
        outG.close();
    }
    cout<<probe+" Fisher matrix saved"<<endl;
}

//Progression bar update.
double XC::barometer(double progress, int NP){
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
    progress += 1./NP;
    return progress;
}

//Destructor
XC::~XC(){
}
