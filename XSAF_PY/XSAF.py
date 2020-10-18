##################################################################################
#Author: S. Yahia-Cherif, F. Dournac, I.Tutusaus
#This script computes the photometric Fisher matrix
##################################################################################

#Modules import.
import sys
import numpy as np
import scipy.integrate as pyint
import os
from os import path
import glob
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import multiprocessing as mp
from multiprocessing import Pool
import scipy.ndimage
from numba import jit, njit
from numba import autojit, prange
from scipy import integrate
import resource
from scipy.linalg import lapack

def b_new(z, w0_p, wa_p):
    return pyint.quad(b_aux, 0, z, args = (w0_p, wa_p))[0]

def b_aux(z, w0_p, wa_p):
    return 1./(1+z)*(1+w0_p+wa_p*z/(1+z))

def E(z, h_p, wm_p, wDE_p, w0_p, wa_p, curvature): 
    if curvature == 'NF':
        return np.sqrt( wm_p*(1+z)**3 + wDE_p*np.exp(3*b_new(z, w0_p, wa_p)) + (1.-wm_p-wDE_p)*(1+z)**2)
    elif curvature == 'F':
        return np.sqrt( wm_p*(1+z)**3 + (1.-wm_p)*np.exp(3*b_new(z, w0_p, wa_p)))
    
def chi_aux_R(z, h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature):
    return c/(100.*h_p*E(z, h_p, wm_p, wDE_p, w0_p, wa_p, curvature))
    
def chi_R(z, h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature):  
    return pyint.quad(chi_aux_R, 0, z, args = (h_p, wm_p, wDE_p, w0_p, wa_p, c, curvature))[0]

def OM_M(z, h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature):
    return (wm_p*(1.+z)**3/(E(z, h_p, wm_p, wDE_p, w0_p, wa_p, curvature)**2))**gamma_p/(1.+z)
    
def DGrowth(z, h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature):
    return np.exp(-pyint.quad(OM_M, 0, z, args = (h_p, wm_p, wDE_p, w0_p, wa_p, gamma_p, curvature))[0])

def parainterp(P_dd_spec, cl, RT500, P_dd_C):
    P_dd_ok = np.zeros(len(P_dd_C))
    for cc in range(len(P_dd_C)): 
        if ((cl+0.5)/RT500[cc] < 35 and (cl+0.5)/RT500[cc] > 0.0005):
            P_dd_ok[cc] = P_dd_spec[cc]((cl+0.5)/RT500[cc])
    return P_dd_ok

class XC:
    
	def __init__(self):

		self.paramo = 0

		self.red_B = 10
		self.der_B = 5
		self.MMM = 60
		self.WWW = 100
		self.zmin = 0.001
		self.zmax = 3.731
		self.lmax_WL, self.lmaxGC = 5000, 3000
		self.stencil = 3 

		self.curvature = 'F'
		self.gamma_MG = 'N'
		self.zcut = 'N'
		self.l_max_WL, self.l_max_GC = 5000, 3000

		self.c = 299792.458
		self.fsky = 15000./41253.0 
		self.zrange_0 = 0.63639610
		self.alpha=1.5 
		self.ng = 30./(8.461594994079999e-08)
		self.cb = 1.0 
		self.zb = 0.0
		self.sig_b = 0.05 
		self.c0 = 1.0 
		self.zo = 0.1 
		self.sig_0 = 0.05
		self.f_out = 0.1 
		self.sig_epsilon = 0.3

		self.Z_NB = np.linspace(self.zmin, self.zmax, 100000)
		self.integrale, self.integrale_bis = 0, 0

		#for i in range(len(self.Z_NB)):
		#	self.integrale += 1./90*(self.Z_NB[i]-self.Z_NB[i-1])*(7*(self.nz(self.Z_NB[i]) + self.nz(self.Z_NB[i-1])) + 32*(self.nz(0.25*(3*self.Z_NB[i]+self.Z_NB[i-1])) + self.nz(0.25*(self.Z_NB[i]+3*self.Z_NB[i-1]))) + 12*self.nz(0.5*(self.Z_NB[i]+self.Z_NB[i-1])));
		self.integrale = 0.5*(self.Z_NB[1]-self.Z_NB[0])*np.sum(self.nz(self.Z_NB[1:]) + self.nz(self.Z_NB[:-1]))
		
		self.bin_N=0./self.red_B;
		self.z_im, self.z_ip = np.zeros(0), np.zeros(self.red_B)
		for i in range(1, len(self.Z_NB)):
			self.integrale_bis = self.integrale_bis + 0.5*(self.Z_NB[1]-self.Z_NB[0])*(self.nz(self.Z_NB[i]) + self.nz(self.Z_NB[i-1]));
			if(self.integrale_bis >= self.bin_N*self.integrale):
				self.z_im = np.insert(self.z_im, len(self.z_im), self.Z_NB[i-1]);
				self.bin_N+=1./self.red_B;
		self.z_ip[0:len(self.z_ip)-1] = self.z_im[1:]
		self.z_ip[len(self.z_ip)-1] = self.zmax

		self.zrange = (self.z_ip+self.z_im)/2

		self.BX = np.zeros((self.red_B,self.der_B))
		for i in range(len(self.BX)):
		    for j in range (len(self.BX[0])):
		        self.BX[i][j] = (1+self.zrange[i])**0.5
		        if self.zcut == 'Y':
		            if i > 4:
		                self.BX[i][j] = 0.

		self.Omega_b, self.h, self.Omega_m, self.ns, self.Omega_DE, self.w0, self.wa, self.sigma_8, self.gamma = 0.05*np.ones(5), 0.67*np.ones(5), 0.32*np.ones(5), 0.96*np.ones(5), 0.68*np.ones(5), -1.*np.ones(5), 0.*np.ones(5), 0.81553388*np.ones(5), 6./11*np.ones(5)
		self.C_IA, self.A_IA, self.B_IA, self.n_IA = 0.0134*np.ones(5), 1.72*np.ones(5), 2.17*np.ones(5), -0.41*np.ones(5)
		self.eps_Omega_b, self.eps_h, self.eps_Omega_m, self.eps_ns, self.eps_Omega_DE, self.eps_w0, self.eps_wa, self.eps_sigma_8, self.eps_gamma, self.eps_b = 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
		self.eps_A_IA, self.eps_n_IA, self.eps_B_IA = 0.001, 0.001, 0.001

		self.param_chain_A = np.array(["wm", "wde", "wb", "w0", "wa", "h", "ns", "s8", "gamma", "A_IA", "n_IA", "B_IA"]);
		self.fid_A = np.array([self.Omega_m[2], self.Omega_DE[2], self.Omega_b[2], self.w0[2], self.wa[2], self.h[2], self.ns[2], self.sigma_8[2], self.gamma[2], self.A_IA[2], self.n_IA[2], self.B_IA[2]])
		self.steps_A = np.array([self.eps_Omega_m, self.eps_Omega_DE, self.eps_Omega_b, self.eps_w0, self.eps_wa, self.eps_h, self.eps_ns, self.eps_sigma_8, self.eps_gamma, self.eps_A_IA, self.eps_n_IA, self.eps_B_IA])
		for i in range(len(self.zrange)):
			self.param_chain_A = np.insert(self.param_chain_A, len(self.param_chain_A), "b"+str(i+1))
			self.fid_A = np.insert(self.fid_A, len(self.fid_A), (self.BX[i][2]))
			self.steps_A = np.insert(self.steps_A, len(self.steps_A), self.eps_b)

		self.z500 = np.linspace(self.zmin, self.zmax, self.MMM)

		self.z_win_x_y = np.linspace(self.zmin,self.zmax, self.WWW)
		self.z_win_3_1 = 1./4*(3*self.z_win_x_y[1:]+self.z_win_x_y[:-1])
		self.z_win_1_3 = 1./4*(self.z_win_x_y[1:]+3*self.z_win_x_y[:-1])
		self.z_win_1_2 = 1./2*(self.z_win_x_y[1:]+self.z_win_x_y[:-1])
		self.z_win = np.zeros(len(self.z_win_x_y)+len(self.z_win_3_1)+len(self.z_win_1_2)+len(self.z_win_1_3))
		i,j=0,0
		while i < len(self.z_win):
		    if i == 0:
		        self.z_win[i] = self.z_win_x_y[j]
		        i=i+1
		        j=j+1
		    elif i > 0 and i < len(self.z_win):
		        self.z_win[i] = self.z_win_1_3[j-1]
		        self.z_win[i+1] = self.z_win_1_2[j-1]
		        self.z_win[i+2] = self.z_win_3_1[j-1]
		        self.z_win[i+3] = self.z_win_x_y[j]
		        i=i+4
		        j=j+1
		self.delta_z500 = self.z500[1]-self.z500[0]

		self.Pk_Path = "WP_Pk/" 
		self.fold_path = np.array([self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid"]);
		self.pre_CC_path = np.array(["Cl_GG/", "Cl_LL/", "Cl_GL/"])
		self.CC_path = np.array(["C_wb_up", "C_wb_up2", "C_fid", "C_wb_dw", "C_wb_dw2"]);
		self.fold_path = [self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid", self.Pk_Path+"fid"]

		self.param_chain_Cl, self.fid_Cl, self.steps_Cl = [], [], []
		self.Pk_files = []
		self.Pk_size_dw, self.Pk_size, self.Pk_size_up = 0.0, 0.0, 0.0

		self.zvals, self.Lumos = np.loadtxt("scaledmeanlum-E2Sa.dat", usecols=(0,1), unpack="True")
		self.Lum = interp1d(self.zvals,self.Lumos,kind="linear")
		self.lum_mean = self.Lum(self.z_win)

		self.l_edge = np.linspace(10, 5000, 61)
		self.l = np.zeros(len(self.l_edge)-1)
		for i in range(1, len(self.l_edge)):
		    self.l[i-1] = (self.l_edge[i] + self.l_edge[i-1])/2
		self.delta_l = self.l[1]-self.l[0]

		self.E_tab, self.E_tab_up, self.E_tab_dw = np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win))
		self.R_tab, self.R_tab_up, self.R_tab_dw = np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win))
		self.RT_tab, self.RT_tab_up, self.RT_tab_dw = np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win))
		self.DG_tab, self.DG_tab_up, self.DG_tab_dw = np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win))

		self.Photoz_tab = np.zeros((len(self.z_win), self.red_B))

		self.WG_tab, self.WG_tab_up, self.WG_tab_dw = np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B))
		self.W_tab, self.W_tab_up, self.W_tab_dw = np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B))
		self.WIA_tab, self.WIA_tab_up, self.WIA_tab_dw = np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B)), np.zeros((len(self.z_win), self.red_B))

		self.C_ij_GG, self.C_ij_GG_up, self.C_ij_GG_dw = np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B))
		self.C_ij_LL, self.C_ij_LL_up, self.C_ij_LL_dw = np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B))
		self.C_ij_GL, self.C_ij_GL_up, self.C_ij_GL_dw = np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B)), np.zeros((self.red_B, self.red_B))

		self.lsize=0;
		for i in range(len(self.l)):
			if(self.l[i]<=min(self.l_max_WL,self.l_max_GC)):
				self.lsize=self.lsize+1

		self.lsize2 = len(self.l)-self.lsize

		self.Dim_x = 0;
		for i in range(1,len(self.zrange)+1):
			self.Dim_x += i;
		self.Dim_y = len(self.zrange)**2

	def Initializers_G(self, N_paramo):
		self.paramo = N_paramo;
		self.param_chain_Cl.append(self.param_chain_A[self.paramo])
		self.fid_Cl.append(self.fid_A[self.paramo])
		self.steps_Cl.append(self.steps_A[self.paramo])

		if self.paramo == 0:
		    self.fold_path = [self.Pk_Path+"wb_up", self.Pk_Path+"wb_up2", self.Pk_Path+"fid", self.Pk_Path+"wb_dw", self.Pk_Path+"wb_dw2"]
		    self.CC_path = ["C_wb_up", "C_wb_up2", "C_fid", "C_wb_dw", "C_wb_dw2"]
		elif self.paramo == 1:
		    self.h = np.array([self.h[0]*(1.+self.eps_h), self.h[1]*(1.+2*self.eps_h), self.h[2], self.h[3]*(1.-self.eps_h), self.h[4]*(1.-2*self.eps_h)])
		    self.fold_path = [self.Pk_Path+"h_up", self.Pk_Path+"h_up2", self.Pk_Path+"fid", self.Pk_Path+"h_dw", self.Pk_Path+"h_dw2"]
		    self.CC_path = ["C_h_up", "C_h_up2", "C_fid", "C_h_dw", "C_h_dw2"]
		elif self.paramo == 2:
		    self.Omega_m = np.array([self.Omega_m[0]*(1.+self.eps_Omega_m), self.Omega_m[1]*(1.+2*self.eps_Omega_m), self.Omega_m[2], self.Omega_m[3]*(1.-self.eps_Omega_m), self.Omega_m[4]*(1.-2*self.eps_Omega_m)])
		    self.fold_path = [self.Pk_Path+"wm_up", self.Pk_Path+"wm_up2", self.Pk_Path+"fid", self.Pk_Path+"wm_dw", self.Pk_Path+"wm_dw2"]
		    self.CC_path = ["C_wm_up", "C_wm_up2", "C_fid", "C_wm_dw", "C_wm_dw2"]
		elif self.paramo == 3:
		    self.fold_path = [self.Pk_Path+"ns_up", self.Pk_Path+"ns_up2", self.Pk_Path+"fid", self.Pk_Path+"ns_dw", self.Pk_Path+"ns_dw2"]
		    self.CC_path = ["C_ns_up", "C_ns_up2", "C_fid", "C_ns_dw", "C_ns_dw2"] 
		elif self.paramo == 4:
		    self.Omega_DE = np.array([self.Omega_DE[0]*(1.+self.eps_Omega_DE), self.Omega_DE[1]*(1.+2*self.eps_Omega_DE), self.Omega_DE[2], self.Omega_DE[3]*(1.-self.eps_Omega_DE), self.Omega_DE[4]*(1.-2*self.eps_Omega_DE)])  
		    self.fold_path = [self.Pk_Path+"wde_up", self.Pk_Path+"wde_up2", self.Pk_Path+"fid", self.Pk_Path+"wde_dw", self.Pk_Path+"wde_dw2"]
		    self.CC_path = ["C_wde_up", "C_wde_up2", "C_fid", "C_wde_dw", "C_wde_dw2"]  
		elif self.paramo == 5:
		    self.w0 = np.array([self.w0[0]*(1.+self.eps_w0), self.w0[1]*(1.+2*self.eps_w0), self.w0[2], self.w0[3]*(1.-self.eps_w0), self.w0[4]*(1.-2*self.eps_w0)])
		    self.fold_path = [self.Pk_Path+"w0_up", self.Pk_Path+"w0_up2", self.Pk_Path+"fid", self.Pk_Path+"w0_dw", self.Pk_Path+"w0_dw2"]
		    self.CC_path = ["C_w0_up", "C_w0_up2", "C_fid", "C_w0_dw", "C_w0_dw2"]
		elif self.paramo == 6:
		    self.wa = np.array([self.wa[0]+self.eps_wa, self.wa[1]+2*self.eps_wa, self.wa[2], self.wa[3]-self.eps_wa, self.wa[4]-2*self.eps_wa])
		    self.fold_path = [self.Pk_Path+"wa_up", self.Pk_Path+"wa_up2", self.Pk_Path+"fid", self.Pk_Path+"wa_dw", self.Pk_Path+"wa_dw2"]
		    self.CC_path = ["C_wa_up", "C_wa_up2", "C_fid", "C_wa_dw", "C_wa_dw2"]
		elif self.paramo == 7:
		    self.fold_path = [self.Pk_Path+"s8_up", self.Pk_Path+"s8_up2", self.Pk_Path+"fid", self.Pk_Path+"s8_dw", self.Pk_Path+"s8_dw2"]
		    self.CC_path = ["C_s8_up", "C_s8_up2", "C_fid", "C_s8_dw", "C_s8_dw2"]
		elif self.paramo == 8:
		    self.gamma = np.array([self.gamma[0]*(1.+self.eps_gamma), self.gamma[1]*(1.+2*self.eps_gamma), self.gamma[2], self.gamma[3]*(1.-self.eps_gamma), self.gamma[4]*(1.-2*self.eps_gamma)])
		    self.CC_path = ["C_gamma_up", "C_gamma_up2", "C_fid", "C_gamma_dw", "C_gamma_dw2"]
		elif self.paramo == 9:
		    self.A_IA = np.array([self.A_IA[0]*(1.+self.eps_A_IA), self.A_IA[1]*(1.+2*self.eps_A_IA), self.A_IA[2], self.A_IA[3]*(1.-self.eps_A_IA), self.A_IA[4]*(1.-2*self.eps_A_IA)])
		    self.CC_path = ["C_A_IA_up", "C_A_IA_up2", "C_fid", "C_A_IA_dw", "C_A_IA_dw2"]
		elif self.paramo == 10:
		    self.n_IA = np.array([self.n_IA[0]*(1.+self.eps_n_IA), self.n_IA[1]*(1.+2*self.eps_n_IA), self.n_IA[2], self.n_IA[3]*(1.-self.eps_n_IA), self.n_IA[4]*(1.-2*self.eps_n_IA)])
		    self.CC_path = ["C_n_IA_up", "C_n_IA_up2", "C_fid", "C_n_IA_dw", "C_n_IA_dw2"]
		elif self.paramo == 11:
		    self.B_IA = np.array([self.B_IA[0]*(1.+self.eps_B_IA), self.B_IA[1]*(1.+2*self.eps_B_IA), self.B_IA[2], self.B_IA[3]*(1.-self.eps_B_IA), self.B_IA[4]*(1.-2*self.eps_B_IA)])
		    self.CC_path = ["C_B_IA_up", "C_B_IA_up2", "C_fid", "C_B_IA_dw", "C_B_IA_dw2"]
		elif self.paramo > 11:
		    self.CC_path = ["C_b"+str(self.paramo-11)+"_up", "C_b"+str(self.paramo-11)+"_up2", "C_fid", "C_b"+str(self.paramo-11)+"_dw", "C_b"+str(self.paramo-11)+"_dw2"]
		    self.BX[self.paramo-12] = [self.BX[self.paramo-12][0]*(1.+self.eps_b), self.BX[self.paramo-12][1]*(1.+2*self.eps_b), self.BX[self.paramo-12][2], self.BX[self.paramo-12][3]*(1.-self.eps_b), self.BX[self.paramo-12][4]*(1.-2*self.eps_b)]

		self.Pk_files = []
		for i in range(self.der_B):
			self.Pk_files.append(self.fold_path[i]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat")

	def Initializers_Pk(self):

		self.P_dd_C = np.genfromtxt(self.Pk_files[2], skip_header=3, unpack=True)
		self.P_dd_C_up = np.genfromtxt(self.Pk_files[0], skip_header=3, unpack=True)
		self.P_dd_C_dw = np.genfromtxt(self.Pk_files[3], skip_header=3, unpack=True)

		self.P_dd_C[:][0] = self.P_dd_C[:][0]*self.h[2]
		self.P_dd_C_up[:][0] = self.P_dd_C_up[:][0]*self.h[0]
		self.P_dd_C_dw[:][0] = self.P_dd_C_dw[:][0]*self.h[3]

		self.P_dd_C[:][1] = self.P_dd_C[:][1]/(self.h[2]**3)
		self.P_dd_C_up[:][1] = self.P_dd_C_up[:][1]/(self.h[0]**3)
		self.P_dd_C_dw[:][1] = self.P_dd_C_dw[:][1]/(self.h[3]**3)

		for i in range(1, len(self.P_dd_C[0])):
			if(self.P_dd_C[0][i] < self.P_dd_C[0][i-1]):
				self.Pk_size = i
				break

		for i in range(1, len(self.P_dd_C_up[0])):
			if(self.P_dd_C_up[0][i] < self.P_dd_C_up[0][i-1]):
				self.Pk_size_up = i
				break

		for i in range(1, len(self.P_dd_C_dw[0])):
			if(self.P_dd_C_dw[0][i] < self.P_dd_C_dw[0][i-1]):
				self.Pk_size_dw = i
				break

	def background(self):
		for i in range(len(self.z_win)):
			self.E_tab[i] = E(self.z_win[i], self.h[2], self.Omega_m[2], self.Omega_DE[2], self.w0[2], self.wa[2], self.curvature)
			self.R_tab[i] = chi_R(self.z_win[i], self.h[2], self.Omega_m[2], self.Omega_DE[2], self.w0[2], self.wa[2], self.c, self.curvature)
			self.DG_tab[i] = DGrowth(self.z_win[i], self.h[2], self.Omega_m[2], self.Omega_DE[2], self.w0[2], self.wa[2], self.gamma[2], self.curvature)
			self.RT_tab[i] = self.R_tab[i]/(self.c/(100*self.h[2]))

			self.E_tab_up[i] = E(self.z_win[i], self.h[0], self.Omega_m[0], self.Omega_DE[0], self.w0[0], self.wa[0], self.curvature)
			self.R_tab_up[i] = chi_R(self.z_win[i], self.h[0], self.Omega_m[0], self.Omega_DE[0], self.w0[0], self.wa[0], self.c, self.curvature)
			self.DG_tab_up[i] = DGrowth(self.z_win[i], self.h[0], self.Omega_m[0], self.Omega_DE[0], self.w0[0], self.wa[0], self.gamma[0], self.curvature)
			self.RT_tab_up[i] = self.R_tab_up[i]/(self.c/(100*self.h[0]))

			self.E_tab_dw[i] = E(self.z_win[i], self.h[3], self.Omega_m[3], self.Omega_DE[3], self.w0[3], self.wa[3], self.curvature)
			self.R_tab_dw[i] = chi_R(self.z_win[i], self.h[3], self.Omega_m[3], self.Omega_DE[3], self.w0[3], self.wa[3], self.c, self.curvature)
			self.DG_tab_dw[i] = DGrowth(self.z_win[i], self.h[3], self.Omega_m[3], self.Omega_DE[3], self.w0[3], self.wa[3], self.gamma[3], self.curvature)
			self.RT_tab_dw[i] = self.R_tab_dw[i]/(self.c/(100*self.h[3]))

	def nz(self, z):
		return (z/self.zrange_0)**2*np.exp(-(z/self.zrange_0)**1.5)

	def P_phot(self, z, zp):
		return (1. -  self.f_out)/(np.sqrt(2.*np.pi)*self.sig_b*(1+z))*np.exp(-0.5*((z - self.cb*zp - self.zb)/(self.sig_b*(1.+z)))**2) + self.f_out/(np.sqrt(2.*np.pi)*self.sig_0*(1+z))*np.exp(-0.5*((z - self.c0*zp - self.zo)/(self.sig_0*(1.+z)))**2)

	def photoz(self):
		z_p, z_m, zp, I_prec = [], [], [[]], 500
		Trap, Trap_Z = 0.0, 0.0
		delta_zp, TZ = [], [[]]
		PZ_out = open("Pz_Table.txt", "w")
		for i in range(len(self.z_win)):
			print(i)
			for j in range(len(self.zrange)):
				Trap, Trap_Z = 0.0, 0.0
				z_p, z_m = self.z_ip[j], self.z_im[j]
				if(i==0):
					if(j != 0):
						zp.append([[]])
					zp[j] = np.linspace(z_m, z_p, I_prec)
					delta_zp.append(zp[j][1]-zp[j][0])

				Trap = delta_zp[j]/90*self.nz(self.z_win[i])*np.sum(7.*(self.P_phot(self.z_win[i], zp[j][1:])+(self.P_phot(self.z_win[i], zp[j][:-1]))) + 32.*(self.P_phot(self.z_win[i], 1./4*(3.*zp[j][1:]+zp[j][:-1])) + self.P_phot(self.z_win[i], 1./4*(zp[j][1:]+3.*zp[j][:-1]))) + 12*(self.P_phot(self.z_win[i], 1./2*(zp[j][1:]+zp[j][:-1]))))
				
				Z_W, Z_P = np.meshgrid(self.z_win, zp[j])
				TZ = self.P_phot(Z_W, Z_P)*self.nz(Z_W)

				TTZ = np.zeros(I_prec)
				for k in range(I_prec):
					TTZ[k] = delta_zp[j]/90*np.sum(7*(TZ[k][1:] + TZ[k][:-1]) + 32*((TZ[k][:-1] + (1./4*(3*self.z_win[1:]+self.z_win[:-1]) - self.z_win[:-1])*(TZ[k][1:]-TZ[k][:-1])/(self.z_win[1:]-self.z_win[:-1])) + (TZ[k][:-1] + (1./4*(self.z_win[1:]+3*self.z_win[:-1]) - self.z_win[:-1])*(TZ[k][1:]-TZ[k][:-1])/(self.z_win[1:]-self.z_win[:-1]))) + 12*(TZ[k][:-1] + (1./2*(self.z_win[1:]+self.z_win[:-1]) - self.z_win[:-1])*(TZ[k][1:]-TZ[k][:-1])/(self.z_win[1:]-self.z_win[:-1])) )

				Trap_Z = (self.z_win[1]-self.z_win[0])/90*np.sum(7*(TTZ[1:] + TTZ[:-1]) + 32*( ((TTZ[:-1] + (1./4*(3*zp[j][1:]+zp[j][:-1]) - zp[j][:-1])*(TTZ[1:]-TTZ[:-1])/(zp[j][1:]-zp[j][:-1]))) + ((TTZ[:-1] + (1./4*(zp[j][1:]+3*zp[j][:-1]) - zp[j][:-1])*(TTZ[1:]-TTZ[:-1])/(zp[j][1:]-zp[j][:-1])))) + 12*(TTZ[:-1] + (1./2*(zp[j][1:]+zp[j][:-1]) - zp[j][:-1])*(TTZ[1:]-TTZ[:-1])/(zp[j][1:]-zp[j][:-1])))
				self.Photoz_tab[i][j] = Trap/Trap_Z;
				PZ_out.write(str("%.16e" % self.Photoz_tab[i][j]))
				PZ_out.write(str(' '))
			PZ_out.write(str('\n'))
		PZ_out.close()

	def photoz_load(self):
		self.Photoz_tab = np.loadtxt("Pz_Table_10000.txt")

	def windows(self):
		BX_W, BX_W_up, BX_W_dw, i_s = np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win)), np.zeros(len(self.z_win))
		delta_zwin = self.z_win[4]-self.z_win[0]
		for i in range(len(self.z_win)):
			for j in range(len(self.z_ip)):
				if (self.z_win[i] <= self.z_ip[j] and self.z_win[i] >= self.z_im[j]):
					i_s[i] = j
					BX_W[i] = self.BX[int(i_s[i])][2]
					BX_W_up[i] = self.BX[int(i_s[i])][0]
					BX_W_dw[i] = self.BX[int(i_s[i])][3]
				if(self.paramo == 0):
					self.W_tab[i][j] = 1.5*100.*self.h[2]*self.Omega_m[2]*(1.+self.z_win[i])/self.c*self.RT_tab[i]/90*delta_zwin*np.sum(7.*(self.Photoz_tab[0:-1:4,j]*(1. - self.RT_tab[i]/self.RT_tab[0:-1:4]) + self.Photoz_tab[4::4,j]*(1. - self.RT_tab[i]/self.RT_tab[4::4])) + 32.*(self.Photoz_tab[3::4,j]*(1. - self.RT_tab[i]/self.RT_tab[3::4]) + self.Photoz_tab[1::4,j]*(1. - self.RT_tab[i]/self.RT_tab[1::4])) + 12.*(self.Photoz_tab[2::4,j]*(1. - self.RT_tab[i]/self.RT_tab[2::4])))
				self.W_tab_up[i][j] = 1.5*100.*self.h[0]*self.Omega_m[0]*(1.+self.z_win[i])/self.c*self.RT_tab_up[i]/90*delta_zwin*np.sum(7.*(self.Photoz_tab[0:-1:4,j]*(1. - self.RT_tab_up[i]/self.RT_tab_up[0:-1:4]) + self.Photoz_tab[4::4,j]*(1. - self.RT_tab_up[i]/self.RT_tab_up[4::4])) + 32.*(self.Photoz_tab[3::4,j]*(1. - self.RT_tab_up[i]/self.RT_tab_up[3::4]) + self.Photoz_tab[1::4,j]*(1. - self.RT_tab_up[i]/self.RT_tab_up[1::4])) + 12.*(self.Photoz_tab[2::4,j]*(1. - self.RT_tab_up[i]/self.RT_tab_up[2::4])))
				self.W_tab_dw[i][j] = 1.5*100.*self.h[3]*self.Omega_m[3]*(1.+self.z_win[i])/self.c*self.RT_tab_dw[i]/90*delta_zwin*np.sum(7.*(self.Photoz_tab[0:-1:4,j]*(1. - self.RT_tab_dw[i]/self.RT_tab_dw[0:-1:4]) + self.Photoz_tab[4::4,j]*(1. - self.RT_tab_dw[i]/self.RT_tab_dw[4::4])) + 32.*(self.Photoz_tab[3::4,j]*(1. - self.RT_tab_dw[i]/self.RT_tab_dw[3::4]) + self.Photoz_tab[1::4,j]*(1. - self.RT_tab_dw[i]/self.RT_tab_dw[1::4])) + 12.*(self.Photoz_tab[2::4,j]*(1. - self.RT_tab_dw[i]/self.RT_tab_dw[2::4])))

		E_tab_W = np.array([self.E_tab]*len(self.zrange)).transpose()
		E_tab_up_W = np.array([self.E_tab_up]*len(self.zrange)).transpose()
		E_tab_dw_W = np.array([self.E_tab_dw]*len(self.zrange)).transpose()

		i_s = np.array([i_s]*len(self.zrange)).transpose()

		BX_W = np.array([BX_W]*len(self.zrange)).transpose()
		BX_W_up = np.array([BX_W_up]*len(self.zrange)).transpose()
		BX_W_dw = np.array([BX_W_dw]*len(self.zrange)).transpose()

		if(self.paramo == 0):
			self.WG_tab = 100.*self.h[2]/self.c*self.Photoz_tab*E_tab_W*BX_W
			self.WIA_tab = 100.*self.h[2]/self.c*self.Photoz_tab*E_tab_W
		self.WG_tab_up = 100.*self.h[0]/self.c*self.Photoz_tab*E_tab_up_W*BX_W_up
		self.WG_tab_dw = 100.*self.h[3]/self.c*self.Photoz_tab*E_tab_dw_W*BX_W_dw
		self.WIA_tab_up = 100.*self.h[0]/self.c*self.Photoz_tab*E_tab_up_W
		self.WIA_tab_dw = 100.*self.h[3]/self.c*self.Photoz_tab*E_tab_dw_W

	def P_shot_WL(self, zpi, zpj):
		if(zpi == zpj):
			return self.sig_epsilon**2/(self.ng/len(self.zrange))
		else:
			return 0

	def P_shot_GC(self, zpi, zpj):
		if(zpi == zpj):
			return 1./(self.ng/len(self.zrange))
		else:
			return 0

	def C_l_computing(self, i, seed=None):
		Pk_conv, Pk_conv_up, Pk_conv_dw = [], [], []
		k_conv = 0.0
		delta_zpm = self.z_win[4]-self.z_win[0]

		if(self.paramo == 0):
			outGG = open(self.pre_CC_path[0]+self.CC_path[2]+"/COVAR_fid_"+str(self.l[i]), "w")
			outLL = open(self.pre_CC_path[1]+self.CC_path[2]+"/COVAR_fid_"+str(self.l[i]), "w")
			outGL = open(self.pre_CC_path[2]+self.CC_path[2]+"/COVAR_fid_"+str(self.l[i]), "w")

		outGGU = open(self.pre_CC_path[0]+self.CC_path[0]+"/COVAR_up_"+str(self.l[i]), "w");
		outGGD = open(self.pre_CC_path[0]+self.CC_path[3]+"/COVAR_dw_"+str(self.l[i]), "w");
		outLLU = open(self.pre_CC_path[1]+self.CC_path[0]+"/COVAR_up_"+str(self.l[i]), "w");
		outLLD = open(self.pre_CC_path[1]+self.CC_path[3]+"/COVAR_dw_"+str(self.l[i]), "w");
		outGLU = open(self.pre_CC_path[2]+self.CC_path[0]+"/COVAR_up_"+str(self.l[i]), "w");
		outGLD = open(self.pre_CC_path[2]+self.CC_path[3]+"/COVAR_dw_"+str(self.l[i]), "w");

		k_conv = (self.l[i]+0.5)/self.R_tab_dw
		for m in range(len(self.z_win)):
			Pk_conv_dw = np.append(Pk_conv_dw, CubicSpline(self.P_dd_C_dw[0][m*self.Pk_size_dw:(m+1)*self.Pk_size_dw], self.P_dd_C_dw[1][m*self.Pk_size_dw:(m+1)*self.Pk_size_dw]))
			if ((k_conv[m] < 35 and k_conv[m] >  0.0005)):
				if(self.paramo == 8):
					Pk_conv_dw[m] =  Pk_conv_dw[m](k_conv[m])*(self.DG_tab_dw[m]/self.DG_tab[m])**2
				else:
					Pk_conv_dw[m] =  Pk_conv_dw[m](k_conv[m])
			else:
				Pk_conv_dw[m] =  0.0

		k_conv = (self.l[i]+0.5)/self.R_tab_up
		for m in range(len(self.z_win)):
			Pk_conv_up = np.append(Pk_conv_up, CubicSpline(self.P_dd_C_up[0][m*self.Pk_size_up:(m+1)*self.Pk_size_up], self.P_dd_C_up[1][m*self.Pk_size_up:(m+1)*self.Pk_size_up]))
			if ((k_conv[m] < 35 and k_conv[m] >  0.0005)):
				if(self.paramo == 8):
					Pk_conv_up[m] =  Pk_conv_up[m](k_conv[m])*(self.DG_tab_up[m]/self.DG_tab[m])**2
				else:
					Pk_conv_up[m] =  Pk_conv_up[m](k_conv[m])
			else:
				Pk_conv_up[m] =  0.0

		if(self.paramo == 0):
			k_conv = (self.l[i]+0.5)/self.R_tab
			for m in range(len(self.z_win)):
				Pk_conv = np.append(Pk_conv, CubicSpline(self.P_dd_C[0][m*self.Pk_size:(m+1)*self.Pk_size], self.P_dd_C[1][m*self.Pk_size:(m+1)*self.Pk_size]))
				if ((k_conv[m] < 35 and k_conv[m] >  0.0005)):
					Pk_conv[m] =  Pk_conv[m](k_conv[m])
				else:
					Pk_conv[m] =  0.0

		for aa in range(len(self.zrange)):
			for bb in range(len(self.zrange)):
				if(aa >= bb):
					if(self.paramo != 9 or self.paramo != 10 or self.paramo != 11):
						if(self.paramo == 0):
							self.C_ij_GG[aa][bb] = self.c/(100.*self.h[2])*delta_zpm/90*np.sum(7.*(self.WG_tab[0:-1:4,aa]*self.WG_tab[0:-1:4,bb]/(self.E_tab[0:-1:4]*pow(self.R_tab[0:-1:4],2))*Pk_conv[0:-1:4] + self.WG_tab[4::4,aa]*self.WG_tab[4::4,bb]/(self.E_tab[4::4]*pow(self.R_tab[4::4],2))*Pk_conv[4::4]) + 32.*(self.WG_tab[1::4,aa]*self.WG_tab[1::4,bb]/(self.E_tab[1::4]*pow(self.R_tab[1::4],2))*Pk_conv[1::4] + self.WG_tab[3::4,aa]*self.WG_tab[3::4,bb]/(self.E_tab[3::4]*pow(self.R_tab[3::4],2))*Pk_conv[3::4]) + 12.*(self.WG_tab[2::4,aa]*self.WG_tab[2::4,bb]/(self.E_tab[2::4]*pow(self.R_tab[2::4],2))*Pk_conv[2::4])) + self.P_shot_GC(self.zrange[aa], self.zrange[bb])
							if(aa != bb):
								self.C_ij_GG[bb][aa] = self.C_ij_GG[aa][bb]
						
						self.C_ij_GG_up[aa][bb] = self.c/(100.*self.h[0])*delta_zpm/90*np.sum(7.*(self.WG_tab_up[0:-1:4,aa]*self.WG_tab_up[0:-1:4,bb]/(self.E_tab_up[0:-1:4]*pow(self.R_tab_up[0:-1:4],2))*Pk_conv_up[0:-1:4] + self.WG_tab_up[4::4,aa]*self.WG_tab_up[4::4,bb]/(self.E_tab_up[4::4]*pow(self.R_tab_up[4::4],2))*Pk_conv_up[4::4]) + 32.*(self.WG_tab_up[1::4,aa]*self.WG_tab_up[1::4,bb]/(self.E_tab_up[1::4]*pow(self.R_tab_up[1::4],2))*Pk_conv_up[1::4] + self.WG_tab_up[3::4,aa]*self.WG_tab_up[3::4,bb]/(self.E_tab_up[3::4]*pow(self.R_tab_up[3::4],2))*Pk_conv_up[3::4]) + 12.*(self.WG_tab_up[2::4,aa]*self.WG_tab_up[2::4,bb]/(self.E_tab_up[2::4]*pow(self.R_tab_up[2::4],2))*Pk_conv_up[2::4])) + self.P_shot_GC(self.zrange[aa], self.zrange[bb])
						self.C_ij_GG_dw[aa][bb] = self.c/(100.*self.h[3])*delta_zpm/90*np.sum(7.*(self.WG_tab_dw[0:-1:4,aa]*self.WG_tab_dw[0:-1:4,bb]/(self.E_tab_dw[0:-1:4]*pow(self.R_tab_dw[0:-1:4],2))*Pk_conv_dw[0:-1:4] + self.WG_tab_dw[4::4,aa]*self.WG_tab_dw[4::4,bb]/(self.E_tab_dw[4::4]*pow(self.R_tab_dw[4::4],2))*Pk_conv_dw[4::4]) + 32.*(self.WG_tab_dw[1::4,aa]*self.WG_tab_dw[1::4,bb]/(self.E_tab_dw[1::4]*pow(self.R_tab_dw[1::4],2))*Pk_conv_dw[1::4] + self.WG_tab_dw[3::4,aa]*self.WG_tab_dw[3::4,bb]/(self.E_tab_dw[3::4]*pow(self.R_tab_dw[3::4],2))*Pk_conv_dw[3::4]) + 12.*(self.WG_tab_dw[2::4,aa]*self.WG_tab_dw[2::4,bb]/(self.E_tab_dw[2::4]*pow(self.R_tab_dw[2::4],2))*Pk_conv_dw[2::4])) + self.P_shot_GC(self.zrange[aa], self.zrange[bb])

						if(aa != bb):
							self.C_ij_GG_up[bb][aa] = self.C_ij_GG_up[aa][bb]
							self.C_ij_GG_dw[bb][aa] = self.C_ij_GG_dw[aa][bb]
					else:
						self.C_ij_GG[aa][bb], self.C_ij_GG_up[aa][bb], self.C_ij_GG_dw[aa][bb] = 0.0, 0.0, 0.0
						self.C_ij_GG[bb][aa], self.C_ij_GG_up[bb][aa], self.C_ij_GG_dw[bb][aa] = 0.0, 0.0, 0.0

					if(self.paramo < 12):
						if(self.paramo == 0):
							self.C_ij_LL[aa][bb] = (self.c/(100.*self.h[2])*delta_zpm/90*np.sum(7.*(self.W_tab[0:-1:4,aa]*self.W_tab[0:-1:4,bb]/(self.E_tab[0:-1:4]*pow(self.R_tab[0:-1:4],2))*Pk_conv[0:-1:4] + self.W_tab[4::4,aa]*self.W_tab[4::4,bb]/(self.E_tab[4::4]*pow(self.R_tab[4::4],2))*Pk_conv[4::4]) + 32.*(self.W_tab[1::4,aa]*self.W_tab[1::4,bb]/(self.E_tab[1::4]*pow(self.R_tab[1::4],2))*Pk_conv[1::4] + self.W_tab[3::4,aa]*self.W_tab[3::4,bb]/(self.E_tab[3::4]*pow(self.R_tab[3::4],2))*Pk_conv[3::4]) + 12.*(self.W_tab[2::4,aa]*self.W_tab[2::4,bb]/(self.E_tab[2::4]*pow(self.R_tab[2::4],2))*Pk_conv[2::4])
							+ 7.*((self.W_tab[0:-1:4,aa]*self.WIA_tab[0:-1:4,bb] + self.WIA_tab[0:-1:4,aa]*self.W_tab[0:-1:4,bb])/(self.E_tab[0:-1:4]*pow(self.R_tab[0:-1:4],2))*(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[2])*pow(self.lum_mean[0:-1:4],self.B_IA[2])*Pk_conv[0:-1:4])
							+ (self.W_tab[4::4,aa]*self.WIA_tab[4::4,bb] + self.WIA_tab[4::4,aa]*self.W_tab[4::4,bb])/(self.E_tab[4::4]*pow(self.R_tab[4::4],2))*(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[4::4]*pow(1.+self.z_win[4::4],self.n_IA[2])*pow(self.lum_mean[4::4],self.B_IA[2])*Pk_conv[4::4]))
							+ 32.*((self.W_tab[1::4,aa]*self.WIA_tab[1::4,bb] + self.WIA_tab[1::4,aa]*self.W_tab[1::4,bb])/(self.E_tab[1::4]*pow(self.R_tab[1::4],2))*(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[1::4]*pow(1.+self.z_win[1::4],self.n_IA[2])*pow(self.lum_mean[1::4],self.B_IA[2])*Pk_conv[1::4])
							+ (self.W_tab[3::4,aa]*self.WIA_tab[3::4,bb] + self.WIA_tab[3::4,aa]*self.W_tab[3::4,bb])/(self.E_tab[3::4]*pow(self.R_tab[3::4],2))*(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[3::4]*pow(1.+self.z_win[3::4],self.n_IA[2])*pow(self.lum_mean[3::4],self.B_IA[2])*Pk_conv[3::4]))
							+ 12.*((self.W_tab[2::4,aa]*self.WIA_tab[2::4,bb] + self.WIA_tab[2::4,aa]*self.W_tab[2::4,bb])/(self.E_tab[2::4]*pow(self.R_tab[2::4],2))*(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[2::4]*pow(1.+self.z_win[2::4],self.n_IA[2])*pow(self.lum_mean[2::4],self.B_IA[2])*Pk_conv[2::4]))
							+ 7.*(self.WIA_tab[0:-1:4,aa]*self.WIA_tab[0:-1:4,bb]/(self.E_tab[0:-1:4]*pow(self.R_tab[0:-1:4],2))*pow(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[2])*pow(self.lum_mean[0:-1:4],self.B_IA[2]),2)*Pk_conv[0:-1:4]
							+ self.WIA_tab[4::4,aa]*self.WIA_tab[4::4,bb]/(self.E_tab[4::4]*pow(self.R_tab[4::4],2))*pow(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[4::4]*pow(1.+self.z_win[4::4],self.n_IA[2])*pow(self.lum_mean[4::4],self.B_IA[2]),2)*Pk_conv[4::4])
							+ 32.*(self.WIA_tab[1::4,aa]*self.WIA_tab[1::4,bb]/(self.E_tab[1::4]*pow(self.R_tab[1::4],2))*pow(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[1::4]*pow(1.+self.z_win[1::4],self.n_IA[2])*pow(self.lum_mean[1::4],self.B_IA[2]),2)*Pk_conv[1::4]
							+ self.WIA_tab[3::4,aa]*self.WIA_tab[3::4,bb]/(self.E_tab[3::4]*pow(self.R_tab[3::4],2))*pow(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[3::4]*pow(1.+self.z_win[3::4],self.n_IA[2])*pow(self.lum_mean[3::4],self.B_IA[2]),2)*Pk_conv[3::4])
							+ 12.*(self.WIA_tab[2::4,aa]*self.WIA_tab[2::4,bb]/(self.E_tab[2::4]*pow(self.R_tab[2::4],2))*pow(-self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[2::4]*pow(1.+self.z_win[2::4],self.n_IA[2])*pow(self.lum_mean[2::4],self.B_IA[2]),2)*Pk_conv[2::4]))
							+ self.P_shot_WL(self.zrange[aa], self.zrange[bb]))

							self.C_ij_LL[bb][aa] = self.C_ij_LL[aa][bb]

						self.C_ij_LL_up[aa][bb] = (self.c/(100.*self.h[0])*delta_zpm/90*np.sum(7.*(self.W_tab_up[0:-1:4,aa]*self.W_tab_up[0:-1:4,bb]/(self.E_tab_up[0:-1:4]*pow(self.R_tab_up[0:-1:4],2))*Pk_conv_up[0:-1:4] + self.W_tab_up[4::4,aa]*self.W_tab_up[4::4,bb]/(self.E_tab_up[4::4]*pow(self.R_tab_up[4::4],2))*Pk_conv_up[4::4]) + 32.*(self.W_tab_up[1::4,aa]*self.W_tab_up[1::4,bb]/(self.E_tab_up[1::4]*pow(self.R_tab_up[1::4],2))*Pk_conv_up[1::4] + self.W_tab_up[3::4,aa]*self.W_tab_up[3::4,bb]/(self.E_tab_up[3::4]*pow(self.R_tab_up[3::4],2))*Pk_conv_up[3::4]) + 12.*(self.W_tab_up[2::4,aa]*self.W_tab_up[2::4,bb]/(self.E_tab_up[2::4]*pow(self.R_tab_up[2::4],2))*Pk_conv_up[2::4])
						+ 7.*((self.W_tab_up[0:-1:4,aa]*self.WIA_tab_up[0:-1:4,bb] + self.WIA_tab_up[0:-1:4,aa]*self.W_tab_up[0:-1:4,bb])/(self.E_tab_up[0:-1:4]*pow(self.R_tab_up[0:-1:4],2))*(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[0])*pow(self.lum_mean[0:-1:4],self.B_IA[0])*Pk_conv_up[0:-1:4])
						+ (self.W_tab_up[4::4,aa]*self.WIA_tab_up[4::4,bb] + self.WIA_tab_up[4::4,aa]*self.W_tab_up[4::4,bb])/(self.E_tab_up[4::4]*pow(self.R_tab_up[4::4],2))*(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[4::4]*pow(1.+self.z_win[4::4],self.n_IA[0])*pow(self.lum_mean[4::4],self.B_IA[0])*Pk_conv_up[4::4]))
						+ 32.*((self.W_tab_up[1::4,aa]*self.WIA_tab_up[1::4,bb] + self.WIA_tab_up[1::4,aa]*self.W_tab_up[1::4,bb])/(self.E_tab_up[1::4]*pow(self.R_tab_up[1::4],2))*(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[1::4]*pow(1.+self.z_win[1::4],self.n_IA[0])*pow(self.lum_mean[1::4],self.B_IA[0])*Pk_conv_up[1::4])
						+ (self.W_tab_up[3::4,aa]*self.WIA_tab_up[3::4,bb] + self.WIA_tab_up[3::4,aa]*self.W_tab_up[3::4,bb])/(self.E_tab_up[3::4]*pow(self.R_tab_up[3::4],2))*(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[3::4]*pow(1.+self.z_win[3::4],self.n_IA[0])*pow(self.lum_mean[3::4],self.B_IA[0])*Pk_conv_up[3::4]))
						+ 12.*((self.W_tab_up[2::4,aa]*self.WIA_tab_up[2::4,bb] + self.WIA_tab_up[2::4,aa]*self.W_tab_up[2::4,bb])/(self.E_tab_up[2::4]*pow(self.R_tab_up[2::4],2))*(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[2::4]*pow(1.+self.z_win[2::4],self.n_IA[0])*pow(self.lum_mean[2::4],self.B_IA[0])*Pk_conv_up[2::4]))
						+ 7.*(self.WIA_tab_up[0:-1:4,aa]*self.WIA_tab_up[0:-1:4,bb]/(self.E_tab_up[0:-1:4]*pow(self.R_tab_up[0:-1:4],2))*pow(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[0])*pow(self.lum_mean[0:-1:4],self.B_IA[0]),2)*Pk_conv_up[0:-1:4]
						+ self.WIA_tab_up[4::4,aa]*self.WIA_tab_up[4::4,bb]/(self.E_tab_up[4::4]*pow(self.R_tab_up[4::4],2))*pow(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[4::4]*pow(1.+self.z_win[4::4],self.n_IA[0])*pow(self.lum_mean[4::4],self.B_IA[0]),2)*Pk_conv_up[4::4])
						+ 32.*(self.WIA_tab_up[1::4,aa]*self.WIA_tab_up[1::4,bb]/(self.E_tab_up[1::4]*pow(self.R_tab_up[1::4],2))*pow(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[1::4]*pow(1.+self.z_win[1::4],self.n_IA[0])*pow(self.lum_mean[1::4],self.B_IA[0]),2)*Pk_conv_up[1::4]
						+ self.WIA_tab_up[3::4,aa]*self.WIA_tab_up[3::4,bb]/(self.E_tab_up[3::4]*pow(self.R_tab_up[3::4],2))*pow(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[3::4]*pow(1.+self.z_win[3::4],self.n_IA[0])*pow(self.lum_mean[3::4],self.B_IA[0]),2)*Pk_conv_up[3::4])
						+ 12.*(self.WIA_tab_up[2::4,aa]*self.WIA_tab_up[2::4,bb]/(self.E_tab_up[2::4]*pow(self.R_tab_up[2::4],2))*pow(-self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[2::4]*pow(1.+self.z_win[2::4],self.n_IA[0])*pow(self.lum_mean[2::4],self.B_IA[0]),2)*Pk_conv_up[2::4]))
						+ self.P_shot_WL(self.zrange[aa], self.zrange[bb]))

						self.C_ij_LL_dw[aa][bb] = (self.c/(100.*self.h[3])*delta_zpm/90*np.sum(7.*(self.W_tab_dw[0:-1:4,aa]*self.W_tab_dw[0:-1:4,bb]/(self.E_tab_dw[0:-1:4]*pow(self.R_tab_dw[0:-1:4],2))*Pk_conv_dw[0:-1:4] + self.W_tab_dw[4::4,aa]*self.W_tab_dw[4::4,bb]/(self.E_tab_dw[4::4]*pow(self.R_tab_dw[4::4],2))*Pk_conv_dw[4::4]) + 32.*(self.W_tab_dw[1::4,aa]*self.W_tab_dw[1::4,bb]/(self.E_tab_dw[1::4]*pow(self.R_tab_dw[1::4],2))*Pk_conv_dw[1::4] + self.W_tab_dw[3::4,aa]*self.W_tab_dw[3::4,bb]/(self.E_tab_dw[3::4]*pow(self.R_tab_dw[3::4],2))*Pk_conv_dw[3::4]) + 12.*(self.W_tab_dw[2::4,aa]*self.W_tab_dw[2::4,bb]/(self.E_tab_dw[2::4]*pow(self.R_tab_dw[2::4],2))*Pk_conv_dw[2::4])
						+ 7.*((self.W_tab_dw[0:-1:4,aa]*self.WIA_tab_dw[0:-1:4,bb] + self.WIA_tab_dw[0:-1:4,aa]*self.W_tab_dw[0:-1:4,bb])/(self.E_tab_dw[0:-1:4]*pow(self.R_tab_dw[0:-1:4],2))*(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[3])*pow(self.lum_mean[0:-1:4],self.B_IA[3])*Pk_conv_dw[0:-1:4])
						+ (self.W_tab_dw[4::4,aa]*self.WIA_tab_dw[4::4,bb] + self.WIA_tab_dw[4::4,aa]*self.W_tab_dw[4::4,bb])/(self.E_tab_dw[4::4]*pow(self.R_tab_dw[4::4],2))*(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[4::4]*pow(1.+self.z_win[4::4],self.n_IA[3])*pow(self.lum_mean[4::4],self.B_IA[3])*Pk_conv_dw[4::4]))
						+ 32.*((self.W_tab_dw[1::4,aa]*self.WIA_tab_dw[1::4,bb] + self.WIA_tab_dw[1::4,aa]*self.W_tab_dw[1::4,bb])/(self.E_tab_dw[1::4]*pow(self.R_tab_dw[1::4],2))*(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[1::4]*pow(1.+self.z_win[1::4],self.n_IA[3])*pow(self.lum_mean[1::4],self.B_IA[3])*Pk_conv_dw[1::4])
						+ (self.W_tab_dw[3::4,aa]*self.WIA_tab_dw[3::4,bb] + self.WIA_tab_dw[3::4,aa]*self.W_tab_dw[3::4,bb])/(self.E_tab_dw[3::4]*pow(self.R_tab_dw[3::4],2))*(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[3::4]*pow(1.+self.z_win[3::4],self.n_IA[3])*pow(self.lum_mean[3::4],self.B_IA[3])*Pk_conv_dw[3::4]))
						+ 12.*((self.W_tab_dw[2::4,aa]*self.WIA_tab_dw[2::4,bb] + self.WIA_tab_dw[2::4,aa]*self.W_tab_dw[2::4,bb])/(self.E_tab_dw[2::4]*pow(self.R_tab_dw[2::4],2))*(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[2::4]*pow(1.+self.z_win[2::4],self.n_IA[3])*pow(self.lum_mean[2::4],self.B_IA[3])*Pk_conv_dw[2::4]))
						+ 7.*(self.WIA_tab_dw[0:-1:4,aa]*self.WIA_tab_dw[0:-1:4,bb]/(self.E_tab_dw[0:-1:4]*pow(self.R_tab_dw[0:-1:4],2))*pow(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[3])*pow(self.lum_mean[0:-1:4],self.B_IA[3]),2)*Pk_conv_dw[0:-1:4]
						+ self.WIA_tab_dw[4::4,aa]*self.WIA_tab_dw[4::4,bb]/(self.E_tab_dw[4::4]*pow(self.R_tab_dw[4::4],2))*pow(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[4::4]*pow(1.+self.z_win[4::4],self.n_IA[3])*pow(self.lum_mean[4::4],self.B_IA[3]),2)*Pk_conv_dw[4::4])
						+ 32.*(self.WIA_tab_dw[1::4,aa]*self.WIA_tab_dw[1::4,bb]/(self.E_tab_dw[1::4]*pow(self.R_tab_dw[1::4],2))*pow(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[1::4]*pow(1.+self.z_win[1::4],self.n_IA[3])*pow(self.lum_mean[1::4],self.B_IA[3]),2)*Pk_conv_dw[1::4]
						+ self.WIA_tab_dw[3::4,aa]*self.WIA_tab_dw[3::4,bb]/(self.E_tab_dw[3::4]*pow(self.R_tab_dw[3::4],2))*pow(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[3::4]*pow(1.+self.z_win[3::4],self.n_IA[3])*pow(self.lum_mean[3::4],self.B_IA[3]),2)*Pk_conv_dw[3::4])
						+ 12.*(self.WIA_tab_dw[2::4,aa]*self.WIA_tab_dw[2::4,bb]/(self.E_tab_dw[2::4]*pow(self.R_tab_dw[2::4],2))*pow(-self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[2::4]*pow(1.+self.z_win[2::4],self.n_IA[3])*pow(self.lum_mean[2::4],self.B_IA[3]),2)*Pk_conv_dw[2::4]))
						+ self.P_shot_WL(self.zrange[aa], self.zrange[bb]))

						self.C_ij_LL_up[bb][aa] = self.C_ij_LL_up[aa][bb]
						self.C_ij_LL_dw[bb][aa] = self.C_ij_LL_dw[aa][bb]

				if(self.paramo == 0):
					self.C_ij_GL[aa][bb] = self.c/(100.*self.h[2])*delta_zpm/90*np.sum(7.*(self.WG_tab[0:-1:4,aa]*(self.W_tab[0:-1:4,bb]-self.WIA_tab[0:-1:4,bb]*self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[2])*pow(self.lum_mean[0:-1:4],self.B_IA[2]))*Pk_conv[0:-1:4]/(self.E_tab[0:-1:4]*pow(self.R_tab[0:-1:4],2))
					+ self.WG_tab[4::4,aa]*(self.W_tab[4::4,bb]-self.WIA_tab[4::4,bb]*self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[4::4]*pow(1.+self.z_win[4::4],self.n_IA[2])*pow(self.lum_mean[4::4],self.B_IA[2]))*Pk_conv[4::4]/(self.E_tab[4::4]*pow(self.R_tab[4::4],2))) 
					+ 32.*(self.WG_tab[1::4,aa]*(self.W_tab[1::4,bb]-self.WIA_tab[1::4,bb]*self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[1::4]*pow(1.+self.z_win[1::4],self.n_IA[2])*pow(self.lum_mean[1::4],self.B_IA[2]))*Pk_conv[1::4]/(self.E_tab[1::4]*pow(self.R_tab[1::4],2)) 
					+ self.WG_tab[3::4,aa]*(self.W_tab[3::4,bb]-self.WIA_tab[3::4,bb]*self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[3::4]*pow(1.+self.z_win[3::4],self.n_IA[2])*pow(self.lum_mean[3::4],self.B_IA[2]))*Pk_conv[3::4]/(self.E_tab[3::4]*pow(self.R_tab[3::4],2)))
					+ 12.*self.WG_tab[2::4,aa]*(self.W_tab[2::4,bb]-self.WIA_tab[2::4,bb]*self.A_IA[2]*self.C_IA[2]*self.Omega_m[2]/self.DG_tab[2::4]*pow(1.+self.z_win[2::4],self.n_IA[2])*pow(self.lum_mean[2::4],self.B_IA[2]))*Pk_conv[2::4]/(self.E_tab[2::4]*pow(self.R_tab[2::4],2)))

				self.C_ij_GL_up[aa][bb] = self.c/(100.*self.h[0])*delta_zpm/90*np.sum(7.*(self.WG_tab_up[0:-1:4,aa]*(self.W_tab_up[0:-1:4,bb]-self.WIA_tab_up[0:-1:4,bb]*self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[0])*pow(self.lum_mean[0:-1:4],self.B_IA[0]))*Pk_conv_up[0:-1:4]/(self.E_tab_up[0:-1:4]*pow(self.R_tab_up[0:-1:4],2))
				+ self.WG_tab_up[4::4,aa]*(self.W_tab_up[4::4,bb]-self.WIA_tab_up[4::4,bb]*self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[4::4]*pow(1.+self.z_win[4::4],self.n_IA[0])*pow(self.lum_mean[4::4],self.B_IA[0]))*Pk_conv_up[4::4]/(self.E_tab_up[4::4]*pow(self.R_tab_up[4::4],2))) 
				+ 32.*(self.WG_tab_up[1::4,aa]*(self.W_tab_up[1::4,bb]-self.WIA_tab_up[1::4,bb]*self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[1::4]*pow(1.+self.z_win[1::4],self.n_IA[0])*pow(self.lum_mean[1::4],self.B_IA[0]))*Pk_conv_up[1::4]/(self.E_tab_up[1::4]*pow(self.R_tab_up[1::4],2)) 
				+ self.WG_tab_up[3::4,aa]*(self.W_tab_up[3::4,bb]-self.WIA_tab_up[3::4,bb]*self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[3::4]*pow(1.+self.z_win[3::4],self.n_IA[0])*pow(self.lum_mean[3::4],self.B_IA[0]))*Pk_conv_up[3::4]/(self.E_tab_up[3::4]*pow(self.R_tab_up[3::4],2)))
				+ 12.*self.WG_tab_up[2::4,aa]*(self.W_tab_up[2::4,bb]-self.WIA_tab_up[2::4,bb]*self.A_IA[0]*self.C_IA[0]*self.Omega_m[0]/self.DG_tab_up[2::4]*pow(1.+self.z_win[2::4],self.n_IA[0])*pow(self.lum_mean[2::4],self.B_IA[0]))*Pk_conv_up[2::4]/(self.E_tab_up[2::4]*pow(self.R_tab_up[2::4],2)))

				self.C_ij_GL_dw[aa][bb] = self.c/(100.*self.h[3])*delta_zpm/90*np.sum(7.*(self.WG_tab_dw[0:-1:4,aa]*(self.W_tab_dw[0:-1:4,bb]-self.WIA_tab_dw[0:-1:4,bb]*self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[0:-1:4]*pow(1.+self.z_win[0:-1:4],self.n_IA[3])*pow(self.lum_mean[0:-1:4],self.B_IA[3]))*Pk_conv_dw[0:-1:4]/(self.E_tab_dw[0:-1:4]*pow(self.R_tab_dw[0:-1:4],2))
				+ self.WG_tab_dw[4::4,aa]*(self.W_tab_dw[4::4,bb]-self.WIA_tab_dw[4::4,bb]*self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[4::4]*pow(1.+self.z_win[4::4],self.n_IA[3])*pow(self.lum_mean[4::4],self.B_IA[3]))*Pk_conv_dw[4::4]/(self.E_tab_dw[4::4]*pow(self.R_tab_dw[4::4],2))) 
				+ 32.*(self.WG_tab_dw[1::4,aa]*(self.W_tab_dw[1::4,bb]-self.WIA_tab_dw[1::4,bb]*self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[1::4]*pow(1.+self.z_win[1::4],self.n_IA[3])*pow(self.lum_mean[1::4],self.B_IA[3]))*Pk_conv_dw[1::4]/(self.E_tab_dw[1::4]*pow(self.R_tab_dw[1::4],2)) 
				+ self.WG_tab_dw[3::4,aa]*(self.W_tab_dw[3::4,bb]-self.WIA_tab_dw[3::4,bb]*self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[3::4]*pow(1.+self.z_win[3::4],self.n_IA[3])*pow(self.lum_mean[3::4],self.B_IA[3]))*Pk_conv_dw[3::4]/(self.E_tab_dw[3::4]*pow(self.R_tab_dw[3::4],2)))
				+ 12.*self.WG_tab_dw[2::4,aa]*(self.W_tab_dw[2::4,bb]-self.WIA_tab_dw[2::4,bb]*self.A_IA[3]*self.C_IA[3]*self.Omega_m[3]/self.DG_tab_dw[2::4]*pow(1.+self.z_win[2::4],self.n_IA[3])*pow(self.lum_mean[2::4],self.B_IA[3]))*Pk_conv_dw[2::4]/(self.E_tab_dw[2::4]*pow(self.R_tab_dw[2::4],2)))

		for aa in range(len(self.zrange)):
			for bb in range(len(self.zrange)):
				if(self.paramo == 0):
					outGG.write(str("%.16e" % self.C_ij_GG[aa][bb]))
					outGG.write(str(' '))
					outLL.write(str("%.16e" % self.C_ij_LL[aa][bb]))
					outLL.write(str(' '))
					outGL.write(str("%.16e" % self.C_ij_GL[aa][bb]))
					outGL.write(str(' '))
				outGGU.write(str("%.16e" % self.C_ij_GG_up[aa][bb]))
				outGGU.write(str(' '))
				outGGD.write(str("%.16e" % self.C_ij_GG_dw[aa][bb]))
				outGGD.write(str(' '))
				outLLU.write(str("%.16e" % self.C_ij_LL_up[aa][bb]))
				outLLU.write(str(' '))
				outLLD.write(str("%.16e" % self.C_ij_LL_dw[aa][bb]))
				outLLD.write(str(' '))
				outGLU.write(str("%.16e" % self.C_ij_GL_up[aa][bb]))
				outGLU.write(str(' '))
				outGLD.write(str("%.16e" % self.C_ij_GL_dw[aa][bb]))
				outGLD.write(str(' '))
			if(self.paramo == 0):
				outGG.write(str('\n'))
				outLL.write(str('\n'))
				outGL.write(str('\n'))
			outGGU.write(str('\n'))
			outGGD.write(str('\n'))
			outLLU.write(str('\n'))
			outLLD.write(str('\n'))
			outGLU.write(str('\n'))
			outGLD.write(str('\n'))
		if(self.paramo == 0):
			outGG.close()
			outLL.close()
			outGL.close()
		outGGU.close()
		outGGD.close()
		outLLU.close()
		outLLD.close()
		outGLU.close()
		outGLD.close()

	def Fisher(self, probe, Fisher_matrix_name, lcut_mn, lcut_pl, run_index, Mat_N, Mat_NTOT):
	
		l_new = []
		for i in range(lcut_mn,lcut_pl):
		    l_new.append(self.l[i]);
		lsize=0;
		for i in range(0,len(l_new)):
		    if(l_new[i]<=min(self.l_max_WL,self.l_max_GC)):
		        lsize=lsize+1

		lsize2 = len(l_new)-lsize
		F_mat_name = Fisher_matrix_name

		print(self.param_chain_Cl)

		param_chain = np.copy(self.param_chain_Cl)
		fid_all = np.copy(self.fid_Cl)
		steps_all = np.copy(self.steps_Cl)

		if(self.zcut == "Y"):
			i=0
			while i<len(param_chain):
				for j in range(5, len(BX)):
					prefix = "b"+str(j+1)
					if(param_chain[i] == prefix):
						param_chain = np.delete(param_chain, i)
						fid_all = np.delete(fid_all, i)
						steps_all = np.delete(steps_all, i)
						i=i-1
				i=i+1

		if(probe == "GCp"):
			i=0
			while i<len(param_chain):
				if(param_chain[i] == "A_IA" or param_chain[i] == "n_IA" or param_chain[i] == "B_IA"):
					param_chain = np.delete(param_chain, i)
					fid_all = np.delete(fid_all, i)
					steps_all = np.delete(steps_all, i)
					i=i-1
				i=i+1

		if(probe == "WL"):
			i=0
			while(i<len(param_chain)):
				for j in range(len(BX)):
					prefix = "b"+str(j+1);
					if(param_chain[i].compare(prefix) == 0):
						param_chain = np.delete(param_chain, i)
						fid_all = np.delete(fid_all, i)
						steps_all = np.delete(steps_all, i)
						i=i-1
				i=i+1

		if(self.curvature == "F"):
			i=0
			while(i<len(param_chain)):
				if(param_chain[i] == "wde"):
					param_chain = np.delete(param_chain, i)
					fid_all = np.delete(fid_all, i)
					steps_all = np.delete(steps_all, i)
					i=i-1
				i=i+1

		lll, FX, FY, k_vec, U1, U2 = 0, 0, 0, 0, 0, 0
		I_55, I_100, I_rect_I1, I_rect_I2, I_rect_I3, I_rect_I4 = 0, 0, 0, 0, 0, 0

		C_folder = np.array([["Cl_GG", "Cl_GL"], ["Cl_GL", "Cl_LL"]])

		Fisher_M = np.zeros((len(param_chain), len(param_chain)))
		relat_index = len(param_chain)+1
		i, yn = 0, 0

		while(i<len(param_chain)):
			if(param_chain[i] == "wa"):
				relat_index = i
				yn=yn+1
			i=i+1

		C_ij_ABCD_GG = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LL = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GL = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LG = np.zeros((len(self.zrange), len(self.zrange)))

		C_ij_ABCD_GG_up_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LL_up_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GL_up_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LG_up_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GG_dw_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LL_dw_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GL_dw_PX = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LG_dw_PX = np.zeros((len(self.zrange), len(self.zrange)))

		C_ij_ABCD_GG_up_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LL_up_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GL_up_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LG_up_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GG_dw_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LL_dw_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_GL_dw_PY = np.zeros((len(self.zrange), len(self.zrange)))
		C_ij_ABCD_LG_dw_PY = np.zeros((len(self.zrange), len(self.zrange)))

		CC_GGGG_R = np.zeros((self.Dim_x, self.Dim_x))
		CC_LLLL_R = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLGL_R = np.zeros((self.Dim_y, self.Dim_y))
		CC_GGGL_R = np.zeros((self.Dim_x, self.Dim_y))
		CC_GGLL_R = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLGG_R = np.zeros((self.Dim_y, self.Dim_x))
		CC_LLGG_R = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLLL_R = np.zeros((self.Dim_y, self.Dim_x))
		CC_LLGL_R = np.zeros((self.Dim_x, self.Dim_y))

		CC_GGGG_DR = np.zeros((self.Dim_x, self.Dim_x))
		CC_LLLL_DR = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLGL_DR = np.zeros((self.Dim_y, self.Dim_y))
		CC_GGGL_DR = np.zeros((self.Dim_x, self.Dim_y))
		CC_GGLL_DR = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLGG_DR = np.zeros((self.Dim_y, self.Dim_x))
		CC_LLGG_DR = np.zeros((self.Dim_x, self.Dim_x))
		CC_GLLL_DR = np.zeros((self.Dim_y, self.Dim_x))
		CC_LLGL_DR = np.zeros((self.Dim_x, self.Dim_y))

		CO_CL = np.zeros((self.lsize*(2*self.Dim_x+self.Dim_y), self.lsize*(2*self.Dim_x+self.Dim_y)))
		CO_CL_D = np.zeros((self.lsize*(2*self.Dim_x+self.Dim_y), self.lsize*(2*self.Dim_x+self.Dim_y)))
		CO_I = np.eye(self.lsize*(2*self.Dim_x+self.Dim_y))
		CO_CL_WL = np.zeros((self.lsize2*self.Dim_x, self.lsize2*self.Dim_x))
		CO_CL_WL_D = np.zeros((self.lsize2*self.Dim_x, self.lsize2*self.Dim_x))
		CO_WL_I = np.eye(self.lsize2*self.Dim_x)
		CO_CL_ref = np.zeros((self.lsize*(2*self.Dim_x+self.Dim_y), self.lsize*(2*self.Dim_x+self.Dim_y)))
		CO_CL_WL_ref = np.zeros((self.lsize2*self.Dim_x, self.lsize2*self.Dim_x))

		for FX in range(len(Fisher_M)):
			for FY in range(FX, len(Fisher_M)):
				for lll in range(lsize):
					print(lll)

					CC_GGGG = np.zeros(self.Dim_x**2)
					CC_LLLL = np.zeros(self.Dim_x**2)
					CC_GLGL = np.zeros(self.Dim_y**2)
					CC_GGGL = np.zeros(self.Dim_x*self.Dim_y)
					CC_GGLL = np.zeros(self.Dim_x**2)
					CC_GLGG = np.zeros(self.Dim_y*self.Dim_x)
					CC_LLGG = np.zeros(self.Dim_x**2)
					CC_GLLL = np.zeros(self.Dim_y*self.Dim_x)
					CC_LLGL = np.zeros(self.Dim_x*self.Dim_y)

					CC_GGGG_D = np.zeros(self.Dim_x**2)
					CC_LLLL_D = np.zeros(self.Dim_x**2)
					CC_GLGL_D = np.zeros(self.Dim_y**2)
					CC_GGGL_D = np.zeros(self.Dim_x*self.Dim_y)
					CC_GGLL_D = np.zeros(self.Dim_x**2)
					CC_GLGG_D = np.zeros(self.Dim_y*self.Dim_x)
					CC_LLGG_D = np.zeros(self.Dim_x**2)
					CC_GLLL_D = np.zeros(self.Dim_y*self.Dim_x)
					CC_LLGL_D = np.zeros(self.Dim_x*self.Dim_y)

					C_ij_ABCD_GG_up_PX = np.loadtxt(C_folder[0][0]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_GG_dw_PX = np.loadtxt(C_folder[0][0]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_LL_up_PX = np.loadtxt(C_folder[1][1]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_LL_dw_PX = np.loadtxt(C_folder[1][1]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_GL_up_PX = np.loadtxt(C_folder[0][1]+"/C_"+param_chain[FX]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_GL_dw_PX = np.loadtxt(C_folder[0][1]+"/C_"+param_chain[FX]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_LG_up_PX = np.transpose(C_ij_ABCD_GL_up_PX)
					C_ij_ABCD_LG_dw_PX = np.transpose(C_ij_ABCD_GL_dw_PX)

					C_ij_ABCD_GG_up_PY = np.loadtxt(C_folder[0][0]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_GG_dw_PY = np.loadtxt(C_folder[0][0]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_LL_up_PY = np.loadtxt(C_folder[1][1]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_LL_dw_PY = np.loadtxt(C_folder[1][1]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_GL_up_PY = np.loadtxt(C_folder[0][1]+"/C_"+param_chain[FY]+"_up"+"/COVAR_up_"+str(l_new[lll]))
					C_ij_ABCD_GL_dw_PY = np.loadtxt(C_folder[0][1]+"/C_"+param_chain[FY]+"_dw"+"/COVAR_dw_"+str(l_new[lll]))
					C_ij_ABCD_LG_up_PY = np.transpose(C_ij_ABCD_GL_up_PY)
					C_ij_ABCD_LG_dw_PY = np.transpose(C_ij_ABCD_GL_dw_PY)

					if(FX == 0 and FY == 0):
						C_ij_ABCD_GG = np.loadtxt(C_folder[0][0]+"/C_fid"+"/COVAR_fid_"+str(l_new[lll]))
						C_ij_ABCD_LL = np.loadtxt(C_folder[1][1]+"/C_fid"+"/COVAR_fid_"+str(l_new[lll]))
						C_ij_ABCD_GL = np.loadtxt(C_folder[0][1]+"/C_fid"+"/COVAR_fid_"+str(l_new[lll]))
						C_ij_ABCD_LG = np.transpose(C_ij_ABCD_GL)

					I_55, I_100, I_rect_I1, I_rect_I2, I_rect_I3, I_rect_I4 = 0, 0, 0, 0, 0, 0

					for I1 in range(len(self.zrange)):
						for I2 in range(len(self.zrange)):
							for I3 in range(len(self.zrange)):
								for I4 in range(len(self.zrange)):

									if(I2 <= I1 and I4 <= I3):
										if(FX == 0 and FY == 0):
											CC_GGGG[I_55] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_GG[I2][I4] + C_ij_ABCD_GG[I1][I4]*C_ij_ABCD_GG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
											CC_LLLL[I_55] = (C_ij_ABCD_LL[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_LL[I1][I4]*C_ij_ABCD_LL[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
											CC_GGLL[I_55] = (C_ij_ABCD_GL[I1][I3]*C_ij_ABCD_GL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_GL[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
											CC_LLGG[I_55] = (C_ij_ABCD_LG[I1][I3]*C_ij_ABCD_LG[I2][I4] + C_ij_ABCD_LG[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
										if(FX == relat_index and FY == relat_index):
											CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY])
										elif(FX == relat_index and FY != relat_index):
											CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
										elif(FX != relat_index and FY == relat_index):
											CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY])
											CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY])
										else:
											CC_GGGG_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_LLLL_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_GGLL_D[I_55] = (C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
											CC_LLGG_D[I_55] = (C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
										I_55=I_55+1

									if(I2 <= I1):
										if(FX == 0 and FY == 0):
											CC_GGGL[I_rect_I3] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_GL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_GG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
											CC_GLGG[I_rect_I3] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_LG[I2][I4] + C_ij_ABCD_GG[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
										if(FX == relat_index and FY == relat_index):
											CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]))
											CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*steps_all[FX]))
										elif(FX == relat_index and FY != relat_index):
											CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]*fid_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]))
											CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]*fid_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*steps_all[FX]))
										elif(FX != relat_index and FY == relat_index):
											CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
											CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
										else:
											CC_GGGL_D[I_rect_I3] = 0.5*((C_ij_ABCD_GG_up_PX[I1][I2]-C_ij_ABCD_GG_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GG_up_PY[I1][I2]-C_ij_ABCD_GG_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
											CC_GLGG_D[I_rect_I3] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GG_up_PY[I3][I4]-C_ij_ABCD_GG_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GG_up_PX[I3][I4]-C_ij_ABCD_GG_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
										I_rect_I3=I_rect_I3+1

									if(I2 <= I1):
										if(FX == 0 and FY == 0):
											CC_LLGL[I_rect_I1] = (C_ij_ABCD_LG[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_LL[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
											CC_GLLL[I_rect_I1] = (C_ij_ABCD_GL[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_LL[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
										if(FX == relat_index and FY == relat_index):
											CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]))
											CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*steps_all[FX]))
										elif(FX == relat_index and FY != relat_index):
											CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*steps_all[FX]))
											CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*steps_all[FX]))
										elif(FX != relat_index and FY == relat_index):
											CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
											CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
										else:
											CC_LLGL_D[I_rect_I1] = 0.5*((C_ij_ABCD_LL_up_PX[I1][I2]-C_ij_ABCD_LL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_LL_up_PY[I1][I2]-C_ij_ABCD_LL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_GL_up_PX[I3][I4]-C_ij_ABCD_GL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
											CC_GLLL_D[I_rect_I1] = 0.5*((C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_LL_up_PY[I3][I4]-C_ij_ABCD_LL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY]) + (C_ij_ABCD_GL_up_PY[I1][I2]-C_ij_ABCD_GL_dw_PY[I1][I2])/(2*fid_all[FY]*steps_all[FY]) * (C_ij_ABCD_LL_up_PX[I3][I4]-C_ij_ABCD_LL_dw_PX[I3][I4])/(2*fid_all[FX]*steps_all[FX]))
										I_rect_I1=I_rect_I1+1

									if(FX == 0 and FY == 0):
										CC_GLGL[I_100] = (C_ij_ABCD_GG[I1][I3]*C_ij_ABCD_LL[I2][I4] + C_ij_ABCD_GL[I1][I4]*C_ij_ABCD_LG[I2][I3])/((2*l_new[lll]+1)*self.fsky*self.delta_l)
									if(FX == relat_index and FY == relat_index):
										CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY])
									elif(FX == relat_index and FY != relat_index):
										CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
									elif(FX != relat_index and FY == relat_index):
										CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*steps_all[FY])
									else:
										CC_GLGL_D[I_100] = (C_ij_ABCD_GL_up_PX[I1][I2]-C_ij_ABCD_GL_dw_PX[I1][I2])/(2*fid_all[FX]*steps_all[FX]) * (C_ij_ABCD_GL_up_PY[I3][I4]-C_ij_ABCD_GL_dw_PY[I3][I4])/(2*fid_all[FY]*steps_all[FY])
									I_100=I_100+1

					CC_GGGG = np.reshape(CC_GGGG, (self.Dim_x,self.Dim_x))
					CC_LLLL = np.reshape(CC_LLLL, (self.Dim_x,self.Dim_x))
					CC_GGLL = np.reshape(CC_GGLL, (self.Dim_x,self.Dim_x))
					CC_LLGG = np.reshape(CC_LLGG, (self.Dim_x,self.Dim_x))
					CC_GLGL = np.reshape(CC_GLGL, (self.Dim_y,self.Dim_y))
					CC_GGGL = np.reshape(CC_GGGL, (self.Dim_x,self.Dim_y))
					CC_LLGL = np.reshape(CC_LLGL, (self.Dim_x,self.Dim_y))
					CC_GLGG = np.reshape(CC_GLGG, (self.Dim_y,self.Dim_x))
					CC_GLLL = np.reshape(CC_GLLL, (self.Dim_y,self.Dim_x))
					CC_GLGG = np.transpose(CC_GGGL)
					CC_GLLL = np.transpose(CC_LLGL)

					CC_GGGG_D = np.reshape(CC_GGGG_D, (self.Dim_x,self.Dim_x))
					CC_LLLL_D = np.reshape(CC_LLLL_D, (self.Dim_x,self.Dim_x))
					CC_GGLL_D = np.reshape(CC_GGLL_D, (self.Dim_x,self.Dim_x))
					CC_LLGG_D = np.reshape(CC_LLGG_D, (self.Dim_x,self.Dim_x))
					CC_GLGL_D = np.reshape(CC_GLGL_D, (self.Dim_y,self.Dim_y))
					CC_GGGL_D = np.reshape(CC_GGGL_D, (self.Dim_x,self.Dim_y))
					CC_LLGL_D = np.reshape(CC_LLGL_D, (self.Dim_x,self.Dim_y))
					CC_GLGG_D = np.reshape(CC_GLGG_D, (self.Dim_y,self.Dim_x))
					CC_GLLL_D = np.reshape(CC_GLLL_D, (self.Dim_y,self.Dim_x))
					CC_GLGG_D = np.transpose(CC_GGGL_D)
					CC_GLLL_D = np.transpose(CC_LLGL_D)

					for z1 in range(self.Dim_x):
						for z2 in range(self.Dim_x):
							if(FX == 0 and FY == 0):
								CO_CL[z1*lsize+lll][z2*lsize+lll] = CC_GGGG[z1][z2]
								CO_CL[lsize*self.Dim_x+z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_LLLL[z1][z2]
								CO_CL[z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_GGLL[z1][z2]
								CO_CL[lsize*self.Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_LLGG[z1][z2]
							CO_CL_D[z1*lsize+lll][z2*lsize+lll] = CC_GGGG_D[z1][z2]
							CO_CL_D[lsize*self.Dim_x+z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_LLLL_D[z1][z2]
							CO_CL_D[z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_GGLL_D[z1][z2]
							CO_CL_D[lsize*self.Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_LLGG_D[z1][z2]

					for z1 in range(self.Dim_y):
						for z2 in range(self.Dim_y):
							if(FX == 0 and FY == 0):
								CO_CL[2*lsize*self.Dim_x+z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_GLGL[z1][z2]
							CO_CL_D[2*lsize*self.Dim_x+z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_GLGL_D[z1][z2]

					for z1 in range(self.Dim_x):
						for z2 in range(self.Dim_y):
							if(FX == 0 and FY == 0):
								CO_CL[z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_GGGL[z1][z2]
								CO_CL[lsize*self.Dim_x+z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_LLGL[z1][z2]
							CO_CL_D[z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_GGGL_D[z1][z2]
							CO_CL_D[lsize*self.Dim_x+z1*lsize+lll][2*lsize*self.Dim_x+z2*lsize+lll] = CC_LLGL_D[z1][z2]

					for z1 in range(self.Dim_y):
						for z2 in range(self.Dim_x):
							if(FX == 0 and FY == 0):
								CO_CL[2*lsize*self.Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_GLGG[z1][z2]
								CO_CL[2*lsize*self.Dim_x+z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_GLLL[z1][z2]
							CO_CL_D[2*lsize*self.Dim_x+z1*lsize+lll][z2*lsize+lll] = CC_GLGG_D[z1][z2]
							CO_CL_D[2*lsize*self.Dim_x+z1*lsize+lll][lsize*self.Dim_x+z2*lsize+lll] = CC_GLLL_D[z1][z2]

				print(np.diag(CO_CL_D))
				if(probe == "GCp"):
					if(FX == 0 and FY == 0):
						vector<vector<double>> CO_CL_temp(lsize*self.Dim_x, vector<double>(lsize*self.Dim_x, 0));
						CO_CL_temp = np.copy(CO_CL)
						CO_CL_temp_D = np.copy(CO_CL_D)


				if(probe == "WL"):
					if(FX == 0 and FY == 0):
						CO_CL_temp = np.zeros((lsize*self.Dim_x, lsize*self.Dim_x))
						for i in range(lsize*self.Dim_x, 2*lsize*self.Dim_x):
							for j in range(lsize*self.Dim_x, 2*lsize*self.Dim_x):
								CO_CL_temp[i-lsize*self.Dim_x][j-lsize*self.Dim_x] = CO_CL[i][j]
						CO_CL = np.copy(CO_CL_temp)

					CO_CL_temp_D = np.zeros((lsize*self.Dim_x, lsize*self.Dim_x))
					for i in range(lsize*self.Dim_x, 2*lsize*self.Dim_x):
						for j in range(lsize*self.Dim_x, 2*lsize*self.Dim_x):
							CO_CL_temp_D[i-lsize*self.Dim_x][j-lsize*self.Dim_x] = CO_CL_D[i][j]
					CO_CL_D = np.copy(CO_CL_temp_D)

				print(np.diag(CO_CL_D))
				if(FX==0 and FY == 0):
					CO_CL = lapack.dposv(CO_CL, CO_I)[1]

				Fisher_M[FX][FY] = np.sum(CO_CL*CO_CL_D)
				Fisher_M[FY][FX] = Fisher_M[FX][FY]
				CO_CL_D = np.copy(CO_CL_ref)
				print(Fisher_M[FX][FY])

resource.setrlimit(resource.RLIMIT_NOFILE, (10000,-1))
tab = XC()

def Main_F(X_index, seed=None):
	#for X_index in range(20):
	print(X_index)
	tab.Initializers_G(X_index)
	"""
	tab.Initializers_Pk()
	tab.background()
	if X_index == 0:
		#tab.photoz()
		tab.photoz_load()
	tab.windows()
	#i = range(60)  
	#if __name__ == '__main__':          
		#pool = mp.Pool(10)
		#pool.map(tab.C_l_computing, i)
	for i in range(60):
		tab.C_l_computing(i)

X_ind = range(22)
"""
if __name__ == '__main__':          
	pool = mp.Pool(8)
	map(Main_F, X_ind)
"""

for X_ind in range(22):
	Main_F(X_ind)

tab.Fisher("XC", "Fisher_GCph_WL_XC_XSAF", 0, 60, 60, 60, 60)


