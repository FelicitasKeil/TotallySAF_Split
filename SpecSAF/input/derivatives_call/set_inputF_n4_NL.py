#####################################################################################
#Author: S. Yahia-Cherif
#This script creates a set name of input functions to launch for the Fisher matrices.
#####################################################################################

#Modules import
import numpy as np

#Load the redshift values, the bias and f(z) 
zrange, b, growth_f = np.loadtxt("../bias_n_growth_baseline.dat",delimiter=' ',
                                 usecols = (0,2,4), unpack = True)
# Redshift Dependency
N_notRD_params = 6
N_RD_params = 5

Parameters_elts = np.loadtxt("../../../QTLauncher/SpecSAF_W.txt", dtype='str')

SpecSAF_elts = {}
for i in range(len(Parameters_elts)):
    SpecSAF_elts[Parameters_elts[i][0]] = Parameters_elts[i][1]

Allder_choice = int(SpecSAF_elts["DerMethodSP_ch"])
choices = np.array(["3", "5", "7"])
choice_der_pts_shape = int(choices[Allder_choice])
choice_der_pts_sig_p = int(choices[Allder_choice])
choice_der_pts_sig_v = int(choices[Allder_choice])
choice_der_pts_Da = int(choices[Allder_choice])
choice_der_pts_H = int(choices[Allder_choice])
choice_der_pts_fs8 = int(choices[Allder_choice])
choice_der_pts_bs8 = int(choices[Allder_choice])

#Number of parameters
N_params = N_RD_params+N_notRD_params

#Array of functions call
input_arr = np.chararray((N_params*len(zrange)), itemsize=1000)

#The files for each blocs are loaded
f_out_LU=open('der_input_NB_LU','wb')
f_out_RD=open('der_input_NB_RD','wb')
f_out_all=open('der_input_NB_all','wb')

i,j=0,0
#All the function call are written and saved in the 3 corresponding files
while i < len(input_arr):
    
    if choice_der_pts_shape == 3:
        input_arr[i] = 'der_wb_3pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_wb_up, P_m_wb_dw, P_m_NW_wb_up, P_m_NW_wb_dw, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+1] = 'der_h_3pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_h_up, P_m_h_dw, P_m_NW_h_up, P_m_NW_h_dw, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+2] = 'der_wm_3pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_wm_up, P_m_wm_dw, P_m_NW_wm_up, P_m_NW_wm_dw, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+3] = 'der_ns_3pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_ns_up, P_m_ns_dw, P_m_NW_ns_up, P_m_NW_ns_dw, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        
    elif choice_der_pts_shape == 5:
        input_arr[i] = 'der_wb_5pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_wb_up, P_m_wb_up2, P_m_wb_dw, P_m_wb_dw2, P_m_NW_wb_up, P_m_NW_wb_up2, P_m_NW_wb_dw, P_m_NW_wb_dw2, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+1] = 'der_h_5pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_h_up, P_m_h_up2, P_m_h_dw, P_m_h_dw2, P_m_NW_h_up, P_m_NW_h_up2, P_m_NW_h_dw, P_m_NW_h_dw2, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+2] = 'der_wm_5pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_wm_up, P_m_wm_up2, P_m_wm_dw, P_m_wm_dw2, P_m_NW_wm_up, P_m_NW_wm_up2, P_m_NW_wm_dw, P_m_NW_wm_dw2, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+3] = 'der_ns_5pts(k_ref, mu_ref, '+str(zrange[j])+', P_m_ns_up, P_m_ns_up2, P_m_ns_dw, P_m_ns_dw2, P_m_NW_ns_up, P_m_NW_ns_up2, P_m_NW_ns_dw, P_m_NW_ns_dw2, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
 
    elif choice_der_pts_shape == 7:
        input_arr[i] = 'der_wb_7pts(k_ref, mu_ref, '+str(zrange[j])+',P_m_wb_up, P_m_wb_up2, P_m_wb_up3, P_m_wb_dw, P_m_wb_dw2, P_m_wb_dw3, P_m_NW_wb_up, P_m_NW_wb_up2, P_m_NW_wb_up3, P_m_NW_wb_dw, P_m_NW_wb_dw2, P_m_NW_wb_dw3, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+1] = 'der_h_7pts(k_ref, mu_ref, '+str(zrange[j])+',P_m_h_up, P_m_h_up2, P_m_h_up3, P_m_h_dw, P_m_h_dw2, P_m_h_dw3, P_m_NW_h_up, P_m_NW_h_up2, P_m_NW_h_up3, P_m_NW_h_dw, P_m_NW_h_dw2, P_m_NW_h_dw3, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+2] = 'der_wm_7pts(k_ref, mu_ref, '+str(zrange[j])+',P_m_wm_up, P_m_wm_up2, P_m_wm_up3, P_m_wm_dw, P_m_wm_dw2, P_m_wm_dw3, P_m_NW_wm_up, P_m_NW_wm_up2, P_m_NW_wm_up3, P_m_NW_wm_dw, P_m_NW_wm_dw2, P_m_NW_wm_dw3, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        input_arr[i+3] = 'der_ns_7pts(k_ref, mu_ref, '+str(zrange[j])+',P_m_ns_up, P_m_ns_up2, P_m_ns_up3, P_m_ns_dw, P_m_ns_dw2, P_m_ns_dw3, P_m_NW_ns_up, P_m_NW_ns_up2, P_m_NW_ns_up3, P_m_NW_ns_dw, P_m_NW_ns_dw2, P_m_NW_ns_dw3, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        
    input_arr[i+4] = 'der_sp_'+str(choice_der_pts_sig_p)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    input_arr[i+5] = 'der_sv_'+str(choice_der_pts_sig_v)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
        
    input_arr[i+6] = 'der_Da_'+str(choice_der_pts_Da)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    input_arr[i+7] = 'der_H_'+str(choice_der_pts_H)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    input_arr[i+8] = 'der_fs8_'+str(choice_der_pts_fs8)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    input_arr[i+9] = 'der_bs8_'+str(choice_der_pts_bs8)+'pts(k_ref, mu_ref, '+str(zrange[j])+', P_m, P_m_NW, '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    input_arr[i+10] = 'der_P_shot(k_ref, mu_ref, '+str(zrange[j])+', 10**P_m(np.log10(k_ref)), 10**P_m_NW(np.log10(k_ref)), '+str(b[j])+'*sig_8_fid, '+str(growth_f[j])+'*sig_8_fid, H('+str(zrange[j])+'), H_ref('+str(zrange[j])+'), D_A('+str(zrange[j])+'), D_A_ref('+str(zrange[j])+'), '+'sig_p_fid['+str(j)+']'+', '+'sig_v_fid['+str(j)+']'+')'
    
    f_out_all.write(input_arr[i])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+1])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+2])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+3])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+4])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+5])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+6])
    f_out_all.write(b'\n') 
    f_out_all.write(input_arr[i+7])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+8])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+9])
    f_out_all.write(b'\n')
    f_out_all.write(input_arr[i+10])
    f_out_all.write(b'\n')
    
    f_out_LU.write(input_arr[i])
    f_out_LU.write(b'\n')
    f_out_LU.write(input_arr[i+1])
    f_out_LU.write(b'\n')
    f_out_LU.write(input_arr[i+2])
    f_out_LU.write(b'\n')
    f_out_LU.write(input_arr[i+3])
    f_out_LU.write(b'\n')
    f_out_LU.write(input_arr[i+4])
    f_out_LU.write(b'\n')
    f_out_LU.write(input_arr[i+5])
    f_out_LU.write(b'\n')
    
    f_out_RD.write(input_arr[i+6])
    f_out_RD.write(b'\n') 
    f_out_RD.write(input_arr[i+7])
    f_out_RD.write(b'\n')
    f_out_RD.write(input_arr[i+8])
    f_out_RD.write(b'\n')
    f_out_RD.write(input_arr[i+9])
    f_out_RD.write(b'\n')
    f_out_RD.write(input_arr[i+10])
    f_out_RD.write(b'\n')
    
    i=i+N_params
    j=j+1
    
f_out_all.close()
f_out_LU.close()
f_out_RD.close()
