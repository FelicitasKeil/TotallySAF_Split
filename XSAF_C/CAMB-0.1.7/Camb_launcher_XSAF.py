###################################################################################################
#Authors: S. Yahia-Cherif
#Last update: 29/06/2020
#This script call Camb to get the matter power spectrums.
###################################################################################################

import os
import numpy as np
import time
import sys, platform
from scipy.interpolate import CubicSpline

#Loading the parameters files.
Codes_elts = np.loadtxt("../../QTLauncher/Codes_W.txt", dtype='str')
X_elts = np.loadtxt("../../QTLauncher/XSAF_W.txt", dtype='str')
Parameters_elts = np.loadtxt("../../QTLauncher/Parameters_W.txt", dtype='str')

#All the parameters are saved in a dictionary.
XSAF_elts = {}
for i in range(len(X_elts)):
    XSAF_elts[X_elts[i][0]] = X_elts[i][1]
for i in range(len(Parameters_elts)):
    XSAF_elts[Parameters_elts[i][0]] = Parameters_elts[i][1]

#Initializing all the variables.
fid = np.array([float(XSAF_elts["VFidOmegab"]), float(XSAF_elts["VFidh"]), float(XSAF_elts["VFidOmegam"]), float(XSAF_elts["VFidns"]), float(XSAF_elts["VFidOmegaDE"]), float(XSAF_elts["VFidw0"]), float(XSAF_elts["VFidwa"]), float(XSAF_elts["VFidAs"])*10**(-9)])

z = np.linspace(float(XSAF_elts["Vzmin"]), float(XSAF_elts["Vzmax"]), int(XSAF_elts["Vprec_Int_z"]))
z_win_3_1 = 1./4*(3*z[1:]+z[:-1])
z_win_1_3 = 1./4*(z[1:]+3*z[:-1])
z_win_1_2 = 1./2*(z[1:]+z[:-1])
z_win = np.zeros(len(z)+len(z_win_3_1)+len(z_win_1_2)+len(z_win_1_3))
i,j=0,0
while i < len(z_win):
    if i == 0:
        z_win[i] = z[j]
        i=i+1
        j=j+1
    elif i > 0 and i < len(z_win):
        z_win[i] = z_win_1_3[j-1]
        z_win[i+1] = z_win_1_2[j-1]
        z_win[i+2] = z_win_3_1[j-1]
        z_win[i+3] = z[j]
        i=i+4
        j=j+1
z = np.copy(z_win)

zrange = range(len(z))
z_str = np.chararray((len(zrange)), itemsize=1000)

z = z[0:len(zrange)]
zrange = zrange[0:len(zrange)]

z_str = np.chararray((len(zrange)), itemsize=1000)
i=0
while i < len(zrange):
    z_str[i] = "iz_"+str(zrange[i])
    i=i+1
i=0
z_str = z_str.decode('utf-8')

d_step = 3
i,l,paramo,t=0,0,0,0 

wb_new, eps_wb = fid[0], float(XSAF_elts["VStepOmegab"])
h_new, eps_h = fid[1], float(XSAF_elts["VSteph"])
wm_new, eps_wm = fid[2], float(XSAF_elts["VStepOmegam"])
ns_new, eps_ns = fid[3], float(XSAF_elts["VStepns"])
wde_new, eps_wde = fid[4], float(XSAF_elts["VStepOmegaDE"])
w0_new, eps_w0 = fid[5], float(XSAF_elts["VStepw0"])
wa_new, eps_wa = fid[6], float(XSAF_elts["VStepwa"])
As_new, eps_As = fid[7], float(XSAF_elts["VStepsigma8"])
w_neu = float(XSAF_elts["VFidWnu"])

path_repository = "WP_Pk"

#The path are all declared and initialized in variables.
fold_path_fid = [path_repository+"/fid"]
fold_path_wb = [path_repository+"/wb_up", path_repository+"/wb_up2", path_repository+"/wb_up3", path_repository+"/wb_dw", path_repository+"/wb_dw2", path_repository+"/wb_dw3"]
fold_path_h = [path_repository+"/h_up", path_repository+"/h_up2", path_repository+"/h_up3", path_repository+"/h_dw", path_repository+"/h_dw2", path_repository+"/h_dw3"]
fold_path_wm = [path_repository+"/wm_up", path_repository+"/wm_up2", path_repository+"/wm_up3", path_repository+"/wm_dw", path_repository+"/wm_dw2", path_repository+"/wm_dw3"]
fold_path_ns = [path_repository+"/ns_up", path_repository+"/ns_up2", path_repository+"/ns_up3", path_repository+"/ns_dw", path_repository+"/ns_dw2", path_repository+"/ns_dw3"]
fold_path_wde = [path_repository+"/wde_up", path_repository+"/wde_up2", path_repository+"/wde_up3", path_repository+"/wde_dw", path_repository+"/wde_dw2", path_repository+"/wde_dw3"]
fold_path_w0 = [path_repository+"/w0_up", path_repository+"/w0_up2", path_repository+"/w0_up3", path_repository+"/w0_dw", path_repository+"/w0_dw2", path_repository+"/w0_dw3"]
fold_path_wa = [path_repository+"/wa_up", path_repository+"/wa_up2", path_repository+"/wa_up3", path_repository+"/wa_dw", path_repository+"/wa_dw2", path_repository+"/wa_dw3"]
fold_path_s8 = [path_repository+"/s8_up", path_repository+"/s8_up2", path_repository+"/s8_up3", path_repository+"/s8_dw", path_repository+"/s8_dw2", path_repository+"/s8_dw3"]

#The directories are all created if they don't exist.
if not os.path.exists(fold_path_fid[0]):
    os.makedirs(fold_path_fid[0])
    
i=0
while i < len(fold_path_wb):
    if not os.path.exists(fold_path_wb[i]):
        os.makedirs(fold_path_wb[i])
    if not os.path.exists(fold_path_h[i]):
        os.makedirs(fold_path_h[i])
    if not os.path.exists(fold_path_wm[i]):
        os.makedirs(fold_path_wm[i])
    if not os.path.exists(fold_path_ns[i]):
        os.makedirs(fold_path_ns[i])
    if not os.path.exists(fold_path_wde[i]):
        os.makedirs(fold_path_wde[i])
    if not os.path.exists(fold_path_w0[i]):
        os.makedirs(fold_path_w0[i])
    if not os.path.exists(fold_path_wa[i]):
        os.makedirs(fold_path_wa[i])
    if not os.path.exists(fold_path_s8[i]):
        os.makedirs(fold_path_s8[i])
    i=i+1

while paramo < len(fid)+1:
        
    st=(d_step-1)/2
            
    if paramo == 0:
        #Initialisation of Camb parameters values for the linear case.
        with open('params_base.ini', 'r') as file :
            filedata = file.readlines()
            filedata[17] = "do_nonlinear = " + str(0) + "\n"
            filedata[34] = "use_physical   = F" + "\n"
            filedata[42] = "w              = "+str(w0_new) + "\n"
            filedata[47] = "wa             = "+str(wa_new) + "\n"
            filedata[53] = "omega_baryon   = "+str(wb_new) + "\n"
            filedata[54] = "omega_cdm      = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[55] = "omega_lambda   = "+str(1.-wm_new) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[86] = "scalar_amp(1)             = "+str(As_new) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  0." + "\n")
                else:
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  "+str(z[len(zrange)-1-bbb]) + "\n")
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata = np.insert(filedata, aaa+bbb, "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n")
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0

        #Call Camb
        os.system('./camb params.ini')   
        
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1

        #Interpolate and compute the integrals of the power spectrums.
        P_m = CubicSpline(np.log10(kh[-1]),np.log10(pk[-1]))
        integrale = 1./(2*np.pi**2) * np.sum((kh[-1][1:]-kh[-1][:-1])/6. * ( (kh[-1][:-1]**2*(3*(np.sin(8*kh[-1][:-1]) - 8*kh[-1][:-1]*np.cos(8*kh[-1][:-1]))/(8*kh[-1][:-1])**3)**2*pk[-1][:-1]) + (kh[-1][1:]**2*(3*(np.sin(8*kh[-1][1:]) - 8*kh[-1][1:]*np.cos(8*kh[-1][1:]))/(8*kh[-1][1:])**3)**2*pk[-1][1:]) + 4.*( ((kh[-1][1:]+kh[-1][:-1])/2.)**2*(3*(np.sin(8*(kh[-1][1:]+kh[-1][:-1])/2.) - 8*(kh[-1][1:]+kh[-1][:-1])/2*np.cos(8*(kh[-1][1:]+kh[-1][:-1])/2.))/(8*(kh[-1][1:]+kh[-1][:-1])/2.)**3)**2*10**P_m(np.log10((kh[-1][:-1]+kh[-1][1:])/2))) ))

        #Compute sigma8 and As.
        s8_fiducial_ref = np.sqrt(integrale)
        As_fiducial_ref = np.copy(As_new)

        #Initialisation of Camb parameters values for the non linear case.
        with open('params_base.ini', 'r') as file :
            filedata = file.readlines()
            filedata[17] = "do_nonlinear = " + str(1) + "\n"
            filedata[34] = "use_physical   = F" + "\n"
            filedata[42] = "w              = "+str(w0_new) + "\n"
            filedata[47] = "wa             = "+str(wa_new) + "\n"
            filedata[53] = "omega_baryon   = "+str(wb_new) + "\n"
            filedata[54] = "omega_cdm      = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[55] = "omega_lambda   = "+str(1.-wm_new) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[86] = "scalar_amp(1)             = "+str(As_fiducial_ref) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  0." + "\n")
                else:
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  "+str(z[len(zrange)-1-bbb]) + "\n")
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata = np.insert(filedata, aaa+bbb, "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n")
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0

        #Call Camb.
        os.system('./camb params.ini')   
        
        #Save all the fiducial power spectrums.
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1

        kh = np.delete(kh, len(kh)-1, axis=0)
        pk = np.delete(pk, len(pk)-1, axis=0)
        
        outP = open(fold_path_fid[0]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
        outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
        outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_new) +"\n")
        outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
        
        files,flines=0,0
        while files < len(z):
            while flines < len(kh[files]):
               outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
               flines=flines+1
            flines=0
            files=files+1
        outP.close()
        paramo=paramo+1
        continue
                
    while l < d_step:
        if (l == st):
            l=l+1
        if (t==0):
            if paramo == 1:
                FPA = fold_path_wb
                wb_new = fid[0]*(1 + (l-st)*eps_wb)
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]
            elif paramo == 2:
                FPA = fold_path_h
                wb_new = fid[0]
                h_new = fid[1]*(1 + (l-st)*eps_h)
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]
            elif paramo == 3:
                FPA = fold_path_wm
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]*(1 + (l-st)*eps_wm)
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]
            elif paramo == 4:
                FPA = fold_path_ns
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]*(1 + (l-st)*eps_ns)
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]
            elif paramo == 5:
                FPA = fold_path_wde
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]
            elif paramo == 6:
                FPA = fold_path_w0
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]*(1 + (l-st)*eps_w0)
                wa_new = fid[6]
            elif paramo == 7:
                FPA = fold_path_wa
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6] + (l-st)*eps_wa 
            elif paramo == 8:
                FPA = fold_path_s8
                wb_new = fid[0]
                h_new = fid[1]
                wm_new = fid[2]
                ns_new = fid[3]
                wde_new = fid[4]
                w0_new = fid[5]
                wa_new = fid[6]

        #Initialisation of Camb parameters values for the linear case (non fiducial).
        with open('params_base.ini', 'r') as file :
            filedata = file.readlines()
            filedata[17] = "do_nonlinear = " + str(0) + "\n"
            filedata[34] = "use_physical   = F" + "\n"
            filedata[42] = "w              = "+str(w0_new) + "\n"
            filedata[47] = "wa             = "+str(wa_new) + "\n"
            filedata[53] = "omega_baryon   = "+str(wb_new) + "\n"
            filedata[54] = "omega_cdm      = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[55] = "omega_lambda   = "+str(1.-wm_new) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[86] = "scalar_amp(1)             = "+str(As_new) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  0." + "\n")
                else:
                    filedata = np.insert(filedata, aaa+bbb, "transfer_redshift("+str(bbb+1)+")    =  "+str(z[len(zrange)-1-bbb]) + "\n")
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata = np.insert(filedata, aaa+bbb, "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n")
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0

        #Call Camb.
        os.system('./camb params.ini')   
        
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1

        #Interpolate and compute the integrals of the power spectrums.
        P_m = CubicSpline(np.log10(kh[-1]),np.log10(pk[-1]))
        integrale = 1./(2*np.pi**2) * np.sum((kh[-1][1:]-kh[-1][:-1])/6. * ( (kh[-1][:-1]**2*(3*(np.sin(8*kh[-1][:-1]) - 8*kh[-1][:-1]*np.cos(8*kh[-1][:-1]))/(8*kh[-1][:-1])**3)**2*pk[-1][:-1]) + (kh[-1][1:]**2*(3*(np.sin(8*kh[-1][1:]) - 8*kh[-1][1:]*np.cos(8*kh[-1][1:]))/(8*kh[-1][1:])**3)**2*pk[-1][1:]) + 4.*( ((kh[-1][1:]+kh[-1][:-1])/2.)**2*(3*(np.sin(8*(kh[-1][1:]+kh[-1][:-1])/2.) - 8*(kh[-1][1:]+kh[-1][:-1])/2*np.cos(8*(kh[-1][1:]+kh[-1][:-1])/2.))/(8*(kh[-1][1:]+kh[-1][:-1])/2.)**3)**2*10**P_m(np.log10((kh[-1][:-1]+kh[-1][1:])/2))) ))

        #Compute sigma8 and As.
        if paramo == 8:
            s8_modified = s8_fiducial_ref*(1 + (l-st)*eps_As)
            As_modified = As_fiducial_ref*(s8_modified/s8_fiducial_ref)**2
        else:
            s8_modified = np.sqrt(integrale)
            As_modified = As_fiducial_ref*(s8_fiducial_ref/s8_modified)**2

        #Initialisation of Camb parameters values for the non linear case (non fiducial).
        with open('params.ini', 'r') as file :
            filedata = file.readlines()
            filedata[17] = "do_nonlinear = " + str(1) + "\n"
            filedata[34] = "use_physical   = F" + "\n"
            filedata[42] = "w              = "+str(w0_new) + "\n"
            filedata[47] = "wa             = "+str(wa_new) + "\n"
            filedata[53] = "omega_baryon   = "+str(wb_new) + "\n"
            filedata[54] = "omega_cdm      = "+str(wm_new-wb_new-w_neu) + "\n"
            filedata[55] = "omega_lambda   = "+str(1.-wm_new) + "\n"
            filedata[39] = "hubble         = "+str(h_new*100) + "\n"
            filedata[86] = "scalar_amp(1)             = "+str(As_modified) + "\n"
            filedata[87] = "scalar_spectral_index(1)  = "+str(ns_new) + "\n"
            filedata[167] = "transfer_num_redshifts  = "+ str(len(zrange)+1) + "\n"
            
            aaa, bbb = 169, 0
            while bbb < len(zrange)+1:
                if bbb == len(zrange):
                    filedata[aaa+bbb] = "transfer_redshift("+str(bbb+1)+")    =  0." + "\n"
                else:
                    filedata[aaa+bbb] = "transfer_redshift("+str(bbb+1)+")    =  "+str(z[len(zrange)-1-bbb]) + "\n"
                bbb=bbb+1
   
            aaa = aaa + bbb
            bbb = 0
            while bbb < len(zrange)+1:
                filedata[aaa+bbb] = "transfer_filename("+str(bbb+1)+")    =  transfer_out"+str(bbb+1)+".dat" + "\n"
                bbb=bbb+1
            
        flin=0
        with open('params.ini', 'w') as file:
            while flin < len(filedata):
                file.write(filedata[flin])
                flin=flin+1
            flin=0
        
        #Call Camb
        os.system('./camb params.ini')   
        
        aaa = len(zrange)
        while aaa >= 0:
            if aaa == len(zrange):
                kh, pk = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
            elif aaa > 0:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(aaa)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            else:
                kh1, pk1 = np.loadtxt("test_matterpower_"+str(len(zrange)+1)+".dat", usecols=(0,1,), unpack=True)
                kh = np.vstack((kh,kh1))
                pk = np.vstack((pk,pk1))
            aaa = aaa-1
        
        #Save all the non fiducial power spectrums.
        files,flines=0,0
        if (l == st+1):
            outP = open(FPA[0]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        elif (l == st+2):
            outP = open(FPA[1]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        elif (l == st+3):
            outP = open(FPA[2]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        elif (l == st-1):
            outP = open(FPA[3]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        elif (l == st-2):
            outP = open(FPA[4]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        elif (l == st-3):
            outP = open(FPA[5]+"/Pks8sqRatio_ist_LogSplineInterpPk.dat",'w')
            outP.write(str("%.12e" % z[0]) + " to " + str("%.12e" % z[len(z)-1]) + "\n")
            outP.write(str(wb_new) + " " + str(h_new) + " " + str(wm_new) + " " + str(ns_new) + " " + str(wde_new) + " " + str(w0_new) + " " + str(wa_new) + " " + str(As_modified) +"\n")
            outP.write("   k(h/Mpc)      Pk/s8^2(Mpc/h)^3 \n")
            while files < len(zrange):
                while flines < len(kh[files]):
                    outP.write(str("%.12e" % kh[files][flines])+" "+str("%.12e" % pk[files][flines])+"\n")
                    flines=flines+1
                flines=0
                files=files+1

        outP.close()
        l=l+1
    l=0
    paramo=paramo+1
paramo=0

#Erase all the temporar files.
os.system("rm -r test_transfer_out*")
os.system("rm -r test_matterpower_*")
os.system("rm -r test_matterpower*")
