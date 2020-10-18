#Import modules.
import sys
import os
import numpy as np

#Loading parameters files.
Codes_elts = np.loadtxt("../QTLauncher/Codes_W.txt", dtype='str')
Spec_elts = np.loadtxt("../QTLauncher/SpecSAF_W.txt", dtype='str')
XC_elts = np.loadtxt("../QTLauncher/XSAF_W.txt", dtype='str')
Parameters_elts = np.loadtxt("../QTLauncher/Parameters_W.txt", dtype='str')

#Declare dictionaries which take the parameters as values and keys.
Use_Spec_Phot = {}
Combine_elts = {}
P_elts = {}
for i in range(len(Codes_elts)):
    Use_Spec_Phot[Codes_elts[i][0]] = Codes_elts[i][1]
for i in range(len(Spec_elts)):
    Combine_elts[Spec_elts[i][0]] = Spec_elts[i][1]
for i in range(len(Spec_elts)):
    Combine_elts[XC_elts[i][0]] = XC_elts[i][1]
for i in range(len(Parameters_elts)):
    P_elts[Parameters_elts[i][0]] = Parameters_elts[i][1]

#Build arrays with used non redshift dependent parameters.
therms_elements = []
therms_use = []
for i in range(len(Parameters_elts)):
	if(Parameters_elts[i][0].startswith("Use") and Parameters_elts[i][1] == "0"):
		therms_elements = np.insert(therms_elements, len(therms_elements), int(Parameters_elts[i][1]))
		therms_use.append(Parameters_elts[i][0])

if(Combine_elts["zcut_ch"] == "0"):
	Combine_elts["VRS_bins"] = "5"

#Adding elements with redshift dependent parameters.
i=0
while i < len(therms_use):
	if(therms_use[i] == "UseGCspecbias_ch"):
		for j in range(int(Combine_elts["redbinsSP"])-1):
			therms_elements = np.insert(therms_elements, i, therms_elements[i])
			therms_use.insert(i, therms_use[i])
			i=i+1
	if(therms_use[i] == "UseGCspecPS_ch"):
		for j in range(int(Combine_elts["redbinsSP"])-1):
			therms_elements = np.insert(therms_elements, i, therms_elements[i])
			therms_use.insert(i, therms_use[i])
			i=i+1
	if(therms_use[i] == "UseGCphotbias_ch"):
		for j in range(int(Combine_elts["VRS_bins"])-1):
			therms_elements = np.insert(therms_elements, i, therms_elements[i])
			therms_use.insert(i, therms_use[i])
			i=i+1
	i=i+1

therms_GCp = np.copy(therms_elements)
therms_GCs = np.copy(therms_elements)
therms_WL = np.copy(therms_elements)
therms_XC = np.copy(therms_elements)

i=0
while i < len(therms_use):
	if(therms_use[i] == "UseGCspecbias_ch"):
		therms_GCp[i] = 1
		therms_WL[i] = 1
		therms_XC[i] = 1
	if(therms_use[i] == "UseGCspecPS_ch"):
		therms_GCp[i] = 1
		therms_WL[i] = 1
		therms_XC[i] = 1
	if(therms_use[i] == "Usesp_ch"):
		therms_GCp[i] = 1
		therms_WL[i] = 1
		therms_XC[i] = 1
	if(therms_use[i] == "Usesv_ch"):
		therms_GCp[i] = 1
		therms_WL[i] = 1
		therms_XC[i] = 1
	if(therms_use[i] == "UseGCphotbias_ch"):
		therms_WL[i] = 1
		therms_GCs[i] = 1
	if(therms_use[i] == "UseWLAia_ch"):
		therms_GCp[i] = 1
		therms_GCs[i] = 1
	if(therms_use[i] == "UseWLBia_ch"):
		therms_GCp[i] = 1
		therms_GCs[i] = 1
	if(therms_use[i] == "UseWLnia_ch"):
		therms_GCp[i] = 1
		therms_GCs[i] = 1
	i=i+1

#Loading the Fisher matrix.
if os.path.exists("Fisher_GCph_WL_XC_XSAF"):
	XC_base = np.loadtxt("Fisher_GCph_WL_XC_XSAF")
if os.path.exists("Fisher_GCph_XSAF"):
	GCp_base = np.loadtxt("Fisher_GCph_XSAF")
if os.path.exists("Fisher_WL_XSAF"):
	WL_base = np.loadtxt("Fisher_WL_XSAF")

if os.path.exists("Fisher_GCs_SpecSAF"):
	SP_base = np.loadtxt("Fisher_GCs_SpecSAF")

#Matrix combination.

#XC+GCs
if os.path.exists("Fisher_GCph_WL_XC_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	XC = np.copy(XC_base)
	SP = np.copy(SP_base)

	for i in range(len(therms_use)):
		if(therms_XC[i] == 1 and therms_GCs[i] == 1):
			XC = np.insert(XC, i, 0, axis=0)
			XC = np.insert(XC, i, 0, axis=1)
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
		if(therms_XC[i] == 1 and therms_GCs[i] == 0):
			XC = np.insert(XC, i, 0, axis=0)
			XC = np.insert(XC, i, 0, axis=1)
		if(therms_XC[i] == 0 and therms_GCs[i] == 1):
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
	i,j=0,0
	while i < len(XC):
		if(therms_XC[j] == 1 and therms_GCs[j] == 1):
			XC = np.delete(XC, i, axis=0)
			XC = np.delete(XC, i, axis=1)
			SP = np.delete(SP, i, axis=0)
			SP = np.delete(SP, i, axis=1)
			i=i-1
		i=i+1
		j=j+1

	XC_SP = XC+SP

#GCp+GCs
if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	GCp = np.copy(GCp_base)
	SP = np.copy(SP_base)
	for i in range(len(therms_use)):
		if(therms_GCp[i] == 1 and therms_GCs[i] == 1):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
		if(therms_GCp[i] == 1 and therms_GCs[i] == 0):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
		if(therms_GCp[i] == 0 and therms_GCs[i] == 1):
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
	i,j=0,0
	while i < len(GCp):
		if(therms_GCp[j] == 1 and therms_GCs[j] == 1):
			GCp = np.delete(GCp, i, axis=0)
			GCp = np.delete(GCp, i, axis=1)
			SP = np.delete(SP, i, axis=0)
			SP = np.delete(SP, i, axis=1)
			i=i-1
		i=i+1
		j=j+1

	Gcp_SP = GCp+SP

#GCs+WL
if os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	WL = np.copy(WL_base)
	SP = np.copy(SP_base)
	for i in range(len(therms_use)):
		if(therms_WL[i] == 1 and therms_GCs[i] == 1):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
		if(therms_WL[i] == 1 and therms_GCs[i] == 0):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
		if(therms_WL[i] == 0 and therms_GCs[i] == 1):
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
	i,j=0,0
	while i < len(WL):
		if(therms_WL[j] == 1 and therms_GCs[j] == 1):
			WL = np.delete(WL, i, axis=0)
			WL = np.delete(WL, i, axis=1)
			SP = np.delete(SP, i, axis=0)
			SP = np.delete(SP, i, axis=1)
			i=i-1
		i=i+1
		j=j+1

	SP_WL = WL+SP

#GCp+GCs+WL
if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	WL = np.copy(WL_base)
	GCp = np.copy(GCp_base)
	SP = np.copy(SP_base)
	for i in range(len(therms_use)):
		if(therms_WL[i] == 1 and therms_GCp[i] == 1 and therms_GCs[i] == 1):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
		if(therms_WL[i] == 1 and therms_GCp[i] == 0):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
		if(therms_WL[i] == 0 and therms_GCp[i] == 1):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
		if(therms_WL[i] == 1 and therms_GCs[i] == 0):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
		if(therms_WL[i] == 0 and therms_GCs[i] == 1):
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
		if(therms_GCp[i] == 1 and therms_GCs[i] == 0):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
		if(therms_GCp[i] == 0 and therms_GCs[i] == 1):
			SP = np.insert(SP, i, 0, axis=0)
			SP = np.insert(SP, i, 0, axis=1)
	i,j=0,0
	while i < len(WL):
		if(therms_WL[j] == 1 and therms_GCp[j] == 1 and therms_GCs[j] == 1):
			GCp = np.delete(GCp, i, axis=0)
			GCp = np.delete(GCp, i, axis=1)
			SP = np.delete(SP, i, axis=0)
			SP = np.delete(SP, i, axis=1)
			WL = np.delete(WL, i, axis=0)
			WL = np.delete(WL, i, axis=1)
			i=i-1
		i=i+1
		j=j+1

	GCp_SP_WL = WL+GCp+SP

#GCp+WL
if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF"):
	WL = np.copy(WL_base)
	GCp = np.copy(GCp_base)
	for i in range(len(therms_use)):
		if(therms_WL[i] == 1 and therms_GCp[i] == 1):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
		if(therms_WL[i] == 1 and therms_GCp[i] == 0):
			WL = np.insert(WL, i, 0, axis=0)
			WL = np.insert(WL, i, 0, axis=1)
		if(therms_WL[i] == 0 and therms_GCp[i] == 1):
			GCp = np.insert(GCp, i, 0, axis=0)
			GCp = np.insert(GCp, i, 0, axis=1)
	i,j=0,0
	while i < len(WL):
		if(therms_WL[j] == 1 and therms_GCp[j] == 1):
			WL = np.delete(WL, i, axis=0)
			WL = np.delete(WL, i, axis=1)
			GCp = np.delete(GCp, i, axis=0)
			GCp = np.delete(GCp, i, axis=1)
			i=i-1
		i=i+1
		j=j+1

	GCp_WL = WL+GCp

ind_w0, ind_wa = 0, 0
w0wa = 0
i=0
while i < len(therms_use):
	if(therms_use[i] == "Usew0_ch" and therms_elements[i] == 0):
		ind_w0 = i
		w0wa = w0wa+1
	if(therms_use[i] == "Usewa_ch" and therms_elements[i] == 0):
		ind_wa = i
		w0wa = w0wa+1
	i=i+1
i=0

#Compute the FoM.
if(w0wa == 2):
	if os.path.exists("Fisher_GCph_WL_XC_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
		C_new = np.linalg.inv(XC_SP)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("Gcp+GCs+WL+XC FoM = " + str(FOM))

	if os.path.exists("Fisher_GCph_WL_XC_XSAF"):
		C_new = np.linalg.inv(XC_base)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("Gcp+WL+XC FoM = " + str(FOM))

	if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
		C_new = np.linalg.inv(GCp_SP_WL)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("Gcp+GCs+WL FoM = " + str(FOM))

	if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF"):
		C_new = np.linalg.inv(GCp_WL)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("Gcp+WL FoM = " + str(FOM))

	if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
		C_new = np.linalg.inv(Gcp_SP)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("Gcp+GCs FoM = " + str(FOM))

	if os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
		C_new = np.linalg.inv(SP_WL)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("GCs+WL FoM = " + str(FOM))

	if os.path.exists("Fisher_GCph_XSAF"):
		C_new = np.linalg.inv(GCp_base)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("GCp FoM = " + str(FOM))

	if os.path.exists("Fisher_GCs_SpecSAF"):
		C_new = np.linalg.inv(SP_base)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("GCs FoM = " + str(FOM))

	if os.path.exists("Fisher_WL_XSAF"):
		C_new = np.linalg.inv(WL_base)
		C_sub = np.array(([(C_new[ind_w0][ind_w0]), C_new[ind_w0][ind_wa]], [C_new[ind_wa][ind_w0], (C_new[ind_wa][ind_wa])]))
		FOM = np.sqrt(1./(np.linalg.det(C_sub)))
		print("WL FoM = " + str(FOM))


#Save the Fisher matrix.
if os.path.exists("Fisher_GCph_WL_XC_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	i,j = 0,0
	out_F = open("Fisher_GCph_GCs_WL_XC_TSAF", 'w')
	while i < len(XC_SP):
		while j < len(XC_SP):
			out_F.write(str("%.12e" % XC_SP[i][j]))
			out_F.write(" ")
			j=j+1
		out_F.write("\n")
		j=0
		i=i+1
	out_F.close()

if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	i,j = 0,0
	out_F = open("Fisher_GCph_GCs_WL_TSAF", 'w')
	while i < len(GCp_SP_WL):
		while j < len(GCp_SP_WL):
			out_F.write(str("%.12e" % GCp_SP_WL[i][j]))
			out_F.write(" ")
			j=j+1
		out_F.write("\n")
		j=0
		i=i+1
	out_F.close()

if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	i,j = 0,0
	out_F = open("Fisher_GCph_GCs_TSAF", 'w')
	while i < len(Gcp_SP):
		while j < len(Gcp_SP):
			out_F.write(str("%.12e" % Gcp_SP[i][j]))
			out_F.write(" ")
			j=j+1
		out_F.write("\n")
		j=0
		i=i+1
	out_F.close()

if os.path.exists("Fisher_WL_XSAF") and os.path.exists("Fisher_GCs_SpecSAF"):
	i,j = 0,0
	out_F = open("Fisher_GCs_WL_TSAF", 'w')
	while i < len(SP_WL):
		while j < len(SP_WL):
			out_F.write(str("%.12e" % SP_WL[i][j]))
			out_F.write(" ")
			j=j+1
		out_F.write("\n")
		j=0
		i=i+1
	out_F.close()

if os.path.exists("Fisher_GCph_XSAF") and os.path.exists("Fisher_WL_XSAF"):
	i,j = 0,0
	out_F = open("Fisher_GCph_WL_XSAF", 'w')
	while i < len(GCp_WL):
		while j < len(GCp_WL):
			out_F.write(str("%.12e" % GCp_WL[i][j]))
			out_F.write(" ")
			j=j+1
		out_F.write("\n")
		j=0
		i=i+1
	out_F.close()

