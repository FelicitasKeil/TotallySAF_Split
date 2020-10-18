"""
Author: S.Yahia-Cherif
Last update: 29/06/2020
Description: This code is the TotallySAF launcher. It calls all the codes needed to run TotallySAF.
"""

#Modules call
import sys
import os
import numpy as np
import time
import datetime
print("")

"""
Wid_org --> arguments: the parameters files produced after calling the QT interface.
Output: the modified parameters files.
This function reorganize the parameter files and erase the repeated entries to keep the entries selected by the user.
"""
def Wid_org(Win):
	i=0
	j=i+1
	while i < len(Win):
		while j < len(Win):
			if(Win[j][0] == Win[i][0]):
				Win[i][1] = Win[j][1]
				Win = np.delete(Win, j, axis=0)
				j=j-1
			j=j+1
		i=i+1
		j=i+1
	return Win

#Make process of the code which dislays the QT interface.
os.system("rm QTLauncher/QTLauncher.app/Contents/MacOS/QTLauncher")
os.chdir("QTLauncher")
os.system("qmake")
os.system("make")
print("")

#If the make process failed, the code stops and display the following error.
if(not os.path.exists("QTLauncher.app/Contents/MacOS/QTLauncher")):
	print("There's at least one error.")
	sys.exit()
else:
	os.system("./QTLauncher.app/Contents/MacOS/QTLauncher")

#Initial time t0 when the user validate the parameters.
t0 = time.time()

#Load the parameters files
Codes_W = np.loadtxt("Codes_W.txt", dtype='str')
Extra_W = np.loadtxt("Extra_W.txt", dtype='str')
XSAF_W = np.loadtxt("XSAF_W.txt", dtype='str')
SpecSAF_W = np.loadtxt("SpecSAF_W.txt", dtype='str')
Parameters_W = np.loadtxt("Parameters_W.txt", dtype='str')

#Call of the function Wid_org: all the parameters changed from the default parameters are erased to keep the last parameter entered.
Codes_W = Wid_org(Codes_W)
Extra_W = Wid_org(Extra_W)
XSAF_W = Wid_org(XSAF_W)
SpecSAF_W = Wid_org(SpecSAF_W)
Parameters_W = Wid_org(Parameters_W)

#Creation of 3 dictionaries. The user choices (Codes, Extra and XSAF choices) are saved in the in these dictionaries.
Codes_choices = {}
Extra_choices = {}
XSAF_choices = {}
for i in range(len(Codes_W)):
	Codes_choices[Codes_W[i][0]] = Codes_W[i][1]
for i in range(len(Extra_W)):
	Extra_choices[Extra_W[i][0]] = Extra_W[i][1]
for i in range(len(XSAF_W)):
	XSAF_choices[XSAF_W[i][0]] = XSAF_W[i][1]

#The modified parameters choices are saved.
C_out = open("Codes_W.txt", "w")
E_out = open("Extra_W.txt", "w")
X_out = open("XSAF_W.txt", "w")
S_out = open("SpecSAF_W.txt", "w")
P_out = open("Parameters_W.txt", "w")
for i in range(len(Codes_W)):
	C_out.write(Codes_W[i][0] + " " + Codes_W[i][1] + "\n")
for i in range(len(Extra_W)):
	E_out.write(Extra_W[i][0] + " " + Extra_W[i][1] + "\n")
for i in range(len(XSAF_W)):
	X_out.write(XSAF_W[i][0] + " " + XSAF_W[i][1] + "\n")
for i in range(len(SpecSAF_W)):
	S_out.write(SpecSAF_W[i][0] + " " + SpecSAF_W[i][1] + "\n")
for i in range(len(Parameters_W)):
	P_out.write(Parameters_W[i][0] + " " + Parameters_W[i][1] + "\n")
C_out.close()
E_out.close()
X_out.close()
S_out.close()
P_out.close()

#Building the Cl paths corresponding to all the parameters.
pre_CC_path = ["Cl_GG/", "Cl_LL/", "Cl_GL/"]
paths_tab = ["wb", "h", "wm", "ns", "wde", "w0", "wa", "s8", "gamma", "A_IA", "n_IA", "B_IA"]
for i in range(int(XSAF_choices["VRS_bins"])):
	paths_tab = np.insert(paths_tab, len(paths_tab), "b"+str(i+1))

os.chdir("../")

#Call XSAF C++ version.
if(Codes_choices["XSAF_ch"] == '0'):
	os.chdir("XSAF_C")
	#Create the Cl directories corresponding to all the parameters.
	for i in range(len(paths_tab)):
		CC_path = ["C_"+paths_tab[i]+"_up", "C_"+paths_tab[i]+"_up2", "C_fid", "C_"+paths_tab[i]+"_dw", "C_"+paths_tab[i]+"_dw2"]
		for j in range(len(CC_path)):
		    if not os.path.exists(pre_CC_path[0]+CC_path[j]):
		        os.makedirs(pre_CC_path[0]+CC_path[j])
		    if not os.path.exists(pre_CC_path[1]+CC_path[j]):
		        os.makedirs(pre_CC_path[1]+CC_path[j])
		    if not os.path.exists(pre_CC_path[2]+CC_path[j]):
		        os.makedirs(pre_CC_path[2]+CC_path[j])

	#Call Camb.
	if Codes_choices["Camb_ch"] == "0":
	    print("Calling Camb")
	    os.chdir('CAMB-0.1.7')
	    os.system('python Camb_launcher_XSAF.py') 
	    os.system('cp -r WP_Pk ../')
	    os.chdir('../')

	#XSAF compilation and execution.
	print("XSAF C++ version is compiling...")
	os.system("g++ -c main.cpp XSAF_C.cpp -fopenmp -mfma -O3 -ffast-math")
	os.system("g++ XSAF_C.o main.cpp -o main.exe -fopenmp -mfma -O3 -ffast-math")
	os.system("./main.exe")
	os.chdir("../")
else:
	print("XSAF won\'t be launched.")

#SpecSAF execution.
if(Codes_choices["SpecSAF_ch"] == '0'):
	os.chdir("SpecSAF")
	os.system("python GCs_forecast.py")
	os.chdir("../")
else:
	print("SpecSAF won\'t be launched.")

#Combine the probes if the user selected this option
if(Extra_choices["SavePrC_ch"] == "0"):
	print("The photometric and spectroscopic matrix will be combined if they are present in the output directories.")
	#The Fisher matrix are moved in the Matrix_combine directory.
	try:
		if os.path.exists("XSAF_C/output/Fisher_GCph_WL_XC_XSAF"):
			os.system("mv XSAF_C/output/Fisher_GCph_WL_XC_XSAF Matrix_combine/")
		if os.path.exists("XSAF_C/output/Fisher_GCph_XSAF"):
			os.system("mv XSAF_C/output/Fisher_GCph_XSAF Matrix_combine/")
		if os.path.exists("XSAF_C/output/Fisher_WL_XSAF"):
			os.system("mv XSAF_C/output/Fisher_WL_XSAF Matrix_combine/")
		if os.path.exists("SpecSAF/output/Fisher_GCs_SpecSAF"):
			os.system("mv SpecSAF/output/Fisher_GCs_SpecSAF Matrix_combine/")
	except:
		print("")
	os.chdir("Matrix_combine")
	os.system("python Probes_combine.py")
	os.chdir("../")

	#The Fisher matrix are moved in the Output directory.
	try:
		if os.path.exists("Matrix_combine/Fisher_GCph_WL_XC_XSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_WL_XC_XSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCph_XSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_XSAF Output")
		if os.path.exists("Matrix_combine/Fisher_WL_XSAF"):
			os.system("mv Matrix_combine/Fisher_WL_XSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCs_SpecSAF"):
			os.system("mv Matrix_combine/Fisher_GCs_SpecSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCph_GCs_TSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_GCs_TSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCph_GCs_WL_TSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_GCs_WL_TSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCph_GCs_WL_XC_TSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_GCs_WL_XC_TSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCs_WL_TSAF"):
			os.system("mv Matrix_combine/Fisher_GCs_WL_TSAF Output")
		if os.path.exists("Matrix_combine/Fisher_GCph_WL_XSAF"):
			os.system("mv Matrix_combine/Fisher_GCph_WL_XSAF Output")
	except:
		print("")
else:
	#The Fisher matrix are moved in the Output directory.
	if os.path.exists("XSAF_C/output/Fisher_GCph_WL_XC_XSAF"):
		os.system("mv XSAF_C/output/Fisher_GCph_WL_XC_XSAF Output/")
	if os.path.exists("XSAF_C/output/Fisher_GCph_XSAF"):
		os.system("mv XSAF_C/output/Fisher_GCph_XSAF Output/")
	if os.path.exists("XSAF_C/output/Fisher_WL_XSAF"):
		os.system("mv XSAF_C/output/Fisher_WL_XSAF Output/")
	if os.path.exists("SpecSAF/output/Fisher_GCs_SpecSAF"):
		os.system("mv SpecSAF/output/Fisher_GCs_SpecSAF Output/")
	print("No matrix combination requested.")


#Message if the user doesn't call any code.
if(Codes_choices["XSAF_ch"] == '1' and Codes_choices["SpecSAF_ch"] == '1'):
	print("")
	print("No modules was run. No matrix has been saved.")
	print("")
else:
	print("")
	print("Your matrices have been saved in the Output directory. If you want to plot some contours, you can use the Triplot directory.")
	print("")

#End time of the run and computation of the total time the codes ran.
t1 = time.time()
total = t1-t0

#End time of the run and computation of the total time the codes ran.
Ttime = str(datetime.timedelta(seconds=total))

#Display the time of the run.
print("Time(h:mm:ss.milli):", Ttime[:-3])
print("")
