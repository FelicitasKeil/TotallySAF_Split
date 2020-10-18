/*
Authors: S.Yahia-Cherif, F.Dournac.
Last Update 29/06/2020.
This is the main script of XSAF. XSAF computes the photometric Cl and the photometric Fisher matrix
*/

#include "XSAF_C.h"

using namespace std;

int main(){

    vector<string> elements_0; vector<double> elements_1;

    //Load the parameters files and stock them in dictionaries.
    int count=0;
    ifstream ifile("../QTLauncher/Parameters_W.txt");
    while(!ifile.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile>>elements_0[count];
        ifile>>elements_1[count];
        count++;
    }
    ifile.close();

    vector<int> PAR_X;
    int PAR_IND = 0;
    for(int i=0; i<elements_0.size(); i++){
        if(elements_0[i] != "Usesp_ch" && elements_0[i] != "Usesv_ch" && elements_0[i] != "UseGCspecbias_ch" && elements_0[i] != "UseGCspecPS_ch"){
            if(elements_0[i].find("Use") != string::npos && elements_1[i] == 0){
                PAR_X.push_back(PAR_IND);
            }
            PAR_IND++;
        }
    }

    ifstream ifile2("../QTLauncher/Codes_W.txt");
    while(!ifile2.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile2>>elements_0[count];
        ifile2>>elements_1[count];
        count++;
    }
    ifile2.close();
    ifstream ifile3("../QTLauncher/XSAF_W.txt");
    while(!ifile3.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile3>>elements_0[count];
        ifile3>>elements_1[count];
        count++;
    }
    ifile3.close();

    ifstream ifile4("../QTLauncher/Extra_W.txt");
    while(!ifile4.fail()){
        elements_0.push_back("");
        elements_1.push_back(0.0);
        ifile4>>elements_0[count];
        ifile4>>elements_1[count];
        count++;
    }
    ifile4.close();

    map<string, double> XSAF_elts;
    for(int i=0; i<elements_0.size(); i++){
        XSAF_elts[elements_0[i]] = double(elements_1[i]);
    }

    PAR_IND = PAR_X[PAR_X.size()-1]+1;
    for(int i=PAR_IND; i<PAR_IND + XSAF_elts["VRS_bins"]-1; i++){
        PAR_X.push_back(i);
    }

    //Reading curvature, zcut and gamma settings.
    string zcut, curv, gma = "";
    if(XSAF_elts["FNF_ch"] == 0){
        curv = "F";
    }
    else{
        curv = "NF";
    }
    if(XSAF_elts["zcut_ch"] == 1){
        zcut = "N";
    }
    else{
        zcut = "Y";
    }
    if(XSAF_elts["Usegamma_ch"] == 1){
        gma = "N";
    }
    else{
        gma = "Y";
    }

    //Call default constructor.
    XC XC(XSAF_elts["VRS_bins"], 5, XSAF_elts["Vlnum"], XSAF_elts["Vlmin_GCWL"], XSAF_elts["Vlmax_WL"], XSAF_elts["Vlmax_GC"], 60, XSAF_elts["Vprec_Int_z"], 3, PAR_X.size(), curv, gma, zcut);

    //Call the methods to compute the Cl.
	for(int X_ind=0; X_ind<PAR_X.size(); X_ind++){
        XC.Initializers_G(PAR_X[X_ind]);
        XC.Initializers_Pk();
        XC.background();
        if(X_ind==0 && XSAF_elts["UsePZ_ch"] == 0){
            XC.photoz();
        }
    	XC.photoz_load();
    	XC.windows();
    	XC.C_l_computing();
    }

    //Call the method to compute the Fisher matrix.
    int run_index = 0;
    for(int i=0; i<XSAF_elts["Cutting_l_V"]; i++){
        if(i == XSAF_elts["Cutting_l_V"]-1){
            run_index = 1;
        }
        if(XSAF_elts["UseXC_ch"] == 0){
            XC.Fisher("XC", "Fisher_GCph_WL_XC_XSAF", XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*i, XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*(i+1), run_index, i, XSAF_elts["Cutting_l_V"]);
        }
        if(XSAF_elts["UseWL_ch"] == 0){
            XC.Fisher("WL", "Fisher_WL_XSAF", XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*i, XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*(i+1), run_index, i, XSAF_elts["Cutting_l_V"]);
        }
        if(XSAF_elts["UseGC_ch"] == 0){
            XC.Fisher("GCp", "Fisher_GCph_XSAF", XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*i, XSAF_elts["Vlnum"]/XSAF_elts["Cutting_l_V"]*(i+1), run_index, i, XSAF_elts["Cutting_l_V"]);
        }
    }
    //Suppress the sub Fishers after the total Fisher matrix is built.
    system("rm output/Fisher_GCph_WL_XC_XSAF_*");
    system("rm output/Fisher_GCph_XSAF_*");
    system("rm output/Fisher_WL_XSAF_*");
    cout<<endl<<endl;
    return 0;
}