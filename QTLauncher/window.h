#ifndef WINDOW_H
#define WINDOW_H

//Import libraries
#include <QWidget>
#include <QDebug>

#include<QDoubleSpinBox>
#include<QGroupBox>
#include<QCheckBox>
#include<QComboBox>
#include<QLabel>
#include<QPushButton>
#include<QDialog>
#include<QTextStream>
#include<QDialogButtonBox>
#include<QVBoxLayout>
#include<QHBoxLayout>
#include<QGridLayout>

//Class Window.
class Window : public QWidget{
    Q_OBJECT

public:
    Window();

private:
    //Main sub window 
    void Codes();
    void Extra();
    void XSAF();
    void SpecSAF();
    void Parameters_set();
    void ValidEx();

    //Declaration of all the widgets names to put in the main sub windows.
    QDoubleSpinBox *RS_bins;
    QDoubleSpinBox *zmin, *zmax;
    QDoubleSpinBox *lmin_GCWL, *lmax_GC, *lmax_WL, *lnum_WL, *lnum;
    QDoubleSpinBox *Gdensity, *SurfArea, *prec_Int_z, *UseCuttingl;
    QComboBox *UseXSAF, *UseSpecSAF, *UseCamb, *UseGC, *UseWL, *UseXC, *UsePZ, *flat_nonflat, *SavePrC;
    QComboBox *zcut;
    QDoubleSpinBox *zrange_0, *alpha, *photoz_cb, *photoz_zb, *photoz_sigb, *photoz_c0, *photoz_z0, *photoz_sig0, *photoz_f_out, *sig_epsilon;

    QDoubleSpinBox *RS_binsSP, *zminSP, *zmaxSP, *Mean_binSP, *kminSP, *kmaxSP;
    QDoubleSpinBox *CMB_TSP, *N_speciesSP;
    QDoubleSpinBox *SurfAreaSP, *SpectroprecSP, *DerprojSP, *IntegprecSP, *ODEPrecSP;
    QComboBox *LNLcaseSP, *SNLXcaseSP, *ODEfitSP, *DerMethodSP, *biasvslnbiasSP;

    QComboBox *UseOmegam, *UseOmegaDE, *UseOmegab, *Usew0, *Usewa, *Useh, *Usens, *Usesigma8, *Usegamma;
    QComboBox *Usesp, *Usesv, *UseGCspecbias, *UseGCspecPS, *UseGCphotbias, *UseWLAia, *UseWLBia, *UseWLnia;
    QDoubleSpinBox *FidOmegam, *FidOmegaDE, *FidOmegab, *Fidw0, *Fidwa, *Fidh, *Fidns, *Fidsigma8, *Fidgamma;
    QDoubleSpinBox *Fidsp, *Fidsv, *FidWLAia, *FidWLBia, *FidWLnia, *FidWLCia;
    QDoubleSpinBox *FidAs, *FidWnu;
    QDoubleSpinBox *StepOmegam, *StepOmegaDE, *StepOmegab, *Stepw0, *Stepwa, *Steph, *Stepns, *Stepsigma8, *Stepgamma;
    QDoubleSpinBox *Stepsp, *Stepsv, *StepGCspecbias, *StepGCspecPS, *StepGCphotbias, *StepWLAia, *StepWLBia, *StepWLnia;
    QDoubleSpinBox *SteplnDa, *SteplnH, *Steplnfs8;
    QDoubleSpinBox *priceSpinBox;
    QDoubleSpinBox *scaleSpinBox;
    QGroupBox *CodesGroup;
    QGroupBox *ExtraGroup;
    QGroupBox *XSAFGroup;
    QGroupBox *editsGroup;
    QGroupBox *SpecSAFGroup;
    QGroupBox *ParametersGroup;
    QGroupBox *ValidExGroup;
    QPushButton *Confirm, *Cancel;
    QDialog *dialog;
};

#endif
