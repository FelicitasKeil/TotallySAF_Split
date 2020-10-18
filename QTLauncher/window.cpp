#include <QtGui>
#include "window.h"

//Window method: Build all the sub windows.
Window::Window(){
 	Codes();
 	Extra();
    XSAF();
    SpecSAF();
    Parameters_set();
    ValidEx();

    //Layout creation.
    QGridLayout *layout = new QGridLayout;
    //The sub windows are built.
    layout->addWidget(CodesGroup, 1, 0);
    layout->addWidget(XSAFGroup, 0, 0);
    layout->addWidget(SpecSAFGroup, 0, 1);
    layout->addWidget(ParametersGroup, 0, 2);
    layout->addWidget(ValidExGroup, 1, 1);
    layout->addWidget(ExtraGroup, 1, 2);
    setLayout(layout);

    setWindowTitle(tr("TotallySAF"));
}

/*Building the Widgets in the Codes windows.
Step 1: The default parameters values are declared in variables and saved in a parameter file.
Step 2: The Spinebox variables are created.
Step 3: The widget are placed in their subwindows layout with their labels.
Each time the user change the parameter value, the new value is written in the parameter file (see connect).
*/
void Window::Codes(){

    int XSAF_ch = 0; int SpecSAF_ch = 0; int FNF_ch = 0; int Camb_ch = 1;

    QString filename="Codes_W.txt";
    QFile file(filename);
    if (file.open(QIODevice::ReadWrite | QIODevice::Text)){
        QTextStream stream( &file );
        stream << "XSAF_ch "<< XSAF_ch << endl;
        stream << "SpecSAF_ch "<< SpecSAF_ch << endl;
        stream << "FNF_ch "<< FNF_ch << endl;
        stream << "Camb_ch "<< Camb_ch << endl;
    }

	CodesGroup = new QGroupBox(tr("Modules and Cosmology"));
	
    QLabel *UseXSAFLabel = new QLabel(tr("Use XSAF:"));
    QComboBox *UseXSAF = new QComboBox(this);
    UseXSAF->addItem("Yes");
    UseXSAF->addItem("No");

    connect(UseXSAF, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename="Codes_W.txt";
        QFile file(filename);
        if (file.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream( &file );
            stream << "XSAF_ch " << index << endl;
        }
     });
     
	QLabel *UseSpecSAFLabel = new QLabel(tr("Use SpecSAF:"));
    QComboBox *UseSpecSAF = new QComboBox(this);
    UseSpecSAF->addItem("Yes");
    UseSpecSAF->addItem("No");

    connect(UseSpecSAF, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename="Codes_W.txt";
        QFile file(filename);
        if (file.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream( &file );
            stream << "SpecSAF_ch " << index << endl;
        }
     });

    QLabel *UseCambLabel = new QLabel(tr("Use Camb:"));
    QComboBox *UseCamb = new QComboBox(this);
    UseCamb->addItem("Yes");
    UseCamb->addItem("No");
    UseCamb->setCurrentIndex(UseCamb->findText("No"));
    
    connect(UseCamb, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename="Codes_W.txt";
        QFile file(filename);
        if (file.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream( &file );
            stream << "Camb_ch " << index << endl;
        }
     });

    QLabel *flat_nonflatLabel = new QLabel(tr("Flat or non flat case:"));
    QComboBox *flat_nonflat = new QComboBox(this);
    flat_nonflat->addItem("Flat");
    flat_nonflat->addItem("Non flat");
    
    connect(flat_nonflat, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
     	QString filename="Codes_W.txt";
		QFile file(filename);
	    if (file.open(QIODevice::ReadWrite | QIODevice::Append)){
	        QTextStream stream( &file );
	        stream << "FNF_ch " << index << endl;
		}
     });

    QGridLayout *spinBoxLayout = new QGridLayout;
    spinBoxLayout->addWidget(UseXSAFLabel, 0, 0);
    spinBoxLayout->addWidget(UseXSAF, 0, 1);
    spinBoxLayout->addWidget(UseSpecSAFLabel, 0, 2);
    spinBoxLayout->addWidget(UseSpecSAF, 0, 3);
    spinBoxLayout->addWidget(flat_nonflatLabel, 1, 0);
    spinBoxLayout->addWidget(flat_nonflat, 1, 1);
    spinBoxLayout->addWidget(UseCambLabel, 1, 2);
    spinBoxLayout->addWidget(UseCamb, 1, 3);

    CodesGroup->setLayout(spinBoxLayout);
}

/*Building the Widgets in the Extra windows.
Step 1: The default parameters values are declared in variables and saved in a parameter file.
Step 2: The Spinebox variables are created.
Step 3: The widget are placed in their subwindows layout with their labels.
Each time the user change the parameter value, the new value is written in the parameter file (see connect).
*/
void Window::Extra(){

    int Cutting_l_V = 60; int SavePrC_ch = 0;

    QString filename2="Extra_W.txt";
    QFile file2(filename2);
    if (file2.open(QIODevice::ReadWrite | QIODevice::Text)){
        QTextStream stream2( &file2);
        stream2 << "Cutting_l_V "<< Cutting_l_V << endl;
        stream2 << "SavePrC_ch "<< SavePrC_ch << endl;
    }

	ExtraGroup = new QGroupBox(tr("Extra options"));

    QLabel *UseCuttinglLabel = new QLabel(tr("l cut optimization:"));
    UseCuttingl = new QDoubleSpinBox;
    UseCuttingl->setDecimals(0);
    UseCuttingl->setRange(1.000, 100000.000); UseCuttingl->setSingleStep(1.000); UseCuttingl->setValue(Cutting_l_V);

    connect(UseCuttingl, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename2="Extra_W.txt";
        QFile file2(filename2);
        if (file2.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream2( &file2 );
            stream2 << "Cutting_l_V "<< UseCuttingl->value() << endl;
        }
    });

    QLabel *SavePrCLabel = new QLabel(tr("Combine the probes:"));
    QComboBox *SavePrC = new QComboBox(this);
    SavePrC->addItem("Yes");
    SavePrC->addItem("No");
    SavePrC->setCurrentIndex(SavePrC->findText("Yes"));

    connect(SavePrC, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged),
     [=](int index){
        QString filename2="Extra_W.txt";
        QFile file2(filename2);
        if (file2.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream2( &file2 );
            stream2 << "SavePrC_ch " << index << endl;
        }
     });

    QGridLayout *spinBoxLayout = new QGridLayout;
	spinBoxLayout->addWidget(UseCuttinglLabel, 0, 0);
    spinBoxLayout->addWidget(UseCuttingl, 0, 1);
    spinBoxLayout->addWidget(SavePrCLabel, 1, 0);
    spinBoxLayout->addWidget(SavePrC, 1, 1);

	ExtraGroup->setLayout(spinBoxLayout);

}

/*Building the Widgets in the XSAF windows.
Step 1: The default parameters values are declared in variables and saved in a parameter file.
Step 2: The Spinebox variables are created.
Step 3: The widget are placed in their subwindows layout with their labels.
Each time the user change the parameter value, the new value is written in the parameter file (see connect).
*/
void Window::XSAF(){

    int UseGC_ch = 0; int UseWL_ch = 0; int UseXC_ch = 0; int UsePZ_ch = 1; int VRS_bins = 10; double Vzmin = 0.001; double Vzmax = 3.731;
    int Vlnum = 60; double Vlmin_GCWL = 10; double Vlmax_GC = 3000; double Vlmax_WL = 5000; double VSurfArea = 15000; double VGdensity = 30.0; int Vprec_Int_z = 100;
    int zcut_ch = 1; double Vzrange_0 = 0.9/pow(2, 0.5); double Valpha = 1.5; double Vsig_epsilon = 0.3; double Vphotoz_cb = 1.0;
    double Vphotoz_zb = 0.0; double Vphotoz_sigb = 0.05; double Vphotoz_c0 = 1.0; double Vphotoz_z0 = 0.1; double Vphotoz_sig0 = 0.05; double Vphotoz_f_out = 0.1;
    
    QString filename3="XSAF_W.txt";
    QFile file3(filename3);
    if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
        QTextStream stream3( &file3 );

        stream3 << "UseGC_ch "<< UseGC_ch << endl;
        stream3 << "UseWL_ch "<< UseWL_ch << endl;
        stream3 << "UseXC_ch "<< UseXC_ch << endl;
        stream3 << "VRS_bins "<< VRS_bins << endl;
        stream3 << "Vzmin "<< Vzmin << endl;
        stream3 << "Vzmax "<< Vzmax << endl;
        stream3 << "Vlnum "<< Vlnum << endl;
        stream3 << "Vlmin_GCWL "<< Vlmin_GCWL << endl;
        stream3 << "Vlmax_GC "<< Vlmax_GC << endl;
        stream3 << "Vlmax_WL "<< Vlmax_WL << endl;
        stream3 << "VSurfArea "<< VSurfArea << endl;
        stream3 << "VGdensity "<< VGdensity << endl;
        stream3 << "Vprec_Int_z "<< Vprec_Int_z << endl;
        stream3 << "zcut_ch "<< zcut_ch << endl;
        stream3 << "Vsig_epsilon "<< Vsig_epsilon << endl;

        stream3 << "UsePZ_ch "<< UsePZ_ch << endl;
        stream3 << "Vzrange_0 "<< Vzrange_0 << endl;
        stream3 << "Valpha "<< Valpha << endl;
        stream3 << "Vphotoz_cb "<< Vphotoz_cb << endl;
        stream3 << "Vphotoz_zb "<< Vphotoz_zb << endl;
        stream3 << "Vphotoz_sigb "<< Vphotoz_sigb << endl;
        stream3 << "Vphotoz_c0 "<< Vphotoz_c0 << endl;
        stream3 << "Vphotoz_z0 "<< Vphotoz_z0 << endl;
        stream3 << "Vphotoz_sig0 "<< Vphotoz_sig0 << endl;
        stream3 << "Vphotoz_f_out "<< Vphotoz_f_out << endl;
    }
     
    XSAFGroup = new QGroupBox(tr("XSAF Module"));

    QLabel *UseGCLabel = new QLabel(tr("Photometric Galaxy Clustering (GCp):"));
    QComboBox *UseGC = new QComboBox(this);
    UseGC->addItem("Yes");
    UseGC->addItem("No");

     connect(UseGC, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "UseGC_ch " << index << endl;
        }
     });

    QLabel *UseWLLabel = new QLabel(tr("Weak Lensing (WL):"));
    QComboBox *UseWL = new QComboBox(this);
    UseWL->addItem("Yes");
    UseWL->addItem("No");

    connect(UseWL, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "UseWL_ch " << index << endl;
        }
     });

    QLabel *UseXCLabel = new QLabel(tr("Cross-Correlations (XC:GCp+WL+XC):"));
    QComboBox *UseXC = new QComboBox(this);
    UseXC->addItem("Yes");
    UseXC->addItem("No");

    connect(UseXC, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "UseXC_ch " << index << endl;
        }
    });

    QLabel *RS_bins_label = new QLabel(tr("Number of redshift bins:"));
    RS_bins = new QDoubleSpinBox;
    RS_bins->setDecimals(0);
    RS_bins->setRange(1.000, 100.000); RS_bins->setSingleStep(1.000); RS_bins->setValue(VRS_bins);

    connect(RS_bins, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "VRS_bins "<< RS_bins->value() << endl;
        }
    });

    QLabel *zmin_bins_label = new QLabel(tr("Redshift min:"));
    zmin = new QDoubleSpinBox;
    zmin->setDecimals(8);
    zmin->setRange(0.0, 1000000); zmin->setSingleStep(0.001); zmin->setValue(Vzmin);

    connect(zmin, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vzmin "<< zmin->value() << endl;
        }
    });

    QLabel *zmax_bins_label = new QLabel(tr("Redshift max:"));
    zmax = new QDoubleSpinBox;
    zmax->setDecimals(8);
    zmax->setRange(0.0, 1000000); zmax->setSingleStep(0.001); zmax->setValue(Vzmax);

    connect(zmax, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vzmax "<< zmax->value() << endl;
        }
    });

     QLabel *lnum_bins_label = new QLabel(tr("Number of multipoles l:"));
     lnum = new QDoubleSpinBox;
     lnum->setDecimals(0);
     lnum->setRange(0.0, 100000); lnum->setSingleStep(1.0); lnum->setValue(Vlnum);

    connect(lnum, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vlnum "<< lnum->value() << endl;
        }
    });

     QLabel *lmin_GCWL_bins_label = new QLabel(tr("l min (GCp and WL):"));
     lmin_GCWL = new QDoubleSpinBox;
     lmin_GCWL->setDecimals(8);
     lmin_GCWL->setRange(0.0, 1000000); lmin_GCWL->setSingleStep(1.0); lmin_GCWL->setValue(Vlmin_GCWL);

    connect(lmin_GCWL, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vlmin_GCWL "<< lmin_GCWL->value() << endl;
        }
    });

     QLabel *lmax_GC_bins_label = new QLabel(tr("l max GCphot:"));
     lmax_GC = new QDoubleSpinBox;
     lmax_GC->setDecimals(8);
     lmax_GC->setRange(0.0, 1000000); lmax_GC->setSingleStep(1.0); lmax_GC->setValue(Vlmax_GC);

    connect(lmax_GC, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vlmax_GC "<< lmax_GC->value() << endl;
        }
    });

    QLabel *lmax_WL_bins_label = new QLabel(tr("l max WL:"));
    lmax_WL = new QDoubleSpinBox;
    lmax_WL->setDecimals(8);
    lmax_WL->setRange(0.0, 1000000); lmax_WL->setSingleStep(1.0); lmax_WL->setValue(Vlmax_WL);

    connect(lmax_WL, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vlmax_WL "<< lmax_WL->value() << endl;
        }
    });

    QLabel *SurfAreaLabel = new QLabel(tr("Surface Area (deg sq):"));
    SurfArea = new QDoubleSpinBox;
    SurfArea->setDecimals(8);
    SurfArea->setRange(0.0,41253.0); SurfArea->setSingleStep(1.); SurfArea->setValue(VSurfArea);

    connect(SurfArea, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "VSurfArea "<< SurfArea->value() << endl;
        }
    });

    QLabel *GdensityLabel = new QLabel(tr("Galaxy density (arcmin^{-2}):"));
    Gdensity = new QDoubleSpinBox;
    Gdensity->setDecimals(8);
    Gdensity->setRange(0.0,1000000000); Gdensity->setSingleStep(1.); Gdensity->setValue(VGdensity);

    connect(Gdensity, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "VGdensity "<< Gdensity->value() << endl;
        }
    });

    QLabel *prec_Int_z_label = new QLabel(tr("Number of steps (redshift integrals):"));
    prec_Int_z = new QDoubleSpinBox;
    prec_Int_z->setDecimals(0);
    prec_Int_z->setRange(0.0, 1000000); prec_Int_z->setSingleStep(1.0); prec_Int_z->setValue(Vprec_Int_z);

    connect(prec_Int_z, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vprec_Int_z "<< prec_Int_z->value() << endl;
        }
    });

    QLabel *zcutLabel = new QLabel(tr("Use zcut option at z>0.9:"));
    QComboBox *zcut = new QComboBox(this);
    zcut->addItem("Yes");
    zcut->addItem("No");
    zcut->setCurrentIndex(zcut->findText("No"));

    connect(zcut, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "zcut_ch " << index << endl;
        }
     });

    QLabel *sig_epsilonLabel = new QLabel(tr("Observed ellipticities variance:"));
    sig_epsilon = new QDoubleSpinBox;
    sig_epsilon->setDecimals(8);
    sig_epsilon->setRange(0.0,1.0); sig_epsilon->setSingleStep(0.001); sig_epsilon->setValue(Vsig_epsilon);

    connect(sig_epsilon, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vsig_epsilon "<< sig_epsilon->value() << endl;
        }
    });
	
    QLabel *UsePZLabel = new QLabel(tr("Compute Photo-z:"));
    QComboBox *UsePZ = new QComboBox(this);
    UsePZ->addItem("Yes");
    UsePZ->addItem("No");
    UsePZ->setCurrentIndex(UsePZ->findText("No"));
    
    connect(UsePZ, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename="XSAF_W.txt";
        QFile file(filename);
        if (file.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream( &file );
            stream << "UsePZ_ch " << index << endl;
        }
     });

    QLabel *zrange_0Label = new QLabel(tr("z0"));
    zrange_0 = new QDoubleSpinBox;
    zrange_0->setDecimals(8);
    zrange_0->setRange(0.0,100.0); zrange_0->setSingleStep(0.001); zrange_0->setValue(Vzrange_0);

    connect(zrange_0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vzrange_0 "<< zrange_0->value() << endl;
        }
    });

    QLabel *alphaLabel = new QLabel(tr("alpha:"));
    alpha = new QDoubleSpinBox;
    alpha->setDecimals(8);
    alpha->setRange(0.0,100.0); alpha->setSingleStep(0.001); alpha->setValue(Valpha);

    connect(alpha, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Valpha "<< alpha->value() << endl;
        }
    });

    QLabel *photoz_cbLabel = new QLabel(tr("cb:"));
    photoz_cb = new QDoubleSpinBox;
    photoz_cb->setDecimals(8);
    photoz_cb->setRange(0.0,1.0); photoz_cb->setSingleStep(0.001); photoz_cb->setValue(Vphotoz_cb);

    connect(photoz_cb, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_cb "<< photoz_cb->value() << endl;
        }
    });

     QLabel *photoz_zbLabel = new QLabel(tr("zb:"));
     photoz_zb = new QDoubleSpinBox;
     photoz_zb->setDecimals(8);
     photoz_zb->setRange(0.0,1.0); photoz_zb->setSingleStep(0.001); photoz_zb->setValue(Vphotoz_zb);

    connect(photoz_zb, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_zb "<< photoz_zb->value() << endl;
        }
    });

    QLabel *photoz_sigbLabel = new QLabel(tr("sig_b:"));
    photoz_sigb = new QDoubleSpinBox;
    photoz_sigb->setDecimals(8);
    photoz_sigb->setRange(0.0,1.0); photoz_sigb->setSingleStep(0.001); photoz_sigb->setValue(Vphotoz_sigb);

    connect(photoz_sigb, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_sigb "<< photoz_sigb->value() << endl;
        }
    });

    QLabel *photoz_c0Label = new QLabel(tr("co:"));
    photoz_c0 = new QDoubleSpinBox;
    photoz_c0->setDecimals(8);
    photoz_c0->setRange(0.0,1.0); photoz_c0->setSingleStep(0.001); photoz_c0->setValue(Vphotoz_c0);

    connect(photoz_c0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_c0 "<< photoz_c0->value() << endl;
        }
    });
     
    QLabel *photoz_z0Label = new QLabel(tr("zo:"));
    photoz_z0 = new QDoubleSpinBox;
    photoz_z0->setDecimals(8);
    photoz_z0->setRange(0.0,1.0); photoz_z0->setSingleStep(0.001); photoz_z0->setValue(Vphotoz_z0);

    connect(photoz_z0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_z0 "<< photoz_z0->value() << endl;
        }
    });

    QLabel *photoz_sig0Label = new QLabel(tr("sig_o:"));
    photoz_sig0 = new QDoubleSpinBox;
    photoz_sig0->setDecimals(8);
    photoz_sig0->setRange(0.0,1.0); photoz_sig0->setSingleStep(0.001); photoz_sig0->setValue(Vphotoz_sig0);

    connect(photoz_sig0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_sig0 "<< photoz_sig0->value() << endl;
        }
    });

    QLabel *photoz_f_outLabel = new QLabel(tr("fraction outliers:"));
    photoz_f_out = new QDoubleSpinBox;
    photoz_f_out->setDecimals(8);
    photoz_f_out->setRange(0.0,1.0); photoz_f_out->setSingleStep(0.001); photoz_f_out->setValue(Vphotoz_f_out);

    connect(photoz_f_out, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename3="XSAF_W.txt";
        QFile file3(filename3);
        if (file3.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream3( &file3 );
            stream3 << "Vphotoz_f_out "<< photoz_f_out->value() << endl;
        }
    });

    QGridLayout *spinBoxLayout = new QGridLayout;
    spinBoxLayout->addWidget(UseGCLabel, 0, 0);
    spinBoxLayout->addWidget(UseGC, 0, 1);
    spinBoxLayout->addWidget(UseWLLabel, 1, 0);
    spinBoxLayout->addWidget(UseWL, 1, 1);
    spinBoxLayout->addWidget(UseXCLabel, 2, 0);
    spinBoxLayout->addWidget(UseXC, 2, 1);
    spinBoxLayout->addWidget(RS_bins_label, 3, 0);
    spinBoxLayout->addWidget(RS_bins, 3, 1);
    spinBoxLayout->addWidget(zmin_bins_label, 4, 0);
    spinBoxLayout->addWidget(zmin, 4, 1);
    spinBoxLayout->addWidget(zmax_bins_label, 5, 0);
    spinBoxLayout->addWidget(zmax, 5, 1);
    spinBoxLayout->addWidget(lnum_bins_label, 6, 0);
    spinBoxLayout->addWidget(lnum, 6, 1);
    spinBoxLayout->addWidget(lmin_GCWL_bins_label, 7, 0);
    spinBoxLayout->addWidget(lmin_GCWL, 7, 1);
    spinBoxLayout->addWidget(lmax_GC_bins_label, 8, 0);
    spinBoxLayout->addWidget(lmax_GC, 8, 1);
    spinBoxLayout->addWidget(lmax_WL_bins_label, 10, 0);
    spinBoxLayout->addWidget(lmax_WL, 10, 1);
    spinBoxLayout->addWidget(SurfAreaLabel, 11, 0);
    spinBoxLayout->addWidget(SurfArea, 11, 1);
    spinBoxLayout->addWidget(GdensityLabel, 12, 0);
    spinBoxLayout->addWidget(Gdensity, 12, 1);
    spinBoxLayout->addWidget(prec_Int_z_label, 13, 0);
    spinBoxLayout->addWidget(prec_Int_z, 13, 1);
    spinBoxLayout->addWidget(zcutLabel, 14, 0);
    spinBoxLayout->addWidget(zcut, 14, 1);
    spinBoxLayout->addWidget(sig_epsilonLabel, 15, 0);
    spinBoxLayout->addWidget(sig_epsilon, 15, 1);
    spinBoxLayout->addWidget(UsePZLabel, 16, 0);
    spinBoxLayout->addWidget(UsePZ, 16, 1);
    spinBoxLayout->addWidget(zrange_0Label, 17, 0);
    spinBoxLayout->addWidget(zrange_0, 17, 1);
    spinBoxLayout->addWidget(alphaLabel, 18, 0);
    spinBoxLayout->addWidget(alpha, 18, 1);
    spinBoxLayout->addWidget(photoz_cbLabel, 19, 0);
    spinBoxLayout->addWidget(photoz_cb, 19, 1);
    spinBoxLayout->addWidget(photoz_zbLabel, 20, 0);
    spinBoxLayout->addWidget(photoz_zb, 20, 1);
    spinBoxLayout->addWidget(photoz_sigbLabel, 21, 0);
    spinBoxLayout->addWidget(photoz_sigb, 21, 1);
    spinBoxLayout->addWidget(photoz_c0Label, 22, 0);
    spinBoxLayout->addWidget(photoz_c0, 22, 1);
    spinBoxLayout->addWidget(photoz_z0Label, 23, 0);
    spinBoxLayout->addWidget(photoz_z0, 23, 1);
    spinBoxLayout->addWidget(photoz_sig0Label, 24, 0);
    spinBoxLayout->addWidget(photoz_sig0, 24, 1);
    spinBoxLayout->addWidget(photoz_f_outLabel, 25, 0);
    spinBoxLayout->addWidget(photoz_f_out, 25, 1);
     
    XSAFGroup->setLayout(spinBoxLayout);
}

/*Building the Widgets in the SpecSAF windows.
Step 1: The default parameters values are declared in variables and saved in a parameter file.
Step 2: The Spinebox variables are created.
Step 3: The widget are placed in their subwindows layout with their labels.
Each time the user change the parameter value, the new value is written in the parameter file (see connect).
*/
void Window::SpecSAF(){

    int redbinsSP = 5; double VzminSP = 0.9; double VzmaxSP = 1.8; int VMean_binSP = 3; double VkminSP = 0.001; double VkmaxSP = 0.3;
    int LNLcaseSP_ch = 1; int SNLXcaseSP_ch = 1; int ODEfitSP_ch = 0; double VSurfAreaSP = 15000; double VSpectroprecSP = 0.001;
    int DerMethodSP_ch = 1; double VIntegprecSP = 100; double VDerprojSP = 0.001; double VODEPrecSP = 1000;
    int biasvslnbiasSP_ch = 0; double VCMB_TSP = 2.7255; double VN_speciesSP = 1.0;

    QString filename4="SpecSAF_W.txt";
    QFile file4(filename4);
    if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
        QTextStream stream4( &file4 );

        stream4 << "redbinsSP "<< redbinsSP << endl;
        stream4 << "VzminSP "<< VzminSP << endl;
        stream4 << "VzmaxSP "<< VzmaxSP << endl;
        stream4 << "VMean_binSP "<< VMean_binSP << endl;
        stream4 << "VkminSP "<< VkminSP << endl;
        stream4 << "VkmaxSP "<< VkmaxSP << endl;
        stream4 << "LNLcaseSP_ch "<< LNLcaseSP_ch << endl;
        stream4 << "SNLXcaseSP_ch "<< SNLXcaseSP_ch << endl;
        stream4 << "ODEfitSP_ch "<< ODEfitSP_ch << endl;
        stream4 << "VCMB_TSP "<< VCMB_TSP << endl;
        stream4 << "VN_speciesSP "<< VN_speciesSP << endl;
        stream4 << "VSurfAreaSP "<< VSurfAreaSP << endl;
        stream4 << "VSpectroprecSP "<< VSpectroprecSP << endl;
        stream4 << "DerMethodSP_ch "<< DerMethodSP_ch << endl;
        stream4 << "VIntegprecSP "<< VIntegprecSP << endl;
        stream4 << "VDerprojSP "<< VDerprojSP << endl;
        stream4 << "VODEPrecSP "<< VODEPrecSP << endl;
        stream4 << "biasvslnbiasSP_ch "<< biasvslnbiasSP_ch << endl;
    }

    SpecSAFGroup = new QGroupBox(tr("SpecSAF Module"));

    QLabel *RS_bins_labelSP = new QLabel(tr("Number of redshift bins:"));
    RS_binsSP = new QDoubleSpinBox;
    RS_binsSP->setDecimals(0);
    RS_binsSP->setRange(1.000, 100.000); RS_binsSP->setSingleStep(1.000); RS_binsSP->setValue(redbinsSP);

    connect(RS_binsSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "redbinsSP "<< RS_binsSP->value() << endl;
        }
    });
    
    QLabel *zmin_bins_labelSP = new QLabel(tr("Redshift min:"));
    zminSP = new QDoubleSpinBox;
    zminSP->setDecimals(8);
    zminSP->setRange(0.0, 1000000); zminSP->setSingleStep(0.001); zminSP->setValue(VzminSP);

    connect(zminSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VzminSP "<< zminSP->value() << endl;
        }
    });

    QLabel *zmax_bins_labelSP = new QLabel(tr("Redshift max:"));
    zmaxSP = new QDoubleSpinBox;
    zmaxSP->setDecimals(8);
    zmaxSP->setRange(0.0, 1000000); zmaxSP->setSingleStep(0.001); zmaxSP->setValue(VzmaxSP);

    connect(zmaxSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VzmaxSP "<< zmaxSP->value() << endl;
        }
    });

    QLabel *Mean_bin_LabelSP = new QLabel(tr("Mean redshift bin:"));
    Mean_binSP = new QDoubleSpinBox;
    Mean_binSP->setDecimals(0);
    Mean_binSP->setRange(1.0, 100); Mean_binSP->setSingleStep(1.0); Mean_binSP->setValue(VMean_binSP);

    connect(Mean_binSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VMean_binSP "<< Mean_binSP->value() << endl;
        }
    });

    QLabel *kmin_bins_labelSP = new QLabel(tr("min scale k (h/Mpc):"));
    kminSP = new QDoubleSpinBox;
    kminSP->setDecimals(8);
    kminSP->setRange(0.0, 1000000); kminSP->setSingleStep(0.001); kminSP->setValue(VkminSP);

    connect(kminSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VkminSP "<< kminSP->value() << endl;
        }
    });

    QLabel *kmax_bins_labelSP = new QLabel(tr("max scale k (h/Mpc):"));
    kmaxSP = new QDoubleSpinBox;
    kmaxSP->setDecimals(8);
    kmaxSP->setRange(0.0, 1000000); kmaxSP->setSingleStep(0.001); kmaxSP->setValue(VkmaxSP);

    connect(kmaxSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VkmaxSP "<< kmaxSP->value() << endl;
        }
    });

    QLabel *LNLcaseLabelSP = new QLabel(tr("Linear or non Linear P(k):"));
    QComboBox *LNLcaseSP = new QComboBox(this);
    LNLcaseSP->addItem("L");
    LNLcaseSP->addItem("SNL");
    LNLcaseSP->setCurrentIndex(LNLcaseSP->findText("SNL"));

    connect(LNLcaseSP, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream4( &file4 );
            stream4 << "LNLcaseSP_ch " << index << endl;
        }
     });

    QLabel *SNLXcaseLabelSP = new QLabel(tr("Power spectrum case:"));
    QComboBox *SNLXcaseSP = new QComboBox(this);
    SNLXcaseSP->addItem("SNL1");
    SNLXcaseSP->addItem("SNL2");
    SNLXcaseSP->setCurrentIndex(SNLXcaseSP->findText("SNL2"));

    connect(SNLXcaseSP, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream4( &file4 );
            stream4 << "SNLXcaseSP_ch " << index << endl;
        }
     });

    QLabel *CMB_T_LabelSP = new QLabel(tr("CMB temperature (NW P(k)):"));
    CMB_TSP = new QDoubleSpinBox;
    CMB_TSP->setDecimals(8);
    CMB_TSP->setRange(0.0, 1000000); CMB_TSP->setSingleStep(0.001); CMB_TSP->setValue(VCMB_TSP);

    connect(CMB_TSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VCMB_TSP "<< CMB_TSP->value() << endl;
        }
    });

    QLabel *N_species_LabelSP = new QLabel(tr("Number of massive neutrinos (NW P(k)):"));
    N_speciesSP = new QDoubleSpinBox;
    N_speciesSP->setDecimals(8);
    N_speciesSP->setRange(0.0, 1000000); N_speciesSP->setSingleStep(0.001); N_speciesSP->setValue(VN_speciesSP);

    connect(N_speciesSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VN_speciesSP "<< N_speciesSP->value() << endl;
        }
    });

    QLabel *ODEfitLabelSP = new QLabel(tr("ODE or Fit (projection):"));
    QComboBox *ODEfitSP = new QComboBox(this);
    ODEfitSP->addItem("ODE");
    ODEfitSP->addItem("Fit");
    ODEfitSP->setCurrentIndex(ODEfitSP->findText("ODE"));

    connect(ODEfitSP, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream4( &file4 );
            stream4 << "ODEfitSP_ch " << index << endl;
        }
     });

    QLabel *SurfAreaLabelSP = new QLabel(tr("Surface Area (deg sq):"));
    SurfAreaSP = new QDoubleSpinBox;
    SurfAreaSP->setDecimals(8);
    SurfAreaSP->setRange(0.0,41253.0); SurfAreaSP->setSingleStep(1.); SurfAreaSP->setValue(VSurfAreaSP);

    connect(SurfAreaSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VSurfAreaSP "<< SurfAreaSP->value() << endl;
        }
    });

    QLabel *SpectroprecLabelSP = new QLabel(tr("Spectroscopic precision:"));
    SpectroprecSP = new QDoubleSpinBox;
    SpectroprecSP->setDecimals(8);
    SpectroprecSP->setRange(0.0,1.0); SpectroprecSP->setSingleStep(0.0001); SpectroprecSP->setValue(VSpectroprecSP);

    connect(SpectroprecSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VSpectroprecSP "<< SpectroprecSP->value() << endl;
        }
    });

    QLabel *DerMethodLabelSP = new QLabel(tr("Derivatives scheme:"));
    QComboBox *DerMethodSP = new QComboBox(this);
    DerMethodSP->addItem("3");
    DerMethodSP->addItem("5");
    DerMethodSP->addItem("7");
    DerMethodSP->setCurrentIndex(DerMethodSP->findText("5"));

    connect(DerMethodSP, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream4( &file4 );
            stream4 << "DerMethodSP_ch " << index << endl;
        }
     });

    QLabel *IntegprecLabelSP = new QLabel(tr("Integrals number of steps:"));
    IntegprecSP = new QDoubleSpinBox;
    IntegprecSP->setDecimals(0);
    IntegprecSP->setRange(2.0, 1000000); IntegprecSP->setSingleStep(1.0); IntegprecSP->setValue(VIntegprecSP);

    connect(IntegprecSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VIntegprecSP "<< IntegprecSP->value() << endl;
        }
    });

    QLabel *DerprojLabelSP = new QLabel(tr("Steps for the projection:"));
    DerprojSP = new QDoubleSpinBox;
    DerprojSP->setDecimals(8);
    DerprojSP->setRange(0.0, 1000000); DerprojSP->setSingleStep(0.001); DerprojSP->setValue(VDerprojSP);

    connect(DerprojSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VDerprojSP "<< DerprojSP->value() << endl;
        }
    });

    QLabel *ODEPrecLabelSP = new QLabel(tr("ODE number of steps:"));
    ODEPrecSP = new QDoubleSpinBox;
    ODEPrecSP->setDecimals(0);
    ODEPrecSP->setRange(2.0, 1000000); ODEPrecSP->setSingleStep(1.0); ODEPrecSP->setValue(VODEPrecSP);

    connect(ODEPrecSP, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream Stream4( &file4 );
            Stream4 << "VODEPrecSP "<< ODEPrecSP->value() << endl;
        }
    });

    QLabel *biasvslnbiasLabelSP = new QLabel(tr("Use b(z) or ln[bs8(z)]:"));
    QComboBox *biasvslnbiasSP = new QComboBox(this);
    biasvslnbiasSP->addItem("b(z)");
    biasvslnbiasSP->addItem("ln[bs8(z)]");
    biasvslnbiasSP->setCurrentIndex(biasvslnbiasSP->findText("b(z)"));

    connect(biasvslnbiasSP, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename4="SpecSAF_W.txt";
        QFile file4(filename4);
        if (file4.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream4( &file4 );
            stream4 << "biasvslnbiasSP_ch " << index << endl;
        }
     });

    QGridLayout *spinBoxLayout = new QGridLayout;
    spinBoxLayout->addWidget(RS_bins_labelSP, 0, 0);
    spinBoxLayout->addWidget(RS_binsSP, 0, 1);
    spinBoxLayout->addWidget(zmin_bins_labelSP, 1, 0);
    spinBoxLayout->addWidget(zminSP, 1, 1);
    spinBoxLayout->addWidget(zmax_bins_labelSP, 2, 0);
    spinBoxLayout->addWidget(zmaxSP, 2, 1);
    spinBoxLayout->addWidget(Mean_bin_LabelSP, 3, 0);
    spinBoxLayout->addWidget(Mean_binSP, 3, 1);
    spinBoxLayout->addWidget(kmin_bins_labelSP, 4, 0);
    spinBoxLayout->addWidget(kminSP, 4, 1);
    spinBoxLayout->addWidget(kmax_bins_labelSP, 5, 0);
    spinBoxLayout->addWidget(kmaxSP, 5, 1);
    spinBoxLayout->addWidget(LNLcaseLabelSP, 6, 0);
    spinBoxLayout->addWidget(LNLcaseSP, 6, 1);
    spinBoxLayout->addWidget(SNLXcaseLabelSP, 7, 0);
    spinBoxLayout->addWidget(SNLXcaseSP, 7, 1);
    spinBoxLayout->addWidget(CMB_T_LabelSP, 8, 0);
    spinBoxLayout->addWidget(CMB_TSP, 8, 1);
    spinBoxLayout->addWidget(N_species_LabelSP, 9, 0);
    spinBoxLayout->addWidget(N_speciesSP, 9, 1);
    spinBoxLayout->addWidget(ODEfitLabelSP, 10, 0);
    spinBoxLayout->addWidget(ODEfitSP, 10, 1);
    spinBoxLayout->addWidget(SurfAreaLabelSP, 11, 0);
    spinBoxLayout->addWidget(SurfAreaSP, 11, 1);
    spinBoxLayout->addWidget(SpectroprecLabelSP, 12, 0);
    spinBoxLayout->addWidget(SpectroprecSP, 12, 1);
    spinBoxLayout->addWidget(DerMethodLabelSP, 13, 0);
    spinBoxLayout->addWidget(DerMethodSP, 13, 1);
    spinBoxLayout->addWidget(IntegprecLabelSP, 14, 0);
    spinBoxLayout->addWidget(IntegprecSP, 14, 1);
    spinBoxLayout->addWidget(DerprojLabelSP, 15, 0);
    spinBoxLayout->addWidget(DerprojSP, 15, 1);
    spinBoxLayout->addWidget(ODEPrecLabelSP, 16, 0);
    spinBoxLayout->addWidget(ODEPrecSP, 16, 1);
    spinBoxLayout->addWidget(biasvslnbiasLabelSP, 17, 0);
    spinBoxLayout->addWidget(biasvslnbiasSP, 17, 1);

    SpecSAFGroup->setLayout(spinBoxLayout);
}

/*Building the Widgets in the Parameters set windows.
Step 1: The default parameters values are declared in variables and saved in a parameter file.
Step 2: The Spinebox variables are created.
Step 3: The widget are placed in their subwindows layout with their labels.
Each time the user change the parameter value, the new value is written in the parameter file (see connect).
*/
void Window::Parameters_set(){

    int UseOmegam_ch = 0; int UseOmegaDE_ch = 1; int UseOmegab_ch = 0; int Usew0_ch = 0; int Usewa_ch = 0; int Useh_ch = 0;
    int Usens_ch = 0; int Usesigma8_ch = 0; int Usegamma_ch = 1; int Usesp_ch = 1; int Usesv_ch = 1; int UseGCspecbias_ch = 0;
    int UseGCspecPS_ch = 0; int UseGCphotbias_ch = 0; int UseWLAia_ch = 0; int UseWLBia_ch = 0; int UseWLnia_ch = 0;

    double VFidOmegam = 0.32; double VFidOmegaDE = 0.68; double VFidOmegab = 0.05; double VFidw0 = -1.0; double VFidwa = 0.0;
    double VFidh = 0.67; double VFidns = 0.96; double VFidsigma8 = 0.8155338; double VFidgamma = 6./11; double VFidWLAia = 1.72; 
    double VFidWLBia = 2.17; double VFidWLnia = -0.41; double VFidWLCia = 0.0134; double VFidAs = 2.12605; double VFidWnu = 0.00143717; 

    double VStepOmegam = 0.001; double VStepOmegaDE = 0.001; double VStepOmegab = 0.001; double VStepw0 = 0.001; double VStepwa = 0.001;
    double VSteph = 0.001; double VStepns = 0.001; double VStepsigma8 = 0.001; double VStepgamma = 0.001; double VStepWLAia = 0.001; 
    double VStepWLBia = 0.001; double VStepWLnia = 0.001; double VStepsp = 0.001; double VStepsv = 0.001; double VStepGCspecbias = 0.0001;
    double VStepGCspecPS = 0.0001; double VStepGCphotbias = 0.0001; double VSteplnDa = 0.0001; double VSteplnH = 0.0001; double VSteplnfs8 = 0.0001;

    QString filename5="Parameters_W.txt";
    QFile file5(filename5);
    if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
        QTextStream stream5( &file5 );

        stream5 << "UseOmegam_ch "<< UseOmegam_ch << endl;
        stream5 << "UseOmegaDE_ch "<< UseOmegaDE_ch << endl;
        stream5 << "UseOmegab_ch "<< UseOmegab_ch << endl;
        stream5 << "Usew0_ch "<< Usew0_ch << endl;
        stream5 << "Usewa_ch "<< Usewa_ch << endl;
        stream5 << "Useh_ch "<< Useh_ch << endl;
        stream5 << "Usens_ch "<< Usens_ch << endl;
        stream5 << "Usesigma8_ch "<< Usesigma8_ch << endl;
        stream5 << "Usegamma_ch "<< Usegamma_ch << endl;
        stream5 << "Usesp_ch "<< Usesp_ch << endl;
        stream5 << "Usesv_ch "<< Usesv_ch << endl;
        stream5 << "UseGCspecbias_ch "<< UseGCspecbias_ch << endl;
        stream5 << "UseGCspecPS_ch "<< UseGCspecPS_ch << endl;
        stream5 << "UseWLAia_ch "<< UseWLAia_ch << endl;
        stream5 << "UseWLBia_ch "<< UseWLBia_ch << endl;
        stream5 << "UseWLnia_ch "<< UseWLnia_ch << endl;
        stream5 << "UseGCphotbias_ch "<< UseGCphotbias_ch << endl;

        stream5 << "VFidOmegam "<< VFidOmegam << endl;
        stream5 << "VFidOmegaDE "<< VFidOmegaDE << endl;
        stream5 << "VFidOmegab "<< VFidOmegab << endl;
        stream5 << "VFidw0 "<< VFidw0 << endl;
        stream5 << "VFidwa "<< VFidwa << endl;
        stream5 << "VFidh "<< VFidh << endl;
        stream5 << "VFidns "<< VFidns << endl;
        stream5 << "VFidsigma8 "<< VFidsigma8 << endl;
        stream5 << "VFidgamma "<< VFidgamma << endl;
        stream5 << "VFidWLAia "<< VFidWLAia << endl;
        stream5 << "VFidWLBia "<< VFidWLBia << endl;
        stream5 << "VFidWLnia "<< VFidWLnia << endl;
        stream5 << "VFidWLCia "<< VFidWLCia << endl;
        stream5 << "VFidAs "<< VFidAs << endl;
        stream5 << "VFidWnu "<< VFidWnu << endl;

        stream5 << "VStepOmegam "<< VStepOmegam << endl;
        stream5 << "VStepOmegaDE "<< VStepOmegaDE << endl;
        stream5 << "VStepOmegab "<< VStepOmegab << endl;
        stream5 << "VStepw0 "<< VStepw0 << endl;
        stream5 << "VStepwa "<< VStepwa << endl;
        stream5 << "VSteph "<< VSteph << endl;
        stream5 << "VStepns "<< VStepns << endl;
        stream5 << "VStepsigma8 "<< VStepsigma8 << endl;
        stream5 << "VStepgamma "<< VStepgamma << endl;
        stream5 << "VStepsp "<< VStepsp << endl;
        stream5 << "VStepsv "<< VStepsv << endl;
        stream5 << "VStepGCspecbias "<< VStepGCspecbias << endl;
        stream5 << "VStepGCspecPS "<< VStepGCspecPS << endl;
        stream5 << "VStepWLAia "<< VStepWLAia << endl;
        stream5 << "VStepWLBia "<< VStepWLBia << endl;
        stream5 << "VStepWLnia "<< VStepWLnia << endl;
        stream5 << "VStepGCphotbias "<< VStepGCphotbias << endl;
        stream5 << "VSteplnDa "<< VSteplnDa << endl;
        stream5 << "VSteplnH "<< VSteplnH << endl;
        stream5 << "VSteplnfs8 "<< VSteplnfs8 << endl;
    }

    ParametersGroup = new QGroupBox(tr("Parameters settings"));

    QLabel *ParamsLabel = new QLabel(tr("Parameters")); QLabel *UsePLabel = new QLabel(tr("Use")); QLabel *FidLabel = new QLabel(tr("Fiducials")); QLabel *StepsLabel = new QLabel(tr("steps"));

    QLabel *UseOmegamLabel = new QLabel(tr("Omega m"));
    QComboBox *UseOmegam = new QComboBox(this);
    UseOmegam->addItem("Yes");
    UseOmegam->addItem("No");
    UseOmegam->setCurrentIndex(UseOmegam->findText("Yes"));
    FidOmegam = new QDoubleSpinBox; FidOmegam->setDecimals(8);
    FidOmegam->setRange(0.0, 1.0); FidOmegam->setSingleStep(0.01); FidOmegam->setValue(VFidOmegam);
    StepOmegam = new QDoubleSpinBox; StepOmegam->setDecimals(8);
    StepOmegam->setRange(0.00000001, 1.0); StepOmegam->setSingleStep(0.0001); StepOmegam->setValue(VStepOmegam);

    connect(UseOmegam, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseOmegam_ch " << index << endl;
        }
    });

    connect(FidOmegam, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidOmegam "<< FidOmegam->value() << endl;
        }
    });

    connect(StepOmegam, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepOmegam "<< StepOmegam->value() << endl;
        }
    });

    QLabel *UseOmegaDELabel = new QLabel(tr("Omega DE"));
    QComboBox *UseOmegaDE = new QComboBox(this);
    UseOmegaDE->addItem("Yes");
    UseOmegaDE->addItem("No");
    UseOmegaDE->setCurrentIndex(UseOmegaDE->findText("No"));
    FidOmegaDE = new QDoubleSpinBox; FidOmegaDE->setDecimals(8);
    FidOmegaDE->setRange(0.0, 1.0); FidOmegaDE->setSingleStep(0.01); FidOmegaDE->setValue(VFidOmegaDE);
    StepOmegaDE = new QDoubleSpinBox; StepOmegaDE->setDecimals(8);
    StepOmegaDE->setRange(0.00000001, 1.0); StepOmegaDE->setSingleStep(0.0001); StepOmegaDE->setValue(VStepOmegaDE);

    connect(UseOmegaDE, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseOmegaDE_ch " << index << endl;
        }
    });

    connect(FidOmegaDE, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidOmegaDE "<< FidOmegaDE->value() << endl;
        }
    });

    connect(StepOmegaDE, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepOmegaDE "<< StepOmegaDE->value() << endl;
        }
    });

    QLabel *UseOmegabLabel = new QLabel(tr("Omega b"));
    QComboBox *UseOmegab = new QComboBox(this);
    UseOmegab->addItem("Yes");
    UseOmegab->addItem("No");
    FidOmegab = new QDoubleSpinBox; FidOmegab->setDecimals(8);
    FidOmegab->setRange(0.0, 1.0); FidOmegab->setSingleStep(0.01); FidOmegab->setValue(VFidOmegab);
    StepOmegab = new QDoubleSpinBox; StepOmegab->setDecimals(8);
    StepOmegab->setRange(0.00000001, 1.0); StepOmegab->setSingleStep(0.0001); StepOmegab->setValue(VStepOmegab);

    connect(UseOmegab, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseOmegab_ch " << index << endl;
        }
    });

    connect(FidOmegab, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidOmegab "<< FidOmegab->value() << endl;
        }
    });

    connect(StepOmegab, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepOmegab "<< StepOmegab->value() << endl;
        }
    });

    QLabel *Usew0Label = new QLabel(tr("w0"));
    QComboBox *Usew0 = new QComboBox(this);
    Usew0->addItem("Yes");
    Usew0->addItem("No");
    Fidw0 = new QDoubleSpinBox; Fidw0->setDecimals(8);
    Fidw0->setRange(-5.0, 5.0); Fidw0->setSingleStep(0.01); Fidw0->setValue(VFidw0);
    Stepw0 = new QDoubleSpinBox; Stepw0->setDecimals(8);
    Stepw0->setRange(0.00000001, 1.0); Stepw0->setSingleStep(0.0001); Stepw0->setValue(VStepw0);

    connect(Usew0, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usew0_ch " << index << endl;
        }
    });

    connect(Fidw0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidw0 "<< Fidw0->value() << endl;
        }
    });

    connect(Stepw0, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepw0 "<< Stepw0->value() << endl;
        }
    });

    QLabel *UsewaLabel = new QLabel(tr("wa"));
    QComboBox *Usewa = new QComboBox(this);
    Usewa->addItem("Yes");
    Usewa->addItem("No");
    Fidwa = new QDoubleSpinBox; Fidwa->setDecimals(8);
    Fidwa->setRange(-5.0, 5.0); Fidwa->setSingleStep(0.01); Fidwa->setValue(VFidwa);
    Stepwa = new QDoubleSpinBox; Stepwa->setDecimals(8);
    Stepwa->setRange(0.00000001, 1.0); Stepwa->setSingleStep(0.0001); Stepwa->setValue(VStepwa);

    connect(Usewa, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usewa_ch " << index << endl;
        }
    });

    connect(Fidwa, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidwa "<< Fidwa->value() << endl;
        }
    });

    connect(Stepwa, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepwa "<< Stepwa->value() << endl;
        }
    });

    QLabel *UsehLabel = new QLabel(tr("h"));
    QComboBox *Useh = new QComboBox(this);
    Useh->addItem("Yes");
    Useh->addItem("No");
    Fidh = new QDoubleSpinBox; Fidh->setDecimals(8);
    Fidh->setRange(0.0, 2.0); Fidh->setSingleStep(0.01); Fidh->setValue(VFidh);
    Steph = new QDoubleSpinBox; Steph->setDecimals(8);
    Steph->setRange(0.00000001, 1.0); Steph->setSingleStep(0.0001); Steph->setValue(VSteph);

    connect(Useh, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Useh_ch " << index << endl;
        }
    });

    connect(Fidh, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidh "<< Fidh->value() << endl;
        }
    });

    connect(Steph, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VSteph "<< Steph->value() << endl;
        }
    });

    QLabel *UsensLabel = new QLabel(tr("ns"));
    QComboBox *Usens = new QComboBox(this);
    Usens->addItem("Yes");
    Usens->addItem("No");
    Fidns = new QDoubleSpinBox; Fidns->setDecimals(8);
    Fidns->setRange(0.0, 5.0); Fidns->setSingleStep(0.01); Fidns->setValue(VFidns);
    Stepns = new QDoubleSpinBox; Stepns->setDecimals(8);
    Stepns->setRange(0.00000001, 1.0); Stepns->setSingleStep(0.0001); Stepns->setValue(VStepns);

    connect(Usens, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usens_ch " << index << endl;
        }
    });

    connect(Fidns, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidns "<< Fidns->value() << endl;
        }
    });

    connect(Stepns, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepns "<< Stepns->value() << endl;
        }
    });

    QLabel *Usesigma8Label = new QLabel(tr("s8"));
    QComboBox *Usesigma8 = new QComboBox(this);
    Usesigma8->addItem("Yes");
    Usesigma8->addItem("No");
    Fidsigma8 = new QDoubleSpinBox; Fidsigma8->setDecimals(8);
    Fidsigma8->setRange(0.0, 1.0); Fidsigma8->setSingleStep(0.01); Fidsigma8->setValue(VFidsigma8);
    Stepsigma8 = new QDoubleSpinBox; Stepsigma8->setDecimals(8);
    Stepsigma8->setRange(0.00000001, 1.0); Stepsigma8->setSingleStep(0.0001); Stepsigma8->setValue(VStepsigma8);

    connect(Fidsigma8, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidsigma8 "<< Fidsigma8->value() << endl;
        }
    });

    connect(Stepsigma8, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepsigma8 "<< Stepsigma8->value() << endl;
        }
    });

    QLabel *UsespLabel = new QLabel(tr("sp"));
    QComboBox *Usesp = new QComboBox(this);
    Usesp->addItem("Yes");
    Usesp->addItem("No");
    Usesp->setCurrentIndex(Usesp->findText("No"));
    Stepsp = new QDoubleSpinBox; Stepsp->setDecimals(8);
    Stepsp->setRange(0.00000001, 1.0); Stepsp->setSingleStep(0.0001); Stepsp->setValue(VStepsp);

    connect(Usesp, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usesp_ch " << index << endl;
        }
    });

    connect(Stepsp, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepsp "<< Stepsp->value() << endl;
        }
    });

    QLabel *UsesvLabel = new QLabel(tr("sv"));
    QComboBox *Usesv = new QComboBox(this);
    Usesv->addItem("Yes");
    Usesv->addItem("No");
    Usesv->setCurrentIndex(Usesv->findText("No"));
    Stepsv = new QDoubleSpinBox; Stepsv->setDecimals(8);
    Stepsv->setRange(0.00000001, 1.0); Stepsv->setSingleStep(0.0001); Stepsv->setValue(VStepsv);

    connect(Usesv, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usesv_ch " << index << endl;
        }
    });

    connect(Stepsv, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepsv "<< Stepsv->value() << endl;
        }
    });

    QLabel *UsegammaLabel = new QLabel(tr("gamma"));
    QComboBox *Usegamma = new QComboBox(this);
    Usegamma->addItem("Yes");
    Usegamma->addItem("No");
    Usegamma->setCurrentIndex(Usegamma->findText("No"));
    Fidgamma = new QDoubleSpinBox; Fidgamma->setDecimals(8);
    Fidgamma->setRange(0.0, 1.0); Fidgamma->setSingleStep(0.01); Fidgamma->setValue(VFidgamma);
    Stepgamma = new QDoubleSpinBox; Stepgamma->setDecimals(8);
    Stepgamma->setRange(0.00000001, 1.0); Stepgamma->setSingleStep(0.0001); Stepgamma->setValue(VStepgamma);

    connect(Usegamma, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "Usegamma_ch " << index << endl;
        }
    });

    connect(Fidgamma, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidgamma "<< Fidgamma->value() << endl;
        }
    });

    connect(Stepgamma, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepgamma "<< Stepgamma->value() << endl;
        }
    });

    QLabel *UseGCspecbiasLabel = new QLabel(tr("GCs bias"));
    QComboBox *UseGCspecbias = new QComboBox(this);
    UseGCspecbias->addItem("Yes");
    UseGCspecbias->addItem("No");
    StepGCspecbias = new QDoubleSpinBox; StepGCspecbias->setDecimals(8);
    StepGCspecbias->setRange(0.00000001, 1.0); StepGCspecbias->setSingleStep(0.0001); StepGCspecbias->setValue(VStepGCspecbias);

    connect(UseGCspecbias, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseGCspecbias_ch " << index << endl;
        }
    });

    connect(StepGCspecbias, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepGCspecbias "<< StepGCspecbias->value() << endl;
        }
    });

    QLabel *UseGCspecPSLabel = new QLabel(tr("GCs shot noise"));
    QComboBox *UseGCspecPS = new QComboBox(this);
    UseGCspecPS->addItem("Yes");
    UseGCspecPS->addItem("No");
    StepGCspecPS = new QDoubleSpinBox; StepGCspecPS->setDecimals(8);
    StepGCspecPS->setRange(0.00000001, 1.0); StepGCspecPS->setSingleStep(0.0001); StepGCspecPS->setValue(VStepGCspecPS);

    connect(UseGCspecPS, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseGCspecPS_ch " << index << endl;
        }
    });

    connect(StepGCspecPS, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepGCspecPS "<< StepGCspecPS->value() << endl;
        }
    });

    QLabel *UseGCphotbiasLabel = new QLabel(tr("GCp bias"));
    QComboBox *UseGCphotbias = new QComboBox(this);
    UseGCphotbias->addItem("Yes");
    UseGCphotbias->addItem("No");
    StepGCphotbias = new QDoubleSpinBox; StepGCphotbias->setDecimals(8);
    StepGCphotbias->setRange(0.00000001, 1.0); StepGCphotbias->setSingleStep(0.0001); StepGCphotbias->setValue(VStepGCphotbias);

    connect(UseGCphotbias, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseGCphotbias_ch " << index << endl;
        }
    });

    connect(StepGCphotbias, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepGCphotbias "<< StepGCphotbias->value() << endl;
        }
    });

    QLabel *UseWLAiaLabel = new QLabel(tr("A IA (WL)"));
    QComboBox *UseWLAia = new QComboBox(this);
    UseWLAia->addItem("Yes");
    UseWLAia->addItem("No");
    FidWLAia = new QDoubleSpinBox; FidWLAia->setDecimals(8);
    FidWLAia->setRange(-10.0, 10.0); FidWLAia->setSingleStep(0.01); FidWLAia->setValue(1.72);
    StepWLAia = new QDoubleSpinBox; StepWLAia->setDecimals(8);
    StepWLAia->setRange(0.00000001, 1.0); StepWLAia->setSingleStep(0.0001); StepWLAia->setValue(VStepWLAia);

    connect(UseWLAia, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseWLAia_ch " << index << endl;
        }
    });

    connect(FidWLAia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidWLAia "<< FidWLAia->value() << endl;
        }
    });

    connect(StepWLAia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepWLAia "<< StepWLAia->value() << endl;
        }
    });

    QLabel *UseWLBiaLabel = new QLabel(tr("B IA (WL)"));
    QComboBox *UseWLBia = new QComboBox(this);
    UseWLBia->addItem("Yes");
    UseWLBia->addItem("No");
    FidWLBia = new QDoubleSpinBox; FidWLBia->setDecimals(8);
    FidWLBia->setRange(-10.0, 10.0); FidWLBia->setSingleStep(0.01); FidWLBia->setValue(VFidWLBia);
    StepWLBia = new QDoubleSpinBox; StepWLBia->setDecimals(8);
    StepWLBia->setRange(0.00000001, 1.0); StepWLBia->setSingleStep(0.0001); StepWLBia->setValue(VStepWLBia);

    connect(UseWLBia, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseWLBia_ch " << index << endl;
        }
    });

    connect(FidWLBia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidWLBia "<< FidWLBia->value() << endl;
        }
    });

    connect(StepWLBia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepWLBia "<< StepWLBia->value() << endl;
        }
    });

    QLabel *UseWLniaLabel = new QLabel(tr("n IA (WL)"));
    QComboBox *UseWLnia = new QComboBox(this);
    UseWLnia->addItem("Yes");
    UseWLnia->addItem("No");
    FidWLnia = new QDoubleSpinBox; FidWLnia->setDecimals(8);
    FidWLnia->setRange(-10.0, 10.0); FidWLnia->setSingleStep(0.01); FidWLnia->setValue(VFidWLnia);
    StepWLnia = new QDoubleSpinBox; StepWLnia->setDecimals(8);
    StepWLnia->setRange(0.00000001, 1.0); StepWLnia->setSingleStep(0.0001); StepWLnia->setValue(VStepWLnia);

    connect(UseWLnia, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int index){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "UseWLnia_ch " << index << endl;
        }
    });

    connect(FidWLnia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidWLnia "<< FidWLnia->value() << endl;
        }
    });

    connect(StepWLnia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VStepWLnia "<< StepWLnia->value() << endl;
        }
    });

    QLabel *UseWLCiaLabel = new QLabel(tr("C IA (WL)"));
    FidWLCia = new QDoubleSpinBox; FidWLCia->setDecimals(8);
    FidWLCia->setRange(-10.0, 10.0); FidWLCia->setSingleStep(0.01); FidWLCia->setValue(VFidWLCia);

    connect(FidWLCia, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidWLCia "<< FidWLCia->value() << endl;
        }
    });

    QLabel *lnDaLabel = new QLabel(tr("ln[Da(z)] (GCs)")); QLabel *lnHLabel = new QLabel(tr("ln[H(z)] (GCs)")); QLabel *lnfsigma8Label = new QLabel(tr("ln[fs8(z)] (GCs)"));

    SteplnDa = new QDoubleSpinBox; SteplnDa->setDecimals(8);
    SteplnDa->setRange(0.00000001, 1.0); SteplnDa->setSingleStep(0.0001); SteplnDa->setValue(VSteplnDa);
    SteplnH = new QDoubleSpinBox; SteplnH->setDecimals(8);
    SteplnH->setRange(0.00000001, 1.0); SteplnH->setSingleStep(0.0001); SteplnH->setValue(VSteplnH);
    Steplnfs8 = new QDoubleSpinBox; Steplnfs8->setDecimals(8);
    Steplnfs8->setRange(0.00000001, 1.0); Steplnfs8->setSingleStep(0.0001); Steplnfs8->setValue(VSteplnfs8);

    connect(SteplnDa, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VSteplnDa "<< SteplnDa->value() << endl;
        }
    });

    connect(SteplnH, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VSteplnH "<< SteplnH->value() << endl;
        }
    });

    connect(Steplnfs8, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VSteplnfs8 "<< Steplnfs8->value() << endl;
        }
    });

    QLabel *UseWnuLabel = new QLabel(tr("Omega nu"));
    FidWnu = new QDoubleSpinBox; FidWnu->setDecimals(8);
    FidWnu->setRange(0.0, 1.0); FidWnu->setSingleStep(0.0001); FidWnu->setValue(VFidWnu);

    connect(FidWnu, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidWnu "<< FidWnu->value() << endl;
        }
    });

    QLabel *UseAsLabel = new QLabel(tr("As (x1e-9)"));
    FidAs = new QDoubleSpinBox; FidAs->setDecimals(8);
    FidAs->setRange(0.0, 100.0); FidAs->setSingleStep(0.0001); FidAs->setValue(VFidAs);

    connect(FidAs, static_cast<void(QDoubleSpinBox::*)(const QString &)>(&QDoubleSpinBox::valueChanged), [=](const QString){
        QString filename5="Parameters_W.txt";
        QFile file5(filename5);
        if (file5.open(QIODevice::ReadWrite | QIODevice::Append)){
            QTextStream stream5( &file5 );
            stream5 << "VFidAs "<< FidAs->value() << endl;
        }
    });

    QGridLayout *spinBoxLayout = new QGridLayout;
    spinBoxLayout->addWidget(ParamsLabel, 0, 0); spinBoxLayout->addWidget(UsePLabel, 0, 1); spinBoxLayout->addWidget(FidLabel, 0, 2); spinBoxLayout->addWidget(StepsLabel, 0, 3);
    spinBoxLayout->addWidget(UseOmegamLabel, 1, 0); spinBoxLayout->addWidget(UseOmegam, 1, 1); spinBoxLayout->addWidget(FidOmegam, 1, 2); spinBoxLayout->addWidget(StepOmegam, 1, 3);
    spinBoxLayout->addWidget(UseOmegaDELabel, 2, 0); spinBoxLayout->addWidget(UseOmegaDE, 2, 1); spinBoxLayout->addWidget(FidOmegaDE, 2, 2); spinBoxLayout->addWidget(StepOmegaDE, 2, 3);
    spinBoxLayout->addWidget(UseOmegabLabel, 3, 0); spinBoxLayout->addWidget(UseOmegab, 3, 1); spinBoxLayout->addWidget(FidOmegab, 3, 2); spinBoxLayout->addWidget(StepOmegab, 3, 3);
    spinBoxLayout->addWidget(Usew0Label, 4, 0); spinBoxLayout->addWidget(Usew0, 4, 1); spinBoxLayout->addWidget(Fidw0, 4, 2); spinBoxLayout->addWidget(Stepw0, 4, 3);
    spinBoxLayout->addWidget(UsewaLabel, 5, 0); spinBoxLayout->addWidget(Usewa, 5, 1); spinBoxLayout->addWidget(Fidwa, 5, 2); spinBoxLayout->addWidget(Stepwa, 5, 3);
    spinBoxLayout->addWidget(UsehLabel, 6, 0); spinBoxLayout->addWidget(Useh, 6, 1); spinBoxLayout->addWidget(Fidh, 6, 2); spinBoxLayout->addWidget(Steph, 6, 3);
    spinBoxLayout->addWidget(UsensLabel, 7, 0); spinBoxLayout->addWidget(Usens, 7, 1); spinBoxLayout->addWidget(Fidns, 7, 2); spinBoxLayout->addWidget(Stepns, 7, 3);
    spinBoxLayout->addWidget(Usesigma8Label, 8, 0); spinBoxLayout->addWidget(Usesigma8, 8, 1); spinBoxLayout->addWidget(Fidsigma8, 8, 2); spinBoxLayout->addWidget(Stepsigma8, 8, 3);
    spinBoxLayout->addWidget(UsegammaLabel, 9, 0); spinBoxLayout->addWidget(Usegamma, 9, 1); spinBoxLayout->addWidget(Fidgamma,9, 2); spinBoxLayout->addWidget(Stepgamma, 9, 3);
    spinBoxLayout->addWidget(UsespLabel, 10, 0); spinBoxLayout->addWidget(Usesp, 10, 1); spinBoxLayout->addWidget(Stepsp, 10, 3);
    spinBoxLayout->addWidget(UsesvLabel, 11, 0); spinBoxLayout->addWidget(Usesv, 11, 1); spinBoxLayout->addWidget(Stepsv, 11, 3);
    spinBoxLayout->addWidget(UseGCspecbiasLabel, 12, 0); spinBoxLayout->addWidget(UseGCspecbias, 12, 1); spinBoxLayout->addWidget(StepGCspecbias, 12, 3);
    spinBoxLayout->addWidget(UseGCspecPSLabel, 13, 0); spinBoxLayout->addWidget(UseGCspecPS, 13, 1); spinBoxLayout->addWidget(StepGCspecPS, 13, 3);
    spinBoxLayout->addWidget(UseWLAiaLabel, 14, 0); spinBoxLayout->addWidget(UseWLAia, 14, 1); spinBoxLayout->addWidget(FidWLAia, 14, 2); spinBoxLayout->addWidget(StepWLAia, 14, 3);
    spinBoxLayout->addWidget(UseWLBiaLabel, 15, 0); spinBoxLayout->addWidget(UseWLnia, 15, 1); spinBoxLayout->addWidget(FidWLnia, 15, 2); spinBoxLayout->addWidget(StepWLnia, 15, 3);
    spinBoxLayout->addWidget(UseWLniaLabel, 16, 0); spinBoxLayout->addWidget(UseWLBia, 16, 1); spinBoxLayout->addWidget(FidWLBia, 16, 2); spinBoxLayout->addWidget(StepWLBia, 16, 3);
    spinBoxLayout->addWidget(UseWLCiaLabel, 17, 0); spinBoxLayout->addWidget(FidWLCia, 17, 2);
    spinBoxLayout->addWidget(UseGCphotbiasLabel, 18, 0); spinBoxLayout->addWidget(UseGCphotbias, 18, 1); spinBoxLayout->addWidget(StepGCphotbias, 18, 3);
    spinBoxLayout->addWidget(lnDaLabel, 19, 0); spinBoxLayout->addWidget(lnHLabel, 20, 0); spinBoxLayout->addWidget(lnfsigma8Label, 21, 0);
    spinBoxLayout->addWidget(SteplnDa, 19, 3);
    spinBoxLayout->addWidget(SteplnH, 20, 3);
    spinBoxLayout->addWidget(Steplnfs8, 21, 3);
    spinBoxLayout->addWidget(UseWnuLabel, 22, 0); spinBoxLayout->addWidget(FidWnu, 22, 2);
    spinBoxLayout->addWidget(UseAsLabel, 23, 0); spinBoxLayout->addWidget(FidAs, 23, 2);
     
    ParametersGroup->setLayout(spinBoxLayout);
}

//Building the Widget validation. Once the user clicks on the confirm button, the codes run.
void Window::ValidEx(){
     
    ValidExGroup = new QGroupBox(tr(""));
    QHBoxLayout *spinBoxLayout = new QHBoxLayout;

    QPushButton* Confirm = new QPushButton("Confirm");
    QObject::connect(Confirm, SIGNAL(clicked()), qApp, SLOT(quit()));
    spinBoxLayout->addWidget(Confirm);

    /*
    QPushButton* Cancel = new QPushButton("Cancel");
    QObject::connect(Cancel, SIGNAL(clicked()), qApp, SLOT(quit()));
    spinBoxLayout->addWidget(Confirm);
    */

    ValidExGroup->setLayout(spinBoxLayout);
}
