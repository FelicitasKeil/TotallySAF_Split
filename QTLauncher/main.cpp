/*
Authors: S.Yahia-Cherif.
Last Update 29/06/2020.
This script computes the Big Fisher matrix and performs the projection into the new parameter space.
*/

//import libraries.
#include <QApplication>
#include "window.h"
#include <unistd.h>

int main(int argc, char *argv[]){
	//Erase the previous parameters files.
 	system("rm Codes_W.txt"); system("rm Extra_W.txt"); system("rm XSAF_W.txt");  system("rm SpecSAF_W.txt"); system("rm Parameters_W.txt");
    QApplication app(argc, argv);
    Window window;
    //Set windows size and display.
    window.setMinimumSize(QSize(100,100));
	window.setMaximumSize(QSize(1920,1080));
    window.showMaximized();
    return app.exec();
}
