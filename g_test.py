#!/usr/bin/env python3

"""
GUI for Diabetes prediction.
"""
import sys

from PyQt5.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QApplication, QMessageBox
from PyQt5.QtGui import QDoubleValidator, QFont
from PyQt5.QtCore import Qt, QLine

import diabetes

class diabetes(QWidget):

    def __init__(self) -> None :
        super(diabetes, self).__init__()
        self.sub_head = QLabel("Patient's Details")
        self.sub_head.setFont(QFont("Times",24, weight=QFont.Bold))
        self.l0 = QLineEdit()
        self.l1 = QLineEdit()
        self.l2 = QLineEdit()
        self.l3 = QLineEdit()
        self.l4 = QLineEdit()
        self.l5 = QLineEdit()
        self.l6 = QLineEdit()
        self.l7 = QLineEdit()
        self.l8 = QLineEdit()
        self.t0 = QLabel("Patient's Name:")
        self.t1 = QLabel("Glucose conc:")
        self.t2 = QLabel("Diastolic_Bp:")
        self.t3 = QLabel("Thickness:")
        self.t4 = QLabel("insulin:")
        self.t5 = QLabel("Bmi:")
        self.t6 = QLabel("Diab_pred:")
        self.t7 = QLabel("age:")
        self.t8 = QLabel("skin:")
        

        self.r1 = QLabel("(1-200)")
        self.r2 = QLabel("(1-150)")
        self.r3 = QLabel("(1-100)")
        self.r4 = QLabel("(0.-1000)")
        self.r5 = QLabel("(0.0-1)")
        self.r6 = QLabel("(0.0-100)")
        self.r7 = QLabel("(0.0-2)")
        self.r8 = QLabel("(0.0-50)")

        self.h1 = QHBoxLayout()
        self.h0 = QHBoxLayout()
        self.h2 = QHBoxLayout()
        self.h3 = QHBoxLayout()
        self.h4 = QHBoxLayout()
        self.h5 = QHBoxLayout()
        self.h6 = QHBoxLayout()
        self.h7 = QHBoxLayout()
        self.h8 = QHBoxLayout()

        self.clbtn = QPushButton("CLEAR")
        self.clbtn.setFixedWidth(100)
        self.submit = QPushButton("SUBMIT")
        self.submit.setFixedWidth(100)
        self.v1_box = QVBoxLayout()
        self.v2_box = QVBoxLayout()
        self.final_hbox = QHBoxLayout()
        self.initui()

    def initui(self) -> None:
        """ The gui is created and widgets elements are set here """
        self.v1_box.addWidget(self.sub_head)
        self.v1_box.addSpacing(10)
        self.v1_box.setSpacing(5)
        self.l1.setValidator(QDoubleValidator())
        self.l2.setValidator(QDoubleValidator())
        self.l3.setValidator(QDoubleValidator())
        self.l4.setValidator(QDoubleValidator())
        self.l5.setValidator(QDoubleValidator())
        self.l6.setValidator(QDoubleValidator())
        self.l7.setValidator(QDoubleValidator())
        self.l8.setValidator(QDoubleValidator())

        self.l0.setToolTip("Enter name here")
        self.l1.setToolTip("2 hours in an oral glucose tolerance test \n 70-180 mg/dl")
        self.l2.setToolTip("Blool pressure")
        self.l3.setToolTip("Enter number fo insulin taken")
        self.l4.setToolTip("15-276mu U/ml")
        self.l5.setToolTip("compulsory to Fill")
        self.l6.setToolTip("compulsory to Fill")
        self.l7.setToolTip("Enter your curret age")
        self.l8.setToolTip("compulsory to Fill")

        self.l0.setFixedSize(275,30)
        self.l1.setFixedSize(60,30)
        self.l2.setFixedSize(60,30)
        self.l3.setFixedSize(60,30)
        self.l4.setFixedSize(60,30)
        self.l5.setFixedSize(60,30)
        self.l6.setFixedSize(60,30)
        self.l7.setFixedSize(60,30)
        self.l8.setFixedSize(60,30)

        self.h0.addWidget(self.t0)
        self.h0.addWidget(self.l0)
        self.v1_box.addLayout(self.h0)
        self.h1.addWidget(self.t1)
        self.h1.addWidget(self.l1)
        self.h1.addWidget(self.r1)        
        self.v1_box.addLayout(self.h1)
        self.h2.addWidget(self.t2)
        self.h2.addWidget(self.l2)
        self.h2.addWidget(self.r2)       
        self.v1_box.addLayout(self.h2)
        self.h3.addWidget(self.t3)
        self.h3.addWidget(self.l3)
        self.h3.addWidget(self.r3)       
        self.v1_box.addLayout(self.h3)
        self.h4.addWidget(self.t4)
        self.h4.addWidget(self.l4)
        self.h4.addWidget(self.r4)      
        self.v1_box.addLayout(self.h4)
        self.h5.addWidget(self.t5)
        self.h5.addWidget(self.l5)
        self.h5.addWidget(self.r5)      
        self.v1_box.addLayout(self.h5)
        self.h6.addWidget(self.t6)
        self.h6.addWidget(self.l6)
        self.h6.addWidget(self.r6)      
        self.v1_box.addLayout(self.h6)
        self.h7.addWidget(self.t7)
        self.h7.addWidget(self.l7)
        self.h7.addWidget(self.r7)      
        self.v1_box.addLayout(self.h7)
        self.h8.addWidget(self.t8)
        self.h8.addWidget(self.l8)
        self.h8.addWidget(self.r8)      
        self.v1_box.addLayout(self.h8)


        self.h9 = QHBoxLayout()
        self.submit.clicked.connect(lambda: self.test_input())
        self.submit.setToolTip("Click to check if patient has diabetes")
        self.clbtn.clicked.connect(lambda: self.clfn())
        self.h9.addWidget(self.submit)
        self.h9.addWidget(self.clbtn)
        self.v1_box.addLayout(self.h9)
        self.report_ui()
        self.final_hbox.addLayout(self.v1_box)
        self.final_hbox.addSpacing(40)
        self.final_hbox.addLayout(self.v2_box)
        self.setLayout(self.final_hbox)

    def report_ui(self):
        self.v2_box.setSpacing(6)
        self.report_subhead = QLabel("About product:")
        self.report_subhead.setAlignment(Qt.AlignCenter)
        self.report_subhead.setFont(QFont("Times",24, weight=QFont.Bold))
        self.v2_box.addWidget(self.report_subhead)
        self.details = QLabel("This will Help you to find you are suffering from Diabetes or Not.\nAnd Help you to stay fit and Active.  ")
        self.details.setFont(QFont("Arial",14, weight=QFont.Bold))
        self.details.setAlignment(Qt.AlignLeft)
        self.details.setWordWrap(True)
        self.model_details = QLabel("Fill details and press submit to see details.")
        self.model_details.setWordWrap(True)
        self.v2_box.addWidget(self.details)
        self.results = QLabel(" ")
        self.results.setWordWrap(True)
        self.v2_box.addWidget(self.results)
        self.v2_box.addWidget(self.model_details)

    def clfn(self):
        """ clear all the text fields via clear button"""
        self.l0.clear()
        self.l1.clear()
        self.l2.clear()
        self.l3.clear()
        self.l3.clear()
        self.l4.clear()
        self.l5.clear()
        self.l6.clear()
        self.l7.clear()
        self.l8.clear()
        self.report_subhead.setText("About")
        self.model_details.setText("Fill details and press submit to see details.")
        self.results.setText(" ")
        self.details.setText("Enter Detials for check again\nif you are have any dought vistie: www.v123@xyz.com")
        #print(self.frameGeometry().width())
        #print(self.frameGeometry().height())

    def test_input(self) -> None:
        """ test for diabetes"""
        my_dict = {"A":float(self.l1.text()), "B":float(self.l2.text()),"C":float(self.l3.text()), "D":float(self.l4.text()), "E": float(self.l5.text()), "F": float(self.l6.text()), "G": float(self.l7.text()), "H": float(self.l8.text())}
        output = diabetes.check_input(my_dict)
        print(output)
        #self.setFixedSize(850, 342)
        self.report_subhead.setText("Reports")
        self.model_details.setText("For More Go to AI Doctor ")
        self.details.setText("Patient's name: {}\nBody Glucose: {} \
\nBlood pressure: {}\nMean Perimeter: {}\ninsulin taken: {}\nH*w ratio: {}\nDiabetes pedigree: {}\nyour curret age: {}\nskin fold: {}".format(self.l0.text(), self.l1.text(), self.l2.text(), self.l3.text(),self.l4.text(),self.l5.text(),self.l6.text(),self.l7.text(),self.l8.text()))
        #
        if output==0:
            self.results.setText("Our suggests that patient DOES NOT SUFFER FROM Diabetes.")
        else:
            self.results.setText("Our suggests patient DOES SUFFER FROM Diabetes\nPlease get checked soon.")
        self.results.setFont(QFont("Arial",14, weight=QFont.Bold))           

    def mwindow(self) -> None:
        """ window features are set here and application is loaded into display"""
        self.setFixedSize(898, 422)
        self.setWindowTitle("Diabetes Detection")
        self.show()


if __name__=="__main__":
    app = QApplication(sys.argv)
    a_window = diabetes()
    a_window.mwindow()
    sys.exit(app.exec_())
