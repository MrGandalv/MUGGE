# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Gui-first-attempt.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1114, 824)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.frame_model = QtWidgets.QFrame(self.centralwidget)
        self.frame_model.setGeometry(QtCore.QRect(340, 30, 731, 401))
        self.frame_model.setMouseTracking(True)
        self.frame_model.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_model.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_model.setObjectName("frame_model")


        self.frame_box_1 = QtWidgets.QFrame(self.frame_model)
        self.frame_box_1.setGeometry(QtCore.QRect(220, 20, 120, 80))
        self.frame_box_1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.frame_box_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_box_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_box_1.setObjectName("frame_box_1")


        self.box_1 = QtWidgets.QGraphicsView(self.frame_box_1)
        self.box_1.setGeometry(QtCore.QRect(-10, -10, 131, 91))
        self.box_1.setObjectName("box_1")


        self.frame_box_2 = QtWidgets.QFrame(self.frame_model)
        self.frame_box_2.setGeometry(QtCore.QRect(220, 110, 120, 80))
        self.frame_box_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_box_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_box_2.setObjectName("frame_box_2")


        self.box_2 = QtWidgets.QGraphicsView(self.frame_box_2)
        self.box_2.setGeometry(QtCore.QRect(-10, -10, 131, 91))
        self.box_2.setObjectName("box_2")


        self.frame_box_3 = QtWidgets.QFrame(self.frame_model)
        self.frame_box_3.setGeometry(QtCore.QRect(220, 200, 120, 80))
        self.frame_box_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_box_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_box_3.setObjectName("frame_box_3")


        self.box_3 = QtWidgets.QGraphicsView(self.frame_box_3)
        self.box_3.setGeometry(QtCore.QRect(-10, -10, 131, 91))
        self.box_3.setObjectName("box_3")


        self.frame_box_4 = QtWidgets.QFrame(self.frame_model)
        self.frame_box_4.setGeometry(QtCore.QRect(220, 290, 120, 80))
        self.frame_box_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_box_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_box_4.setObjectName("frame_box_4")


        self.box_4 = QtWidgets.QGraphicsView(self.frame_box_4)
        self.box_4.setGeometry(QtCore.QRect(-10, -10, 131, 91))
        self.box_4.setObjectName("box_4")


        self.widget_decision_box = QtWidgets.QWidget(self.frame_model)
        self.widget_decision_box.setGeometry(QtCore.QRect(370, 50, 120, 291))
        self.widget_decision_box.setObjectName("widget_decision_box")


        self.graphicsView_5 = QtWidgets.QGraphicsView(self.widget_decision_box)
        self.graphicsView_5.setGeometry(QtCore.QRect(-10, -10, 131, 311))
        self.graphicsView_5.setObjectName("graphicsView_5")


        self.frame_input = QtWidgets.QFrame(self.frame_model)
        self.frame_input.setGeometry(QtCore.QRect(9, 50, 151, 291))
        self.frame_input.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_input.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_input.setObjectName("frame_input")


        self.text_input = QtWidgets.QTextEdit(self.frame_input)
        self.text_input.setGeometry(QtCore.QRect(0, 0, 131, 291))
        self.text_input.setObjectName("text_input")


        self.frame_box_outcome = QtWidgets.QFrame(self.frame_model)
        self.frame_box_outcome.setGeometry(QtCore.QRect(530, 50, 201, 291))
        self.frame_box_outcome.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_box_outcome.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_box_outcome.setObjectName("frame_box_outcome")


        self.graphicsView_6 = QtWidgets.QGraphicsView(self.frame_box_outcome)
        self.graphicsView_6.setGeometry(QtCore.QRect(-10, -10, 221, 311))
        self.graphicsView_6.setObjectName("graphicsView_6")


        self.frame_info = QtWidgets.QFrame(self.centralwidget)
        self.frame_info.setGeometry(QtCore.QRect(40, 460, 1031, 231))
        self.frame_info.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_info.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_info.setObjectName("frame_info")


        self.frame_info_pic_1 = QtWidgets.QFrame(self.frame_info)
        self.frame_info_pic_1.setGeometry(QtCore.QRect(0, 0, 331, 231))
        self.frame_info_pic_1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_info_pic_1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_info_pic_1.setObjectName("frame_info_pic_1")


        self.graphicsView_2 = QtWidgets.QGraphicsView(self.frame_info_pic_1)
        self.graphicsView_2.setGeometry(QtCore.QRect(-10, -10, 351, 251))
        self.graphicsView_2.setObjectName("graphicsView_2")


        self.frame_info_pic_2 = QtWidgets.QFrame(self.frame_info)
        self.frame_info_pic_2.setGeometry(QtCore.QRect(350, 0, 331, 231))
        self.frame_info_pic_2.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.frame_info_pic_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_info_pic_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_info_pic_2.setObjectName("frame_info_pic_2")


        self.graphicsView_3 = QtWidgets.QGraphicsView(self.frame_info_pic_2)
        self.graphicsView_3.setGeometry(QtCore.QRect(-10, -9, 351, 251))
        self.graphicsView_3.setObjectName("graphicsView_3")


        self.frame_info_pic_3 = QtWidgets.QFrame(self.frame_info)
        self.frame_info_pic_3.setGeometry(QtCore.QRect(700, 0, 331, 231))
        self.frame_info_pic_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_info_pic_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_info_pic_3.setObjectName("frame_info_pic_3")


        self.graphicsView_4 = QtWidgets.QGraphicsView(self.frame_info_pic_3)
        self.graphicsView_4.setGeometry(QtCore.QRect(-10, -10, 351, 251))
        self.graphicsView_4.setObjectName("graphicsView_4")


        self.frame_description = QtWidgets.QFrame(self.centralwidget)
        self.frame_description.setGeometry(QtCore.QRect(40, 30, 281, 401))
        self.frame_description.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_description.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_description.setObjectName("frame_description")


        self.frame_group_pic = QtWidgets.QFrame(self.frame_description)
        self.frame_group_pic.setGeometry(QtCore.QRect(0, 0, 281, 191))
        self.frame_group_pic.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
        self.frame_group_pic.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_group_pic.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_group_pic.setObjectName("frame_group_pic")
        self.graphicsView = QtWidgets.QGraphicsView(self.frame_group_pic)
        self.graphicsView.setGeometry(QtCore.QRect(-10, -10, 301, 211))
        self.graphicsView.setObjectName("graphicsView")


        self.frame_description_text = QtWidgets.QFrame(self.frame_description)
        self.frame_description_text.setGeometry(QtCore.QRect(0, 190, 281, 211))
        self.frame_description_text.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_description_text.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_description_text.setObjectName("frame_description_text")


        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.frame_description_text)
        self.plainTextEdit.setGeometry(QtCore.QRect(-7, -4, 291, 221))
        self.plainTextEdit.setObjectName("plainTextEdit")


        self.pushButton_classify = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_classify.setGeometry(QtCore.QRect(60, 710, 51, 51))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(232, 232, 232))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(232, 232, 232))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Highlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 120, 215))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Highlight, brush)
        self.pushButton_classify.setPalette(palette)
        self.pushButton_classify.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_classify.setCheckable(True)
        self.pushButton_classify.setChecked(False)
        self.pushButton_classify.setFlat(True)
        self.pushButton_classify.setObjectName("pushButton_classify")


        self.dial_volume = QtWidgets.QDial(self.centralwidget)
        self.dial_volume.setGeometry(QtCore.QRect(830, 690, 81, 91))
        self.dial_volume.setObjectName("dial_volume")


        self.dial_2 = QtWidgets.QDial(self.centralwidget)
        self.dial_2.setGeometry(QtCore.QRect(920, 690, 81, 91))
        self.dial_2.setObjectName("dial_2")


        self.lcd_time_frame = QtWidgets.QLCDNumber(self.centralwidget)
        self.lcd_time_frame.setGeometry(QtCore.QRect(390, 710, 331, 51))
        self.lcd_time_frame.setCursor(QtGui.QCursor(QtCore.Qt.BlankCursor))
        self.lcd_time_frame.setObjectName("lcd_time_frame")


        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(120, 710, 51, 51))
        self.pushButton_3.setFlat(True)
        self.pushButton_3.setObjectName("pushButton_3")


        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(180, 710, 51, 51))
        self.pushButton_4.setObjectName("pushButton_4")


        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(240, 710, 51, 51))
        self.pushButton_5.setObjectName("pushButton_5")


        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(300, 710, 51, 51))
        self.pushButton_6.setObjectName("pushButton_6")


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1114, 26))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.text_input.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:7.8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:14pt; font-weight:600;\">U P L O A D  M U S I C  F I L E</span></p></body></html>"))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "\n"
"\n"
"   M.U.G.G.E. - Mostly unfinished genre grating engine\n"
""))
        self.pushButton_classify.setText(_translate("MainWindow", "classify"))
        self.pushButton_3.setText(_translate("MainWindow", "classify"))
        self.pushButton_4.setText(_translate("MainWindow", "classify"))
        self.pushButton_5.setText(_translate("MainWindow", "classify"))
        self.pushButton_6.setText(_translate("MainWindow", "classify"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
