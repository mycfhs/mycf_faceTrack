import sys
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import (
    QPushButton, QApplication,QMessageBox,QDesktopWidget,QMainWindow)
from PyQt5.QtGui import QIcon
import tkinter as tk
from tkinter import filedialog
from MyFaceTrackForUI import MyFaceTrack



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.extractorName = "cv2"
        self.trackerName = "cv2"
        self.Filepath = "0"
        self.initUI()

    def initUI(self):
        self.setObjectName("MainWindow")
        self.setStyleSheet("#MainWindow{border-image:url(background.png)}")  # 这里使用相对路径，也可以使用绝对路径
        qbtn = QPushButton('设置视频路径', self)#按钮
        qbtn.resize(qbtn.sizeHint())
        qbtn.move(600, 50)
        btn = QPushButton('人脸追踪(摄像头)', self)#按钮
        btn.resize(btn.sizeHint())
        btn.move(80, 50)
        btn2 = QPushButton('人脸追踪(视频)', self)  # 按钮
        btn2.resize(btn.sizeHint())
        btn2.move(350, 50)
        self.comboBoxl = QComboBox(self)
        self.comboBoxl.move(80,150)
        self.comboBoxl.addItems(['cv2', 'align', 'yolo'])
        self.comboBoxm = QComboBox(self)
        self.comboBoxm.move(350, 150)
        self.comboBoxm.addItems(['cv2','sort'])
        self.comboBoxl.currentIndexChanged[str].connect(self.vary_extractor_parametric)
        self.comboBoxm.currentIndexChanged[str].connect(self.vary_trackerName_parametric)
        qbtn.clicked.connect(self.buttonClicked)
        btn.clicked.connect(self.buttonClicked)
        btn2.clicked.connect(self.buttonClicked)
        self.statusBar()


        self.resize(795, 600)
        self.center()
        self.setWindowTitle('人脸追踪')
        self.setWindowIcon(QIcon('icon.jpg'))
        self.show()



    def center(self):#窗口居中

        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    def vary_extractor_parametric(self, parametric):
        self.extractorName = parametric

    def vary_trackerName_parametric(self, parametric):
        self.trackerName = parametric

    def closeEvent(self, event):#关闭窗口时提示
        reply = QMessageBox.question(self, 'Message',
                                     "您确定要关闭吗?", QMessageBox.Yes |
                                     QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def buttonClicked(self):
        sender = self.sender()
        if sender.text() == "设置视频路径":
            '''打开选择文件夹对话框'''
            root = tk.Tk()
            root.withdraw()
            self.Filepath = filedialog.askopenfilename()  # 获得选择好的文件

        if sender.text() == "人脸追踪(摄像头)":
            tracker = MyFaceTrack(self.extractorName, self.trackerName)
            tracker.trackMulti(0)
        if sender.text() == "人脸追踪(视频)":
            if self.Filepath != "0":
                tracker = MyFaceTrack(self.extractorName, self.trackerName)
                tracker.trackMulti(self.Filepath)



def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())#窗口退出


if __name__ == '__main__':
    main()