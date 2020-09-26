import os
import sys, os
import PySide
import MARS_output_format as mof
from PySide import QtCore
from PySide import QtGui
from PySide.QtCore import QRect
from PySide.QtGui import (QApplication, QMainWindow, QMessageBox,
                          QIcon, QAction, QWidget, QGridLayout, QLabel,
                          QTextEdit, QMenuBar, QMenu, QStatusBar, QDesktopWidget,
                          QPushButton, QLineEdit, QCheckBox)
from PySide.QtCore import *
from PySide.QtGui import *
import xlwt
import pdb

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        #widnow setup
        resolution = QDesktopWidget().screenGeometry()
        self.screen_w = resolution.width()
        self.screen_h = resolution.height()
        self.setGeometry(0, 0, 650, 200)
        self.setWindowTitle('MARS change version' + mof.get_version_suffix())
        self.setWindowIcon(QIcon('icons/rename.jpeg'))

        #center window
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        #adjust size
        self.resize(self.screen_w / 2, self.screen_h / 10)
        self.Menu()
        self.Layout()

        central_widget = QtGui.QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

    def Menu(self):
        #this creates an action exit, a shortcut and status tip
        exitAction = QAction(QIcon('icons/exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)

        openFile = QAction(QIcon('icons/open.png'), '&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.browse)

        runAction = QAction(QIcon('icons/run.png'), '&Run', self)
        runAction.setShortcut('Ctrl+R')
        runAction.setStatusTip('Run Mars')
        runAction.triggered.connect(self.run_event)

    def Layout(self):
        #LAYOUT
        self.directory_prompt = QLabel(self)
        self.directory_prompt.setText("Directory Selected:")
        self.directory_prompt.move(25,50)
        self.directory_prompt.resize(150,30)
        self.directory_prompt.setVisible(False)

        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.move(130, 50)
        self.browse_btn.resize(50,30)

        self.browse_btn.setStatusTip(" Browse Folder")
        # self.browse_btn.setStyleSheet("background-color: rgb(186, 186, 186); border-radius: 15px;border-style: solid;border-width: 2px;border-color: black;");
        self.browse_btn.clicked.connect(self.browse)

        self.dir_shower = QLabel(self)
        self.dir_shower.setText("")
        self.dir_shower.setVisible(False)

        self.label_version = QLabel(self)
        self.label_version.setText("Version to rename")
        self.label_version.setVisible(False)

        self.version  = QTextEdit()
        self.version.setStatusTip("Version to rename example: 1_6")
        self.version.setFixedHeight(30)
        self.version.setFixedWidth(40)
        self.version.setVisible(False)

        self.run_mars = QPushButton("RENAME", self)
        self.run_mars.setVisible(False)
        self.run_mars.move(50, 200)
        self.run_mars.resize(self.screen_w / 2 - 150, 50)
        self.run_mars.setStatusTip('')
        self.run_mars.setStyleSheet("background-color: rgb(142, 229, 171); border-radius: 15px;");
        self.run_mars.clicked.connect(self.run_event)

        self.menu_layout = QtGui.QHBoxLayout()
        self.menu_layout.addWidget(self.browse_btn)
        self.menu_layout.addWidget(self.directory_prompt)
        self.menu_layout.addWidget(self.dir_shower)
        self.menu_layout.addStretch()


        self.version_layout = QtGui.QHBoxLayout()
        self.version_layout.addWidget(self.label_version)
        self.version_layout.addWidget(self.version)
        self.version_layout.addStretch()

        self.run_layout = QtGui.QHBoxLayout()
        self.run_layout.addWidget(self.run_mars)

        self.main_layout = QtGui.QVBoxLayout()
        self.main_layout.addLayout(self.menu_layout)
        self.main_layout.addLayout(self.version_layout)
        self.main_layout.addLayout(self.run_layout)
        self.main_layout.addStretch()


    def browse(self):
        # sender = self.sender()
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.Directory)
        dialog.setOption(QtGui.QFileDialog.ShowDirsOnly)
        self.dirname = dialog.getExistingDirectory(self, 'Choose Directory', os.path.curdir)
        if os.path.exists(self.dirname) and os.path.isdir(self.dirname):
            self.dir_shower.setText(self.dirname)
            self.dir_shower.setVisible(True)
            self.directory_prompt.setVisible(True)
            self.version.setVisible(True)
            self.label_version.setVisible(True)
            self.run_mars.setVisible(True)
            return
        else:
            QMessageBox.information(self, " Wrong file selected", "Select a folder containing .seq files!")



    def run_event(self):
            self.genericThread = GenericThread(self.dirname,self.version.toPlainText())
            self.genericThread.start()



#Inherit from QThread
class GenericThread(QtCore.QThread):
    update_th = QtCore.Signal(int)
    update_progress = QtCore.Signal(str,str)
    update_progbar_sig = QtCore.Signal(int,int)
    clear_sig = QtCore.Signal()
    done_th = QtCore.Signal()

    #You can do any extra things in this init you need, but for this example
    #nothing else needs to be done expect call the super's init
    def __init__(self,  *args,**kwargs):
        QtCore.QThread.__init__(self, parent=None)
        self.args = args
        self.kwargs = kwargs


    #A QThread is run by calling it's start() function, which calls this run()
    #function in it's own "thread".
    def run(self):
        # Get the queue of directory paths we'll start from.
        root_path = self.args[0]
        version = self.args[1]
        rename(root_path,version)


def rename(path, ver_old):
    ver = mof.get_version_suffix()
    ver_old = 'v'+ver_old

    for dir, subdirs, filenames in os.walk(path):

        if ver_old in dir:
            old_dir=dir
            new_dir = dir.replace(ver_old,ver)
            if not os.path.exists(new_dir): os.rename(old_dir,new_dir)
        else: new_dir=dir

        for numv, fname in enumerate(filenames):
            if ver_old in fname:
                oldf = fname
                newf = fname.replace(ver_old, ver)
                if os.path.exists(os.path.join(new_dir, oldf)):
                    os.rename(os.path.join(new_dir,oldf), os.path.join(new_dir,newf))



if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the form
    frame = MainWindow()
    frame.show()

    # Run the main Qt loop
    sys.exit(app.exec_())