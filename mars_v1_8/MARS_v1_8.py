import sys, os
import PySide2
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2.QtCore import QRect
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (QApplication, QMainWindow, QMessageBox,
                          QAction, QWidget, QGridLayout, QLabel, QFileDialog,
                          QTextEdit, QMenuBar, QMenu, QStatusBar, QDesktopWidget,
                          QPushButton, QLineEdit, QCheckBox, QToolBar, QFrame,
                          QProgressBar, QHBoxLayout, QVBoxLayout)
import progressbar
from multiprocessing import Queue
import fnmatch
import time
import datetime
import MARS_pose_extractor as mpe
import MARS_feature_extractor as mfe
import MARS_classification_extractor as mce
import MARS_classification_machinery as mcm
import MARS_output_format as mof
import MARS_create_video as mcv
import pdb
import numpy as np


QApplication.setStyle('plastique')

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        #window setup
        resolution = QDesktopWidget().screenGeometry()
        self.screen_w = resolution.width()
        self.screen_h = resolution.height()
        self.setGeometry(0, 0, 650, 550)
        self.setWindowTitle('MARS_v1_8')
        self.setWindowIcon(QIcon('icons/mouse.png'))
        self.queue = Queue()
        self.queue_list = []
        self.str_proc = ''
        self.fname = ''

        #center window
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        #adjust size
        self.resize(self.screen_w / 2, self.screen_h / 2)
        self.Menu()
        self.Layout()

        central_widget = QWidget()
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

        resetAction = QAction(QIcon('icons/reset.png'), '&Reset', self)
        resetAction.setShortcut('Ctrl+X')
        resetAction.setStatusTip('Reset Mars')
        resetAction.triggered.connect(self.reset)


        #create status bar to show tooltip
        statusbar = QStatusBar(self)
        self.setStatusBar(statusbar)

        #create menu
        self.menu_File = QMenu('&File', self)
        self.menuBar().addMenu(self.menu_File)
        self.menu_File.addAction(openFile)
        self.menu_File.addAction(runAction)
        self.menu_File.addAction(resetAction)
        self.menu_File.addSeparator()
        self.menu_File.addAction(exitAction)

        #create toolbar
        self.toolbar = QToolBar("Toolbar", self)
        self.toolbar.addAction(openFile)
        self.toolbar.addAction(runAction)
        self.toolbar.addAction(resetAction)
        self.toolbar.addSeparator()
        # self.toolbar.move(5, 5)
        self.toolbar2 = QToolBar("ToolbarExit", self)
        self.toolbar2.addAction(exitAction)
 


        self.toto = QFrame(self)
        self.toto.setFrameShape(QFrame.HLine)
        self.toto.setFrameShadow(QFrame.Sunken)
        self.toto.setLineWidth(2)


    def Layout(self):
        #LAYOUT
        self.select_video = QLabel(self)
        self.select_video.setText("Folder Selected:")
        self.select_video.setVisible(False)

        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.setStatusTip(" Browse Folder")
        # self.browse_btn.setStyleSheet("background-color: rgb(186, 186, 186); border-radius: 15px;border-style: solid;border-width: 2px;border-color: black;");
        self.browse_btn.clicked.connect(self.browse)

        self.VideoName = QLabel(self)

        self.todo = QLabel(self)
        self.todo.setText("What do you want to do?")
        self.todo.setVisible(False)

        ## Various checkboxes for activities to perform within MARS.
        self.doPose = 0
        self.pose_chbox = QCheckBox('[Pose]', self)
        self.pose_chbox.stateChanged.connect(self.checkDoPose)
        self.pose_chbox.setVisible(False)

        self.doFeats = 0
        self.feat_chbox = QCheckBox('[Features]', self)
        self.feat_chbox.stateChanged.connect(self.checkDoFeat)
        self.feat_chbox.setVisible(False)

        self.doActions = 0
        self.actions_chbox = QCheckBox('[Classify Actions]', self)
        self.actions_chbox.stateChanged.connect(self.checkDoActions)
        self.actions_chbox.setVisible(False)

        # self.ddlist_label = QLabel(self)
        # self.ddlist_label.setText("Classifier:")
        # self.ddlist_label.move(200, 150)
        # self.ddlist_label.resize(150, 30)
        # self.ddlist_label.setVisible(False)
        #
        # self.ddlist = QComboBox(self)
        # self.ddlist.setVisible(False)
        # self.ddlist.setStatusTip('Choose the classifier you\'d like to use.')
        # self.ddlist.move(220, 120)
        # self.ddlist.resize(150, 50)
        # self.ddlist.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        self.doVideo = 0
        self.video_chbox = QCheckBox('[Produce Video]', self)
        self.video_chbox.stateChanged.connect(self.checkDoVideo)
        self.video_chbox.setVisible(False)

        self.doOverwrite = 0
        self.overwrite_chbox = QCheckBox('[Overwrite]', self)
        self.overwrite_chbox.setStyleSheet("background-color: #ff7a7a")
        self.overwrite_chbox.stateChanged.connect(self.checkDoOverwrite)
        self.overwrite_chbox.setVisible(False)

        ## Checkboxes that pick which view(s) to use, as well as the internal values they represent.
        self.doTop = 0
        self.top_chbox = QCheckBox('[Top]', self)
        self.top_chbox.stateChanged.connect(self.checkDoTop)
        self.top_chbox.setVisible(False)

        self.doToppcf = 0
        self.toppcf_chbox = QCheckBox('[Top (w/ Front pixel features)]', self)
        self.toppcf_chbox.stateChanged.connect(self.checkDoToppcf)
        self.toppcf_chbox.setVisible(False)

        self.doFront = 0
        self.front_chbox = QCheckBox('[Front]', self)
        self.front_chbox.stateChanged.connect(self.checkDoFront)
        self.front_chbox.setVisible(False)

        # Button to run MARS.
        self.run_mars = QPushButton("[Run MARS]", self)
        self.run_mars.setVisible(False)
        self.run_mars.setStatusTip('Run detection and actions classification')
        self.run_mars.setStyleSheet("background-color: rgb(142, 229, 171); border-radius: 15px;");
        self.run_mars.clicked.connect(self.run_event)

        # Button to reset the form for MARS.
        self.reset_btn = QPushButton("[Reset]", self)
        self.reset_btn.setVisible(False)
        self.reset_btn.setStatusTip('Reset buttons')
        self.reset_btn.setStyleSheet("background-color: rgb(229, 200, 142);border-radius: 15px");
        self.reset_btn.clicked.connect(self.reset)

        # Button for adding things to queue.
        self.add2queue_btn = QPushButton("[Enqueue]", self)
        self.add2queue_btn.setVisible(False)
        self.add2queue_btn.setStyleSheet("background-color: rgb(216,191,216);border-radius: 50px");
        self.add2queue_btn.clicked.connect(self.addToQueue)

        self.progress = QLabel(self)
        self.progress.setVisible(True)

        # Progress bar above the global progress, shows the progress on the current task.
        self.progbar = QProgressBar(self)
        self.progbar.setStyleSheet("background-color: #FFA07A; border: 3px solid #000000;");
        self.progbar.setVisible(True)
        self.progbar.setAlignment(QtCore.Qt.AlignCenter)
        # Label for progress bar.
        self.progbar_label = QLabel(self)
        self.progbar_label.setText("Current Task Progress:")

        # Big progress bar at the bottom. Global progress.
        self.big_progbar = QProgressBar(self)
        self.big_progbar.setVisible(True)
        self.big_progbar.setStyleSheet("background-color: #add8e6; border: 3px solid #FFFFFF;");
        self.big_progbar.setAlignment(QtCore.Qt.AlignCenter)
        # Label for big progress bar.
        self.big_progbar_label = QLabel(self)
        self.big_progbar_label.setText("Global Video Progress:")

        # Layout for the browsing span.
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.browse_btn)
        self.button_layout.addWidget(self.select_video)
        self.button_layout.addWidget(self.VideoName)
        self.button_layout.addWidget(self.add2queue_btn)
        self.button_layout.addStretch(0.5)

        # Layout for the menu at the top.
        self.menu_layout = QHBoxLayout()
        self.menu_layout.addWidget(self.toolbar)
        self.menu_layout.addStretch()
        self.menu_layout.addWidget(self.toolbar2)

        # Layout for the view selection (Top, Toppcf, Front)
        self.view_layout = QHBoxLayout()
        self.view_layout.addWidget(self.top_chbox)
        self.view_layout.addWidget(self.toppcf_chbox)
        self.view_layout.addWidget(self.front_chbox)
        self.view_layout.addStretch()

        # Layout for the checkboxes.
        self.chbox_layout = QHBoxLayout()
        self.chbox_layout.setSpacing(10)
        self.chbox_layout.addWidget(self.pose_chbox)
        self.chbox_layout.addWidget(self.feat_chbox)
        self.chbox_layout.addWidget(self.actions_chbox)
        self.chbox_layout.addWidget(self.video_chbox)
        self.chbox_layout.addWidget(self.overwrite_chbox)
        self.chbox_layout.addStretch(1)

        # Layout for the activity buttons, RUN and RESET.
        self.active_layout = QHBoxLayout()
        self.active_layout.addWidget(self.run_mars,stretch=2)
        self.active_layout.addWidget(self.reset_btn, stretch=1)

        # # Layout for the task progress bar.
        # self.task_progbar_layout = QtGui.QHBoxLayout()
        # self.task_progbar_layout.addWidget(self.progbar_label)
        # self.task_progbar_layout.addWidget(self.progbar, stretch=1)
        #
        # # Layout for the global progress bar.
        # self.global_progbar_layout = QtGui.QHBoxLayout()
        # self.global_progbar_layout.addWidget(self.big_progbar_label)
        # self.global_progbar_layout.addWidget(self.big_progbar)

        # Layout for the labels, to get ther vertically-aligned.
        self.progbar_label_layout = QVBoxLayout()
        self.progbar_label_layout.addWidget(self.progbar_label)
        self.progbar_label_layout.addWidget(self.big_progbar_label)

        # Layout for the progress bars themselves, to get them vertically-aligned.
        self.progbar_bar_layout = QVBoxLayout()
        self.progbar_bar_layout.addWidget(self.progbar)
        self.progbar_bar_layout.addWidget(self.big_progbar)

        # Layout for the combined progress bars and labels.
        self.progbar_layout = QHBoxLayout()
        self.progbar_layout.addLayout(self.progbar_label_layout)
        self.progbar_layout.addLayout(self.progbar_bar_layout, stretch = 1)

        # This layout puts everything on the screen.
        self.main_layout = QVBoxLayout()
        self.main_layout.addLayout(self.menu_layout)
        self.main_layout.addWidget(self.toto)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addWidget(self.todo)
        self.main_layout.addLayout(self.view_layout)
        self.main_layout.addLayout(self.chbox_layout)
        self.main_layout.addLayout(self.active_layout)
        self.main_layout.addWidget(self.progress)

        self.main_layout.addStretch()
        self.main_layout.addLayout(self.progbar_layout)
#        self.main_layout.addLayout(self.task_progbar_layout)
#        self.main_layout.addLayout(self.global_progbar_layout)

    def addToQueue(self):
        self.queue.put(self.fname)
        self.queue_list.append(self.fname)
        barMsg = self.fname + " added to the list!\n"
        msg = barMsg
        self.updateProgess( barMsg, msg)

    def browse(self):
        # sender = self.sender()
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly)
        if os.path.exists(self.fname):
            dir_to_use = self.fname
        else:
            dir_to_use = os.path.curdir

        self.fname = dialog.getExistingDirectory(self, 'Choose Directory', dir_to_use)
        self.statusBar().showMessage(self.fname + ' selected ')
        if os.path.exists(self.fname) and os.path.isdir(self.fname) and self.fname!='':
            files = os.listdir(self.fname)
            seq_files = [f for f in files if f.endswith('.seq') or f.endswith('.avi')]
            self.vid_name = self.fname.split('/')[-1]
            #if len(seq_files) >= 2:
            self.VideoName.setText(self.fname)
            self.todo.setVisible(True)

            self.select_video.setVisible(True)
            self.add2queue_btn.setVisible(True)
            # self.ddlist_label.setVisible(True)
            # self.ddlist.setVisible(True)

            self.pose_chbox.setVisible(True)
            self.feat_chbox.setVisible(True)
            self.actions_chbox.setVisible(True)
            self.video_chbox.setVisible(True)

            self.front_chbox.setVisible(True)
            self.top_chbox.setVisible(True)
            self.toppcf_chbox.setVisible(True)


            self.run_mars.setVisible(True)
            self.reset_btn.setVisible(True)
            self.overwrite_chbox.setVisible(True)
            #else:
            #    QMessageBox.information(self, "Not all needed files exists", "Select a folder containing .seq files!")
        else:
            QMessageBox.information(self, " Wrong file selected", "Select a folder containing .seq files!")
            self.fname = dir_to_use


    def checkDoPose(self, state):
        if state ==QtCore.Qt.Checked:
            self.doPose = 1
        else:
            self.doPose = 0

    def checkDoFeat(self, state):
        if state ==QtCore.Qt.Checked:
            self.doFeats = 1
        else:
            self.doFeats = 0

    def checkDoActions(self, state):
        if state ==QtCore.Qt.Checked:
            self.doActions = 1
        else:
            self.doActions = 0

    def checkDoVideo(self, state):
        if state ==QtCore.Qt.Checked:
            self.doVideo = 1
        else:
            self.doVideo = 0

    def checkDoOverwrite(self,state):
        if state == QtCore.Qt.Checked:
            self.doOverwrite = 1
        else:
            self.doOverwrite = 0

    def checkDoTop(self,state):
        if state == QtCore.Qt.Checked:
            self.doTop = 1
            # self.ddlist.addItem("top mlp")
            # self.ddlist.addItem("top xgb")
            # self.ddlist.addItem("top mlp wnd")
            # self.ddlist.addItem("top xgb wnd")
        else:
            self.doTop = 0
            # self.ddlist.clear()

    def checkDoToppcf(self,state):
        if state == QtCore.Qt.Checked:
            self.doToppcf = 1
            # self.ddlist.addItem("top pcf mlp")
            # self.ddlist.addItem("top pcf xgb")
            # self.ddlist.addItem("top pcf mlp wnd")
            # self.ddlist.addItem("top pcf xgb wnd")
        else:
            self.doToppcf = 0
            # self.ddlist.clear()

    def checkDoFront(self,state):
        if state == QtCore.Qt.Checked:
            self.doFront = 1
            # self.ddlist.addItem("topfront mlp")
            # self.ddlist.addItem("topfront xgb")
            # self.ddlist.addItem("topfront mlp wnd")
            # self.ddlist.addItem("topfront xgb wnd")
        else:
            self.doFront = 0
            # self.ddlist.clear()

    def reset(self):
        todo = [self.doPose, self.doFeats, self.doActions]
        # if not self.todo.isVisible() or  sum(todo)== 0:
        #     QMessageBox.information(self, "Reset", "Nothing to reset")
        # else:
        self.doPose = 0
        self.doFeats = 0
        self.doActions = 0
        self.doVideo = 0
        self.doOverwrite = 0

        self.doFront = 0
        self.doTop = 0
        self.doToppcf = 0

        self.pose_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.feat_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.actions_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.video_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.overwrite_chbox.setCheckState(QtCore.Qt.Unchecked)

        self.front_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.top_chbox.setCheckState(QtCore.Qt.Unchecked)
        self.toppcf_chbox.setCheckState(QtCore.Qt.Unchecked)

        self.str_proc = ''
        self.progress.setText(self.str_proc)

        self.VideoName.setText('')
        self.fname = ''

        self.statusBar().showMessage('')
        self.changeEnable_wdj(True)
        self.clearProgress()


    def changeEnable_wdj(self, b=False):
        self.run_mars.setEnabled(b)
        self.reset_btn.setEnabled(b)

        self.pose_chbox.setEnabled(b)
        self.feat_chbox.setEnabled(b)
        self.actions_chbox.setEnabled(b)
        self.video_chbox.setEnabled(b)
        self.overwrite_chbox.setEnabled(b)

        self.add2queue_btn.setEnabled(b)
        # self.ddlist.setEnabled(b)

        self.front_chbox.setEnabled(b)
        self.top_chbox.setEnabled(b)
        self.toppcf_chbox.setEnabled(b)


    # def stop_event(self):
    #     print('Stopped')
    #     # self.genericThread.stop()
    #     # self.genericThread.wait()
    #     self.statusBar().showMessage('Stopped processing')
    #     self.changeEnable_wdj(True)
    #     # self.stop_run.setVisible(False)

    def update_thread(self, prog):
        if prog ==1:
            print('Thread pose done')
        if prog ==2:
            print('Thread features done')
        if prog ==3:
            print('Thread actions done')
        if prog ==4:
            print('Thread video done')

    def thread_done(self):
        print('Thread ended')
        # self.changeEnable_wdj(True)
        # self.stop_run.setVisible(False)

    def queue_done(self):
        print('Queue ended')
        self.changeEnable_wdj(True)

    def updateProgess(self,barMsg,msg):
        self.statusBar().showMessage(barMsg)
        self.str_proc += msg
        self.progress.setText(self.str_proc)
        self.scrollText()

    def updateProgbar(self,value,set_max):
        if set_max != 0:
            self.progbar.setMaximum(set_max)
        self.progbar.setValue(value)

    def updateBigProgbar(self,value,set_max):
        if set_max != 0:
            self.big_progbar.setMaximum(set_max)
        self.big_progbar.setValue(value)

    def clearProgress(self):
        self.str_proc = 'Console cleared. \n'
        self.progress.setText(self.str_proc)
        self.resize(self.screen_w / 2, self.screen_h / 2)
        # self.adjustSize()
        # self.changeEnable_wdj(True)

    def scrollText(self):
        MAX_LINES = 20
        all_lines = self.str_proc.splitlines()
        if len(all_lines) > MAX_LINES:
            renewed_lines = all_lines[-MAX_LINES:]
            self.str_proc = '\n'.join(renewed_lines) + '\n'


    def run_event(self):
        todo = [self.doPose, self.doFeats, self.doActions, self.doVideo]
        if not self.todo.isVisible():
            QMessageBox.information(self, "Empty selection", "Select a folder to process.")
        elif sum(todo)== 0:
            QMessageBox.information(self, "Empty selection", "Select at least one task to do.")
        else:
            self.str_proc = ''
            self.progress.setVisible(True)
            self.genericThread = GenericThread(self.doPose, self.doFeats, self.doActions, self.doVideo,self.doOverwrite,
                                               self.doFront, self.doTop, self.doToppcf,self.queue, self.fname)
            self.genericThread.update_th.connect(self.update_thread)
            self.genericThread.done_th.connect(self.thread_done)
            self.genericThread.done_queue.connect(self.queue_done)

            self.genericThread.update_progbar_sig.connect(self.updateProgbar)
            self.genericThread.update_big_progbar_sig.connect(self.updateBigProgbar)
            self.genericThread.update_progress.connect(self.updateProgess)

            # self.genericThread.classifier_to_use = self.ddlist.currentText()
            self.genericThread.clear_sig.connect(self.clearProgress)

            self.genericThread.start()
            # self.stop_run.setVisible(True)
            self.changeEnable_wdj(False)


#Inherit from QThread
class GenericThread(QtCore.QThread):
    update_th = QtCore.Signal(int)
    update_progress = QtCore.Signal(str,str)
    update_progbar_sig = QtCore.Signal(float,float)
    update_big_progbar_sig = QtCore.Signal(int,int)

    clear_sig = QtCore.Signal()
    done_th = QtCore.Signal()
    done_queue = QtCore.Signal()

    # You can do any extra things in this init you need, but for this example
    # nothing else needs to be done expect call the super's init
    def __init__(self,  *args,**kwargs):
        QtCore.QThread.__init__(self, parent=None)
        self.args = args
        self.kwargs = kwargs

    # A QThread is run by calling its start() function, which calls this run()
    # function in its own "thread".
    def run(self):
        # Get the queue of directory paths we'll start from.
        queue = self.args[-2]

        # These are the checkboxes used.
        doPose = self.args[0]
        doFeats = self.args[1]
        doActions = self.args[2]
        doVideo = self.args[3]
        doOverwrite = self.args[4]

        doFront = self.args[5]
        doTop = self.args[6]
        doToppcf = self.args[7]

        root_count = 0 # A counter for the number of paths we've processed.

        while not (queue.empty()):
            # While there are still things in the queue...
            time.sleep(0.5) # (This pause exists to let us see things happening in the GUI.)

            # Get the next path.
            root_path = queue.get()

            # This is a counter for the number of files in this path we've processed. After we reach MAX_FILE_LOGS,
            #   we clear everything. Set MAX_FILE_LOGS to -1 to avoid this feature.
            count = 0
            MAX_FILE_LOGS = 4

            msg = "Processing root path " + str(root_count) + " : " + root_path + "\n"
            self.update_progress.emit(msg,msg)
            trials_to_run = []
            fullpaths = []

            # Walk through the subdirectories and get trials to process.
            for path, subdirs, filenames in os.walk(root_path):
                for numv,fname in enumerate(filenames):
                    cond1 = all(x not in fname for x in ['.seq','.avi','.mpg'])
                    cond2 = 'skipped' in path
                    cond3 = 'Top' not in fname
                    if cond1|cond2|cond3:
                        continue

                    if 'Top' in fname:
                        front_fname, top_fname, mouse_name  = mof.get_names(fname)
                        if top_fname != fname: continue
                        fullpath_to_front = os.path.join(path, front_fname)
                        fullpath_to_top = os.path.join(path, top_fname)
                        cond1 = not (os.path.exists(fullpath_to_front))
                        cond2 = doFront
                        cond3 = not doTop
                        cond4 = not doToppcf
                        # If the front video doesn't exist and we need it, continue.
                        if (cond1 and cond2):#|(cond3 and cond4):
                            continue
                    elif 'Front' in fname:
                        front_fname = fname
                        front_fname, top_fname, mouse_name = mof.get_names(front_fname)
                        # if front_fname != fname: continue
                        fullpath_to_top = os.path.join(path, top_fname)
                        fullpath_to_front = os.path.join(path, front_fname)
                        cond1 = not (os.path.exists(fullpath_to_top))
                        cond2 = doTop
                        cond3 = not doFront
                        cond4 = not doToppcf

                        if (cond1 and cond2 and cond4)|cond3:
                            continue
                    else:
                        # This is a seq file, but doesnt have "Top" or "Front" in it. Let's skip it.
                        continue

                    # Save the paths we want to use.
                    mouse_trial = dict()
                    mouse_trial['top'] = fullpath_to_top
                    mouse_trial['front'] = fullpath_to_front
                    trials_to_run.append(mouse_trial)
                    fullpaths.append(fullpath_to_top)

            # Get the indices to sort the trials by.
            idxs = np.argsort(fullpaths)

            # Sort the trials.
            placeholder = [trials_to_run[idx] for idx in idxs]
            trials_to_run = placeholder

            # Get the number of videos we're going to process, to give people an impression of time.
            total_valid_videos = len(trials_to_run)
            # Update our global progress bar.
            self.update_big_progbar_sig.emit(0, total_valid_videos-1)
            videos_processed = 0
            for trial in trials_to_run:
                try:
                    fullpath_to_top = trial['top']
                    fullpath_to_front = trial['front']

                    top_fname = os.path.basename(fullpath_to_top)
                    front_fname = os.path.basename(fullpath_to_front)

                    cumulative_progress = ' '.join([" |",
                                                    str(videos_processed + 1),
                                                    "out of",
                                                    str(total_valid_videos), "total videos"])

                    self.update_progress.emit('Processing ' + top_fname + cumulative_progress ,
                                              'Processing ' + top_fname + cumulative_progress + ' \n')

                    if doPose:

                        if doFront:
                            self.update_progress.emit("Extracting front pose from " + front_fname,
                                                      "Extracting front pose .... ")
                            # Extract the front pose.
                            mpe.extract_pose_wrapper(video_fullpath=fullpath_to_front,
                                                     view= 'front',
                                                     doOverwrite=doOverwrite,
                                                     progress_bar_signal=self.update_progbar_sig,
                                                     verbose=0)

                            self.update_progress.emit('', "saved.\n")

                        if doTop or doToppcf:
                            self.update_progress.emit("Extracting op pose from " + top_fname ,
                                                      "Extracting top pose .... ")
                            # Extract the front pose.
                            mpe.extract_pose_wrapper(video_fullpath=fullpath_to_top,
                                                     view= 'top',
                                                     doOverwrite=doOverwrite,
                                                     progress_bar_signal=self.update_progbar_sig,
                                                     verbose=0)

                            self.update_progress.emit('', "saved.\n")

                        if not (doFront|doTop|doToppcf):
                            view_msg = "ERROR: You need to select at least one view to use."
                            raise ValueError(view_msg)

                    if doFeats:
                        # TODO: If we wanted to do something special with the Front and Top views, we'd do it here.
                        # if doFront and doTop:
                        #     # Notify the GUI that feature extraction is starting.
                        #     self.update_progress.emit('Extracting features', 'Extracting features ....\n')
                        #
                        #     # Extract the features.
                        #     mfe.extract_both_features_wrapper(top_video_fullpath = fullpath_to_top,
                        #                                     front_video_fullpath = fullpath_to_front,
                        #                                     doOverwrite=doOverwrite,
                        #                                     output_suffix='')
                        #
                        #     # Notify the GUI that Features were extracted.
                        #     self.update_progress.emit('', 'Features saved\n')
                        #     self.update_th.emit(2)
                        if doTop:
                            self.update_progress.emit('Extracting top features from '+ top_fname ,
                                                      'Extracting top features  ....  ')

                            # Extract the features.
                            mfe.extract_top_features_wrapper(top_video_fullpath = fullpath_to_top,
                                                            doOverwrite=doOverwrite,
                                                            progress_bar_sig=self.update_progbar_sig,
                                                            output_suffix='')

                            # Notify the GUI that Features were extracted.
                            self.update_progress.emit('', 'Saved\n')
                            self.update_th.emit(2)

                        if doToppcf:
                            self.update_progress.emit('Extracting top pcf features from '+ top_fname ,
                                                      'Extracting top pcf features .... ')
                            # Extract the features.
                            mfe.extract_top_pcf_features_wrapper(top_video_fullpath=fullpath_to_top,
                                                                front_video_fullpath=fullpath_to_front,
                                                                doOverwrite=doOverwrite,
                                                                progress_bar_sig=self.update_progbar_sig,
                                                                output_suffix='')

                            # Notify the GUI that Features were extracted.
                            self.update_progress.emit('', 'Saved\n')
                            self.update_th.emit(2)

                        if doFront:
                            self.update_progress.emit('Extracting front features from '+ front_fname,
                                                      'Extracting front features ....  ')

                            # # Extract the features.
                            mfe.extract_front_features_wrapper(top_video_fullpath = fullpath_to_top,
                                                            front_video_fullpath = fullpath_to_front,
                                                            progress_bar_sig=self.update_progbar_sig,
                                                            doOverwrite=doOverwrite,
                                                            output_suffix='')

                            # Notify the GUI that Features were extracted.
                            self.update_progress.emit('', 'Saved\n')
                            self.update_th.emit(2)
                        if not (doTop | doToppcf):
                            view_msg = "Warning: You need to include a top view to use the classifier later."
                            raise Warning(view_msg)

                    if doActions:
                        if doToppcf:
                            # allowed_classifier=["top pcf mlp","top pcf xgb", "top pcf mlp wnd","top pcf xgb wnd"]
                            # if not classifier_to_use in allowed_classifier:
                            #     print('Selected not allowed classifier, default classifier will be used')
                            self.update_progress.emit('Prediting top pcf actions from ' + top_fname,
                                                      'Predicting top pcf actions ....  ')
                            # TODO: Don't hardcode this maybe?
                            classifier_type='top_pcf_tm_xgb500_wnd'

                            mce.classify_actions_wrapper(top_video_fullpath=fullpath_to_top,
                                                         front_video_fullpath='',
                                                         doOverwrite=doOverwrite,
                                                         view='toppcf',
                                                         classifier_path='models/classifier/' + classifier_type)
                            self.update_progress.emit('', "Saved\n")
                            self.update_th.emit(3)

                        if doTop:
                            self.update_progress.emit('Prediting top actions from ' + top_fname,
                                                      'Predicting top actions ....  ')
                            # TODO: Don't hardcode this maybe?

                            classifier_type='top_tm_xgb500_wnd'

                            mce.classify_actions_wrapper(top_video_fullpath= fullpath_to_top,
                                                            front_video_fullpath='',
                                                            doOverwrite=doOverwrite,
                                                            view='top',
                                                            classifier_path='models/classifier/' + classifier_type)
                            self.update_progress.emit('', "saved.\n")
                            self.update_th.emit(3)
                            # TODO: If you wanted to do something special with combined classification output, you'd do it here
                        if not doTop and not doToppcf:
                            view_msg = "ERROR: You need to select a top view to classify."
                            raise ValueError(view_msg)
                        mcm.dump_bento(video_fullpath=fullpath_to_top,
                                       basepath=root_path)

                    if doVideo:
                        if not (doFront | doTop | doToppcf):
                            view_msg = "ERROR: You need to select a view."
                            raise ValueError(view_msg)
                        else:
                            # TODO: Don't hardcode this maybe?
                            classifier_type='top_pcf_tm_xgb500_wnd' if doToppcf else 'top_tm_xgb500_wnd'

                            self.update_progress.emit('Creating results video from ' + top_fname,
                                                      'Creating results video .... ')
                            mcv.create_video_results_wrapper(top_video_fullpath=fullpath_to_top,
                                                             classifier_path='models/classifier/' + classifier_type,
                                                             progress_bar_signal=self.update_progbar_sig,
                                                             view='top',
                                                             doOverwrite=doOverwrite,
                                                             )
                            self.update_progress.emit('', 'saved\n')
                            self.update_th.emit(4)

                    # self.update_progress.emit(fname + ' done ', 'Done\n')
                    self.done_th.emit()
                    count += 1
                    videos_processed += 1
                    self.update_big_progbar_sig.emit(videos_processed,0)
                    if count == MAX_FILE_LOGS:
                        time.sleep(0.5)
                        self.clear_sig.emit()
                        count = 0

                except Exception as e:
                    print(e)
                    error_msg = 'ERROR: ' + os.path.join(path,fname) + ' has failed. \n' + str(e)
                    self.update_progress.emit(error_msg, error_msg + ' \n')
                    continue
                            # End of try-except block
                        # End of particular fname
                    # End of the particular root_path
            root_count += 1
            # End of the queue while loop
        self.clear_sig.emit()
        self.done_queue.emit()
        queue_msg =  'Finished processing all the data in the queue.\n'
        queue_msg += 'Add items to the queue (by selecting a folder and pressing [Enqueue]), '
        queue_msg += 'then press [Run MARS] to continue. \n'
        self.update_progress.emit(queue_msg, queue_msg)
        return


def get_time():
    ### Gets the current time and date in a string
    t = datetime.datetime.now()
    s = t.strftime('%Y-%m-%d %H:%M:%S.%f')
    return s



if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the form
    frame = MainWindow()
    frame.show()

    # Run the main Qt loop
    sys.exit(app.exec_())

