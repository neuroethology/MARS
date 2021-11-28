import os
import MARS_output_format as mof
import sys, os
import PySide
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
# for path_count, root_path in enumerate(path_list):
#     # While there are still things in the queue...
#     time.sleep(0.5)  # (This pause exists to let us see things happening in the GUI.)
#
#     count = 0
#     MAX_FILE_LOGS = 4
#
#     msg = "Processing root path " + str(path_count) + " : " + root_path + "\n"
#     print(msg)
#     # Walk through the subdirectories and files and track the videos within.

def dump_bento_across_dir(root_path):
    ''' This function makes a bento file for a specific directory.'''
    wb = xlwt.Workbook(encoding='utf-8')
    ws1 = wb.add_sheet('Sheet1', cell_overwrite_ok=True)
    ws1.write(0, 0, os.path.abspath(root_path))  # A1
    ws1.write(0, 1, 'Ca framerate:')  # B1
    ws1.write(0, 2, 0)  # C1
    ws1.write(0, 3, 'Annot framerate:')  # D1
    ws1.write(0, 4, 30)  # E1
    ws1.write(0, 5, 'Multiple trials/Ca file:')  # F1
    ws1.write(0, 6, 0)  # G1
    ws1.write(0, 7, 'Multiple trails/annot file')  # H1
    ws1.write(0, 8, 0)  # I1
    ws1.write(0, 9, 'Includes behavior movies:')  # J1
    ws1.write(0, 10, 1)  # K1
    ws1.write(0, 11, 'Offset (in seconds; positive values = annot starts before Ca):')  # L1
    ws1.write(0, 12, 0)  # M1
    ws1.write(0,13,'Includes tracking data:')
    ws1.write(0,14, 0)
    ws1.write(0,15, 'Includes audio files:')
    ws1.write(0,16, 0)

    ws1.write(1, 0, 'Mouse')  # A2
    mouse_col_num = 0
    session_col_num = 1
    trial_col_num = 2
    ws1.write(1, 1, 'Sessn')  # B2
    ws1.write(1, 2, 'Trial')  # C2
    ws1.write(1, 3, 'Stim')  # D2
    ws1.write(1, 4, 'Calcium imaging file')  # E2
    ws1.write(1, 5, 'Start Ca')  # F2
    ws1.write(1, 6, 'Stop Ca')  # G2
    ws1.write(1, 7, 'FR Ca')  # H2
    ws1.write(1, 8, 'Alignments')  # I2
    ws1.write(1, 9, 'Annotation file')  # J2
    annot_file_col_num = 9
    ws1.write(1, 10, 'Start Anno')  # K2
    ws1.write(1, 11, 'Stop Anno')  # L2
    ws1.write(1, 12, 'FR Anno')  # M2
    ws1.write(1, 13, 'Offset')  # N2
    ws1.write(1, 14, 'Behavior movie')  # O2
    behavior_movie_col_num = 14
    ws1.write(1, 15, 'Tracking')  # P2
    tracking_file_col_num = 15
    ws1.write(1, 16, 'Audio file')
    audio_file_col_num = 16
    ws1.write(1, 17, 'tSNE')

    # ws1.write(2, 0, 1)  # A2
    # ws1.write(2, 1, 1)  # B2
    # ws1.write(2, 2, 1)  # C2
    # ws1.write(2, 3, '')  # D2
    # ws1.write(2, 4, '')  # E2
    # ws1.write(2, 5, '')  # F2
    # ws1.write(2, 6, '')  # G2
    # ws1.write(2, 7, '')  # H2
    # ws1.write(2, 8, '')  # I2
    mouse_number = 0
    row_num = 2
    # Going through everything in this directory.
    audio_filenames = []
    add_audio_count = 0
    nonaudio_filenames = []
    for path, subdirs, filenames in os.walk(root_path):
        for fname in sorted(filenames):
            fname = os.path.join(path,fname)
            if fname.endswith('.wav'):
                audio_filenames.append(fname)
            else:
                nonaudio_filenames.append(fname)

        audio_filenames = sorted(audio_filenames)
        nonaudio_filenames = sorted(nonaudio_filenames)


    for fname in nonaudio_filenames:
        try:
            cond1 = any(x in fname for x in mof.get_supported_formats())
            if not cond1:
                continue

            front_fname, top_fname, mouse_name = mof.get_names(fname, pair_files=False)
            fullpath_to_top = os.path.join(path, top_fname)

            # Add their info to the bento file at the appropriate level.

            video_fullpath = fullpath_to_top

            output_suffix= ''
            video_path = os.path.dirname(video_fullpath)
            video_name = os.path.basename(video_fullpath)

            # Get the output folder for this specific mouse.
            output_folder = mof.get_mouse_output_dir(dir_output_should_be_in=video_path, video_name=video_name,
                                                     output_suffix=output_suffix)
            _,_,mouse_name = mof.get_names(video_name=video_name)


            pose_basename = mof.get_pose_no_ext(video_fullpath=video_fullpath,
                                                    output_folder=output_folder,
                                                    view='top',
                                                    output_suffix=output_suffix)

            top_pose_fullpath = pose_basename + '.json'

            same_path_ann = [os.path.join(root_path,f)
                             for f in os.listdir(root_path) if is_annotation_file(f, mouse_name)]

            ann = [os.path.join(output_folder, f)
                   for f in os.listdir(output_folder) if is_annotation_file(f,mouse_name)]

            ann = sorted(ann)
            ann = [get_normrel_path(f,root_path) for f in ann]

            pose_cond = os.path.exists(top_pose_fullpath)
            video_cond = os.path.exists(video_fullpath)

            should_write = (pose_cond and video_cond)

            if should_write:
                old_mouse_number = mouse_number
                mouse_number = get_mouse_number(video_fullpath)

                mouse_cond = (old_mouse_number == mouse_number)
                # TODO: Session condition
                sess_cond = (True)

                if mouse_cond and sess_cond:
                    trial_count += 1
                else:
                    trial_count = 1

                ws1.write(row_num, mouse_col_num, mouse_number)  # A2
                ws1.write(row_num, session_col_num, 1)  # B2
                ws1.write(row_num, trial_col_num, trial_count)  # C2

                ws1.write(row_num, annot_file_col_num, ';'.join(ann))  # J2
                ws1.write(row_num, 10, '')  # K2
                ws1.write(row_num, 11, '')  # L2
                ws1.write(row_num, 12, '')  # M2
                ws1.write(row_num, 13, '')  # N2


                track_file = get_normrel_path(top_pose_fullpath, root_path)

                ws1.write(row_num, behavior_movie_col_num, get_normrel_path(fullpath_to_top, root_path)) # O2
                ws1.write(row_num, tracking_file_col_num, track_file)  # P2
                row_num += 1
        except Exception as e:
                print(e)
                error_msg = 'ERROR: ' + fname + ' has failed. ' + str(e)

                continue
                # End of try-except block
                # End of particular fname
                # End of the particular root_path

    last_row = row_num
    row_num = 2
    for audio_file_count, audio_file in enumerate(audio_filenames):
        # Write the files in order.
        ws1.write(row_num + audio_file_count, audio_file_col_num + 2, get_normrel_path(audio_file,root_path))


    bento_name = 'bento_' + mof.get_version_suffix() + '.xls'
    wb.save(os.path.join(root_path, bento_name))
    return


def get_normrel_path(path, start=''):
    return (os.path.relpath(path,start))


def is_annotation_file(filename, mouse_name):
    cond1 = filename.startswith(mouse_name)
    cond2 = filename.endswith('.txt')
    cond3 = os.path.exists(filename)
    cond4 = filename.endswith('.annot')
    return (cond1 and(cond2 |cond4))


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        #widnow setup
        resolution = QDesktopWidget().screenGeometry()
        self.screen_w = resolution.width()
        self.screen_h = resolution.height()
        self.setGeometry(0, 0, 650, 200)
        self.setWindowTitle('bento dumper' + mof.get_version_suffix())
        self.setWindowIcon(QIcon('icons/run.png'))

        #center window
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        #adjust size
        self.resize(self.screen_w / 2, self.screen_h / 16)
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

        self.browse_btn = QPushButton("Browse", self)
        self.browse_btn.move(130, 50)
        self.browse_btn.setStatusTip(" Browse Folder")
        # self.browse_btn.setStyleSheet("background-color: rgb(186, 186, 186); border-radius: 15px;border-style: solid;border-width: 2px;border-color: black;");
        self.browse_btn.clicked.connect(self.browse)

        self.dir_shower = QLabel(self)
        self.dir_shower.setText("Directory")


        self.run_mars = QPushButton("Dump BENTO", self)
        self.run_mars.setVisible(True)
        self.run_mars.move(25, 160)
        self.run_mars.resize(self.screen_w / 2 - 150, 50)
        self.run_mars.setStatusTip('')
        self.run_mars.setStyleSheet("background-color: rgb(142, 229, 171); border-radius: 15px;");
        self.run_mars.clicked.connect(self.run_event)

        self.menu_layout = QtGui.QHBoxLayout()
        self.menu_layout.addWidget(self.browse_btn)
        self.menu_layout.addWidget(self.directory_prompt)
        self.menu_layout.addWidget(self.dir_shower)
        self.menu_layout.addStretch()

        self.run_layout = QtGui.QHBoxLayout()
        self.run_layout.addWidget(self.run_mars)



        self.main_layout = QtGui.QVBoxLayout()
        self.main_layout.addLayout(self.menu_layout)
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
            return
        else:
            QMessageBox.information(self, " Wrong file selected", "Select a folder containing .seq files!")



    def run_event(self):
            self.genericThread = GenericThread(self.dirname)
            self.genericThread.start()
            # self.stop_run.setVisible(True)






#Inherit from QThread
class GenericThread(QtCore.QThread):
    update_th = QtCore.Signal(int)
    update_progress = QtCore.Signal(str,str)
    update_progbar_sig = QtCore.Signal(int,int)
    clear_sig = QtCore.Signal()
    done_th = QtCore.Signal()
    classifier_to_use = "tomomi_fix"

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
        dump_bento_across_dir(root_path=root_path)

def get_mouse_number(video_fullpath):
    # Assumes path is something like: path/to/Mouse[NUM]_(yadda_yadda.ext)
    video_name = os.path.basename(video_fullpath)
    mouse_name_and_number = video_name.split('_')[0]
    mouse_num = mouse_name_and_number[5:]
    if mouse_num == '':
        mouse_num = 1
    else:
        mouse_num = int(mouse_num)
    return mouse_num



if __name__ == '__main__':
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the form
    frame = MainWindow()
    frame.show()

    # Run the main Qt loop
    sys.exit(app.exec_())
