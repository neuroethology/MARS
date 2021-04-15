import sys
import time
from platform import system
from multiprocessing import Queue, set_start_method
from pathlib import Path
from MARS_queue import *

def get_mars_default_options():
    with open('config.yml') as f:
        default_opts = yaml.load(f)
    return default_opts


def run_MARS(folders, user_opts=dict()):

    # Get the queue of directory paths we'll start from.
    queue = Queue()
    for fid in folders:
        queue.put(str(Path(fid)))
    time.sleep(0.1)  # let the queue finish building

    mars_queue_engine(queue, user_opts, 'terminal')

    print('Finished processing all the data in the queue!')
    return


if __name__ == '__main__':
    # avoid multiprocessing problem on Macs
    if system() == 'Darwin':
        set_start_method('forkserver', force=True)
            
    # launch MARS gui if call came from terminal
    # more imports are needed for gui support
    from MARS_gui import *

    # Create the Qt Application
    app = QApplication(sys.argv)

    # Create and show the form
    frame = MainWindow()
    frame.show()

    # Run the main Qt loop
    sys.exit(app.exec_())

