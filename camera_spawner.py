import subprocess
import sys
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets


def spawn_camera():
    subprocess.run([sys.executable,
        "-m", "camera_gui",
        "-s", "10.255.6.127",
        "-p", "4444",
        "-z", "5555"])


def main():
    app = QtWidgets.QApplication(sys.argv)

    spawn = QtGui.QPushButton("Spawn")
    spawn.clicked.connect(spawn_camera)

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(spawn)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()