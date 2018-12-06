import sys
#pyqtgraph import needed to make QtGui import work...>
import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets, QtCore

from camera_gui import local, remote

class ConnectionDialog(QtWidgets.QDialog):
    """
    Dialogue window to initiate connections.

    - Can spawn multiple network clients
    - Clients stay alive even if dialog window is closed
    - Untested with local cameras
    - Untested with multiple local cameras
    """
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_local_connection(self):
        print("called local", flush=True)

    def init_remote_connection(self):
        server = self.ip_address.text()
        try:
            ctl_port = int(self.artiq_port.text())
        except ValueError as e:
            self.error_message.setText("Error: Invalid ARTIQ port")
            self.error_message.show()
            return

        try:
            zmq_port = int(self.zmq_port.text())
        except ValueError as e:
            self.error_message.setText("Error: Invalid ZMQ port")
            self.error_message.show()
            return

        try:
            remote(server, ctl_port, zmq_port)
        except Exception as e:
            self.error_message.setText("Error: {}".format(e))
            self.error_message.show()

    def init_connection(self):
        if self.conn_type.currentIndex() == 0:
            self.init_remote_connection()
        elif self.conn_type.currentIndex() == 1:
            self.init_local_connection()

    def init_ui(self):
        self.conn_type = QtGui.QComboBox()
        self.conn_type.insertItems(0, ["Remote", "Local"])
        self.conn = QtGui.QPushButton("Connect")
        self.conn.clicked.connect(self.init_connection)
        self.conn.setAutoDefault(True)

        self.error_message = QtGui.QLabel("")
        self.error_message.hide()

        # Remote connection params
        self.ip_address = QtGui.QLineEdit()
        self.artiq_port = QtGui.QLineEdit()
        self.zmq_port = QtGui.QLineEdit()
        self.ip_address.setText("127.0.0.1")
        self.artiq_port.setText("4000")
        self.zmq_port.setText("5555")

        # Local connection parameters
        # Could add a fetch numbers and replace this with a QComboBox
        self.serial_number = QtGui.QLineEdit()

        self.conn_params = QtGui.QStackedWidget()
        self.conn_type.currentIndexChanged.connect(
            self.conn_params.setCurrentIndex)

        # Layout everything
        self.conn_type_layout = QtWidgets.QFormLayout()
        self.conn_type_layout.addRow(
            QtGui.QLabel("Connection Type:"), self.conn_type)
        self.conn_type_widget = QtGui.QWidget()
        self.conn_type_widget.setLayout(self.conn_type_layout)

        self.remote_layout = QtWidgets.QFormLayout()
        self.remote_layout.addRow("IP Address:", self.ip_address)
        self.remote_layout.addRow("ARTIQ port:", self.artiq_port)
        self.remote_layout.addRow("ZMQ port:", self.zmq_port)
        self.remote_params_widget = QtGui.QWidget()
        self.remote_params_widget.setLayout(self.remote_layout)

        self.local_layout = QtWidgets.QFormLayout()
        self.local_layout.addRow("Serial Number:", self.serial_number)
        self.local_params_widget = QtGui.QWidget()
        self.local_params_widget.setLayout(self.local_layout)

        self.conn_params.addWidget(self.remote_params_widget)
        self.conn_params.addWidget(self.local_params_widget)

        divider = QtGui.QFrame()
        divider.setFrameShape(QtGui.QFrame.HLine)
        divider.setLineWidth(1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.conn_type_widget)
        self.layout.addWidget(divider)
        self.layout.addWidget(self.conn_params)
        self.layout.addWidget(self.conn)
        self.layout.addWidget(self.error_message)


def main():
    app  = QtWidgets.QApplication(sys.argv)

    conn = ConnectionDialog()
    conn.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
