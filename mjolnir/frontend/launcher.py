#!/usr/bin/env python3.5
import sys
from PyQt5 import QtGui, QtWidgets, QtCore

from mjolnir.frontend.gui import get_argparser
from mjolnir.drivers.camera import list_serial_numbers


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
        self.parser = get_argparser()
        self.init_ui()

    def init_connection(self):
        argv = [str(self.conn_type.currentText().lower())]
        if self.conn_type.currentIndex() == 0:
            argv.extend([
                    "-s", str(self.server.text()),
                    "-p", str(self.artiq_port.text()),
                    "-z", str(self.zmq_port.text())
                ])
        elif self.conn_type.currentIndex() == 1:
            argv.extend([
                    "-d", str(self.serial_number.currentText())
                ])

        try:
            args = self.parser.parse_args(argv)
        except Exception as e:
            self.error_message.setText(
                "Error parsing args: {}".format(e))
            self.error_message.show()
            return

        args.func(args)
        self.done(0)
        # try:
        #     args.func(args)
        # except Exception as e:
        #     self.error_message.setText(
        #         "Error initialising: {}".format(e))

    def _get_cam_list(self):
        self.serials = list_serial_numbers()

        self.serial_number.clear()
        self.serial_number.insertItems(0, self.serials)

    def init_ui(self):
        self.conn_type = QtGui.QComboBox()
        self.conn_type.insertItems(0, ["Remote", "Local"])
        self.conn = QtGui.QPushButton("Connect")
        self.conn.clicked.connect(self.init_connection)
        self.conn.setAutoDefault(True)

        self.error_message = QtGui.QLabel("")
        self.error_message.hide()

        # Remote connection params
        self.server = QtGui.QLineEdit()
        self.artiq_port = QtGui.QLineEdit()
        self.zmq_port = QtGui.QLineEdit()
        self.server.setText("127.0.0.1")
        self.artiq_port.setText("4000")
        self.zmq_port.setText("5555")

        # Local connection parameters
        # Could add a fetch numbers and replace this with a QComboBox
        self.get_cam_list = QtGui.QPushButton("Get Camera List")
        self.get_cam_list.clicked.connect(self._get_cam_list)
        self.serial_number = QtGui.QComboBox()
        self.serial_number.setEditable(True)

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
        self.remote_layout.addRow("IP Address:", self.server)
        self.remote_layout.addRow("ARTIQ port:", self.artiq_port)
        self.remote_layout.addRow("ZMQ port:", self.zmq_port)
        self.remote_params_widget = QtGui.QWidget()
        self.remote_params_widget.setLayout(self.remote_layout)

        self.local_layout = QtWidgets.QFormLayout()
        self.local_layout.addRow(self.get_cam_list)
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
