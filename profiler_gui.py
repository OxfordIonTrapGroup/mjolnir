

from camera_controller import camera_controller

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import sys
import traceback
import numpy as np
import png
import datetime
import image_tools

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'b')
pg.setConfigOption('antialias', True)


silder_hack_value = 1000.0

class ProfilerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Profiler GUI")
        self.fit_results = {}

        self.camera = camera_controller.start()
        self.camera.start_acquisition()

        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.setInterval(100)
        self.update_timer.start()


    def initUI(self):
        self.layout = QtWidgets.QHBoxLayout(self)
        self.widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.widget)

        self.info_pane = QtWidgets.QWidget(self)
        self.info_pane_layout = QtWidgets.QVBoxLayout(self)
        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)

        self.waist_title = QtWidgets.QLabel(self)
        self.waist_title.setText('Waists:')
        self.waist_title.setAlignment(QtCore.Qt.AlignLeft)

        self.h_waist_label = QtWidgets.QLineEdit(self)
        self.h_waist_label.setText('Wx: ')
        self.h_waist_label.setAlignment(QtCore.Qt.AlignLeft)
        self.h_waist_label.setReadOnly(True)

        self.v_waist_label = QtWidgets.QLineEdit(self)
        self.v_waist_label.setText('Wy: ')
        self.v_waist_label.setAlignment(QtCore.Qt.AlignLeft)
        self.v_waist_label.setReadOnly(True)


        self.amplitude_label = QtWidgets.QLabel(self)
        self.amplitude_label.setText('Amplitudes:')
        self.amplitude_label.setAlignment(QtCore.Qt.AlignLeft)

        self.h_amplitude_label = QtWidgets.QLineEdit(self)
        self.h_amplitude_label.setText('Ax: ')
        self.h_amplitude_label.setAlignment(QtCore.Qt.AlignLeft)
        self.h_amplitude_label.setReadOnly(True)

        self.v_amplitude_label = QtWidgets.QLineEdit(self)
        self.v_amplitude_label.setText('Ay: ')
        self.v_amplitude_label.setAlignment(QtCore.Qt.AlignLeft)
        self.v_amplitude_label.setReadOnly(True)


        self.positions_label = QtWidgets.QLabel(self)
        self.positions_label.setText('Centers:')
        self.positions_label.setAlignment(QtCore.Qt.AlignLeft)

        self.h_position_label = QtWidgets.QLineEdit(self)
        self.h_position_label.setText('x: ')
        self.h_position_label.setAlignment(QtCore.Qt.AlignLeft)
        self.h_position_label.setReadOnly(True)

        self.v_position_label = QtWidgets.QLineEdit(self)
        self.v_position_label.setText('y: ')
        self.v_position_label.setAlignment(QtCore.Qt.AlignLeft)
        self.v_position_label.setReadOnly(True)


        self.connection_box = QtWidgets.QCheckBox(self)
        self.connection_box.setText('Connected')
        self.connection_box.setEnabled(False)
        self.connection_box.setChecked(False)

        self.auto_expose_box = QtWidgets.QCheckBox(self)
        self.auto_expose_box.setText('Auto exposure')
        self.auto_expose_box.setChecked(True)

        self.exposure_slider = QtWidgets.QSlider(self)
        self.exposure_slider.setRange(0,100)
        self.exposure_slider.setValue(1)
        self.exposure_slider.setOrientation(QtCore.Qt.Horizontal)

        self.exposure_label = QtWidgets.QLineEdit(self)
        self.exposure_label.setText('Exposure: ')
        self.exposure_label.setAlignment(QtCore.Qt.AlignLeft)
        self.exposure_label.setReadOnly(True)

        self.save_button = QtWidgets.QPushButton(self)
        self.save_button.setText("Save")
        self.save_button.clicked.connect(self.save_button_clicked)

        self.info_pane_layout.addWidget(self.connection_box)
        self.info_pane_layout.addWidget(self.auto_expose_box)
        self.info_pane_layout.addWidget(self.exposure_slider)
        self.info_pane_layout.addWidget(self.exposure_label)

        self.info_pane_layout.addWidget(self.positions_label)
        self.info_pane_layout.addWidget(self.h_position_label)
        self.info_pane_layout.addWidget(self.v_position_label)


        self.info_pane_layout.addWidget(self.waist_title)
        self.info_pane_layout.addWidget(self.h_waist_label)
        self.info_pane_layout.addWidget(self.v_waist_label)


        self.info_pane_layout.addWidget(self.amplitude_label)
        self.info_pane_layout.addWidget(self.h_amplitude_label)
        self.info_pane_layout.addWidget(self.v_amplitude_label)

        self.info_pane_layout.addWidget(self.save_button)


        self.info_pane.setLayout(self.info_pane_layout)


        self.g_layout = pg.GraphicsLayoutWidget(self)

        self.vb_image = self.g_layout.addViewBox(row=0, col=0, lockAspect=True, enableMouse=False, invertY=True)
        self.vb_right = self.g_layout.addViewBox(row=0, col=1, enableMouse=False, border='k', invertY=True)
        self.vb_bottom = self.g_layout.addViewBox(row=1, col=0, enableMouse=False, border='k')
        self.vb_small_image = self.g_layout.addViewBox(row=1, col=1, lockAspect=True, enableMouse=False, invertY=True)

        qGraphicsGridLayout = self.g_layout.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 2)
        qGraphicsGridLayout.setRowStretchFactor(0, 2)

        #img = cam.acquire().transpose()
        #img = loadImage('Waists/colimated.bmp')[1]
        img = np.zeros((2,2))
        self.item_MainImage = pg.ImageItem(img)
        self.vb_image.addItem(self.item_MainImage)

        self.item_SmallImage = pg.ImageItem(img)
        self.vb_small_image.addItem(self.item_SmallImage)

        self.item_SmallImageOutline = pg.ImageItem(img)
        self.item_SmallImageOutline.setCompositionMode(QtGui.QPainter.CompositionMode_Plus)
        self.vb_small_image.addItem(self.item_SmallImageOutline)
        self.item_SmallImageOutline.setAutoDownsample(True)

        self.item_HPlotFit = pg.PlotDataItem(np.zeros(2),pen={'width':2})
        self.item_HPlot = pg.PlotDataItem(np.zeros(2),pen=None, symbol='o', pxMode=True, symbolSize=4)

        self.vb_bottom.addItem(self.item_HPlot)
        self.vb_bottom.addItem(self.item_HPlotFit)
        self.vb_bottom.setRange(yRange=[0,255])

        self.item_VPlotFit = pg.PlotDataItem(np.zeros(2),pen={'width':2})
        self.item_VPlot = pg.PlotDataItem(np.zeros(2),pen=None, symbol='o', pxMode=True, symbolSize=4)

        self.vb_right.addItem(self.item_VPlot)
        self.vb_right.addItem(self.item_VPlotFit)
        self.vb_right.setRange(xRange=[0,255])

        self.setGeometry(300, 300,900, 600)
        self.layout.addWidget(self.g_layout,stretch=2)
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.show()


    def update(self):

        if self.camera.connected:
            self.connection_box.setChecked(True)
        else:
            self.connection_box.setChecked(False)

        # Update auto-expsure
        auto_exposure = self.auto_expose_box.isChecked()
        self.camera.set_auto_exposure(auto_exposure)

        self.exposure_slider.setRange(
            self.camera.exposure_min*silder_hack_value,
            self.camera.exposure_max*silder_hack_value)
        self.exposure_slider.setSingleStep(
            self.camera.exposure_inc**silder_hack_value)

        if auto_exposure:
            self.exposure_slider.setValue(
                self.camera.exposure*silder_hack_value)
        else:
            self.camera.set_exposure(
                self.exposure_slider.value()/silder_hack_value)

        self.exposure_label.setText(
            'Exposure: {:.2f} ms'.format(self.camera.exposure))

        if not self.camera.new_image:
            return

        self.process_image()

    def process_image(self):
        self.camera.new_image = False
        image = self.camera.get_image()


        try:

            self.fit_results = image_tools.fit_image(image)
            r = self.fit_results

            # image_zoom, outlined_image,\
            #     row_x_fit, row_y_fit, row_x_data, row_y_data,\
            #     col_y_fit, col_x_fit, col_y_data, col_x_data,\
            #     wx, wy, x,y,\
            #     amp_x, amp_y\
            #      = self.fit_results

            self.h_waist_label.setText('Wx: {:.1f} um'.format(r["wx"]))
            self.v_waist_label.setText('Wy: {:.1f} um'.format(r["wy"]))

            self.h_amplitude_label.setText('Ax: {:.1f}'.format(r["amp_x"]))
            self.v_amplitude_label.setText('Ay: {:.1f}'.format(r["amp_y"]))

            self.h_position_label.setText('x: {:.1f} um'.format(r["x"]))
            self.v_position_label.setText('y: {:.1f} um'.format(r["y"]))

            self.item_MainImage.setImage(r["image_zoom"].T)
            self.item_SmallImage.setImage(image.T)
            self.item_SmallImageOutline.setImage(r["outlined_image"].T)

            self.item_HPlotFit.setData(r["row_x_fit"], r["row_y_fit"])
            self.item_HPlot.setData(r["row_x_data"], r["row_y_data"])

            self.item_VPlotFit.setData(r["col_y_fit"], r["col_x_fit"])
            self.item_VPlot.setData(r["col_y_data"], r["col_x_data"])

        except Exception as e:
            traceback.print_exc()
            self.item_MainImage.setImage(image.T)

            self.item_SmallImage.setImage(np.zeros((2,2)))
            self.item_SmallImageOutline.setImage(np.zeros((2,2)))

    def save_button_clicked(self):
        """Respond to the save button"""

        image = self.camera._image
        # Generate iso format time string
        file_name_base = datetime.datetime.now().isoformat().split(".")[0]
        file_name_base = file_name_base.replace(":","")


        f = open(file_name_base+".png", "wb")
        writer = png.Writer(greyscale=True,
            width=image.shape[1], height=image.shape[0])
        writer.write(f, image.tolist())
        f.close()

        output_dictionary = self.fit_results
        del output_dictionary["image_zoom"]
        del output_dictionary["outlined_image"]
        del output_dictionary["row_x_fit"]
        del output_dictionary["row_y_fit"]
        del output_dictionary["row_x_data"]
        del output_dictionary["row_y_data"]
        del output_dictionary["col_y_fit"]
        del output_dictionary["col_x_fit"]
        del output_dictionary["col_y_data"]
        del output_dictionary["col_x_data"]

        f = open(file_name_base+".txt", "w")
        for key in output_dictionary:
            f.write("{}, {}\n".format(key, output_dictionary[key]))
        f.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = ProfilerGUI()
    sys.exit(app.exec_())

