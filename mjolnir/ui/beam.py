import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
from collections import deque

from mjolnir.ui.worker import Worker
from mjolnir.tools import tools


class BeamDisplay(QtWidgets.QMainWindow):
    new_image = QtCore.pyqtSignal()

    def __init__(self, camera):
        super().__init__()

        self.cam = camera
        # Pixel width in microns (get from camera?)
        self._px_width = 5.2

        # Deques discard the oldest value when full
        self.imageq = deque(maxlen=3)
        self.updateq = deque(maxlen=3)
        self.worker = Worker(self.imageq, self.updateq)
        self.thread = QtCore.QThread()
        self.worker.moveToThread(self.thread)

        # must occur after moving worker!
        self.new_image.connect(self.worker.process_imageq)
        self.worker.new_update.connect(self.update)
        self.thread.start()

        self._fps = None
        self._last_update = QtCore.QTime.currentTime()
        self._mark = None
        self._residual_levels = [-10,10]

        self.init_ui()
        self.show()

    @QtCore.pyqtSlot(np.ndarray)
    def queue_image(self, im):
        """Queues an image for fitting and plotting"""
        self.imageq.append(im)
        self.new_image.emit()

    @QtCore.pyqtSlot()
    def update(self):
        try:
            up = self.updateq.popleft()
        except IndexError:
            return

        options = {'autoRange': False, 'autoLevels': False}
        self.image.setImage(up['im'], **options)
        self.zoom.setImage(up['im_crop'], **options)
        self.residuals.setImage(up['im_res'], lut=self.residual_LUT, **options)

        self.x_slice.setData(up['x'], up['x_slice'])
        self.x_fit.setData(up['x'], up['x_fit'])

        self.y_slice.setData(up['y_slice'], up['y'])
        self.y_fit.setData(up['y_fit'], up['y'])

        # cache the centroid in case we need to set a mark
        self._centre = up['x0']

        # Sub-pixel position works with QPointF
        point = QtCore.QPointF(*up['x0'])
        self.fit_v_line.setValue(point)
        self.fit_h_line.setValue(point)

        nopen = pg.mkPen(style=QtCore.Qt.NoPen)
        self.history.append(up['x0'])
        self.history_plot.setData(
            pos=self.history,
            pen=nopen,
            brush=self.history_brushes[-len(self.history):])

        # 'centre' is a QPointF
        self.fit_maj_line.setValue(up['zoom_centre'])
        self.fit_min_line.setValue(up['zoom_centre'])
        self.fit_maj_line.setAngle(up['semimaj_angle'])
        self.fit_min_line.setAngle(up['semimin_angle'])

        self.isocurve.setLevel(up['iso_level'])
        self.isocurve.setData(up['im_fit'])

        def px_string(px):
            return "{:.1f}μm ({:.1f}px)".format(px*self._px_width, px)

        self.maj_radius.setText(px_string(up['semimaj']))
        self.min_radius.setText(px_string(up['semimin']))
        self.avg_radius.setText(px_string(up['avg_radius']))
        self.x_radius.setText(px_string(up['x_radius']))
        self.y_radius.setText(px_string(up['y_radius']))
        self.x_centroid.setText(px_string(up['x0'][0]))
        self.y_centroid.setText(px_string(up['x0'][1]))
        self.ellipticity.setText("{:.3f}".format(up['e']))

        if self._mark is not None:
            delta = up['x0'] - self._mark
            self.x_delta.setText(px_string(delta[0]))
            self.y_delta.setText(px_string(delta[1]))

        self._last_update, self._fps = tools.update_rate(
            self._last_update, self._fps)
        self.fps.setText("{:.1f} fps".format(self._fps))
        self.cps.setText("{:.1f} cps".format(up['cps']))

    def get_exposure_params(self):
        val, min_, max_, step = self.cam.get_exposure_params()
        self.exposure.setRange(min_, max_)
        self.exposure.setSingleStep(step)
        self.exposure.setValue(val)

    def exposure_cb(self):
        exp = self.exposure.value()
        self.cam.set_exposure_ms(exp)

    def aoi_cb(self):
        pass

    def mark_cb(self):
        self._mark = self._centre
        point = QtCore.QPointF(*self._mark)
        self.mark_v_line.setValue(point)
        self.mark_h_line.setValue(point)
        self.mark_v_line.show()
        self.mark_h_line.show()

        # Not ideal but don't want to duplicate px_string from update
        # If the user is using in continuous mode they'll never notice...
        self.x_delta.setText('')
        self.y_delta.setText('')
        self.x_delta.show()
        self.y_delta.show()

    def unmark_cb(self):
        self._mark = None
        self.mark_v_line.hide()
        self.mark_h_line.hide()
        self.x_delta.hide()
        self.y_delta.hide()

    def get_color_map(self):
        # Colour map for residuals is transparent when residual is zero
        colors = np.array([
            (0, 255, 255, 255),
            (0, 0, 255, 191),
            (0, 0, 0, 0),
            (255, 0, 0, 191),
            (255, 255, 0, 255)
        ], dtype=np.uint8)
        positions = [0.25 * (2 + i) for i in range(-2,3)]
        return pg.ColorMap(positions, colors)

    def init_ui(self):
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()

        self.init_graphics()
        self.init_info_pane()

        self.layout.addWidget(self.g_layout, stretch=2)
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setGeometry(300, 300, 1500, 600)

    def init_info_pane(self):
        self.single = QtGui.QPushButton("Single Acquisition")
        self.start = QtGui.QPushButton("Start Acquisition")
        self.stop = QtGui.QPushButton("Stop Acquisition")
        self.single.clicked.connect(lambda: self.cam.single_acquisition())
        self.start.clicked.connect(lambda: self.cam.start_acquisition())
        self.stop.clicked.connect(lambda: self.cam.stop_acquisition())

        self.exposure = QtGui.QDoubleSpinBox()
        self.exposure.setSuffix(" ms")
        self.get_exposure_params()
        # connect after finding params so we don't send accidental update
        self.exposure.valueChanged.connect(self.exposure_cb)

        self.maj_radius = QtGui.QLabel()
        self.min_radius = QtGui.QLabel()
        self.avg_radius = QtGui.QLabel()
        self.ellipticity = QtGui.QLabel()
        self.x_radius = QtGui.QLabel()
        self.y_radius = QtGui.QLabel()
        self.x_centroid = QtGui.QLabel()
        self.y_centroid = QtGui.QLabel()

        # Mark current beam position
        self.mark = QtGui.QPushButton("Mark")
        self.unmark = QtGui.QPushButton("Unmark")
        self.mark.clicked.connect(self.mark_cb)
        self.unmark.clicked.connect(self.unmark_cb)

        # Distance from marked location
        self.x_delta = QtGui.QLabel()
        self.y_delta = QtGui.QLabel()
        self.x_delta.hide()
        self.y_delta.hide()

        self.fps = QtGui.QLabel()
        self.cps = QtGui.QLabel()

        #
        # Add the widgets to their layouts
        #
        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.addRow(QtGui.QLabel("Beam Parameters"))
        self.param_layout.addRow(QtGui.QLabel("(all radii are 1/e^2)"))
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("Semi-major radius:", self.maj_radius)
        self.param_layout.addRow("Semi-minor radius:", self.min_radius)
        self.param_layout.addRow("Average radius:", self.avg_radius)
        self.param_layout.addRow("Ellipticity:", self.ellipticity)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("X radius:", self.x_radius)
        self.param_layout.addRow("Y radius:", self.y_radius)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow("X position:", self.x_centroid)
        self.param_layout.addRow("Y position:", self.y_centroid)
        self.param_layout.addRow(QtGui.QWidget())
        self.param_layout.addRow(self.mark, self.unmark)
        self.param_layout.addRow("ΔX:", self.x_delta)
        self.param_layout.addRow("ΔY:", self.y_delta)

        self.param_widget = QtGui.QWidget()
        self.param_widget.setLayout(self.param_layout)

        self.info_pane_layout = QtWidgets.QVBoxLayout()
        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)
        self.info_pane_layout.addWidget(self.start)
        self.info_pane_layout.addWidget(self.single)
        self.info_pane_layout.addWidget(self.stop)
        self.info_pane_layout.addWidget(self.exposure)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.param_widget)
        self.info_pane_layout.addStretch(3)
        self.info_pane_layout.addWidget(self.fps)
        self.info_pane_layout.addWidget(self.cps)

        self.info_pane = QtWidgets.QWidget(self)
        self.info_pane.setLayout(self.info_pane_layout)

    def init_graphics(self):
        # Graphics layout object to place viewboxes in
        self.g_layout = pg.GraphicsLayoutWidget(border=(80, 80, 80))

        m, n = 1280, 1024
        self.image = pg.ImageItem(np.zeros((m,n)))
        self.zoom = pg.ImageItem(np.zeros((50,50)))
        self.residuals = pg.ImageItem(np.zeros((50,50)))
        self.residuals.setLevels(self._residual_levels)
        self.x_fit = pg.PlotDataItem(np.zeros(m), pen={'width':2})
        self.x_slice = pg.PlotDataItem(np.zeros(m), pen=None, symbol='o', pxMode=True, symbolSize=4)
        self.y_fit = pg.PlotDataItem(np.zeros(n), pen={'width':2})
        self.y_slice = pg.PlotDataItem(np.zeros(n), pen=None, symbol='o', pxMode=True, symbolSize=4)

        # Viewboxes for images
        # aspect locked so that pixels are square
        # y inverted so that (0,0) is top left as in Thorlabs software
        options = {"lockAspect":True, "invertY":True}
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=2, **options)
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=2, **options)
        self.vb_residuals = self.g_layout.addViewBox(row=1, col=2, **options)

        # Only the residuals have any sort of false color - initialise the
        # lookup table and the legend
        cmap = self.get_color_map()
        self.residual_LUT = cmap.getLookupTable(nPts=256)
        self.res_legend = pg.GradientLegend((10,255),(0, 20))
        self.res_legend.setGradient(cmap.getGradient())
        self.res_legend.setParentItem(self.vb_residuals)
        n_ticks = 5
        self.res_legend.setLabels({"{}".format(level):val
            for (level, val) in zip(
                np.linspace(*self._residual_levels, n_ticks),
                np.linspace(0, 1, n_ticks))})

        ypen = pg.mkPen(color=(255,255,0,85), width=3)
        rpen = pg.mkPen(color=(255,0,0,127), width=3, style=QtCore.Qt.DotLine)

        # Centroid position markers in main image, aligned with x,y
        self.fit_v_line = pg.InfiniteLine(pos=1, angle=90, pen=ypen)
        self.fit_h_line = pg.InfiniteLine(pos=1, angle=0, pen=ypen)

        # Plot fading recent position markers
        n_history = 5
        self.history = deque(maxlen=n_history)
        self.history_plot = pg.ScatterPlotItem()
        self.history_brushes = [pg.mkBrush(
            color=(255,255,0,int((i+1)*255/n_history)))
            for i in range(n_history)]

        # User marked position
        self.mark_v_line = pg.InfiniteLine(pos=1, angle=90, pen=rpen)
        self.mark_h_line = pg.InfiniteLine(pos=1, angle=0, pen=rpen)
        self.mark_v_line.hide()
        self.mark_h_line.hide()

        # Centroid position markers in zoomed image, aligned with beam
        # ellipse axes
        zoom_centre = QtCore.QPointF(25,25)
        self.fit_maj_line = pg.InfiniteLine(pos=zoom_centre, angle=90, pen=ypen)
        self.fit_min_line = pg.InfiniteLine(pos=zoom_centre, angle=0, pen=ypen)

        # Shows 1/e^2 ellipse of beam
        isopen = pg.mkPen(color=(255,255,0,85), width=3, style=QtCore.Qt.DotLine)
        self.isocurve = pg.IsocurveItem(pen=isopen)
        self.isocurve.setParentItem(self.zoom)

        # Viewboxes for slice data
        # Both boxes have mouse disabled - range is fixed so we don't want to
        # scale them accidentally
        # Y box has y inverted to match the main image
        # Y box has x inverted so that zero pixel value is far from the image
        options = {"enableMouse":False, "enableMenu": False}
        self.vb_x = self.g_layout.addViewBox(row=2, col=0, **options)
        self.vb_y = self.g_layout.addViewBox(row=0, col=1, rowspan=2,
            invertX=True, invertY=True, **options)

        # Link the slice axes to the main image so that when we zoom/pan the
        # main image, our slices zoom/pan also
        self.vb_x.setXLink(self.vb_image)
        self.vb_y.setYLink(self.vb_image)

        # Disable autoscaling and fix range to maximum pixel intensity
        self.vb_x.setRange(yRange=(0,255))
        self.vb_y.setRange(xRange=(0,255))
        self.vb_x.disableAutoRange(axis=self.vb_x.YAxis)
        self.vb_y.disableAutoRange(axis=self.vb_y.XAxis)

        # Background color must not be black so that we can see where images
        # start/end
        color = pg.mkColor(40,40,40)
        self.vb_image.setBackgroundColor(color)
        self.vb_zoom.setBackgroundColor(color)
        self.vb_residuals.setBackgroundColor(color)
        self.vb_x.setBackgroundColor(color)
        self.vb_y.setBackgroundColor(color)
        self.g_layout.setBackground(color)

        self.vb_image.addItem(self.image)
        self.vb_image.addItem(self.fit_v_line)
        self.vb_image.addItem(self.fit_h_line)
        self.vb_image.addItem(self.mark_v_line)
        self.vb_image.addItem(self.mark_h_line)
        self.vb_image.addItem(self.history_plot)
        # Figure out how to overlay properly?
        # self.vb_image.addItem(self.x_slice)
        # self.vb_image.addItem(self.x_fit)
        # self.vb_image.addItem(self.y_slice)
        # self.vb_image.addItem(self.y_fit)
        self.vb_zoom.addItem(self.zoom)
        self.vb_zoom.addItem(self.fit_maj_line)
        self.vb_zoom.addItem(self.fit_min_line)
        self.vb_residuals.addItem(self.residuals)
        self.vb_x.addItem(self.x_slice)
        self.vb_x.addItem(self.x_fit)
        self.vb_y.addItem(self.y_slice)
        self.vb_y.addItem(self.y_fit)

        self.vb_image.setRange(QtCore.QRectF(0, 0, 1280, 1024))
        self.vb_zoom.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_residuals.setRange(QtCore.QRectF(0, 0, 50, 50))

        #
        # Size hints below here
        #
        self.g_layout.ci.layout.setColumnStretchFactor(0, 4)
        self.g_layout.ci.layout.setColumnStretchFactor(1, 1)
        self.g_layout.ci.layout.setColumnStretchFactor(2, 2)
        self.g_layout.ci.layout.setRowStretchFactor(0, 2)
        self.g_layout.ci.layout.setRowStretchFactor(1, 2)
        self.g_layout.ci.layout.setRowStretchFactor(2, 1)

        self.vb_x.setMinimumHeight(50)
        self.vb_y.setMinimumWidth(50)
        self.vb_x.setMaximumHeight(100)
        self.vb_y.setMaximumWidth(100)
        self.vb_image.setMinimumSize(640, 512)
        self.vb_zoom.setMinimumSize(320, 320)
        self.vb_residuals.setMinimumSize(320, 320)

        self.g_layout.setMinimumSize(1100,562)

    def add_tooltips(self):
        #TODO
        pass
