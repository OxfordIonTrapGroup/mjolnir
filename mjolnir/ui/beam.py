import pyqtgraph as pg
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore
import collections
import pickle
import time
import os

from mjolnir.ui.worker import Worker
from mjolnir.tools import tools


class BeamDisplay(QtWidgets.QMainWindow):
    new_image = QtCore.pyqtSignal()

    def __init__(self, camera):
        super().__init__()

        self.cam = camera
        
        # Pixel width in microns (get from camera if not tsi)
        if self.cam.is_tsi_cam:
            self._px_width = self.cam.get_pixel_width()
        else:
            self._px_width = 5.2 # change this value depending on camera

        # Deques discard the oldest value when full
        self.imageq = collections.deque(maxlen=3)
        self.updateq = collections.deque(maxlen=3)
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
        self._centroid = None

        self._residual_levels = [-2,2]

        self._history_timer = QtCore.QTimer(self)
        self._history_timer.timeout.connect(self.age_history)
        self._history_timer.setInterval(150)
        self._history_timer.start()

        self._up = None

        self.mark_widgets = []

        self.init_ui()
        self.show()

    @QtCore.pyqtSlot(np.ndarray)
    def queue_image(self, im):
        """Queues an image for fitting and plotting"""
        self.imageq.append(im)
        self.new_image.emit()

    @QtCore.pyqtSlot()
    def update(self):
        def finish():
            if not self.cam._is_single_or_stop:
                self.exp.setText("{:.2f} ms".format(self._exp))
                self._last_update, self._fps = tools.update_rate(
                    self._last_update, self._fps)
                self.fps.setText("{:.2f} fps".format(self._fps))
                self.exp.show()
                self.fps.show()
                # fps adjustment within the acquisition thread in camera.py
                if self.cam.is_tsi_cam:
                    if self._fps < self._frame_rate_control:
                        self.cam.fps_adjustment -= 0.01
                    elif self._fps > self._frame_rate_control:
                        self.cam.fps_adjustment += 0.01
            else:
                self.single_acq.setEnabled(True)
        try:
            up = self.updateq.popleft()
        except IndexError:
            return

        self._exp, self._exp_min, self._exp_max, self._exp_inc = self.cam.get_exposure_params()

        up.update({'exposure': self._exp})
        up.update({'px_width': self._px_width})        
        up.update({'residual_LUT': self.residual_LUT})
        up.update({'residual_levels': self._residual_levels})

        self._up = up

        options = {'autoRange': False, 'autoLevels': False}
        self.image.setImage(up['im'], **options)

        try:
            self.zoom.setImage(up['im_crop'], **options)
        except KeyError:
            pass

        failure = up.get('failure', None)
        if failure:
            self.message.setText(failure)
            self.message.show()
            finish()
            return
        else:
            self.message.hide()

        try:
            self.residuals.setImage(up['im_res'], lut=self.residual_LUT, **options)

            self.x_slice.setData(up['x'], up['x_slice'])
            self.x_fit.setData(up['x'], up['x_fit'])

            self.y_slice.setData(up['y_slice'], up['y'])
            self.y_fit.setData(up['y_fit'], up['y'])

            # Sub-pixel position works with QPointF
            centroid = QtCore.QPointF(*up['pxc'])
            self.fit_v_line.setPos(centroid)
            self.fit_h_line.setPos(centroid)

            # cache the centroid in case we need to set a mark
            self._centroid = centroid

            self.history.append(up['pxc'])
            self.replot_history()
            self._history_timer.start()

            # 'zoom_centre' is NOT a QPointF
            zoom_centre = QtCore.QPointF(*up['zoom_centre'])
            self.fit_maj_line.setPos(zoom_centre)
            self.fit_min_line.setPos(zoom_centre)
            self.fit_maj_line.setAngle(up['semimaj_angle'])
            self.fit_min_line.setAngle(up['semimin_angle'])

            self.isocurve.setLevel(up['iso_level'])
            self.isocurve.setData(up['im_fit'])

            self.maj_radius.setText(self.px_string(up['semimaj']))
            self.min_radius.setText(self.px_string(up['semimin']))
            self.avg_radius.setText(self.px_string(up['avg_radius']))
            self.x_radius.setText(self.px_string(up['x_radius']))
            self.y_radius.setText(self.px_string(up['y_radius']))
            self.x_centroid.setText(self.px_string(up['pxc'][0]))
            self.y_centroid.setText(self.px_string(up['pxc'][1]))
            self.ellipticity.setText("{:.3f}".format(up['e']))
            
        except KeyError:
            return

        if self._mark is not None:
            self.update_deltas()

        finish()

    def px_string(self, px):
        return "{:.1f}μm ({:.1f}px)".format(px*self._px_width, px)

    def px_to_um(self, px):
        return self._px_width * px

    def replot_history(self):
        """Update history plot with currently stored points"""
        nopen = pg.mkPen(style=QtCore.Qt.NoPen)
        self.history_plot.setData(
            pos=self.history,
            pen=nopen,
            brush=self.history_brushes[-len(self.history):])

    @QtCore.pyqtSlot()
    def age_history(self):
        try:
            _ = self.history.popleft()
        except IndexError:
            return
        if not self.history:
            return
        self.replot_history()

    def enable_acq_buttons(self, start, single, stop):
        self.start_acq.setEnabled(start)
        self.single_acq.setEnabled(single)
        self.stop_acq.setEnabled(stop)

    def single_acq_cb(self):
        self.cam.single_acquisition()
        self.enable_acq_buttons(True, False, False)
        self.status.setText("Single")
        self.hide_exp_fps()
        if self.cam.is_tsi_cam:
            self.frame_rate.setEnabled(True)
        else:
            self.pixel_clock.setEnabled(True)

    def start_acq_cb(self):
        self.cam.start_acquisition()
        self.enable_acq_buttons(False, False, True)
        self.status.setText("Started")
        if self.cam.is_tsi_cam:
            self.frame_rate.setEnabled(False)
        else:
            self.pixel_clock.setEnabled(False)

    def stop_acq_cb(self):
        self.cam.stop_acquisition()
        self.enable_acq_buttons(True, True, False)
        self.hide_exp_fps()
        if self.cam.is_tsi_cam:
            self.frame_rate.setEnabled(True)
        else:
            self.pixel_clock.setEnabled(True)

    def hide_exp_fps(self):
        self.exp.hide()
        self.fps.hide()

    def get_exposure_params(self):
        val, min_, max_, step = self.cam.get_exposure_params()
        self.exposure.setRange(min_, max_)
        self.exposure.setSingleStep(step)
        self.exposure.setValue(val)

    def exposure_cb(self):
        exp = self.exposure.value()
        self.cam.set_exposure_ms(exp)
        real_exp = self.cam.get_exposure_params()[0]
        self.exposure.setValue(real_exp)

    def get_frame_rate_params(self):
        val, min_, max_, step = self.cam.get_frame_rate_params()
        self.frame_rate.setRange(min_,max_)
        self.frame_rate.setSingleStep(step)
        self.frame_rate.setValue(val)
        self._frame_rate_control = val

    def frame_rate_cb(self):
        fps = self.frame_rate.value()
        self.cam.set_frame_rate(fps)
        self._frame_rate_control = fps

    def get_pixel_clock_params(self):
        val, min_, max_, step = self.cam.get_pixel_clock_params()
        self.pixel_clock.setRange(min_,max_)
        self.pixel_clock.setSingleStep(step)
        self.pixel_clock.setValue(val)
        self._pixel_clock_val = val

    def pixel_clock_cb(self):
        clock_val = self.pixel_clock.value()
        self.cam.set_pixel_clock(clock_val)
        self.get_exposure_params()
        self._pixel_clock_val = clock_val

    def save_cb(self):
        '''Saves the current frame. The function pickles the self._up dictionary.
        Any ongoing acquisition is halted.
        Raises an error if there is no data available in the current frame.'''
        
        try:
            self.stop_acq_cb()
            self.status.setText("Stopped: Save Frame")
            
            # Crude test for save error
            error_test = self._up['semimaj']
            
            name, _filters = QtGui.QFileDialog.getSaveFileName(self, "Save Frame", "untitled",
                                                               "Pickle (*.pickle)")
            with open(name, 'wb') as f:
                pickle.dump(self._up, f, pickle.HIGHEST_PROTOCOL)
        
        except (KeyError, TypeError):
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Critical)
            msg.setText("Error: No data to save.")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec()

    def load_cb(self):
        '''Loads a saved frame. The function unpickles a dictionary that was previously pickled via save_cb.
        Any ongoing acquisition is halted.
        Once the dictionary is unpickled, the function assigns the proper variables to display the frame.
        The assignment code is copied from update().
        Raises an error if the frame does not load properly.'''

        try:
            self.stop_acq_cb()
            self.status.setText("Stopped: Load Frame")
            
            name, _filters = QtGui.QFileDialog.getOpenFileName(self, "Load Frame", "",
                                                               "Pickle (*.pickle)")
            fname = os.path.basename(name)
            
            with open(name, 'rb') as f:
                d = pickle.load(f)

            # Crude test for load error
            error_test = d['semimaj']

            up = d
            self._up = up

            def finish():
                self._exp, self._exp_min, self._exp_max, self._exp_inc = self.cam.get_exposure_params()
                self.exp.setText("{:.2f} ms".format(self._exp))
                self._last_update, self._fps = tools.update_rate(
                    self._last_update, self._fps)
                self.fps.setText("{:.2f} fps".format(self._fps))
                self.exp.show()
                self.fps.show()

            options = {'autoRange': False, 'autoLevels': False}
            self.image.setImage(up['im'], **options)

            try:
                self.zoom.setImage(up['im_crop'], **options)
            except KeyError:
                pass

            failure = up.get('failure', None)
            if failure:
                self.message.setText(failure)
                self.message.show()
                finish()
                return
            else:
                self.message.hide()

            try:
                self.residuals.setImage(up['im_res'], lut=self.residual_LUT, **options)

                self.x_slice.setData(up['x'], up['x_slice'])
                self.x_fit.setData(up['x'], up['x_fit'])

                self.y_slice.setData(up['y_slice'], up['y'])
                self.y_fit.setData(up['y_fit'], up['y'])

                # Sub-pixel position works with QPointF
                centroid = QtCore.QPointF(*up['pxc'])
                self.fit_v_line.setPos(centroid)
                self.fit_h_line.setPos(centroid)

                # cache the centroid in case we need to set a mark
                self._centroid = centroid

                self.history.append(up['pxc'])
                self.replot_history()
                self._history_timer.start()

                # 'zoom_centre' is NOT a QPointF
                zoom_centre = QtCore.QPointF(*up['zoom_centre'])
                self.fit_maj_line.setPos(zoom_centre)
                self.fit_min_line.setPos(zoom_centre)
                self.fit_maj_line.setAngle(up['semimaj_angle'])
                self.fit_min_line.setAngle(up['semimin_angle'])

                self.isocurve.setLevel(up['iso_level'])
                self.isocurve.setData(up['im_fit'])

                self.maj_radius.setText(self.px_string(up['semimaj']))
                self.min_radius.setText(self.px_string(up['semimin']))
                self.avg_radius.setText(self.px_string(up['avg_radius']))
                self.x_radius.setText(self.px_string(up['x_radius']))
                self.y_radius.setText(self.px_string(up['y_radius']))
                self.x_centroid.setText(self.px_string(up['pxc'][0]))
                self.y_centroid.setText(self.px_string(up['pxc'][1]))
                self.ellipticity.setText("{:.3f}".format(up['e']))

                saved_exp = up['exposure']
                
            except KeyError:
                return

            if self._mark is not None:
                self.update_deltas()

            self.status.setText("Displaying saved frame \""+ fname + "\" (exposure = {:.2f} ms)".format(saved_exp))
            self.fps.hide()
            
        except:
            msg = QtGui.QMessageBox()
            msg.setIcon(QtGui.QMessageBox.Critical)
            msg.setText("Error: Unable to load frame. File incompatible.")
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec()
    
    def reset_view_cb(self):
        self.vb_image.autoRange()
        self.vb_zoom.autoRange()
        self.vb_residuals.autoRange()
        
    def aoi_cb(self):
        pass

    def mark_cb(self):
        self.new_mark(mark=None)

    def new_mark(self, mark=None):
        if mark is None:
            if self._centroid is not None:
                self._mark = self._centroid
            else:
                self._mark = QtCore.QPointF(0, 0)
        else:
            self._mark = mark
        self.mark_v_line.setPos(self._mark)
        self.mark_h_line.setPos(self._mark)

        self.mark_x.setText(self.px_string(self._mark.x()))
        self.mark_y.setText(self.px_string(self._mark.y()))

        self.update_deltas()

        for w in self.mark_widgets:
            w.show()

    def unmark_cb(self):
        self._mark = None

        for w in self.mark_widgets:
            w.hide()

    def update_deltas(self):
        if self._mark is None or self._centroid is None:
            return
        delta = self._centroid - self._mark
        self.x_delta.setText(self.px_string(delta.x()))
        self.y_delta.setText(self.px_string(delta.y()))
        self.beam_delta.setPos(self._centroid)
        self.beam_delta.setText("Δ = ({:.1f}, {:.1f}) μm".format(
            self.px_to_um(delta.x()),
            self.px_to_um(delta.y())))

    def is_within_image_meta(self, viewbox, imageitem, scene_pos):
        if viewbox.sceneBoundingRect().contains(scene_pos):
            pos = viewbox.mapSceneToView(scene_pos)
            upper = np.array(imageitem.image.shape)
            lower = np.zeros(2)
            pos_tup = np.array((pos.x(), pos.y()))

            # check whether the point is within the image
            if np.all(np.logical_and(lower <= pos_tup, pos_tup <= upper)):
                return True
        return False

    def is_within_image(self, scene_pos):
        return self.is_within_image_meta(self.vb_image, self.image, scene_pos)

    def is_within_zoom(self, scene_pos):
        return self.is_within_image_meta(self.vb_zoom, self.zoom, scene_pos)

    def is_within_residuals(self, scene_pos):
        return self.is_within_image_meta(
            self.vb_residuals, self.residuals, scene_pos)

    @QtCore.pyqtSlot(tuple)
    def cursor_cb(self, scene_pos):
        """Called when the cursor is within the graphics view"""
        if self.is_within_image(scene_pos):
            pos = self.vb_image.mapSceneToView(scene_pos)

            self.cursor_v.setPos(pos)
            self.cursor_h.setPos(pos)
            self.cursor_text.setText(
                "({:.1f}, {:.1f}) px".format(pos.x(), pos.y()))
            if self._mark is not None:
                delta = pos - self._mark
                self.cursor_delta.setPos(pos)
                self.cursor_delta.setText(
                    "Δ = ({:.1f}, {:.1f}) μm".format(
                        self.px_to_um(delta.x()), self.px_to_um(delta.y())))

            self.cursor_v.show()
            self.cursor_h.show()
            self.cursor_text.show()
            self.cursor_delta.show()

        elif self.is_within_zoom(scene_pos):
            pos = self.vb_zoom.mapSceneToView(scene_pos)

            if self._up is not None:
                self.zoom_text.setPos(pos)
                self.zoom_text.setText("I = {:.0f}".format(
                    self.zoom.image[int(pos.x()), int(pos.y())]))
                self.zoom_text.show()

        elif self.is_within_residuals(scene_pos):
            pos = self.vb_residuals.mapSceneToView(scene_pos)

            if self._up is not None:
                self.residuals_text.setPos(pos)
                self.residuals_text.setText("r = {:.2f}".format(
                    self.residuals.image[int(pos.x()),int(pos.y())]))
                self.residuals_text.show()

        else:
            for w in [self.cursor_v, self.cursor_h,
                    self.cursor_text, self.cursor_delta,
                    self.zoom_text, self.residuals_text]:
                w.hide()

    @QtCore.pyqtSlot(tuple)
    def clicked_cb(self, evt):
        scene_pos = evt.scenePos()
        if (evt.button() == 1 and not evt.double()
                and self.is_within_image(scene_pos)):
            pos = self.vb_image.mapSceneToView(scene_pos)
            self.cursor_delta.setText("Δ = (0.0, 0.0) μm")
            self.new_mark(pos)

    def get_color_map(self):
        # Bipolar color map
        colors = np.array([
            (0, 255, 255, 255),
            (0, 0, 255, 255),
            (0, 0, 0, 255),
            (255, 0, 0, 255),
            (255, 255, 0, 255),
        ], dtype=np.uint8)
        positions = [0.25 * (2 + i) for i in range(-2,3)]
        return pg.ColorMap(positions, colors)

    def init_ui(self):
        self.widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout()

        self.init_graphics()
        self.init_info_pane()
        self.add_tooltips()
        self.layout_graphics()
        self.layout_info_pane()
        self.connect_actions()

        self.layout.addWidget(self.g_layout, stretch=2)
        self.layout.addWidget(self.info_pane)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)
        self.setGeometry(300, 300, 1500, 600)
        self.setWindowTitle("mjolnir")

    def connect_actions(self):
        """Connect triggers to their actions"""
        self.single_acq.clicked.connect(lambda: self.single_acq_cb())
        self.start_acq.clicked.connect(lambda: self.start_acq_cb())
        self.stop_acq.clicked.connect(lambda: self.stop_acq_cb())
        self.stop_acq.clicked.connect(lambda: self.status.setText("Stopped"))
        self.reset_view.clicked.connect(lambda: self.reset_view_cb())
        # connect after finding params so we don't send accidental update
        self.exposure.valueChanged.connect(self.exposure_cb)
        if self.cam.is_tsi_cam:
            self.frame_rate.valueChanged.connect(self.frame_rate_cb)
        else:
            self.pixel_clock.valueChanged.connect(self.pixel_clock_cb)
        self.save.clicked.connect(self.save_cb)
        self.load.clicked.connect(self.load_cb)
        self.mark.clicked.connect(self.mark_cb)
        self.unmark.clicked.connect(self.unmark_cb)

        proxy = pg.SignalProxy(self.g_layout.scene().sigMouseMoved,
            rateLimit=20, slot=self.cursor_cb)
        self.g_layout.scene().sigMouseMoved.connect(self.cursor_cb)
        self.g_layout.scene().sigMouseClicked.connect(self.clicked_cb)

    def init_info_pane(self):
        """Initialise the info pane's permanent widgets"""
        self.single_acq = QtGui.QPushButton("Single Acquisition")
        self.start_acq = QtGui.QPushButton("Start Acquisition")
        self.stop_acq = QtGui.QPushButton("Stop Acquisition")
        
        self.exposure_label = QtGui.QLabel("Exposure:")
        self.exposure = QtGui.QDoubleSpinBox()
        self.exposure.setSuffix(" ms")
        self.get_exposure_params()

        if self.cam.is_tsi_cam:
            self.frame_rate_label = QtGui.QLabel("Target frame rate:")
            self.frame_rate = QtGui.QDoubleSpinBox()
            self.frame_rate.setSuffix(" fps")
            self.get_frame_rate_params()
        else:
            self.pixel_clock_label = QtGui.QLabel("Pixel clock:")
            self.pixel_clock = QtGui.QDoubleSpinBox()
            self.pixel_clock.setSuffix(" MHz")
            self.get_pixel_clock_params()

        self.reset_view = QtGui.QPushButton("Reset View")
        self.save = QtGui.QPushButton("Save Frame")
        self.load = QtGui.QPushButton("Load Frame")

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

        # Mark location
        self.mark_x = QtGui.QLabel()
        self.mark_y = QtGui.QLabel()

        # Beam distance from marked location
        self.x_delta = QtGui.QLabel()
        self.y_delta = QtGui.QLabel()

        # Keep a list of mark sub-widgets so we can hide/show them
        # Obviously we don't want to hide the mark buttons themselves
        self.mark_widgets.extend([
            self.mark_x, self.mark_y,
            # self.x_delta, self.y_delta,
        ])

        self.exp = QtGui.QLabel()
        self.fps = QtGui.QLabel()
        self.message = QtGui.QLabel()
        self.status = QtGui.QLabel("Stopped")
        self.enable_acq_buttons(True, True, False)

    def init_graphics(self):
        """Initialise the important graphics items"""
        m, n = 1280, 1024
        self.image = pg.ImageItem(np.zeros((m,n)))
        self.zoom = pg.ImageItem(np.zeros((50,50)))
        self.residuals = pg.ImageItem(np.zeros((50,50)))
        self.residuals.setLevels(self._residual_levels)
        self.x_fit = pg.PlotDataItem(np.zeros(m), pen={'width':2})
        self.x_slice = pg.PlotDataItem(np.zeros(m), pen=None, symbol='o', pxMode=True, symbolSize=4)
        self.y_fit = pg.PlotDataItem(np.zeros(n), pen={'width':2})
        self.y_slice = pg.PlotDataItem(np.zeros(n), pen=None, symbol='o', pxMode=True, symbolSize=4)

        # Only the residuals have any sort of false color - initialise the
        # lookup table and the legend
        cmap = self.get_color_map()
        self.residual_LUT = cmap.getLookupTable(nPts=256)
        self.res_legend = pg.GradientLegend(size=(10,255), offset=(0,20))
        self.res_legend.setGradient(cmap.getGradient())
        n_ticks = 5
        self.res_legend.setLabels({"{}".format(level):val
            for (level, val) in zip(
                np.linspace(*self._residual_levels, n_ticks),
                np.linspace(0, 1, n_ticks))})

        ypen = pg.mkPen(color=(255,255,0,85), width=3)

        # Centroid position markers in main image, aligned with x,y
        self.fit_v_line = pg.InfiniteLine(pos=1, angle=90, pen=ypen)
        self.fit_h_line = pg.InfiniteLine(pos=1, angle=0, pen=ypen)

        # Plot fading recent position markers
        n_history = 5
        self.history = collections.deque(maxlen=n_history)
        self.history_plot = pg.ScatterPlotItem()
        self.history_brushes = [pg.mkBrush(
            color=(255,255,0,int((i+1)*255/n_history)))
            for i in range(n_history)]

        # User marked position
        rpen = pg.mkPen(color=(255,0,0,127), width=3, style=QtCore.Qt.DotLine)
        self.mark_v_line = pg.InfiniteLine(pos=1, angle=90, pen=rpen)
        self.mark_h_line = pg.InfiniteLine(pos=1, angle=0, pen=rpen)
        self.mark_widgets.extend([
            self.mark_v_line, self.mark_h_line,
        ])

        # Mouse cursor
        wpen = pg.mkPen(color=(255,255,255,63), width=3)
        red = pg.mkColor(255,0,0,223)
        yellow = pg.mkColor(255,255,0,223)
        self.cursor_v = pg.InfiniteLine(pos=1, angle=90, pen=wpen)
        self.cursor_h = pg.InfiniteLine(pos=1, angle=0, pen=wpen)
        self.cursor_text = pg.TextItem()
        self.cursor_delta = pg.TextItem(anchor=(-0.1, -0.1), color=red)
        self.beam_delta = pg.TextItem(anchor=(-0.1, -0.1), color=yellow)
        self.zoom_text = pg.TextItem(anchor=(-0.1, -0.1), color=yellow)
        self.residuals_text = pg.TextItem(anchor=(-0.1, -0.1))
        self.mark_widgets.append(self.cursor_delta)
        self.mark_widgets.append(self.beam_delta)

        # Centroid position markers in zoomed image, aligned with beam
        # ellipse axes
        zoom_centre = QtCore.QPointF(25,25)
        self.fit_maj_line = pg.InfiniteLine(pos=zoom_centre, angle=90, pen=ypen)
        self.fit_min_line = pg.InfiniteLine(pos=zoom_centre, angle=0, pen=ypen)

        # Shows 1/e^2 ellipse of beam
        isopen = pg.mkPen(color=(255,255,0,85), width=3, style=QtCore.Qt.DotLine)
        self.isocurve = pg.IsocurveItem(pen=isopen)
        self.isocurve.setParentItem(self.zoom)

    def layout_info_pane(self):
        """Add info pane widgets to their layout"""
        self.param_layout = QtWidgets.QFormLayout()
        self.param_layout.addRow(QtGui.QLabel("<b>Beam Parameters</b>"))
        self.param_layout.addRow(QtGui.QLabel("<i>(all radii are 1/e<sup>2</sup>)</i>"))
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

        mark_x_label = QtGui.QLabel("Mark X:")
        mark_y_label = QtGui.QLabel("Mark Y:")
        dx_label = QtGui.QLabel("ΔX:")
        dy_label = QtGui.QLabel("ΔY:")
        self.mark_widgets.extend([
            mark_x_label, mark_y_label,
            # dx_label, dy_label,
        ])
        self.param_layout.addRow(self.reset_view)
        self.param_layout.addRow(self.mark, self.unmark)
        self.param_layout.addRow(mark_x_label, self.mark_x)
        self.param_layout.addRow(mark_y_label, self.mark_y)
        # self.param_layout.addRow(dx_label, self.x_delta)
        # self.param_layout.addRow(dy_label, self.y_delta)
        for w in self.mark_widgets:
            w.hide()

        self.param_widget = QtGui.QWidget()
        self.param_widget.setLayout(self.param_layout)

        self.exp_fps_layout = QtWidgets.QFormLayout()
        self.exp_fps_layout.addRow(self.exposure_label, self.exposure)
        if self.cam.is_tsi_cam:
            self.exp_fps_layout.addRow(self.frame_rate_label, self.frame_rate)
        else:
            self.exp_fps_layout.addRow(self.pixel_clock_label, self.pixel_clock)
        self.exp_fps_layout.addRow("Measured exposure:", self.exp)
        self.exp_fps_layout.addRow("Measured frame rate:", self.fps)
        
        self.exp_fps_widget = QtGui.QWidget()
        self.exp_fps_widget.setLayout(self.exp_fps_layout)

        self.info_pane_layout = QtWidgets.QVBoxLayout()
        self.info_pane_layout.setAlignment(QtCore.Qt.AlignTop)
        self.info_pane_layout.addWidget(self.start_acq)
        self.info_pane_layout.addWidget(self.single_acq)
        self.info_pane_layout.addWidget(self.stop_acq)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.save)
        self.info_pane_layout.addWidget(self.load)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.exp_fps_widget)
        self.info_pane_layout.addStretch(1)
        self.info_pane_layout.addWidget(self.param_widget)
        self.info_pane_layout.addStretch(3)
        self.info_pane_layout.addWidget(self.message)
        self.info_pane_layout.addWidget(self.status)

        self.info_pane = QtWidgets.QWidget(self)
        self.info_pane.setLayout(self.info_pane_layout)

    def layout_graphics(self):
        """Put graphics items in the layout"""
        # Graphics layout object to place viewboxes in
        self.g_layout = pg.GraphicsLayoutWidget(border=(80, 80, 80))
        self.g_layout.setCursor(QtCore.Qt.CrossCursor)

        # Viewboxes for images
        # aspect locked so that pixels are square
        # y inverted so that (0,0) is top left as in Thorlabs software
        options = {"lockAspect":True, "invertY":True}
        self.vb_image = self.g_layout.addViewBox(row=0, col=0, rowspan=2, **options)
        self.vb_zoom = self.g_layout.addViewBox(row=0, col=2, **options)
        self.vb_residuals = self.g_layout.addViewBox(row=1, col=2, **options)

        # Link zoom and residual views
        self.vb_zoom.setXLink(self.vb_residuals)
        self.vb_zoom.setYLink(self.vb_residuals)

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
        # self.vb_image.addItem(self.cursor_text)
        self.vb_image.addItem(self.cursor_delta)
        self.vb_image.addItem(self.beam_delta)
        self.vb_image.addItem(self.history_plot)
        # Figure out how to overlay properly?
        # self.vb_image.addItem(self.x_slice)
        # self.vb_image.addItem(self.x_fit)
        # self.vb_image.addItem(self.y_slice)
        # self.vb_image.addItem(self.y_fit)
        self.vb_zoom.addItem(self.zoom)
        self.vb_zoom.addItem(self.fit_maj_line)
        self.vb_zoom.addItem(self.fit_min_line)
        self.vb_zoom.addItem(self.zoom_text)
        self.vb_residuals.addItem(self.residuals)
        self.vb_residuals.addItem(self.residuals_text)
        self.vb_x.addItem(self.x_slice)
        self.vb_x.addItem(self.x_fit)
        self.vb_x.addItem(self.cursor_v)
        self.vb_y.addItem(self.y_slice)
        self.vb_y.addItem(self.y_fit)
        self.vb_y.addItem(self.cursor_h)

        self.res_legend.setParentItem(self.vb_residuals)
        self.cursor_text.setParentItem(self.vb_image)

        self.vb_image.setRange(QtCore.QRectF(0, 0, 1280, 1024))
        self.vb_zoom.setRange(QtCore.QRectF(0, 0, 50, 50))
        self.vb_residuals.setRange(QtCore.QRectF(0, 0, 50, 50))

        # Fit to viewboxes on first acquisition
        self.vb_zoom.enableAutoRange()
        self.vb_residuals.enableAutoRange()

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
        self.mark.setToolTip("Place a mark at the current beam position")
        self.unmark.setToolTip("Remove the mark")
