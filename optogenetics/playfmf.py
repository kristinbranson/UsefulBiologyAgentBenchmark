"""
playfmf.py

Standalone viewer for UFMF videos. FMF, SBFMF, and AVI files also work, since
frames are read through movies.Movie. A Python/Qt counterpart of JAABA's
playfmf.m.

Usage:
    python playfmf.py [moviefile] [--fps FPS] [--zero-based]

If no movie file is given, a file dialog opens. Keyboard and mouse controls
are listed in CONTROLS_HELP below (also shown by --help and Help > Controls).

Playback is clocked to wall time: at speed s, the frame shown is the one
recorded (elapsed seconds * s) after the frame playback started from. If
reading frames cannot keep up, or the requested frame rate is higher than
MAX_DISPLAY_RATE_HZ, frames are skipped rather than playing slower than
requested.

Side effect: the directory of the last movie opened is remembered between
sessions via QSettings (on Linux, ~/.config/playfmf/playfmf.conf).
"""

import argparse
import os
import sys
import time

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from movies import Movie

# Frame rate (frames/s) assumed when the movie's timestamps do not give a
# usable one and --fps is not set.
DEFAULT_FPS = 30.0
# Playback never redraws faster than this (Hz). Faster than a typical monitor
# refresh would only spend time decoding frames that are never seen.
MAX_DISPLAY_RATE_HZ = 60.0
# Allowed playback speeds, as multiples of the movie's frame rate.
MIN_SPEED = 0.01
MAX_SPEED = 100.0

# Zoom factor per mouse wheel notch or zoom key press.
ZOOM_STEP = 1.25
# Allowed zoom range, in screen pixels per image pixel.
MIN_ZOOM = 0.05
MAX_ZOOM = 64.0
# QWheelEvent.angleDelta() units in one notch of a standard mouse wheel.
WHEEL_DELTA_PER_NOTCH = 120

INITIAL_WINDOW_SIZE = (1000, 1000)  # width, height in screen pixels
VIEW_BACKGROUND_GRAY = 40  # 0-255 gray level around the frame

MOVIE_FILE_FILTER = (
    "Movies (*.ufmf *.fmf *.sbfmf *.avi);;"
    "UFMF (*.ufmf);;FMF (*.fmf);;SBFMF (*.sbfmf);;AVI (*.avi);;All files (*)"
)
SETTINGS_NAME = "playfmf"  # QSettings organization and application name
LAST_DIRECTORY_KEY = "last_directory"

CONTROLS_HELP = """\
Space                 play / pause
Left / Right arrow    previous / next frame (pauses playback)
Frame box             type a frame number, press Enter to jump there
Slider                scrub through the movie
Speed box             playback speed, as a multiple of the recorded frame rate
Mouse wheel           zoom in / out around the cursor
+ / -                 zoom in / out around the center of the view
F                     fit the whole frame in the window
Left-drag             pan when zoomed in
Ctrl+O                open another movie
Ctrl+Q                quit"""


def to_uint8(frame):
    """
    Convert a frame to uint8 for display, scaling like playfmf.m.

    Inputs:
        frame: np.ndarray of any shape. Integer dtypes are scaled so the
            dtype's maximum maps to 255. Float frames with maximum <= 1 are
            scaled by 255; otherwise they are assumed to already be in [0, 255].
    Outputs:
        frame_uint8: np.ndarray, same shape, dtype uint8. The input itself if
            it is already uint8.
    """
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.integer):
        scale = 255.0 / np.iinfo(frame.dtype).max
    else:
        scale = 255.0 if frame.max() <= 1 else 1.0
    return np.clip(frame * scale, 0, 255).astype(np.uint8)


def frame_to_qimage(frame):
    """
    Convert a movie frame to a QImage.

    Inputs:
        frame: np.ndarray, shape (height, width) grayscale, (height, width, 1),
            or (height, width, 3) color in BGR channel order (as OpenCV-based
            readers return). Any dtype accepted by to_uint8.
    Outputs:
        image: QImage of size width x height, Format_Grayscale8 or
            Format_BGR888, owning its own copy of the pixel data.
    """
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]  # (height, width, 1) -> (height, width)
    frame = np.ascontiguousarray(to_uint8(frame))
    height, width = frame.shape[:2]
    if frame.ndim == 2:
        image_format = QImage.Format.Format_Grayscale8
    elif frame.ndim == 3 and frame.shape[2] == 3:
        image_format = QImage.Format.Format_BGR888
    else:
        raise ValueError(f"Cannot display frame of shape {frame.shape}")
    image = QImage(frame.data, width, height, frame.strides[0], image_format)
    # QImage wraps the numpy buffer without copying; copy so the image stays
    # valid after the array is freed.
    return image.copy()


def estimate_frame_rate(timestamps):
    """
    Estimate a movie's frame rate from its per-frame timestamps.

    Inputs:
        timestamps: sequence of floats, length num_frames, in seconds.
    Outputs:
        fps: float, 1 / median inter-frame interval (frames/s), or None if
            there are fewer than two timestamps or the median interval is not
            a positive finite number.
    """
    timestamps = np.asarray(timestamps, dtype=float)
    if timestamps.size < 2:
        return None
    interval = np.median(np.diff(timestamps))
    if not np.isfinite(interval) or interval <= 0:
        return None
    return 1.0 / interval


class VideoView(QGraphicsView):
    """
    Zoomable, pannable display of a single video frame.

    The view starts in fit mode: the whole frame is scaled to fit the window,
    and is refit whenever the window is resized. Zooming leaves fit mode until
    fit_to_window() is called again. Left-drag pans (when zoomed in past the
    window size).

    Signals:
        zoom_changed(float): emitted with the new zoom, in screen pixels per
            image pixel, whenever the zoom changes.
    """

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        """
        Inputs:
            parent: optional parent QWidget.
        """
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.fit_mode = True
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # zoom_by() positions the view itself after scaling.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(VIEW_BACKGROUND_GRAY, VIEW_BACKGROUND_GRAY, VIEW_BACKGROUND_GRAY))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, image):
        """
        Show a new frame. If its size differs from the previous frame's (e.g. a
        new movie), the scene is resized and the view refit to the window.

        Inputs:
            image: QImage to display.
        """
        size_changed = image.size() != self.pixmap_item.pixmap().size()
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        if size_changed:
            self.scene().setSceneRect(self.pixmap_item.boundingRect())
            self.fit_to_window()

    def zoom(self):
        """
        Outputs:
            zoom: float, current screen pixels per image pixel.
        """
        return self.transform().m11()

    def fit_to_window(self):
        """Scale the whole frame to fit the window and enter fit mode."""
        if self.pixmap_item.pixmap().isNull():
            return
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.fit_mode = True
        self._on_zoom_changed()

    def zoom_by(self, factor, anchor=None):
        """
        Multiply the zoom by factor, clamped to [MIN_ZOOM, MAX_ZOOM], and leave
        fit mode.

        Inputs:
            factor: float, > 1 zooms in, < 1 zooms out.
            anchor: QPoint in viewport pixel coordinates, or None for the
                center of the view. The image point at the anchor stays fixed
                on screen, as far as the scroll range allows (when the whole
                image fits in the view, it is centered instead).
        """
        current_zoom = self.zoom()
        new_zoom = float(np.clip(current_zoom * factor, MIN_ZOOM, MAX_ZOOM))
        if new_zoom == current_zoom:
            return
        if anchor is None:
            anchor = self.viewport().rect().center()
        anchor_scene_point = self.mapToScene(anchor)
        relative_scale = new_zoom / current_zoom
        self.scale(relative_scale, relative_scale)
        # Scroll so the anchored image point is back under the anchor.
        shift = self.mapFromScene(anchor_scene_point) - anchor  # viewport pixels
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + shift.x())
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + shift.y())
        self.fit_mode = False
        self._on_zoom_changed()

    def zoom_in(self):
        """Zoom in by ZOOM_STEP around the center of the view."""
        self.zoom_by(ZOOM_STEP)

    def zoom_out(self):
        """Zoom out by ZOOM_STEP around the center of the view."""
        self.zoom_by(1.0 / ZOOM_STEP)

    def _on_zoom_changed(self):
        """
        Pick the pixel interpolation for the new zoom and emit zoom_changed.
        Nearest-neighbor when magnified, so individual pixels stay crisp;
        smooth filtering when shrunk, so the downsampled frame does not alias.
        """
        if self.zoom() < 1:
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        else:
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.zoom_changed.emit(self.zoom())

    def wheelEvent(self, event):
        """Zoom around the cursor by ZOOM_STEP per wheel notch."""
        notches = event.angleDelta().y() / WHEEL_DELTA_PER_NOTCH
        if notches == 0:
            super().wheelEvent(event)
            return
        self.zoom_by(ZOOM_STEP**notches, anchor=event.position().toPoint())
        event.accept()

    def resizeEvent(self, event):
        """Keep the whole frame visible on window resize when in fit mode."""
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_to_window()


class PlayerWindow(QMainWindow):
    """
    Main window: video view, frame slider, play/pause button, frame number
    box, playback speed box, and zoom buttons.

    Frames are indexed from 0 internally (frame_index, matching
    Movie.get_frame); the number shown to the user is frame_index +
    first_frame_number.
    """

    def __init__(self, fps_override=None, first_frame_number=1):
        """
        Inputs:
            fps_override: float or None. Frame rate (frames/s) that speed 1x
                corresponds to. If None, estimated from each movie's timestamps.
            first_frame_number: int, 0 or 1. Number displayed for the first
                frame of the movie.
        """
        super().__init__()
        self.fps_override = fps_override
        self.first_frame_number = first_frame_number

        self.movie = None
        self.movie_path = None
        self.num_frames = 0
        self.frame_index = 0  # 0-based index of the frame on screen
        self.native_fps = DEFAULT_FPS  # frames/s that speed 1x plays at
        self.first_timestamp = 0.0  # seconds; time display is relative to it
        self.speed = 1.0  # multiple of native_fps

        # Playback clock: at time play_start_time (time.perf_counter() seconds)
        # the frame on screen was play_start_index. Reset whenever the frame is
        # changed by hand or the speed changes during playback.
        self.play_start_time = 0.0
        self.play_start_index = 0
        self.play_timer = QTimer(self)
        self.play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.play_timer.timeout.connect(self.advance_playback)

        self.settings = QSettings(SETTINGS_NAME, SETTINGS_NAME)

        self._build_controls()
        self._build_menus()
        self._set_controls_enabled(False)
        self.setWindowTitle("playfmf")
        self.resize(*INITIAL_WINDOW_SIZE)

    # ------------------------------------------------------------------
    # UI construction

    def _build_controls(self):
        """Create the video view, slider, control row, and status bar."""
        self.view = VideoView()
        self.view.zoom_changed.connect(self._update_zoom_label)

        # Buttons and slider never take keyboard focus, so arrow keys and
        # Space always reach the window's shortcuts.
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(self.seek)

        self.play_button = QPushButton()
        self.play_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.play_button.clicked.connect(self.toggle_playback)
        self._update_play_button()

        self.frame_edit = QLineEdit()
        self.frame_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.frame_edit.setToolTip("Current frame. Type a frame number and press Enter to go there.")
        self.frame_edit.editingFinished.connect(self._on_frame_edited)
        self.frame_count_label = QLabel("/ 0")

        self.speed_box = QDoubleSpinBox()
        self.speed_box.setRange(MIN_SPEED, MAX_SPEED)
        self.speed_box.setDecimals(2)
        self.speed_box.setValue(self.speed)
        self.speed_box.setSuffix("x")
        self.speed_box.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        # Only apply the speed on Enter, focus loss, or the step arrows, not on
        # every keystroke while typing.
        self.speed_box.setKeyboardTracking(False)
        self.speed_box.setToolTip("Playback speed, as a multiple of the movie's recorded frame rate")
        self.speed_box.valueChanged.connect(self._on_speed_changed)
        self.speed_box.editingFinished.connect(self.view.setFocus)
        self.playback_fps_label = QLabel()

        zoom_out_button = QToolButton()
        zoom_out_button.setText("−")  # minus sign
        zoom_out_button.setToolTip("Zoom out (-)")
        zoom_out_button.clicked.connect(self.view.zoom_out)
        zoom_in_button = QToolButton()
        zoom_in_button.setText("+")
        zoom_in_button.setToolTip("Zoom in (+)")
        zoom_in_button.clicked.connect(self.view.zoom_in)
        fit_button = QToolButton()
        fit_button.setText("Fit")
        fit_button.setToolTip("Fit frame to window (F)")
        fit_button.clicked.connect(self.view.fit_to_window)
        zoom_buttons = [zoom_out_button, zoom_in_button, fit_button]
        for button in zoom_buttons:
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        control_row = QHBoxLayout()
        control_row.addWidget(self.play_button)
        control_row.addSpacing(12)
        control_row.addWidget(QLabel("Frame"))
        control_row.addWidget(self.frame_edit)
        control_row.addWidget(self.frame_count_label)
        control_row.addSpacing(12)
        control_row.addWidget(QLabel("Speed"))
        control_row.addWidget(self.speed_box)
        control_row.addWidget(self.playback_fps_label)
        control_row.addStretch(1)
        control_row.addWidget(QLabel("Zoom"))
        for button in zoom_buttons:
            control_row.addWidget(button)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.slider)
        layout.addLayout(control_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.time_label = QLabel()
        self.zoom_label = QLabel()
        # Wide enough for the longest zoom text, so the label does not get clipped.
        widest_zoom_text = f"zoom {100 * MAX_ZOOM:.0f}%"
        self.zoom_label.setMinimumWidth(self.zoom_label.fontMetrics().horizontalAdvance(widest_zoom_text))
        self.statusBar().addPermanentWidget(self.time_label)
        self.statusBar().addPermanentWidget(self.zoom_label)

    def _build_menus(self):
        """
        Create the File, View, Playback, and Help menus. Keyboard shortcuts
        are attached to these menu actions.
        """
        file_menu = self.menuBar().addMenu("&File")
        self._add_action(file_menu, "&Open...", self.open_movie_dialog, [QKeySequence.StandardKey.Open])
        self._add_action(file_menu, "&Quit", self.close, [QKeySequence.StandardKey.Quit])

        view_menu = self.menuBar().addMenu("&View")
        self.zoom_in_action = self._add_action(
            view_menu, "Zoom &in", self.view.zoom_in, ["+", "=", QKeySequence.StandardKey.ZoomIn]
        )
        self.zoom_out_action = self._add_action(
            view_menu, "Zoom &out", self.view.zoom_out, ["-", QKeySequence.StandardKey.ZoomOut]
        )
        self.fit_action = self._add_action(view_menu, "&Fit to window", self.view.fit_to_window, ["F"])

        playback_menu = self.menuBar().addMenu("&Playback")
        self.play_action = self._add_action(playback_menu, "&Play / pause", self.toggle_playback, ["Space"])
        self.previous_frame_action = self._add_action(
            playback_menu, "P&revious frame", lambda: self.step(-1), ["Left"]
        )
        self.next_frame_action = self._add_action(playback_menu, "&Next frame", lambda: self.step(1), ["Right"])

        help_menu = self.menuBar().addMenu("&Help")
        self._add_action(help_menu, "&Controls", self.show_controls_help, [])

    def _add_action(self, menu, text, callback, shortcuts):
        """
        Add an action to a menu.

        Inputs:
            menu: QMenu to add to.
            text: str, menu item label.
            callback: callable with no arguments, run when the action triggers.
            shortcuts: list of str or QKeySequence.StandardKey; may be empty.
        Outputs:
            action: the new QAction.
        """
        action = QAction(text, self)
        action.setShortcuts([QKeySequence(shortcut) for shortcut in shortcuts])
        action.triggered.connect(callback)
        menu.addAction(action)
        return action

    def _set_controls_enabled(self, enabled):
        """
        Enable or disable the controls that need an open movie.

        Inputs:
            enabled: bool.
        """
        widgets = [self.slider, self.play_button, self.frame_edit, self.speed_box]
        actions = [
            self.zoom_in_action,
            self.zoom_out_action,
            self.fit_action,
            self.play_action,
            self.previous_frame_action,
            self.next_frame_action,
        ]
        for item in widgets + actions:
            item.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Opening movies

    def open_movie_dialog(self):
        """Ask for a movie file and open it, starting in the last directory used."""
        start_directory = self.settings.value(LAST_DIRECTORY_KEY, os.getcwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open movie", start_directory, MOVIE_FILE_FILTER)
        if path:
            self.open_movie(path)

    def open_movie(self, path):
        """
        Open a movie and show its first frame, replacing any open movie.

        Inputs:
            path: str, path to a .ufmf, .fmf, .sbfmf, or .avi file.
        Outputs:
            success: bool. On failure an error dialog is shown and any
                previously open movie stays open.
        Side effects:
            Stores the movie's directory in QSettings as the default for the
            next file dialog.
        """
        self.stop_playback()
        try:
            movie = Movie(path)
            num_frames = movie.get_n_frames()
            timestamps = movie.get_some_timestamps()
        except Exception as error:
            QMessageBox.critical(self, "Error opening movie", f"Could not open {path}:\n{error}")
            return False
        if num_frames < 1:
            QMessageBox.critical(self, "Error opening movie", f"{path} has no frames.")
            return False

        self._close_movie()
        self.movie = movie
        self.movie_path = path
        self.num_frames = num_frames
        self.first_timestamp = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        if self.fps_override is not None:
            self.native_fps = self.fps_override
        else:
            self.native_fps = estimate_frame_rate(timestamps) or DEFAULT_FPS
        self._update_playback_fps_label()

        self.slider.blockSignals(True)
        self.slider.setRange(0, num_frames - 1)
        self.slider.blockSignals(False)
        last_frame_number = num_frames - 1 + self.first_frame_number
        self.frame_count_label.setText(f"/ {last_frame_number}")
        # Wide enough for the largest frame number plus a little margin.
        digit_width = self.frame_edit.fontMetrics().horizontalAdvance("0")
        self.frame_edit.setFixedWidth(digit_width * (len(str(last_frame_number)) + 3))

        self.setWindowTitle(f"playfmf - {os.path.basename(path)}")
        self.settings.setValue(LAST_DIRECTORY_KEY, os.path.dirname(os.path.abspath(path)))
        self._set_controls_enabled(True)
        self.show_frame(0)
        self.view.fit_to_window()
        self.view.setFocus()
        return True

    def _close_movie(self):
        """Close the open movie file, if any."""
        if self.movie is not None:
            self.movie.close()
            self.movie = None

    # ------------------------------------------------------------------
    # Frame display and navigation

    def show_frame(self, index):
        """
        Read and display one frame, and update the slider, frame box, and
        time readout to match. Does not touch the playback clock.

        Inputs:
            index: int, 0-based frame index, in [0, num_frames).
        Outputs:
            success: bool. On a read error playback stops and the error is
                shown in the status bar.
        """
        try:
            frame, timestamp = self.movie.get_frame(index)
            image = frame_to_qimage(frame)
        except Exception as error:
            self.stop_playback()
            frame_number = index + self.first_frame_number
            self.statusBar().showMessage(f"Could not read frame {frame_number}: {error}")
            return False
        self.frame_index = index
        self.view.set_image(image)

        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)
        # Leave the frame box alone while the user is typing in it.
        if not (self.frame_edit.hasFocus() and self.frame_edit.isModified()):
            self.frame_edit.setText(str(index + self.first_frame_number))
        self.time_label.setText(f"t = {timestamp - self.first_timestamp:.4f} s")
        return True

    def seek(self, index):
        """
        Jump to a frame chosen by the user (slider, frame box, arrow keys).
        If playing, playback continues from the new frame.

        Inputs:
            index: int, 0-based frame index. Clamped to [0, num_frames).
        """
        if self.movie is None:
            return
        index = int(np.clip(index, 0, self.num_frames - 1))
        self.show_frame(index)
        if self.is_playing():
            self._reset_play_clock()

    def step(self, frame_offset):
        """
        Pause, then move by frame_offset frames.

        Inputs:
            frame_offset: int, e.g. -1 for the previous frame, +1 for the next.
        """
        self.stop_playback()
        self.seek(self.frame_index + frame_offset)

    def _on_frame_edited(self):
        """Go to the frame number typed in the frame box, then return focus to the video."""
        if not self.frame_edit.isModified():
            return
        try:
            frame_number = int(self.frame_edit.text().strip())
        except ValueError:
            frame_number = self.frame_index + self.first_frame_number  # restore current frame
        self.frame_edit.setModified(False)
        self.seek(frame_number - self.first_frame_number)
        # seek() clamps, so rewrite the box in case the typed number was out of range.
        self.frame_edit.setText(str(self.frame_index + self.first_frame_number))
        self.view.setFocus()

    # ------------------------------------------------------------------
    # Playback

    def is_playing(self):
        """
        Outputs:
            playing: bool, True while playback is running.
        """
        return self.play_timer.isActive()

    def toggle_playback(self):
        """Play if paused, pause if playing."""
        if self.is_playing():
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        """Start playing from the current frame, or from the first frame if at the end."""
        if self.movie is None:
            return
        if self.frame_index >= self.num_frames - 1:
            self.show_frame(0)
        self._reset_play_clock()
        self.play_timer.start(self._timer_interval_ms())
        self._update_play_button()

    def stop_playback(self):
        """Pause playback. Does nothing if already paused."""
        self.play_timer.stop()
        self._update_play_button()

    def advance_playback(self):
        """
        Timer callback: show the frame that should be on screen now according
        to the playback clock. Stops at the last frame.
        """
        elapsed = time.perf_counter() - self.play_start_time  # seconds
        target_index = self.play_start_index + int(elapsed * self.native_fps * self.speed)
        if target_index >= self.num_frames - 1:
            self.show_frame(self.num_frames - 1)
            self.stop_playback()
        elif target_index != self.frame_index:
            self.show_frame(target_index)

    def _reset_play_clock(self):
        """Restart the playback clock from the frame currently on screen."""
        self.play_start_time = time.perf_counter()
        self.play_start_index = self.frame_index

    def _timer_interval_ms(self):
        """
        Outputs:
            interval: int, ms between playback timer ticks. One frame period at
                the current speed, but no shorter than 1 / MAX_DISPLAY_RATE_HZ.
        """
        frame_period_ms = 1000.0 / (self.native_fps * self.speed)
        min_period_ms = 1000.0 / MAX_DISPLAY_RATE_HZ
        return max(1, round(max(frame_period_ms, min_period_ms)))

    def _on_speed_changed(self, speed):
        """
        Apply a new playback speed, without jumping frames if playing.

        Inputs:
            speed: float, multiple of the movie's frame rate.
        """
        self.speed = speed
        self._update_playback_fps_label()
        if self.is_playing():
            self._reset_play_clock()
            self.play_timer.setInterval(self._timer_interval_ms())

    # ------------------------------------------------------------------
    # Labels and dialogs

    def _update_play_button(self):
        """Show a play or pause icon/label on the play button to match the playback state."""
        if self.is_playing():
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            text = "Pause"
        else:
            icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            text = "Play"
        self.play_button.setIcon(icon)
        self.play_button.setText(text)

    def _update_playback_fps_label(self):
        """Show the frame rate that the current speed corresponds to."""
        self.playback_fps_label.setText(f"= {self.native_fps * self.speed:.4g} fps")

    def _update_zoom_label(self, zoom):
        """
        Inputs:
            zoom: float, screen pixels per image pixel.
        """
        self.zoom_label.setText(f"zoom {100 * zoom:.0f}%")

    def show_controls_help(self):
        """Show the keyboard and mouse controls in a dialog."""
        box = QMessageBox(self)
        box.setWindowTitle("playfmf controls")
        box.setText(f"<pre>{CONTROLS_HELP}</pre>")
        box.exec()

    def closeEvent(self, event):
        """Stop playback and close the movie file when the window closes."""
        self.stop_playback()
        self._close_movie()
        super().closeEvent(event)


def parse_args(argv):
    """
    Parse command line arguments.

    Inputs:
        argv: list of str, arguments after the program name.
    Outputs:
        args: argparse.Namespace with fields moviefile (str or None),
            fps (float or None), zero_based (bool).
    """
    parser = argparse.ArgumentParser(
        description="Play UFMF (and FMF, SBFMF, AVI) movies.",
        epilog="Controls:\n" + CONTROLS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("moviefile", nargs="?", help="Movie to open. If omitted, a file dialog opens.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate (frames/s) for 1x playback speed. Default: estimated from the movie's timestamps.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Number frames starting from 0 (default: from 1, as in playfmf.m).",
    )
    args = parser.parse_args(argv)
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def main(argv=None):
    """
    Run the player.

    Inputs:
        argv: list of str or None. Arguments after the program name; None
            means sys.argv[1:].
    Outputs:
        exit_code: int, from the Qt event loop.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    # Qt gets only the program name; all options are handled by argparse.
    app = QApplication(sys.argv[:1])
    window = PlayerWindow(fps_override=args.fps, first_frame_number=0 if args.zero_based else 1)
    window.show()
    if args.moviefile:
        window.open_movie(args.moviefile)
    else:
        QTimer.singleShot(0, window.open_movie_dialog)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
