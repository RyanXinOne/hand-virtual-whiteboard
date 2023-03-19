import time
from math import sqrt
import numpy as np
from PyQt6.QtCore import QTimer, pyqtSignal, QPoint, QPointF, QRectF
from PyQt6.QtSvg import QSvgRenderer

from gui.canvas import Canvas
from detect import DetectEngine


class GesturePointBuffer:
    '''A buffer storing detected gesture points of the same class.
    '''
    RESPONSE_DELAY = 0
    MIN_DISTANCE = 12

    def __init__(self):
        self.class_ = ''
        self.buffer = []

    def add(self, class_, point, save_point=True):
        '''Add a point to the buffer. If a new class, the buffer is cleared first. Filter out point by MIN_DISTANCE.

        Return:
            True if switched to a new class.
        '''
        if class_ != self.class_:
            self.class_ = class_
            self.buffer.clear()
            if save_point:
                self.buffer.append(point)
            return True
        else:
            if save_point and (not self.buffer or self._pointDistance(self.buffer[-1], point) >= self.MIN_DISTANCE):
                self.buffer.append(point)
            return False

    def getClass(self):
        return self.class_

    def nextPoints(self, num=1, filter=lambda x: x):
        '''Get next n points from buffer if available. The first n - 1 points are removed from the buffer.

        Filter function is applied to each point before returning.
        '''
        if len(self.buffer) >= self.RESPONSE_DELAY + num:
            points = self.buffer[:num]
            self.buffer = self.buffer[num-1:]
            return list(map(filter, points))
        else:
            return None

    def clear(self):
        self.class_ = ''
        self.buffer.clear()

    def _pointDistance(self, p1, p2):
        return sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)


class HandCanvas(Canvas):
    '''Canvas controlled by hand.
    '''
    HAND_STROKE_UNIT = 3
    END_STROKE_IN_SEC = 1.5
    CLEAR_CANVAS_IN_SEC = 3
    CURSOR_SIZE = 20

    # define gesture signal
    onGesture = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_buffer = GesturePointBuffer()
        self.ges_class = ''
        self.ges_point = QPointF()

        self.engine = DetectEngine()
        # dry run to make engine prepared
        self.engine.detect(np.empty((1, 1, 3), dtype=np.uint8))

        def endGestureTimeoutSlot():
            self.ges_class = ''
            self.point_buffer.clear()
            self.update()

        self.noGestureTimer = QTimer(self)
        self.noGestureTimer.setSingleShot(True)
        self.noGestureTimer.timeout.connect(endGestureTimeoutSlot)

        self.clearIniTimestamp = 0

        self.penCursorRenderer = QSvgRenderer('assets/pen.svg')
        self.eraserCursorRenderer = QSvgRenderer('assets/eraser.svg')
        self.pageCursorRenderer = QSvgRenderer('assets/hand.svg')
        self.clearCursorRenderer = QSvgRenderer('assets/clear.svg')

    def timerEvent(self, e):
        super().timerEvent(e)
        image = self.getCameraArray()
        if image.size == 0:
            return
        detections = self.engine.detect(image)
        if not detections:
            return

        if self.show_camera:
            for detection in detections:
                image = self.engine.drawDetection(image, detection)
                self.setCameraArray(image)

        x, y, bx1, by1, bx2, by2, conf, cls_n = detections[0]
        if cls_n not in ('fist', 'one', 'two_up', 'stop', 'dislike'):
            return

        self.noGestureTimer.start(int(self.END_STROKE_IN_SEC * 1000))

        if x > -1:
            self.ges_point = QPointF(x, y)
        else:
            self.ges_point = QPointF((bx1 + bx2) / 2, (by1 + by2) / 2)

        is_new_class = self.point_buffer.add(cls_n, self.ges_point, save_point=(x > -1))
        self.ges_class = self.point_buffer.getClass()
        if is_new_class:
            self.onGesture.emit(self.ges_class)

        if self.ges_class == 'dislike':
            # clear canvas
            timeDiff = time.time() - self.clearIniTimestamp
            if timeDiff >= self.CLEAR_CANVAS_IN_SEC:
                if timeDiff < self.CLEAR_CANVAS_IN_SEC + 0.5:
                    self.clearStrokes()
                self.clearIniTimestamp = time.time()
        else:
            self.clearIniTimestamp = 0
        if self.ges_class == 'one' or self.ges_class == 'two_up':
            # pen stroke
            points = self.point_buffer.nextPoints(self.HAND_STROKE_UNIT, filter=self._cameraToGeoPos)
            if points is not None:
                self.drawStroke(*points)
        elif self.ges_class == 'stop':
            # page drag
            points = self.point_buffer.nextPoints(2, filter=self._cameraToGeoPos)
            if points is not None:
                self.updateOffset(points[0] - points[1])
        elif self.ges_class == 'fist':
            # end the previous gesture
            pass

        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        # draw hand cursor
        geo_point = self._cameraToGeoPos(self.ges_point)
        self.painter.begin(self)
        if self.ges_class == 'one':
            self.penCursorRenderer.render(self.painter, QRectF(
                geo_point.x() - 2,
                geo_point.y() - self.CURSOR_SIZE + 2,
                self.CURSOR_SIZE,
                self.CURSOR_SIZE))
        elif self.ges_class == 'two_up':
            self.eraserCursorRenderer.render(self.painter, QRectF(
                geo_point.x() - 3,
                geo_point.y() - self.CURSOR_SIZE + 3,
                self.CURSOR_SIZE,
                self.CURSOR_SIZE))
        elif self.ges_class == 'stop':
            self.pageCursorRenderer.render(self.painter, QRectF(
                geo_point.x() - self.CURSOR_SIZE / 2,
                geo_point.y(),
                self.CURSOR_SIZE,
                self.CURSOR_SIZE))
        elif self.ges_class == 'dislike':
            self.clearCursorRenderer.render(self.painter, QRectF(
                geo_point.x() - self.CURSOR_SIZE / 2,
                geo_point.y() - self.CURSOR_SIZE / 2,
                self.CURSOR_SIZE,
                self.CURSOR_SIZE))
        self.painter.end()

    def _cameraToGeoPos(self, point):
        '''Transform point from camera coordinate to geometry coordinate.
        '''
        if point.isNull():
            return QPoint()

        rect = self.getCameraRect()
        geo_point_x = (point.x() - rect.x()) * self.width() / rect.width()
        geo_point_y = (point.y() - rect.y()) * self.height() / rect.height()
        return QPoint(round(geo_point_x), round(geo_point_y))


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    canvas = HandCanvas()
    canvas.toggleCamera(True)
    canvas.show()
    sys.exit(app.exec())
