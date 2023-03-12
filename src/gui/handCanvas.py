from math import sqrt
import numpy as np
from PyQt6.QtCore import QPoint, QTimer, pyqtSignal

from gui.canvas import Canvas
from detect import DetectEngine


class HandPointBuffer:
    '''A buffer storing detected hand points of the same class.
    '''
    RESPONSE_DELAY = 0
    MIN_DISTANCE = 6

    def __init__(self):
        self.class_ = ''
        self.buffer = []

    def add(self, class_, point):
        '''Add a point to the buffer. If a new class, the buffer is cleared first. Filter out point by MIN_DISTANCE.

        Return:
            True if switched to a new class.
        '''
        if class_ != self.class_:
            self.class_ = class_
            self.buffer.clear()
            self.buffer.append(point)
            return True
        else:
            if self._pointDistance(self.buffer[-1], point) >= self.MIN_DISTANCE:
                self.buffer.append(point)
            return False

    def getClass(self):
        return self.class_

    def nextPoints(self, num=1):
        '''Get next n points from buffer if available. The first n - 1 points are removed from the buffer.
        '''
        if len(self.buffer) >= self.RESPONSE_DELAY + num:
            points = self.buffer[:num]
            self.buffer = self.buffer[num-1:]
            return points
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
    HAND_STROKE_UNIT = 4
    END_STROKE_IN_SEC = 1

    # define gesture signal
    onGesture = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.point_buffer = HandPointBuffer()

        self.engine = DetectEngine()
        # dry run to make engine prepared
        self.engine.detect(np.empty((1, 1, 3), dtype=np.uint8))

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.point_buffer.clear)

    def timerEvent(self, e):
        super().timerEvent(e)
        image = self.getCameraArray()
        if image.size == 0:
            return
        detection = self.engine.detect(image)
        if detection is None:
            return

        if self.show_camera:
            image = self.engine.drawDetection(image, detection)
            self.setCameraArray(image)

        x, y, bx1, by1, bx2, by2, conf, cls_n = detection
        if cls_n not in ('one', 'two_up', 'stop'):
            return

        if x > -1:
            # fingertip coordinate transform from camera to canvas
            rect = self.getCameraRect()
            point_x = (x - rect.x()) * self.width() / rect.width()
            point_y = (y - rect.y()) * self.height() / rect.height()
            point = QPoint(round(point_x), round(point_y))

            is_new_class = self.point_buffer.add(cls_n, point)
            self.timer.start(self.END_STROKE_IN_SEC * 1000)

            ges_class = self.point_buffer.getClass()
            if is_new_class:
                self.onGesture.emit(ges_class)
        else:
            self.timer.stop()
            self.point_buffer.clear()
            ges_class = cls_n

        if ges_class in ('one', 'two_up'):
            # pen stroke
            points = self.point_buffer.nextPoints(self.HAND_STROKE_UNIT)
            if points is not None:
                self.drawStroke(*points)
                self.update()
        elif ges_class == 'stop':
            # page move
            points = self.point_buffer.nextPoints(2)
            if points is not None:
                self.updateOffset(points[0] - points[1])
                self.update()


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    canvas = HandCanvas()
    canvas.toggleCamera(True)
    canvas.show()
    sys.exit(app.exec())
