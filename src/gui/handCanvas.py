from math import sqrt
import numpy as np
from PyQt6.QtCore import QPoint, QTimer

from gui.canvas import Canvas
from detect import DetectEngine


class HandCanvas(Canvas):
    '''Canvas controlled by hand.
    '''
    HAND_STROKE_UNIT = 4
    END_STROKE_IN_SEC = 1
    MINIMUM_STROKE_DISTANCE = 6
    POINT_REDUNDANCY = 2  # TODO

    def __init__(self, parent=None):
        super().__init__(parent)

        self.engine = DetectEngine()
        # dry run to make engine prepared
        self.engine.detect(np.empty((1, 1, 3), dtype=np.uint8))

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._endStroke)

    def timerEvent(self, e):
        super().timerEvent(e)
        image = self.getCameraArray()
        if image.size == 0:
            return
        detection = self.engine.detect(image)
        if detection is None:
            return
        x, y, bx1, by1, bx2, by2, conf, cls_ = detection
        # coordinate transform from camera to canvas
        rect = self.getCameraRect()
        point_x = (x - rect.x()) * self.width() / rect.width()
        point_y = (y - rect.y()) * self.height() / rect.height()
        point = QPoint(int(point_x), int(point_y))

        isFirstPoint = not self.timer.isActive()
        self.timer.start(self.END_STROKE_IN_SEC * 1000)
        if isFirstPoint:
            # start hand stroke
            self.hand_points = [point]
        else:
            # continue hand stroke
            if self._pointDistance(self.hand_points[-1], point) >= self.MINIMUM_STROKE_DISTANCE:
                self.hand_points.append(point)
                if len(self.hand_points) >= self.HAND_STROKE_UNIT:
                    self.drawStroke(*self.hand_points)
                    self.hand_points = self.hand_points[-1:]
                    self.update()

    def _endStroke(self):
        # end hand stroke
        # self.drawStroke(*self.hand_points)
        del self.hand_points
        # self.update()

    def _pointDistance(self, p1, p2):
        return sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    canvas = HandCanvas()
    canvas.toggleCamera()
    canvas.show()
    sys.exit(app.exec())
