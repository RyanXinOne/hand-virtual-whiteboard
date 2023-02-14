from PyQt6.QtCore import QPoint

from gui.canvas import WCanvas
from detect import DetectEngine


class HCanvas(WCanvas):
    '''Canvas controlled by hand.
    '''

    def __init__(self, parent=None):
        super().__init__(parent)

        self.engine = DetectEngine()
        # dry run to make engine prepared
        self.engine.detect(self.getCameraImageArray())

    def timerEvent(self, e):
        super().timerEvent(e)
        image = self.getCameraImageArray()
        rect = self.getCameraRect()
        detection = self.engine.detect(image)
        if detection is not None:
            x, y, bx1, by1, bx2, by2, conf, cls_ = detection
            # coordinate transform from camera to canvas
            point_x = (x - rect.x()) * self.width() / rect.width()
            point_y = (y - rect.y()) * self.height() / rect.height()
            point = QPoint(int(point_x), int(point_y))
            self.drawStroke(point)


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    canvas = HCanvas()
    canvas.toggleCamera()
    canvas.show()
    sys.exit(app.exec())
