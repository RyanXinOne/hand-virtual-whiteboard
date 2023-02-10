import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QImage, QPen, QPainter, QColorConstants
from PyQt6.QtCore import Qt, QSize, QPoint, QRect


class WCanvas(QWidget):
    """Widget canvas.
    """
    BACKGROUND_COLOR = QColorConstants.White
    MIN_SIZE = QSize(128, 128)
    MAX_SIZE = QSize(8192, 8192)
    PREFERRED_SIZE = QSize(640, 480)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(self.MIN_SIZE)
        self.setMaximumSize(self.MAX_SIZE)
        self.resetOffset()

        self.background_board = QPixmap(self.MAX_SIZE)
        self.background_board.fill(self.BACKGROUND_COLOR)
        self.strokes_board = QPixmap(self.MAX_SIZE)
        self.strokes_board.fill(QColorConstants.Transparent)

        self.pen = QPen(QColorConstants.Black, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.painter = QPainter()

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.PREFERRED_SIZE.width())
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.PREFERRED_SIZE.height())

        self.show_camera = False
        self._updateCamera()

        self.startTimer(0)

    def clear(self):
        self.strokes_board.fill(QColorConstants.Transparent)
        self.update()

    def changeColor(self, color):
        self.pen.setColor(color)

    def changeThickness(self, thickness):
        self.pen.setWidth(thickness)

    def toImage(self):
        pass

    def toggleCamera(self):
        '''Show/hide camera display on canvas.
        '''
        self.show_camera = not self.show_camera
        self._updateCamera()
        self.update()

    def updateOffset(self, point):
        '''Update user offset by the amount of point.
        '''
        self.user_offset += point
        self._updateAbsOffset()

    def resetOffset(self):
        '''Reset user offset to (0, 0).
        '''
        self.user_offset = QPoint(0, 0)
        self._updateAbsOffset()

    def mousePressEvent(self, e):
        self.pos0 = e.pos() + self.abs_offset
        self._drawStroke(self.pos0)
        self.update()

    def mouseMoveEvent(self, e):
        self.pos1 = e.pos() + self.abs_offset
        self._drawStroke(self.pos0, self.pos1)
        self.pos0 = self.pos1
        self.update()

    def resizeEvent(self, e):
        self._updateAbsOffset()

    def paintEvent(self, e):
        target_rect = e.rect()
        board_rect = QRect(self.abs_offset, self.size())
        camera_rect = self._getValidCameraRect()
        self.painter.begin(self)
        # draw background
        self.painter.drawPixmap(target_rect, self.background_board, board_rect)
        # draw camera image
        self.painter.drawImage(target_rect, self.camera_image, camera_rect)
        # draw strokes
        self.painter.drawPixmap(target_rect, self.strokes_board, board_rect)
        self.painter.end()

    def timerEvent(self, e):
        self._updateCamera()
        if not self.camera_image.isNull():
            self.update()

    def sizeHint(self):
        return self.PREFERRED_SIZE

    def _drawStroke(self, point0, point1=None):
        '''Draw stroke point/line on strokes board.
        '''
        self.painter.begin(self.strokes_board)
        self.painter.setPen(self.pen)
        if point1 is None:
            self.painter.drawPoint(point0)
        else:
            self.painter.drawLine(point0, point1)
        self.painter.end()

    def _updateAbsOffset(self):
        '''Update absolute offset by base offset and user offset.
        '''
        base_offset_size = self.MAX_SIZE / 2 - self.size() / 2
        self.abs_offset = QPoint(base_offset_size.width(), base_offset_size.height()) + self.user_offset

    def _updateCamera(self):
        '''Read image from camera caption. Camera image is stored in camera_array as np.ndarray. Camera image is converted to QImage and stored in camera_image.
        '''
        # prepare camera ndarray
        if self.camera.isOpened():
            _, self.camera_array = self.camera.read()
            self.camera_array = cv2.flip(self.camera_array, 1)
            self.camera_array = cv2.cvtColor(self.camera_array, cv2.COLOR_BGR2RGB)
        else:
            self.camera_array = np.empty((0, 0, 3), dtype=np.uint8)

        # prepare camera QImage
        if self.show_camera:
            self.camera_image = QImage(self.camera_array.data, self.camera_array.shape[1], self.camera_array.shape[0], QImage.Format.Format_RGB888)
        else:
            self.camera_image = QImage()

    def _getValidCameraRect(self):
        '''Return absolute rect indicating valid camera image area that is visible on canvas.
        '''
        if self.camera_image.isNull():
            return QRect()
        canvas_aspect_ratio = self.width() / self.height()
        camera_aspect_ratio = self.camera_image.width() / self.camera_image.height()
        if canvas_aspect_ratio < camera_aspect_ratio:
            h = self.camera_image.height()
            w = int(h * canvas_aspect_ratio)
            x = (self.camera_image.width() - w) // 2
            y = 0
            return QRect(x, y, w, h)
        else:
            w = self.camera_image.width()
            h = int(w / canvas_aspect_ratio)
            x = 0
            y = (self.camera_image.height() - h) // 2
            return QRect(x, y, w, h)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = WCanvas()
    w.show()
    sys.exit(app.exec())
