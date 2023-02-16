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

    def drawStroke(self, point0, point1=None):
        '''Draw stroke point/line on strokes board, board offset is automatically applied.
        '''
        self.painter.begin(self.strokes_board)
        self.painter.setPen(self.pen)
        point0 = point0 + self.abs_offset
        if point1 is None:
            self.painter.drawPoint(point0)
        else:
            point1 = point1 + self.abs_offset
            self.painter.drawLine(point0, point1)
        self.painter.end()

    def getCameraArray(self):
        '''Return camera image as np.ndarray in RGB format.
        '''
        return cv2.cvtColor(self.camera_array, cv2.COLOR_BGR2RGB)

    def getCameraRect(self):
        '''Return absolute rect indicating valid camera array area that can be visible on canvas. Regardless of camera image.
        '''
        if self.camera_array.size == 0:
            return QRect()
        camera_w, camera_h = self.camera_array.shape[1], self.camera_array.shape[0]
        canvas_aspect_ratio = self.width() / self.height()
        camera_aspect_ratio = camera_w / camera_h
        if canvas_aspect_ratio < camera_aspect_ratio:
            h = camera_h
            w = int(h * canvas_aspect_ratio)
            x = (camera_w - w) // 2
            y = 0
            return QRect(x, y, w, h)
        else:
            w = camera_w
            h = int(w / canvas_aspect_ratio)
            x = 0
            y = (camera_h - h) // 2
            return QRect(x, y, w, h)

    def mousePressEvent(self, e):
        self.pos0 = e.pos()
        self.drawStroke(self.pos0)
        self.update()

    def mouseMoveEvent(self, e):
        self.pos1 = e.pos()
        self.drawStroke(self.pos0, self.pos1)
        self.pos0 = self.pos1
        self.update()

    def resizeEvent(self, e):
        self._updateAbsOffset()

    def timerEvent(self, e):
        self._updateCamera()
        if not self.camera_image.isNull():
            self.update()

    def paintEvent(self, e):
        target_rect = e.rect()
        board_rect = QRect(self.abs_offset, self.size())
        camera_rect = self.getCameraRect()
        self.painter.begin(self)
        # draw background
        self.painter.drawPixmap(target_rect, self.background_board, board_rect)
        # draw camera image
        self.painter.drawImage(target_rect, self.camera_image, camera_rect)
        # draw strokes
        self.painter.drawPixmap(target_rect, self.strokes_board, board_rect)
        self.painter.end()

    def sizeHint(self):
        return self.PREFERRED_SIZE

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
        else:
            self.camera_array = np.empty((0, 0, 3), dtype=np.uint8)

        # prepare camera QImage
        if self.show_camera:
            self.camera_image = QImage(self.camera_array.data, self.camera_array.shape[1], self.camera_array.shape[0], QImage.Format.Format_BGR888)
        else:
            self.camera_image = QImage()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = WCanvas()
    canvas.toggleCamera()
    canvas.show()
    sys.exit(app.exec())
