import time
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QImage, QPen, QFont, QColorConstants, QPainter, QPainterPath
from PyQt6.QtCore import Qt, QSize, QPoint, QPointF, QRect


class Canvas(QWidget):
    """Widget canvas.
    """
    MIN_SIZE = QSize(128, 128)
    MAX_SIZE = QSize(8192, 8192)
    PREFERRED_SIZE = QSize(640, 480)
    MOUSE_STROKE_UNIT = 3  # 2->line, 3->quad curve, 4->cubic curve

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(self.MIN_SIZE)
        self.setMaximumSize(self.MAX_SIZE)

        self.resetOffset()

        self.background_board = QPixmap(self.MAX_SIZE)
        self.setBackgroundColor(QColorConstants.White)
        self.strokes_board = QPixmap(self.MAX_SIZE)
        self.clearStrokes()

        self.pen = QPen(QColorConstants.Black, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.eraser = QPen(QColorConstants.Transparent, 25, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.painter = QPainter()
        self.setPaintingMode('draw')
        self.setMouseTool('pen')

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.PREFERRED_SIZE.width())
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.PREFERRED_SIZE.height())

        self.fps = 0
        self.last_frame_time = time.perf_counter()

        self.toggleCamera(False)

        self.startTimer(0)

    def setMouseTool(self, tool):
        '''Set mouse tool among ('pen', 'page').
        '''
        if tool in ('pen', 'page'):
            self.mouse_tool = tool
        else:
            raise ValueError('Invalid mouse tool')

    def setPaintingMode(self, mode):
        '''Set painting mode among ('draw', 'erase').
        '''
        if mode in ('draw', 'erase'):
            self.painting_mode = mode
        else:
            raise ValueError('Invalid drawing mode')

    def setPenColor(self, color):
        self.pen.setColor(color)

    def setPenThickness(self, thickness):
        self.pen.setWidth(thickness)

    def setEraserThickness(self, thickness):
        self.eraser.setWidth(thickness)

    def setBackgroundColor(self, color):
        self.background_board.fill(color)

    def toggleCamera(self, state):
        '''Show/hide camera display on canvas.
        '''
        self.show_camera = state
        self._updateCamera()

    def clearStrokes(self):
        self.strokes_board.fill(QColorConstants.Transparent)

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

    def drawStroke(self, point0, *points):
        '''Draw strokes on board, board offset is automatically applied.

        Type of stroke is depend on number of points given.
        Points num:
            1 - point
            2 - straight line
            3 - quadratic Bezier curve
            4 - cubic Bezier curve
        '''
        point0 = QPointF(point0 + self.abs_offset)
        points = [QPointF(point + self.abs_offset) for point in points]
        proceed_num = len(points)
        self.painter.begin(self.strokes_board)
        if self.painting_mode == 'draw':
            self.painter.setPen(self.pen)
            self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        elif self.painting_mode == 'erase':
            self.painter.setPen(self.eraser)
            self.painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        path = QPainterPath(point0)
        if proceed_num == 0:
            self.painter.drawPoint(point0)
        elif proceed_num == 1:
            path.lineTo(points[0])
        elif proceed_num == 2:
            path.quadTo(points[0], points[1])
        else:
            path.cubicTo(points[0], points[1], points[2])
        self.painter.drawPath(path)
        self.painter.end()

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
            w = round(h * canvas_aspect_ratio)
            x = (camera_w - w) // 2
            y = 0
            return QRect(x, y, w, h)
        else:
            w = camera_w
            h = round(w / canvas_aspect_ratio)
            x = 0
            y = (camera_h - h) // 2
            return QRect(x, y, w, h)

    def getCameraArray(self):
        '''Return camera image as np.ndarray in RGB format.
        '''
        return cv2.cvtColor(self.camera_array, cv2.COLOR_BGR2RGB)

    def setCameraArray(self, array, fromRGB=True):
        '''Set camera_array and camera_image in BGR format.
        '''
        if fromRGB:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        self.camera_array = array

        # prepare camera QImage
        if self.show_camera:
            self.camera_image = QImage(self.camera_array.data, self.camera_array.shape[1], self.camera_array.shape[0], QImage.Format.Format_BGR888)
        else:
            self.camera_image = QImage()

        if not self.camera_image.isNull():
            self.update()

    def mousePressEvent(self, e):
        self.mouse_points = [e.pos()]

    def mouseMoveEvent(self, e):
        self.mouse_points.append(e.pos())
        if self.mouse_tool == 'pen':
            if len(self.mouse_points) >= self.MOUSE_STROKE_UNIT:
                self.drawStroke(*self.mouse_points)
                self.mouse_points = self.mouse_points[-1:]
                self.update()
        elif self.mouse_tool == 'page':
            self.updateOffset(self.mouse_points[-2] - self.mouse_points[-1])
            self.mouse_points = self.mouse_points[-1:]
            self.update()

    def mouseReleaseEvent(self, e):
        if self.mouse_tool == 'pen':
            self.drawStroke(*self.mouse_points)
            self.update()
        elif self.mouse_tool == 'page':
            pass
        del self.mouse_points

    def resizeEvent(self, e):
        self._updateAbsOffset()

    def timerEvent(self, e):
        self.fps = 1 / (time.perf_counter() - self.last_frame_time)
        self.last_frame_time = time.perf_counter()
        self._updateCamera()

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
        # draw fps
        if self.show_camera:
            self.painter.setPen(QPen(QColorConstants.Black))
            self.painter.setFont(QFont('Arial', 11))
            self.painter.drawText(10, 20, f'fps: {self.fps:.1f}')
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
            _, array = self.camera.read()
            array = cv2.flip(array, 1)
        else:
            array = np.empty((0, 0, 3), dtype=np.uint8)

        self.setCameraArray(array, fromRGB=False)


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    canvas = Canvas()
    canvas.toggleCamera(True)
    canvas.show()
    sys.exit(app.exec())
