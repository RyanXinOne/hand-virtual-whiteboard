from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QPen, QPainter, QColorConstants
from PyQt6.QtCore import Qt, QSize, QPoint


class WCanvas(QWidget):
    """Widget canvas.
    """
    BACKGROUND_COLOR = QColorConstants.White
    MIN_SIZE = QSize(300, 250)
    MAX_SIZE = QSize(8192, 8192)
    PREFERRED_SIZE = QSize(600, 500)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.board = QPixmap(self.MAX_SIZE)
        self.board.fill(self.BACKGROUND_COLOR)

        self.pen = QPen(QColorConstants.Black, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.painter = QPainter()

        self.setMinimumSize(self.MIN_SIZE)
        self.setMaximumSize(self.MAX_SIZE)
        self._updateOffset()

    def clear(self):
        self.board.fill(self.BACKGROUND_COLOR)
        self.update()

    def changeColor(self, color):
        self.pen.setColor(color)

    def changeThickness(self, thickness):
        self.pen.setWidth(thickness)

    def toImage(self):
        return self.board.toImage()

    def mousePressEvent(self, e):
        self.pos0 = e.pos() + self.offset

    def mouseMoveEvent(self, e):
        self.pos1 = e.pos() + self.offset
        self.painter.begin(self.board)
        self.painter.setPen(self.pen)

        self.painter.drawLine(self.pos0, self.pos1)

        self.painter.end()
        self.pos0 = self.pos1

        self.update()

    def paintEvent(self, e):
        self.painter.begin(self)
        self.painter.drawPixmap(0, 0, self.board, self.offset.x(), self.offset.y(), self.width(), self.height())
        self.painter.end()

    def resizeEvent(self, e):
        self._updateOffset()

    def sizeHint(self):
        return self.PREFERRED_SIZE

    def _updateOffset(self):
        offset_size = (self.MAX_SIZE - self.size()) / 2
        self.offset = QPoint(offset_size.width(), offset_size.height())


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = WCanvas()
    w.show()
    sys.exit(app.exec())
