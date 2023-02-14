from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QAction

from gui.hcanvas import HCanvas


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        canvas = HCanvas(self)
        self.setCentralWidget(canvas)

        toolbar = self.addToolBar('Toolbar')

        cameraAct = QAction('Toggle Camera', self)
        cameraAct.triggered.connect(canvas.toggleCamera)
        toolbar.addAction(cameraAct)

        clearAct = QAction('Clear', self)
        clearAct.triggered.connect(canvas.clear)
        toolbar.addAction(clearAct)

        self.setWindowTitle('Hand Virtual Whiteboard')
        self.show()


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
