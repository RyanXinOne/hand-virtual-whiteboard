from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QAction

from gui.canvas import WCanvas


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        toolbar = self.addToolBar('Toolbar')

        cameraAct = QAction('Toggle Camera', self)
        cameraAct.triggered.connect(self.close)
        toolbar.addAction(cameraAct)

        canvas = WCanvas(self)
        self.setCentralWidget(canvas)

        self.setWindowTitle('Hand Virtual Whiteboard')
        self.show()


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
