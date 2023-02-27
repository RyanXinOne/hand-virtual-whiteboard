from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QAction, QActionGroup

from gui.handCanvas import HandCanvas


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        canvas = HandCanvas(self)
        self.setCentralWidget(canvas)

        # ===== Setup Toolbar =====

        cameraAct = QAction('Toggle Camera', self)
        cameraAct.setCheckable(True)
        cameraAct.toggled.connect(lambda: canvas.toggleCamera(cameraAct.isChecked()))

        clearAct = QAction('Clear', self)
        clearAct.triggered.connect(canvas.clear)

        penAct = QAction('Pen', self)
        penAct.setCheckable(True)
        penAct.toggled.connect(lambda: canvas.setMouseTool('pen') if penAct.isChecked() else None)
        penAct.setChecked(True)
        pageMoveAct = QAction('Page Move', self)
        pageMoveAct.setCheckable(True)
        pageMoveAct.toggled.connect(lambda: canvas.setMouseTool('page') if pageMoveAct.isChecked() else None)

        cvToolGroup = QActionGroup(self)
        cvToolGroup.setExclusive(True)
        cvToolGroup.addAction(penAct)
        cvToolGroup.addAction(pageMoveAct)

        toolbar = self.addToolBar('Toolbar')
        toolbar.addAction(cameraAct)
        toolbar.addAction(clearAct)
        toolbar.addSeparator()
        toolbar.addAction(penAct)
        toolbar.addAction(pageMoveAct)

        # ===== Setup Window =====

        self.setWindowTitle('Hand Virtual Whiteboard')
        self.show()


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
