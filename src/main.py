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
        def cameraSlot():
            canvas.toggleCamera(cameraAct.isChecked())
            canvas.update()
        cameraAct.toggled.connect(cameraSlot)

        clearAct = QAction('Clear', self)
        def clearSlot():
            canvas.clearStrokes()
            canvas.update()
        clearAct.triggered.connect(clearSlot)

        penAct = QAction('Pen', self)
        penAct.setCheckable(True)
        def penSlot():
            if penAct.isChecked():
                canvas.setMouseTool('pen')
                canvas.setPaintingMode('draw')
        penAct.toggled.connect(penSlot)
        penAct.setChecked(True)
        
        eraserAct = QAction('Eraser', self)
        eraserAct.setCheckable(True)
        def eraserSlot():
            if eraserAct.isChecked():
                canvas.setMouseTool('pen')
                canvas.setPaintingMode('erase')
        eraserAct.toggled.connect(eraserSlot)

        pageMoveAct = QAction('Page Move', self)
        pageMoveAct.setCheckable(True)
        def pageMoveSlot():
            if pageMoveAct.isChecked():
                canvas.setMouseTool('page')
        pageMoveAct.toggled.connect(pageMoveSlot)

        cvToolGroup = QActionGroup(self)
        cvToolGroup.setExclusive(True)
        cvToolGroup.addAction(penAct)
        cvToolGroup.addAction(eraserAct)
        cvToolGroup.addAction(pageMoveAct)

        toolbar = self.addToolBar('Toolbar')
        toolbar.addAction(cameraAct)
        toolbar.addAction(clearAct)
        toolbar.addSeparator()
        toolbar.addAction(penAct)
        toolbar.addAction(eraserAct)
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
