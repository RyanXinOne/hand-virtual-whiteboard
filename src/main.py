from PyQt6.QtWidgets import QMainWindow, QToolBar
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon
from PyQt6.QtCore import Qt

from gui.handCanvas import HandCanvas
from gui.stackedToolbar import StackedToolbar


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.canvas = HandCanvas(self)

        self._setupAuxToolbar()
        self._setupMainToolbar()

        self.setCentralWidget(self.canvas)

        self.setWindowTitle('Hand Virtual Whiteboard')
        self.show()

    def _setupMainToolbar(self):
        '''Main toolbar lies on top.
        '''
        def cameraSlot():
            self.canvas.toggleCamera(cameraAct.isChecked())
            self.canvas.update()

        def clearSlot():
            self.canvas.clearStrokes()
            self.canvas.update()

        def penSlot():
            if self.penAct.isChecked():
                self.canvas.setMouseTool('pen')
                self.canvas.setPaintingMode('draw')
                self.auxToolbar.switchToolbar(self.auxToolbarIds['pen'])

        def eraserSlot():
            if self.eraserAct.isChecked():
                self.canvas.setMouseTool('pen')
                self.canvas.setPaintingMode('erase')
                self.auxToolbar.switchToolbar(self.auxToolbarIds['eraser'])

        def pageMoveSlot():
            if self.pageMoveAct.isChecked():
                self.canvas.setMouseTool('page')
                self.auxToolbar.switchToolbar(self.auxToolbarIds['empty'])

        cameraAct = QAction(QIcon('assets/camera.svg'), 'Toggle Camera', self)
        cameraAct.setCheckable(True)
        cameraAct.toggled.connect(cameraSlot)

        clearAct = QAction(QIcon('assets/clear.svg'), 'Clear', self)
        clearAct.triggered.connect(clearSlot)

        self.penAct = QAction(QIcon('assets/pen.svg'), 'Pen', self)
        self.penAct.setCheckable(True)
        self.penAct.toggled.connect(penSlot)

        self.eraserAct = QAction(QIcon('assets/eraser.svg'), 'Eraser', self)
        self.eraserAct.setCheckable(True)
        self.eraserAct.toggled.connect(eraserSlot)

        self.pageMoveAct = QAction(QIcon('assets/hand.svg'), 'Pan around', self)
        self.pageMoveAct.setCheckable(True)
        self.pageMoveAct.toggled.connect(pageMoveSlot)

        toolExcGroup = QActionGroup(self)
        toolExcGroup.setExclusive(True)
        toolExcGroup.addAction(self.penAct)
        toolExcGroup.addAction(self.eraserAct)
        toolExcGroup.addAction(self.pageMoveAct)
        self.penAct.setChecked(True)

        mainToolbar = QToolBar('Main Toolbar', self)
        mainToolbar.addAction(cameraAct)
        mainToolbar.addAction(clearAct)
        mainToolbar.addSeparator()
        mainToolbar.addAction(self.penAct)
        mainToolbar.addAction(self.eraserAct)
        mainToolbar.addAction(self.pageMoveAct)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, mainToolbar)

    def _setupAuxToolbar(self):
        '''Setup auxiliary toolbar lies on the left.
        '''
        def blackSlot():
            if blackAct.isChecked():
                self.canvas.setPenColor(QColor('#212121'))

        def redSlot():
            if redAct.isChecked():
                self.canvas.setPenColor(QColor('#e53935'))

        def yellowSlot():
            if yellowAct.isChecked():
                self.canvas.setPenColor(QColor('#fdd835'))

        def blueSlot():
            if blueAct.isChecked():
                self.canvas.setPenColor(QColor('#1e88e5'))

        def greenSlot():
            if greenAct.isChecked():
                self.canvas.setPenColor(QColor('#43a047'))

        blackAct = QAction(QIcon('assets/black.svg'), 'Black', self)
        blackAct.setCheckable(True)
        blackAct.toggled.connect(blackSlot)

        redAct = QAction(QIcon('assets/red.svg'), 'Red', self)
        redAct.setCheckable(True)
        redAct.toggled.connect(redSlot)

        yellowAct = QAction(QIcon('assets/yellow.svg'), 'Yellow', self)
        yellowAct.setCheckable(True)
        yellowAct.toggled.connect(yellowSlot)

        blueAct = QAction(QIcon('assets/blue.svg'), 'Blue', self)
        blueAct.setCheckable(True)
        blueAct.toggled.connect(blueSlot)

        greenAct = QAction(QIcon('assets/green.svg'), 'Green', self)
        greenAct.setCheckable(True)
        greenAct.toggled.connect(greenSlot)

        colorExcGroup = QActionGroup(self)
        colorExcGroup.setExclusive(True)
        colorExcGroup.addAction(blackAct)
        colorExcGroup.addAction(redAct)
        colorExcGroup.addAction(yellowAct)
        colorExcGroup.addAction(blueAct)
        colorExcGroup.addAction(greenAct)
        blackAct.setChecked(True)

        penToolbar = QToolBar(self)
        penToolbar.addAction(blackAct)
        penToolbar.addAction(redAct)
        penToolbar.addAction(yellowAct)
        penToolbar.addAction(blueAct)
        penToolbar.addAction(greenAct)

        eraserToolbar = QToolBar(self)

        emptyToolbar = QToolBar(self)

        self.auxToolbar = StackedToolbar('Auxiliary Toolbar', self)
        self.auxToolbarIds = {}
        self.auxToolbarIds['pen'] = self.auxToolbar.addToolbar(penToolbar)
        self.auxToolbarIds['eraser'] = self.auxToolbar.addToolbar(eraserToolbar)
        self.auxToolbarIds['empty'] = self.auxToolbar.addToolbar(emptyToolbar)

        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.auxToolbar)

    def activatePen(self):
        if not self.penAct.isChecked():
            self.penAct.setChecked(True)

    def activateEraser(self):
        if not self.eraserAct.isChecked():
            self.eraserAct.setChecked(True)

    def activatePageMove(self):
        if not self.pageMoveAct.isChecked():
            self.pageMoveAct.setChecked(True)


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
