from PyQt6.QtWidgets import QMainWindow, QToolBar, QLabel
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QFont
from PyQt6.QtCore import Qt, QSize

from gui.handCanvas import HandCanvas
from gui.stackedToolbar import StackedToolbar


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hand Virtual Whiteboard')
        self.setWindowIcon(QIcon('assets/icon.svg'))
        loadingLabel = QLabel('Loading...', self)
        loadingLabel.setFont(QFont('Arial', 13))
        loadingLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(loadingLabel)
        self.show()

        def gestureSlot(ges_class):
            if ges_class == 'one':
                # activate pen
                if not self.penAct.isChecked():
                    self.penAct.toggle()
            elif ges_class == 'two_up':
                # activate eraser
                if not self.eraserAct.isChecked():
                    self.eraserAct.toggle()
            elif ges_class == 'stop':
                # activate page drag
                if not self.pageDragAct.isChecked():
                    self.pageDragAct.toggle()
            elif ges_class == 'dislike':
                # perform clear
                self.clearAct.trigger()
            elif ges_class == 'ok':
                # perform save
                self.saveAct.trigger()

        self.canvas = HandCanvas(self)
        self.canvas.onGesture.connect(gestureSlot)

        self._setupAuxToolbar()
        self._setupMainToolbar()

        self.setCentralWidget(self.canvas)

    def _setupMainToolbar(self):
        '''Main toolbar lies on top.
        '''
        def cameraSlot():
            self.canvas.toggleCamera(cameraAct.isChecked())
            self.canvas.update()

        def saveSlot():
            self.canvas.saveCanvas()

        def clearSlot():
            self.canvas.clearCanvas()
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

        def pageDragSlot():
            if self.pageDragAct.isChecked():
                self.canvas.setMouseTool('page')
                self.auxToolbar.switchToolbar(self.auxToolbarIds['empty'])

        cameraAct = QAction(QIcon('assets/camera.svg'), 'Toggle Camera', self)
        cameraAct.setCheckable(True)
        cameraAct.toggled.connect(cameraSlot)

        self.saveAct = QAction(QIcon('assets/download.svg'), 'Export', self)
        self.saveAct.triggered.connect(saveSlot)

        self.clearAct = QAction(QIcon('assets/clear.svg'), 'Clear', self)
        self.clearAct.triggered.connect(clearSlot)

        self.penAct = QAction(QIcon('assets/pen.svg'), 'Pen', self)
        self.penAct.setCheckable(True)
        self.penAct.toggled.connect(penSlot)

        self.eraserAct = QAction(QIcon('assets/eraser.svg'), 'Eraser', self)
        self.eraserAct.setCheckable(True)
        self.eraserAct.toggled.connect(eraserSlot)

        self.pageDragAct = QAction(QIcon('assets/hand.svg'), 'Pan around', self)
        self.pageDragAct.setCheckable(True)
        self.pageDragAct.toggled.connect(pageDragSlot)

        toolExcGroup = QActionGroup(self)
        toolExcGroup.setExclusive(True)
        toolExcGroup.addAction(self.penAct)
        toolExcGroup.addAction(self.eraserAct)
        toolExcGroup.addAction(self.pageDragAct)
        self.penAct.setChecked(True)

        mainToolbar = QToolBar('Main Toolbar', self)
        mainToolbar.setAllowedAreas(Qt.ToolBarArea.TopToolBarArea | Qt.ToolBarArea.BottomToolBarArea)
        mainToolbar.setIconSize(QSize(24, 24))
        mainToolbar.addAction(cameraAct)
        mainToolbar.addAction(self.saveAct)
        mainToolbar.addAction(self.clearAct)
        mainToolbar.addSeparator()
        mainToolbar.addAction(self.penAct)
        mainToolbar.addAction(self.eraserAct)
        mainToolbar.addAction(self.pageDragAct)

        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, mainToolbar)

    def _setupAuxToolbar(self):
        '''Setup auxiliary toolbar lies on the left.
        '''
        penToolbar = self._setupAuxPenToolbar()
        eraserToolbar = self._setupAusEraserToolbar()
        emptyToolbar = QToolBar(self)

        self.auxToolbar = StackedToolbar('Auxiliary Toolbar', self)
        self.auxToolbar.setAllowedAreas(Qt.ToolBarArea.LeftToolBarArea | Qt.ToolBarArea.RightToolBarArea)
        self.auxToolbar.setIconSize(QSize(28, 28))
        self.auxToolbar.setSpacing(5)
        self.auxToolbarIds = {
            'pen': self.auxToolbar.addToolbar(penToolbar),
            'eraser': self.auxToolbar.addToolbar(eraserToolbar),
            'empty': self.auxToolbar.addToolbar(emptyToolbar)
        }

        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.auxToolbar)

    def _setupAuxPenToolbar(self):
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

        def thinSlot():
            if thinAct.isChecked():
                self.canvas.setPenThickness(5)

        def mediumSlot():
            if mediumAct.isChecked():
                self.canvas.setPenThickness(15)

        def thickSlot():
            if thickAct.isChecked():
                self.canvas.setPenThickness(30)

        blackAct = QAction(QIcon('assets/color/black.svg'), 'Black', self)
        blackAct.setCheckable(True)
        blackAct.toggled.connect(blackSlot)

        redAct = QAction(QIcon('assets/color/red.svg'), 'Red', self)
        redAct.setCheckable(True)
        redAct.toggled.connect(redSlot)

        yellowAct = QAction(QIcon('assets/color/yellow.svg'), 'Yellow', self)
        yellowAct.setCheckable(True)
        yellowAct.toggled.connect(yellowSlot)

        blueAct = QAction(QIcon('assets/color/blue.svg'), 'Blue', self)
        blueAct.setCheckable(True)
        blueAct.toggled.connect(blueSlot)

        greenAct = QAction(QIcon('assets/color/green.svg'), 'Green', self)
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

        thinAct = QAction(QIcon('assets/thickness/thin.svg'), 'Thin', self)
        thinAct.setCheckable(True)
        thinAct.toggled.connect(thinSlot)

        mediumAct = QAction(QIcon('assets/thickness/medium.svg'), 'Medium', self)
        mediumAct.setCheckable(True)
        mediumAct.toggled.connect(mediumSlot)

        thickAct = QAction(QIcon('assets/thickness/thick.svg'), 'Thick', self)
        thickAct.setCheckable(True)
        thickAct.toggled.connect(thickSlot)

        thicknessExcGroup = QActionGroup(self)
        thicknessExcGroup.setExclusive(True)
        thicknessExcGroup.addAction(thinAct)
        thicknessExcGroup.addAction(mediumAct)
        thicknessExcGroup.addAction(thickAct)
        thinAct.setChecked(True)

        penToolbar = QToolBar(self)
        penToolbar.addAction(blackAct)
        penToolbar.addAction(redAct)
        penToolbar.addAction(yellowAct)
        penToolbar.addAction(blueAct)
        penToolbar.addAction(greenAct)
        penToolbar.addSeparator()
        penToolbar.addAction(thinAct)
        penToolbar.addAction(mediumAct)
        penToolbar.addAction(thickAct)

        return penToolbar

    def _setupAusEraserToolbar(self):
        def thinSlot():
            if thinAct.isChecked():
                self.canvas.setEraserThickness(10)

        def mediumSlot():
            if mediumAct.isChecked():
                self.canvas.setEraserThickness(25)

        def thickSlot():
            if thickAct.isChecked():
                self.canvas.setEraserThickness(50)

        thinAct = QAction(QIcon('assets/thickness/thin.svg'), 'Thin', self)
        thinAct.setCheckable(True)
        thinAct.toggled.connect(thinSlot)

        mediumAct = QAction(QIcon('assets/thickness/medium.svg'), 'Medium', self)
        mediumAct.setCheckable(True)
        mediumAct.toggled.connect(mediumSlot)

        thickAct = QAction(QIcon('assets/thickness/thick.svg'), 'Thick', self)
        thickAct.setCheckable(True)
        thickAct.toggled.connect(thickSlot)

        thicknessExcGroup = QActionGroup(self)
        thicknessExcGroup.setExclusive(True)
        thicknessExcGroup.addAction(thinAct)
        thicknessExcGroup.addAction(mediumAct)
        thicknessExcGroup.addAction(thickAct)
        mediumAct.setChecked(True)

        eraserToolbar = QToolBar(self)
        eraserToolbar.addAction(thinAct)
        eraserToolbar.addAction(mediumAct)
        eraserToolbar.addAction(thickAct)

        return eraserToolbar

    def sizeHint(self):
        return QSize(800, 600)


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
