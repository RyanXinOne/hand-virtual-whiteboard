from PyQt6.QtWidgets import QToolBar, QStackedWidget


class StackedToolbar(QToolBar):
    '''Provides a stack of toolbars where only one toolbar is visible at a time.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)
        self.addWidget(self.stack)
        self.orientationChanged.connect(self._onOrientationChanged)

    def addToolbar(self, toolbar):
        toolbar.setOrientation(self.orientation())
        return self.stack.addWidget(toolbar)

    def switchToolbar(self, index):
        self.stack.setCurrentIndex(index)

    def _onOrientationChanged(self, orientation):
        for i in range(self.stack.count()):
            self.stack.widget(i).setOrientation(orientation)
