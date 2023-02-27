from PyQt6.QtWidgets import QToolBar, QStackedLayout


class StackedToolbar(QToolBar):
    '''Provides a stack of toolbars where only one toolbar is visible at a time.

    Following the public methods of QStackedWidget.
    '''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stack = QStackedLayout()
        self.setLayout(self.stack)

    def addWidget(self, widget):
        self.stack.addWidget(widget)

    def count(self):
        return self.stack.count()

    def currentIndex(self):
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.stack.currentWidget()

    def indexOf(self, widget):
        return self.stack.indexOf(widget)

    def insertWidget(self, index, widget):
        self.stack.insertWidget(index, widget)

    def removeWidget(self, widget):
        self.stack.removeWidget(widget)

    def widget(self, index):
        return self.stack.widget(index)

    def setCurrentIndex(self, index):
        self.stack.setCurrentIndex(index)

    def setCurrentWidget(self, widget):
        self.stack.setCurrentWidget(widget)
