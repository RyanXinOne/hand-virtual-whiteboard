import sys
sys.path.append('./src')

from PyQt6.QtWidgets import QApplication
from main import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mwindow = MainWindow()
    sys.exit(app.exec())
