import sys
from PyQt5.QtWidgets import QApplication
from controller5 import MainController

if __name__ == "__main__":
    app = QApplication(sys.argv)
    controller = MainController()
    controller.show()
    sys.exit(app.exec())
    