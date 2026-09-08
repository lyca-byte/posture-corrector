# import sys
# from PyQt5.QtWidgets import QApplication
# from controller5 import MainController

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     controller = MainController()
#     controller.show()
#     sys.exit(app.exec())
    


print("========== MAIN START ==========")

import mediapipe as mp

print("MAIN: MediaPipe OK:", mp.__version__)

from PyQt5.QtWidgets import QApplication

print("MAIN: PyQt5 OK")

from controller5 import MainController

print("MAIN: controller5 OK")


if __name__ == "__main__":
    print("MAIN: Creating QApplication")

    app = QApplication([])

    print("MAIN: Creating MainController")

    controller = MainController()

    print("MAIN: Showing window")

    controller.show()

    print("MAIN: Starting event loop")

    app.exec_()