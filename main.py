import sys
from PyQt6 import QtWidgets
from PyQt6 import QtGui
from ui.main_window import MainWindow

if __name__ == '__main__':
    try:
        from ctypes import windll
        appid = 'Calculate Beet'
        windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)
    except ImportError:
        pass
    app = QtWidgets.QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
