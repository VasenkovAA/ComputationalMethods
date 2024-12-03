from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
from ui.input_data_pages import InputPagesController
from ui.plot_page import PlotPage

class MainWindow(QtWidgets.QMainWindow):
    """Основное окно приложения"""

    def __init__(self):
        super(MainWindow, self).__init__()

        # Устанавливаем минимальные размеры для окна
        self.setMinimumHeight(720)  # Минимальная высота окна 720 пикселей
        self.setMinimumWidth(900)  # Минимальная ширина окна 1600 пикселей (если нужно)

        self.resize(720, 1100)  # Размер по умолчанию при старте
        self.setWindowTitle('Calculate Beet')

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName('MainWidget')
        self.setCentralWidget(self.central_widget)

        # Используем вертикальный макет для простоты размещения
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Создаем страницы графика и ввода данных
        self.plot_page = PlotPage()
        self.input_pages_controller = InputPagesController(self.plot_page, self)

        # Добавляем график сверху
        self.layout.addWidget(self.plot_page)

        # Добавляем контроллер вкладок снизу
        self.layout.addWidget(self.input_pages_controller)

        # Создаем меню
        self._create_menubar()

    def _create_menubar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        settings_menu = menubar.addMenu("Настройки")

        save_calculations_action = QtGui.QAction("Сохранить расчеты", self)
        save_calculations_action.triggered.connect(self.save_calculations)
        file_menu.addAction(save_calculations_action)

        load_calculations_action = QtGui.QAction("Загрузить расчеты", self)
        load_calculations_action.triggered.connect(self.load_calculations)
        file_menu.addAction(load_calculations_action)

        interface_settings_action = QtGui.QAction("Настройки интерфейса", self)
        interface_settings_action.triggered.connect(self.show_interface_settings)
        settings_menu.addAction(interface_settings_action)

    def save_calculations(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранить расчеты", "", "JSON (*.json)")
        if file_path:
            print(f"Сохранение расчетов в: {file_path}")

    def load_calculations(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Загрузить расчеты", "", "JSON (*.json)")
        if file_path:
            print(f"Загрузка расчетов из: {file_path}")

    def show_interface_settings(self):
        settings_dialog = QtWidgets.QDialog(self)
        settings_dialog.setWindowTitle("Настройки интерфейса")
        settings_dialog.exec()
