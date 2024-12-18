from PyQt6 import QtWidgets
from PyQt6 import QtCore
from ui.matrix_page import InputMatrixPage
from ui.experiment_page import InputExperimentPage
from ui.plot_page import PlotPage


class InputPagesController(QtWidgets.QTabWidget):
    """Управление страницами ввода данных"""

    def __init__(self, plot_page: PlotPage, parent: QtWidgets.QMainWindow | None = None):
        super(InputPagesController, self).__init__(parent)
        self.setObjectName('PagesController')

        self.setMinimumSize(500, 150)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed
        )

        _translate = QtCore.QCoreApplication.translate
        self.input_matrix_page = InputMatrixPage(plot_page)
        self.input_experiment_page = InputExperimentPage(plot_page)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.addTab(self.input_experiment_page, '')
        self.addTab(self.input_matrix_page, '')

        self.setTabText(self.indexOf(self.input_experiment_page), _translate("PagesController",
                                                                             "Эксперимент"))
        self.setTabText(self.indexOf(self.input_matrix_page), _translate("PagesController", "Расчёт"))

