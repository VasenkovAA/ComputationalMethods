import numpy as np
from PyQt6 import QtWidgets
from PyQt6 import QtCore
from PyQt6 import QtGui
from ui.plot_page import PlotPage
from core.algorithms import Algorithms
from ui.input_feild_validation import Int_1_1000_Validator, Double_0_100000_Validator


class InputMatrixPage (QtWidgets.QWidget):
    """Страница ручного ввода данных"""

    def __init__(self, plot_page: PlotPage):
        super(InputMatrixPage, self).__init__()
        self.plot_page = plot_page
        self.setObjectName("InputMatrixPage")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        self.gridLayout_3 = QtWidgets.QGridLayout(self)
        self.gridLayout_3.setHorizontalSpacing(30)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setVerticalSpacing(7)
        self.gridLayout_5.setObjectName("gridLayout_5")

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.gridLayout_5.addLayout(self.formLayout, 4, 0, 1, 2)

        self.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)

        self.label_8 = QtWidgets.QLabel(parent=self)
        self.label_8.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.label_8.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft |
            QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_5.addWidget(self.label_8, 0, 0, 1, 1)

        self.lineEdit_5 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_5.textChanged.connect(self.change_display_create_matrix_btn)
        self.lineEdit_5.setValidator(Int_1_1000_Validator())
        self.lineEdit_5.setMaximumWidth(218)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_5.sizePolicy().hasHeightForWidth())
        self.lineEdit_5.setSizePolicy(sizePolicy)
        self.lineEdit_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_5.addWidget(self.lineEdit_5, 0, 1, 1, 1)

        self.pushButton_2 = QtWidgets.QPushButton(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.pushButton_2.setDefault(True)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_5.addWidget(self.pushButton_2, 2, 1, 1, 1)

        self.label_9 = QtWidgets.QLabel(parent=self)
        self.label_9.setText("")
        self.label_9.setWordWrap(True)
        self.label_9.setObjectName("label_9")
        self.gridLayout_5.addWidget(self.label_9, 3, 1, 1, 1)


        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_4.setObjectName("gridLayout_4")

        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_4.addItem(spacerItem7, 0, 2, 1, 1)

        self.label_13 = QtWidgets.QLabel(parent=self)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 5, 0, 1, 2)

        self.algorithm_radiobuttons = [QtWidgets.QRadioButton(parent=self) for _ in range(6)]

        for i, radiobutton in enumerate(self.algorithm_radiobuttons):
            radiobutton.setObjectName(f'radioButton_{i + 3}')
            self.gridLayout_5.addWidget(radiobutton, i // 2 + 6, i % 2, 1, 1)
        self.algorithm_radiobuttons[0].setChecked(True)

        self.algorithm_radiobuttons[0].clicked.connect(lambda: self.color_selected_cells(0))
        self.algorithm_radiobuttons[1].clicked.connect(lambda: self.color_selected_cells(1))
        self.algorithm_radiobuttons[2].clicked.connect(lambda: self.color_selected_cells(2))
        self.algorithm_radiobuttons[3].clicked.connect(lambda: self.color_selected_cells(3))
        self.algorithm_radiobuttons[4].clicked.connect(lambda: self.color_selected_cells(4))
        self.algorithm_radiobuttons[5].clicked.connect(lambda: self.color_selected_cells(5))

        self.algorithm_colors = ('#A04D73', '#6A8DC1', '#267A8B', '#A3D4AF', '#B086A8', '#CDAFA5')

        self.pushButton_3 = QtWidgets.QPushButton(parent=self)
        self.pushButton_3.setDefault(True)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setVisible(False)
        self.gridLayout_5.addWidget(self.pushButton_3, 10, 0, 1, 1)

        self.label_10 = QtWidgets.QLabel(parent=self)
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.gridLayout_5.addWidget(self.label_10, 11, 0, 1, 1)

        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_6.setObjectName("gridLayout_6")

        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_6.addItem(spacerItem8, 0, 2, 1, 1)

        self.ans_label_24 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_24.sizePolicy().hasHeightForWidth())
        self.ans_label_24.setSizePolicy(sizePolicy)
        self.ans_label_24.setText("")
        self.ans_label_24.setObjectName("ans_label_24")
        self.gridLayout_6.addWidget(self.ans_label_24, 2, 4, 1, 1)
        self.ans_label_23 = QtWidgets.QLabel(parent=self)
        self.ans_label_23.setText("")
        self.ans_label_23.setObjectName("ans_label_23")
        self.gridLayout_6.addWidget(self.ans_label_23, 2, 3, 1, 1)
        self.ans_label_13 = QtWidgets.QLabel(parent=self)
        self.ans_label_13.setText("")
        self.ans_label_13.setObjectName("ans_label_13")
        self.gridLayout_6.addWidget(self.ans_label_13, 0, 0, 1, 1)
        self.ans_label_20 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_20.sizePolicy().hasHeightForWidth())
        self.ans_label_20.setSizePolicy(sizePolicy)
        self.ans_label_20.setText("")
        self.ans_label_20.setObjectName("ans_label_20")
        self.gridLayout_6.addWidget(self.ans_label_20, 1, 4, 1, 1)
        self.ans_label_16 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_16.sizePolicy().hasHeightForWidth())
        self.ans_label_16.setSizePolicy(sizePolicy)
        self.ans_label_16.setText("")
        self.ans_label_16.setObjectName("ans_label_16")
        self.gridLayout_6.addWidget(self.ans_label_16, 0, 4, 1, 1)
        self.ans_label_14 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_14.sizePolicy().hasHeightForWidth())
        self.ans_label_14.setSizePolicy(sizePolicy)
        self.ans_label_14.setText("")
        self.ans_label_14.setObjectName("ans_label_14")
        self.gridLayout_6.addWidget(self.ans_label_14, 0, 1, 1, 1)
        self.ans_label_18 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_18.sizePolicy().hasHeightForWidth())
        self.ans_label_18.setSizePolicy(sizePolicy)
        self.ans_label_18.setText("")
        self.ans_label_18.setObjectName("ans_label_18")
        self.gridLayout_6.addWidget(self.ans_label_18, 1, 1, 1, 1)
        self.ans_label_17 = QtWidgets.QLabel(parent=self)
        self.ans_label_17.setText("")
        self.ans_label_17.setObjectName("ans_label_17")
        self.gridLayout_6.addWidget(self.ans_label_17, 1, 0, 1, 1)
        self.ans_label_19 = QtWidgets.QLabel(parent=self)
        self.ans_label_19.setText("")
        self.ans_label_19.setObjectName("ans_label_19")
        self.gridLayout_6.addWidget(self.ans_label_19, 1, 3, 1, 1)
        self.ans_label_21 = QtWidgets.QLabel(parent=self)
        self.ans_label_21.setText("")
        self.ans_label_21.setObjectName("ans_label_21")
        self.gridLayout_6.addWidget(self.ans_label_21, 2, 0, 1, 1)
        self.ans_label_15 = QtWidgets.QLabel(parent=self)
        self.ans_label_15.setText("")
        self.ans_label_15.setObjectName("ans_label_15")
        self.gridLayout_6.addWidget(self.ans_label_15, 0, 3, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(parent=self)
        self.pushButton_5.setDefault(True)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.save_to_file)
        self.gridLayout_6.addWidget(self.pushButton_5, 3, 0, 1, 2)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_6.addItem(spacerItem8, 0, 2, 1, 1)
        self.ans_label_22 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_22.sizePolicy().hasHeightForWidth())
        self.ans_label_22.setSizePolicy(sizePolicy)
        self.ans_label_22.setText("")
        self.ans_label_22.setObjectName("ans_label_22")
        self.gridLayout_6.addWidget(self.ans_label_22, 2, 1, 1, 1)

        self.ans_label_26 = QtWidgets.QLabel(parent=self)
        self.ans_label_26.setText("")
        self.ans_label_26.setObjectName("ans_label_26")
        self.gridLayout_6.addWidget(self.ans_label_26, 3, 3, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_6, 14, 0, 1, 2)

        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_3.addItem(spacerItem9, 15, 0, 1, 1)

        self.label_13 = QtWidgets.QLabel(parent=self)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 5, 0, 1, 2)

        self.label_15 = QtWidgets.QLabel(parent=self)
        self.label_15.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.label_15.setText("")
        self.label_15.setWordWrap(True)
        self.label_15.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 3, 0, 1, 1)

        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                             QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout_5.addItem(spacerItem10, 1, 0, 1, 1)

        self.pushButton_7 = QtWidgets.QPushButton(parent=self)
        self.pushButton_7.clicked.connect(self.read_matrix_from_file)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_5.addWidget(self.pushButton_7, 2, 0, 1, 1)

        spacerItem11 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                             QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout_5.addItem(spacerItem11, 9, 0, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_5, 0, 0, 1, 2)

        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                             QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout_3.addItem(spacerItem12, 13, 0, 1, 1)

        self.label_back_6 = QtWidgets.QLabel()
        self.label_back_6.setObjectName('label_back_6')
        self.gridLayout_5.addWidget(self.label_back_6, 0, 0, 2, 2)
        self.label_back_6.lower()

        self.label_back_7 = QtWidgets.QLabel()
        self.label_back_7.setObjectName('label_back_7')
        self.gridLayout_5.addWidget(self.label_back_7, 4, 0, 1, 2);
        self.label_back_7.lower()
        self.label_back_7.setVisible(False)

        self.label_back_8 = QtWidgets.QLabel()
        self.label_back_8.setObjectName('label_back_8')
        self.gridLayout_5.addWidget(self.label_back_8, 5, 0, 5, 2)
        self.label_back_8.lower()

        self.vertical_view: QtWidgets.QGraphicsView | None = None

        self.setVisible_radiobuttons_tab_2(False)
        self.pushButton_5.setVisible(False)
        self.retranslateUi()

        self.setTabOrder(self.lineEdit_5, self.pushButton_7)
        self.setTabOrder(self.pushButton_7, self.pushButton_2)
        self.setTabOrder(self.pushButton_2, self.algorithm_radiobuttons[0])
        for i in range(len(self.algorithm_radiobuttons) - 1):
            self.setTabOrder(self.algorithm_radiobuttons[i], self.algorithm_radiobuttons[i + 1])

        self.setTabOrder(self.algorithm_radiobuttons[-1], self.pushButton_3)
        self.setTabOrder(self.pushButton_3, self.pushButton_5)

        self.position_of_matrix = 2
        self.elements_in_form_layout = 3

        self.num_varieties: int = 0
        self.matrix_from_file: list[list[float]] = []

        self.algorithms: Algorithms | None = None

        self.pushButton_2.clicked.connect(lambda: self.create_matrix_tab_2())

        self.pushButton_3.clicked.connect(self.read_matrix_tab_2)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.label_8.setText(_translate("InputMatrixPage", "Количество сортов свёклы"))
        self.label_8.setProperty("heading", _translate("InputMatrixPage", "true"))
        self.label_8.setProperty("exp_style", _translate("MainWindow", "true"))

        self.pushButton_2.setText(_translate("InputMatrixPage", "Создать матрицу"))
        self.pushButton_2.setProperty("exp_style", _translate("MainWindow", "true"))

        self.label_9.setProperty("error", _translate("InputMatrixPage", "true"))
        self.label_9.setProperty("exp_style", _translate("MainWindow", "true"))

        self.lineEdit_5.setProperty("exp_style", _translate("MainWindow", "true"))

        self.pushButton_3.setText(_translate("InputMatrixPage", "Рассчитать"))
        self.pushButton_3.setProperty("exp_style", _translate("MainWindow", "true"))

        self.label_10.setProperty("error", _translate("InputMatrixPage", "true"))
        self.label_10.setProperty("exp_style", _translate("MainWindow", "true"))

        self.label_13.setText(_translate("MainWindow", "Алгоритм, отображаемый на матрице"))
        self.label_13.setProperty("heading", _translate("InputMatrixPage", "true"))
        self.label_13.setProperty("exp_style", _translate("MainWindow", "true"))

        self.label_15.setProperty("heading", _translate("MainWindow", "true"))
        self.label_15.setProperty("exp_style", _translate("MainWindow", "true"))

        self.pushButton_5.setText(_translate("MainWindow", "Сохранить"))
        self.pushButton_5.setProperty("exp_style", _translate("MainWindow", "true"))

        self.pushButton_7.setText(_translate("MainWindow", "Загрузить данные из файла"))
        self.pushButton_7.setProperty("exp_style", _translate("MainWindow", "true"))

        self.label_15.setProperty("heading", _translate("MainWindow", "true"))

        algorithm_names = ["Венгерский макс", "Венгерский мин", "Жадный",
                           "Бережливый", "Жадно-бережливый", "Бережливо-жадный"]
        for i in range(len(self.algorithm_radiobuttons)):
            self.algorithm_radiobuttons[i].setText(_translate("InputMatrixPage", algorithm_names[i]))
            self.algorithm_radiobuttons[i].setProperty("exp_style", _translate("MainWindow", "true"))
            self.algorithm_radiobuttons[i].setProperty("color_of_back_label", _translate("MainWindow", "red"))
            self.algorithm_radiobuttons[i].setProperty("subheading", _translate("MainWindow", "true"))

        self.ans_label_24.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_24.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_23.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_23.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_13.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_13.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_20.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_20.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_16.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_16.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_14.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_14.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_18.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_18.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_17.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_17.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_19.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_19.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_21.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_21.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_15.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_15.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_22.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_22.setProperty("ans_style", _translate("MainWindow", "true"))

        self.ans_label_26.setProperty("exp_style", _translate("MainWindow", "true"))
        self.ans_label_26.setProperty("ans_style", _translate("MainWindow", "true"))

    def write_err_tab_2(self, s: str):
        self.label_9.setText(s)

    def write_err_2_tab_2(self, s: str):
        self.label_10.setText(s)

    def write_answers_tab_2(self, algorithms: Algorithms):
        self.pushButton_5.setVisible(True)

        name_labels = [self.ans_label_13, self.ans_label_15, self.ans_label_17,
                       self.ans_label_19, self.ans_label_21, self.ans_label_23]

        value_labels = [self.ans_label_14, self.ans_label_16, self.ans_label_18,
                        self.ans_label_20, self.ans_label_22, self.ans_label_24]

        for i in range(len(name_labels)):
            name_labels[i].setText(algorithms[i].name + ':')
            value_labels[i].setText(str(round(algorithms[i].ans[-1], 2)))

    def clear_answers_tab_2(self):
        self.pushButton_5.setVisible(False)
        self.setVisible_radiobuttons_tab_2(False)
        self.clear_selected_cells()
        self.plot_page.clear_graph()

        labels = [self.ans_label_13, self.ans_label_14, self.ans_label_15, self.ans_label_16,
                  self.ans_label_17, self.ans_label_18, self.ans_label_19, self.ans_label_20,
                  self.ans_label_21, self.ans_label_22, self.ans_label_23, self.ans_label_24]

        for label in labels:
            label.setText('')

    def setVisible_radiobuttons_tab_2(self, bool_variable: bool):
        self.label_13.setVisible(bool_variable)
        self.label_back_8.setVisible(bool_variable)
        for radiobutton in self.algorithm_radiobuttons:
            radiobutton.setVisible(bool_variable)

    def delete_old_matrix(self):
        if self.formLayout.rowCount() > 0:
            for _ in range(self.elements_in_form_layout):
                self.formLayout.removeRow(0)
            self.label_back_7.setVisible(False)

    def delete_errors(self):
        self.write_err_tab_2('')
        self.write_err_2_tab_2('')

    def create_vertical_label(self):
        scene_for_label = QtWidgets.QGraphicsScene()
        scene_for_label.setBackgroundBrush(QtGui.QColor(234, 234, 234))

        self.vertical_view = QtWidgets.QGraphicsView()
        self.vertical_view.setObjectName('vertical_view')
        self.vertical_view.setScene(scene_for_label)
        self.vertical_view.setFixedWidth(36)

        vertical_label = scene_for_label.addText('Дни', font=QtGui.QFont('Century Gothic', 9))
        vertical_label.setRotation(-90)
        vertical_label.setDefaultTextColor(QtGui.QColor(28, 69, 112))

    def create_matrix_tab_2(self, matrix: list[list[float]] | None = None):
        self.matrix_from_file = []
        self.change_display_create_matrix_btn()
        self.algorithms = None
        self.clear_answers_tab_2()
        fl = 0
        n = 0
        try:
            n = int(self.lineEdit_5.text())
            self.write_err_tab_2("")
        except:
            fl = 1
            self.pushButton_3.setVisible(False)
            self.setVisible_radiobuttons_tab_2(False)
            self.write_err_tab_2("Введите число")

        self.num_varieties = n
        self.delete_old_matrix()
        if fl == 0:
            lbl_1 = QtWidgets.QLabel()
            lbl_1.setObjectName('lbl_1')
            lbl_1.setText("Введите матрицу")
            lbl_1.setProperty("exp_style", "true")
            lbl_1.setProperty("heading", "true")

            lbl_2 = QtWidgets.QLabel()
            lbl_2.setObjectName('lbl_2')
            lbl_2.setText("Сорта")
            lbl_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            lbl_2.setProperty("exp_style", "true")

            self.formLayout.insertRow(self.position_of_matrix - 2, lbl_1)
            self.formLayout.insertRow(self.position_of_matrix - 1, lbl_2)

            layout_grid_input = QtWidgets.QGridLayout()
            for i in range(n):
                for j in range(n):
                    fld = QtWidgets.QLineEdit("fld_" + str(j + i * n))
                    fld.setValidator(Double_0_100000_Validator())
                    fld.setText("" if matrix is None else str(matrix[i][j]))
                    fld.textEdited.connect(self.clear_answers_tab_2)
                    fld.setCursorPosition(0)
                    fld.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
                    if n > 8:
                        fld.setFixedSize(40, 20)
                        fld.setProperty('small_cell', 'true')
                    else:
                        fld.setFixedSize(80, 40)
                    layout_grid_input.addWidget(fld, i, j)

            label_role = QtWidgets.QFormLayout.ItemRole.LabelRole
            layout_grid_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
            layout_grid_input.setContentsMargins(10 + 36, 10, 10, 10)
            self.formLayout.insertRow(self.position_of_matrix, layout_grid_input)
            self.create_vertical_label()
            self.formLayout.setWidget(self.position_of_matrix, label_role, self.vertical_view)
            self.label_back_7.setVisible(True)

            spanning_role = QtWidgets.QFormLayout.ItemRole.SpanningRole
            self.setTabOrder(self.pushButton_2, self.formLayout.itemAt(self.position_of_matrix,
                                                                       spanning_role).itemAtPosition(0, 0).widget())
            for i in range(n):
                for j in range(n):
                    if i == n - 1 and j == n - 1:
                        break
                    next_i = i + (j + 1) // n
                    next_j = (j + 1) % n
                    self.setTabOrder(self.formLayout.itemAt(self.position_of_matrix,
                                                            spanning_role).itemAtPosition(i, j).widget(),
                                     self.formLayout.itemAt(self.position_of_matrix,
                                                            spanning_role).itemAtPosition(next_i, next_j).widget())
            self.setTabOrder(self.formLayout.itemAt(self.position_of_matrix,
                                                    spanning_role).itemAtPosition(n - 1, n - 1).widget(),
                             self.pushButton_3)

            self.pushButton_3.setVisible(True)

    def clear_selected_cells(self):
        if self.algorithms and self.formLayout.rowCount() > 0:
            n = self.algorithms.num_days
            spanning_role = QtWidgets.QFormLayout.ItemRole.SpanningRole
            for i in range(n):
                for j in range(n):
                    self.formLayout.itemAt(self.position_of_matrix,
                                           spanning_role).itemAtPosition(j, i).widget().setStyleSheet(
                        "background-color: rgba(0, 0, 0, 0);")

    def color_selected_cells(self, algorithm_ind: int):
        if self.algorithm_radiobuttons[algorithm_ind].isChecked():
            n = self.algorithms.num_days
            spanning_role = QtWidgets.QFormLayout.ItemRole.SpanningRole

            self.clear_selected_cells()
            if algorithm_ind is not None:
                for i in range(n):
                    self.formLayout.itemAt(self.position_of_matrix, spanning_role).itemAtPosition(
                        i, self.algorithms[algorithm_ind].column_indexes[i]).widget().setStyleSheet(
                        f"background-color: {self.algorithm_colors[algorithm_ind]};")

    def calculate_answer(self, matrix: list[list[float]]):
        self.algorithms = Algorithms(self.num_varieties)
        self.algorithms.run_algorithms(np.matrix(matrix))
        self.algorithms.calculate_average(1)

    def show_answer(self):
        self.plot_page.print_plots(self.algorithms)
        self.write_answers_tab_2(self.algorithms)

    def show_info_input(self):
        self.label_15.setText('Чтобы создать матрицу,\nвведите размер меньше 15\nили воспользуйтесь\n'
                              'вводом из файла')


    def change_display_create_matrix_btn(self):
        label_value = self.lineEdit_5.text()
        if not label_value.isdigit() or int(label_value) == 0:
            self.pushButton_2.setDisabled(True)
            self.write_err_tab_2("Введите натуральное число")
            return
        else:
            self.write_err_tab_2("")

        if int(label_value) > 15:
            self.pushButton_2.setDisabled(True)
            if len(self.matrix_from_file) == 0:
                self.show_info_input()
        else:
            self.pushButton_2.setDisabled(False)
            if len(self.matrix_from_file) == 0:
                self.label_15.setText('')

    def read_matrix_tab_2(self):
        try:
            n = int(self.lineEdit_5.text())
            if self.num_varieties != n:
                self.write_err_tab_2("Количество сортов должно совпадать с размерами матрицы")
                return
            self.write_err_tab_2("")
        except:
            self.write_err_tab_2("Введите натуральное число")
            return
        arr = [[-1.0] * n for i in range(n)]
        f = 0
        spanning_role = QtWidgets.QFormLayout.ItemRole.SpanningRole
        if n < 16:
            for i in range(n):
                for j in range(n):
                    try:
                        ln = float(self.formLayout.itemAt(self.position_of_matrix,
                                                          spanning_role).itemAtPosition(i, j).widget().text())
                        arr[i][j] = ln
                        self.write_err_2_tab_2("")
                    except:
                        self.write_err_2_tab_2("Заполните все поля числами")
                        self.clear_answers_tab_2()
                        self.plot_page.clear_graph()
                        return
        elif len(self.matrix_from_file) > 0:
            arr = self.matrix_from_file
        else:
            return

        self.calculate_answer(arr)
        self.show_answer()
        if n < 16:
            self.setVisible_radiobuttons_tab_2(True)

            for i in range(len(self.algorithm_radiobuttons)):
                if self.algorithm_radiobuttons[i].isChecked():
                    self.color_selected_cells(i)
                    break

        errors = self.algorithms.calculate_error()

    def read_matrix_from_file(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Выбор файла', '/', '*.txt')[0]


    def save_to_file(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить данные', '/матрица сортов.csv',
                                                     '*.csv')[0]

