
from PyQt6 import QtGui
from ui.plot_page import PlotPage
from core.algorithms import Algorithms
from core.experement import experiment


from ui.input_feild_validation import Int_1_1000_Validator, Double_0_100000_Validator, Double_0_1_Validator

from PyQt6 import QtWidgets
from PyQt6 import QtCore

class InputExperimentPage(QtWidgets.QWidget):
    """Страница ввода параметров для эксперимента"""

    def __init__(self, plot_page: PlotPage):
        super(InputExperimentPage, self).__init__()
        self.plot_page = plot_page
        self.setObjectName('InputExperimentPage')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                           QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)

        self.gridLayout_8 = QtWidgets.QGridLayout(self)
        self.gridLayout_8.setContentsMargins(20, 20, 20, 20)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                           QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem, 15, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_3.setValidator(Double_0_100000_Validator())
        self.lineEdit_3.setMaximumWidth(218)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
        self.lineEdit_3.setSizePolicy(sizePolicy)
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_3.textChanged.connect(self.clear_answers_tab)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 4, 1, 1, 1)

        self.label_5 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 6, 0, 1, 1)

        self.label = QtWidgets.QLabel(parent=self)
        self.label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_2 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)

        self.lineEdit_1 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_1.setValidator(Int_1_1000_Validator())
        self.lineEdit_1.setMaximumWidth(218)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_1.sizePolicy().hasHeightForWidth())
        self.lineEdit_1.textChanged.connect(self.clear_answers_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_1.sizePolicy().hasHeightForWidth())
        self.lineEdit_1.setSizePolicy(sizePolicy)
        self.lineEdit_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.gridLayout.addWidget(self.lineEdit_1, 0, 1, 1, 1)

        self.label_3 = QtWidgets.QLabel(parent=self)
        self.label_3.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)

        self.label_4 = QtWidgets.QLabel(parent=self)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(-1)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignRight
            | QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 4, 0, 1, 1)

        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_2.setValidator(Double_0_100000_Validator())
        self.lineEdit_2.setMaximumWidth(218)
        self.lineEdit_2.textChanged.connect(self.clear_answers_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_2.sizePolicy().hasHeightForWidth())
        self.lineEdit_2.setSizePolicy(sizePolicy)
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 3, 1, 1, 1)

        self.label = QtWidgets.QLabel(parent=self)
        self.label.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.label_11 = QtWidgets.QLabel(parent=self)
        self.label_11.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing |
            QtCore.Qt.AlignmentFlag.AlignTrailing | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 9, 0, 1, 1)

        self.lineEdit_8 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_8.setValidator(Double_0_1_Validator())
        self.lineEdit_8.setMaximumWidth(218)
        self.lineEdit_8.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout.addWidget(self.lineEdit_8, 10, 1, 1, 1)

        self.lineEdit_7 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_7.setValidator(Double_0_1_Validator())
        self.lineEdit_7.setMaximumWidth(218)
        self.lineEdit_7.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_7.textChanged.connect(self.clear_answers_tab)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout.addWidget(self.lineEdit_7, 9, 1, 1, 1)

        self.label_12 = QtWidgets.QLabel(parent=self)
        self.label_12.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTrailing |
            QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 10, 0, 1, 1)

        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem1, 5, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem2, 15, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem3, 13, 0, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem4, 1, 0, 1, 1)

        self.label_6 = QtWidgets.QLabel(parent=self)
        self.label_6.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 14, 0, 1, 1)

        self.pushButton_6 = QtWidgets.QPushButton(parent=self)
        self.pushButton_6.clicked.connect(self.fill_params_from_file)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 16, 0, 1, 1)

        self.lineEdit_4 = QtWidgets.QLineEdit(parent=self)
        self.lineEdit_4.setValidator(Int_1_1000_Validator())
        self.lineEdit_4.setMaximumWidth(218)
        self.lineEdit_4.textChanged.connect(self.clear_answers_tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_4.sizePolicy().hasHeightForWidth())
        self.lineEdit_4.setSizePolicy(sizePolicy)
        self.lineEdit_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout.addWidget(self.lineEdit_4, 14, 1, 1, 1)

        self.pushButton = QtWidgets.QPushButton(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.pushButton.setDefault(True)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 17, 0, 1, 1)

        spacerItem5 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout.addItem(spacerItem5, 11, 0, 1, 1)

        self.checkBox = QtWidgets.QCheckBox(parent=self)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.stateChanged.connect(self.clear_answers_tab)
        self.gridLayout.addWidget(self.checkBox, 12, 0, 1, 2)

        self.radioButton_1 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_1.setObjectName("radioButton_1")
        self.radioButton_1.toggled.connect(self.clear_answers_tab)
        self.gridLayout.addWidget(self.radioButton_1, 8, 0, 1, 2)

        self.radioButton_2 = QtWidgets.QRadioButton(parent=self)
        self.radioButton_2.setChecked(True)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_2.toggled.connect(self.clear_answers_tab)
        self.gridLayout.addWidget(self.radioButton_2, 7, 0, 1, 2)

        self.label_7 = QtWidgets.QLabel(parent=self)
        self.label_7.setText("")
        self.label_7.setWordWrap(True)
        self.label_7.setObjectName("label_7")
        self.label_7.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft |
            QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.gridLayout.addWidget(self.label_7, 17, 1, 1, 1)

        self.gridLayout_8.addLayout(self.gridLayout, 0, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum,
                                            QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_8.addItem(spacerItem6, 1, 0, 1, 1)

        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.gridLayout_4.setObjectName("gridLayout_4")

        self.ans_label_1 = QtWidgets.QLabel(parent=self)
        self.ans_label_1.setText("")
        self.ans_label_1.setObjectName("ans_label_1")
        self.gridLayout_4.addWidget(self.ans_label_1, 0, 0, 1, 1)
        self.ans_label_11 = QtWidgets.QLabel(parent=self)
        self.ans_label_11.setText("")
        self.ans_label_11.setObjectName("ans_label_11")
        self.gridLayout_4.addWidget(self.ans_label_11, 2, 3, 1, 1)
        self.ans_label_12 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_12.sizePolicy().hasHeightForWidth())
        self.ans_label_12.setSizePolicy(sizePolicy)
        self.ans_label_12.setText("")
        self.ans_label_12.setObjectName("ans_label_12")
        self.gridLayout_4.addWidget(self.ans_label_12, 2, 4, 1, 1)

        self.ans_label_2 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_2.sizePolicy().hasHeightForWidth())
        self.ans_label_2.setSizePolicy(sizePolicy)
        self.ans_label_2.setText("")
        self.ans_label_2.setObjectName("ans_label_2")
        self.gridLayout_4.addWidget(self.ans_label_2, 0, 1, 1, 1)

        self.ans_label_8 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_8.sizePolicy().hasHeightForWidth())
        self.ans_label_8.setSizePolicy(sizePolicy)
        self.ans_label_8.setText("")
        self.ans_label_8.setObjectName("ans_label_8")
        self.gridLayout_4.addWidget(self.ans_label_8, 1, 4, 1, 1)

        self.ans_label_9 = QtWidgets.QLabel(parent=self)
        self.ans_label_9.setText("")
        self.ans_label_9.setObjectName("ans_label_9")
        self.gridLayout_4.addWidget(self.ans_label_9, 2, 0, 1, 1)

        self.ans_label_4 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_4.sizePolicy().hasHeightForWidth())
        self.ans_label_4.setSizePolicy(sizePolicy)
        self.ans_label_4.setText("")
        self.ans_label_4.setObjectName("ans_label_4")
        self.gridLayout_4.addWidget(self.ans_label_4, 0, 4, 1, 1)

        self.ans_label_6 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_6.sizePolicy().hasHeightForWidth())
        self.ans_label_6.setSizePolicy(sizePolicy)
        self.ans_label_6.setText("")
        self.ans_label_6.setObjectName("ans_label_6")
        self.gridLayout_4.addWidget(self.ans_label_6, 1, 1, 1, 1)

        self.ans_label_7 = QtWidgets.QLabel(parent=self)
        self.ans_label_7.setText("")
        self.ans_label_7.setObjectName("ans_label_7")
        self.gridLayout_4.addWidget(self.ans_label_7, 1, 3, 1, 1)

        self.ans_label_10 = QtWidgets.QLabel(parent=self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ans_label_10.sizePolicy().hasHeightForWidth())
        self.ans_label_10.setSizePolicy(sizePolicy)
        self.ans_label_10.setText("")
        self.ans_label_10.setObjectName("ans_label_10")
        self.gridLayout_4.addWidget(self.ans_label_10, 2, 1, 1, 1)

        self.ans_label_5 = QtWidgets.QLabel(parent=self)
        self.ans_label_5.setText("")
        self.ans_label_5.setObjectName("ans_label_5")
        self.gridLayout_4.addWidget(self.ans_label_5, 1, 0, 1, 1)

        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_4.addItem(spacerItem7, 0, 2, 1, 1)

        self.ans_label_3 = QtWidgets.QLabel(parent=self)
        self.ans_label_3.setText("")
        self.ans_label_3.setObjectName("ans_label_3")
        self.gridLayout_4.addWidget(self.ans_label_3, 0, 3, 1, 1)

        self.pushButton_4 = QtWidgets.QPushButton(parent=self)
        self.pushButton_4.setDefault(True)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setVisible(False)
        self.gridLayout_4.addWidget(self.pushButton_4, 3, 0, 1, 2)

        self.ans_label_25 = QtWidgets.QLabel(parent=self)
        self.ans_label_25.setText("")
        self.ans_label_25.setObjectName("ans_label_25")
        self.gridLayout_4.addWidget(self.ans_label_25, 3, 3, 1, 1)

        self.gridLayout_8.addLayout(self.gridLayout_4, 2, 0, 1, 1)

        self.label_back_1 = QtWidgets.QLabel()
        self.label_back_1.setObjectName('label_back_1')
        self.gridLayout.addWidget(self.label_back_1, 0, 0, 2, 2);
        self.label_back_1.lower()

        self.label_back_2 = QtWidgets.QLabel()
        self.label_back_2.setObjectName('label_back_2')
        self.gridLayout.addWidget(self.label_back_2, 2, 0, 4, 2)
        self.label_back_2.lower()

        self.label_back_3 = QtWidgets.QLabel()
        self.label_back_3.setObjectName('label_back_3')
        self.gridLayout.addWidget(self.label_back_3, 6, 0, 6, 2)
        self.label_back_3.lower()

        self.label_back_4 = QtWidgets.QLabel()
        self.label_back_4.setObjectName('label_back_4')
        self.gridLayout.addWidget(self.label_back_4, 12, 0, 2, 2)
        self.label_back_4.lower()

        self.label_back_5 = QtWidgets.QLabel()
        self.label_back_5.setObjectName('label_back_5')
        self.gridLayout.addWidget(self.label_back_5, 14, 0, 2, 2)
        self.label_back_5.lower()

        self.retranslateUi()

        self.setTabOrder(self, self.lineEdit_1)
        self.setTabOrder(self.lineEdit_1, self.lineEdit_2)
        self.setTabOrder(self.lineEdit_2, self.lineEdit_3)
        self.setTabOrder(self.lineEdit_3, self.radioButton_2)
        self.setTabOrder(self.radioButton_2, self.radioButton_1)
        self.setTabOrder(self.radioButton_1, self.lineEdit_7)
        self.setTabOrder(self.lineEdit_7, self.lineEdit_8)
        self.setTabOrder(self.lineEdit_8, self.checkBox)
        self.setTabOrder(self.checkBox, self.lineEdit_4)
        self.setTabOrder(self.lineEdit_4, self.pushButton_6)
        self.setTabOrder(self.pushButton_6, self.pushButton)
        self.setTabOrder(self.pushButton, self.pushButton_4)

        self.radioButton_2.clicked.connect(self.show_normal_params)
        self.radioButton_1.clicked.connect(self.show_uniform_params)
        self.pushButton.clicked.connect(self.run_experiment)
        self.pushButton_4.clicked.connect(self.save_to_file)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.pushButton.setText(_translate("InputExperimentPage", "Рассчитать"))
        self.pushButton.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.pushButton_6.setText(_translate("MainWindow", "Загрузить данные из файла"))
        self.pushButton_6.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label.setText(_translate("InputExperimentPage", "Количество сортов свёклы"))
        self.label.setProperty("heading", _translate("InputExperimentPage", "true"))
        self.label.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label_2.setText(_translate("InputExperimentPage", "Сахаристость до переработки"))
        self.label_2.setProperty("heading", _translate("InputExperimentPage", "true"))
        self.label_2.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label_5.setText(_translate("InputExperimentPage", "Распределение деградации"))
        self.label_5.setProperty("heading", _translate("InputExperimentPage", "true"))
        self.label_5.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label_4.setText(_translate("InputExperimentPage", "Максимальное значение:"))
        self.label_4.setProperty("subheading", _translate("InputExperimentPage", "true"))
        self.label_4.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label_3.setText(_translate("InputExperimentPage", "Минимальное значение:"))
        self.label_3.setProperty("subheading", _translate("InputExperimentPage", "true"))
        self.label_3.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.radioButton_1.setText(_translate("InputExperimentPage", "Равномерное распределение"))
        self.radioButton_1.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.radioButton_1.setProperty("subheading", _translate("InputExperimentPage", "true"))
        self.radioButton_1.setProperty("color_of_back_label", _translate("MainWindow", "white"))

        self.checkBox.setText(_translate("InputExperimentPage", "Учитывать влияние неорганики"))
        self.checkBox.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.checkBox.setProperty("subheading", _translate("InputExperimentPage", "true"))
        self.checkBox.setProperty("color_of_back_label", _translate("MainWindow", "red"))

        self.label_6.setText(_translate("InputExperimentPage", "Количество экспериментов"))
        self.label_6.setProperty("heading", _translate("InputExperimentPage", "true"))
        self.label_6.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.label_7.setProperty("error", _translate("InputExperimentPage", "true"))
        self.label_7.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_7.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.radioButton_2.setText(_translate("InputExperimentPage", "Нормальное распределение"))
        self.radioButton_2.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.radioButton_2.setProperty("subheading", _translate("InputExperimentPage", "true"))
        self.radioButton_2.setProperty("color_of_back_label", _translate("MainWindow", "white"))

        self.label_11.setText(_translate("InputExperimentPage", "Среднее значение:"))
        self.label_11.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.label_11.setProperty("subheading", _translate("InputExperimentPage", "true"))

        self.label_12.setText(_translate("InputExperimentPage", "Отклонение:"))
        self.label_12.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.label_12.setProperty("subheading", _translate("InputExperimentPage", "true"))

        self.pushButton_4.setText(_translate("InputExperimentPage", "Сохранить"))
        self.pushButton_4.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.ans_label_25.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_25.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_1.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_1.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_11.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_11.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_12.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_12.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_2.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_2.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_8.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_8.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_9.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_9.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_4.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_4.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_6.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_6.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_7.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_7.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_10.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_10.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_5.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_5.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.ans_label_3.setProperty("exp_style", _translate("InputExperimentPage", "true"))
        self.ans_label_3.setProperty("ans_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_3.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_1.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_2.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_8.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_7.setProperty("exp_style", _translate("InputExperimentPage", "true"))

        self.lineEdit_4.setProperty("exp_style", _translate("InputExperimentPage", "true"))

    def write_answers(self, algorithms: Algorithms):
        self.pushButton_4.setVisible(True)
        name_labels = [self.ans_label_1, self.ans_label_3, self.ans_label_5,
                       self.ans_label_7, self.ans_label_9, self.ans_label_11]

        value_labels = [self.ans_label_2, self.ans_label_4, self.ans_label_6,
                        self.ans_label_8, self.ans_label_10, self.ans_label_12]

        # Инициализация переменных для определения наилучшего алгоритма
        best_algorithm_index = None
        best_value = float('inf')  # Предполагаем, что мы минимизируем значение

        for i in range(len(name_labels)):
            name_labels[i].setText(algorithms[i].name + ':')
            value_labels[i].setText(str(round(algorithms[i].ans[-1], 2)))

            # Проверка наилучшего алгоритма
            if algorithms[i].ans[-1] < best_value:
                best_value = algorithms[i].ans[-1]
                best_algorithm_index = i

        # Вывод сообщения о наилучшем алгоритме
        if best_algorithm_index is not None:
            best_algorithm_name = algorithms[best_algorithm_index].name
            message = f"Наилучший алгоритм: {best_algorithm_name} с значением {round(best_value, 2)}"
            # Здесь можно использовать QLabel или QMessageBox для отображения сообщения
            QtWidgets.QMessageBox.information(self, "Результат", message)

    def clear_answers_tab(self):
        self.pushButton_4.setVisible(False)
        self.plot_page.clear_graph()
        labels = [self.ans_label_1, self.ans_label_2, self.ans_label_3, self.ans_label_4,
                  self.ans_label_5, self.ans_label_6, self.ans_label_7, self.ans_label_8,
                  self.ans_label_9, self.ans_label_10, self.ans_label_11, self.ans_label_12]

        for label in labels:
            label.setText('')

    def show_normal_params(self):
        self.label_11.setText("Среднее значение:")
        self.label_12.setText("Отклонение:")

    def show_uniform_params(self):
        self.label_11.setText("Минимальное значение:")
        self.label_12.setText("Максимальное значение:")

    def write_err_tab(self, s: str):
        self.label_7.setText(s)

    def __check_params(self, n: int, t: int, min_a: float, max_a: float,
                     min_b: float, max_b: float, is_normal: bool) -> bool:
        if min_a > max_a:
            self.write_err_tab('Минимальная сахаристость должна быть не больше максимальной')
            return False

        if not is_normal and min_b > max_b:
            self.write_err_tab('Минимальная деградация\nдолжна быть не больше максимальной')
            return False

        max_num_experiment = 1000000 // n ** 2
        if t > max_num_experiment:
            self.write_err_tab(f'Макс кол-во экспериментов для данной матрицы - {max_num_experiment}')
            return False
        return True

    def run_experiment(self):
        is_normal = False
        try:
            n = int(self.lineEdit_1.text())
            min_a = float(self.lineEdit_2.text())
            max_a = float(self.lineEdit_3.text())
            min_b = float(self.lineEdit_7.text())
            max_b = float(self.lineEdit_8.text())
            t = int(self.lineEdit_4.text())
            if self.radioButton_1.isChecked():
                is_normal = False
            if self.radioButton_2.isChecked():
                is_normal = True

            if self.checkBox.isChecked():
                organic = True
            else:
                organic = False
            self.write_err_tab("")
        except:
            self.write_err_tab("Заполните все поля числами")
            self.clear_answers_tab()
            return

        if not self.__check_params(n, t, min_a, max_a, min_b, max_b, is_normal):
            self.clear_answers_tab()
            return

        algorithms: Algorithms = experiment(n, t, min_a, max_a, min_b, max_b, organic, is_normal)
        self.write_answers(algorithms)
        self.plot_page.print_plots(algorithms)
        errors = algorithms.calculate_error()


    def fill_params_from_file(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Выбор файла', '/', '*.txt')[0]
        # if path:
        #     info_msg = QtWidgets.QMessageBox()
        #     info_msg.setWindowTitle('Ввод параметров эксперимента')
        #     info_msg.setWindowIcon(QtGui.QIcon(Paths.path_to_logo))
        #     try:
        #         params = FileController.read_experiment_params(path)
        #         self.lineEdit_1.setText(str(params[0]))
        #         self.lineEdit_2.setText(str(params[1]))
        #         self.lineEdit_3.setText(str(params[2]))
        #         if params[3]:
        #             self.radioButton_2.setChecked(True)
        #         else:
        #             self.radioButton_1.setChecked(True)
        #         self.lineEdit_7.setText(str(params[4]))
        #         self.lineEdit_8.setText(str(params[5]))
        #         if params[6]:
        #             self.checkBox.setChecked(True)
        #         else:
        #             self.checkBox.setChecked(False)
        #         self.lineEdit_4.setText(str(params[7]))
        #     except ValueError:
        #         info_msg.setText('Неверный формат данных!')
        #         info_msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        #         info_msg.exec()
        #     except Exception as error:
        #         info_msg.setText('Произошла ошибка при вводе параметров из файла!')
        #         info_msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        #         info_msg.exec()
        #         print(error)

    def save_to_file(self):
        path = QtWidgets.QFileDialog.getSaveFileName(self, 'Сохранить данные', '/параметры эксперимента.csv',
                                                     '*.csv')[0]
        # if path:
        #     info_msg = QtWidgets.QMessageBox()
        #     info_msg.setWindowTitle('Сохранение')
        #     info_msg.setWindowIcon(QtGui.QIcon(Paths.path_to_logo))
        #     try:
        #         FileController.user_save_experiment(path)
        #         info_msg.setText('Файл успешно сохранен!')
        #         info_msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        #     except Exception as error:
        #         info_msg.setText('Произошла ошибка при сохранении файла!')
        #         info_msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        #         print(error)
        #     finally:
        #         info_msg.exec()
