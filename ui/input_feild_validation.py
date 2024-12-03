from PyQt6 import QtGui
from PyQt6 import QtCore
import typing


class Int_1_1000_Validator(QtGui.QValidator):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Int_1_1000_Validator, cls).__new__(cls)
        return cls.instance

    def validate(self, a0: typing.Optional[str], a1: int) -> typing.Tuple['QtGui.QValidator.State', str, int]:
        try:
            if a0 == '' or 0 <= int(a0) <= 1000:
                return QtGui.QValidator.State.Acceptable, a0, a1
        except Exception:
            pass
        return QtGui.QValidator.State.Invalid, a0, a1


class Double_0_100000_Validator(QtGui.QValidator):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Double_0_100000_Validator, cls).__new__(cls)
        return cls.instance

    def validate(self, a0: typing.Optional[str], a1: int) -> typing.Tuple['QtGui.QValidator.State', str, int]:
        try:
            if a0 == '' or 0 <= float(a0) <= 100000:
                return QtGui.QValidator.State.Acceptable, a0, a1
        except Exception:
            pass
        return QtGui.QValidator.State.Invalid, a0, a1


class Double_0_1_Validator(QtGui.QValidator):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Double_0_1_Validator, cls).__new__(cls)
        return cls.instance

    def validate(self, a0: typing.Optional[str], a1: int) -> typing.Tuple['QtGui.QValidator.State', str, int]:
        try:
            if a0 == '' or 0 <= float(a0) <= 1:
                return QtGui.QValidator.State.Acceptable, a0, a1
        except Exception:
            pass
        return QtGui.QValidator.State.Invalid, a0, a1


class DistributionValidator(QtGui.QValidator):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DistributionValidator, cls).__new__(cls)
        return cls.instance

    def validate(self, a0: typing.Optional[str], a1: int) -> typing.Tuple['QtGui.QValidator.State', str, int]:
        if a0.lower() in ['0', '1', 'нормальное', 'равномерное']:
            return QtGui.QValidator.State.Acceptable, a0, a1
        return QtGui.QValidator.State.Invalid, a0, a1


class InorganicValidator(QtGui.QValidator):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(InorganicValidator, cls).__new__(cls)
        return cls.instance

    def validate(self, a0: typing.Optional[str], a1: int) -> typing.Tuple['QtGui.QValidator.State', str, int]:
        if a0.lower() in ['0', '1', 'да', 'нет']:
            return QtGui.QValidator.State.Acceptable, a0, a1
        return QtGui.QValidator.State.Invalid, a0, a1

