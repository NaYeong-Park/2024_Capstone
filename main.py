from PyQt5 import QtWidgets
import sys
from pictureUI import AugmentationApp


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = AugmentationApp()
    mainWin.show()
    sys.exit(app.exec_())
