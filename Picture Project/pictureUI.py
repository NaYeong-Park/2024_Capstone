
from PyQt5 import QtWidgets, uic, Qt, QtGui, sip, QtCore
from PyQt5.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem, QCursor, QPainter
from PyQt5.QtCore import Qt, QPointF, QRectF, QPoint
import tkinter as tk
import numpy as np
import cv2, os, logging
from PyQt5.QtWidgets import *

logging.basicConfig(filename='augmentation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class AugmentationApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.root = tk.Tk()
        self.listbox = tk.Listbox(self.root)


        # UI 파일 로드
        _ui = os.path.join(os.path.dirname(__file__), "picture UI.ui")  # Ensure the path is correct
        uic.loadUi(_ui, self)

        self.scene = QGraphicsScene()  # 이미지를 표시할 QGraphicsScene 객체 생성
        self.graphicsView.setScene(self.scene)  # graphicsView에 scene 설정
        self.scene2 = QGraphicsScene()
        self.graphicsView2.setScene(self.scene2)

        self.image_path = ""
        self.augmented_images = []
        self.points = []
        self.ellipses = []

        self.isDragging = False  # 전역 변수를 클래스 변수로 선언
        self.isSelecting = False  # ROI 선택 모드를 위한 변수 추가
        self.mode_free_roi = False  # 자유형 ROI 모드 활성화 초기화
        self.roi_path_item = None  # 자유형 ROI 경로 아이템 초기화
        self.roi_path = None  # 그리기를 위한 QPainterPath 초기화
        self.point_mode = False  # 포인트 모드 변수 초기화
        self.sticker_mode = False  # 스티커 모드 플래그 초기화
        self.roi_selected = False

        self.graphicsView.setMouseTracking(True)  # 마우스 추적 활성화
        self.selectionRect = None  # 초기 roi 사각형 초기화

        self.original_image = None
        self.modified_image = None
        self.modified2_image = None
        self.free_roi_image = None
        self.thres_image = None
        self.canny_image = None
        self.morph_image = None
        self.blur_image = None

        self.loadButton.clicked.connect(self.load_image)

        self.brightButton.clicked.connect(self.adjust_brightness)
        self.rotateButton.clicked.connect(self.rotate_image)
        self.resizeButton.clicked.connect(self.resize_image)

        self.brightButton_2.clicked.connect(self.roi_adjust_brightness)
        self.rotateButton_2.clicked.connect(self.roi_rotate_image)
        self.resizeButton_2.clicked.connect(self.roi_resize_image)

        self.saveButton.clicked.connect(self.save_image)
        self.roi_saveButton.clicked.connect(self.roi_save_image)

        self.roi_button.clicked.connect(self.toggle_roi_mode)
        self.free_roi_button.clicked.connect(self.free_roi_mode)
        self.sticker_button.clicked.connect(self.toggle_sticker_mode)
        self.point_button.clicked.connect(self.toggle_point_mode)

        self.gray_button.toggled.connect(self.gray_change)
        self.thres_Button.clicked.connect(self.thres_change)
        self.canny_Button.clicked.connect(self.canny_change)
        self.morph_Button.clicked.connect(self.morph_change)
        self.blur_Button.clicked.connect(self.blur_change)
        self.shar_Button.clicked.connect(self.shar_change)
        self.sticker_remove_button.clicked.connect(self.sticker_remove)

        self.graphicsView.mousePressEvent = self.onMousePress
        self.graphicsView.mouseMoveEvent = self.onMouseMove
        self.graphicsView.mouseReleaseEvent = self.onMouseRelease

        self.model = QStandardItemModel()  # 모델 초기화
        self.log_listView.setModel(self.model)  # ListView에 모델 할당

        self.add_log("요이땅")

    def load_image(self):
        self.modified2_image = None
        self.image_path, _ = QFileDialog.getOpenFileName(self, "사진 선택", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not os.path.isfile(self.image_path):
            #QMessageBox.warning(self, "Error", "파일 존재 안함.")
            #logging.info("파일이 없어요")
            self.add_log("파일이 없습니다.")
            return
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is not None:
            #self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.modified_image = self.original_image.copy()
            self.display_image(self.modified_image)
            # logging.info(f"이미지 로드 성공: {self.image_path}")
            self.add_log(f"이미지 로드 성공: {self.image_path}")
        else:
            self.add_log("이미지를 불러오는데 실패했습니다.")

    def show_image_load_popup(self):
        QMessageBox.warning(self, "이미지 로드", "이미지를 먼저 선택하세요.")


    def add_log(self, message):
        logging.info(message)  # 로그 메시지 파일에 기록
        item = QStandardItem(message)  # 로그 메시지 아이템 생성
        self.model.appendRow(item)  # 기존 모델에 아이템 추가
        self.log_listView.scrollToBottom()  # 리스트뷰 스크롤을 최하단으로 이동

    def display_image(self, image):
        image = np.ascontiguousarray(image)  # 배열을 C 연속형으로 변환
        qImg = QImage(sip.voidptr(image.data), image.shape[1], image.shape[0], image.strides[0], QImage.Format_BGR888)  # BGR 포맷 사용
        pixmap = QPixmap.fromImage(qImg) # QPixmap 객체 생성
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene.clear()
        self.scene2.clear()
        self.scene.addItem(pixmapItem)
        # QGraphicsScene 설정 및 이미지 맞춤 조정
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setSceneRect(pixmapItem.boundingRect()) #주어진 점을 감싸는 최소 크기 사각형(바운딩박스)를 반환
        self.graphicsView2.fitInView(pixmapItem, Qt.KeepAspectRatio)  # 이미지 크기를 View에 맞추어 조정

    def display_image2(self, image):
        image = np.ascontiguousarray(image)  # 배열을 C 연속형으로 변환
        qImg = QImage(sip.voidptr(image.data), image.shape[1], image.shape[0], image.strides[0], QImage.Format_BGR888)  # BGR 포맷 사용
        pixmap = QPixmap.fromImage(qImg)
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene2.clear()
        self.scene2.addItem(pixmapItem)
        self.graphicsView2.setScene(self.scene2)
        self.graphicsView2.setSceneRect(pixmapItem.boundingRect())


    def display_image3(self, image):
        image = np.ascontiguousarray(image)  # 배열을 C 연속형으로 변환
        qImg = QImage(sip.voidptr(image.data), image.shape[1], image.shape[0], image.strides[0], QImage.Format_BGR888)  # BGR 포맷 사용
        pixmap = QPixmap.fromImage(qImg)
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene2.clear()
        self.scene2.addItem(pixmapItem)
        self.graphicsView2.setScene(self.scene2)
        self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
        self.graphicsView2.fitInView(pixmapItem, Qt.KeepAspectRatio)  # 이미지 크기를 View에 맞추어 조정

    def display_free_image(self, image):
        qImg = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qImg)
        pixmapItem = QGraphicsPixmapItem(pixmap)
        self.scene2.clear()
        self.scene2.addItem(pixmapItem)
        self.graphicsView2.setScene(self.scene2)
        self.graphicsView2.setSceneRect(pixmapItem.boundingRect())

    def deactivate_all_buttons(self):
        self.roi_button.setStyleSheet("")
        self.sticker_button.setStyleSheet("")
        self.point_button.setStyleSheet("")
        self.isSelecting = False
        self.sticker_mode = False
        self.point_mode = False

    #########################  Mouse Event ###############################################################################################################################################################################################
    def onMousePress(self, event):
        scenePoint = self.graphicsView.mapToScene(event.pos())
        self.add_log(f"좌표: {scenePoint.x()}, {scenePoint.y()} ")
        if event.button() == Qt.LeftButton:
            if self.mode_free_roi:
                self.roi_path = QtGui.QPainterPath()
                self.roi_path.moveTo(scenePoint)
                self.roi_path_item = QGraphicsPathItem()
                self.roi_path_item.setPath(self.roi_path)
                self.roi_path_item.setPen(QtGui.QPen(Qt.red, 2))
                self.scene.addItem(self.roi_path_item)
            elif self.isSelecting:
                self.x0 = scenePoint.x()
                self.y0 = scenePoint.y()
                self.isDragging = True
                if not self.selectionRect:
                    self.selectionRect = self.scene.addRect(self.x0, self.y0, 0, 0, pen=QtGui.QPen(Qt.red, 2))
                else:
                    self.selectionRect.setRect(self.x0, self.y0, 0, 0)
            elif self.sticker_mode:
                self.add_log("sticker mode가 활성화 되었습니다.")
                self.place_sticker(scenePoint)
            elif self.point_mode:
                self.calculate_distance(scenePoint)
            else:
                self.add_log("ROI를 설정해주세요")

    def onMouseMove(self, event):
        scenePoint = self.graphicsView.mapToScene(event.pos())
        if event.buttons() == Qt.LeftButton:
            if self.mode_free_roi and self.roi_path_item and self.roi_path:
                self.roi_path.lineTo(scenePoint)
                self.roi_path_item.setPath(self.roi_path)
            elif self.isDragging and self.isSelecting:
                self.currentX = scenePoint.x()
                self.currentY = scenePoint.y()
                width = self.currentX - self.x0
                height = self.currentY - self.y0
                if self.selectionRect:
                    self.selectionRect.setRect(self.x0, self.y0, width, height)

    def onMouseRelease(self, event):
        if self.mode_free_roi and self.roi_path_item and self.roi_path:
            min_x, min_y, max_x, max_y = self.roi_path.boundingRect().getCoords()
            cropped_image = self.modified_image[int(min_y):int(max_y), int(min_x):int(max_x)]

            mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
            path_offset = QtGui.QPainterPath(self.roi_path)
            path_offset.translate(-min_x, -min_y)

            mask_image = QImage(mask.shape[1], mask.shape[0], QImage.Format_Grayscale8)
            mask_image.fill(0)

            painter = QtGui.QPainter(mask_image)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            painter.fillPath(path_offset, QtGui.QBrush(QtGui.QColor(255)))
            painter.end()

            ptr = mask_image.bits()
            ptr.setsize(mask_image.width() * mask_image.height())
            mask = np.array(ptr, dtype=np.uint8).reshape((mask_image.height(), mask_image.width()))

            cropped_rgba = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
            cropped_rgba[..., 3] = mask

            self.free_roi_image = cropped_rgba
            self.display_free_image(self.free_roi_image)

            self.mode_free_roi = False
            self.add_log("자유형 ROI 생성 완료")

            self.scene.removeItem(self.roi_path_item)
            self.roi_path = None
            self.roi_path_item = None
            self.update_button_styles()
            self.graphicsView.setCursor(Qt.ArrowCursor)
        elif self.isDragging and self.isSelecting:
            self.isDragging = False
            self.graphicsView.setCursor(Qt.ArrowCursor)
            self.currentPoint = self.graphicsView.mapToScene(event.pos())
            w = self.currentPoint.x() - self.x0
            h = self.currentPoint.y() - self.y0
            if w > 0 and h > 0:
                self.modified2_image = self.modified_image[int(self.y0):int(self.y0 + h), int(self.x0):int(self.x0 + w)]
                self.roi_selected = True
                self.display_image2(self.modified2_image)
            if self.selectionRect:
                self.scene.removeItem(self.selectionRect)
                self.selectionRect = None
                self.add_log('ROI가 선택되었습니다. - x:%d, y:%d, w:%d, h:%d' % (self.x0, self.y0, w, h))
                self.lineEdit_roi.setText(str('x:%d, y:%d, w:%d, h:%d' % (self.x0, self.y0, w, h)))

            self.isSelecting = False
            self.update_button_styles()
            self.add_log("ROI 모드 비활성화")

    def toggle_roi_mode(self):
        if self.modified_image is None:
            self.show_image_load_popup()
            return
        self.scene2.clear()
        self.isSelecting = not self.isSelecting
        self.update_button_styles()
        for ellipsis in self.ellipses:
            self.scene.removeItem(ellipsis)
        self.ellipses.clear()
        self.add_log("toggle_roi_mode callled")
        if self.isSelecting:
            self.add_log("ROI mode 활성화 됨")
            self.graphicsView.setCursor(Qt.CrossCursor)  # ROI 선택 모드 시 커서 변경
        else:
            self.add_log("ROI mode 비활성화 됨")
            self.graphicsView.setCursor(Qt.ArrowCursor)  # 커서를 기본 상태로 복원

    def free_roi_mode(self):
        if self.modified_image is None:
            self.show_image_load_popup()
            return

        self.mode_free_roi = not self.mode_free_roi
        self.update_button_styles()
        self.scene2.clear()

        for ellipse in self.ellipses:
            self.scene.removeItem(ellipse)
        self.ellipses.clear()

        if self.mode_free_roi:
            self.graphicsView.setCursor(Qt.CrossCursor)  # 커서를 교차 모양으로 변경
            self.add_log("자유형 ROI 모드 활성화됨")
        else:
            if self.roi_path_item:  # 활성화 상태를 끝내면서 경로가 있다면 제거
                self.scene.removeItem(self.roi_path_item)
                self.roi_path_item = None
                self.roi_path = None
            self.graphicsView.setCursor(Qt.ArrowCursor)  # 커서를 기본 상태로 복원
            self.add_log("자유형 ROI 모드 비활성화됨")


    # def free_roi_final(self):
    #     if self.roi_path:
    #         # RGBA 이미지 생성
    #         rgba_image = np.zeros((self.modified_image.shape[0], self.modified_image.shape[1], 4), dtype=np.uint8)
    #         rgba_image[..., :3] = self.modified_image  # RGB 채널에 원본 이미지 복사
    #         rgba_image[..., 3] = 0  # 알파 채널을 0 (완전 투명)으로 초기화
    #
    #         # QPainterPath에서 numpy 배열로 변환
    #         polygon = self.roi_path.toFillPolygon().toPolygon()
    #         points = np.array([[pt.x(), pt.y()] for pt in polygon], dtype=np.int32)
    #
    #         if len(points) > 0:
    #             mask = np.zeros((self.modified_image.shape[0], self.modified_image.shape[1]), dtype=np.uint8)
    #             cv2.fillPoly(mask, [points], 255)  # 마스크 내부를 255로 채움
    #             for i in range(3):  # RGB 채널에 마스크 적용
    #                 rgba_image[..., i] = cv2.bitwise_and(rgba_image[..., i], mask)
    #             rgba_image[..., 3] = mask  # 알파 채널에 마스크 적용
    #
    #             self.free_roi_image = rgba_image
    #             self.display_free_image(self.free_roi_image)
    #         else:
    #             self.add_log("자유형 ROI 생성 실패: 포인트 배열이 비어 있음")
    #
    #         self.mode_free_roi = False
    #         self.add_log("자유형 ROI 생성 완료")

    def toggle_sticker_mode(self):
        if self.modified_image is None:
            self.show_image_load_popup()
            return

        self.sticker_mode = not self.sticker_mode
        self.update_button_styles()

        for ellipsis in self.ellipses:
            self.scene.removeItem(ellipsis)
        self.ellipses.clear()

        if self.sticker_mode:
            self.add_log("Sticker mode 활성화")
            self.graphicsView.setCursor(Qt.CrossCursor)  # 커서 변경
        else:
            self.add_log("Sticker mode 비활성화")
            self.graphicsView.setCursor(Qt.ArrowCursor)  # 커서 변경


    def toggle_point_mode(self):
        for ellipse in self.ellipses:
            self.scene.removeItem(ellipse)
        self.ellipses.clear()
        if self.modified_image is None:
            self.show_image_load_popup()
            return
        self.deactivate_all_buttons()
        self.point_mode = True
        self.update_button_styles()
        self.add_log("측정시작")



    # def place_sticker(self, position):
    #     #sticker_image = self.free_roi_image if self.mode_free_roi else self.modified2_image
    #     if self.mode_free_roi:
    #         sticker_image = self.free_roi_image
    #     else:
    #         sticker_image = self.modified2_image
    #
    #     if sticker_image is not None:
    #         sticker_image = np.ascontiguousarray(sticker_image)  # 배열을 연속형으로 변환
    #         if self.mode_free_roi:
    #             qImg = QImage(sticker_image.data, sticker_image.shape[1], sticker_image.shape[0],
    #                           sticker_image.strides[0], QImage.Format_RGBA8888)  # RGBA 포맷 사용
    #         else:
    #             qImg = QImage(sticker_image.data, sticker_image.shape[1], sticker_image.shape[0],
    #                           sticker_image.strides[0], QImage.Format_BGR888)  # BGR 포맷 사용
    #
    #         pixmap = QPixmap.fromImage(qImg)
    #         pixmapItem = QGraphicsPixmapItem(pixmap)
    #         pixmapItem.setPos(position.x() - pixmap.width() / 2, position.y() - pixmap.height() / 2)
    #         self.scene.addItem(pixmapItem)
    #         self.add_log("스티커가 중심 좌표 : ({}, {})에 배치되었습니다.".format(position.x(), position.y()))


    def place_sticker(self, position):
        sticker_image = self.free_roi_image if self.free_roi_image is not None else self.modified2_image

        if sticker_image is not None:
            sticker_image = np.ascontiguousarray(sticker_image)  # 배열을 연속형으로 변환
            qImg = QImage(sticker_image.data, sticker_image.shape[1], sticker_image.shape[0],
                          sticker_image.strides[0],
                          QImage.Format_RGBA8888 if self.mode_free_roi else QImage.Format_BGR888)  # 포맷 선택
            pixmap = QPixmap.fromImage(qImg)
            pixmapItem = QGraphicsPixmapItem(pixmap)
            pixmapItem.setPos(position.x() - pixmap.width() / 2, position.y() - pixmap.height() / 2)
            self.scene.addItem(pixmapItem)
            self.add_log("스티커가 중심 좌표 : ({}, {})에 배치되었습니다.".format(position.x(), position.y()))
        else:
            self.add_log("Error: 스티커 이미지를 로드하지 못했습니다.")

    def sticker_remove(self):
        self.scene.clear()
        self.scene2.clear()
        self.modified2_image = None
        self.free_roi_image = None
        self.display_image(self.modified_image)

    def calculate_distance(self, point):
        self.point_lineEdit.clear()
        self.point2_textEdit.clear()
        self.points.append((point.x(), point.y()))
        radius = 5
        ellipse = self.scene.addEllipse(point.x() - radius, point.y() - radius, 2 * radius, 2 * radius,
                                        pen=QtGui.QPen(QtGui.QColor(0, 0, 255), 2),
                                        brush=QtGui.QBrush(QtGui.QColor(0, 0, 255)))
        self.ellipses.append(ellipse)

        if len(self.points) == 2:  # 수정: self.points1을 self.points로 변경
            p1, p2 = self.points
            x_distance = abs(p2[0] - p1[0])
            y_distance = abs(p2[1] - p1[1])
            distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
            self.point_lineEdit.setText(f"({p1[0]}, {p1[1]}) and ({p2[0]}, {p2[1]})")
            self.point2_textEdit.setText(f"총 거리: {distance:.2f}\nx축 거리: {x_distance:.2f}\nY축 거리: {y_distance:.2f}")

            self.add_log(f"거리 계산됨: 총 거리 {distance:.2f}, X축 {x_distance:.2f}, Y축 {y_distance:.2f}")
            self.add_log("측정끝")  # 측정 끝 로그 추가
            self.points = []  # 포인트 목록 초기화

            self.deactivate_all_buttons()
            self.update_button_styles()


    def update_button_styles(self):
        if self.isSelecting:
            self.roi_button.setStyleSheet("QPushButton { background-color: green; color: white; }")
        else:
            self.roi_button.setStyleSheet("")

        if self.mode_free_roi:
            self.free_roi_button.setStyleSheet("QPushButton { background-color: green; color: black; }")
        else:
            self.free_roi_button.setStyleSheet("")

        if self.sticker_mode:
            self.sticker_button.setStyleSheet("QPushButton { background-color: blue; color: white; }")
        else:
            self.sticker_button.setStyleSheet("")

        if self.point_mode:
            self.point_button.setStyleSheet("QPushButton { background-color: yellow; color: black; }")
        else:
            self.point_button.setStyleSheet("")



    def cv_image_to_qpixmap(self, cv_image):
        """OpenCV 이미지를 QPixmap으로 변환"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(cv_image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)




###########################  사진증강  #########################################################################################################################################################

    def adjust_brightness(self):
        try:
            factor = float(self.brightLineEdit.text())
            if self.modified_image is None:
                #logging.info("Error-이미지를 열지못했습니다.")
                #QMessageBox.warning(self, "Error", "이미지를 열지 못합니다.")
                self.add_log("Error-이미지를 열지못했습니다.")
                return
            self.modified2_image = cv2.convertScaleAbs(self.original_image, alpha=factor, beta=factor)
            self.display_image(self.modified2_image)

            self.lineEdit_br.setText(str(factor))
            #logging.info("%s 배 이미지 밝아짐",factor)
            self.add_log(f"이미지가 {factor}배 밝아짐")
        except ValueError:
            #logging.info("Error - 숫자를 입력좀 해요")
            self.add_log(("Error - 숫자를 입력좀 해요"))

    def rotate_image(self):
        try:
            angle = float(self.rotateLineEdit.text())  # 사용자로부터 입력받은 각도
            if self.modified_image is None:
                self.add_log("Error-이미지를 열지 못했습니다.")
                return

            # 이미지 중심을 계산
            image_center = tuple(np.array(self.modified_image.shape[1::-1]) / 2)

            # 회전 매트릭스 계산
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

            # 회전된 이미지의 새로운 크기를 계산
            abs_cos = abs(rot_mat[0, 0])
            abs_sin = abs(rot_mat[0, 1])
            bound_w = int(self.modified_image.shape[1] * abs_cos + self.modified_image.shape[0] * abs_sin)
            bound_h = int(self.modified_image.shape[1] * abs_sin + self.modified_image.shape[0] * abs_cos)

            # 회전 매트릭스 수정하여 이미지가 중앙에 위치하도록 조정
            rot_mat[0, 2] += bound_w / 2 - image_center[0]
            rot_mat[1, 2] += bound_h / 2 - image_center[1]

            # 이미지 회전 적용
            rotated_image = cv2.warpAffine(self.modified_image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)

            # 결과 이미지를 표시
            self.modified2_image = rotated_image
            self.display_image2(self.modified2_image)

            self.lineEdit_ro.setText(str(angle))
            self.add_log(f"이미지 회전: 각도 {angle}도")
        except ValueError:
            self.add_log("Error - 숫자를 입력좀 해요")

    def resize_image(self):
        try:
            # 사용자 입력 값을 실수로 변환
            scale_factor = float(self.resizeLineEdit.text())
            #logging.info("%s 배 이미지 확대",scale_factor)
            self.add_log(f"이미지가 {scale_factor}배 커짐")
        except ValueError:
            #logging.info("Error - 유효한 숫자를 입력해주세요.")
            self.add_log(f"숫자입력해")
            return

        if self.modified_image is None:
            #logging.info("Error - 이미지를 열지 못합니다.")
            self.add_log("Error - 이미지를 열지 못합니다.")
            return

        # 새로운 이미지 차원 계산
        new_dimensions = (int(self.original_image.shape[1] * scale_factor), int(self.original_image.shape[0] * scale_factor))
        self.modified2_image = cv2.resize(self.original_image, new_dimensions, interpolation=cv2.INTER_LINEAR)
        self.display_image2(self.modified2_image)
        self.lineEdit_re.setText(str(scale_factor))

#######################    roi 증강  ##################################################################################################################################################################
    def add_log(self, message):
        logging.info(message)  # 로그 메시지 파일에 기록
        item = QStandardItem(message)  # 로그 메시지 아이템 생성
        self.model.appendRow(item)  # 기존 모델에 아이템 추가
        self.log_listView.scrollToBottom()  # 리스트뷰 스크롤을 최하단으로 이동

    def roi_adjust_brightness(self):
        try:
            factor = float(self.brightLineEdit_2.text())
            if self.modified2_image is None:
                logging.info("Error-이미지를 열지못했습니다.")
                #QMessageBox.warning(self, "Error", "이미지를 열지 못합니다.")
                #self.add_log("Error-이미지를 열지못했습니다.")
                return
            self.modified2_image = cv2.convertScaleAbs(self.modified2_image, alpha=factor, beta=factor)
           # self.modified2_image = cv2.cvtColor(self.modified2_image, cv2.COLOR_BGR2RGB)
            self.display_image3(self.modified2_image)
            self.lineEdit_roi_br.setText(str(factor))
            #logging.info("%s 배 이미지 밝아짐",factor)
            self.add_log(f"이미지가 {factor}배 밝아짐")
        except ValueError:
            #logging.info("Error - 숫자를 입력좀 해요")
            self.add_log(("Error - 숫자를 입력좀 해요"))

    def roi_rotate_image(self):
        try:
            angle = float(self.rotateLineEdit_2.text())
            if self.modified2_image is None:
                #QMessageBox.warning(self, "Error", "이미지를 열지 못합니다.")
                #logging.info("Error: 이미지를 열지 못했습니다.")
                self.add_log("Error-이미지를 열지못했습니다.")
                return
            center = tuple(np.array(self.modified2_image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            # self.modified2_image = cv2.warpAffine(self.modified2_image, rot_mat, self.original_image.shape[1::-1], flags=cv2.INTER_LINEAR) #dsize - 원본이미지 사이즈로
            # self.modified2_image = cv2.warpAffine(self.modified2_image, rot_mat, (self.modified2_image.shape[1], self.modified2_image.shape[0]), flags=cv2.INTER_LINEAR) # dsize - roi 이미지로
            rotated_image = cv2.warpAffine(self.modified2_image, rot_mat,
                                           (self.modified2_image.shape[1], self.modified2_image.shape[0]),
                                           flags=cv2.INTER_LINEAR)
            self.modified2_image = rotated_image
          #  self.modified2_image = cv2.cvtColor(self.modified2_image, cv2.COLOR_BGR2RGB)
            self.display_image3(self.modified2_image)
            self.lineEdit_roi_ro.setText(str(angle))
            self.add_log(f"이미지 회전: 각도 {angle}도")
        except ValueError:
            logging.info("Error - 숫자 입력 좀 하자고오")
            self.add_log("Error - 숫자를 입력좀 해요")

    def roi_resize_image(self):
        try:
            # 사용자 입력 값을 실수로 변환
            scale_factor = float(self.resizeLineEdit_2.text())
            #logging.info("%s 배 이미지 확대",scale_factor)
            self.add_log(f"이미지가 {scale_factor}배 커짐")
        except ValueError:
            #logging.info("Error - 유효한 숫자를 입력해주세요.")
            self.add_log(f"숫자입력해")
            return

        if self.modified2_image is None:
            #logging.info("Error - 이미지를 열지 못합니다.")
            self.add_log("Error - 이미지를 열지 못합니다.")
            return

        # 새로운 이미지 차원 계산
        new_dimensions2 = (int(self.modified2_image.shape[1] * scale_factor), int(self.modified2_image.shape[0] * scale_factor))
        self.modified2_image = cv2.resize(self.modified2_image, new_dimensions2, interpolation=cv2.INTER_LINEAR)
       # self.modified2_image = cv2.cvtColor(self.modified2_image, cv2.COLOR_BGR2RGB)
        self.display_image3(self.modified2_image)
        self.lineEdit_roi_re.setText(str(scale_factor))

    def save_image(self):
        if self.scene is not None:
            # Scene의 크기를 가져옵니다.
            rect = self.scene.sceneRect()
            # QImage 객체를 생성합니다. 이 객체는 scene의 렌더링을 저장할 것입니다.
            image = QImage(rect.size().toSize(), QImage.Format_ARGB32)
            image.fill(Qt.white)  # 배경을 흰색으로 설정합니다.

            # QPainter 객체를 생성하고 QImage 객체에 scene을 렌더링합니다.
            painter = QPainter(image)
            self.scene.render(painter)
            painter.end()

            # QImage를 QPixmap으로 변환합니다.
            pixmap = QPixmap.fromImage(image)

            # QPixmap을 OpenCV 이미지로 변환합니다.
            qimage = pixmap.toImage()
            buffer = qimage.bits().asstring(qimage.byteCount())
            img = np.frombuffer(buffer, dtype=np.uint8).reshape((qimage.height(), qimage.width(), 4))

            # OpenCV를 사용하여 이미지를 저장합니다.
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
            if save_path:  # 사용자가 경로를 선택한 경우에만 이미지를 저장
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
                self.add_log("이미지가 성공적으로 저장되었습니다.")
        else:
            self.add_log("저장할 이미지가 없습니다.")
        # if self.modified_image is not None:
        #     save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "","PNG (*.png);;JPEG (*.jpg *.jpeg)")  # 사용자에게 저장할 경로 선택
        #     if save_path:  # 사용자가 경로를 선택한 경우에만 이미지를 저장
        #         cv2.imwrite(save_path, self.modified_image)  # OpenCV를 사용하여 이미지를 저장
        #         self.add_log("이미지가 성공적으로 저장되었습니다.")
        # else:
        #     self.add_log("저장할 이미지가 없습니다.")

    def roi_save_image(self):
        if self.modified2_image is not None:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "","PNG (*.png);;JPEG (*.jpg *.jpeg)")  # 사용자에게 저장할 경로 선택
            if save_path:  # 사용자가 경로를 선택한 경우에만 이미지를 저장
                cv2.imwrite(save_path, cv2.cvtColor(self.modified2_image, cv2.COLOR_RGB2BGR))  # OpenCV를 사용하여 이미지를 저장
                self.add_log("이미지가 성공적으로 저장되었습니다.")
        else:
            self.add_log("저장할 이미지가 없습니다.")

    #######################    image filter   ##################################################################################################################################################################

    def gray_change(self, checked):
        if checked:
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY) # 원본 이미지를 회색조로 변환
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB) # OpenCV의 회색조 이미지를 RGB 이미지로 변환 (QPixmap에 맞게 조정)
            qImg = QImage(gray_image.data, gray_image.shape[1], gray_image.shape[0], gray_image.strides[0], QImage.Format_RGB888) # QImage 객체 생성
            gray_pixmap = QPixmap.fromImage(qImg) # QPixmap으로 변환
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(gray_pixmap) # QGraphicsScene에 QPixmap 객체 추가
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
            self.add_log("Gray Image 생성 완료")
        else:
            self.display_image(self.modified_image) # 회색조 변경이 아닌 경우 원래 이미지로 되돌림

    def thres_change(self):
        if not self.gray_button.isChecked():
            self.add_log("Error-Please convert to grayscale first.")
            return

        try:
            threshold_value = int(self.thres_lineEdit.text())
            if not (0 <= threshold_value <= 255):
                self.add_log("Error-Threshold value must be between 0 and 255.")
                return

            # 회색조 이미지를 이진화 처리
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)  # 원본 이미지를 회색조로 변환
            _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)  # 이진화된 이미지를 RGB 이미지로 변환
            qImg = QImage(binary_image.data, binary_image.shape[1], binary_image.shape[0], binary_image.strides[0],
                          QImage.Format_RGB888)
            binary_pixmap = QPixmap.fromImage(qImg)
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(binary_pixmap)  # QGraphicsScene에 QPixmap 객체 추가
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
            self.add_log("2진화 성공: {}".format(threshold_value))
        except ValueError:
            self.add_log("Error-Invalid input. Please enter a valid integer for threshold.")

    def canny_change(self):
        if not self.gray_button.isChecked():
            self.add_log("Error-Please convert to grayscale first.")
            return

        try:
            canny_value = int(self.canny_lineEdit.text())
            if not (0 <= canny_value <= 255):
                self.add_log("Error-Threshold value must be between 0 and 255.")
                return

            # 회색조 이미지를 이용한 Canny 에지 검출
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)  # 원본 이미지를 회색조로 변환
            canny_image = cv2.Canny(gray_image, canny_value, canny_value * 2)  # Canny 함수로 에지 검출
            canny_image = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)  # 에지 검출된 이미지를 RGB 이미지로 변환
            qImg = QImage(canny_image.data, canny_image.shape[1], canny_image.shape[0], canny_image.strides[0], QImage.Format_RGB888)
            canny_pixmap = QPixmap.fromImage(qImg)
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(canny_pixmap)  # QGraphicsScene에 QPixmap 객체 추가
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
            self.add_log("canny 성공: {}".format(canny_value))
        except ValueError:
            self.add_log("Error-Invalid input. Please enter a valid integer for threshold.")

    def morph_change(self):
        if not self.gray_button.isChecked():
            self.add_log("Error-Please convert to grayscale first.")
            return

        try:
            morph_value = int(self.morph_lineEdit.text())
            if not (0 <= morph_value <= 255):
                self.add_log("Error-Threshold value must be between 0 and 255.")
                return

            # 회색조 이미지를 이진화 처리
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, morph_value, 255, cv2.THRESH_BINARY)

            # 모폴로지 연산 적용: 여기서는 예로 침식과 팽창을 사용
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5x5 크기의 사각형 커널
            eroded_image = cv2.erode(binary_image, kernel, iterations=1)  # 침식 연산
            dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)  # 팽창 연산

            # 결과 이미지를 RGB로 변환
            result_image = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB)
            qImg = QImage(result_image.data, result_image.shape[1], result_image.shape[0], result_image.strides[0], QImage.Format_RGB888)
            result_pixmap = QPixmap.fromImage(qImg)

            # 결과 표시
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(result_pixmap)
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
            self.add_log("morph 성공: {}".format(morph_value))
        except ValueError:
            self.add_log("Error-Invalid input. Please enter a valid integer for threshold.")

    def blur_change(self):
        if not self.gray_button.isChecked():
            self.add_log("Error-Please apply grayscale first.")
            return

        try:
            blur_amount = int(self.blur_lineEdit.text())
            if not (1 <= blur_amount <= 255):
                self.add_log("Error-Blur amount must be between 1 and 255.")
                return

            if blur_amount % 2 == 0:
                blur_amount += 1

            # 회색조 이미지로 블러 적용
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (blur_amount, blur_amount), 0)
            blurred_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2RGB)  # 다시 RGB로 변환하여 QGraphicsView에서 표시

            qImg = QImage(blurred_image.data, blurred_image.shape[1], blurred_image.shape[0], blurred_image.strides[0], QImage.Format_RGB888)
            blur_pixmap = QPixmap.fromImage(qImg)
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(blur_pixmap)  # QGraphicsScene에 QPixmap 객체 추가
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())

            self.add_log("Blur applied with intensity: {}".format(blur_amount))
        except ValueError:
            self.add_log("Error-Invalid input.")

    def shar_change(self):
        if not self.gray_button.isChecked():
            self.add_log("Error-Please apply grayscale first.")
            return

        try:
            shar_amount = int(self.shar_lineEdit.text())
            if not (0 <= shar_amount <= 255):
                self.add_log("Error-Sharpness amount must be between 0 and 255.")
                return

            # 회색조 이미지 가져오기
            gray_image = cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2GRAY)

            # 샤프닝 커널 정의
            b = (1 - shar_amount) / 8
            kernel = np.array([[b, b, b],
                               [b, shar_amount, b],
                               [b, b, b]])


            # 샤프닝 적용
            sharpened_image = cv2.filter2D(gray_image, -1, kernel)
            sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2RGB)  # RGB로 변환하여 QGraphicsView에서 표시

            # 이미지 표시를 위한 QImage 객체 생성
            qImg = QImage(sharpened_image.data, sharpened_image.shape[1], sharpened_image.shape[0], sharpened_image.strides[0], QImage.Format_RGB888)
            shar_pixmap = QPixmap.fromImage(qImg)
            self.scene2.clear()
            pixmapItem = QGraphicsPixmapItem(shar_pixmap)  # QGraphicsScene에 QPixmap 객체 추가
            self.scene2.addItem(pixmapItem)
            self.graphicsView2.setScene(self.scene2)
            self.graphicsView2.setSceneRect(pixmapItem.boundingRect())
            self.add_log("Sharpness applied with intensity: {}".format(shar_amount))
        except ValueError:
            self.add_log("Error-Invalid input. Please enter a valid integer for sharpness.")
