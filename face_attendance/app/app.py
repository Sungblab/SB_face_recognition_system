from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QTableWidget, QTableWidgetItem, QFileDialog, QLineEdit,
    QListWidget, QListWidgetItem, QGridLayout, QScrollArea, QMessageBox, QCheckBox,
    QTabWidget, QHeaderView, QInputDialog, QComboBox, QGraphicsOpacityEffect
)
from PyQt5.QtCore import Qt, QTimer, QPoint, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import QSize, QParallelAnimationGroup
import sys
import cv2
import face_recognition
import datetime
import numpy as np
import csv
import os
import shutil
from PIL import Image

class AttendanceTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        # 카메라 및 얼굴인식 관련 변수 초기화
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        self.frame_count = 0  # frame_count 변수 추가

        # 얼굴 인식 데이터 초기화
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # 출석한 학생 목록
        self.present_students = set()
        self.update_absent_list()

    def initUI(self):
        # 메인 레이아웃을 수평 레이아웃으로 변경
        layout = QHBoxLayout()

        # notification_label 추가
        self.notification_label = QLabel()
        self.notification_label.setAlignment(Qt.AlignCenter)
        self.notification_label.hide()  # 초기에는 숨김
        self.notification_label.setStyleSheet("""
            QLabel {
                color: white;
                padding: 15px 30px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 16px;
                background-color: rgba(76, 175, 80, 0.95);
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
        """)
        layout.addWidget(self.notification_label)

        # 왼쪽 부분 (카메라 뷰 + 컨트롤)
        left_layout = QVBoxLayout()

        # 카메라 뷰 크기 확대
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(800, 600)  # 크기 증가
        self.camera_label.setStyleSheet("border: 2px solid #ccc; border-radius: 8px;")
        left_layout.addWidget(self.camera_label)

        # 컨트롤 버튼 컨테이너
        control_layout = QHBoxLayout()

        self.start_btn = QPushButton('출석 시작')
        self.start_btn.clicked.connect(self.start_attendance)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton('출석 종료')
        self.stop_btn.clicked.connect(self.stop_attendance)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        left_layout.addLayout(control_layout)

        # 상태 표시
        self.status_label = QLabel('시스템 대기 중...')
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #666;
                padding: 15px;
                background-color: white;
                border-radius: 8px;
                margin: 10px 0;
                font-size: 16px;
            }
        """)
        left_layout.addWidget(self.status_label)

        # 최근 인식된 얼굴 표시 크기 증가
        self.recent_face_label = QLabel()
        self.recent_face_label.setMinimumSize(250, 250)  # 크기 증가
        self.recent_face_label.setMaximumSize(250, 250)
        self.recent_face_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 8px;
                background-color: white;
            }
        """)
        self.recent_face_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.recent_face_label)

        layout.addLayout(left_layout, stretch=1)

        # 오른쪽 부분 (출석 기록)
        right_layout = QVBoxLayout()

        attendance_group = QGroupBox("출석 기록")
        attendance_layout = QVBoxLayout()

        self.attendance_table = QTableWidget()
        self.attendance_table.setColumnCount(4)
        self.attendance_table.setHorizontalHeaderLabels(['이름', '시간', '상태', '사진'])
        self.attendance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        attendance_layout.addWidget(self.attendance_table)

        # 출석 기록 내보내기 버튼
        self.export_btn = QPushButton('출석 기록 내보내기')
        self.export_btn.clicked.connect(self.export_attendance)
        attendance_layout.addWidget(self.export_btn)

        attendance_group.setLayout(attendance_layout)
        right_layout.addWidget(attendance_group)

        # 미출석자 목록 추가
        absent_group = QGroupBox("미출석자 목록")
        absent_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-top: 1ex;
                padding: 10px;
                background: white;
            }
            QGroupBox::title {
                color: #424242;
            }
        """)
        absent_layout = QVBoxLayout()

        self.absent_list = QListWidget()
        self.absent_list.setStyleSheet("""
            QListWidget {
                border: none;
                background: white;
            }
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border-radius: 5px;
                background: #ffebee;
                color: #d32f2f;
            }
        """)
        absent_layout.addWidget(self.absent_list)
        absent_group.setLayout(absent_layout)
        right_layout.addWidget(absent_group)

        # 시간대별 출석 현황 그래프 추가
        chart_group = QGroupBox("시간대별 출석 현황")
        chart_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                margin-top: 1ex;
                padding: 10px;
                background: white;
            }
            QGroupBox::title {
                color: #424242;
            }
        """)
        chart_layout = QVBoxLayout()
        self.chart_view = QLabel("그래프 영역")  # 실제로는 matplotlib 등을 사용하여 그래프 표시
        chart_layout.addWidget(self.chart_view)
        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)

        # 출석 필터 추가
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["전체", "출석", "미출석"])
        self.filter_combo.setStyleSheet("""
            QComboBox {
                padding: 5px 10px;
                border: 2px solid #e0e0e0;
                border-radius: 15px;
                background: white;
                min-width: 100px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        self.filter_combo.currentTextChanged.connect(self.filter_attendance)
        filter_layout.addWidget(QLabel("출석 상태:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addStretch()
        attendance_layout.insertLayout(1, filter_layout)

        # 실시간 알림 개선
        self.notification_label.setStyleSheet("""
            QLabel {
                color: white;
                padding: 15px 30px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 16px;
                background-color: rgba(76, 175, 80, 0.95);
                border: 2px solid rgba(255, 255, 255, 0.3);
            }
        """)

        layout.addLayout(right_layout, stretch=1)

        self.setLayout(layout)

    def load_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []

        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            return

        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                try:
                    image_path = os.path.join(faces_dir, filename)
                    # PIL로 이미지 읽기
                    face_image = Image.open(image_path)
                    # numpy 배열로 변환
                    face_image_np = np.array(face_image)
                    # face_recognition 라이브러리 사용
                    face_encodings = face_recognition.face_encodings(face_image_np)

                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        name = os.path.splitext(filename)[0]

                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                    else:
                        print(f"경고: {filename}에서 얼굴을 찾을 수 없습니다.")

                except Exception as e:
                    print(f"얼굴 로딩 중 오류 발생 ({filename}): {str(e)}")

    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            # 프레임 처리 최적화
            process_this_frame = self.frame_count % 3 == 0  # 3프레임마다 처리
            self.frame_count += 1

            if process_this_frame:
                # 프레임 크기 조정 최적화
                height, width = frame.shape[:2]
                small_frame = cv2.resize(frame, (width//4, height//4))
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                try:
                    # 얼굴 인식 처리
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # CPU 최적화

                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            if len(self.known_face_encodings) > 0:
                                # 벡터화된 연산으로 최적화
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)

                                if face_distances[best_match_index] < 0.6:
                                    name = self.known_face_names[best_match_index]
                                    self.record_attendance(name)
                                    self.show_notification(name)  # 알림 표시
                                else:
                                    name = "미등록"
                            else:
                                name = "미등록"

                            # 좌표 변환 및 박스 그리기
                            scale = 4
                            cv2.rectangle(frame, 
                                        (left*scale, top*scale), 
                                        (right*scale, bottom*scale), 
                                        (0, 255, 0), 3)  # 박스 두께 증가

                            # 텍스트 표시 최적화
                            y_position = bottom*scale + 30
                            cv2.rectangle(frame,
                                        (left*scale, y_position-30),
                                        (right*scale, y_position+10),
                                        (0, 255, 0), cv2.FILLED)

                            # OpenCV putText로 변경 (성능 향상)
                            cv2.putText(frame, name,
                                      (left*scale + 10, y_position),
                                      cv2.FONT_HERSHEY_DUPLEX, 0.8,  # 글자 크기 증가
                                      (255, 255, 255), 2)  # 글자 두께 증가

                except Exception as e:
                    print(f"프레임 처리 중 오류 발생: {str(e)}")

            # OpenCV 프레임을 Qt 이미지로 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_image = qt_image.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(QPixmap.fromImage(scaled_image))

    def record_attendance(self, name):
        try:
            current_time = datetime.datetime.now()
            time_string = current_time.strftime('%H:%M:%S')

            # 최근 출석 확인을 위한 시간 체크 (5분 이내 중복 방지)
            if name in self.present_students:
                return

            # 얼굴 이미지 처리
            faces_dir = "faces"
            image_path = os.path.join(faces_dir, f"{name}.jpg")
            if os.path.exists(image_path):
                # 최근 인식된 얼굴 표시 - 원본 이미지 사용
                pixmap = QPixmap(image_path)
                pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.recent_face_label.setPixmap(pixmap)

                # 상태 메시지 업데이트
                self.status_label.setStyleSheet("""
                    QLabel {
                        color: #4CAF50;
                        padding: 15px;
                        background-color: #E8F5E9;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 16px;
                    }
                """)
                self.status_label.setText(f'✓ {name}님 출석이 확인되었습니다.')

                # 테이블에 기록 추가 - 원본 이미지 사용
                table_pixmap = QPixmap(image_path)
                table_pixmap = table_pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label = QLabel()
                image_label.setPixmap(table_pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                image_label.setStyleSheet("""
                    QLabel {
                        border-radius: 40px;
                        padding: 2px;
                        background-color: white;
                    }
                """)

                row_position = self.attendance_table.rowCount()
                self.attendance_table.insertRow(row_position)

                # 테이블 아이템 설정
                name_item = QTableWidgetItem(name)
                time_item = QTableWidgetItem(time_string)
                status_item = QTableWidgetItem("출석")

                # 아이템 정렬 및 스타일 설정
                for item in [name_item, time_item, status_item]:
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFont(QFont("Arial", 12))

                self.attendance_table.setItem(row_position, 0, name_item)
                self.attendance_table.setItem(row_position, 1, time_item)
                self.attendance_table.setItem(row_position, 2, status_item)
                self.attendance_table.setCellWidget(row_position, 3, image_label)

                # CSV 기록
                self.save_attendance_record(name, time_string)

                # 출석 상태 업데이트
                self.present_students.add(name)
                self.update_absent_list()

        except Exception as e:
            print(f"출석 기록 중 오류 발생: {str(e)}")
            self.status_label.setText(f'오류 발생: {str(e)}')
            self.status_label.setStyleSheet("color: #F44336; padding: 10px; font-size: 16px;")

    def save_attendance_record(self, name, time_string):
        """출석 기록을 CSV 파일에 저장하는 별도의 메서드"""
        date_string = datetime.datetime.now().strftime('%Y-%m-%d')
        filename = f'attendance_{date_string}.csv'

        # 파일이 없으면 헤더 추가
        if not os.path.exists(filename):
            with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['이름', '시간', '상태'])

        # 출석 기록 추가
        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([name, time_string, "출석"])

    def export_attendance(self):
        date_string = datetime.datetime.now().strftime('%Y-%m-%d')
        file_name, _ = QFileDialog.getSaveFileName(self, "출석 기록 저장",
                                                 f"attendance_{date_string}.csv",
                                                 "CSV files (*.csv)")
        if file_name:
            try:
                with open(file_name, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(['이름', '시간', '상태'])

                    for row in range(self.attendance_table.rowCount()):
                        name = self.attendance_table.item(row, 0).text()
                        time = self.attendance_table.item(row, 1).text()
                        status = self.attendance_table.item(row, 2).text()
                        writer.writerow([name, time, status])

                QMessageBox.information(self, '내보내기 완료', f'출석 기록이 {file_name}에 저장되었습니다.')
            except Exception as e:
                QMessageBox.warning(self, '오류', f'출석 기록 내보내기 중 오류가 발생했습니다: {str(e)}')

    def start_attendance(self):
        if not self.is_running:
            # 여러 카메라 인덱스 시도
            for i in range(2):  # 0과 1 시도
                self.camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow 사용
                if self.camera.isOpened():
                    # 카메라 설정
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.camera.set(cv2.CAP_PROP_FPS, 30)
                    break

            if not self.camera.isOpened():
                QMessageBox.warning(self, '오류', '카메라를 열 수 없니다.')
                return

            self.timer.start(30)  # 30ms 간격으로 프레임 업데이트
            self.is_running = True
            self.status_label.setText('출석 확인 중...')
            self.status_label.setStyleSheet("color: #4CAF50; padding: 10px; font-size: 16px;")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

    def stop_attendance(self):
        if self.is_running:
            self.timer.stop()
            self.camera.release()
            self.is_running = False
            self.camera_label.clear()
            self.status_label.setText('시스템 중지됨')
            self.status_label.setStyleSheet("color: #F44336; padding: 10px; font-size: 16px;")
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def filter_attendance(self, filter_status):
        """출석 상태별 필터링"""
        for row in range(self.attendance_table.rowCount()):
            status = self.attendance_table.item(row, 2).text()
            should_show = (
                filter_status == "전체" or
                (filter_status == "출석" and status == "출석") or
                (filter_status == "미출석" and status == "미출석")
            )
            self.attendance_table.setRowHidden(row, not should_show)

    def update_absent_list(self):
        """미출석자 목록 업데이트"""
        self.absent_list.clear()
        present_names = self.present_students

        faces_dir = "faces"
        if os.path.exists(faces_dir):
            for filename in os.listdir(faces_dir):
                if filename.endswith((".jpg", ".jpeg", ".png")):
                    name = os.path.splitext(filename)[0]
                    if name not in present_names:
                        item = QListWidgetItem(f"⚠ {name}")
                        self.absent_list.addItem(item)

    def update_chart(self):
        """시간대별 출석 현황 그래프 업데이트"""
        # matplotlib을 사용하여 그래프 생성
        # 시간대별 출석 현황을 시각화
        pass

    def show_notification(self, name):
        """개선된 알림 효과"""
        self.notification_label.setText(f"✓ {name}님이 출석했습니다")

        # 화면 중앙 상단에 표시
        pos = self.mapToGlobal(QPoint(0, 0))
        notification_x = pos.x() + (self.width() - self.notification_label.width()) // 2
        notification_y = pos.y() + 20

        # 애니메이션 효과
        self.notification_label.move(notification_x, notification_y - 50)  # 시작 위치
        self.notification_label.show()

        # 슬라이드 다운 + 페이드 인 효과
        pos_anim = QPropertyAnimation(self.notification_label, b"pos")
        pos_anim.setDuration(500)
        pos_anim.setStartValue(QPoint(notification_x, notification_y - 50))
        pos_anim.setEndValue(QPoint(notification_x, notification_y))
        pos_anim.setEasingCurve(QEasingCurve.OutBack)

        effect = QGraphicsOpacityEffect(self.notification_label)
        self.notification_label.setGraphicsEffect(effect)

        opacity_anim = QPropertyAnimation(effect, b"opacity")
        opacity_anim.setDuration(500)
        opacity_anim.setStartValue(0)
        opacity_anim.setEndValue(1)

        # 애니메이션 그룹
        group = QParallelAnimationGroup()
        group.addAnimation(pos_anim)
        group.addAnimation(opacity_anim)
        group.start()

        # 3초 후 숨기기
        QTimer.singleShot(3000, self.hide_notification)

    def hide_notification(self):
        """알림을 숨기는 메서드"""
        # 페이드 아웃 효과
        effect = self.notification_label.graphicsEffect()
        opacity_anim = QPropertyAnimation(effect, b"opacity")
        opacity_anim.setDuration(500)
        opacity_anim.setStartValue(1)
        opacity_anim.setEndValue(0)
        opacity_anim.finished.connect(self.notification_label.hide)
        opacity_anim.start()


class ManagementTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_page = 0
        self.items_per_page = 20  # 페이지당 표시할 얼굴 수
        self.initUI()
        self.load_known_faces()

    def initUI(self):
        layout = QVBoxLayout()

        # 검색 바 추가
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("이름으로 검색...")
        self.search_bar.textChanged.connect(self.update_face_grid)
        search_layout.addWidget(QLabel("검색:"))
        search_layout.addWidget(self.search_bar)
        layout.addLayout(search_layout)

        # 얼굴 목록을 스크롤 가능한 그리드로 변경
        face_group = QGroupBox("등록된 얼굴 목록")
        face_layout = QVBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        self.scroll_area.setWidget(self.grid_widget)

        face_layout.addWidget(self.scroll_area)
        face_group.setLayout(face_layout)
        layout.addWidget(face_group)

        # 페이지 네게이션 추가
        pagination_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton("이전 페이지")
        self.prev_page_btn.clicked.connect(self.prev_page)
        self.next_page_btn = QPushButton("다음 페이지")
        self.next_page_btn.clicked.connect(self.next_page)
        pagination_layout.addWidget(self.prev_page_btn)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.next_page_btn)
        layout.addLayout(pagination_layout)

        # 얼굴 등록 및 삭제 버튼
        button_layout = QHBoxLayout()
        self.register_btn = QPushButton('얼굴 등록')
        self.register_btn.clicked.connect(self.register_face)
        self.bulk_register_btn = QPushButton('대량 등록')
        self.bulk_register_btn.clicked.connect(self.bulk_register_faces)
        self.delete_face_btn = QPushButton('선택한 얼굴 삭제')
        self.delete_face_btn.clicked.connect(self.delete_faces)
        
        button_layout.addWidget(self.register_btn)
        button_layout.addWidget(self.bulk_register_btn)
        button_layout.addWidget(self.delete_face_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def load_known_faces(self):
        self.known_faces = []
        faces_dir = "faces"
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            return

        for filename in os.listdir(faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(faces_dir, filename)
                self.known_faces.append((name, image_path))

        self.update_face_grid()

    def update_face_grid(self):
        # 그리드 초기화
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # 검색 필터 적용
        search_text = self.search_bar.text().lower()
        filtered_faces = [
            face for face in self.known_faces 
            if search_text in face[0].lower()
        ]

        # 페이지네이션 적용
        total_pages = (len(filtered_faces) - 1) // self.items_per_page + 1
        self.current_page = max(0, min(self.current_page, total_pages - 1))

        start_index = self.current_page * self.items_per_page
        end_index = start_index + self.items_per_page
        page_faces = filtered_faces[start_index:end_index]

        # 그리드에 얼굴 추가
        columns = 4
        for idx, (name, image_path) in enumerate(page_faces):
            row = idx // columns
            col = idx % columns
            face_widget = self.create_face_widget(name, image_path)
            self.grid_layout.addWidget(face_widget, row, col)

    def create_face_widget(self, name, image_path):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)

        # 얼굴 이미지 처리 수정
        image = Image.open(image_path)
        image = image.convert('RGB')  # RGB 모드로 변환
        image = image.resize((100, 100), Image.Resampling.LANCZOS)
        
        # PIL 이미지를 QPixmap으로 변환
        img_data = image.tobytes('raw', 'RGB')
        qimg = QImage(img_data, image.width, image.height, image.width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setFixedSize(110, 110)
        image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 3px;
                background-color: white;
            }
        """)
        layout.addWidget(image_label)

        # 이름 레이블
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-size: 12px;
                margin-top: 5px;
            }
        """)
        layout.addWidget(name_label)

        # 체크박스 추가
        checkbox = QCheckBox()
        layout.addWidget(checkbox, alignment=Qt.AlignCenter)

        widget.setLayout(layout)
        widget.setProperty('name', name)
        return widget

    def register_face(self):
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self, "얼굴 이미지 선택", "", 
                "Image files (*.jpg *.jpeg *.png)"
            )
            if file_name:
                name, ok = QInputDialog.getText(
                    self, '이름 입력', '등록할 사람의 이름을 입력하세요:'
                )
                if ok and name:
                    faces_dir = "faces"
                    if not os.path.exists(faces_dir):
                        os.makedirs(faces_dir)

                    # 파일 이름 중복 방지
                    new_path = os.path.join(faces_dir, f"{name}.jpg")
                    counter = 1
                    while os.path.exists(new_path):
                        new_path = os.path.join(faces_dir, f"{name}_{counter}.jpg")
                        counter += 1

                    # 이미지 복사 및 저장
                    shutil.copyfile(file_name, new_path)

                    self.known_faces.append((name, new_path))
                    self.update_face_grid()
                    self.load_known_faces()

                    QMessageBox.information(
                        self, '등록 완료', f'{name}의 얼굴이 등록되었습니다.'
                    )

        except Exception as e:
            QMessageBox.warning(
                self, '오류', f'얼굴 등록 중 오류가 발생했습니다: {str(e)}'
            )

    def delete_faces(self):
        selected_faces = []
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    name = widget.property('name')
                    selected_faces.append(name)

        if not selected_faces:
            QMessageBox.information(self, '삭제할 얼굴 없음', '삭제할 얼굴을 선택하세요.')
            return

        reply = QMessageBox.question(
            self, '확인', 
            f'선택한 {len(selected_faces)}명의 얼굴을 ��제하시겠습니까?',
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            faces_dir = "faces"
            for name in selected_faces:
                # 파일 경로 찾기
                for face in self.known_faces:
                    if face[0] == name:
                        file_path = face[1]
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        self.known_faces.remove(face)
                        break
            self.update_face_grid()
            self.load_known_faces()
            QMessageBox.information(self, '삭제 완료', '선택한 얼굴이 삭제되었습니다.')

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_face_grid()

    def next_page(self):
        total_faces = len([
            face for face in self.known_faces 
            if self.search_bar.text().lower() in face[0].lower()
        ])
        total_pages = (total_faces - 1) // self.items_per_page + 1
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_face_grid()

    def bulk_register_faces(self):
        try:
            # 폴더 선택 다이얼로그
            folder_path = QFileDialog.getExistingDirectory(
                self, "이미지가 있는 폴더 선택"
            )
            
            if folder_path:
                # 결과 저장을 위한 변수들
                success_count = 0
                skip_count = 0
                fail_count = 0
                error_files = []
                
                # faces 디렉토리 확인
                faces_dir = "faces"
                if not os.path.exists(faces_dir):
                    os.makedirs(faces_dir)
                
                # 진행 상황을 보여줄 메시지 박스
                progress = QMessageBox()
                progress.setWindowTitle("등록 진행 중")
                progress.setText("얼굴 등록을 진행하고 있습니다...")
                progress.show()
                
                # 지원하는 이미지 확장자
                valid_extensions = ('.jpg', '.jpeg', '.png')
                
                # 폴더 내의 모든 이미지 파일 처리
                for filename in os.listdir(folder_path):
                    if filename.lower().endswith(valid_extensions):
                        try:
                            # 파일 경로
                            file_path = os.path.join(folder_path, filename)
                            
                            # 이름 추출 (확장자 제외)
                            name = os.path.splitext(filename)[0]
                            
                            # 이미 존재하는 얼굴인지 확인
                            existing_path = os.path.join(faces_dir, f"{name}.jpg")
                            if os.path.exists(existing_path):
                                skip_count += 1
                                continue
                            
                            # 얼굴 인식 확인
                            image = face_recognition.load_image_file(file_path)
                            face_locations = face_recognition.face_locations(image)
                            
                            if len(face_locations) > 0:
                                # 이미지를 faces 폴더에 복사
                                new_path = os.path.join(faces_dir, f"{name}.jpg")
                                
                                # PIL을 사용하여 이미지 처리 및 저장
                                img = Image.open(file_path)
                                img = img.convert('RGB')
                                img.save(new_path, 'JPEG', quality=95)
                                
                                self.known_faces.append((name, new_path))
                                success_count += 1
                            else:
                                fail_count += 1
                                error_files.append(f"{filename} (얼굴 감지 실패)")
                                
                        except Exception as e:
                            fail_count += 1
                            error_files.append(f"{filename} (오류: {str(e)})")
                
                # 진행 상황 메시지 박스 닫기
                progress.close()
                
                # 결과 업데이트
                self.update_face_grid()
                self.load_known_faces()
                
                # 결과 메시지 생성
                result_message = f"등록 완료:\n\n성공: {success_count}개\n건너뜀: {skip_count}개\n실패: {fail_count}개"
                if error_files:
                    result_message += "\n\n실패한 파일들:\n" + "\n".join(error_files)
                
                # 결과 표시
                QMessageBox.information(self, '대량 등록 완료', result_message)
                
        except Exception as e:
            QMessageBox.warning(
                self, '오류',
                f'대량 등록 중 오류가 발생했습니다: {str(e)}'
            )


class AttendanceSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # 전체화면으로 시작
        self.showMaximized()

    def initUI(self):
        self.setWindowTitle('얼굴인식 출결 시스템')
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#deleteButton {
                background-color: #f44336;
            }
            QPushButton#deleteButton:hover {
                background-color: #d32f2f;
            }
            QLabel {
                font-size: 14px;
            }
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: white;
            }
            QTableWidget::item {
                padding: 8px;
            }
        """)

        # 메인 위젯과 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 탭 위젯
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.West)

        # 출결 탭
        self.attendance_tab = AttendanceTab()
        tabs.addTab(self.attendance_tab, QIcon(), "출결")

        # 관리 탭
        self.management_tab = ManagementTab()
        tabs.addTab(self.management_tab, QIcon(), "관리")

        main_layout.addWidget(tabs)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AttendanceSystem()
    ex.show()
    sys.exit(app.exec_())
