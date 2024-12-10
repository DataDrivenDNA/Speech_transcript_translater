from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizeGrip
from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtGui import QPalette, QColor, QFont

class CaptionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.Window |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool  # This prevents the window from showing in taskbar
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMinimumSize(200, 50)  # Set minimum size
        self.setup_ui()
        
        # For window dragging
        self.dragging = False
        self.offset = QPoint()
        self.resizing = False
        self.resize_offset = QPoint()
        
    def setup_ui(self):
        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(10, 5, 10, 5)
        
        # Text label
        self.text_label = QLabel()
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignCenter)
        
        # Style the label
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.text_label.setFont(font)
        self.text_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0, 0, 0, 0.7);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        
        # Size grip (resize handle)
        self.size_grip = QSizeGrip(self)
        self.size_grip.setStyleSheet("""
            QSizeGrip {
                background-color: rgba(255, 255, 255, 0.5);
                border-radius: 5px;
                width: 10px;
                height: 10px;
            }
            QSizeGrip:hover {
                background-color: rgba(255, 255, 255, 0.8);
            }
        """)
        
        # Add widgets to layout
        self.main_layout.addWidget(self.text_label)
        self.setLayout(self.main_layout)
        
        # Position the size grip
        self.size_grip.setFixedSize(15, 15)
        
        # Set a default size
        self.resize(400, 100)
        
    def update_text(self, text):
        """Update the caption text"""
        self.text_label.setText(text)
        
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.LeftButton:
            # Check if click is in the resize area (bottom-right corner)
            resize_area = 15  # Size of the resize area
            if (self.width() - event.pos().x() <= resize_area and 
                self.height() - event.pos().y() <= resize_area):
                self.resizing = True
                self.resize_offset = event.globalPos() - self.geometry().bottomRight()
            else:
                self.dragging = True
                self.offset = event.pos()

    def mouseMoveEvent(self, event):
        """Handle window dragging and resizing"""
        if self.resizing:
            new_size = event.globalPos() - self.pos() - self.resize_offset
            new_width = max(self.minimumWidth(), new_size.x())
            new_height = max(self.minimumHeight(), new_size.y())
            self.resize(new_width, new_height)
        elif self.dragging:
            self.move(event.globalPos() - self.offset)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for window dragging and resizing"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.resizing = False

    def resizeEvent(self, event):
        """Handle resize events"""
        super().resizeEvent(event)
        # Update size grip position
        self.size_grip.move(
            self.width() - self.size_grip.width() - 5,
            self.height() - self.size_grip.height() - 5
        )

    def enterEvent(self, event):
        """Show resize handle when mouse enters window"""
        self.size_grip.show()

    def leaveEvent(self, event):
        """Hide resize handle when mouse leaves window"""
        if not self.geometry().contains(self.mapFromGlobal(self.cursor().pos())):
            self.size_grip.hide()