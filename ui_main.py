from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 800)
        # Central widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Main vertical layout
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")

        # Title
        self.titleLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.titleLabel.setFont(font)
        self.titleLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.titleLabel.setObjectName("titleLabel")
        self.verticalLayout.addWidget(self.titleLabel)

        # Settings Section
        self.settingsGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.settingsGroupBox.setTitle("Settings")
        self.settingsGroupBox.setObjectName("settingsGroupBox")
        self.settingsLayout = QtWidgets.QFormLayout(self.settingsGroupBox)
        self.settingsLayout.setObjectName("settingsLayout")

        # Language Selection
        self.languageComboBox = QtWidgets.QComboBox(self.settingsGroupBox)
        self.languageComboBox.setObjectName("languageComboBox")
        self.languageComboBox.setAccessibleName("Language Selection")
        self.languageComboBox.setToolTip("Select the source language for transcription.")
        self.languageComboBox.addItems([
            "Auto Detect",  # First option for auto language detection
            "English",
            "Chinese",
            "German",
            "Spanish",
            "Russian",
            "Korean",
            "French",
            "Japanese",
            "Portuguese",
            "Turkish",
            "Polish",
            "Catalan",
            "Dutch",
            "Arabic",
            "Swedish",
            "Italian",
            "Indonesian",
            "Hindi",
            "Finnish",
            "Vietnamese",
            "Hebrew",
            "Ukrainian",
            "Greek",
            "Malay",
            "Czech",
            "Romanian",
            "Danish",
            "Hungarian",
            "Tamil",
            "Norwegian",
            "Thai",
            "Urdu",
            "Croatian",
            "Bulgarian",
            "Lithuanian",
            "Latin",
            "Maori",
            "Malayalam",
            "Welsh",
            "Slovak",
            "Telugu",
            "Persian",
            "Latvian",
            "Bengali",
            "Serbian",
            "Azerbaijani",
            "Slovenian",
            "Kannada",
            "Estonian",
            "Macedonian",
            "Breton",
            "Basque",
            "Icelandic",
            "Armenian",
            "Nepali",
            "Mongolian",
            "Bosnian",
            "Kazakh",
            "Albanian",
            "Swahili",
            "Galician",
            "Marathi",
            "Punjabi",
            "Sinhala",
            "Khmer",
            "Shona",
            "Yoruba",
            "Somali",
            "Afrikaans",
            "Occitan",
            "Georgian",
            "Belarusian",
            "Tajik",
            "Sindhi",
            "Gujarati",
            "Amharic",
            "Yiddish",
            "Lao",
            "Uzbek",
            "Faroese",
            "Haitian Creole",
            "Pashto",
            "Turkmen",
            "Nynorsk",
            "Maltese",
            "Sanskrit",
            "Luxembourgish",
            "Myanmar",
            "Tibetan",
            "Tagalog",
            "Malagasy",
            "Assamese",
            "Tatar",
            "Hawaiian",
            "Lingala",
            "Hausa",
            "Bashkir",
            "Javanese",
            "Sundanese",
            "Cantonese",
            "Burmese",
            "Valencian",
            "Flemish",
            "Haitian",
            "Letzeburgesch",
            "Pushto",
            "Panjabi",
            "Moldavian",
            "Moldovan",
            "Sinhalese",
            "Castilian",
            "Mandarin"
        ])
        self.settingsLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, QtWidgets.QLabel("Language:"))
        self.settingsLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.languageComboBox)

        # Chunk Count Label
        self.chunkCountLabel = QtWidgets.QLabel(self.settingsGroupBox)
        self.chunkCountLabel.setObjectName("chunkCountLabel")
        self.chunkCountLabel.setText("Segments in Queue: 0")
        self.settingsLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, QtWidgets.QLabel("Segments in Queue:"))
        self.settingsLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.chunkCountLabel)

        # VAD Window Size
        self.vadWindowSizeSpinBox = QtWidgets.QDoubleSpinBox(self.settingsGroupBox)
        self.vadWindowSizeSpinBox.setObjectName("vadWindowSizeSpinBox")
        self.vadWindowSizeSpinBox.setDecimals(1)
        self.vadWindowSizeSpinBox.setSingleStep(0.1)
        self.vadWindowSizeSpinBox.setRange(0.5, 5.0)
        self.vadWindowSizeSpinBox.setValue(1.0)  # Default value
        self.vadWindowSizeSpinBox.setToolTip("Set the VAD window size in seconds.")
        self.settingsLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, QtWidgets.QLabel("VAD Window Size (seconds):"))
        self.settingsLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.vadWindowSizeSpinBox)

        # VAD Hop Size
        self.vadHopSizeSpinBox = QtWidgets.QDoubleSpinBox(self.settingsGroupBox)
        self.vadHopSizeSpinBox.setObjectName("vadHopSizeSpinBox")
        self.vadHopSizeSpinBox.setDecimals(1)
        self.vadHopSizeSpinBox.setSingleStep(0.1)
        self.vadHopSizeSpinBox.setRange(0.1, 2.5)
        self.vadHopSizeSpinBox.setValue(0.5)  # Default value
        self.vadHopSizeSpinBox.setToolTip("Set the VAD hop size in seconds.")
        self.settingsLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, QtWidgets.QLabel("VAD Hop Size (seconds):"))
        self.settingsLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.vadHopSizeSpinBox)

        # No Speech Threshold
        self.noSpeechThresholdSpinBox = QtWidgets.QDoubleSpinBox(self.settingsGroupBox)
        self.noSpeechThresholdSpinBox.setObjectName("noSpeechThresholdSpinBox")
        self.noSpeechThresholdSpinBox.setDecimals(1)
        self.noSpeechThresholdSpinBox.setSingleStep(0.1)
        self.noSpeechThresholdSpinBox.setRange(0.1, 1.0)
        self.noSpeechThresholdSpinBox.setValue(0.5)  # Default value
        self.noSpeechThresholdSpinBox.setToolTip("Adjust the sensitivity for detecting the end of speech. Lower values make VAD less sensitive, higher values make it more sensitive.")
        self.settingsLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, QtWidgets.QLabel("No Speech Threshold:"))
        self.settingsLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.noSpeechThresholdSpinBox)

        self.verticalLayout.addWidget(self.settingsGroupBox)

        # Buttons layout
        self.buttonsLayout = QtWidgets.QHBoxLayout()
        self.buttonsLayout.setObjectName("buttonsLayout")

        # Start Button
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setObjectName("startButton")
        self.startButton.setAccessibleName("Start Transcription")
        self.startButton.setToolTip("Start real-time transcription and translation.")
        self.buttonsLayout.addWidget(self.startButton)

        # Stop Button
        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setObjectName("stopButton")
        self.stopButton.setAccessibleName("Stop Transcription")
        self.stopButton.setToolTip("Stop real-time transcription and translation.")
        self.buttonsLayout.addWidget(self.stopButton)

        self.verticalLayout.addLayout(self.buttonsLayout)

        # Transcription Area
        self.transcriptionLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.transcriptionLabel.setFont(font)
        self.transcriptionLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.transcriptionLabel.setObjectName("transcriptionLabel")
        self.verticalLayout.addWidget(self.transcriptionLabel)

        self.transcriptionArea = QtWidgets.QTextEdit(self.centralwidget)
        self.transcriptionArea.setObjectName("transcriptionArea")
        self.transcriptionArea.setStyleSheet("background-color: #f0f0f0;")
        self.verticalLayout.addWidget(self.transcriptionArea)

        # Translation Area
        self.translationLabel = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.translationLabel.setFont(font)
        self.translationLabel.setAlignment(QtCore.Qt.AlignLeft)
        self.translationLabel.setObjectName("translationLabel")
        self.verticalLayout.addWidget(self.translationLabel)

        self.translationArea = QtWidgets.QTextEdit(self.centralwidget)
        self.translationArea.setObjectName("translationArea")
        self.translationArea.setStyleSheet("background-color: #f0f0f0;")
        self.verticalLayout.addWidget(self.translationArea)

        # Set central widget
        MainWindow.setCentralWidget(self.centralwidget)

        # Retranslate UI
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # Accessibility Enhancements
        self.startButton.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.stopButton.setShortcut(QtGui.QKeySequence("Ctrl+Q"))
        self.languageComboBox.setFocus()
        self.startButton.setDefault(True)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Real-Time Transcription and Translation"))
        self.titleLabel.setText(_translate("MainWindow", "Real-Time Transcription and Translation"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.stopButton.setText(_translate("MainWindow", "Stop"))
        self.transcriptionLabel.setText(_translate("MainWindow", "Transcription:"))
        self.translationLabel.setText(_translate("MainWindow", "Translation:"))
        self.chunkCountLabel.setText(_translate("MainWindow", "Segments in Queue: 0"))
        self.vadWindowSizeSpinBox.setSuffix(" s")
        self.vadHopSizeSpinBox.setSuffix(" s")
        self.noSpeechThresholdSpinBox.setSuffix(" s")
