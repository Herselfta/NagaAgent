"""
优雅的设置界面组件
统一风格的设置界面，包含API配置、系统配置等多个选项
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QCheckBox, QSpinBox, 
                            QDoubleSpinBox, QComboBox, QFrame, QScrollArea,
                            QSlider, QTextEdit, QGroupBox, QGridLayout, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPainter, QColor
import sys
import os
import json

# 添加项目根目录到path，以便导入配置
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from config import config

class SettingCard(QWidget):
    """单个设置卡片"""
    value_changed = pyqtSignal(str, object)  # 设置名, 新值
    
    def __init__(self, title, description, control_widget, setting_key=None, parent=None):
        super().__init__(parent)
        self.setting_key = setting_key
        self.control_widget = control_widget
        self.setup_ui(title, description)
        
    def setup_ui(self, title, description):
        """初始化卡片UI"""
        self.setFixedHeight(80)
        self.setStyleSheet("""
            SettingCard {
                background: rgba(255, 255, 255, 8);
                border: 1px solid rgba(255, 255, 255, 20);
                border-radius: 10px;
                margin: 2px;
            }
            SettingCard:hover {
                background: rgba(255, 255, 255, 15);
                border: 1px solid rgba(255, 255, 255, 40);
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        
        # 左侧文本区域
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        # 标题
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #fff;
                font: 12pt 'Lucida Console';
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        text_layout.addWidget(title_label)
        
        # 描述
        desc_label = QLabel(description)
        desc_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 120);
                font: 9pt 'Lucida Console';
                background: transparent;
                border: none;
            }
        """)
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)
        
        layout.addLayout(text_layout, 1)
        
        # 右侧控件区域
        control_container = QWidget()
        control_container.setFixedWidth(200)
        control_layout = QHBoxLayout(control_container)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addWidget(self.control_widget)
        
        layout.addWidget(control_container)
        
        # 连接控件信号
        self.connect_signals()
        
    def connect_signals(self):
        """连接控件信号"""
        if isinstance(self.control_widget, QLineEdit):
            self.control_widget.textChanged.connect(self.on_value_changed)
        elif isinstance(self.control_widget, QCheckBox):
            self.control_widget.toggled.connect(self.on_value_changed)
        elif isinstance(self.control_widget, (QSpinBox, QDoubleSpinBox)):
            self.control_widget.valueChanged.connect(self.on_value_changed)
        elif isinstance(self.control_widget, QComboBox):
            self.control_widget.currentTextChanged.connect(self.on_value_changed)
        elif isinstance(self.control_widget, QSlider):
            self.control_widget.valueChanged.connect(self.on_value_changed)
            
    def on_value_changed(self, value):
        """处理值变化"""
        if self.setting_key:
            self.value_changed.emit(self.setting_key, value)

class SettingGroup(QWidget):
    """设置组"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.cards = []
        self.setup_ui(title)
        
    def setup_ui(self, title):
        """初始化组UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # 组标题
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                color: #fff;
                font: 16pt 'Lucida Console';
                font-weight: bold;
                background: transparent;
                border: none;
                margin-bottom: 10px;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 30);
            }
        """)
        layout.addWidget(title_label)
        
        # 卡片容器
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(4)
        
        layout.addWidget(self.cards_container)
        
    def add_card(self, card):
        """添加设置卡片"""
        self.cards.append(card)
        self.cards_layout.addWidget(card)

class ElegantSettingsWidget(QWidget):
    """优雅的设置界面"""
    
    settings_changed = pyqtSignal(str, object)  # 设置名, 新值
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pending_changes = {}  # 待保存的更改
        self.setup_ui()
        self.load_current_settings()
        
    def setup_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 20);
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 60);
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 80);
            }
        """)
        
        # 滚动内容
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(20)
        
        # 创建设置组
        self.create_system_group(scroll_layout)
        self.create_api_group(scroll_layout)
        self.create_interface_group(scroll_layout)
        self.create_xiayuan_group(scroll_layout)
        self.create_tts_group(scroll_layout)
        self.create_weather_group(scroll_layout)
        self.create_mqtt_group(scroll_layout)
        self.create_save_section(scroll_layout)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
    def create_api_group(self, parent_layout):
        group = SettingGroup("API 配置")
        # API Key
        if hasattr(config.api, "api_key"):
            api_key_input = QLineEdit()
            api_key_input.setText(config.api.api_key)
            api_key_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            api_key_card = SettingCard("API Key", "用于连接API的密钥", api_key_input, "api.api_key")
            api_key_card.value_changed.connect(self.on_setting_changed)
            group.add_card(api_key_card)
            self.api_key_input = api_key_input
        # Base URL
        if hasattr(config.api, "base_url"):
            base_url_input = QLineEdit()
            base_url_input.setText(config.api.base_url)
            base_url_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            base_url_card = SettingCard("API Base URL", "API基础URL", base_url_input, "api.base_url")
            base_url_card.value_changed.connect(self.on_setting_changed)
            group.add_card(base_url_card)
            self.base_url_input = base_url_input
        # Model
        if hasattr(config.api, "model"):
            model_combo = QComboBox()
            model_combo.addItems([config.api.model])
            model_combo.setCurrentText(config.api.model)
            model_combo.setStyleSheet(self.get_combo_style() + "color: #fff;")
            model_card = SettingCard("AI模型", "选择用于对话的AI模型", model_combo, "api.model")
            model_card.value_changed.connect(self.on_setting_changed)
            group.add_card(model_card)
            self.model_combo = model_combo
        # Max Tokens
        if hasattr(config.api, "max_tokens"):
            max_tokens_spin = QSpinBox()
            max_tokens_spin.setRange(100, 8000)
            max_tokens_spin.setValue(config.api.max_tokens)
            max_tokens_spin.setStyleSheet(self.get_spin_style() + "color: #fff;")
            max_tokens_card = SettingCard("最大Token数", "单次对话的最大长度限制", max_tokens_spin, "api.max_tokens")
            max_tokens_card.value_changed.connect(self.on_setting_changed)
            group.add_card(max_tokens_card)
            self.max_tokens_spin = max_tokens_spin
        # Temperature
        if hasattr(config.api, "temperature"):
            temp_slider = QSlider(Qt.Horizontal)
            temp_slider.setRange(0, 200)
            temp_slider.setValue(int(config.api.temperature * 100))
            temp_slider.setStyleSheet(self.get_slider_style())
            temp_card = SettingCard("温度参数", "控制AI回复的随机性和创造性", temp_slider, "api.temperature")
            temp_card.value_changed.connect(self.on_setting_changed)
            group.add_card(temp_card)
            self.temp_slider = temp_slider
            
        # Max History Rounds
        # 该项表示上下文对话轮数，即系统会保留最近多少轮对话内容作为上下文
        if hasattr(config.api, "max_history_rounds"):
            history_spin = QSpinBox()
            history_spin.setRange(1, 50)
            history_spin.setValue(config.api.max_history_rounds)
            history_spin.setStyleSheet(self.get_spin_style() + "color: #fff;")
            history_card = SettingCard("历史轮数", "上下文对话轮数（系统会保留最近多少轮对话内容作为上下文）", history_spin, "api.max_history_rounds")
            history_card.value_changed.connect(self.on_setting_changed)
            group.add_card(history_card)
            self.history_spin = history_spin
        parent_layout.addWidget(group)

    def create_system_group(self, parent_layout):
        group = SettingGroup("系统配置")
        # version 只读
        if hasattr(config.system, "version"):
            version_label = QLabel(str(config.system.version))
            version_label.setStyleSheet("color: #fff;")
            version_card = SettingCard("系统版本", "当前系统版本号", version_label, None)
            group.add_card(version_card)
        # voice_enabled
        if hasattr(config.system, "voice_enabled"):
            voice_checkbox = QCheckBox()
            voice_checkbox.setChecked(config.system.voice_enabled)
            voice_checkbox.setStyleSheet(self.get_checkbox_style() + "color: #fff;")
            voice_card = SettingCard("语音交互", "启用语音输入和输出功能", voice_checkbox, "system.voice_enabled")
            voice_card.value_changed.connect(self.on_setting_changed)
            group.add_card(voice_card)
            self.voice_checkbox = voice_checkbox
        # stream_mode
        if hasattr(config.system, "stream_mode"):
            stream_checkbox = QCheckBox()
            stream_checkbox.setChecked(config.system.stream_mode)
            stream_checkbox.setStyleSheet(self.get_checkbox_style() + "color: #fff;")
            stream_card = SettingCard("流式响应", "启用实时流式响应显示", stream_checkbox, "system.stream_mode")
            stream_card.value_changed.connect(self.on_setting_changed)
            group.add_card(stream_card)
            self.stream_checkbox = stream_checkbox
        # debug
        if hasattr(config.system, "debug"):
            debug_checkbox = QCheckBox()
            debug_checkbox.setChecked(config.system.debug)
            debug_checkbox.setStyleSheet(self.get_checkbox_style() + "color: #fff;")
            debug_card = SettingCard("调试模式", "启用详细的调试信息输出", debug_checkbox, "system.debug")
            debug_card.value_changed.connect(self.on_setting_changed)
            group.add_card(debug_card)
            self.debug_checkbox = debug_checkbox
        # log_level
        if hasattr(config.system, "log_level"):
            log_combo = QComboBox()
            log_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
            log_combo.setCurrentText(config.system.log_level)
            log_combo.setStyleSheet(self.get_combo_style() + "color: #fff;")
            log_card = SettingCard("日志级别", "系统日志输出级别", log_combo, "system.log_level")
            log_card.value_changed.connect(self.on_setting_changed)
            group.add_card(log_card)
            self.log_combo = log_combo
        parent_layout.addWidget(group)

    def create_interface_group(self, parent_layout):
        group = SettingGroup("界面配置")
        # user_name
        if hasattr(config.ui, "user_name"):
            user_name_input = QLineEdit()
            user_name_input.setText(config.ui.user_name)
            user_name_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            user_name_card = SettingCard("用户名", "界面显示的用户名", user_name_input, "ui.user_name")
            user_name_card.value_changed.connect(self.on_setting_changed)
            group.add_card(user_name_card)
            self.user_name_input = user_name_input
        # bg_alpha
        if hasattr(config.ui, "bg_alpha"):
            alpha_slider = QSlider(Qt.Horizontal)
            alpha_slider.setRange(30, 100)
            alpha_slider.setValue(int(config.ui.bg_alpha * 100))
            alpha_slider.setStyleSheet(self.get_slider_style())
            alpha_card = SettingCard("背景透明度", "调整界面背景的透明程度", alpha_slider, "ui.bg_alpha")
            alpha_card.value_changed.connect(self.on_setting_changed)
            group.add_card(alpha_card)
            self.alpha_slider = alpha_slider
        # window_bg_alpha
        if hasattr(config.ui, "window_bg_alpha"):
            window_bg_spin = QSpinBox()
            window_bg_spin.setRange(0, 255)
            window_bg_spin.setValue(config.ui.window_bg_alpha)
            window_bg_spin.setStyleSheet(self.get_spin_style() + "color: #fff;")
            window_bg_card = SettingCard("窗口背景透明度", "主窗口背景透明度", window_bg_spin, "ui.window_bg_alpha")
            window_bg_card.value_changed.connect(self.on_setting_changed)
            group.add_card(window_bg_card)
            self.window_bg_spin = window_bg_spin
        parent_layout.addWidget(group)
        
    def create_xiayuan_group(self, parent_layout):
        group = SettingGroup("夏园记忆系统")
        # grag部分
        if hasattr(config.grag, "neo4j_uri"):
            neo4j_uri_input = QLineEdit()
            neo4j_uri_input.setText(config.grag.neo4j_uri)
            neo4j_uri_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            neo4j_uri_card = SettingCard("Neo4j URI", "知识图谱数据库地址", neo4j_uri_input, "grag.neo4j_uri")
            neo4j_uri_card.value_changed.connect(self.on_setting_changed)
            group.add_card(neo4j_uri_card)
        if hasattr(config.grag, "neo4j_user"):
            neo4j_user_input = QLineEdit()
            neo4j_user_input.setText(config.grag.neo4j_user)
            neo4j_user_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            neo4j_user_card = SettingCard("Neo4j 用户名", "知识图谱数据库用户名", neo4j_user_input, "grag.neo4j_user")
            neo4j_user_card.value_changed.connect(self.on_setting_changed)
            group.add_card(neo4j_user_card)
        if hasattr(config.grag, "neo4j_password"):
            neo4j_pwd_input = QLineEdit()
            neo4j_pwd_input.setText(config.grag.neo4j_password)
            neo4j_pwd_input.setEchoMode(QLineEdit.Password)
            neo4j_pwd_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            neo4j_pwd_card = SettingCard("Neo4j 密码", "知识图谱数据库密码", neo4j_pwd_input, "grag.neo4j_password")
            neo4j_pwd_card.value_changed.connect(self.on_setting_changed)
            group.add_card(neo4j_pwd_card)
        # quick_model部分
        if hasattr(config.quick_model, "base_url"):
            qm_url_input = QLineEdit()
            qm_url_input.setText(config.quick_model.base_url)
            qm_url_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            qm_url_card = SettingCard("快速模型URL", "快速模型API地址", qm_url_input, "quick_model.base_url")
            qm_url_card.value_changed.connect(self.on_setting_changed)
            group.add_card(qm_url_card)
        if hasattr(config.quick_model, "api_key"):
            qm_api_input = QLineEdit()
            qm_api_input.setText(config.quick_model.api_key)
            qm_api_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            qm_api_card = SettingCard("快速模型API Key", "快速模型API密钥", qm_api_input, "quick_model.api_key")
            qm_api_card.value_changed.connect(self.on_setting_changed)
            group.add_card(qm_api_card)
            
        # Similarity Threshold
        if hasattr(config.grag, "similarity_threshold"):
            sim_slider = QSlider(Qt.Horizontal)
            sim_slider.setRange(0, 100)
            sim_slider.setValue(int(config.grag.similarity_threshold * 100))
            sim_slider.setStyleSheet(self.get_slider_style())
            sim_card = SettingCard("相似度阈值", "知识图谱检索的相似度阈值", sim_slider, "grag.similarity_threshold")
            sim_card.value_changed.connect(self.on_setting_changed)
            group.add_card(sim_card)
            self.sim_slider = sim_slider
            
        parent_layout.addWidget(group)

    def create_tts_group(self, parent_layout):
        group = SettingGroup("TTS 配置")
        if hasattr(config.tts, "api_key"):
            tts_api_input = QLineEdit()
            tts_api_input.setText(config.tts.api_key)
            tts_api_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            tts_api_card = SettingCard("TTS API Key", "TTS服务API密钥", tts_api_input, "tts.api_key")
            tts_api_card.value_changed.connect(self.on_setting_changed)
            group.add_card(tts_api_card)
        if hasattr(config.tts, "port"):
            tts_port_spin = QSpinBox()
            tts_port_spin.setRange(1, 65535)
            tts_port_spin.setValue(config.tts.port)
            tts_port_spin.setStyleSheet(self.get_spin_style() + "color: #fff;")
            tts_port_card = SettingCard("TTS端口", "TTS服务端口", tts_port_spin, "tts.port")
            tts_port_card.value_changed.connect(self.on_setting_changed)
            group.add_card(tts_port_card)
        if hasattr(config.tts, "keep_audio_files"):
            keep_audio_checkbox = QCheckBox()
            keep_audio_checkbox.setChecked(config.tts.keep_audio_files)
            keep_audio_checkbox.setStyleSheet(self.get_checkbox_style() + "color: #fff;")
            keep_audio_card = SettingCard("保留音频文件", "保留TTS生成的音频文件用于调试", keep_audio_checkbox, "tts.keep_audio_files")
            keep_audio_card.value_changed.connect(self.on_setting_changed)
            group.add_card(keep_audio_checkbox)
        parent_layout.addWidget(group)

    def create_weather_group(self, parent_layout):
        group = SettingGroup("天气服务配置")
        if hasattr(config.weather, "api_key"):
            weather_api_input = QLineEdit()
            weather_api_input.setText(config.weather.api_key)
            weather_api_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            weather_api_card = SettingCard("天气API Key", "天气服务API密钥", weather_api_input, "weather.api_key")
            weather_api_card.value_changed.connect(self.on_setting_changed)
            group.add_card(weather_api_card)
        parent_layout.addWidget(group)

    def create_mqtt_group(self, parent_layout):
        group = SettingGroup("MQTT 配置")
        if hasattr(config.mqtt, "broker"):
            mqtt_broker_input = QLineEdit()
            mqtt_broker_input.setText(config.mqtt.broker)
            mqtt_broker_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            mqtt_broker_card = SettingCard("MQTT Broker", "MQTT服务器地址", mqtt_broker_input, "mqtt.broker")
            mqtt_broker_card.value_changed.connect(self.on_setting_changed)
            group.add_card(mqtt_broker_card)
        if hasattr(config.mqtt, "port"):
            mqtt_port_spin = QSpinBox()
            mqtt_port_spin.setRange(1, 65535)
            mqtt_port_spin.setValue(config.mqtt.port)
            mqtt_port_spin.setStyleSheet(self.get_spin_style() + "color: #fff;")
            mqtt_port_card = SettingCard("MQTT端口", "MQTT服务器端口", mqtt_port_spin, "mqtt.port")
            mqtt_port_card.value_changed.connect(self.on_setting_changed)
            group.add_card(mqtt_port_card)
        if hasattr(config.mqtt, "username"):
            mqtt_user_input = QLineEdit()
            mqtt_user_input.setText(config.mqtt.username)
            mqtt_user_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            mqtt_user_card = SettingCard("MQTT用户名", "MQTT服务器用户名", mqtt_user_input, "mqtt.username")
            mqtt_user_card.value_changed.connect(self.on_setting_changed)
            group.add_card(mqtt_user_card)
        if hasattr(config.mqtt, "password"):
            mqtt_pwd_input = QLineEdit()
            mqtt_pwd_input.setText(config.mqtt.password)
            mqtt_pwd_input.setEchoMode(QLineEdit.Password)
            mqtt_pwd_input.setStyleSheet(self.get_input_style() + "color: #fff;")
            mqtt_pwd_card = SettingCard("MQTT密码", "MQTT服务器密码", mqtt_pwd_input, "mqtt.password")
            mqtt_pwd_card.value_changed.connect(self.on_setting_changed)
            group.add_card(mqtt_pwd_card)
        parent_layout.addWidget(group)
        
    def create_save_section(self, parent_layout):
        """创建保存区域"""
        save_container = QWidget()
        save_container.setFixedHeight(60)
        save_layout = QHBoxLayout(save_container)
        save_layout.setContentsMargins(0, 10, 0, 10)
        
        # 状态提示
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 120);
                font: 10pt 'Lucida Console';
                background: transparent;
                border: none;
            }
        """)
        save_layout.addWidget(self.status_label)
        
        save_layout.addStretch()
        
        # 重置按钮
        reset_btn = QPushButton("重置")
        reset_btn.setFixedSize(80, 36)
        reset_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 100, 100, 150);
                color: #fff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 8px;
                padding: 6px 12px;
                font: 11pt 'Lucida Console';
            }
            QPushButton:hover {
                border: 1px solid rgba(255, 255, 255, 80);
                background: rgba(120, 120, 120, 180);
            }
            QPushButton:pressed {
                background: rgba(80, 80, 80, 200);
            }
        """)
        reset_btn.clicked.connect(self.reset_settings)
        save_layout.addWidget(reset_btn)
        
        # 保存按钮
        self.save_btn = QPushButton("保存设置")
        self.save_btn.setFixedSize(100, 36)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: rgba(100, 200, 100, 150);
                color: #fff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 8px;
                padding: 6px 12px;
                font: 11pt 'Lucida Console';
                font-weight: bold;
            }
            QPushButton:hover {
                border: 1px solid rgba(255, 255, 255, 80);
                background: rgba(120, 220, 120, 180);
            }
            QPushButton:pressed {
                background: rgba(80, 180, 80, 200);
            }
        """)
        self.save_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(self.save_btn)
        
        parent_layout.addWidget(save_container)
        
    def get_input_style(self):
        """获取输入框样式"""
        return """
            QLineEdit {
                background: rgba(17,17,17,180);
                color: #fff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 6px;
                padding: 6px 10px;
                font: 10pt 'Lucida Console';
            }
            QLineEdit:focus {
                border: 1px solid rgba(100, 200, 255, 100);
            }
        """
        
    def get_combo_style(self):
        """获取下拉框样式"""
        return """
            QComboBox {
                background: rgba(17,17,17,180);
                color: #fff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 6px;
                padding: 6px 10px;
                font: 10pt 'Lucida Console';
            }
            QComboBox:hover {
                border: 1px solid rgba(255, 255, 255, 80);
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
        """
        
    def get_checkbox_style(self):
        """获取复选框样式"""
        return """
            QCheckBox {
                color: #fff;
                font: 10pt 'Lucida Console';
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 50);
                background: rgba(17,17,17,180);
            }
            QCheckBox::indicator:checked {
                background: rgba(100, 200, 255, 150);
                border: 1px solid rgba(100, 200, 255, 200);
            }
        """
        
    def get_slider_style(self):
        """获取滑块样式"""
        return """
            QSlider::groove:horizontal {
                border: 1px solid rgba(255, 255, 255, 30);
                height: 6px;
                background: rgba(17,17,17,180);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: rgba(100, 200, 255, 150);
                border: 1px solid rgba(100, 200, 255, 200);
                width: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::handle:horizontal:hover {
                background: rgba(120, 220, 255, 180);
            }
        """
        
    def get_spin_style(self):
        """获取数字输入框样式"""
        return """
            QSpinBox {
                background: rgba(17,17,17,180);
                color: #fff;
                border: 1px solid rgba(255, 255, 255, 50);
                border-radius: 6px;
                padding: 6px 10px;
                font: 10pt 'Lucida Console';
            }
            QSpinBox:focus {
                border: 1px solid rgba(100, 200, 255, 100);
            }
        """
        
    def on_setting_changed(self, setting_key, value):
        """处理设置变化"""
        self.pending_changes[setting_key] = value
        self.update_status_label(f"● {setting_key} 已修改")
        
    def update_status_label(self, text):
        """更新状态标签"""
        self.status_label.setText(text)
        # 3秒后清空状态
        QTimer.singleShot(3000, lambda: self.status_label.setText(""))
        
    def load_current_settings(self):
        """加载当前设置"""
        try:
            # API设置
            self.api_key_input.setText(config.api.api_key if config.api.api_key != "sk-placeholder-key-not-set" else "")
            self.base_url_input.setText(config.api.base_url)
            
            index = self.model_combo.findText(config.api.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                
            # 系统设置
            self.temp_slider.setValue(int(config.api.temperature * 100))
            self.max_tokens_spin.setValue(config.api.max_tokens)
            self.history_spin.setValue(config.api.max_history_rounds)
            
            # 界面设置
            self.stream_checkbox.setChecked(config.system.stream_mode)
            self.voice_checkbox.setChecked(config.system.voice_enabled)
            
            # 高级设置
            self.debug_checkbox.setChecked(config.system.debug)
            self.sim_slider.setValue(int(config.grag.similarity_threshold * 100))
            
        except Exception as e:
            print(f"加载设置失败: {e}")
            
    def save_settings(self):
        """保存所有设置到config.json"""
        try:
            changes_count = len(self.pending_changes)
            
            if changes_count == 0:
                self.update_status_label("● 没有需要保存的更改")
                return
            
            # 加载当前config.json
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception:
                config_data = {}
            
            success_count = 0
            
            # 更新配置数据
            for setting_key, value in self.pending_changes.items():
                try:
                    # 解析嵌套的配置键 (例如 "api.api_key")
                    keys = setting_key.split('.')
                    current = config_data
                    
                    # 导航到父级
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    
                    # 设置值
                    final_key = keys[-1]
                    if setting_key in ['api.temperature', 'grag.similarity_threshold', 'ui.bg_alpha']:
                        # 温度、相似度、透明度值从0-100转换为0.0-1.0
                        current[final_key] = value / 100.0
                    else:
                        current[final_key] = value
                    
                    success_count += 1
                        
                except Exception as e:
                    print(f"保存设置 {setting_key} 失败: {e}")
            
            # 保存到config.json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            
            # 动态更新config对象
            from config import load_config
            global config
            config = load_config()
                    
            self.update_status_label(f"✓ 已保存 {success_count}/{changes_count} 项设置")
            self.pending_changes.clear()
            
            # 发送设置变化信号
            self.settings_changed.emit("all", None)
            
        except Exception as e:
            self.update_status_label(f"✗ 保存失败: {str(e)}")
            
            
    def reset_settings(self):
        """重置所有设置"""
        self.pending_changes.clear()
        self.load_current_settings()
        self.update_status_label("● 设置已重置")


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication([])
    
    # 创建测试窗口
    test_window = QWidget()
    test_window.setStyleSheet("""
        QWidget {
            background: rgba(25, 25, 25, 220);
            color: white;
        }
    """)
    test_window.resize(800, 600)
    
    layout = QVBoxLayout(test_window)
    
    # 添加设置界面
    settings = ElegantSettingsWidget()
    settings.settings_changed.connect(
        lambda key, value: print(f"设置变化: {key} = {value}")
    )
    
    layout.addWidget(settings)
    
    test_window.show()
    app.exec_() 
