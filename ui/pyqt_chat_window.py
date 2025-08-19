import sys, os; sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from .styles.button_factory import ButtonFactory
import sys, datetime
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QSizePolicy, QGraphicsBlurEffect, QHBoxLayout, QLabel, QVBoxLayout, QStackedLayout, QPushButton, QStackedWidget, QDesktopWidget, QScrollArea, QSplitter, QGraphicsDropShadowEffect, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal, QParallelAnimationGroup, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QColor, QPainter, QBrush, QFont, QPixmap, QPalette, QPen, QIcon
from conversation_core import NagaConversation
import os
from config import config # 导入统一配置
from ui.response_utils import extract_message  # 新增：引入消息提取工具
from ui.progress_widget import EnhancedProgressWidget  # 导入进度组件
from ui.enhanced_worker import StreamingWorker, BatchWorker  # 导入增强Worker
from ui.elegant_settings_widget import ElegantSettingsWidget
import asyncio
import json
import threading
from PyQt5.QtCore import QObject, pyqtSignal as Signal
import requests
import shutil
from pathlib import Path
import time
import os

# 使用统一配置系统
BG_ALPHA = config.ui.bg_alpha
WINDOW_BG_ALPHA = config.ui.window_bg_alpha
USER_NAME = config.ui.user_name
MAC_BTN_SIZE = config.ui.mac_btn_size
MAC_BTN_MARGIN = config.ui.mac_btn_margin
MAC_BTN_GAP = config.ui.mac_btn_gap
ANIMATION_DURATION = config.ui.animation_duration

class TitleBar(QWidget):
    def __init__(s, text, parent=None):
        super().__init__(parent)
        s.text = text
        s.setFixedHeight(100)
        s.setAttribute(Qt.WA_TranslucentBackground)
        s._offset = None
        # mac风格按钮
        for i,(txt,color,hover,cb) in enumerate([
            ('-','#FFBD2E','#ffe084',lambda:s.parent().showMinimized()),
            ('×','#FF5F57','#ff8783',lambda:s.parent().close())]):
            btn=QPushButton(txt,s)
            btn.setGeometry(s.width()-MAC_BTN_MARGIN-MAC_BTN_SIZE*(2-i)-MAC_BTN_GAP*(1-i),36,MAC_BTN_SIZE,MAC_BTN_SIZE)
            btn.setStyleSheet(f"QPushButton{{background:{color};border:none;border-radius:{MAC_BTN_SIZE//2}px;color:#fff;font:18pt;}}QPushButton:hover{{background:{hover};}}")
            btn.clicked.connect(cb)
            setattr(s,f'btn_{"min close".split()[i]}',btn)
    def mousePressEvent(s, e):
        if e.button()==Qt.LeftButton: s._offset = e.globalPos()-s.parent().frameGeometry().topLeft()
    def mouseMoveEvent(s, e):
        if s._offset and e.buttons()&Qt.LeftButton:
            s.parent().move(e.globalPos()-s._offset)
    def mouseReleaseEvent(s,e):s._offset=None
    def paintEvent(s, e):
        qp = QPainter(s)
        qp.setRenderHint(QPainter.Antialiasing)
        w, h = s.width(), s.height()
        qp.setPen(QColor(255,255,255,180))
        qp.drawLine(0, 2, w, 2)
        qp.drawLine(0, h-3, w, h-3)
        font = QFont("Consolas", max(10, (h-40)//2), QFont.Bold)
        qp.setFont(font)
        rect = QRect(0, 20, w, h-40)
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            qp.setPen(QColor(0,0,0))
            qp.drawText(rect.translated(dx,dy), Qt.AlignCenter, s.text)
        qp.setPen(QColor(255,255,255))
        qp.drawText(rect, Qt.AlignCenter, s.text)

    def resizeEvent(s, e):
        """
        Ensure the mac-style control buttons stay at the right top when the titlebar resizes.
        """
        try:
            x = s.width() - MAC_BTN_MARGIN
            for i, btn in enumerate([s.btn_min, s.btn_close]):
                btn.move(x - MAC_BTN_SIZE*(2-i) - MAC_BTN_GAP*(1-i), 36)
        except Exception:
            # safe-fail if buttons not yet created
            pass

class AnimatedSideWidget(QWidget):
    """
    自定义侧栏Widget，支持动画发光效果
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_alpha = int(BG_ALPHA * 255)
        self.border_alpha = 50
        self.glow_intensity = 0  # 发光强度 0-20
        self.is_glowing = False
        
    def set_background_alpha(self, alpha):
        """
        设置背景透明度
        """
        self.bg_alpha = alpha
        self.update()
        
    def set_border_alpha(self, alpha):
        """
        设置边框透明度
        """
        self.border_alpha = alpha
        self.update()
        
    def set_glow_intensity(self, intensity):
        """
        设置发光强度 0-20
        """
        self.glow_intensity = max(0, min(20, intensity))
        self.update()
        
    def start_glow_animation(self):
        """
        开始发光动画
        """
        self.is_glowing = True
        self.update()
        
    def stop_glow_animation(self):
        """
        停止发光动画
        """
        self.is_glowing = False
        self.glow_intensity = 0
        self.update()
        
    def paintEvent(self, event):
        """
        自定义绘制方法
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # 绘制发光效果（如果有）
        if self.glow_intensity > 0:
            glow_rect = rect.adjusted(-2, -2, 2, 2)
            glow_color = QColor(100, 200, 255, self.glow_intensity)
            painter.setPen(QPen(glow_color, 2))
            painter.setBrush(QBrush(Qt.NoBrush))
            painter.drawRoundedRect(glow_rect, 17, 17)
        
        # 绘制主要背景
        bg_color = QColor(17, 17, 17, self.bg_alpha)
        painter.setBrush(QBrush(bg_color))
        
        # 绘制边框
        border_color = QColor(255, 255, 255, self.border_alpha)
        painter.setPen(QPen(border_color, 1))
        
        # 绘制圆角矩形
        painter.drawRoundedRect(rect, 15, 15)
        
        super().paintEvent(event)
        
    def start_glow_animation(self):
        """开始发光动画"""
        self.is_glowing = True
        self.update()
        
    def stop_glow_animation(self):
        """停止发光动画"""
        self.is_glowing = False
        self.glow_intensity = 0
        self.update()
        
    def paintEvent(self, event):
        """自定义绘制方法"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect()
        
        # 绘制发光效果（如果有）
        if self.glow_intensity > 0:
            glow_rect = rect.adjusted(-2, -2, 2, 2)
            glow_color = QColor(100, 200, 255, self.glow_intensity)
            painter.setPen(QPen(glow_color, 2))
            painter.setBrush(QBrush(Qt.NoBrush))
            painter.drawRoundedRect(glow_rect, 17, 17)
        
        # 绘制主要背景
        bg_color = QColor(17, 17, 17, self.bg_alpha)
        painter.setBrush(QBrush(bg_color))
        
        # 绘制边框
        border_color = QColor(255, 255, 255, self.border_alpha)
        painter.setPen(QPen(border_color, 1))
        
        # 绘制圆角矩形
        painter.drawRoundedRect(rect, 15, 15)
        
        super().paintEvent(event)

class AutoFitLabel(QLabel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.setWordWrap(True)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 自动调整字体大小以适应标签大小
        font = self.font()
        font_size = min(self.width() // 20, self.height() // 2, 16)
        font.setPointSize(max(font_size, 8))
        self.setFont(font)

class ChatWindow(QWidget):
    def __init__(s):
        super().__init__()
        
        # 获取屏幕大小并自适应
        desktop = QDesktopWidget()
        screen_rect = desktop.screenGeometry()
        # 设置为屏幕大小的80%
        window_width = int(screen_rect.width() * 0.8)
        window_height = int(screen_rect.height() * 0.8)
        s.resize(window_width, window_height)
        
        # 窗口居中显示
        x = (screen_rect.width() - window_width) // 2
        y = (screen_rect.height() - window_height) // 2
        s.move(x, y)
        
        # 移除置顶标志，保留无边框
        s.setWindowFlags(Qt.FramelessWindowHint)
        s.setAttribute(Qt.WA_TranslucentBackground)
        
        # 添加窗口背景和拖动支持
        s._offset = None
        s.setStyleSheet(f"""
            ChatWindow {{
                background: rgba(25, 25, 25, {WINDOW_BG_ALPHA});
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 30);
            }}
        """)
        
        fontfam,fontbig,fontsize='Lucida Console',16,16
        
        # 创建主分割器，替换原来的HBoxLayout
        s.main_splitter = QSplitter(Qt.Horizontal, s)
        s.main_splitter.setStyleSheet("""
            QSplitter {
                background: transparent;
            }
            QSplitter::handle {
                background: rgba(255, 255, 255, 30);
                width: 2px;
                border-radius: 1px;
            }
            QSplitter::handle:hover {
                background: rgba(255, 255, 255, 60);
                width: 3px;
            }
        """)
        
        # 聊天区域容器
        chat_area=QWidget()
        chat_area.setMinimumWidth(400)  # 设置最小宽度
        vlay=QVBoxLayout(chat_area);vlay.setContentsMargins(0,0,0,0);vlay.setSpacing(10)
        
        # 用QStackedWidget管理聊天区和设置页
        s.chat_stack = QStackedWidget(chat_area)
        s.chat_stack.setStyleSheet("""
            QStackedWidget {
                background: transparent;
                border: none;
            }
        """) # 保证背景穿透
        s.text = QTextEdit() # 聊天历史
        s.text.setReadOnly(True)
        s.text.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(17,17,17,{int(BG_ALPHA*255)});
                color: #fff;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 50);
                font: 16pt 'Lucida Console';
                padding: 10px;
            }}
        """)
        s.chat_stack.addWidget(s.text) # index 0 聊天页
        s.settings_page = s.create_settings_page() # index 1 设置页
        s.chat_stack.addWidget(s.settings_page)
        vlay.addWidget(s.chat_stack, 1)
        
        # 添加进度显示组件
        s.progress_widget = EnhancedProgressWidget(chat_area)
        vlay.addWidget(s.progress_widget)
        
        s.input_wrap=QWidget(chat_area)
        s.input_wrap.setFixedHeight(60)  # 增加输入框包装器的高度，与字体大小匹配
        hlay=QHBoxLayout(s.input_wrap);hlay.setContentsMargins(0,0,0,0);hlay.setSpacing(8)
        s.prompt=QLabel('>',s.input_wrap)
        s.prompt.setStyleSheet(f"color:#fff;font:{fontsize}pt '{fontfam}';background:transparent;")
        hlay.addWidget(s.prompt)
        s.input = QTextEdit(s.input_wrap)
        s.input.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(17,17,17,{int(BG_ALPHA*255)});
                color: #fff;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 50);
                font: {fontsize}pt '{fontfam}';
                padding: 8px;
            }}
        """)
        s.input.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        s.input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        hlay.addWidget(s.input)
        
        # 添加文档上传按钮
        s.upload_btn = ButtonFactory.create_action_button("upload", s.input_wrap)
        hlay.addWidget(s.upload_btn)
        
        # 添加心智云图按钮
        s.mind_map_btn = ButtonFactory.create_action_button("mind_map", s.input_wrap)
        hlay.addWidget(s.mind_map_btn)
        
        vlay.addWidget(s.input_wrap,0)
        
        # 将聊天区域添加到分割器
        s.main_splitter.addWidget(chat_area)
        
        # 侧栏（图片显示区域）- 使用自定义动画Widget
        s.side = AnimatedSideWidget()
        s.collapsed_width = 400  # 收缩状态宽度
        s.expanded_width = 800  # 展开状态宽度
        s.side.setMinimumWidth(s.collapsed_width)  # 设置最小宽度为收缩状态
        s.side.setMaximumWidth(s.collapsed_width)  # 初始状态为收缩
        
        # 优化侧栏的悬停效果，使用QPainter绘制
        def setup_side_hover_effects():
            def original_enter(e):
                s.side.set_background_alpha(int(BG_ALPHA * 0.5 * 255))
                s.side.set_border_alpha(80)
            def original_leave(e):
                s.side.set_background_alpha(int(BG_ALPHA * 255))
                s.side.set_border_alpha(50)
            return original_enter, original_leave
        
        s.side_hover_enter, s.side_hover_leave = setup_side_hover_effects()
        s.side.enterEvent = s.side_hover_enter
        s.side.leaveEvent = s.side_hover_leave
        
        # 设置鼠标指针，提示可点击
        s.side.setCursor(Qt.PointingHandCursor)
        
        stack=QStackedLayout(s.side);stack.setContentsMargins(5,5,5,5)
        s.img=QLabel(s.side)
        s.img.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
        s.img.setAlignment(Qt.AlignCenter)
        s.img.setMinimumSize(1,1)
        s.img.setMaximumSize(16777215,16777215)
        s.img.setStyleSheet('background:transparent; border: none;')
        stack.addWidget(s.img)
        nick=QLabel(f"● 娜迦{config.system.version}",s.side)
        nick.setStyleSheet("""
            QLabel {
                color: #fff;
                font: 18pt 'Consolas';
                background: rgba(0,0,0,100);
                padding: 12px 0 12px 0;
                border-radius: 10px;
                border: none;
            }
        """)
        nick.setAlignment(Qt.AlignHCenter|Qt.AlignTop)
        nick.setAttribute(Qt.WA_TransparentForMouseEvents)
        nick.hide()  # 隐藏昵称
        stack.addWidget(nick)
        
        # 将侧栏添加到分割器
        s.main_splitter.addWidget(s.side)
        
        # 设置分割器的初始比例 - 侧栏收缩状态
        s.main_splitter.setSizes([window_width - s.collapsed_width - 20, s.collapsed_width])  # 大部分给聊天区域
        
        # 创建包含分割器的主布局
        main=QVBoxLayout(s)
        main.setContentsMargins(10,110,10,10)
        main.addWidget(s.main_splitter)
        
        s.nick=nick
        s.naga=NagaConversation()  # 第三次初始化：ChatWindow构造函数中创建
        s.worker=None
        s.full_img=0 # 立绘展开标志，0=收缩状态，1=展开状态
        s.streaming_mode = config.system.stream_mode  # 根据配置决定是否使用流式模式
        s.current_response = ""  # 当前响应缓冲
        s.animating = False  # 动画标志位，动画期间为True
        s._img_inited = False  # 标志变量，图片自适应只在初始化时触发一次
        
        # 连接进度组件信号
        s.progress_widget.cancel_requested.connect(s.cancel_current_task)
        
        s.input.textChanged.connect(s.adjust_input_height)
        s.input.installEventFilter(s)
        
        # 连接文档上传按钮
        s.upload_btn.clicked.connect(s.upload_document)
        
        # 连接心智云图按钮
        s.mind_map_btn.clicked.connect(s.open_mind_map)
        
        s.setLayout(main)
        s.titlebar = TitleBar('NAGA AGENT', s)
        s.titlebar.setGeometry(0,0,s.width(),100)
        s.side.mousePressEvent=s.toggle_full_img # 侧栏点击切换聊天/设置
        s.resizeEvent(None)  # 强制自适应一次，修复图片初始尺寸

    def create_settings_page(s):
        page = QWidget()
        page.setObjectName("SettingsPage")
        page.setStyleSheet("""
            #SettingsPage {
                background: transparent;
                border-radius: 24px;
                padding: 12px;
            }
        """)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 创建滚动区域
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
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 滚动内容
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(20)
        # 只保留系统设置界面
        s.settings_widget = ElegantSettingsWidget(scroll_content)
        s.settings_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        s.settings_widget.settings_changed.connect(s.on_settings_changed)
        scroll_layout.addWidget(s.settings_widget, 1)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)
        return page

    def resizeEvent(s, e):
        if getattr(s, '_animating', False):  # 动画期间跳过所有重绘操作，避免卡顿
            return
        if hasattr(s,'img'):
            s.img.resize(s.img.parent().width(), s.img.parent().height())
            # 延迟图片缩放操作，避免频繁重绘
            if not hasattr(s, '_resize_timer'):
                s._resize_timer = QTimer()
                s._resize_timer.setSingleShot(True)
                s._resize_timer.timeout.connect(s._delayed_image_resize)
            s._resize_timer.start(50)  # 50ms后执行图片缩放
            
    def _delayed_image_resize(s):
        """延迟执行的图片缩放，避免频繁重绘"""
        if hasattr(s, 'img') and not getattr(s, '_animating', False):
            p = os.path.join(os.path.dirname(__file__), 'standby.png')
            q = QPixmap(p)
            if os.path.exists(p) and not q.isNull():
                # 确保图片完全填满侧栏，无空隙
                parent_width = s.img.parent().width()
                parent_height = s.img.parent().height()
                s.img.setPixmap(q.scaled(parent_width, parent_height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    def adjust_input_height(s):
        doc = s.input.document()
        s.input_wrap.setFixedHeight(s.input.height())
        
    def eventFilter(s, obj, event):
        if obj is s.input and event.type()==6:
            if event.key()==Qt.Key_Return and not (event.modifiers()&Qt.ShiftModifier):
                s.on_send();return True
        return False
    def add_user_message(s, name, content):
        # 先把\n转成\n，再把\n转成<br>，适配所有换行
        from ui.response_utils import extract_message
        msg = extract_message(content)
        content_html = str(msg).replace('\\n', '\n').replace('\n', '<br>')
        
        # 添加消息到UI
        s.text.append(f"<span style='color:#fff;font-size:12pt;font-family:Lucida Console;'>{name}</span>")
        s.text.append(f"<span style='color:#fff;font-size:16pt;font-family:Lucida Console;'>{content_html}</span>")
        
        # 滚动到底部
        s.text.verticalScrollBar().setValue(s.text.verticalScrollBar().maximum())
        
        # 返回消息的起始位置，用于后续更新
        return len(s.text.toPlainText())
    
    def update_last_message(s, name, content):
        """更新最后一条消息的内容"""
        from ui.response_utils import extract_message
        msg = extract_message(content)
        content_html = str(msg).replace('\\n', '\n').replace('\n', '<br>')
        
        # 获取当前文本内容
        current_text = s.text.toHtml()
        
        # 查找最后一个消息块并替换
        lines = current_text.split('\n')
        if len(lines) >= 2:
            # 替换最后两个元素（名字和内容）
            lines[-2] = f"<span style='color:#fff;font-size:12pt;font-family:Lucida Console;'>{name}</span>"
            lines[-1] = f"<span style='color:#fff;font-size:16pt;font-family:Lucida Console;'>{content_html}</span>"
            
            # 重新组合HTML
            updated_html = '\n'.join(lines)
            s.text.setHtml(updated_html)
            
        # 滚动到底部
        s.text.verticalScrollBar().setValue(s.text.verticalScrollBar().maximum())
    def on_send(s):
        u = s.input.toPlainText().strip()
        if u:
            s.add_user_message(USER_NAME, u)
            s.input.clear()

            # 如果已有任务在运行，先取消
            if s.worker and s.worker.isRunning():
                s.cancel_current_task()
                return

            # 清空当前响应缓冲
            s.current_response = ""

            # 确保worker被清理
            if s.worker:
                s.worker.deleteLater()
                s.worker = None

            # 根据模式选择Worker类型，创建全新实例
            if s.streaming_mode:
                s.worker = StreamingWorker(s.naga, u)
                s.setup_streaming_worker()
            else:
                s.worker = BatchWorker(s.naga, u)
                s.setup_batch_worker()

            # 启动进度显示 - 恢复原来的调用方式
            s.progress_widget.set_thinking_mode()

            # 启动Worker
            s.worker.start()
    
    def setup_streaming_worker(s):
        """配置流式Worker的信号连接"""
        s.worker.progress_updated.connect(s.progress_widget.update_progress)
        s.worker.status_changed.connect(lambda status: s.progress_widget.status_label.setText(status))
        s.worker.error_occurred.connect(s.handle_error)
        
        # 流式专用信号
        s.worker.stream_chunk.connect(s.append_response_chunk)
        s.worker.stream_complete.connect(s.finalize_streaming_response)
        s.worker.finished.connect(s.on_response_finished)
    
    def setup_batch_worker(s):
        """配置批量Worker的信号连接"""
        s.worker.progress_updated.connect(s.progress_widget.update_progress)
        s.worker.status_changed.connect(lambda status: s.progress_widget.status_label.setText(status))
        s.worker.error_occurred.connect(s.handle_error)
        s.worker.finished.connect(s.on_batch_response_finished)
    
    def append_response_chunk(s, chunk):
        """追加响应片段（流式模式）- 实时显示"""
        s.current_response += chunk
        
        # 实时更新显示 - 立即显示到UI
        if not hasattr(s, '_current_message_id'):
            # 第一次收到chunk时，创建新消息
            s._current_message_id = s.add_user_message("娜迦", chunk)
        else:
            # 后续chunk，更新现有消息 - 立即更新
            s.update_last_message("娜迦", s.current_response)
            
        # 强制UI更新
        s.text.repaint()
    
    def finalize_streaming_response(s):
        """完成流式响应 - 立即处理"""
        if s.current_response:
            # 对累积的完整响应进行消息提取（多步自动\n分隔）
            from ui.response_utils import extract_message
            final_message = extract_message(s.current_response)
            
            # 更新最终消息
            if hasattr(s, '_current_message_id'):
                s.update_last_message("娜迦", final_message)
                delattr(s, '_current_message_id')
            else:
                s.add_user_message("娜迦", final_message)
        
        # 立即停止加载状态
        s.progress_widget.stop_loading()
    
    def on_response_finished(s, response):
        """处理完成的响应（流式模式后备）"""
        # 检查是否是取消操作的响应
        if response == "操作已取消":
            return  # 不显示，因为已经在cancel_current_task中显示了
        if not s.current_response:  # 如果流式没有收到数据，使用最终结果
            from ui.response_utils import extract_message
            final_message = extract_message(response)
            s.add_user_message("娜迦", final_message)
        s.progress_widget.stop_loading()
    
    def on_batch_response_finished(s, response):
        """处理完成的响应（批量模式）"""
        # 检查是否是取消操作的响应
        if response == "操作已取消":
            return  # 不显示，因为已经在cancel_current_task中显示了
        from ui.response_utils import extract_message
        final_message = extract_message(response)
        s.add_user_message("娜迦", final_message)
        s.progress_widget.stop_loading()
    
    def handle_error(s, error_msg):
        """处理错误"""
        s.add_user_message("系统", f"❌ {error_msg}")
        s.progress_widget.stop_loading()
    
    def cancel_current_task(s):
        """取消当前任务 - 优化版本，减少卡顿"""
        if s.worker and s.worker.isRunning():
            # 立即设置取消标志
            s.worker.cancel()
            
            # 非阻塞方式处理线程清理
            s.progress_widget.stop_loading()
            s.add_user_message("系统", "🚫 操作已取消")
            
            # 清空当前响应缓冲，避免部分响应显示
            s.current_response = ""
            
            # 使用QTimer延迟处理线程清理，避免UI卡顿
            def cleanup_worker():
                if s.worker:
                    s.worker.quit()
                    if not s.worker.wait(500):  # 只等待500ms
                        s.worker.terminate()
                        s.worker.wait(200)  # 再等待200ms
                    s.worker.deleteLater()
                    s.worker = None
            
            # 50ms后异步清理，避免阻塞UI
            QTimer.singleShot(50, cleanup_worker)
        else:
            s.progress_widget.stop_loading()

    def toggle_full_img(s,e):
        if getattr(s, '_animating', False):  # 动画期间禁止重复点击
            return
        s._animating = True  # 设置动画标志位
        s.full_img^=1  # 立绘展开标志切换
        target_width = s.expanded_width if s.full_img else s.collapsed_width  # 目标宽度：展开或收缩
        
        # --- 立即切换界面状态 ---
        if s.full_img:  # 展开状态 - 进入设置页面
            s.input_wrap.hide()  # 隐藏输入框
            s.chat_stack.setCurrentIndex(1)  # 切换到设置页
            s.side.setCursor(Qt.PointingHandCursor)  # 保持点击指针，可点击收缩
            s.titlebar.text = "SETTING PAGE"
            s.titlebar.update()
            s.side.setStyleSheet(f"""
                QWidget {{
                    background: rgba(17,17,17,{int(BG_ALPHA*255*0.9)});
                    border-radius: 15px;
                    border: 1px solid rgba(255, 255, 255, 80);
                }}
            """)
        else:  # 收缩状态 - 主界面聊天模式
            s.input_wrap.show()  # 显示输入框
            s.chat_stack.setCurrentIndex(0)  # 切换到聊天页
            s.input.setFocus()  # 恢复输入焦点
            s.side.setCursor(Qt.PointingHandCursor)  # 保持点击指针
            s.titlebar.text = "NAGA AGENT"
            s.titlebar.update()
            s.side.setStyleSheet(f"""
                QWidget {{
                    background: rgba(17,17,17,{int(BG_ALPHA*255*0.7)});
                    border-radius: 15px;
                    border: 1px solid rgba(255, 255, 255, 40);
                }}
            """)
        # --- 立即切换界面状态 END ---
        
        # 创建优化后的动画组
        group = QParallelAnimationGroup(s)
        
        # 侧栏宽度动画 - 合并为单个动画
        side_anim = QPropertyAnimation(s.side, b"minimumWidth", s)
        side_anim.setDuration(ANIMATION_DURATION)
        side_anim.setStartValue(s.side.width())
        side_anim.setEndValue(target_width)
        side_anim.setEasingCurve(QEasingCurve.OutCubic)  # 使用更流畅的缓动
        group.addAnimation(side_anim)
        
        side_anim2 = QPropertyAnimation(s.side, b"maximumWidth", s)
        side_anim2.setDuration(ANIMATION_DURATION)
        side_anim2.setStartValue(s.side.width())
        side_anim2.setEndValue(target_width)
        side_anim2.setEasingCurve(QEasingCurve.OutCubic)
        group.addAnimation(side_anim2)
        
        # 输入框动画 - 进入设置时隐藏，退出时显示
        if s.full_img:
            input_hide_anim = QPropertyAnimation(s.input_wrap, b"maximumHeight", s)
            input_hide_anim.setDuration(ANIMATION_DURATION // 2)
            input_hide_anim.setStartValue(s.input_wrap.height())
            input_hide_anim.setEndValue(0)
            input_hide_anim.setEasingCurve(QEasingCurve.OutQuad)
            group.addAnimation(input_hide_anim)
        else:
            input_show_anim = QPropertyAnimation(s.input_wrap, b"maximumHeight", s)
            input_show_anim.setDuration(ANIMATION_DURATION // 2)
            input_show_anim.setStartValue(0)
            input_show_anim.setEndValue(60)
            input_show_anim.setEasingCurve(QEasingCurve.OutQuad)
            group.addAnimation(input_show_anim)
        
        # 预加载原始图片，避免重复加载
        if not hasattr(s, '_original_pixmap'):
            p = os.path.join(os.path.dirname(__file__), 'standby.png')
            if os.path.exists(p):
                s._original_pixmap = QPixmap(p)
        
        def on_side_width_changed():
            """侧栏宽度变化时实时更新图片"""
            if hasattr(s, '_original_pixmap') and not s._original_pixmap.isNull():
                current_width = s.side.width() - 10  # 减去margin
                current_height = s.side.height() - 10
                
                if current_width > 50 and current_height > 50:  # 避免过小尺寸
                    # 实时缩放并设置图片
                    scaled_pixmap = s._original_pixmap.scaled(
                        current_width, current_height, 
                        Qt.KeepAspectRatioByExpanding, 
                        Qt.FastTransformation  # 使用快速变换，提高性能
                    )
                    s.img.setPixmap(scaled_pixmap)
                    s.img.resize(current_width, current_height)
                    
                    # 昵称始终隐藏
        
        def on_animation_finished():
            s._animating = False  # 动画结束标志
            # 最终使用高质量变换
            if hasattr(s, '_original_pixmap') and not s._original_pixmap.isNull():
                actual_width = target_width - 10
                actual_height = s.side.height() - 10
                final_pixmap = s._original_pixmap.scaled(
                    actual_width, actual_height,
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation  # 最终使用高质量变换
                )
                s.img.setPixmap(final_pixmap)
                s.img.resize(actual_width, actual_height)
                
                # 昵称始终隐藏
        
        # 连接信号
        side_anim.valueChanged.connect(on_side_width_changed)
        group.finished.connect(on_animation_finished)
        group.start()
        

    # 添加整个窗口的拖动支持
    def mousePressEvent(s, event):
        if event.button() == Qt.LeftButton:
            s._offset = event.globalPos() - s.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(s, event):
        if s._offset and event.buttons() & Qt.LeftButton:
            s.move(event.globalPos() - s._offset)
            event.accept()

    def mouseReleaseEvent(s, event):
        s._offset = None
        event.accept()

    def paintEvent(s, event):
        """绘制窗口背景"""
        painter = QPainter(s)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制主窗口背景 - 使用可调节的透明度
        painter.setBrush(QBrush(QColor(25, 25, 25, WINDOW_BG_ALPHA)))
        painter.setPen(QColor(255, 255, 255, 30))
        painter.drawRoundedRect(s.rect(), 20, 20)

    def on_settings_changed(s, setting_key, value):
        """处理设置变化"""
        print(f"设置变化: {setting_key} = {value}")
        
        # 这里可以实时应用某些设置变化
        if setting_key in ("all", "ui.bg_alpha", "ui.window_bg_alpha"):  # UI透明度变化 #
            s.apply_opacity_from_config()  # 立即应用 #
            return
        if setting_key in ("system.stream_mode", "STREAM_MODE"):
            s.streaming_mode = value if setting_key == "system.stream_mode" else value  # 兼容新旧键名 #
            s.add_user_message("系统", f"● 流式模式已{'启用' if s.streaming_mode else '禁用'}")
        elif setting_key in ("system.voice_enabled", "VOICE_ENABLED"):
            s.add_user_message("系统", f"● 语音功能已{'启用' if value else '禁用'}")
        elif setting_key in ("system.debug", "DEBUG"):
            s.add_user_message("系统", f"● 调试模式已{'启用' if value else '禁用'}")
        
        # 发送设置变化信号给其他组件
        # 这里可以根据需要添加更多处理逻辑

    def set_window_background_alpha(s, alpha):
        """设置整个窗口的背景透明度
        Args:
            alpha: 透明度值，可以是:
                   - 0-255的整数 (PyQt原生格式)
                   - 0.0-1.0的浮点数 (百分比格式)
        """
        global WINDOW_BG_ALPHA
        
        # 处理不同格式的输入
        if isinstance(alpha, float) and 0.0 <= alpha <= 1.0:
            # 浮点数格式：0.0-1.0 转换为 0-255
            WINDOW_BG_ALPHA = int(alpha * 255)
        elif isinstance(alpha, int) and 0 <= alpha <= 255:
            # 整数格式：0-255
            WINDOW_BG_ALPHA = alpha
        else:
            print(f"警告：无效的透明度值 {alpha}，应为0-255的整数或0.0-1.0的浮点数")
            return
        
        # 更新CSS样式表
        s.setStyleSheet(f"""
            ChatWindow {{
                background: rgba(25, 25, 25, {WINDOW_BG_ALPHA});
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 30);
            }}
        """)
    
        # 触发重绘
        s.update()
        
        print(f"✅ 窗口背景透明度已设置为: {WINDOW_BG_ALPHA}/255 ({WINDOW_BG_ALPHA/255*100:.1f}%不透明度)")

    def apply_opacity_from_config(s):
        """从配置中应用UI透明度(聊天区/输入框/侧栏/窗口)"""
        # 更新全局变量，保持其它逻辑一致 #
        global BG_ALPHA, WINDOW_BG_ALPHA
        BG_ALPHA = config.ui.bg_alpha
        WINDOW_BG_ALPHA = config.ui.window_bg_alpha if isinstance(config.ui.window_bg_alpha, int) else int(config.ui.window_bg_alpha * 255)

        # 计算alpha #
        alpha_px = int(BG_ALPHA * 255)

        # 更新聊天历史背景 #
        s.text.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(17,17,17,{alpha_px});
                color: #fff;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 50);
                font: 16pt 'Lucida Console';
                padding: 10px;
            }}
        """)

        # 更新输入框背景 #
        fontfam, fontsize = 'Lucida Console', 16
        s.input.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(17,17,17,{alpha_px});
                color: #fff;
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 50);
                font: {fontsize}pt '{fontfam}';
                padding: 8px;
            }}
        """)

        # 更新侧栏背景 #
        if hasattr(s, 'side') and isinstance(s.side, QWidget):
            try:
                s.side.set_background_alpha(alpha_px)
            except Exception:
                pass

        # 更新主窗口背景 #
        s.set_window_background_alpha(WINDOW_BG_ALPHA)

    def showEvent(s, event):
        """窗口显示事件"""
        super().showEvent(event)
        
        # 其他初始化代码...
        s.setFocus()
        s.input.setFocus()
        if not getattr(s, '_img_inited', False) and not getattr(s, '_animating', False):
            if hasattr(s, 'img'):
                # 获取实际的侧栏尺寸（减去margin）
                parent_width = s.img.parent().width()
                parent_height = s.img.parent().height()
                actual_width = parent_width - 10  # 减去左右margin 5px
                actual_height = parent_height - 10  # 减去上下margin 5px
                
                s.img.resize(actual_width, actual_height)
                p = os.path.join(os.path.dirname(__file__), 'standby.png')
                q = QPixmap(p)
                if os.path.exists(p) and not q.isNull():
                    s.img.setPixmap(q.scaled(actual_width, actual_height, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            s._img_inited = True

    def upload_document(s):
        """上传文档功能"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                s,
                "选择要上传的文档",
                "",
                "支持的文档格式 (*.docx *.txt *.md);;Word文档 (*.docx);;文本文件 (*.txt);;Markdown文件 (*.md);;所有文件 (*)"
            )
            
            if not file_path:
                return  # 用户取消选择
            
            # 检查文件格式
            file_ext = Path(file_path).suffix.lower()
            supported_formats = ['.docx', '.txt', '.md']
            
            if file_ext not in supported_formats:
                QMessageBox.warning(s, "格式不支持", 
                                   f"不支持的文件格式: {file_ext}\n\n支持的格式: {', '.join(supported_formats)}")
                return
            
            # 检查文件大小 (限制为10MB)
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:  # 10MB
                QMessageBox.warning(s, "文件过大", "文件大小不能超过10MB")
                return
            
            # 上传文件到API服务器
            s.upload_file_to_server(file_path)
            
        except Exception as e:
            QMessageBox.critical(s, "上传错误", f"文档上传失败:\n{str(e)}")
    
    def upload_file_to_server(s, file_path):
        """将文件上传到API服务器"""
        try:
            # 显示上传进度
            s.add_user_message("系统", f"📤 正在上传文档: {Path(file_path).name}")
            s.progress_widget.set_thinking_mode()
            s.progress_widget.status_label.setText("上传文档中...")
            
            # 准备上传数据
            api_url = "http://localhost:8000/upload/document"
            
            with open(file_path, 'rb') as f:
                files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
                data = {'description': f'通过NAGA聊天界面上传的文档'}
                
                # 发送上传请求
                response = requests.post(api_url, files=files, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                s.progress_widget.stop_loading()
                s.add_user_message("系统", f"✅ 文档上传成功: {result['filename']}")
                
                # 询问用户想要进行什么操作
                s.show_document_options(result['file_path'], result['filename'])
            else:
                s.progress_widget.stop_loading()
                s.add_user_message("系统", f"❌ 上传失败: {response.text}")
                
        except requests.exceptions.ConnectionError:
            s.progress_widget.stop_loading()
            s.add_user_message("系统", "❌ 无法连接到API服务器，请确保服务器正在运行")
        except Exception as e:
            s.progress_widget.stop_loading()
            s.add_user_message("系统", f"❌ 上传失败: {str(e)}")
    
    def show_document_options(s, file_path, filename):
        """显示文档处理选项"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QFrame, QPushButton
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFont
        
        dialog = QDialog(s)
        dialog.setWindowTitle("文档处理选项")
        dialog.setFixedSize(650, 480)
        # 隐藏标题栏的图标按钮
        dialog.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        dialog.setStyleSheet("""
            QDialog {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 10px;
            }
        """)
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(30, 25, 30, 25)
        layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("文档上传成功")
        title_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 25px; padding: 15px; min-height: 40px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 文件信息
        info_label = QLabel(f"文件名: {filename}")
        info_label.setStyleSheet("color: #34495e; font-size: 14px; padding: 10px;")
        layout.addWidget(info_label)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #bdc3c7;")
        layout.addWidget(line)
        
        # 操作按钮
        actions = [
            ("📖 读取内容", "read", "读取文档的完整内容"),
            ("🔍 分析文档", "analyze", "分析文档结构和内容"),
            ("📝 生成摘要", "summarize", "生成文档的简洁摘要")
        ]
        
        for btn_text, action, description in actions:
            btn = ButtonFactory.create_document_action_button(btn_text)
            
            # 添加描述标签
            desc_label = QLabel(description)
            desc_label.setStyleSheet("color: #7f8c8d; font-size: 12px; margin-bottom: 10px;")
            layout.addWidget(desc_label)
            layout.addWidget(btn)
            
            # 连接按钮事件
            btn.clicked.connect(lambda checked, f=file_path, a=action, d=dialog: s.process_document(f, a, d))
        
        # 取消按钮
        cancel_btn = ButtonFactory.create_cancel_button()
        cancel_btn.clicked.connect(dialog.close)
        layout.addWidget(cancel_btn)
        
        dialog.exec_()
    
    def process_document(s, file_path, action, dialog=None):
        """处理文档"""
        if dialog:
            dialog.close()
        
        try:
            s.add_user_message("系统", f"🔄 正在处理文档: {Path(file_path).name}")
            s.progress_widget.set_thinking_mode()
            s.progress_widget.status_label.setText("处理文档中...")
            
            # 调用API处理文档
            api_url = "http://localhost:8000/document/process"
            data = {
                "file_path": file_path,
                "action": action
            }
            
            response = requests.post(api_url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                s.progress_widget.stop_loading()
                
                if action == "read":
                    s.add_user_message("娜迦", f"📖 文档内容:\n\n{result['content']}")
                elif action == "analyze":
                    s.add_user_message("娜迦", f"🔍 文档分析:\n\n{result['analysis']}")
                elif action == "summarize":
                    s.add_user_message("娜迦", f"📝 文档摘要:\n\n{result['summary']}")
            else:
                s.progress_widget.stop_loading()
                s.add_user_message("系统", f"❌ 文档处理失败: {response.text}")
                
        except requests.exceptions.ConnectionError:
            s.progress_widget.stop_loading()
            s.add_user_message("系统", "❌ 无法连接到API服务器，请确保服务器正在运行")
        except Exception as e:
            s.progress_widget.stop_loading()
            s.add_user_message("系统", f"❌ 文档处理失败: {str(e)}")
    
    def open_mind_map(s):
        """打开心智云图"""
        try:
            # 检查是否存在知识图谱文件
            graph_file = "logs/knowledge_graph/graph.html"
            quintuples_file = "logs/knowledge_graph/quintuples.json"
            
            print("[MindMap] 用户点击了心智云图按钮")
            print(f"[MindMap] 检查文件: graph={os.path.abspath(graph_file)}, quintuples={os.path.abspath(quintuples_file)}")

            # 判断是否需要重新生成（HTML不存在或数据更新比HTML新）
            def _needs_regen(html_path, data_path):
                try:
                    if not os.path.exists(html_path):
                        return True
                    if not os.path.exists(data_path):
                        return False
                    return os.path.getmtime(data_path) > os.path.getmtime(html_path)
                except Exception:
                    return False

            if _needs_regen(graph_file, quintuples_file):
                if os.path.exists(quintuples_file):
                    # 有五元组数据，生成HTML
                    s.add_user_message("系统", "🔄 正在生成心智云图...")
                    try:
                        print("[MindMap] 检测到五元组数据，开始生成HTML...")
                        from summer_memory.quintuple_visualize_v2 import visualize_quintuples
                        visualize_quintuples()
                        if os.path.exists(graph_file):
                            import webbrowser
                            # use absolute file:// URI so browser can open regardless of current working directory
                            try:
                                print("[MindMap] 生成完成，尝试打开浏览器...")
                                webbrowser.open(Path(graph_file).resolve().as_uri())
                            except Exception:
                                # fallback to original behavior
                                print("[MindMap] 以绝对URI打开失败，尝试直接路径打开...")
                                webbrowser.open(graph_file)
                            s.add_user_message("系统", "🧠 心智云图已生成并打开")
                            print("[MindMap] 心智云图已生成并打开")
                        else:
                            s.add_user_message("系统", "❌ 心智云图生成失败")
                            print("[MindMap] 心智云图生成失败：未找到graph.html")
                    except Exception as e:
                        s.add_user_message("系统", f"❌ 生成心智云图失败: {str(e)}")
                        print(f"[MindMap] 生成异常: {type(e).__name__}: {e}")
                else:
                    # 没有五元组数据，提示用户
                    s.add_user_message("系统", "❌ 未找到五元组数据，请先进行对话以生成知识图谱")
                    print("[MindMap] 未找到五元组数据，无法生成")
            else:
                # HTML文件存在，直接打开
                import webbrowser
                # open using absolute file:// URI to avoid issues when browser cwd != project root
                try:
                    print("[MindMap] 检测到现有HTML，尝试打开浏览器...")
                    webbrowser.open(Path(graph_file).resolve().as_uri())
                except Exception:
                    webbrowser.open(graph_file)
                s.add_user_message("系统", "🧠 心智云图已打开")
                print("[MindMap] 心智云图已打开")
        except Exception as e:
            s.add_user_message("系统", f"❌ 打开心智云图失败: {str(e)}")
            print(f"[MindMap] 打开失败: {type(e).__name__}: {e}")

    def closeEvent(s, event):
        """窗口关闭时确保清理语音临时资源"""
        try:
            from voice.voice_integration import get_voice_integration
            vi = get_voice_integration()
            if vi:
                vi.shutdown()
        except Exception as e:
            print(f"[VoiceCleanup] 关闭时清理失败: {e}")
        super().closeEvent(event)

if __name__=="__main__":
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec_())
