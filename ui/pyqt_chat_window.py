import sys, os; sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import sys, datetime
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QSizePolicy, QGraphicsBlurEffect, QHBoxLayout, QLabel, QVBoxLayout, QStackedLayout, QPushButton, QStackedWidget, QDesktopWidget, QScrollArea, QSplitter, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal, QParallelAnimationGroup, QPropertyAnimation, QEasingCurve, QTimer
from PyQt5.QtGui import QColor, QPainter, QBrush, QFont, QPixmap, QPalette, QPen
from conversation_core import NagaConversation
import os
import config # 导入全局配置
from ui.response_utils import extract_message  # 新增：引入消息提取工具
from ui.progress_widget import EnhancedProgressWidget  # 导入进度组件
from ui.enhanced_worker import StreamingWorker, BatchWorker  # 导入增强Worker
from ui.elegant_personality_selector import ElegantPersonalitySelector
from ui.elegant_settings_widget import ElegantSettingsWidget
BG_ALPHA=0.5 # 聊天背景透明度50%
WINDOW_BG_ALPHA=110 # 主窗口背景透明度 (0-255，220约为86%不透明度)
USER_NAME=os.getenv('COMPUTERNAME')or os.getenv('USERNAME')or'用户' # 自动识别电脑名
MAC_BTN_SIZE=36 # mac圆按钮直径扩大1.5倍
MAC_BTN_MARGIN=16 # 右侧边距
MAC_BTN_GAP=12 # 按钮间距
ANIMATION_DURATION = 600  # 动画时长统一配置，增加到600ms让动画更丝滑

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
    def resizeEvent(s,e):
        x=s.width()-MAC_BTN_MARGIN
        for i,btn in enumerate([s.btn_min,s.btn_close]):btn.move(x-MAC_BTN_SIZE*(2-i)-MAC_BTN_GAP*(1-i),36)

class AnimatedSideWidget(QWidget):
    """自定义侧栏Widget，支持动画发光效果"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bg_alpha = int(BG_ALPHA * 255)
        self.border_alpha = 50
        self.glow_intensity = 0  # 发光强度 0-20
        self.is_glowing = False
        
    def set_background_alpha(self, alpha):
        """设置背景透明度"""
        self.bg_alpha = alpha
        self.update()
        
    def set_border_alpha(self, alpha):
        """设置边框透明度"""
        self.border_alpha = alpha
        self.update()
        
    def set_glow_intensity(self, intensity):
        """设置发光强度 0-20"""
        self.glow_intensity = max(0, min(20, intensity))
        self.update()
        
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
        s.input_wrap.setFixedHeight(48)
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
        vlay.addWidget(s.input_wrap,0)
        
        # 将聊天区域添加到分割器
        s.main_splitter.addWidget(chat_area)
        
        # 侧栏（图片显示区域）- 使用自定义动画Widget
        s.side = AnimatedSideWidget()
        s.side.setMinimumWidth(300)  # 设置最小宽度
        s.side.setMaximumWidth(800)  # 设置最大宽度
        
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
        nick=QLabel(f"● 娜迦{config.NAGA_VERSION}",s.side)
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
        stack.addWidget(nick)
        
        # 将侧栏添加到分割器
        s.main_splitter.addWidget(s.side)
        
        # 设置分割器的初始比例
        s.main_splitter.setSizes([window_width * 2 // 3, window_width // 3])  # 2:1的比例
        
        # 创建包含分割器的主布局
        main=QVBoxLayout(s)
        main.setContentsMargins(10,110,10,10)
        main.addWidget(s.main_splitter)
        
        s.nick=nick
        s.naga=NagaConversation()
        s.worker=None
        s.full_img=0 # 立绘展开标志
        s.streaming_mode = True  # 默认启用流式模式
        s.current_response = ""  # 当前响应缓冲
        s._animating = False  # 动画标志位，动画期间为True
        
        # 连接进度组件信号
        s.progress_widget.cancel_requested.connect(s.cancel_current_task)
        
        s.input.textChanged.connect(s.adjust_input_height)
        s.input.installEventFilter(s)
        s.setLayout(main)
        s.titlebar = TitleBar('NAGA AGENT', s)
        s.titlebar.setGeometry(0,0,s.width(),100)
        s.side.mousePressEvent=s.toggle_full_img # 侧栏点击切换聊天/设置

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
        
        # 滚动内容
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(12, 12, 12, 12)
        scroll_layout.setSpacing(20)
        
        # 性格配置区域标题
        personality_title = QLabel("AI 性格配置")
        personality_title.setStyleSheet("""
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
        scroll_layout.addWidget(personality_title)
        
        # 添加性格选择器
        s.personality_selector = ElegantPersonalitySelector(scroll_content)
        s.personality_selector.setStyleSheet("""
            ElegantPersonalitySelector {
                background: transparent;
                border: none;
                margin: 0 0 30px 0;
                padding: 10px;
            }
        """)
        scroll_layout.addWidget(s.personality_selector)
        
        # 连接性格选择器信号
        s.personality_selector.personality_changed.connect(s.on_personality_changed)
        
        # 添加完整的系统设置界面
        s.settings_widget = ElegantSettingsWidget(scroll_content)
        s.settings_widget.settings_changed.connect(s.on_settings_changed)
        scroll_layout.addWidget(s.settings_widget)
        
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
        
        return page

    def resizeEvent(s, e):
        if getattr(s, '_animating', False):  # 动画期间跳过自适应刷新，提升动画流畅度
            return
        if hasattr(s,'img') and hasattr(s,'nick'):
            s.img.resize(s.img.parent().width(), s.img.parent().height())
            s.nick.resize(s.img.width(), 48) # 48为昵称高度，可自调
            s.nick.move(0,0)
            p=os.path.join(os.path.dirname(__file__),'standby.png')
            q=QPixmap(p)
            if os.path.exists(p) and not q.isNull():
                s.img.setPixmap(q.scaled(s.img.width(),s.img.height(),Qt.KeepAspectRatioByExpanding,Qt.SmoothTransformation))

    def adjust_input_height(s):
        doc = s.input.document()
        h = int(doc.size().height())+10
        s.input.setFixedHeight(min(max(48, h), 120))
        s.input_wrap.setFixedHeight(s.input.height())

    def eventFilter(s, obj, event):
        if obj is s.input and event.type()==6:
            if event.key()==Qt.Key_Return and not (event.modifiers()&Qt.ShiftModifier):
                s.on_send();return True
        return False
    def add_user_message(s, name, content):
        # 先把\n转成\n，再把\n转成<br>，适配所有换行
        content_html = str(content).replace('\\n', '\n').replace('\n', '<br>')
        s.text.append(f"<span style='color:#fff;font-size:12pt;font-family:Lucida Console;'>{name}</span>")
        s.text.append(f"<span style='color:#fff;font-size:16pt;font-family:Lucida Console;'>{content_html}</span>")
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
        """追加响应片段（流式模式）"""
        s.current_response += chunk
        # 实时更新显示（可选，避免过于频繁的更新）
        # s.update_last_message("娜迦", s.current_response)
    
    def finalize_streaming_response(s):
        """完成流式响应"""
        if s.current_response:
            # 对累积的完整响应进行消息提取
            from ui.response_utils import extract_message
            final_message = extract_message(s.current_response)
            s.add_user_message("娜迦", final_message)
        s.progress_widget.stop_loading()
    
    def on_response_finished(s, response):
        """处理完成的响应（流式模式后备）"""
        # 检查是否是取消操作的响应
        if response == "操作已取消":
            return  # 不显示，因为已经在cancel_current_task中显示了
        
        if not s.current_response:  # 如果流式没有收到数据，使用最终结果
            s.add_user_message("娜迦", response)
        s.progress_widget.stop_loading()
    
    def on_batch_response_finished(s, response):
        """处理完成的响应（批量模式）"""
        # 检查是否是取消操作的响应
        if response == "操作已取消":
            return  # 不显示，因为已经在cancel_current_task中显示了
            
        s.add_user_message("娜迦", response)
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
        
        # 获取当前分割器尺寸
        current_sizes = s.main_splitter.sizes()
        total_width = sum(current_sizes)
        
        # --- 立即切换界面状态 ---
        if s.full_img:
            # 展开模式：设置区域占大部分空间
            target_sizes = [total_width // 4, total_width * 3 // 4]  # 1:3比例
            s.input_wrap.hide()  # 立即隐藏输入框
            s.chat_stack.setCurrentIndex(1)  # 立即切换到设置页
            s.side.setCursor(Qt.ArrowCursor)  # 放大模式下恢复普通指针
            s.titlebar.text = "SETTING PAGE"
            s.titlebar.update()
            # 使用QPainter绘制替代QSS
            s.side.set_background_alpha(150)
            s.side.set_border_alpha(80)
            s.side.enterEvent = s.side.leaveEvent = lambda e: None
        else:
            # 收起模式：恢复正常比例
            target_sizes = [total_width * 2 // 3, total_width // 3]  # 2:1比例
            s.input_wrap.show()  # 立即显示输入框
            s.chat_stack.setCurrentIndex(0)  # 立即切换到聊天页
            s.input.setFocus()  # 恢复输入焦点
            s.side.setCursor(Qt.PointingHandCursor)  # 恢复点击指针
            s.titlebar.text = "NAGA AGENT"
            s.titlebar.update()
            # 恢复正常样式
            s.side.set_background_alpha(int(BG_ALPHA * 255))
            s.side.set_border_alpha(50)
            s.side.enterEvent = s.side_hover_enter
            s.side.leaveEvent = s.side_hover_leave
        # --- 立即切换界面状态 END ---
        
        # 创建动画组
        group = QParallelAnimationGroup(s)
        
        # 使用简单的Timer分割器动画来避免QPropertyAnimation的target问题
        s.animation_progress = 0.0
        s.animation_start_sizes = current_sizes
        s.animation_target_sizes = target_sizes
        
        def update_splitter_sizes():
            if s.animation_progress >= 1.0:
                s.main_splitter.setSizes(s.animation_target_sizes)
                s.splitter_timer.stop()
                return
            
            # 使用OutExpo缓动函数
            eased_progress = 1 - (1 - s.animation_progress) ** 3
            
            current_anim_sizes = []
            for start, end in zip(s.animation_start_sizes, s.animation_target_sizes):
                current = start + (end - start) * eased_progress
                current_anim_sizes.append(int(current))
            
            s.main_splitter.setSizes(current_anim_sizes)
            s.animation_progress += 1.0 / (ANIMATION_DURATION / 16.0)  # 60fps
        
        s.splitter_timer = QTimer()
        s.splitter_timer.timeout.connect(update_splitter_sizes)
        s.splitter_timer.start(16)  # 约60fps
        
        # 输入框高度动画
        input_hide_anim = QPropertyAnimation(s.input_wrap, b"maximumHeight", s)
        input_hide_anim.setDuration(ANIMATION_DURATION // 3)
        input_hide_anim.setStartValue(s.input_wrap.height())
        input_hide_anim.setEndValue(0 if s.full_img else 48)
        input_hide_anim.setEasingCurve(QEasingCurve.InOutQuart)
        group.addAnimation(input_hide_anim)
        
        # 输入框透明度动画
        input_opacity_anim = QPropertyAnimation(s.input, b"windowOpacity", s)
        input_opacity_anim.setDuration(ANIMATION_DURATION // 4)
        input_opacity_anim.setStartValue(1.0)
        input_opacity_anim.setEndValue(0.0 if s.full_img else 1.0)
        input_opacity_anim.setEasingCurve(QEasingCurve.InOutQuart)
        group.addAnimation(input_opacity_anim)
        
        # 图片缩放动画 - 高性能优化版本
        p = os.path.join(os.path.dirname(__file__), 'standby.png')
        if os.path.exists(p):
            original_pixmap = QPixmap(p)
            if not original_pixmap.isNull():
                # 预生成不同尺寸的图片，避免实时缩放
                s.img_animation_progress = 0.0
                s.original_img_rect = s.img.geometry()
                target_side_width = target_sizes[1]
                s.target_img_rect = QRect(0, 0, target_side_width, s.side.height())
                
                # 预生成起始和目标尺寸的高质量图片
                start_w, start_h = s.original_img_rect.width(), s.original_img_rect.height()
                target_w, target_h = s.target_img_rect.width(), s.target_img_rect.height()
                
                if s.full_img:
                    # 展开：预生成大尺寸图片
                    s.cached_start_pixmap = original_pixmap.scaled(
                        start_w, start_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    s.cached_target_pixmap = original_pixmap.scaled(
                        target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                else:
                    # 收起：预生成小尺寸图片
                    s.cached_start_pixmap = original_pixmap.scaled(
                        start_w, start_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                    s.cached_target_pixmap = original_pixmap.scaled(
                        target_w, target_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                
                # 创建流畅的图片更新函数（与分割器动画同步）
                def update_image_smoothly():
                    if s.img_animation_progress >= 1.0:
                        s.img_timer.stop()
                        return
                    
                    # 使用与分割器相同的缓动函数，确保同步
                    eased_progress = 1 - (1 - s.img_animation_progress) ** 3
                    
                    # 计算当前几何位置（与分割器动画同步）
                    current_x = s.original_img_rect.x() + (s.target_img_rect.x() - s.original_img_rect.x()) * eased_progress
                    current_y = s.original_img_rect.y() + (s.target_img_rect.y() - s.original_img_rect.y()) * eased_progress
                    current_w = s.original_img_rect.width() + (s.target_img_rect.width() - s.original_img_rect.width()) * eased_progress
                    current_h = s.original_img_rect.height() + (s.target_img_rect.height() - s.original_img_rect.height()) * eased_progress
                    
                    # 实时更新几何位置
                    s.img.setGeometry(int(current_x), int(current_y), int(current_w), int(current_h))
                    
                    # 智能图片更新策略 - 减少缩放次数
                    if eased_progress < 0.1 or eased_progress > 0.9:
                        # 动画开始和结束时使用预生成的高质量图片
                        if eased_progress < 0.5:
                            s.img.setPixmap(s.cached_start_pixmap)
                        else:
                            s.img.setPixmap(s.cached_target_pixmap)
                    else:
                        # 中间过程：使用快速缩放（降低质量但提高性能）
                        if s.full_img:
                            scaled_pixmap = original_pixmap.scaled(
                                int(current_w), int(current_h), 
                                Qt.KeepAspectRatio, Qt.FastTransformation  # 使用快速变换
                            )
                        else:
                            scaled_pixmap = original_pixmap.scaled(
                                int(current_w), int(current_h), 
                                Qt.KeepAspectRatioByExpanding, Qt.FastTransformation  # 使用快速变换
                            )
                        s.img.setPixmap(scaled_pixmap)
                    
                    # 简化透明度效果，减少计算
                    if s.full_img and eased_progress < 0.2:
                        # 展开时轻微淡入效果
                        opacity = 0.95 + 0.05 * (eased_progress / 0.2)
                        s.img.setWindowOpacity(opacity)
                    else:
                        s.img.setWindowOpacity(1.0)
                    
                    # 与分割器动画保持完全同步的进度更新
                    s.img_animation_progress += 1.0 / (ANIMATION_DURATION / 16.0)  # 60fps
                
                # 启动图片动画定时器
                s.img_timer = QTimer()
                s.img_timer.timeout.connect(update_image_smoothly)
                s.img_timer.start(16)  # 60fps
                
                # 添加侧栏发光动画效果 - 使用QPainter绘制
                if s.full_img:
                    # 展开时添加发光效果
                    s.side_glow_progress = 0.0
                    def update_side_glow():
                        if s.side_glow_progress >= 1.0:
                            s.side_glow_timer.stop()
                            return
                        
                        # 计算发光强度和边框透明度
                        border_alpha = int(50 + 30 * s.side_glow_progress)  # 50到80的渐变
                        glow_intensity = int(20 * s.side_glow_progress)     # 0到20的发光强度
                        
                        # 使用QPainter方法更新样式
                        s.side.set_border_alpha(border_alpha)
                        s.side.set_glow_intensity(glow_intensity)
                        
                        s.side_glow_progress += 1.0 / (ANIMATION_DURATION / 16.0)
                    
                    s.side_glow_timer = QTimer()
                    s.side_glow_timer.timeout.connect(update_side_glow)
                    s.side_glow_timer.start(16)
        
        def on_animation_finished():
            # 确保所有动画定时器完全停止
            if hasattr(s, 'splitter_timer'):
                s.splitter_timer.stop()
            if hasattr(s, 'img_timer'):
                s.img_timer.stop()
            if hasattr(s, 'side_glow_timer'):
                s.side_glow_timer.stop()
            
            # 最终设置精确的分割器尺寸
            s.main_splitter.setSizes(target_sizes)
            
            # 动画完成后的图片最终处理
            p = os.path.join(os.path.dirname(__file__), 'standby.png')
            if os.path.exists(p):
                q = QPixmap(p)
                if not q.isNull():
                    side_width = s.side.width()
                    side_height = s.side.height()
                    
                    # 确保图片几何位置准确
                    s.img.setGeometry(0, 0, side_width, side_height)
                    
                    if s.full_img:
                        # 展开模式：适应容器，高画质
                        final_pixmap = q.scaled(side_width, side_height, 
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        s.img.setPixmap(final_pixmap)
                        s.img.setWindowOpacity(1.0)  # 确保完全不透明
                        # 展开时设置更大的最大宽度
                        s.side.setMaximumWidth(1200)
                        # 最终的发光效果
                        s.side.set_background_alpha(150)
                        s.side.set_border_alpha(80)
                        s.side.set_glow_intensity(15)  # 保持轻微发光
                    else:
                        # 收起模式：填充容器
                        final_pixmap = q.scaled(side_width, side_height, 
                                              Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                        s.img.setPixmap(final_pixmap)
                        s.img.setWindowOpacity(1.0)  # 确保完全不透明
                        # 收起时恢复原来的最大宽度
                        s.side.setMaximumWidth(800)
                        # 恢复原始样式
                        s.side.set_background_alpha(int(BG_ALPHA * 255))
                        s.side.set_border_alpha(50)
                        s.side.set_glow_intensity(0)  # 取消发光
            
            s._animating = False  # 动画结束，允许后续操作
            s.resizeEvent(None)  # 动画结束后手动刷新一次，保证布局和图片同步
        
        # 延迟执行完成回调，确保所有动画都完成
        QTimer.singleShot(ANIMATION_DURATION + 100, on_animation_finished)
        
        group.finished.connect(lambda: None)  # 空回调，避免冲突
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

    def on_personality_changed(s, personality_code, personality_config):
        """处理性格切换"""
        print(f"性格切换到: {personality_code} - {personality_config.get('name', '')}")
        
        # 更新娜迦对话实例的系统提示
        if hasattr(s.naga, 'set_personality'):
            s.naga.set_personality(personality_code, personality_config)
        else:
            # 如果NagaConversation没有set_personality方法，更新系统消息
            if personality_code != "DEFAULT" and 'prompt' in personality_config:
                s.naga.system_message = personality_config['prompt']
            else:
                # 恢复默认系统消息
                s.naga.system_message = "你是一个helpful的AI助手，名叫娜迦(Naga)。"
        
        # 在聊天记录中显示性格切换提示
        personality_name = personality_config.get('name', personality_code)
        s.text.append(f"<span style='color:#64C8FF;font-size:12pt;font-family:Lucida Console;font-style:italic;'>● 性格模式已切换为: {personality_name}</span>")

    def on_settings_changed(s, setting_key, value):
        """处理设置变化"""
        print(f"设置变化: {setting_key} = {value}")
        
        # 这里可以实时应用某些设置变化
        if setting_key == "STREAM_MODE":
            s.streaming_mode = value
            s.add_user_message("系统", f"● 流式模式已{'启用' if value else '禁用'}")
        elif setting_key == "BG_ALPHA":
            # 实时更新背景透明度
            global BG_ALPHA
            BG_ALPHA = value / 100.0
            # 这里可以添加实时更新UI的代码
        elif setting_key == "VOICE_ENABLED":
            s.add_user_message("系统", f"● 语音功能已{'启用' if value else '禁用'}")
        elif setting_key == "DEBUG":
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

if __name__=="__main__":
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec_())
