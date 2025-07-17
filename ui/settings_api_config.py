import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import config # 导入全局配置

def read_api_key():
    # 优先读取.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('API_KEY'):
                    return line.strip().split('=', 1)[-1].strip()
    # 退回读取config.py
    return str(getattr(config, 'API_KEY', ''))

def write_api_key(new_key):
    # 写入.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    env_lines = []
    found = False
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            env_lines = f.readlines()
        for i, line in enumerate(env_lines):
            if line.strip().startswith('API_KEY'):
                env_lines[i] = f'API_KEY={new_key}\n'
                found = True
                break
    if not found:
        env_lines.append(f'API_KEY={new_key}\n')
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(env_lines)
    # 写入config.py
    config_path = os.path.join(os.path.dirname(config.__file__), 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith('API_KEY'):
            lines[i] = f"API_KEY = '{new_key}'\n"
            break
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

class ApiConfigWidget(QWidget):
    def __init__(s, parent=None):
        super().__init__(parent)
        s.setWindowTitle("API配置")
        s.setStyleSheet("border-radius:24px;")
        layout = QVBoxLayout(s)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题
        title = QLabel("API配置", s)
        title.setFont(QFont("Consolas", 20, QFont.Bold))  # 字号略小
        title.setStyleSheet("color:#fff;margin:0;padding:0;line-height:16px;")
        layout.addWidget(title, alignment=Qt.AlignLeft | Qt.AlignTop)

        # 输入框
        s.api_input = QLineEdit(s)
        s.api_input.setText(read_api_key())
        s.api_input.setStyleSheet(
            "background:#222;color:#fff;font:16pt 'Consolas';"
            "border-radius:8px;padding:6px 12px 6px 12px;margin:0;margin-top:-6px;border:none;"
        )
        layout.addWidget(s.api_input)

        # 保存按钮
        save_btn = QPushButton("保存", s)
        save_btn.setStyleSheet("background:#444;color:#fff;font:14pt 'Consolas';border-radius:8px;padding:6px 18px;")
        save_btn.clicked.connect(s.save_api_key)
        layout.addWidget(save_btn, alignment=Qt.AlignRight)

    def save_api_key(s):
        new_key = s.api_input.text().strip()
        write_api_key(new_key)
        s.api_input.setText(new_key)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ApiConfigWidget()
    win.show()
    sys.exit(app.exec_()) 
