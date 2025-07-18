#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频系统诊断工具
用于检测和修复音频播放问题
"""
import os
import sys
import subprocess
import platform


def check_audio_devices():
    """检查音频设备"""
    print("=== 音频设备检查 ===")
    
    system = platform.system()
    
    if system == "Linux":
        # 检查ALSA设备
        try:
            result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ ALSA音频设备:")
                print(result.stdout)
            else:
                print("❌ ALSA设备检查失败")
        except Exception as e:
            print(f"❌ ALSA检查异常: {e}")
        
        # 检查PulseAudio
        try:
            result = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ PulseAudio可用")
            else:
                print("❌ PulseAudio不可用")
        except Exception as e:
            print(f"❌ PulseAudio检查异常: {e}")
            
    elif system == "Windows":
        print("Windows音频设备检查...")
        # Windows音频检查
        pass
    elif system == "Darwin":
        print("macOS音频设备检查...")
        # macOS音频检查
        pass


def check_audio_players():
    """检查可用的音频播放器"""
    print("\n=== 音频播放器检查 ===")
    
    players = [
        "mpv", "aplay", "paplay", "mpg123", 
        "vlc", "ffplay", "cvlc", "mplayer"
    ]
    
    available_players = []
    
    for player in players:
        try:
            result = subprocess.run([player, "--version"], 
                                  capture_output=True, timeout=3)
            if result.returncode == 0:
                print(f"✅ {player} 可用")
                available_players.append(player)
            else:
                print(f"❌ {player} 不可用")
        except FileNotFoundError:
            print(f"❌ {player} 未安装")
        except Exception as e:
            print(f"❌ {player} 检查失败: {e}")
    
    return available_players


def check_python_audio_libs():
    """检查Python音频库"""
    print("\n=== Python音频库检查 ===")
    
    libs = [
        ("pydub", "音频处理"),
        ("pyaudio", "音频播放"),
        ("pygame", "游戏/音频库"),
        ("playsound", "简单音频播放"),
        ("simpleaudio", "简单音频播放")
    ]
    
    available_libs = []
    
    for lib, desc in libs:
        try:
            __import__(lib)
            print(f"✅ {lib} 可用 ({desc})")
            available_libs.append(lib)
        except ImportError:
            print(f"❌ {lib} 未安装 ({desc})")
    
    return available_libs


def test_audio_playback():
    """测试音频播放"""
    print("\n=== 音频播放测试 ===")
    
    # 生成测试音频文件
    try:
        # 尝试使用pydub生成测试音频
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # 生成1秒的440Hz正弦波
        tone = Sine(440).to_audio_segment(duration=1000)
        test_file = "audio_test.mp3"
        tone.export(test_file, format="mp3")
        print(f"✅ 生成测试文件: {test_file}")
        
        # 测试播放
        try:
            from pydub.playback import play
            print("测试pydub播放...")
            play(tone)
            print("✅ pydub播放成功")
        except Exception as e:
            print(f"❌ pydub播放失败: {e}")
            
            # 尝试系统播放器
            system = platform.system()
            if system == "Linux":
                players = ["mpv", "aplay", "paplay"]
                for player in players:
                    try:
                        subprocess.run([player, test_file], 
                                     check=True, timeout=3)
                        print(f"✅ {player} 播放成功")
                        break
                    except Exception:
                        continue
                else:
                    print("❌ 所有播放器都失败")
        
        # 清理测试文件
        try:
            os.unlink(test_file)
        except:
            pass
            
    except ImportError:
        print("❌ pydub未安装，无法生成测试音频")


def provide_solutions():
    """提供解决方案"""
    print("\n=== 解决方案建议 ===")
    
    system = platform.system()
    
    if system == "Linux":
        print("Linux音频问题解决方案:")
        print("1. 安装音频播放器:")
        print("   sudo apt install mpv")
        print("   sudo apt install alsa-utils")
        print("   sudo apt install pulseaudio")
        print()
        print("2. 检查音频服务:")
        print("   systemctl --user status pulseaudio")
        print("   pulseaudio --start")
        print()
        print("3. 安装Python音频库:")
        print("   pip install pydub[playback]")
        print("   pip install pyaudio")
        print()
        print("4. 如果在WSL中:")
        print("   需要安装PulseAudio并配置X11转发")
        print("   或使用Windows端的音频播放器")
        
    elif system == "Windows":
        print("Windows音频问题解决方案:")
        print("1. 检查音频驱动")
        print("2. 安装Python音频库:")
        print("   pip install pydub[playback]")
        print("   pip install pyaudio")
        
    elif system == "Darwin":
        print("macOS音频问题解决方案:")
        print("1. 检查音频权限")
        print("2. 安装Python音频库:")
        print("   pip install pydub[playback]")


def check_wsl():
    """检查是否在WSL环境"""
    print("\n=== WSL环境检查 ===")
    
    try:
        with open('/proc/version', 'r') as f:
            version = f.read()
        if 'Microsoft' in version or 'WSL' in version:
            print("✅ 检测到WSL环境")
            print("WSL中的音频播放需要特殊配置:")
            print("1. 安装PulseAudio: sudo apt install pulseaudio")
            print("2. 配置音频服务")
            print("3. 或者保存音频文件到Windows文件系统手动播放")
            return True
        else:
            print("❌ 不是WSL环境")
            return False
    except Exception:
        print("❌ 无法检测WSL环境")
        return False


def main():
    """主函数"""
    print("🎵 NagaAgent 音频系统诊断工具")
    print("=" * 50)
    
    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {sys.version}")
    print()
    
    # 检查WSL
    is_wsl = check_wsl()
    
    # 检查音频设备
    check_audio_devices()
    
    # 检查播放器
    available_players = check_audio_players()
    
    # 检查Python库
    available_libs = check_python_audio_libs()
    
    # 测试播放
    if available_libs:
        test_audio_playback()
    
    # 提供解决方案
    provide_solutions()
    
    # 总结
    print("\n=== 诊断总结 ===")
    if available_players:
        print(f"✅ 可用播放器: {', '.join(available_players)}")
    else:
        print("❌ 没有找到可用的音频播放器")
    
    if available_libs:
        print(f"✅ 可用Python库: {', '.join(available_libs)}")
    else:
        print("❌ 没有找到可用的Python音频库")
    
    if is_wsl:
        print("⚠️  WSL环境需要特殊配置")
    
    print("\n建议:")
    if not available_players and not available_libs:
        print("1. 安装音频播放器 (推荐mpv)")
        print("2. 安装Python音频库 (推荐pydub)")
    elif "pydub" not in available_libs:
        print("1. 安装pydub: pip install pydub[playback]")
    
    print("\n如果问题仍然存在，音频文件会保存到本地供手动播放。")


if __name__ == "__main__":
    main()
