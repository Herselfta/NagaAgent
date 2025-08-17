# -*- coding: utf-8 -*-
from config import config
from voice.gpt_sovits_tts import GPTSoVITS_TTS
# 可以在这里导入其他的TTS处理器，例如 from voice.azure_tts import AzureTTS

class TTSFactory:
    """
    TTS处理器工厂类，根据配置创建并返回相应的TTS处理器实例。
    """
    @staticmethod
    def create_tts_handler():
        """
        根据配置文件中的 TTS_CHOICE 创建并返回一个TTS处理器。
        """
        tts_choice = getattr(config.tts, "TTS_CHOICE", "DISABLE").upper()

        if tts_choice == "GPT-SOVITS":
            # 检查GPT-SoVITS是否在配置中启用
            gpt_sovits_config = getattr(config.tts, "GPT_SoVITS", None)
            if gpt_sovits_config and getattr(gpt_sovits_config, "is_enabled", False):
                return GPTSoVITS_TTS()
            else:
                return None  # 如果未启用，则不返回处理器
        
        # elif tts_choice == "AZURE":
        #     return AzureTTS() # 示例：返回Azure TTS处理器
        
        # 默认情况下，或当TTS_CHOICE为DISABLE时，不返回任何处理器
        return None

# 创建一个全局的tts_handler实例
tts_handler = TTSFactory.create_tts_handler()
