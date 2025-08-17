# -*- coding: utf-8 -*-
import asyncio
import logging
import httpx
from config import config
import os

logger = logging.getLogger(__name__)

class GPTSoVITS_TTS:
    """
    封装 GPT-SoVITS API 的调用
    """
    def __init__(self):
        # 从配置中读取GPT-SoVITS的相关设置
        gpt_sovits_config = config.tts.GPT_SoVITS
        self.is_enabled = gpt_sovits_config.is_enabled
        self.api_url = gpt_sovits_config.gpt_sovits_api_url
        self.ref_audio_path = gpt_sovits_config.gpt_sovits_refer_wav_path
        self.aux_ref_audio_paths = gpt_sovits_config.aux_ref_audio_paths
        self.prompt_text = gpt_sovits_config.gpt_sovits_prompt_text
        self.prompt_lang = gpt_sovits_config.gpt_sovits_prompt_language
        self.text_lang = gpt_sovits_config.gpt_sovits_text_language
        
        # 高级参数
        self.top_k = gpt_sovits_config.top_k
        self.top_p = gpt_sovits_config.top_p
        self.temperature = gpt_sovits_config.temperature
        self.text_split_method = gpt_sovits_config.text_split_method
        self.batch_size = gpt_sovits_config.batch_size
        self.speed_factor = gpt_sovits_config.speed_factor

    async def speak(self, text: str) -> bytes:
        """
        调用 GPT-SoVITS API 生成语音

        Args:
            text (str): 要转换为语音的文本.

        Returns:
            bytes: 音频数据, 或者在失败时返回 None.
        """
        if not self.is_enabled:
            logger.warning("GPT-SoVITS TTS 未启用，跳过语音合成。")
            return None
            
        required_params = {
            "api_url": self.api_url,
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "text_lang": self.text_lang
        }
        
        missing_params = [k for k, v in required_params.items() if not v]
        if missing_params:
            logger.error(f"GPT-SoVITS 配置不完整，缺少必要的参数: {', '.join(missing_params)}")
            return None

        if not os.path.exists(self.ref_audio_path):
            logger.error(f"主参考音频文件不存在: {self.ref_audio_path}")
            return None

        headers = {
            "accept": "audio/wav",
            "Content-Type": "application/json"
        }
        
        # 构建完整的请求数据
        data = {
            "text": text,
            "text_lang": self.text_lang,
            "ref_audio_path": self.ref_audio_path,
            "aux_ref_audio_paths": self.aux_ref_audio_paths,
            "prompt_text": self.prompt_text,
            "prompt_lang": self.prompt_lang,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "speed_factor": self.speed_factor,
            "media_type": "wav",
            "streaming_mode": True,
            # 附加参数
            "batch_threshold": 0.75,
            "split_bucket": True,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 32,
            "super_sampling": False
        }

        try:
            total_chunks = []
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", self.api_url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        logger.debug(f"收到音频数据块: {len(chunk)} 字节")
                        total_chunks.append(chunk)
            
            full_audio = b''.join(total_chunks)
            if not full_audio:
                logger.error("未收到任何音频数据")
                return None
                
            logger.info(f"音频合成完成: {len(full_audio)} 字节")
            return full_audio
            
        except httpx.RequestError as exc:
            logger.error(f"调用 GPT-SoVITS API 时发生网络错误: {exc}")
        except httpx.HTTPStatusError as exc:
            logger.error(f"GPT-SoVITS API 返回错误状态: {exc.response.status_code} - {exc.response.text}")
        except IOError as exc:
            logger.error(f"读取参考音频文件时出错: {exc}")
        except Exception as exc:
            logger.error(f"生成语音时发生未知错误: {exc}")
            
        return None

async def main():
    """
    测试函数
    """
    # 需要确保config已正确加载
    from config import load_config
    load_config()
    
    tts = GPTSoVITS_TTS()
    if not tts.is_enabled:
        print("测试跳过：GPT-SoVITS 未在配置中启用。")
        return

    try:
        print("开始生成语音...")
        audio_data = await tts.speak("你好，这是一个使用新的接口实现的测试。")
        if audio_data:
            with open("test_output_gsv.wav", "wb") as f:
                f.write(audio_data)
            print("测试音频已保存到 test_output_gsv.wav")
        else:
            print("测试失败，未能获取音频数据。")
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
