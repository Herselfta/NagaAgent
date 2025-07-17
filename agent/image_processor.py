# agent/image_processor.py
# 图片处理器插件
import base64
import io
import re
from typing import List, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger("ImageProcessor")

class ImageProcessor:
    """图片处理器 - 处理消息中的图片内容"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.max_image_size = self.config.get('max_image_size', 1024)  # 最大图片尺寸
        self.quality = self.config.get('quality', 85)  # JPEG质量
        self.debug_mode = self.config.get('DebugMode', False)
    
    async def process_messages(self, messages: List[Dict], config: Dict = None) -> List[Dict]:
        """处理消息中的图片内容"""
        if self.debug_mode:
            logger.info(f"开始处理 {len(messages)} 条消息中的图片")
        
        processed_messages = []
        for msg in messages:
            processed_msg = await self._process_single_message(msg)
            processed_messages.append(processed_msg)
        
        if self.debug_mode:
            logger.info("图片处理完成")
        
        return processed_messages
    
    async def _process_single_message(self, message: Dict) -> Dict:
        """处理单条消息中的图片"""
        processed_msg = message.copy()
        
        # 处理字符串内容中的图片URL
        if isinstance(processed_msg.get('content'), str):
            processed_msg['content'] = await self._process_text_content(processed_msg['content'])
        
        # 处理多模态内容
        elif isinstance(processed_msg.get('content'), list):
            processed_content = []
            for part in processed_msg['content']:
                if isinstance(part, dict):
                    processed_part = await self._process_content_part(part)
                    processed_content.append(processed_part)
                else:
                    processed_content.append(part)
            processed_msg['content'] = processed_content
        
        return processed_msg
    
    async def _process_text_content(self, text: str) -> str:
        """处理文本内容中的图片URL"""
        # 查找图片URL模式
        image_url_pattern = r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|webp|bmp)'
        
        def replace_image_url(match):
            image_url = match.group(0)
            try:
                # 这里可以下载图片并转换为base64
                # 暂时返回原始URL
                return f"[图片: {image_url}]"
            except Exception as e:
                logger.error(f"处理图片URL失败: {e}")
                return f"[图片处理失败: {image_url}]"
        
        text = re.sub(image_url_pattern, replace_image_url, text)
        return text
    
    async def _process_content_part(self, part: Dict) -> Dict:
        """处理内容部分"""
        if part.get('type') == 'image_url':
            # 处理图片URL部分
            image_url = part.get('image_url', {})
            if isinstance(image_url, dict):
                url = image_url.get('url', '')
                if url.startswith('data:image/'):
                    # 已经是base64格式
                    return part
                else:
                    # 需要转换为base64
                    try:
                        base64_data = await self._url_to_base64(url)
                        part['image_url']['url'] = base64_data
                    except Exception as e:
                        logger.error(f"转换图片URL失败: {e}")
                        part['image_url']['url'] = f"data:image/png;base64,{base64.b64encode(b'[Image load failed]').decode()}"
        
        elif part.get('type') == 'text':
            # 处理文本部分中的图片引用
            text = part.get('text', '')
            part['text'] = await self._process_text_content(text)
        
        return part
    
    async def _url_to_base64(self, url: str) -> str:
        """将图片URL转换为base64格式"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        # 处理图片
                        processed_image_data = await self._process_image_data(image_data)
                        # 转换为base64
                        base64_data = base64.b64encode(processed_image_data).decode()
                        return f"data:image/png;base64,{base64_data}"
                    else:
                        raise Exception(f"HTTP {response.status}")
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            raise
    
    async def _process_image_data(self, image_data: bytes) -> bytes:
        """处理图片数据（调整大小、压缩等）"""
        try:
            # 打开图片
            image = Image.open(io.BytesIO(image_data))
            
            # 调整大小
            if max(image.size) > self.max_image_size:
                ratio = self.max_image_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # 转换为RGB模式（如果需要）
            if image.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            
            # 保存为JPEG格式
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=self.quality, optimize=True)
            return output_buffer.getvalue()
        
        except Exception as e:
            logger.error(f"处理图片数据失败: {e}")
            # 返回原始数据
            return image_data
    
    def extract_image_urls(self, text: str) -> List[str]:
        """从文本中提取图片URL"""
        image_url_pattern = r'https?://[^\s<>"]+\.(?:jpg|jpeg|png|gif|webp|bmp)'
        return re.findall(image_url_pattern, text)
    
    def is_image_url(self, url: str) -> bool:
        """判断是否为图片URL"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp')
        return any(url.lower().endswith(ext) for ext in image_extensions)

# 插件接口函数
async def processMessages(messages: List[Dict], config: Dict = None) -> List[Dict]:
    """插件接口函数 - 处理消息中的图片"""
    processor = ImageProcessor(config)
    return await processor.process_messages(messages, config)

async def initialize(config: Dict = None):
    """插件初始化函数"""
    logger.info("图片处理器插件已初始化")
    return True

async def shutdown():
    """插件关闭函数"""
    logger.info("图片处理器插件已关闭")
    return True 