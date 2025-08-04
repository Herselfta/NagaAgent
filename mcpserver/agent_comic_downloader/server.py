"""
Comic Downloader MCP Server
提供HTTP API接口的MCP服务器
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .comic_downloader_agent import (
    download_comic_tool,
    get_download_status_tool,
    cancel_download_tool,
    get_all_status_tool
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Comic Downloader MCP Server",
    description="A simplified comic downloader that downloads comics to desktop",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class DownloadRequest(BaseModel):
    album_id: str

class StatusRequest(BaseModel):
    album_id: str

class CancelRequest(BaseModel):
    album_id: str

# 响应模型
class DownloadResponse(BaseModel):
    success: bool
    album_id: str
    message: str
    album_title: Optional[str] = None
    album_author: Optional[str] = None
    download_path: Optional[str] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    progress: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class CancelResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None

# API路由
@app.post("/download", response_model=DownloadResponse)
async def download_comic(request: DownloadRequest):
    """
    下载漫画
    
    Args:
        request: 包含漫画ID的请求
        
    Returns:
        下载结果
    """
    try:
        logger.info(f"收到下载请求: {request.album_id}")
        result = await download_comic_tool(request.album_id)
        
        return DownloadResponse(
            success=result.get('success', False),
            album_id=request.album_id,
            message=result.get('message', ''),
            album_title=result.get('result', {}).get('album_title'),
            album_author=result.get('result', {}).get('album_author'),
            download_path=result.get('result', {}).get('download_path'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"下载漫画时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{album_id}", response_model=StatusResponse)
async def get_download_status(album_id: str):
    """
    获取下载状态
    
    Args:
        album_id: 漫画ID
        
    Returns:
        下载状态
    """
    try:
        logger.info(f"获取下载状态: {album_id}")
        result = await get_download_status_tool(album_id)
        
        return StatusResponse(
            status=result.get('status', 'unknown'),
            message=result.get('message', ''),
            progress=result.get('progress'),
            result=result.get('result'),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"获取下载状态时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel", response_model=CancelResponse)
async def cancel_download(request: CancelRequest):
    """
    取消下载
    
    Args:
        request: 包含漫画ID的请求
        
    Returns:
        取消结果
    """
    try:
        logger.info(f"取消下载: {request.album_id}")
        result = await cancel_download_tool(request.album_id)
        
        return CancelResponse(
            success=result.get('success', False),
            message=result.get('message', ''),
            error=result.get('error')
        )
        
    except Exception as e:
        logger.error(f"取消下载时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status", response_model=Dict[str, Any])
async def get_all_status():
    """
    获取所有下载状态
    
    Returns:
        所有任务状态
    """
    try:
        logger.info("获取所有下载状态")
        result = await get_all_status_tool()
        return result
        
    except Exception as e:
        logger.error(f"获取所有状态时发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    健康检查
    
    Returns:
        服务状态
    """
    return {"status": "healthy", "service": "comic_downloader"}

@app.get("/")
async def root():
    """
    根路径
    
    Returns:
        服务信息
    """
    return {
        "service": "Comic Downloader MCP Server",
        "version": "1.0.0",
        "description": "A simplified comic downloader that downloads comics to desktop",
        "endpoints": {
            "download": "POST /download",
            "status": "GET /status/{album_id}",
            "cancel": "POST /cancel",
            "all_status": "GET /status",
            "health": "GET /health"
        }
    }

def start_server(host: str = "localhost", port: int = 8080):
    """
    启动服务器
    
    Args:
        host: 主机地址
        port: 端口号
    """
    logger.info(f"启动Comic Downloader MCP服务器: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server() 