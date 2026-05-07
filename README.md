# ComfyUI Chat

**通过自然语言对话控制本地ComfyUI，生成图片、视频。**

> 独立项目，不依赖DistillAI，纯本地GPU驱动。

## 快速开始

```bash
# 1. 启动ComfyUI (默认 localhost:8188)
# 2. 运行
python comfyui_chat.py
```

## 功能

- **图片生成**: 文生图 / 图生图
- **视频生成**: SVD-XT / CogVideoX
- **工作流**: 自定义JSON工作流导入执行

## 命令示例

```
画一只在月球上的猫
视频 日落海边风景
工作流        # 查看已保存工作流
状态          # 查看连接状态
帮助          # 查看所有命令
```

## 环境变量

```bash
COMFYUI_HOST=192.168.1.100  # ComfyUI地址
COMFYUI_PORT=8188            # 端口
COMFYUI_USER=admin           # 用户名(可选)
COMFYUI_PASS=xxx             # 密码(可选)
```