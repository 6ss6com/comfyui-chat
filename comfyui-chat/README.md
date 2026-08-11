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

---

## 中文说明

**comfyui-chat** 是用自然语言控制本地 ComfyUI 的工具 — 你说一句话，它帮你搭工作流、跑图、调参。

### 特点
- 自然语言驱动（不用记节点）
- 本地优先（数据不出本机）
- 内置工作流模板库

### 快速开始
```bash
git clone https://github.com/6ss6com/comfyui-chat.git
cd comfyui-chat
pip install -r requirements.txt
python main.py
```

### 路线图
- [x] v0.1 基础自然语言 → 工作流
- [ ] v0.2 工作流模板市场
- [ ] v0.3 多模型支持 (SDXL / Flux)
- [ ] v1.0 Web UI


---

## 内置工作流示例 (workflows/)

- workflows/ltx-2.3-turbo-stable.json - LTX 2.3 Turbo Stable 4V4A
- workflows/minimax-h3-turbo.json - MiniMax H3 Turbo 4-step

完整 JSON 从 GitHub 下载 (GFW 风险): https://github.com/T8mars/comfyui-minimax-h3-audio-T8
