#!/usr/bin/env python3
"""
ComfyUI Chat - 自然语言控制ComfyUI

独立项目，纯本地ComfyUI驱动。
不依赖DistillAI。

用法:
    python comfyui_chat.py
    # 或指定ComfyUI地址
    COMFYUI_HOST=192.168.1.100 COMFYUI_PORT=8188 python comfyui_chat.py
"""

import os, sys, re, json, base64, time, tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# ============================================================
# ComfyUI API Client
# ============================================================

class ComfyUIClient:
    """ComfyUI REST API客户端"""

    def __init__(self, host: str = None, port: int = None):
        self.host = host or os.environ.get("COMFYUI_HOST", "localhost")
        self.port = port or int(os.environ.get("COMFYUI_PORT", "8188"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.username = os.environ.get("COMFYUI_USER")
        self.password = os.environ.get("COMFYUI_PASS")

        import requests
        self.session = requests.Session()
        if self.username and self.password:
            self.session.auth = (self.username, self.password)

    def is_alive(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def get_history(self, prompt_id: str) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            return r.json() if r.status_code == 200 else {}
        except:
            return {}

    def get_image(self, filename: str) -> bytes:
        try:
            r = self.session.get(
                f"{self.base_url}/view",
                params={"filename": filename},
                timeout=30
            )
            return r.content if r.status_code == 200 else b""
        except:
            return b""

    def queue_prompt(self, workflow: dict) -> Tuple[str, str]:
        try:
            payload = {"prompt": workflow}
            r = self.session.post(f"{self.base_url}/prompt", json=payload, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data.get("prompt_id", ""), str(data.get("number", ""))
        except Exception as e:
            print(f"    [ERROR] Queue failed: {e}")
        return "", ""

    def wait_for_result(self, prompt_id: str, timeout: int = 120) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        return {}

    def list_checkpoints(self) -> List[str]:
        try:
            r = self.session.get(f"{self.base_url}/storage/list?type=checkpoints", timeout=10)
            if r.status_code == 200:
                return r.json().get("names", [])
        except:
            pass
        return []


# ============================================================
# Generation Functions
# ============================================================

class ComfyUIChat:
    """ComfyUI对话生成器"""

    def __init__(self, client: ComfyUIClient = None):
        self.client = client or ComfyUIClient()
        self.output_dir = Path(__file__).parent / "outputs"
        self.output_dir.mkdir(exist_ok=True)
        self.workflows_dir = Path(__file__).parent / "workflows"
        self.workflows_dir.mkdir(exist_ok=True)
        self.default_size = (512, 512)
        self.default_model = "sd15"

    # ===== 图片生成 =====
    def txt2img(self, prompt: str, negative: str = "", model: str = None,
                width: int = 512, height: int = 512, steps: int = 25,
                cfg: float = 7.0, seed: int = 0) -> Dict:
        """文生图"""
        if not self.client.is_alive():
            return {"error": "ComfyUI未连接", "tip": "请确认ComfyUI正在运行"}

        model = model or self.default_model

        workflow = {
            "1": {"inputs": {"text": prompt, "clip": ["3", 0]}, "class_type": "CLIPTextEncode"},
            "2": {"inputs": {"text": negative or "low quality, blurry, watermark", "clip": ["3", 0]}, "class_type": "CLIPTextEncode"},
            "3": {"inputs": {"ckpt_name": model}, "class_type": "CheckpointLoaderSimple"},
            "4": {"inputs": {
                "width": width, "height": height, "batch_size": 1
            }, "class_type": "EmptyLatentImage"},
            "5": {"inputs": {
                "seed": seed or 0, "steps": steps, "cfg": cfg,
                "sampler_name": "euler", "scheduler": "normal",
                "positive": ["1", 0], "negative": ["2", 0],
                "model": ["3", 0], "latent_image": ["4", 0]
            }, "class_type": "KSampler"},
            "6": {"inputs": {"samples": ["5", 0], "model": ["3", 0]}, "class_type": "VAEDecode"},
            "7": {"inputs": {
                "filename": f"txt2img_{int(time.time())}.png",
                "images": ["6", 0]
            }, "class_type": "SaveImage"},
        }

        print(f"    [生成中] 文生图: {prompt[:50]}...")
        prompt_id, _ = self.client.queue_prompt(workflow)
        if not prompt_id:
            return {"error": "队列失败"}

        result = self.client.wait_for_result(prompt_id)
        return self._parse_image_result(result)

    def img2img(self, image_path: str, prompt: str, strength: float = 0.7,
                 model: str = None, seed: int = 0) -> Dict:
        """图生图"""
        if not self.client.is_alive():
            return {"error": "ComfyUI未连接"}

        model = model or self.default_model

        # 上传图片
        with open(image_path, "rb") as f:
            files = {"Image": ("input.png", f.read(), "image/png")}
            r = self.client.session.post(
                f"{self.client.base_url}/upload/image",
                files=files, timeout=30
            )
        if r.status_code != 200:
            return {"error": "图片上传失败"}

        uploaded_name = r.json().get("name", "")

        workflow = {
            "1": {"inputs": {"image": uploaded_name}, "class_type": "LoadImage"},
            "2": {"inputs": {"text": prompt, "clip": ["4", 0]}, "class_type": "CLIPTextEncode"},
            "3": {"inputs": {"text": "", "clip": ["4", 0]}, "class_type": "CLIPTextEncode"},
            "4": {"inputs": {"ckpt_name": model}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"强度": strength}, "class_type": "VAEDecode"},
        }

        print(f"    [生成中] 图生图: {prompt[:50]}...")
        prompt_id, _ = self.client.queue_prompt(workflow)
        if not prompt_id:
            return {"error": "队列失败"}

        result = self.client.wait_for_result(prompt_id)
        return self._parse_image_result(result)

    def _parse_image_result(self, result: dict) -> Dict:
        outputs = result.get("outputs", {})
        saved_paths = []

        for node_id, node_data in outputs.items():
            if "images" in node_data:
                for img_info in node_data["images"]:
                    filename = img_info["filename"]
                    img_bytes = self.client.get_image(filename)
                    if img_bytes:
                        out_path = self.output_dir / filename
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        saved_paths.append(str(out_path))

        if saved_paths:
            return {
                "success": True,
                "count": len(saved_paths),
                "paths": saved_paths,
            }
        return {"error": "未生成图片"}

    # ===== 视频生成 =====
    def txt2video(self, prompt: str, model: str = "svd-xt",
                  frames: int = 61, fps: int = 8) -> Dict:
        """文生视频 (SVD-XT / CogVideoX)"""
        if not self.client.is_alive():
            return {"error": "ComfyUI未连接"}

        if model == "cogvideo":
            return self._cogvideo(prompt, frames)
        else:
            return self._svd(prompt, frames, fps)

    def _svd(self, prompt: str, frames: int, fps: int) -> Dict:
        """SVD-XT视频生成"""
        workflow = {
            "1": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
            "2": {"inputs": {
                "model_name": "svd",
                "width": 1024, "height": 576,
                "video_frames": frames, "fps": fps
            }, "class_type": "SVD_img2vid"},
            "3": {"inputs": {"filename": f"svd_{int(time.time())}.mp4"}, "class_type": "VideoCombine"},
        }

        print(f"    [生成中] 视频: {prompt[:50]}... ({frames}帧)")
        prompt_id, _ = self.client.queue_prompt(workflow)
        if not prompt_id:
            return {"error": "队列失败"}

        result = self.client.wait_for_result(prompt_id, timeout=300)
        return self._parse_video_result(result)

    def _cogvideo(self, prompt: str, frames: int) -> Dict:
        """CogVideoX视频生成"""
        workflow = {
            "1": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
            "2": {"inputs": {
                "model_name": "CogVideoX-5b",
                "width": 720, "height": 480,
                "frames": frames
            }, "class_type": "CogVideoX"},
            "3": {"inputs": {"filename": f"cog_{int(time.time())}.mp4"}, "class_type": "VideoCombine"},
        }

        print(f"    [生成中] CogVideoX: {prompt[:50]}...")
        prompt_id, _ = self.client.queue_prompt(workflow)
        if not prompt_id:
            return {"error": "队列失败"}

        result = self.client.wait_for_result(prompt_id, timeout=600)
        return self._parse_video_result(result)

    def _parse_video_result(self, result: dict) -> Dict:
        outputs = result.get("outputs", {})
        for node_id, node_data in outputs.items():
            if "gifs" in node_data or "videos" in node_data:
                key = "gifs" if "gifs" in node_data else "videos"
                for vid_info in node_data[key]:
                    filename = vid_info["filename"]
                    vid_path = self.output_dir / filename
                    # 视频文件需要从输出目录获取
                    return {
                        "success": True,
                        "filename": filename,
                        "path": str(vid_path),
                    }
        return {"error": "视频生成失败"}

    # ===== 工作流 =====
    def run_workflow(self, workflow_path: str = None,
                     workflow_dict: dict = None, **params) -> Dict:
        """执行自定义工作流"""
        if not self.client.is_alive():
            return {"error": "ComfyUI未连接"}

        if workflow_path:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        elif workflow_dict:
            workflow = workflow_dict
        else:
            return {"error": "未提供工作流"}

        # 注入参数
        for key, value in params.items():
            for node_id in workflow:
                if key in workflow[node_id].get("inputs", {}):
                    workflow[node_id]["inputs"][key] = value

        print(f"    [执行] 工作流...")
        prompt_id, _ = self.client.queue_prompt(workflow)
        if not prompt_id:
            return {"error": "队列失败"}

        result = self.client.wait_for_result(prompt_id)
        return {"success": bool(result.get("outputs")), "prompt_id": prompt_id}

    def import_workflow(self, workflow_json: dict, name: str) -> str:
        """导入工作流"""
        safe_name = re.sub(r'[^\w\-]', '_', name)
        path = self.workflows_dir / f"{safe_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(workflow_json, f, ensure_ascii=False, indent=2)
        return str(path)

    def list_workflows(self) -> List[str]:
        return [f.name for f in self.workflows_dir.glob("*.json")]


# ============================================================
# Chat Interface
# ============================================================

class ChatInterface:
    """对话界面"""

    COMMANDS = {
        # 图片
        r'^画?(.*)$': 'txt2img',
        r'^生成.*图.*?(.*)$': 'txt2img',
        r'^文生图\s*(.*)$': 'txt2img',
        r'^图生图\s*(.*)$': 'img2img',
        r'^上传.*图\s*(.*)$': 'img2img',
        # 视频
        r'^生成.*视频\s*(.*)$': 'txt2video',
        r'^视频\s*(.*)$': 'txt2video',
        r'^文生视频\s*(.*)$': 'txt2video',
        # 工作流
        r'^工作流\s*$': 'list_workflows',
        r'^导入工作流\s*(.*)$': 'import_workflow',
        r'^执行\s*(.*)$': 'run_workflow',
        # 系统
        r'^状态\s*$': 'status',
        r'^帮助\s*$': 'help',
        r'^quit|exit|退出\s*$': 'quit',
    }

    def __init__(self):
        self.comfy = ComfyUIChat()
        self.running = True

    def start(self):
        print("=" * 50)
        print("  ComfyUI Chat - 自然语言控制ComfyUI")
        print("=" * 50)
        print()

        # 检查连接
        alive = self.comfy.client.is_alive()
        if alive:
            print("[OK] ComfyUI已连接")
            checkpoints = self.comfy.client.list_checkpoints()
            if checkpoints:
                print(f"[OK] 可用模型: {', '.join(checkpoints[:3])}")
            self.comfy.default_model = checkpoints[0] if checkpoints else "sd15"
        else:
            print("[WARN] ComfyUI未连接，请先启动ComfyUI")

        print()
        print("输入 help 查看命令，输入 quit 退出")
        print()

        while self.running:
            try:
                user_input = input("\n[你] ").strip()
                if not user_input:
                    continue
                self.handle(user_input)
            except (KeyboardInterrupt, EOFError):
                print("\n\n再见!")
                break

    def handle(self, text: str):
        text = text.strip()

        # 检查命令
        for pattern, action in self.COMMANDS.items():
            m = re.match(pattern, text)
            if m:
                getattr(self, f"cmd_{action}")(m.group(1) if m.lastindex else text)
                return

        # 通用生成（默认文生图）
        self.cmd_txt2img(text)

    def cmd_txt2img(self, prompt: str):
        if not prompt:
            print("[用法] 画 <描述> 或直接输入描述")
            return
        result = self.comfy.txt2img(prompt)
        self._show_result(result)

    def cmd_img2img(self, prompt: str):
        print("[用法] 图生图需要先上传图片路径")
        print("[提示] 输入: 上传图片 <本地路径> <风格描述>")

    def cmd_txt2video(self, prompt: str):
        if not prompt:
            print("[用法] 视频 <描述>")
            return
        print("[模式] 视频生成中，请耐心等待(可能需要1-3分钟)...")
        result = self.comfy.txt2video(prompt)
        self._show_result(result)

    def cmd_list_workflows(self, _):
        workflows = self.comfy.list_workflows()
        if workflows:
            print("[工作流]")
            for w in workflows:
                print(f"  - {w}")
        else:
            print("[工作流] 暂无已保存工作流")

    def cmd_import_workflow(self, path: str):
        if not path:
            print("[用法] 导入工作流 <json文件路径>")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                wf = json.load(f)
            name = Path(path).stem
            saved = self.comfy.import_workflow(wf, name)
            print(f"[OK] 工作流已导入: {saved}")
        except Exception as e:
            print(f"[ERROR] 导入失败: {e}")

    def cmd_run_workflow(self, name: str):
        if not name:
            print("[用法] 执行 <工作流名>")
            return
        path = self.comfy.workflows_dir / f"{name}.json"
        if not path.exists():
            print(f"[ERROR] 工作流不存在: {path}")
            return
        result = self.comfy.run_workflow(str(path))
        self._show_result(result)

    def cmd_status(self, _):
        alive = self.comfy.client.is_alive()
        print(f"[状态] ComfyUI: {'在线' if alive else '离线'}")
        if alive:
            models = self.comfy.client.list_checkpoints()
            print(f"[模型] {', '.join(models) if models else '无'}")

    def cmd_help(self, _):
        print("""
[命令]

  图片生成:
    画 <描述>           - 文生图
    视频 <描述>         - 文生视频

  工作流:
    工作流              - 列出已保存的工作流
    导入工作流 <路径>   - 导入JSON工作流
    执行 <名称>         - 执行已保存的工作流

  系统:
    状态               - 查看ComfyUI连接状态
    帮助               - 显示此帮助
    quit               - 退出
""")

    def cmd_quit(self, _):
        self.running = False

    def _show_result(self, result: Dict):
        if result.get("success"):
            count = result.get("count", 1)
            paths = result.get("paths", [])
            print(f"[OK] 生成完成! 共{count}个文件")
            for p in paths[:3]:
                print(f"    {p}")
        else:
            error = result.get("error", "未知错误")
            tip = result.get("tip", "")
            print(f"[ERROR] {error}")
            if tip:
                print(f"[提示] {tip}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("ComfyUI Chat - Local ComfyUI Control")
    interface = ChatInterface()
    interface.start()