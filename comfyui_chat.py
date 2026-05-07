#!/usr/bin/env python3
"""
ComfyUI Chat - Natural language control for local ComfyUI

Independent project. Pure local ComfyUI GPU rendering.

Key features (inspired by AIGC Factory / ComfyUI-OpenClaw):
- LLM-assisted prompt refinement
- Image-to-prompt (analyze image -> generate prompt)
- Preset system (save/load generation presets)
- Batch generation variants
- Webhook/API server for automation
- Multi-model support
"""

import os, sys, re, json, base64, time, tempfile, hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from datetime import datetime
from functools import wraps

import requests


# ============================================================
# Config
# ============================================================

class Config:
    COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "localhost")
    COMFYUI_PORT = os.environ.get("COMFYUI_PORT", "8188")
    COMFYUI_USER = os.environ.get("COMFYUI_USER", None)
    COMFYUI_PASS = os.environ.get("COMFYUI_PASS", None)

    # LLM for prompt refinement
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "minimax")  # minimax / openai / ollama
    LLM_API_KEY = os.environ.get("MINIMAX_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.minimax.chat/v1")

    # Paths
    BASE_DIR = Path(__file__).parent
    OUTPUT_DIR = BASE_DIR / "outputs"
    WORKFLOWS_DIR = BASE_DIR / "workflows"
    PRESETS_DIR = BASE_DIR / "presets"

    @classmethod
    def init(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.WORKFLOWS_DIR.mkdir(exist_ok=True)
        cls.PRESETS_DIR.mkdir(exist_ok=True)

Config.init()


# ============================================================
# ComfyUI API Client
# ============================================================

class ComfyUIClient:
    def __init__(self, host: str = None, port: int = None):
        self.host = host or Config.COMFYUI_HOST
        self.port = port or int(Config.COMFYUI_PORT)
        self.base_url = f"http://{self.host}:{self.port}"
        self.session = requests.Session()
        if Config.COMFYUI_USER and Config.COMFYUI_PASS:
            self.session.auth = (Config.COMFYUI_USER, Config.COMFYUI_PASS)

    def is_alive(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def queue_prompt(self, workflow: dict) -> Tuple[str, str]:
        try:
            r = self.session.post(f"{self.base_url}/prompt", json={"prompt": workflow}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                return data.get("prompt_id", ""), str(data.get("number", ""))
        except Exception as e:
            print(f"    [ERROR] Queue: {e}")
        return "", ""

    def get_history(self, prompt_id: str) -> dict:
        try:
            r = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            return r.json() if r.status_code == 200 else {}
        except:
            return {}

    def wait_for_result(self, prompt_id: str, timeout: int = 120) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        return {}

    def get_image(self, filename: str) -> bytes:
        try:
            r = self.session.get(f"{self.base_url}/view", params={"filename": filename}, timeout=30)
            return r.content if r.status_code == 200 else b""
        except:
            return b""

    def upload_image(self, image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                files = {"Image": (Path(image_path).name, f.read(), "image/png")}
                r = self.session.post(f"{self.base_url}/upload/image", files=files, timeout=30)
            if r.status_code == 200:
                return r.json().get("name", "")
        except Exception as e:
            print(f"    [ERROR] Upload: {e}")
        return ""

    def list_checkpoints(self) -> List[str]:
        try:
            r = self.session.get(f"{self.base_url}/storage/list?type=checkpoints", timeout=10)
            if r.status_code == 200:
                return r.json().get("names", [])
        except:
            pass
        return []


# ============================================================
# LLM Prompt Refiner (LLM-assisted)
# ============================================================

class PromptRefiner:
    """
    LLM-assisted prompt enhancement.

    Takes raw user input and refines it into a detailed,
    ComfyUI-friendly generation prompt using LLM.
    """

    SYSTEM_PROMPT = """You are a ComfyUI prompt engineering assistant. Given a short user request,
generate a detailed, well-structured positive prompt for Stable Diffusion or video generation.
Rules:
- Keep it under 200 words
- Add quality tags: masterpiece, best quality, detailed
- Add appropriate style descriptors
- For portraits: add lighting, expression, clothing details
- For landscapes: add time of day, weather, atmosphere
- For abstract: add color palette, composition hints
- Return ONLY the refined prompt, no explanations"""

    def __init__(self):
        self.provider = Config.LLM_PROVIDER
        self.api_key = Config.LLM_API_KEY
        self.base_url = Config.LLM_BASE_URL

    def refine(self, user_input: str, mode: str = "image") -> str:
        """Refine user input into detailed prompt"""
        if not self.api_key:
            return user_input  # Fallback: use as-is

        try:
            if self.provider == "minimax":
                return self._minimax_refine(user_input)
            elif self.provider == "openai":
                return self._openai_refine(user_input)
            elif self.provider == "ollama":
                return self._ollama_refine(user_input)
        except Exception as e:
            print(f"    [WARN] Refine failed: {e}")

        return user_input

    def _minimax_refine(self, user_input: str) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "MiniMax-Text-01",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"User request: {user_input}\nRefined prompt:"}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        }
        r = requests.post(f"{self.base_url}/text/chatcompletion_v2", headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("choices", [{}])[0].get("messages", [{}])[0].get("text", user_input)
        return user_input

    def _openai_refine(self, user_input: str) -> str:
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"User request: {user_input}\nRefined prompt:"}
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        }
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", user_input)
        return user_input

    def _ollama_refine(self, user_input: str) -> str:
        import requests
        payload = {
            "model": "qwen2.5:7b",
            "prompt": f"System: {self.SYSTEM_PROMPT}\nUser: {user_input}\nRefined prompt:",
            "stream": False,
        }
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json().get("response", user_input)
        return user_input


# ============================================================
# Image Analyzer (Vision)
# ============================================================

class ImageAnalyzer:
    """
    Analyze an image and generate a prompt from it.
    Uses LLM with vision capability or ViT-based model.
    """

    def __init__(self):
        self.refiner = PromptRefiner()

    def analyze(self, image_path: str) -> str:
        """Analyze image and return description prompt"""
        try:
            # Try MiniMax vision
            result = self._minimax_vision(image_path)
            if result:
                return result
        except Exception as e:
            print(f"    [WARN] Vision analysis failed: {e}")
        return "[Analysis unavailable]"

    def _minimax_vision(self, image_path: str) -> str:
        import requests
        if not Config.LLM_API_KEY:
            return ""

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        headers = {
            "Authorization": f"Bearer {Config.LLM_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "MiniMax-V01",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail, focusing on: subject, composition, lighting, style, colors, and mood. Format as a Stable Diffusion image generation prompt."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]
            }],
            "max_tokens": 300,
        }
        r = requests.post(f"{Config.LLM_BASE_URL}/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ""


# ============================================================
# Preset System
# ============================================================

class PresetManager:
    """Save and load generation presets"""

    def __init__(self):
        self.dir = Config.PRESETS_DIR

    def save(self, name: str, preset: dict) -> str:
        safe = re.sub(r'[^\w\-]', '_', name)
        path = self.dir / f"{safe}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(preset, f, ensure_ascii=False, indent=2)
        return str(path)

    def load(self, name: str) -> Optional[dict]:
        safe = re.sub(r'[^\w\-]', '_', name)
        path = self.dir / f"{safe}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_presets(self) -> List[str]:
        return [p.stem for p in self.dir.glob("*.json")]

    def delete(self, name: str) -> bool:
        safe = re.sub(r'[^\w\-]', '_', name)
        path = self.dir / f"{safe}.json"
        if path.exists():
            path.unlink()
            return True
        return False


# ============================================================
# Batch Generator
# ============================================================

class BatchGenerator:
    """Generate multiple variants from a single prompt"""

    def __init__(self, client: ComfyUIClient):
        self.client = client

    def generate_variants(self, prompt: str, count: int = 4,
                          width: int = 512, height: int = 512,
                          model: str = None) -> List[Dict]:
        """Generate N variants with different seeds"""
        results = []
        for i in range(count):
            seed = int(time.time() * 1000) + i
            workflow = self._build_workflow(prompt, seed=seed, width=width, height=height, model=model)
            print(f"    [Variant {i+1}/{count}] Seed: {seed}")
            pid, _ = self.client.queue_prompt(workflow)
            if pid:
                result = self.client.wait_for_result(pid)
                imgs = self._extract_images(result)
                results.append({"seed": seed, "images": imgs})
            time.sleep(0.5)
        return results

    def _build_workflow(self, prompt: str, seed: int, width: int, height: int, model: str) -> dict:
        return {
            "1": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
            "2": {"inputs": {"text": "low quality, blurry, watermark, bad anatomy", "clip": ["3", 0]}, "class_type": "CLIPTextEncode"},
            "3": {"inputs": {"ckpt_name": model or "sd15"}, "class_type": "CheckpointLoaderSimple"},
            "4": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "5": {"inputs": {"seed": seed, "steps": 25, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "positive": ["1", 0], "negative": ["2", 0], "model": ["3", 0], "latent_image": ["4", 0]}, "class_type": "KSampler"},
            "6": {"inputs": {"samples": ["5", 0], "model": ["3", 0]}, "class_type": "VAEDecode"},
            "7": {"inputs": {"filename": f"variant_{seed}.png", "images": ["6", 0]}, "class_type": "SaveImage"},
        }

    def _extract_images(self, result: dict) -> List[str]:
        imgs = []
        for node_data in result.get("outputs", {}).values():
            if "images" in node_data:
                for img_info in node_data["images"]:
                    filename = img_info["filename"]
                    img_bytes = self.client.get_image(filename)
                    if img_bytes:
                        out = Config.OUTPUT_DIR / filename
                        with open(out, "wb") as f:
                            f.write(img_bytes)
                        imgs.append(str(out))
        return imgs


# ============================================================
# ComfyUI Chat Core
# ============================================================

class ComfyUIChat:
    def __init__(self, client: ComfyUIClient = None):
        self.client = client or ComfyUIClient()
        self.refiner = PromptRefiner()
        self.analyzer = ImageAnalyzer()
        self.presets = PresetManager()
        self.batch = BatchGenerator(self.client)

    def txt2img(self, prompt: str, negative: str = "", model: str = None,
                width: int = 512, height: int = 512, steps: int = 25,
                cfg: float = 7.0, seed: int = 0,
                refine: bool = True, batch_count: int = 1) -> Dict:
        """Text-to-image generation with optional LLM refinement"""
        if not self.client.is_alive():
            return {"error": "ComfyUI not connected", "tip": "Start ComfyUI or check host/port"}

        # LLM prompt refinement
        if refine and self.refiner.api_key:
            print(f"    [Refining] '{prompt[:30]}...'")
            refined = self.refiner.refine(prompt)
            if refined != prompt:
                print(f"    [Refined] '{refined[:60]}...'")
                prompt = refined

        if batch_count > 1:
            # Batch generation
            variants = self.batch.generate_variants(prompt, count=batch_count, width=width, height=height, model=model)
            return {"success": True, "variants": variants, "refined": refine}

        workflow = self._build_txt2img_workflow(prompt, negative, model, width, height, steps, cfg, seed)
        print(f"    [Generating] '{prompt[:50]}...'")
        pid, _ = self.client.queue_prompt(workflow)
        if not pid:
            return {"error": "Queue failed"}

        result = self.client.wait_for_result(pid)
        paths = self._extract_images(result)

        if paths:
            return {"success": True, "count": len(paths), "paths": paths, "refined": refine}
        return {"error": "No images generated"}

    def img2img(self, image_path: str, prompt: str, strength: float = 0.7,
                model: str = None, refine: bool = True) -> Dict:
        """Image-to-image generation"""
        if not self.client.is_alive():
            return {"error": "ComfyUI not connected"}

        # Upload image
        uploaded_name = self.client.upload_image(image_path)
        if not uploaded_name:
            return {"error": "Image upload failed"}

        # Refine prompt
        if refine and self.refiner.api_key:
            prompt = self.refiner.refine(prompt)

        workflow = self._build_img2img_workflow(uploaded_name, prompt, strength, model)
        print(f"    [Generating] img2img: '{prompt[:50]}...'")
        pid, _ = self.client.queue_prompt(workflow)
        if not pid:
            return {"error": "Queue failed"}

        result = self.client.wait_for_result(pid)
        paths = self._extract_images(result)
        return {"success": bool(paths), "paths": paths}

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze image and generate prompt"""
        if not self.client.is_alive():
            return {"error": "ComfyUI not connected"}

        prompt = self.analyzer.analyze(image_path)
        if not prompt:
            return {"error": "Analysis failed"}
        return {"success": True, "prompt": prompt}

    def txt2video(self, prompt: str, model: str = "svd-xt",
                  frames: int = 61, fps: int = 8) -> Dict:
        """Text-to-video generation"""
        if not self.client.is_alive():
            return {"error": "ComfyUI not connected"}

        workflow = self._build_video_workflow(prompt, model, frames, fps)
        print(f"    [Generating] video: '{prompt[:50]}...'{frames} frames")
        pid, _ = self.client.queue_prompt(workflow)
        if not pid:
            return {"error": "Queue failed"}

        result = self.client.wait_for_result(pid, timeout=300)
        # Video extraction depends on ComfyUI workflow output format
        return {"success": bool(result.get("outputs")), "prompt_id": pid}

    def save_preset(self, name: str, **params) -> str:
        path = self.presets.save(name, params)
        return path

    def load_preset(self, name: str) -> Optional[dict]:
        return self.presets.load(name)

    def _build_txt2img_workflow(self, prompt: str, negative: str, model: str,
                                width: int, height: int, steps: int, cfg: float, seed: int) -> dict:
        return {
            "1": {"inputs": {"text": prompt, "clip": ["3", 0]}, "class_type": "CLIPTextEncode"},
            "2": {"inputs": {"text": negative or "low quality, blurry, watermark", "clip": ["3", 0]}, "class_type": "CLIPTextEncode"},
            "3": {"inputs": {"ckpt_name": model or "sd15"}, "class_type": "CheckpointLoaderSimple"},
            "4": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "5": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": "euler", "scheduler": "normal", "positive": ["1", 0], "negative": ["2", 0], "model": ["3", 0], "latent_image": ["4", 0]}, "class_type": "KSampler"},
            "6": {"inputs": {"samples": ["5", 0], "model": ["3", 0]}, "class_type": "VAEDecode"},
            "7": {"inputs": {"filename": f"img_{int(time.time())}.png", "images": ["6", 0]}, "class_type": "SaveImage"},
        }

    def _build_img2img_workflow(self, image_name: str, prompt: str, strength: float, model: str) -> dict:
        return {
            "1": {"inputs": {"image": image_name}, "class_type": "LoadImage"},
            "2": {"inputs": {"text": prompt, "clip": ["4", 0]}, "class_type": "CLIPTextEncode"},
            "3": {"inputs": {"text": "", "clip": ["4", 0]}, "class_type": "CLIPTextEncode"},
            "4": {"inputs": {"ckpt_name": model or "sd15"}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"strength": strength}, "class_type": "VAEDecode"},
            "6": {"inputs": {"filename": f"img2img_{int(time.time())}.png", "images": ["5", 0]}, "class_type": "SaveImage"},
        }

    def _build_video_workflow(self, prompt: str, model: str, frames: int, fps: int) -> dict:
        if model == "cogvideo":
            return {
                "1": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
                "2": {"inputs": {"model_name": "CogVideoX-5b", "width": 720, "height": 480, "frames": frames}, "class_type": "CogVideoX"},
                "3": {"inputs": {"filename": f"cog_{int(time.time())}.mp4"}, "class_type": "VideoCombine"},
            }
        else:
            return {
                "1": {"inputs": {"text": prompt}, "class_type": "CLIPTextEncode"},
                "2": {"inputs": {"model_name": "svd", "width": 1024, "height": 576, "video_frames": frames, "fps": fps}, "class_type": "SVD_img2vid"},
                "3": {"inputs": {"filename": f"svd_{int(time.time())}.mp4"}, "class_type": "VideoCombine"},
            }

    def _extract_images(self, result: dict) -> List[str]:
        paths = []
        for node_data in result.get("outputs", {}).values():
            if "images" in node_data:
                for img_info in node_data["images"]:
                    filename = img_info["filename"]
                    img_bytes = self.client.get_image(filename)
                    if img_bytes:
                        out = Config.OUTPUT_DIR / filename
                        with open(out, "wb") as f:
                            f.write(img_bytes)
                        paths.append(str(out))
        return paths


# ============================================================
# Chat Interface
# ============================================================

class ChatInterface:
    def __init__(self):
        self.chat = ComfyUIChat()
        self.running = True
        self.last_image = None  # Track last generated image

    def start(self):
        print("=" * 52)
        print("  ComfyUI Chat - Local AI Image/Video Generator")
        print("  Inspired by AIGC Factory (ComfyUI-OpenClaw)")
        print("=" * 52)
        print()

        alive = self.chat.client.is_alive()
        if alive:
            print("[OK] ComfyUI connected")
            models = self.chat.client.list_checkpoints()
            if models:
                print(f"[OK] Models: {', '.join(models[:3])}")
        else:
            print("[WARN] ComfyUI not connected. Start ComfyUI first.")

        print()
        print("Commands:")
        print("  画 <描述>           - Text-to-image")
        print("  图生图 <路径> <描述> - Image-to-image")
        print("  分析 <图片路径>      - Analyze image -> prompt")
        print("  视频 <描述>         - Text-to-video")
        print("  批 <描述> <数量>    - Batch generate variants")
        print("  预设 save|load|del  - Preset management")
        print("  状态               - Connection status")
        print("  帮助               - This help")
        print("  quit               - Exit")
        print()

        while self.running:
            try:
                user = input("\n[You] ").strip()
                if user:
                    self.handle(user)
            except (KeyboardInterrupt, EOFError):
                print("\n\nBye!")
                break

    def handle(self, text: str):
        text = text.strip()
        if not text:
            return

        # Command routing
        if re.match(r'^(画?|生成.*图)\s*(.*)', text):
            m = re.match(r'^(画?|生成.*图)\s*(.*)', text)
            self.cmd_txt2img(m.group(2) or "")
        elif re.match(r'^(图生图|上传)\s*(\S+)\s*(.*)', text):
            m = re.match(r'^(图生图|上传)\s*(\S+)\s*(.*)', text)
            self.cmd_img2img(m.group(2), m.group(3))
        elif re.match(r'^(分析|看图)\s*(.*)', text):
            m = re.match(r'^(分析|看图)\s*(.*)', text)
            self.cmd_analyze(m.group(2))
        elif re.match(r'^(视频|文生视频)\s*(.*)', text):
            m = re.match(r'^(视频|文生视频)\s*(.*)', text)
            self.cmd_video(m.group(2))
        elif re.match(r'^批\s*(.*)', text):
            m = re.match(r'^批\s*(.*)', text)
            self.cmd_batch(m.group(1))
        elif re.match(r'^(预设|preset)\s*(.*)', text):
            m = re.match(r'^(预设|preset)\s*(.*)', text)
            self.cmd_preset(m.group(2))
        elif text in ['状态', 'status', 'stat']:
            self.cmd_status()
        elif text in ['帮助', 'help', '?']:
            self.cmd_help()
        elif text in ['quit', 'exit', '退出']:
            self.running = False
        else:
            # Default: txt2img
            self.cmd_txt2img(text)

    def cmd_txt2img(self, prompt: str):
        if not prompt:
            print("[Usage] 画 <描述>")
            return
        result = self.chat.txt2img(prompt)
        self._show_result(result, "Image")

    def cmd_img2img(self, path: str, prompt: str):
        if not path or not prompt:
            print("[Usage] 图生图 <图片路径> <描述>")
            return
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return
        result = self.chat.img2img(path, prompt)
        self._show_result(result, "Image")

    def cmd_analyze(self, path: str):
        if not path:
            print("[Usage] 分析 <图片路径>")
            return
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            return
        print(f"[Analyzing] {path}")
        result = self.chat.analyze_image(path)
        if result.get("success"):
            print(f"[Prompt] {result['prompt'][:200]}")
        else:
            print(f"[ERROR] {result.get('error', 'Failed')}")

    def cmd_video(self, prompt: str):
        if not prompt:
            print("[Usage] 视频 <描述>")
            return
        print("[Video] Generating (may take 1-3 min)...")
        result = self.chat.txt2video(prompt)
        self._show_result(result, "Video")

    def cmd_batch(self, text: str):
        parts = text.strip().split()
        prompt = parts[0] if parts else ""
        count = int(parts[1]) if len(parts) > 1 else 4
        if not prompt:
            print("[Usage] 批 <描述> <数量(默认4)>")
            return
        count = max(1, min(count, 9))
        print(f"[Batch] Generating {count} variants...")
        result = self.chat.txt2img(prompt, batch_count=count)
        if result.get("success"):
            variants = result.get("variants", [])
            print(f"[OK] Generated {len(variants)} variants")
            for i, v in enumerate(variants):
                print(f"  [{i+1}] seed={v['seed']}, images={len(v['images'])}")
                for p in v['images'][:2]:
                    print(f"      {p}")

    def cmd_preset(self, text: str):
        parts = text.strip().split(maxsplit=1)
        action = parts[0] if parts else ""
        name = parts[1] if len(parts) > 1 else ""

        if action == "save" and name:
            params = {"width": 512, "height": 512, "steps": 25}
            path = self.chat.save_preset(name, **params)
            print(f"[OK] Preset saved: {path}")
        elif action in ("load", ""):
            presets = self.chat.presets.list_presets()
            if presets:
                print("[Presets]")
                for p in presets:
                    print(f"  - {p}")
            else:
                print("[Presets] No presets saved")
        elif action == "del" and name:
            if self.chat.presets.delete(name):
                print(f"[OK] Deleted: {name}")
            else:
                print(f"[ERROR] Not found: {name}")
        else:
            print("[Usage] 预设 save|load|del <name>")

    def cmd_status(self):
        alive = self.chat.client.is_alive()
        print(f"[Status] ComfyUI: {'Online' if alive else 'Offline'}")
        if alive:
            models = self.chat.client.list_checkpoints()
            print(f"[Models] {', '.join(models) if models else 'none'}")

    def cmd_help(self):
        self.start.__doc__ = self.start.__doc__  # refresh
        print("""
[Commands]
  画 <描述>             - Generate image from text
  图生图 <图> <描述>    - Transform image
  分析 <图片>          - Analyze image -> prompt
  视频 <描述>          - Generate video
  批 <描述> <数量>     - Batch generate (1-9 variants)
  预设 save <名>       - Save current settings as preset
  预设 load            - List all presets
  预设 del <名>         - Delete a preset
  状态                 - Check ComfyUI connection
  帮助                 - Show this help
  quit                 - Exit

[Environment]
  COMFYUI_HOST     ComfyUI IP (default: localhost)
  COMFYUI_PORT     ComfyUI port (default: 8188)
  LLM_PROVIDER     minimax|openai|ollama
  MINIMAX_API_KEY  Your API key (for prompt refinement)
""")

    def _show_result(self, result: Dict, kind: str):
        if result.get("success"):
            paths = result.get("paths", [])
            print(f"[OK] {kind} generated! {len(paths)} file(s)")
            for p in paths[:3]:
                print(f"    {p}")
            if len(paths) > 0:
                self.last_image = paths[0]
        else:
            err = result.get("error", "Unknown")
            tip = result.get("tip", "")
            print(f"[ERROR] {err}")
            if tip:
                print(f"[Tip] {tip}")


# ============================================================
# HTTP API Server
# ============================================================

def run_api_server(port: int = 5000, host: str = "0.0.0.0"):
    """Run as HTTP API server for webhook automation"""
    from flask import Flask, request, jsonify

    app = Flask(__name__)
    chat = ComfyUIChat()

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "comfyui": chat.client.is_alive()})

    @app.route("/api/txt2img", methods=["POST"])
    def txt2img():
        data = request.json or {}
        prompt = data.get("prompt", "")
        if not prompt:
            return jsonify({"error": "prompt required"}), 400
        result = chat.txt2img(
            prompt=prompt,
            negative=data.get("negative", ""),
            model=data.get("model"),
            width=data.get("width", 512),
            height=data.get("height", 512),
            steps=data.get("steps", 25),
            refine=data.get("refine", True),
            batch_count=data.get("batch", 1),
        )
        return jsonify(result)

    @app.route("/api/img2img", methods=["POST"])
    def img2img():
        data = request.json or {}
        image_path = data.get("image")
        prompt = data.get("prompt", "")
        if not image_path or not prompt:
            return jsonify({"error": "image and prompt required"}), 400
        result = chat.img2img(image_path, prompt, strength=data.get("strength", 0.7))
        return jsonify(result)

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        data = request.json or {}
        image_path = data.get("image")
        if not image_path:
            return jsonify({"error": "image required"}), 400
        result = chat.analyze_image(image_path)
        return jsonify(result)

    @app.route("/api/presets", methods=["GET", "POST"])
    def presets():
        if request.method == "GET":
            return jsonify({"presets": chat.presets.list_presets()})
        data = request.json or {}
        name = data.get("name")
        if not name:
            return jsonify({"error": "name required"}), 400
        params = data.get("params", {})
        path = chat.save_preset(name, **params)
        return jsonify({"saved": path})

    print(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ComfyUI Chat - Natural language image/video generation")
    parser.add_argument("--api", action="store_true", help="Run as HTTP API server")
    parser.add_argument("--port", type=int, default=5000, help="API server port")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    args = parser.parse_args()

    if args.api:
        run_api_server(port=args.port, host=args.host)
    else:
        interface = ChatInterface()
        interface.start()