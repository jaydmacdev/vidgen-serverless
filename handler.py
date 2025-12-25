"""
VidGen Production Handler
Optimized for RunPod Serverless with better error handling
"""

import runpod
import base64
import io
import os
import time
import gc
import sys
from PIL import Image
import torch

print("=" * 70)
print("🚀 VidGen Handler Initializing...")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 70)

# Import after checking CUDA
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# Global pipeline
pipeline = None
model_loaded = False

def load_model():
    """Load model with comprehensive error handling"""
    global pipeline, model_loaded
    
    if model_loaded:
        return pipeline
    
    try:
        print("\n" + "=" * 70)
        print("📦 Loading Wan2.2-I2V Model...")
        print("=" * 70)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load with explicit settings
        print("Step 1/4: Downloading model from Hugging Face...")
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("Step 2/4: Moving to GPU...")
        pipeline = pipeline.to("cuda")
        
        print("Step 3/4: Enabling optimizations...")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        
        # Try xformers
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("   ✅ xFormers enabled")
        except Exception as e:
            print(f"   ⚠️ xFormers failed: {e}")
            try:
                pipeline.enable_attention_slicing(1)
                print("   ✅ Attention slicing enabled")
            except:
                pass
        
        print("Step 4/4: Loading LoRA weights...")
        try:
            pipeline.load_lora_weights(
                "lightx2v/Wan2.2-Distill-Loras",
                weight_name="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
            )
            print("   ✅ 4-step LoRA loaded")
        except Exception as e:
            print(f"   ⚠️ LoRA loading failed: {e}")
            print("   ℹ️ Continuing without LoRA (will use 25 steps)")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        model_loaded = True
        print("\n✅ Model loaded successfully!")
        print("=" * 70)
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ CRITICAL: Model loading failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

def base64_to_image(base64_string):
    """Convert base64 to PIL Image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Image decode failed: {e}")

def video_to_base64(video_path):
    """Convert video to base64"""
    try:
        with open(video_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Video encode failed: {e}")

def handler(event):
    """Main handler with comprehensive error handling"""
    start_time = time.time()
    output_path = None
    
    try:
        print("\n" + "=" * 70)
        print("🎬 New Generation Request")
        print("=" * 70)
        
        # Parse input
        input_data = event.get("input", {})
        if not input_data:
            return {"error": "No input provided"}
        
        image_base64 = input_data.get("image_base64")
        if not image_base64:
            return {"error": "Missing image_base64"}
        
        # Get parameters
        prompt = input_data.get("prompt", "").strip()
        quality = input_data.get("quality", "standard").lower()
        fps = input_data.get("fps", 16)
        
        # Map quality to frames
        quality_map = {
            "draft": 49,
            "standard": 65,
            "high": 81
        }
        num_frames = input_data.get("num_frames", quality_map.get(quality, 65))
        
        print(f"\n⚙️ Configuration:")
        print(f"   Quality: {quality}")
        print(f"   Frames: {num_frames} (~{num_frames/fps:.1f}s)")
        print(f"   FPS: {fps}")
        if prompt:
            print(f"   Prompt: '{prompt}'")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\n💾 GPU Memory: {mem_allocated:.2f} GB allocated")
        
        # Load model
        pipe = load_model()
        
        # Process image
        print(f"\n🖼️ Processing image...")
        image = base64_to_image(image_base64)
        print(f"   Original: {image.size}")
        
        image = image.resize((1280, 720), Image.LANCZOS)
        print(f"   Resized: {image.size}")
        
        # Clear memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate
        print(f"\n🎨 Generating video...")
        gen_start = time.time()
        
        # Use 4-step if LoRA loaded, else 25 steps
        num_inference_steps = 4 if model_loaded else 25
        print(f"   Using {num_inference_steps} inference steps")
        
        with torch.no_grad():
            if prompt:
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=720,
                    width=1280
                )
            else:
                output = pipe(
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    height=720,
                    width=1280
                )
        
        frames = output.frames[0]
        gen_time = time.time() - gen_start
        
        print(f"   ✅ Generated in {gen_time:.2f}s")
        print(f"   Speed: {num_frames/gen_time:.1f} fps")
        
        # Clear memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save video
        print(f"\n💾 Exporting video...")
        output_path = f"/tmp/video_{int(time.time())}.mp4"
        export_to_video(frames, output_path, fps=fps)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"   File: {file_size_mb:.2f} MB")
        
        # Encode
        print(f"\n📤 Encoding to base64...")
        video_base64 = video_to_base64(output_path)
        
        # Cleanup
        try:
            os.remove(output_path)
        except:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        
        print(f"\n✅ SUCCESS!")
        print(f"   Total: {total_time:.2f}s")
        print(f"   Generation: {gen_time:.2f}s")
        print("=" * 70)
        
        result = {
            "video_base64": video_base64,
            "width": 1280,
            "height": 720,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": round(total_time, 2),
            "generation_time": round(gen_time, 2),
            "quality": quality,
            "file_size_mb": round(file_size_mb, 2)
        }
        
        if prompt:
            result["prompt_used"] = prompt
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"\n❌ ERROR: {error_msg}")
        print(f"\n📋 Traceback:\n{error_trace}")
        print("=" * 70)
        
        # Cleanup
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            "error": error_msg,
            "traceback": error_trace
        }

# Start RunPod serverless
if __name__ == "__main__":
    print("\n✅ Handler ready. Waiting for requests...")
    print("=" * 70 + "\n")
    runpod.serverless.start({"handler": handler})
