"""
VidGen Production Handler - Wan2.2 I2V with 4-Step Distilled LoRA
Based on: https://huggingface.co/lightx2v/Wan2.2-Distill-Loras
Model: Wan2.2-I2V-A14B with Low Noise LoRA (4-step inference)
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

print("=" * 80)
print("🚀 VidGen Handler - Wan2.2 I2V (4-Step Distilled)")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 80)

# Import diffusers
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# Global variables
pipeline = None
model_loaded = False

def load_model():
    """
    Load Wan2.2-I2V-A14B base model + 4-step distilled LoRA
    """
    global pipeline, model_loaded
    
    if model_loaded:
        return pipeline
    
    try:
        print("\n" + "=" * 80)
        print("📦 Loading Wan2.2-I2V-A14B with 4-Step Distilled LoRA")
        print("=" * 80)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Step 1: Load base model
        print("\n[1/3] Loading Wan2.2-I2V-A14B base model...")
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B",
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True
        )
        print("   ✅ Base model loaded")
        
        # Step 2: Move to GPU
        print("\n[2/3] Moving to GPU...")
        pipeline = pipeline.to("cuda")
        
        # Enable optimizations
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        
        # Try xformers
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("   ✅ xFormers enabled")
        except Exception as e:
            print(f"   ⚠️  xFormers not available: {e}")
            try:
                pipeline.enable_attention_slicing(1)
                print("   ✅ Attention slicing enabled")
            except:
                pass
        
        # Step 3: Load 4-step distilled LoRA
        print("\n[3/3] Loading 4-step distilled LoRA...")
        try:
            # Load low noise LoRA for more stable outputs
            pipeline.load_lora_weights(
                "lightx2v/Wan2.2-Distill-Loras",
                weight_name="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors",
                adapter_name="wan_4step"
            )
            
            # Set LoRA scale to 1.0 (full strength)
            pipeline.set_adapters(["wan_4step"], adapter_weights=[1.0])
            
            print("   ✅ 4-step LoRA loaded (low noise)")
            print("   ℹ️  Inference: 4 steps total")
        except Exception as e:
            print(f"   ⚠️  LoRA loading failed: {e}")
            print("   ℹ️  Falling back to base model (25 steps)")
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        model_loaded = True
        print("\n✅ Model fully loaded and ready!")
        print("=" * 80)
        
        return pipeline
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: Model loading failed!")
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
    """
    Main RunPod handler
    
    Input:
    {
        "input": {
            "image_base64": "base64_encoded_image",
            "prompt": "camera zoom in",  // OPTIONAL
            "quality": "standard",  // draft, standard, high
            "fps": 16  // OPTIONAL
        }
    }
    """
    start_time = time.time()
    output_path = None
    
    try:
        print("\n" + "=" * 80)
        print("🎬 New Generation Request")
        print("=" * 80)
        
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
        
        # Map quality to frames (Wan2.2 supports variable lengths)
        quality_map = {
            "draft": 49,    # ~3 seconds at 16fps
            "standard": 65, # ~4 seconds at 16fps
            "high": 81      # ~5 seconds at 16fps
        }
        num_frames = input_data.get("num_frames", quality_map.get(quality, 65))
        
        print(f"\n⚙️  Configuration:")
        print(f"   Quality: {quality}")
        print(f"   Frames: {num_frames} (~{num_frames/fps:.1f}s)")
        print(f"   FPS: {fps}")
        if prompt:
            print(f"   Prompt: '{prompt}'")
        else:
            print(f"   Prompt: None (image only)")
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"\n💾 GPU Memory: {mem_allocated:.2f} GB allocated")
        
        # Load model
        pipe = load_model()
        
        # Process image
        print(f"\n🖼️  Processing input image...")
        image = base64_to_image(image_base64)
        original_size = image.size
        print(f"   Original: {original_size}")
        
        # Resize to 720p (1280x720) - Wan2.2 optimized resolution
        image = image.resize((1280, 720), Image.LANCZOS)
        print(f"   Resized: {image.size}")
        
        # Clear memory before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate video
        print(f"\n🎨 Generating video...")
        gen_start = time.time()
        
        # Use 4-step inference with distilled LoRA
        # Recommended timesteps for 4-step: [1000, 750, 500, 250]
        num_inference_steps = 4
        timesteps = [1000, 750, 500, 250]
        
        print(f"   Inference: {num_inference_steps} steps")
        print(f"   Timesteps: {timesteps}")
        
        with torch.no_grad():
            if prompt:
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=720,
                    width=1280,
                    timesteps=timesteps
                )
            else:
                # Image-only mode (no prompt)
                output = pipe(
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    height=720,
                    width=1280,
                    timesteps=timesteps
                )
        
        frames = output.frames[0]
        gen_time = time.time() - gen_start
        
        print(f"   ✅ Generated in {gen_time:.2f}s")
        print(f"   Speed: {num_frames/gen_time:.1f} fps processing")
        
        # Clear memory after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Export video
        print(f"\n💾 Exporting video...")
        output_path = f"/tmp/video_{int(time.time())}.mp4"
        export_to_video(frames, output_path, fps=fps)
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"   Size: {file_size_mb:.2f} MB")
        
        # Encode to base64
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
        print(f"   File: {file_size_mb:.2f} MB")
        print("=" * 80)
        
        result = {
            "video_base64": video_base64,
            "width": 1280,
            "height": 720,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": round(total_time, 2),
            "generation_time": round(gen_time, 2),
            "quality": quality,
            "file_size_mb": round(file_size_mb, 2),
            "inference_steps": num_inference_steps,
            "model": "Wan2.2-I2V-A14B-4Step"
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
        print("=" * 80)
        
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
    print("\n✅ Handler ready for requests...")
    print("   Model: Wan2.2-I2V-A14B")
    print("   LoRA: 4-step distilled (low noise)")
    print("   Resolution: 1280x720")
    print("   Inference: 4 steps")
    print("=" * 80 + "\n")
    
    runpod.serverless.start({"handler": handler})
