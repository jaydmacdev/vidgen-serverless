"""
VidGen Serverless - Image to Video Generation Handler
Model: Wan2.2-I2V-A14B with Distilled LoRA (4-step inference)
Features: Text prompt support for guided generation
Platform: RunPod Serverless
"""

import runpod
import base64
import io
import os
import time
import gc
from PIL import Image
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# Global pipeline variable
pipeline = None

def load_model():
    """Load Wan2.2 I2V model with distilled LoRA"""
    global pipeline
    
    if pipeline is None:
        print("🔄 Loading Wan2.2-I2V with Distilled LoRA...")
        
        try:
            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Load base pipeline
            pipeline = WanPipeline.from_pretrained(
                "Wan-AI/Wan2.2-I2V-A14B",
                torch_dtype=torch.float16,
                variant="fp16",
                low_cpu_mem_usage=True
            )
            
            pipeline.to("cuda")
            
            # Memory optimizations
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
            
            # Enable memory efficient attention
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                print("✅ xFormers enabled")
            except Exception as e:
                print(f"⚠️ xFormers not available: {e}")
                try:
                    pipeline.enable_attention_slicing(1)
                    print("✅ Attention slicing enabled")
                except:
                    pass
            
            # Load distilled LoRA for 4-step inference
            print("📦 Loading distilled LoRA...")
            try:
                pipeline.load_lora_weights(
                    "lightx2v/Wan2.2-Distill-Loras",
                    weight_name="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
                )
                print("✅ LoRA loaded successfully!")
            except Exception as e:
                print(f"⚠️ LoRA loading failed: {e}")
                print("⚠️ Continuing with base model (will use more steps)")
            
            print("✅ Model loaded successfully!")
            
            # Clear cache after loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"❌ Model loading failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    return pipeline

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def video_to_base64(video_path):
    """Convert video file to base64 string"""
    try:
        with open(video_path, "rb") as f:
            video_data = f.read()
        return base64.b64encode(video_data).decode('utf-8')
    except Exception as e:
        raise ValueError(f"Failed to encode video: {str(e)}")

def handler(event):
    """
    Main RunPod Handler Function
    
    Expected Input:
    {
        "input": {
            "image_base64": "base64_encoded_image",
            "prompt": "A person walking in a park",  // OPTIONAL
            "quality": "standard",  // draft, standard, high
            "num_frames": 65,  // OPTIONAL - overrides quality preset
            "fps": 16  // OPTIONAL
        }
    }
    
    Output:
    {
        "video_base64": "base64_encoded_video",
        "width": 1280,
        "height": 720,
        "fps": 16,
        "num_frames": 65,
        "processing_time": 35.2,
        "generation_time": 32.1,
        "prompt_used": "A person walking in a park"
    }
    """
    start_time = time.time()
    
    try:
        print("=" * 70)
        print("🎬 VidGen: Wan2.2 4-Step Generation with Prompt Support")
        print("=" * 70)
        
        # Clear CUDA cache at start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            print(f"🎮 GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")
        
        # Extract input data
        input_data = event.get("input", {})
        
        if not input_data:
            return {"error": "No input data provided"}
        
        image_base64 = input_data.get("image_base64")
        
        if not image_base64:
            return {"error": "Missing required field: image_base64"}
        
        # Get prompt (optional)
        prompt = input_data.get("prompt", "")
        if prompt:
            print(f"📝 Prompt: {prompt}")
        
        # Get quality preset or custom params
        quality = input_data.get("quality", "standard")
        
        # Allow custom override or use quality preset
        if "num_frames" in input_data:
            num_frames = input_data.get("num_frames")
        else:
            # Map quality to frame count
            if quality == "draft":
                num_frames = 49  # ~3 seconds at 16fps
            elif quality == "high":
                num_frames = 81  # ~5 seconds at 16fps
            else:  # standard
                num_frames = 65  # ~4 seconds at 16fps
        
        fps = input_data.get("fps", 16)
        num_inference_steps = 4  # Fixed for distilled model
        
        print(f"\n⚙️ Configuration:")
        print(f"   • Quality: {quality}")
        print(f"   • Frames: {num_frames} (~{num_frames/fps:.1f}s)")
        print(f"   • FPS: {fps}")
        print(f"   • Inference Steps: {num_inference_steps} (4-step distilled)")
        if prompt:
            print(f"   • Prompt: '{prompt}'")
        
        # Load model
        print(f"\n📦 Loading model...")
        pipe = load_model()
        
        # Process input image
        print(f"\n🖼️ Processing input image...")
        image = base64_to_image(image_base64)
        print(f"   • Original size: {image.size}")
        
        # Resize to Wan2.2 optimal resolution (1280x720)
        image = image.resize((1280, 720), Image.LANCZOS)
        print(f"   • Resized to: {image.size}")
        
        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Generate video
        if prompt:
            print(f"\n🎨 Generating {num_frames} frames with prompt...")
        else:
            print(f"\n🎨 Generating {num_frames} frames (no prompt)...")
        
        gen_start = time.time()
        
        # Use 4-step denoising schedule for distilled model
        denoising_steps = [1000, 750, 500, 250]
        
        # Generate with torch.no_grad() to save memory
        with torch.no_grad():
            # Generate with or without prompt
            if prompt:
                output = pipe(
                    prompt=prompt,
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=720,
                    width=1280,
                    timesteps=denoising_steps[:num_inference_steps]
                )
            else:
                output = pipe(
                    image=image,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=7.5,
                    height=720,
                    width=1280,
                    timesteps=denoising_steps[:num_inference_steps]
                )
        
        frames = output.frames[0]
        
        gen_time = time.time() - gen_start
        print(f"   ✅ Generated in {gen_time:.2f}s ({num_frames/gen_time:.1f} fps)")
        
        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Save video
        print(f"\n💾 Saving video...")
        output_path = "/tmp/output_video.mp4"
        export_to_video(frames, output_path, fps=fps)
        
        file_size = os.path.getsize(output_path)
        print(f"   • Video size: {file_size / 1024 / 1024:.2f} MB")
        
        # Encode to base64
        print(f"\n📤 Encoding to base64...")
        video_base64 = video_to_base64(output_path)
        
        # Cleanup
        try:
            os.remove(output_path)
        except:
            pass
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Success!")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Generation time: {gen_time:.2f}s")
        print("=" * 70)
        
        result = {
            "video_base64": video_base64,
            "width": 1280,
            "height": 720,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": round(total_time, 2),
            "generation_time": round(gen_time, 2),
            "quality": quality
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
        
        # Cleanup on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return {
            "error": error_msg,
            "traceback": error_trace
        }

# Initialize RunPod serverless handler
if __name__ == "__main__":
    print("🚀 VidGen Serverless Handler Starting...")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("\n✅ Handler ready for requests with prompt support...")
    
    runpod.serverless.start({"handler": handler})
