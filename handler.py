"""
VidGen Serverless - Image to Video Generation Handler
Model: Wan2.2-I2V-A14B with Distilled LoRA (4-step inference)
Platform: RunPod Serverless
"""

import runpod
import base64
import io
import os
import time
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
        
        # Load base pipeline
        pipeline = WanPipeline.from_pretrained(
            "Wan-AI/Wan2.2-I2V-A14B",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipeline.to("cuda")
        
        # Memory optimizations
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            print("✅ xFormers enabled")
        except:
            print("⚠️ xFormers not available")
        
        # Load distilled LoRA for 4-step inference
        print("📦 Loading distilled LoRA...")
        pipeline.load_lora_weights(
            "lightx2v/Wan2.2-Distill-Loras",
            weight_name="wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors"
        )
        
        print("✅ Model loaded successfully with 4-step LoRA!")
    
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
            "num_frames": 81,
            "fps": 16,
            "quality": "standard"
        }
    }
    
    Output:
    {
        "video_base64": "base64_encoded_video",
        "width": 1280,
        "height": 720,
        "fps": 16,
        "num_frames": 81,
        "processing_time": 35.2,
        "generation_time": 32.1
    }
    """
    start_time = time.time()
    
    try:
        print("=" * 70)
        print("🎬 VidGen: Wan2.2 4-Step Generation")
        print("=" * 70)
        
        # Extract input data
        input_data = event.get("input", {})
        
        if not input_data:
            return {"error": "No input data provided"}
        
        image_base64 = input_data.get("image_base64")
        
        if not image_base64:
            return {"error": "Missing required field: image_base64"}
        
        # Parameters with defaults
        quality = input_data.get("quality", "standard")  # draft, standard, high
        
        # Map quality to parameters
        if quality == "draft":
            num_frames = 49  # ~3 seconds at 16fps
            num_inference_steps = 4
        elif quality == "high":
            num_frames = 81  # ~5 seconds at 16fps
            num_inference_steps = 4
        else:  # standard
            num_frames = 65  # ~4 seconds at 16fps
            num_inference_steps = 4
        
        fps = 16  # Wan2.2 optimal FPS
        
        print(f"\n⚙️  Configuration:")
        print(f"   • Quality: {quality}")
        print(f"   • Frames: {num_frames}")
        print(f"   • FPS: {fps}")
        print(f"   • Inference Steps: {num_inference_steps} (4-step distilled)")
        
        # Load model
        print(f"\n📦 Loading model...")
        pipe = load_model()
        
        # Process input image
        print(f"\n🖼️  Processing input image...")
        image = base64_to_image(image_base64)
        original_size = image.size
        print(f"   • Original size: {original_size}")
        
        # Resize to Wan2.2 optimal resolution (1280x720)
        image = image.resize((1280, 720), Image.LANCZOS)
        print(f"   • Resized to: {image.size}")
        
        # Generate video
        print(f"\n🎨 Generating {num_frames} frames with 4-step inference...")
        gen_start = time.time()
        
        # Use 4-step denoising schedule for distilled model
        denoising_steps = [1000, 750, 500, 250]
        
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
        print(f"   ✅ Generated in {gen_time:.2f}s")
        
        # Save video to temporary file
        print(f"\n💾 Saving video...")
        output_path = "/tmp/output_video.mp4"
        export_to_video(frames, output_path, fps=fps)
        
        # Get file size
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
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Success!")
        print(f"   • Total time: {total_time:.2f}s")
        print(f"   • Generation time: {gen_time:.2f}s")
        print(f"   • Speed: {num_frames / gen_time:.1f} fps")
        print("=" * 70)
        
        return {
            "video_base64": video_base64,
            "width": 1280,
            "height": 720,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": round(total_time, 2),
            "generation_time": round(gen_time, 2),
            "quality": quality
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"\n❌ ERROR: {error_msg}")
        print(f"\n📋 Traceback:\n{error_trace}")
        
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
    
    print("\n✅ Handler ready, waiting for requests...")
    
    runpod.serverless.start({"handler": handler})
