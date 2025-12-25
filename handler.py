"""
VidGen Serverless - Image to Video Generation Handler
Model: Stable Video Diffusion (SVD)
Platform: RunPod Serverless
"""

import runpod
import base64
import io
import os
import time
from PIL import Image
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

# Global pipeline variable
pipeline = None

def load_model():
    """Load Stable Video Diffusion model"""
    global pipeline
    
    if pipeline is None:
        print("🔄 Loading Stable Video Diffusion XL...")
        
        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        pipeline.to("cuda")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_vae_slicing()
        
        print("✅ Model loaded successfully!")
    
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
            "num_frames": 25,
            "fps": 8,
            "motion_bucket_id": 127,
            "noise_aug_strength": 0.02,
            "num_inference_steps": 25
        }
    }
    
    Output:
    {
        "video_base64": "base64_encoded_video",
        "width": 1024,
        "height": 576,
        "fps": 8,
        "num_frames": 25,
        "processing_time": 45.2,
        "generation_time": 42.1
    }
    """
    start_time = time.time()
    
    try:
        print("=" * 70)
        print("🎬 VidGen: Starting Image-to-Video Generation")
        print("=" * 70)
        
        # Extract input data
        input_data = event.get("input", {})
        
        if not input_data:
            return {"error": "No input data provided"}
        
        image_base64 = input_data.get("image_base64")
        
        if not image_base64:
            return {"error": "Missing required field: image_base64"}
        
        # Parameters with defaults
        num_frames = input_data.get("num_frames", 25)
        fps = input_data.get("fps", 8)
        motion_bucket_id = input_data.get("motion_bucket_id", 127)
        noise_aug_strength = input_data.get("noise_aug_strength", 0.02)
        num_inference_steps = input_data.get("num_inference_steps", 25)
        
        print(f"\n⚙️  Configuration:")
        print(f"   • Frames: {num_frames}")
        print(f"   • FPS: {fps}")
        print(f"   • Motion Intensity: {motion_bucket_id}")
        print(f"   • Inference Steps: {num_inference_steps}")
        
        # Load model (cached after first call)
        print(f"\n📦 Loading model...")
        pipe = load_model()
        
        # Process input image
        print(f"\n🖼️  Processing input image...")
        image = base64_to_image(image_base64)
        original_size = image.size
        print(f"   • Original size: {original_size}")
        
        # Resize to optimal SVD resolution (1024x576)
        image = image.resize((1024, 576), Image.LANCZOS)
        print(f"   • Resized to: {image.size}")
        
        # Generate video
        print(f"\n🎨 Generating {num_frames} frames...")
        gen_start = time.time()
        
        frames = pipe(
            image=image,
            num_frames=num_frames,
            decode_chunk_size=8,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            num_inference_steps=num_inference_steps
        ).frames[0]
        
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
        print("=" * 70)
        
        return {
            "video_base64": video_base64,
            "width": 1024,
            "height": 576,
            "fps": fps,
            "num_frames": num_frames,
            "processing_time": round(total_time, 2),
            "generation_time": round(gen_time, 2)
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
