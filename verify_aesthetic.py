import os
import shutil
# Check if PIL is installed, if not we can't run this locally but it will run in docker
try:
    from hooks import create_hook_image
except ImportError:
    print("⚠️ PIL not found locally. Please run this inside the Docker container.")
    # Mocking for local check if needed or just exit
    exit(1)

def verify():
    print("🧪 Verifying Hook Aesthetics...")
    
    test_text = "POV: You are testing\nthe new aesthetic feature\nwith explicit lines."
    output_path = "aesthetic_hook.png"
    target_width = 800
    
    try:
        path, w, h = create_hook_image(test_text, target_width, output_image_path=output_path)
        
        print(f"✅ Image generated at {path}")
        print(f"   Dimensions including shadow: {w}x{h}")
        
        # Verify it's larger than the text box would be (due to shadow/padding)
        # Just rudimentary checks
        if not os.path.exists(path):
            print("❌ File does not exist")
            return False
            
        print("✨ Verification Successful! (Inspect aesthetic_hook.png visually)")
        return True
    except Exception as e:
        print(f"❌ Verification Failed: {e}")
        return False

if __name__ == "__main__":
    verify()
