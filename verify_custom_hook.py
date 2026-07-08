import os
try:
    from hooks import create_hook_image
except ImportError:
    print("⚠️ PIL not found locally. Run inside Docker.")
    exit(1)

def verify():
    print("🧪 Verifying Hook Customization...")
    test_text = "Custom Position\n& Size Test"
    
    # Test 1: Small + Top
    print("   Testing Small + Top...")
    p1, w1, h1 = create_hook_image(test_text, 800, "hook_small.png", font_scale=0.8)
    print(f"   ✅ Small: {w1}x{h1}")
    
    # Test 2: Large + Center
    print("   Testing Large...")
    p2, w2, h2 = create_hook_image(test_text, 800, "hook_large.png", font_scale=1.3)
    print(f"   ✅ Large: {w2}x{h2}")
    
    if w2 > w1 and h2 > h1:
        print("   ✅ Scaling logic works (Large > Small)")
    else:
        print("   ❌ Scaling logic failed")
        
    # Cleanup
    if os.path.exists(p1): os.remove(p1)
    if os.path.exists(p2): os.remove(p2)

if __name__ == "__main__":
    verify()
