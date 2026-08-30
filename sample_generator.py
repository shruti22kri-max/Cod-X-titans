"""
Synthetic Side-Scan Sonar Sample Generator
Generates realistic side-scan sonar demo images for:
1. Small Object Detection
2. Unknown Anomaly Detection
3. Ghost Net Detection
"""

import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter

def generate_sonar_texture(width=900, height=600, style="amber"):
    """
    Creates a base side-scan sonar acoustic scan texture with nadir track and acoustic speckle.
    """
    img = Image.new("RGB", (width, height), (15, 10, 5) if style == "amber" else (10, 15, 25))
    draw = ImageDraw.Draw(img)
    
    center_x = width // 2
    nadir_width = 40
    
    # Fill seafloor with sonar reverberation gradient
    for y in range(height):
        scan_line_noise = random.randint(-8, 8)
        
        # Left port side beam
        for x in range(0, center_x - nadir_width // 2, 4):
            dist_from_nadir = (center_x - nadir_width // 2 - x) / (center_x - nadir_width // 2)
            intensity = int(max(0, min(255, 135 * (1.0 - 0.45 * dist_from_nadir) + random.randint(-20, 20) + scan_line_noise)))
            ripple = math.sin(y * 0.045 + x * 0.018) * 14
            intensity = max(0, min(255, int(intensity + ripple)))
            
            if style == "amber":
                color = (int(intensity * 1.0), int(intensity * 0.76), int(intensity * 0.36))
            else:
                color = (int(intensity * 0.3), int(intensity * 0.7), int(intensity * 0.9))
                
            draw.rectangle([x, y, x + 3, y], fill=color)
            
        # Right starboard side beam
        for x in range(center_x + nadir_width // 2, width, 4):
            dist_from_nadir = (x - (center_x + nadir_width // 2)) / (width - (center_x + nadir_width // 2))
            intensity = int(max(0, min(255, 135 * (1.0 - 0.45 * dist_from_nadir) + random.randint(-20, 20) + scan_line_noise)))
            ripple = math.sin(y * 0.045 - x * 0.018) * 14
            intensity = max(0, min(255, int(intensity + ripple)))
            
            if style == "amber":
                color = (int(intensity * 1.0), int(intensity * 0.76), int(intensity * 0.36))
            else:
                color = (int(intensity * 0.3), int(intensity * 0.7), int(intensity * 0.9))
                
            draw.rectangle([x, y, x + 3, y], fill=color)
            
    # Central nadir water column (black acoustic void)
    draw.rectangle([center_x - nadir_width // 2, 0, center_x + nadir_width // 2, height], fill=(4, 4, 4))
    
    # Altitude boundary line
    draw.line([center_x - nadir_width // 2, 0, center_x - nadir_width // 2, height], fill=(220, 180, 100), width=1)
    draw.line([center_x + nadir_width // 2, 0, center_x + nadir_width // 2, height], fill=(220, 180, 100), width=1)
    
    return img

def add_small_object_target(img, x, y, w=48, h=40):
    """
    Renders a compact small object (canister/lost equipment/small metal piece)
    with a pinpoint high-reflectivity acoustic highlight and tight acoustic shadow.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    center_x = width // 2
    is_starboard = (x >= center_x)
    shadow_dir = 1 if is_starboard else -1
    
    # Acoustic Shadow trailing away from nadir
    shadow_len = int(w * 2.2)
    if is_starboard:
        shadow_box = [x + w - 4, y + 4, x + w + shadow_len, y + h - 4]
    else:
        shadow_box = [x - shadow_len, y + 4, x + 4, y + h - 4]
    draw.ellipse(shadow_box, fill=(4, 3, 2))
    
    # Pinpoint high-contrast acoustic highlight
    draw.ellipse([x, y, x + w, y + h], fill=(255, 248, 210), outline=(255, 255, 240), width=2)
    draw.ellipse([x + 6, y + 6, x + w - 6, y + h - 6], fill=(255, 255, 230))
    return img

def add_unknown_anomaly_target(img, x, y, w=110, h=95):
    """
    Renders an unknown seabed acoustic anomaly (unidentified seafloor disturbance,
    irregular geological or unclassified target return with acoustic halo).
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    center_x = width // 2
    is_starboard = (x >= center_x)
    
    # Irregular dark acoustic shadow
    shadow_len = int(w * 1.6)
    if is_starboard:
        shadow_poly = [
            (x + w - 10, y + 5), (x + w + shadow_len, y - 10),
            (x + w + shadow_len + 15, y + h + 15), (x + w - 5, y + h - 5)
        ]
    else:
        shadow_poly = [
            (x + 10, y + 5), (x - shadow_len, y - 10),
            (x - shadow_len - 15, y + h + 15), (x + 5, y + h - 5)
        ]
    draw.polygon(shadow_poly, fill=(5, 3, 2))
    
    # Diffuse acoustic disturbance outer ring
    draw.ellipse([x - 12, y - 10, x + w + 12, y + h + 10], outline=(180, 140, 70), width=3)
    
    # Complex irregular anomaly shape
    points = [
        (x + 15, y), (x + w - 20, y + 10), (x + w, y + h // 2),
        (x + w - 15, y + h), (x + 25, y + h - 8), (x, y + h // 2 + 10)
    ]
    draw.polygon(points, fill=(255, 220, 140), outline=(255, 255, 210))
    
    # Internal acoustic perturbation lines
    for i in range(4):
        offset = 12 * i
        draw.arc([x + offset, y + offset // 2, x + w - offset, y + h - offset // 2], 0, 260, fill=(255, 255, 230), width=2)
        
    return img

def add_ghost_net_target(img, x, y, w=150, h=110):
    """
    Renders a ghost fishing net draped over seafloor: tangled acoustic netting filaments,
    diffuse acoustic backscatter, and irregular trailing shadow contours.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size
    center_x = width // 2
    is_starboard = (x >= center_x)
    
    # Netting shadow profile
    shadow_len = int(w * 1.4)
    if is_starboard:
        shadow_pts = [
            (x + w - 15, y + 10), (x + w + shadow_len, y + 25),
            (x + w + shadow_len - 20, y + h + 10), (x + w - 10, y + h - 10)
        ]
    else:
        shadow_pts = [
            (x + 15, y + 10), (x - shadow_len, y + 25),
            (x - shadow_len + 20, y + h + 10), (x + 10, y + h - 10)
        ]
    draw.polygon(shadow_pts, fill=(6, 4, 3))
    
    # Intertwined filament netting mesh pattern
    step = 16
    for i in range(0, w, step):
        # Wavy warp filaments
        pts = []
        for j in range(0, h, 8):
            wave = math.sin((i + j) * 0.15) * 6
            pts.append((x + i + wave, y + j))
        draw.line(pts, fill=(245, 230, 175), width=2)
        
    for j in range(0, h, step):
        # Wavy weft filaments
        pts = []
        for i in range(0, w, 8):
            wave = math.cos((i + j) * 0.15) * 6
            pts.append((x + i, y + j + wave))
        draw.line(pts, fill=(240, 220, 160), width=2)
        
    # Boundary floaters / leadline weights acoustic bright spots
    for bx in range(x, x + w, 28):
        draw.ellipse([bx, y - 2, bx + 8, y + 6], fill=(255, 255, 220), outline=(255, 240, 180))
        draw.ellipse([bx + 4, y + h - 6, bx + 12, y + h + 2], fill=(255, 255, 220), outline=(255, 240, 180))
        
    return img

def generate_sample_images(output_dir="sample_images"):
    """
    Generates the 3 primary test samples:
    1. Small Object Detection
    2. Unknown Anomaly Detection
    3. Ghost Net Detection
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Small Object Detection Sonar Scan
    img1 = generate_sonar_texture(900, 600, "amber")
    img1 = add_small_object_target(img1, 380, 220, 52, 44)
    path1 = os.path.join(output_dir, "sample_small_object.jpg")
    img1.save(path1, "JPEG", quality=92)
    
    # 2. Unknown Anomaly Detection Sonar Scan
    img2 = generate_sonar_texture(900, 600, "amber")
    img2 = add_unknown_anomaly_target(img2, 340, 190, 115, 100)
    path2 = os.path.join(output_dir, "sample_unknown_anomaly.jpg")
    img2.save(path2, "JPEG", quality=92)
    
    # 3. Ghost Net Detection Sonar Scan
    img3 = generate_sonar_texture(900, 600, "amber")
    img3 = add_ghost_net_target(img3, 310, 175, 155, 115)
    path3 = os.path.join(output_dir, "sample_ghost_net.jpg")
    img3.save(path3, "JPEG", quality=92)
    
    print("Generated 3 new synthetic sonar samples:")
    print(" - sample_small_object.jpg")
    print(" - sample_unknown_anomaly.jpg")
    print(" - sample_ghost_net.jpg")

if __name__ == "__main__":
    generate_sample_images()
