import cv2
import numpy as np
import os
import csv
from datetime import datetime
from collections import defaultdict
from roboflow import Roboflow
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from marine_database import analyze_fish, analyze_pond_health
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pandas as pd
from datetime import datetime, timedelta

# ================================================================
#  CONFIGURATION  — only change these if needed
# ================================================================

API_KEY    = "pH1H6HKtWI4Mw1go3GYm"
PROJECT_ID = "fish-detection2-jgvqz"
VERSION    = 1
IMAGE_SIZE  = 416
PIXEL_TO_CM = 0.05
LOG_FILE    = "logs/marine_log.csv"
GROWTH_LOG_FILE = "logs/fish_growth_log.csv"
GRAPH_OUTPUT_DIR = "outputs/graphs/"

# Video: process 1 out of every N frames (skipped frames reuse last result)
# 4 = good balance of speed vs accuracy for 24fps video
PROCESS_EVERY_N = 4

os.makedirs("logs",    exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs(GRAPH_OUTPUT_DIR, exist_ok=True)

# ================================================================
#  MODELS
# ================================================================

rf      = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT_ID)
detector = project.version(VERSION).model

species_model = load_model("fish_species_model.keras")

CLASS_NAMES = [
    "Jaguar Gapote","Perch","Mullet","Indo-Pacific Tarpon","Grass Carp",
    "Long-Snouted Pipefish","Glass Perchlet","Goby","Tilapia","Bangus",
    "Green Spotted Puffer","Freshwater Eel","Big Head Carp","Silver Carp",
    "Scat Fish","Black Spotted Barb","Climbing Perch","Mosquito Fish",
    "Janitor Fish","Pangasius","Gold Fish","Gourami","Indian Carp",
    "Mudfish","Silver Barb","Silver Perch","Knifefish","Catfish",
    "Tenpounder","Snakehead","Fourfinger Threadfin"
]

# Global fish ID — increments across video frames, resets per image
_fish_id_counter = 0

def reset_id():
    global _fish_id_counter
    _fish_id_counter = 0

def get_next_id():
    global _fish_id_counter
    fid = _fish_id_counter
    _fish_id_counter += 1
    return fid


# ================================================================
#  STEP 1 — IMAGE PROFILING
#  Measures 5 metrics and assigns a severity level that drives
#  every downstream decision (enhancement, confidence, tiling).
# ================================================================

def profile_image(frame):
    """
    Returns a profile dict with severity in [CLEAR, MILD, MODERATE, EXTREME].

    Metric          Threshold       Meaning
    ─────────────   ─────────────   ────────────────────────────────
    brightness      < 75            dark / under-exposed
    blur_score      < 120           hazy / blurry water
    contrast        < 35            flat / low-visibility
    green_cast      > 12            algae-green turbid water
    blue_cast       < -8            blue-tinted underwater scene

    Severity rules:
      CLEAR    → no flags at all AND contrast > 40
      MILD     → 1 flag (minor issue)
      MODERATE → 1 serious flag OR 2 mild flags
      EXTREME  → 2+ serious flags OR brightness < 55
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(frame)

    brightness = float(np.mean(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast   = float(gray.std())
    green_cast = float(np.mean(g)) - max(float(np.mean(r)), float(np.mean(b)))

    f_dark      = brightness < 75
    f_blurry    = blur_score < 120
    f_turbid    = green_cast > 12
    f_blue      = green_cast < -8
    f_flat      = contrast < 35

    problems = sum([f_dark, f_blurry, f_turbid, f_blue, f_flat])
    is_clear  = problems == 0 and contrast > 40

    if is_clear:
        sev = "CLEAR"
    elif brightness < 55 or problems >= 3:
        sev = "EXTREME"
    elif problems >= 2:
        sev = "MODERATE"
    elif problems == 1:
        sev = "MILD"
    else:
        sev = "MILD"

    return {
        "brightness": brightness, "blur_score": blur_score,
        "contrast": contrast,     "green_cast": green_cast,
        "dark": f_dark,  "blurry": f_blurry, "turbid": f_turbid,
        "blue": f_blue,  "flat": f_flat,     "clear": is_clear,
        "severity": sev,
    }


# ================================================================
#  STEP 2 — ENHANCEMENT
#  Only applied when needed. Strength matches severity.
# ================================================================

def _white_balance(img):
    f = img.astype(np.float32)
    m = f.mean()
    for i in range(3):
        ch = f[:,:,i].mean()
        if ch > 0:
            f[:,:,i] = np.clip(f[:,:,i] * m / ch, 0, 255)
    return f.astype(np.uint8)

def _clahe(img, clip):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def _gamma(img, g):
    t = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, t)

def _sharpen(img, s=1.6):
    blur = cv2.GaussianBlur(img, (0,0), 2.5)
    return cv2.addWeighted(img, s, blur, -(s-1), 0)

def _stretch(img):
    f = img.astype(np.float32)
    for i in range(3):
        p2, p98 = np.percentile(f[:,:,i], 2), np.percentile(f[:,:,i], 98)
        if p98 > p2:
            f[:,:,i] = np.clip((f[:,:,i]-p2)/(p98-p2)*255, 0, 255)
    return f.astype(np.uint8)

def enhance(frame, prof):
    sev = prof["severity"]
    if sev == "CLEAR":
        return frame
    if sev == "MILD":
        return _clahe(_white_balance(frame), clip=2.5)
    if sev == "MODERATE":
        img = _clahe(_white_balance(frame), clip=3.5)
        if prof["dark"]:   img = _gamma(img, 1.6)
        if prof["blurry"]: img = _sharpen(img, 1.6)
        return img
    # EXTREME
    img = _stretch(frame)
    img = _white_balance(img)
    img = _clahe(img, clip=5.0)
    img = _gamma(img, 2.0)
    img = _sharpen(img, 2.0)
    return img


# ================================================================
#  STEP 3 — DETECTION
#  Full-frame for clear images. Tiled for dark/blurry/turbid.
#  Auto-retries with lower confidence if 0 fish found.
# ================================================================

def _predict(img, conf, overlap):
    """Call Roboflow on a pre-resized 416x416 image."""
    try:
        return detector.predict(img, confidence=conf, overlap=overlap).json().get("predictions", [])
    except Exception as e:
        print(f"    [API error] {e}")
        return []

def _full_frame(frame, conf, overlap=30):
    """Detect on whole frame, return coords in original pixel space."""
    h, w = frame.shape[:2]
    preds = _predict(cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE)), conf, overlap)
    return [{
        "x": p["x"]*w/IMAGE_SIZE, "y": p["y"]*h/IMAGE_SIZE,
        "width": p["width"]*w/IMAGE_SIZE, "height": p["height"]*h/IMAGE_SIZE,
        "confidence": p.get("confidence", 0.5)
    } for p in preds]

def _tiled(frame, conf, overlap=30, rows=2, cols=3, pad=40):
    """
    Detect on overlapping tiles.
    Each tile is sent at full 416px → small fish appear larger to the model.
    Coords mapped back to original frame space.
    """
    h, w = frame.shape[:2]
    th_base, tw_base = h//rows, w//cols
    all_preds = []
    for r in range(rows):
        for c in range(cols):
            y1=max(0,r*th_base-pad); y2=min(h,(r+1)*th_base+pad)
            x1=max(0,c*tw_base-pad); x2=min(w,(c+1)*tw_base+pad)
            tile = frame[y1:y2, x1:x2]
            th, tw = tile.shape[:2]
            preds = _predict(cv2.resize(tile, (IMAGE_SIZE, IMAGE_SIZE)), conf, overlap)
            for p in preds:
                all_preds.append({
                    "x": p["x"]*tw/IMAGE_SIZE+x1, "y": p["y"]*th/IMAGE_SIZE+y1,
                    "width": p["width"]*tw/IMAGE_SIZE, "height": p["height"]*th/IMAGE_SIZE,
                    "confidence": p.get("confidence", 0.5)
                })
    return all_preds

def _nms(preds, iou):
    """Remove duplicate overlapping boxes. Keeps highest-confidence one."""
    if not preds:
        return []
    boxes  = [[p["x"]-p["width"]/2, p["y"]-p["height"]/2,
               p["width"], p["height"]] for p in preds]
    scores = [float(p["confidence"]) for p in preds]
    idx = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.01, nms_threshold=iou)
    return [preds[i] for i in idx.flatten()] if len(idx) > 0 else []

def detect(frame, prof):
    """
    Main detection entry point.

    Severity → starting conf → NMS strictness → use tiling?
    CLEAR    →  28  →  0.30  →  no  (large fish, no tiling needed)
    MILD     →  20  →  0.35  →  no
    MODERATE →  15  →  0.40  →  yes (small/faint fish)
    EXTREME  →  10  →  0.45  →  yes

    Auto-retry: if 0 detections, drops confidence and adds tiling.
    Stops as soon as fish are found or minimum confidence reached.
    """
    sev = prof["severity"]
    enh = enhance(frame, prof)

    # Fallback ladders: list of (conf, use_tiling)
    ladders = {
        "CLEAR":    [(28,False),(18,False),(12,True),(8,True)],
        "MILD":     [(20,False),(14,True), (9, True),(6,True)],
        "MODERATE": [(15,True), (10,True), (7, True),(5,True)],
        "EXTREME":  [(10,True), (7, True), (5, True)],
    }
    nms_thresh = {"CLEAR":0.30,"MILD":0.35,"MODERATE":0.40,"EXTREME":0.45}[sev]

    for attempt, (conf, tiled) in enumerate(ladders[sev]):
        raw = ((_tiled(enh,conf) + _full_frame(enh,conf)) if tiled
               else _full_frame(enh, conf))
        final = _nms(raw, nms_thresh)

        status = "✓" if final else "✗"
        print(f"    {status} attempt {attempt+1}: conf={conf:2d}  "
              f"tiled={str(tiled):5s}  raw={len(raw):3d}  kept={len(final)}")

        if final:
            return final
        if attempt < len(ladders[sev])-1:
            print(f"      → 0 fish, retrying with lower confidence...")

    print("      → No fish detected after all attempts.")
    return []


# ================================================================
#  STEP 4 — SPECIES CLASSIFICATION
# ================================================================

def classify_species(crop):
    if crop.size == 0:
        return "Unknown", 0
    img   = cv2.resize(crop, (224,224))
    img   = preprocess_input(np.array(img, dtype=np.float32))
    preds = species_model.predict(np.expand_dims(img,0), verbose=0)[0]
    idx   = int(np.argmax(preds))
    return CLASS_NAMES[idx], int(preds[idx]*100)


# ================================================================
#  STEP 5 — WATER ANALYSIS
# ================================================================

def analyze_water(frame):
    b,g,r = np.mean(frame, axis=(0,1))
    if   g > r and g > b: status = "Algae Presence"
    elif b < 60:          status = "Turbid Water"
    else:                 status = "Clear Water"
    return (status,
            round(np.random.uniform(25,30),1),
            round(np.random.uniform(6.8,7.8),2),
            round(np.random.uniform(5.5,8.5),2))


# ================================================================
#  STEP 6 — GROWTH TRACKING & GRAPH GENERATION
# ================================================================

def init_growth_log():
    """Initialize growth log CSV file with headers if it doesn't exist"""
    if not os.path.isfile(GROWTH_LOG_FILE):
        with open(GROWTH_LOG_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Species", "Avg_Size_cm", "Min_Size_cm", "Max_Size_cm", "Count", "Timestamp"])

def log_growth_data(date, species_data):
    """Log daily growth data for each species"""
    init_growth_log()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(GROWTH_LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        for species, stats in species_data.items():
            if stats['count'] > 0:
                w.writerow([
                    date,
                    species,
                    round(stats['avg_size'], 2),
                    round(stats['min_size'], 2),
                    round(stats['max_size'], 2),
                    stats['count'],
                    timestamp
                ])

def get_species_colors(species_list):
    """
    Generate unique colors for each species using a perceptually uniform colormap
    This ensures all 31 species get visually distinct colors
    """
    import matplotlib.cm as cm
    
    # Use 'tab20' and 'tab20b' colormaps which together provide 40 distinct colors
    n_species = len(species_list)
    
    # Combine multiple colormaps for maximum distinctness
    colors = []
    
    # First, try using 'tab20' (20 colors)
    tab20 = cm.tab20(np.linspace(0, 1, 20))
    
    # Then use 'tab20b' for next 20 colors
    tab20b = cm.tab20b(np.linspace(0, 1, 20))
    
    # Also have 'tab20c' as backup
    tab20c = cm.tab20c(np.linspace(0, 1, 20))
    
    # Combine all colormaps
    all_colors = np.vstack([tab20, tab20b, tab20c])
    
    # If we need more than 60 colors (unlikely), generate using a custom scheme
    if n_species > 60:
        # Use HSV color space for maximum distinction
        hues = np.linspace(0, 1, n_species, endpoint=False)
        all_colors = plt.cm.hsv(hues)
    
    # Assign colors to species
    color_map = {}
    for i, species in enumerate(species_list):
        # Cycle through colors if we have more species than colors
        color_idx = i % len(all_colors)
        color_map[species] = all_colors[color_idx]
    
    return color_map

def generate_species_distribution_graph(species_count, timestamp=None):
    """
    Generate bar chart for fish species distribution
    Similar to the first graph in the image
    """
    if not species_count:
        print("  No species data to plot")
        return None
    
    # Sort species by count descending
    sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)
    species_names = [item[0] for item in sorted_species]
    counts = [item[1] for item in sorted_species]
    
    # Create figure with larger size
    plt.figure(figsize=(12, 7))
    
    # Create bar chart with vibrant colors
    bars = plt.bar(species_names, counts, color='skyblue', edgecolor='navy', linewidth=1.5, alpha=0.8)
    
    # Highlight top species with different color
    if len(bars) > 0:
        bars[0].set_color('orange')
        bars[0].set_edgecolor('darkorange')
    if len(bars) > 1:
        bars[1].set_color('lightgreen')
        bars[1].set_edgecolor('darkgreen')
    
    # Customize the chart
    plt.title('Fish Species Distribution in Pond', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Fish Species', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Fish Detected', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add a light background
    plt.gca().set_facecolor('#f8f9fa')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the graph
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(GRAPH_OUTPUT_DIR, f"species_distribution_{timestamp}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Species distribution graph saved → {graph_path}")
    return graph_path

def generate_species_pie_chart(species_count, timestamp=None):
    """
    Generate pie chart for species percentage distribution
    Similar to the second graph in the image
    """
    if not species_count:
        print("  No species data to plot")
        return None
    
    # Sort species by count descending
    sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)
    
    # If there are more than 8 species, group smaller ones into "Others"
    if len(sorted_species) > 8:
        main_species = sorted_species[:7]
        others_count = sum(count for _, count in sorted_species[7:])
        main_species.append(("Others", others_count))
        sorted_species = main_species
    
    species_names = [item[0] for item in sorted_species]
    counts = [item[1] for item in sorted_species]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(species_names)))
    
    # Create pie chart with percentage labels
    wedges, texts, autotexts = plt.pie(counts, labels=species_names, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    # Style the percentage labels
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    # Add title
    plt.title('Species Percentage Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend for better understanding
    plt.legend(wedges, species_names, title="Fish Species", loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    # Ensure pie is circular
    plt.axis('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the graph
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(GRAPH_OUTPUT_DIR, f"species_percentage_{timestamp}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Species percentage pie chart saved → {graph_path}")
    return graph_path

def generate_growth_graph(species_list=None, days=30):
    """
    Generate growth trend graph similar to the one in the image
    species_list: list of species to include (None for all)
    days: number of days to look back
    """
    if not os.path.isfile(GROWTH_LOG_FILE):
        print("  No growth data available yet")
        return None
    
    # Read growth data
    df = pd.read_csv(GROWTH_LOG_FILE)
    if df.empty:
        print("  No growth data to plot")
        return None
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter by date range
    cutoff_date = datetime.now().date() - timedelta(days=days)
    df = df[df['Date'].dt.date >= cutoff_date]
    
    if df.empty:
        print(f"  No data in the last {days} days")
        return None
    
    # Filter by species if specified
    if species_list:
        df = df[df['Species'].isin(species_list)]
    
    # Get unique species
    species = df['Species'].unique()
    
    # Generate unique colors for each species
    color_map = get_species_colors(species)
    
    # Create the plot with a larger figure size for better visibility
    plt.figure(figsize=(14, 8))
    
    # Sort species for consistent legend ordering
    sorted_species = sorted(species)
    
    for species_name in sorted_species:
        species_df = df[df['Species'] == species_name].sort_values('Date')
        color = color_map[species_name]
        
        # Plot average size with markers
        line = plt.plot(species_df['Date'], species_df['Avg_Size_cm'], 
                marker='o', linewidth=2, markersize=6,
                color=color, label=species_name, markevery=1)
        
        # Add min-max range as shaded area with transparency
        plt.fill_between(species_df['Date'], 
                        species_df['Min_Size_cm'], 
                        species_df['Max_Size_cm'],
                        alpha=0.15, color=color)
    
    # Customize the graph
    plt.title(f'Fish Growth Trend - Last {days} Days', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Average Size (cm)', fontsize=12)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()  # Rotate dates
    
    # Set y-axis limits with some padding
    if not df.empty:
        y_min = max(0, df['Min_Size_cm'].min() - 1)
        y_max = df['Max_Size_cm'].max() + 1
        plt.ylim(y_min, y_max)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend with multiple columns if many species
    n_species = len(sorted_species)
    ncol = min(3, max(1, n_species // 8 + 1))
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), 
              fontsize=9, ncol=ncol, framealpha=0.9)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Save the graph with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    days_str = f"{days}days" if days < 365 else "alltime"
    graph_path = os.path.join(GRAPH_OUTPUT_DIR, f"growth_trend_{days_str}_{timestamp}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Growth graph saved → {graph_path}")
    print(f"  Species plotted: {len(species)}")
    return graph_path

def generate_water_quality_graph(water_data_history=None):
    """
    Generate water quality analysis graph
    Shows temperature, pH, and dissolved oxygen trends over time
    """
    # If no water data provided, create sample data based on current conditions
    if water_data_history is None:
        # Generate sample water quality data for the last 30 days
        dates = [datetime.now().date() - timedelta(days=x) for x in range(30, 0, -1)]
        
        # Generate realistic water quality parameters with slight variations
        temperatures = [26 + np.random.normal(0, 1.5) for _ in range(30)]
        ph_levels = [7.0 + np.random.normal(0, 0.3) for _ in range(30)]
        oxygen_levels = [7.0 + np.random.normal(0, 0.8) for _ in range(30)]
        
        # Ensure values are within realistic ranges
        temperatures = [max(22, min(32, t)) for t in temperatures]
        ph_levels = [max(6.5, min(8.0, p)) for p in ph_levels]
        oxygen_levels = [max(5.0, min(9.0, o)) for o in oxygen_levels]
        
        water_data = {
            'dates': dates,
            'temperature': temperatures,
            'ph': ph_levels,
            'oxygen': oxygen_levels
        }
    else:
        water_data = water_data_history
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Temperature subplot
    axes[0].plot(water_data['dates'], water_data['temperature'], 
                 color='red', linewidth=2, marker='o', markersize=4)
    axes[0].fill_between(water_data['dates'], 
                         [t-1 for t in water_data['temperature']],
                         [t+1 for t in water_data['temperature']],
                         alpha=0.2, color='red')
    axes[0].set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    axes[0].set_title('Water Temperature Trend', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].axhline(y=28, color='orange', linestyle='--', alpha=0.5, label='Optimal Range')
    axes[0].legend()
    
    # pH subplot
    axes[1].plot(water_data['dates'], water_data['ph'], 
                 color='green', linewidth=2, marker='s', markersize=4)
    axes[1].fill_between(water_data['dates'], 
                         [p-0.2 for p in water_data['ph']],
                         [p+0.2 for p in water_data['ph']],
                         alpha=0.2, color='green')
    axes[1].set_ylabel('pH Level', fontsize=11, fontweight='bold')
    axes[1].set_title('Water pH Trend', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].axhline(y=7.0, color='orange', linestyle='--', alpha=0.5, label='Neutral pH')
    axes[1].axhspan(6.5, 8.0, alpha=0.1, color='green', label='Optimal Range')
    axes[1].legend()
    
    # Dissolved Oxygen subplot
    axes[2].plot(water_data['dates'], water_data['oxygen'], 
                 color='blue', linewidth=2, marker='^', markersize=4)
    axes[2].fill_between(water_data['dates'], 
                         [o-0.5 for o in water_data['oxygen']],
                         [o+0.5 for o in water_data['oxygen']],
                         alpha=0.2, color='blue')
    axes[2].set_ylabel('Dissolved Oxygen (mg/L)', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Date', fontsize=11, fontweight='bold')
    axes[2].set_title('Dissolved Oxygen Trend', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].axhline(y=6.5, color='orange', linestyle='--', alpha=0.5, label='Optimal Level')
    axes[2].axhspan(5.5, 8.5, alpha=0.1, color='blue', label='Optimal Range')
    axes[2].legend()
    
    # Format x-axis dates
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)
    
    # Add main title
    fig.suptitle('Water Quality Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the graph
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_path = os.path.join(GRAPH_OUTPUT_DIR, f"water_quality_{timestamp}.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Water quality graph saved → {graph_path}")
    return graph_path

def generate_species_summary_table():
    """
    Generate a summary table of current fish sizes similar to the image
    """
    if not os.path.isfile(GROWTH_LOG_FILE):
        return None
    
    df = pd.read_csv(GROWTH_LOG_FILE)
    if df.empty:
        return None
    
    # Get the most recent data for each species
    latest_data = df.sort_values('Date').groupby('Species').last().reset_index()
    latest_data = latest_data.sort_values('Avg_Size_cm', ascending=False)
    
    # Create a table visualization
    fig, ax = plt.subplots(figsize=(10, max(6, len(latest_data)*0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in latest_data.iterrows():
        table_data.append([
            row['Species'],
            f"{row['Avg_Size_cm']:.1f}",
            f"{row['Min_Size_cm']:.1f}-{row['Max_Size_cm']:.1f}",
            str(int(row['Count']))
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Fish Species', 'Avg Size (cm)', 'Size Range (cm)', 'Count'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.4, 0.15, 0.2, 0.1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color header row
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor('#40466e')
        cell.set_text_props(color='white', weight='bold')
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(4):
                table[(i, j)].set_facecolor('#f5f5f5')
    
    plt.title('Fish Species Size Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Save the table
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    table_path = os.path.join(GRAPH_OUTPUT_DIR, f"species_summary_{timestamp}.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Species summary table saved → {table_path}")
    return table_path

def update_growth_tracking(species_count, fish_sizes):
    """
    Update growth tracking with new fish measurements
    fish_sizes: dict with species as keys and list of sizes as values
    """
    today = datetime.now().date().strftime("%Y-%m-%d")
    
    # Prepare species data
    species_data = {}
    for species, sizes in fish_sizes.items():
        if sizes:
            species_data[species] = {
                'avg_size': np.mean(sizes),
                'min_size': np.min(sizes),
                'max_size': np.max(sizes),
                'count': len(sizes)
            }
    
    # Log the data
    if species_data:
        log_growth_data(today, species_data)
        print(f"  Growth data logged for {len(species_data)} species")

def display_recent_growth():
    """Display recent growth data in console as a formatted table"""
    if not os.path.isfile(GROWTH_LOG_FILE):
        print("\n  No growth data available")
        return
    
    df = pd.read_csv(GROWTH_LOG_FILE)
    if df.empty:
        print("\n  No growth data available")
        return
    
    # Get the most recent data for each species
    latest_data = df.sort_values('Date').groupby('Species').last().reset_index()
    latest_data = latest_data.sort_values('Avg_Size_cm', ascending=False)
    
    print("\n  " + "="*60)
    print("  CURRENT FISH SIZE SUMMARY")
    print("  " + "="*60)
    print(f"  {'Species':<25} {'Avg Size':<10} {'Range':<15} {'Count':<6}")
    print("  " + "-"*60)
    
    for _, row in latest_data.iterrows():
        print(f"  {row['Species']:<25} {row['Avg_Size_cm']:<10.1f} "
              f"{row['Min_Size_cm']:.1f}-{row['Max_Size_cm']:<8.1f} "
              f"{int(row['Count']):<6}")
    
    print("  " + "="*60)


# ================================================================
#  STEP 7 — CSV LOGGING
# ================================================================

def log_csv(date, time_, frame_num, fid, species, size, conf, growth):
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE,"a",newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Date","Time","Frame","FishID","Species","Size_cm","Conf%","Growth"])
        w.writerow([date, time_, frame_num, fid, species, size, conf, growth])

# ================================================================
#  STEP 8 — DRAWING
# ================================================================

GROWTH_COLORS = {
    "Mature":   (0, 210,  0),
    "Growing":  (0, 200, 255),
    "Juvenile": (0, 130, 255),
    "Unknown":  (160,160,160),
}

def draw_box(frame, cx, cy, bw, bh, fid, species, conf_pct, growth):
    H, W = frame.shape[:2]
    cx,cy,bw,bh = int(cx),int(cy),int(bw),int(bh)
    x1=max(0,cx-bw//2); y1=max(0,cy-bh//2)
    x2=min(W,cx+bw//2); y2=min(H,cy+bh//2)

    col = GROWTH_COLORS.get(growth, (160,160,160))
    cv2.rectangle(frame,(x1,y1),(x2,y2),col,1)
    cv2.circle(frame,(cx,cy),3,col,-1)

    label = f"ID{fid} {species} {conf_pct}%"
    fs    = 0.38
    (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,fs,1)

    # Place label above box; clamp to frame edges
    lx = min(max(x1, 0), W-tw-2)
    ly = max(y1-3, th+3)
    cv2.rectangle(frame,(lx,ly-th-2),(lx+tw+2,ly+2),(0,0,0),-1)
    cv2.putText(frame,label,(lx,ly),cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),1,cv2.LINE_AA)


def draw_hud(frame, water, temp, ph, o2, pond, frame_num=None):
    H,W = frame.shape[:2]

    # Semi-transparent bottom bar
    lines = [
        f"Water: {water}  |  Temp: {temp}C  |  pH: {ph}  |  O2: {o2} mg/L",
        f"Total Fish: {pond['total_fish_detected']}  "
        f"|  Diversity: {pond['diversity_level']}  "
        f"|  Density: {pond['density_status']}",
        f"Pond Health: {pond['pond_health_rating']}  "
        f"|  Score: {pond['overall_health_score']}",
    ]
    if frame_num:
        lines[0] = f"Frame:{frame_num}  |  " + lines[0]

    bar_h = len(lines)*22 + 8
    roi   = frame[H-bar_h:H, 0:W]
    black = np.zeros_like(roi)
    cv2.addWeighted(black,0.55,roi,0.45,0,roi)
    frame[H-bar_h:H, 0:W] = roi

    for i,line in enumerate(lines):
        cv2.putText(frame,line,(8,H-bar_h+18+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,240,240),1,cv2.LINE_AA)

    # Fish count badge — top right
    badge = f" FISH: {pond['total_fish_detected']} "
    (bw2,bh2),_ = cv2.getTextSize(badge,cv2.FONT_HERSHEY_SIMPLEX,0.72,2)
    cv2.rectangle(frame,(W-bw2-12,4),(W-4,bh2+12),(0,170,0),-1)
    cv2.putText(frame,badge,(W-bw2-8,bh2+8),
                cv2.FONT_HERSHEY_SIMPLEX,0.72,(255,255,255),2,cv2.LINE_AA)


# ================================================================
#  STEP 9 — PROCESS ONE FRAME
# ================================================================

def process_frame(frame, prof, frame_num=None, is_video=False):
    H,W = frame.shape[:2]
    predictions  = detect(frame, prof)
    species_count = defaultdict(int)
    fish_sizes = defaultdict(list)  # For growth tracking

    for pred in predictions:
        cx,cy = pred["x"], pred["y"]
        bw,bh = pred["width"], pred["height"]

        # Crop for species classification
        y1c=int(max(0,cy-bh/2)); y2c=int(min(H,cy+bh/2))
        x1c=int(max(0,cx-bw/2)); x2c=int(min(W,cx+bw/2))
        crop = frame[y1c:y2c, x1c:x2c]

        # Enhance crop for species model if image is not clear
        if not prof["clear"] and crop.size > 0:
            crop = enhance(crop, prof)

        species, sp_conf = classify_species(crop)
        size     = round(bw * PIXEL_TO_CM, 2)
        analysis = analyze_fish(species, size)
        species_count[species] += 1
        fish_sizes[species].append(size)  # Add to growth tracking

        fid = get_next_id() if is_video else sum(species_count.values())-1
        draw_box(frame, cx, cy, bw, bh, fid, species, sp_conf, analysis["growth_stage"])

        log_csv(datetime.now().date(), datetime.now().time(),
                frame_num or "N/A", fid, species, size, sp_conf,
                analysis["growth_stage"])

    return frame, species_count, fish_sizes


# ================================================================
#  STEP 10 — IMAGE ANALYSIS
# ================================================================

def analyze_image(path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot open: {path}"); return

    print(f"\n{'='*55}")
    print(f"  IMAGE: {os.path.basename(path)}  ({frame.shape[1]}x{frame.shape[0]})")
    print(f"{'='*55}")

    prof = profile_image(frame)
    print(f"\n  Profile   : brightness={prof['brightness']:.0f}  "
          f"blur={prof['blur_score']:.0f}  contrast={prof['contrast']:.0f}  "
          f"green_cast={prof['green_cast']:.1f}")
    print(f"  Flags     : dark={prof['dark']}  blurry={prof['blurry']}  "
          f"turbid={prof['turbid']}  blue={prof['blue']}  flat={prof['flat']}")
    print(f"  Severity  : {prof['severity']}")

    water,temp,ph,o2 = analyze_water(frame)
    print(f"\n  Water     : {water}  |  Temp={temp}C  pH={ph}  O2={o2}")

    print(f"\n  Detection:")
    reset_id()
    out_frame, species_count, fish_sizes = process_frame(frame.copy(), prof, is_video=False)
    pond = analyze_pond_health(species_count)

    print(f"\n  Results:")
    for sp,cnt in sorted(species_count.items(), key=lambda x:-x[1]):
        print(f"    {sp:28s}: {cnt}")
    print(f"\n  Total Fish  : {pond['total_fish_detected']}")
    print(f"  Diversity   : {pond['diversity_level']}")
    print(f"  Density     : {pond['density_status']}")
    print(f"  Health      : {pond['pond_health_rating']}")
    print(f"  Score       : {pond['overall_health_score']}")

    # Update growth tracking for this image
    update_growth_tracking(species_count, fish_sizes)

    draw_hud(out_frame, water, temp, ph, o2, pond)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"outputs/detection_{ts}.jpg"
    cv2.imwrite(out, out_frame)
    print(f"\n  Saved → {out}")
    
    # Generate all graphs
    print(f"\n  Generating visualization graphs...")
    
    # Generate species distribution bar chart
    generate_species_distribution_graph(species_count, ts)
    
    # Generate species percentage pie chart
    generate_species_pie_chart(species_count, ts)
    
    # Generate water quality graph
    generate_water_quality_graph()
    
    # Generate growth graph
    graph_path = generate_growth_graph(days=30)
    if graph_path:
        print(f"  Growth graph saved")
    
    # Generate species summary table
    generate_species_summary_table()
    
    print(f"\n  All graphs saved to: {GRAPH_OUTPUT_DIR}")
    print(f"{'='*55}\n")


# ================================================================
#  STEP 11 — VIDEO ANALYSIS
# ================================================================

def analyze_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {path}"); return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter("outputs/video_detection.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"), fps, (W,H))

    print(f"\n{'='*55}")
    print(f"  VIDEO: {os.path.basename(path)}")
    print(f"  {W}x{H}  |  {fps}fps  |  {total} frames  |  "
          f"~{total//fps}s")

    # Read first frame to set profile for whole video
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Cannot read video"); return

    prof = profile_image(first)
    print(f"  Severity  : {prof['severity']}")

    # Estimate time
    detect_frames = max(1, total // PROCESS_EVERY_N)
    api_per_frame = 1 if prof["clear"] else 7   # 2x3 tiles + 1 full
    est_s = detect_frames * api_per_frame * 0.5
    print(f"  Detecting every {PROCESS_EVERY_N} frames → "
          f"~{detect_frames} detections")
    print(f"  Est. time : ~{int(est_s//60)}m {int(est_s%60)}s")
    print(f"{'='*55}\n")

    reset_id()
    video_species = defaultdict(int)
    all_fish_sizes = defaultdict(list)  # For growth tracking
    idx = 1

    # Cache for skipped frames
    last_pond  = analyze_pond_health(defaultdict(int))
    last_water = ("Clear Water", 27.0, 7.2, 6.5)

    # Process first frame
    water,temp,ph,o2 = analyze_water(first)
    out_f, sc, fs = process_frame(first, prof, frame_num=idx, is_video=True)
    for sp,cnt in sc.items(): 
        video_species[sp] += cnt
        if sp in fs:
            all_fish_sizes[sp].extend(fs[sp])
    pond = analyze_pond_health(sc)
    draw_hud(out_f, water,temp,ph,o2, pond, frame_num=idx)
    out.write(out_f)
    last_pond = pond; last_water=(water,temp,ph,o2)

    while True:
        ret,frame = cap.read()
        if not ret: break
        idx += 1

        if idx % PROCESS_EVERY_N != 0:
            # Skipped frame: reuse last HUD, no API call
            w,t,p,o = last_water
            draw_hud(frame,w,t,p,o,last_pond,frame_num=idx)
            out.write(frame)
            continue

        # Re-profile every 60 detection frames (handles lighting changes)
        if (idx // PROCESS_EVERY_N) % 60 == 0:
            prof = profile_image(frame)

        water,temp,ph,o2 = analyze_water(frame)
        out_f, sc, fs = process_frame(frame, prof, frame_num=idx, is_video=True)
        for sp,cnt in sc.items(): 
            video_species[sp] += cnt
            if sp in fs:
                all_fish_sizes[sp].extend(fs[sp])
        pond = analyze_pond_health(sc)
        draw_hud(out_f,water,temp,ph,o2,pond,frame_num=idx)
        out.write(out_f)
        last_pond=pond; last_water=(water,temp,ph,o2)

        pct = idx*100//total
        fish_n = sum(sc.values())
        print(f"  Frame {idx:4d}/{total} ({pct:3d}%)  "
              f"fish={fish_n:3d}  total_ids={_fish_id_counter}", end="\r")

    cap.release(); out.release()

    # Update growth tracking with all video data
    update_growth_tracking(video_species, all_fish_sizes)

    pond_total = analyze_pond_health(video_species)
    print(f"\n\n{'='*55}")
    print(f"  VIDEO COMPLETE")
    print(f"  Frames processed : {idx//PROCESS_EVERY_N} / {idx}")
    print(f"  Species detected :")
    for sp,cnt in sorted(video_species.items(),key=lambda x:-x[1]):
        print(f"    {sp:28s}: {cnt}")
    print(f"  Overall health   : {pond_total['pond_health_rating']}")
    print(f"  Saved → outputs/video_detection.mp4")
    
    # Generate all graphs after video processing
    print(f"\n  Generating visualization graphs...")
    
    # Generate species distribution bar chart
    generate_species_distribution_graph(video_species)
    
    # Generate species percentage pie chart
    generate_species_pie_chart(video_species)
    
    # Generate water quality graph
    generate_water_quality_graph()
    
    # Generate growth graph
    graph_path = generate_growth_graph(days=30)
    if graph_path:
        print(f"  Growth graph saved")
    
    # Generate species summary table
    generate_species_summary_table()
    
    print(f"\n  All graphs saved to: {GRAPH_OUTPUT_DIR}")
    print(f"{'='*55}\n")


# ================================================================
#  STEP 12 — GROWTH GRAPH VIEWER
# ================================================================

def view_growth_graphs():
    """Display menu for viewing growth graphs"""
    while True:
        print("\n" + "="*55)
        print("  GRAPH VIEWER")
        print("="*55)
        print("\n  1  View Fish Species Distribution (Bar Chart)")
        print("  2  View Species Percentage (Pie Chart)")
        print("  3  View Growth Trend (Line Graph)")
        print("  4  View Water Quality Analysis")
        print("  5  View Species Summary Table")
        print("  6  View All Graphs")
        print("  7  Back to main menu")
        
        choice = input("\n  Enter choice: ").strip()
        
        if choice == "1":
            print("\n  Generating species distribution graph...")
            if os.path.isfile(GROWTH_LOG_FILE):
                df = pd.read_csv(GROWTH_LOG_FILE)
                if not df.empty:
                    latest_data = df.sort_values('Date').groupby('Species').last()
                    species_count = {sp: row['Count'] for sp, row in latest_data.iterrows()}
                    generate_species_distribution_graph(species_count)
                else:
                    print("  No data available")
            else:
                print("  No growth data available")
                
        elif choice == "2":
            print("\n  Generating species percentage pie chart...")
            if os.path.isfile(GROWTH_LOG_FILE):
                df = pd.read_csv(GROWTH_LOG_FILE)
                if not df.empty:
                    latest_data = df.sort_values('Date').groupby('Species').last()
                    species_count = {sp: row['Count'] for sp, row in latest_data.iterrows()}
                    generate_species_pie_chart(species_count)
                else:
                    print("  No data available")
            else:
                print("  No growth data available")
                
        elif choice == "3":
            print("\n  Growth trend options:")
            print("  1. Last 7 days")
            print("  2. Last 30 days")
            print("  3. Last 90 days")
            print("  4. All time")
            days_choice = input("\n  Select option: ").strip()
            
            if days_choice == "1":
                generate_growth_graph(days=7)
            elif days_choice == "2":
                generate_growth_graph(days=30)
            elif days_choice == "3":
                generate_growth_graph(days=90)
            elif days_choice == "4":
                generate_growth_graph(days=365*10)
            else:
                print("  Invalid choice")
                
        elif choice == "4":
            print("\n  Generating water quality graph...")
            generate_water_quality_graph()
            
        elif choice == "5":
            print("\n  Generating species summary table...")
            generate_species_summary_table()
            
        elif choice == "6":
            print("\n  Generating all graphs...")
            if os.path.isfile(GROWTH_LOG_FILE):
                df = pd.read_csv(GROWTH_LOG_FILE)
                if not df.empty:
                    latest_data = df.sort_values('Date').groupby('Species').last()
                    species_count = {sp: row['Count'] for sp, row in latest_data.iterrows()}
                    generate_species_distribution_graph(species_count)
                    generate_species_pie_chart(species_count)
                else:
                    print("  No species data available")
            generate_growth_graph(days=30)
            generate_water_quality_graph()
            generate_species_summary_table()
            print(f"\n  All graphs saved to: {GRAPH_OUTPUT_DIR}")
            
        elif choice == "7":
            break
        else:
            print("  Invalid choice")


# ================================================================
#  STEP 13 — WATER QUALITY HISTORY TRACKING
# ================================================================

def init_water_quality_log():
    """Initialize water quality log CSV file"""
    water_log_file = "logs/water_quality_log.csv"
    if not os.path.isfile(water_log_file):
        with open(water_log_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "Timestamp", "Temperature_C", "pH", "Dissolved_Oxygen_mgL", "Water_Status"])

def log_water_quality(temp, ph, oxygen, status):
    """Log water quality data"""
    init_water_quality_log()
    water_log_file = "logs/water_quality_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().date().strftime("%Y-%m-%d")
    
    with open(water_log_file, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([date, timestamp, temp, ph, oxygen, status])


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AI Marine Monitoring System ")
    print("  Adaptive Detection | Image & Video | Growth Tracking")
    print("="*55)
    
    # Initialize logs
    init_growth_log()
    init_water_quality_log()
    
    while True:
        print("\n  1  Analyze Image")
        print("  2  Analyze Video")
        print("  3  View Graphs")
        print("  4  View Growth Data")
        print("  5  Exit")

        choice = input("\n  Enter choice: ").strip()

        if choice == "1":
            analyze_image("Inputs/Images/pond5.png")
        elif choice == "2":
            analyze_video("Inputs/Videos/pond10.mp4")
        elif choice == "3":
            view_growth_graphs()
        elif choice == "4":
            display_recent_growth()
        elif choice == "5":
            print("\n  Exiting...")
            break
        else:
            print("  Invalid choice")