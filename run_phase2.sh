#!/bin/bash

# Default values
# Calculate default week (current week) if not provided
# Note: This depends on the system's date command implementation. 
# Adjust format if necessary (e.g. %V for ISO week number).
WEEK=$(date +%Y-W%V)
MARKETPLACE="US"
STORAGE="output/weekly_report"

#active the venv 
source myenv/bin/activate

# Initialize empty scenes array
SCENES=()
SCENES_FILE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --week) WEEK="$2"; shift ;;
        --scene) SCENES+=("$2"); shift ;;
        --scenes-file) SCENES_FILE="$2"; shift ;;
        --marketplace) MARKETPLACE="$2"; shift ;;
        --storage) STORAGE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Load scenes from file if specified
if [[ -n "$SCENES_FILE" ]]; then
    if [[ ! -f "$SCENES_FILE" ]]; then
        echo "Error: Scenes file not found: $SCENES_FILE"
        exit 1
    fi
    echo "Loading scenes from file: $SCENES_FILE"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        SCENES+=("$line")
    done < "$SCENES_FILE"
fi

# If no scenes specified, show usage
if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "Usage: $0 --scene <scene_name> [--scene <scene_name2>...] [OPTIONS]"
    echo "   OR: $0 --scenes-file <file_path> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --week <YYYY-Www>        Week to process (default: current week)"
    echo "  --scene <name>           Scene to process (can be specified multiple times)"
    echo "  --scenes-file <path>     File containing scene names (one per line)"
    echo "  --marketplace <code>     Marketplace code (default: US)"
    echo "  --storage <path>         Storage path (default: output/weekly_report)"
    echo ""
    echo "Example:"
    echo "  $0 --week 2025-W45 --scene 浴室袋 --scene 收纳袋"
    echo "  $0 --week 2025-W45 --scenes-file scenes.txt"
    exit 1
fi

echo "=================================================="
echo "Running Phase 2 Pipeline"
echo "Week: $WEEK"
echo "Marketplace: $MARKETPLACE"
echo "Storage: $STORAGE"
echo "Scenes: ${#SCENES[@]} scenes to process"
echo "=================================================="

# Step A: Global Diff Generation (Run once per week)
echo ""
echo "[Step A] Generating ASIN Week Diffs (Global)..."
echo "Command: python -m scpc.jobs.asin_week_diff_job --week $WEEK"
python -m scpc.jobs.asin_week_diff_job --week "$WEEK"

if [ $? -ne 0 ]; then
    echo "Error: Step A (ASIN Week Diff) failed. Exiting."
    exit 1
fi

# Loop through scenes for B, C, D
for SCENE in "${SCENES[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Processing Scene: $SCENE"
    echo "--------------------------------------------------"

    # Step B: Traffic JSON
    echo "[Step B] Generating Traffic JSON..."
    python -m scpc.jobs.generate_scene_traffic_json \
        --week "$WEEK" \
        --scene_tag "$SCENE" \
        --marketplace "$MARKETPLACE" \
        --storage "$STORAGE"
    
    if [ $? -ne 0 ]; then echo "Warning: Step B failed for $SCENE"; continue; fi

    # Step C: Traffic Report
    echo "[Step C] Generating Traffic Report (Markdown)..."
    python -m scpc.jobs.generate_scene_traffic_report \
        --week "$WEEK" \
        --scene_tag "$SCENE" \
        --marketplace "$MARKETPLACE" \
        --storage "$STORAGE"

    if [ $? -ne 0 ]; then echo "Warning: Step C failed for $SCENE"; continue; fi

    # Step D1: Weekly Scene JSON
    echo "[Step D1] Generating Weekly Scene JSON (Modules 1-5)..."
    python -m scpc.jobs.generate_weekly_scene_json \
        --week "$WEEK" \
        --scene_tag "$SCENE" \
        --marketplace "$MARKETPLACE" \
        --storage "$STORAGE"

    if [ $? -ne 0 ]; then echo "Warning: Step D1 failed for $SCENE"; continue; fi

    # Step D2: Weekly Scene Report
    echo "[Step D2] Generating Weekly Scene Report (Markdown)..."
    python -m scpc.jobs.generate_weekly_scene_report \
        --week "$WEEK" \
        --scene_tag "$SCENE" \
        --marketplace "$MARKETPLACE" \
        --storage "$STORAGE"

    if [ $? -ne 0 ]; then echo "Warning: Step D2 failed for $SCENE"; continue; fi
    
    echo "Completed processing for $SCENE"
done

echo ""
echo "=================================================="
echo "Phase 2 Pipeline Completed."
echo "=================================================="

#deactivate the venv
deactivate
