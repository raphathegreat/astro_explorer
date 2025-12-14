# 🍓 Raspberry Pi / AstroPi ML Model Setup Guide

## Overview

The application **automatically detects** whether you have EdgeTPU hardware and uses the appropriate model:

1. **EdgeTPU hardware available** → Uses `model_edgetpu.tflite` (fast, hardware-accelerated)
2. **No EdgeTPU hardware** → Falls back to `model_unquant.tflite` (runs on CPU)

## Scenario 1: With Coral USB Accelerator (EdgeTPU Hardware) ✅

If you have a **Coral USB Accelerator** attached to your Raspberry Pi:

### Setup Steps:

1. **Install pycoral library:**
   ```bash
   pip install pycoral
   ```

2. **Verify EdgeTPU is detected:**
   ```bash
   python3 -c "from pycoral.utils import edgetpu; print('EdgeTPU devices:', edgetpu.list_edge_tpus())"
   ```

3. **Place your model files:**
   - `model_edgetpu.tflite` → Project root
   - `edgetpu_labels.txt` → Project root

4. **Run the app:**
   ```bash
   python3 iss_speed_html_dashboard_v2_clean.py
   ```

5. **Expected output:**
   ```
   ✅ EdgeTPU device found, using hardware acceleration
   ✅ EdgeTPU ML model loaded successfully. Classes: ['GOOD', 'NOT_GOOD']
   ```

### Benefits:
- ⚡ **Much faster** inference (hardware acceleration)
- 🔋 **Lower power consumption**
- 🎯 **Optimized for real-time processing**

---

## Scenario 2: Without EdgeTPU Hardware (CPU Only) ✅

If you **don't have** a Coral USB Accelerator:

### Setup Steps:

1. **Export regular TensorFlow Lite model from Teachable Machine:**
   - Go to https://teachablemachine.withgoogle.com
   - Click "Export Model"
   - Select **"TensorFlow Lite"** (NOT "Edge TPU")
   - Download the model

2. **Place model files in project root:**
   - `model_unquant.tflite` → Project root
   - `labels.txt` → Project root
   - Format of `labels.txt`:
     ```
     0 GOOD
     1 NOT_GOOD
     ```

3. **Run the app:**
   ```bash
   python3 iss_speed_html_dashboard_v2_clean.py
   ```

4. **Expected output:**
   ```
   ⚠️ Failed to load EdgeTPU model... (expected - no EdgeTPU hardware)
   ✅ Regular ML model loaded successfully. Classes: ['GOOD', 'NOT_GOOD']
   ```

### Performance:
- ⏱️ **Slower** than EdgeTPU (runs on Raspberry Pi CPU)
- ✅ **Still works** - Raspberry Pi 4+ handles TensorFlow Lite well
- 💾 **Lower memory usage** than EdgeTPU model

---

## Automatic Fallback Logic

The code **automatically** handles both scenarios:

```python
# 1. Try EdgeTPU model first (if available)
if model_edgetpu.tflite exists and pycoral installed:
    try:
        load EdgeTPU model with hardware acceleration
    except:
        fall back to regular model

# 2. Try regular TensorFlow Lite model
if model_unquant.tflite exists:
    load regular model (CPU)
```

**No configuration needed** - the app detects what's available!

---

## Troubleshooting

### "EdgeTPU device not found"
- ✅ **Normal** if you don't have Coral USB Accelerator
- ✅ App will automatically use regular TFLite model

### "ML model files not found"
- Check that model files are in the **project root directory**
- Verify file names match exactly:
  - `model_edgetpu.tflite` (for EdgeTPU)
  - `model_unquant.tflite` (for CPU)
  - `edgetpu_labels.txt` or `labels.txt`

### "All images show Unknown"
- ✅ **Fixed in latest code** - class name normalization now handles "GOOD"/"NOT_GOOD"
- Check console logs to see what classifications are being made
- Verify labels file format is correct

### Performance Issues on Raspberry Pi
- **EdgeTPU**: Should be fast (hardware acceleration)
- **CPU**: Slower but should still work
  - Consider reducing image processing batch size
  - Ensure adequate power supply (2.5A+ recommended for Pi 4)
  - Close other applications to free up resources

---

## AstroPi ISS Hardware ✅

**Good news!** The AstroPi computers on the International Space Station **ARE equipped with Coral USB Accelerators**!

This means:
- ✅ Your `model_edgetpu.tflite` will work perfectly on the ISS AstroPi
- ✅ Hardware acceleration available for fast ML inference
- ✅ Code automatically detects and uses EdgeTPU hardware

**For ISS deployment:**
1. Ensure `model_edgetpu.tflite` and `edgetpu_labels.txt` are included
2. Ensure `pycoral` is in `requirements.txt` or installed on AstroPi
3. Code will automatically use EdgeTPU when available

## Recommended Setup

**Best Performance (ISS AstroPi):**
- ✅ Coral USB Accelerator (already installed on ISS)
- ✅ EdgeTPU model (`model_edgetpu.tflite`)
- ✅ Install `pycoral` library

**Local Development / Testing:**
- Raspberry Pi 4 (any model)
- If you have Coral USB Accelerator → Use EdgeTPU model
- If not → Use regular TensorFlow Lite model (`model_unquant.tflite`)

---

## Quick Test

After setup, test the model:

```bash
# In the dashboard:
1. Load images from a photos folder
2. Enable "ML Classification Filter" in Section 3
3. Click "Refresh Filters"
4. Check Section 6 graph - should show Good (green) / Not_Good (red)
```

If you see "Unknown" for all images:
- Check console logs for ML classification output
- Verify model files are loaded correctly
- Ensure labels file matches model classes

