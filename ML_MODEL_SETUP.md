# ML Model Setup Guide

## Current Issue
Your EdgeTPU model (`model_edgetpu.tflite`) requires the `pycoral` library and EdgeTPU hardware, which is not available on Railway or most systems.

## Solution: Export Regular TensorFlow Lite Model

### Steps to Export from Teachable Machine:

1. **Go to your Teachable Machine project**: https://teachablemachine.withgoogle.com
2. **Click "Export Model"**
3. **Select "TensorFlow Lite"** (NOT "Edge TPU")
4. **Download the model file**
5. **Save as**: `model_unquant.tflite` in the project root
6. **Export labels**: Download labels.txt and save in project root

### Expected Files:
- `model_unquant.tflite` - Regular TensorFlow Lite model (will work on Railway)
- `labels.txt` - Label file with format:
  ```
  0 GOOD
  1 NOT_GOOD
  ```

## Model Priority (Current Implementation)

The app will try models in this order:

1. **EdgeTPU Model** (if `pycoral` installed):
   - `model_edgetpu.tflite`
   - `edgetpu_labels.txt`
   
2. **Regular TensorFlow Lite Model** (fallback):
   - `model_unquant.tflite`
   - `labels.txt`

## Testing

After adding the regular model:
1. Load images in the dashboard
2. Enable "ML Classification Filter"
3. Click "Refresh Filters"
4. Check console logs for ML classification results
5. Images should be classified as "Good" or "Not_Good"

## Troubleshooting

### All images show "Unknown":
- ✅ **Fixed**: Class name normalization now handles "GOOD"/"NOT_GOOD" → "Good"/"Not_Good"
- Check console logs to see if model is loading
- Verify model file exists and is accessible

### Model not loading:
- Check console for error messages
- Verify TensorFlow is installed: `pip install tensorflow`
- Verify PIL/Pillow is installed: `pip install pillow`







