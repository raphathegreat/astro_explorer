# ISS Speed Analysis Dashboard v2 - Clean Version

A clean, maintainable version of the ISS Speed Analysis Dashboard with improved code organization, documentation, and maintainability.

## 📋 Development Rules

**Important:** See [DEVELOPMENT_RULES.md](DEVELOPMENT_RULES.md) for complete development guidelines and rules.

Key Rules:
- **Rule 1**: Test-Driven Development (TDD) - Create tests before code
- **Rule 2**: Version Archiving - Archive working versions  
- **Rule 3**: Test Pack - Use realistic_test.py as the designated test pack
- **Rule 4**: Cache Maintenance - Update caching for Section 2 changes
- **Rule 5**: Issue Investigation Protocol - Always check logs when user reports issues
- **Rule 6**: GSD Configuration Default Behavior - Disabled by default
- **Rule 7**: Filter Application Logic - Only send enabled filters

## 📁 File Structure

```
├── iss_speed_html_dashboard_v2_clean.py     # Clean backend (Flask application)
├── templates/
│   └── dashboard_v2_clean.html             # Clean frontend (HTML/CSS/JavaScript)
├── iss_speed_html_dashboard_v2_working_backup.py  # Backup of working version
├── templates/
│   └── dashboard_v2_working_backup.html    # Backup of working frontend
└── README_CLEAN_VERSION.md                 # This documentation
```

## 🚀 Features

### Core Functionality
- **Image Processing**: Process ISS image sequences using SIFT, ORB, and cchan083 algorithms
- **Speed Calculation**: Calculate ISS speed using pixel distances and GPS data
- **Real-time Visualization**: Interactive plots showing speed distributions and pair analysis
- **Cache Management**: Efficient caching system for processed data
- **Algorithm Comparison**: Compare different computer vision algorithms

### Data Processing
- **Multiple Algorithms**: Support for ORB, SIFT, and custom cchan083 algorithms
- **Feature Matching**: Advanced feature detection and matching with FLANN and RANSAC
- **GPS Integration**: Extract and use GPS coordinates for accurate speed calculation
- **Contrast Enhancement**: CLAHE and other enhancement methods for better feature detection

### User Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Filters**: Filter data by speed percentiles, standard deviation, cloudiness
- **Real-time Updates**: Live progress tracking and statistics updates
- **Modern UI**: Clean, professional interface with smooth animations

## 🛠️ Installation & Setup

### Prerequisites
```bash
pip install flask numpy opencv-python exif matplotlib tensorflow
```

### Running the Application
```bash
python iss_speed_html_dashboard_v2_clean.py
```

The dashboard will be available at: `http://localhost:5002`

## 📊 Usage Guide

### 1. Data Processing
1. **Select Photo Folder**: Choose a folder containing ISS images (photos-* format)
2. **Set Range**: Specify start and end indices for image pairs
3. **Choose Algorithm**: Select ORB, SIFT, or both
4. **Configure Options**: Enable FLANN matching, RANSAC homography, contrast enhancement
5. **Load Data**: Click "Load Data" to start processing

### 2. Viewing Results
- **Section 3 - Statistics**: View mean, median, mode, and total match counts
- **Section 5 - Speed Distribution**: Histogram showing speed distribution of all keypoints
- **Section 6 - Pair Analysis**: Scatter plot showing mean and median speeds per image pair

### 3. Data Filtering
- **Speed Filtering**: Filter by keypoint percentiles or standard deviation
- **Cloudiness Filtering**: Include/exclude partly or mostly cloudy images
- **Real-time Updates**: Filters apply immediately to all visualizations

### 4. Algorithm Comparison
1. **Select Algorithms**: Choose which algorithms to compare
2. **Run Comparison**: Click "Run Comparison" to process with multiple algorithms
3. **View Results**: Compare accuracy, success rates, and average speeds

## 🏗️ Code Architecture

### Backend (Python/Flask)
```
├── Configuration & Global State
├── Utility Functions (Logging, Cache Management)
├── Image Processing (Feature Detection, GPS Extraction)
├── Statistics & Filtering
├── Flask Routes (API Endpoints)
├── Algorithm Comparison
└── Main Application
```

### Frontend (HTML/CSS/JavaScript)
```
├── Global Styles (Responsive Design)
├── Section Styles (Cards, Forms, Plots)
├── Component Styles (Buttons, Progress, Messages)
├── Utility Functions (Logging, Message Display)
├── Data Loading & Processing
├── Statistics Display
├── Plot Creation (Plotly.js)
├── Filter Management
├── Cache Management
├── Algorithm Comparison
└── Event Listeners
```

## 🔧 Key Improvements in Clean Version

### Code Organization
- **Modular Structure**: Clear separation of concerns with well-defined sections
- **Comprehensive Documentation**: Detailed comments and docstrings throughout
- **Consistent Naming**: Standardized variable and function naming conventions
- **Error Handling**: Robust error handling with user-friendly messages

### Performance Optimizations
- **Efficient Caching**: Smart cache management with version control
- **Memory Management**: Proper cleanup and resource management
- **Optimized Processing**: Streamlined image processing pipeline
- **Responsive UI**: Smooth animations and efficient DOM updates

### Maintainability
- **Clean Code**: Removed debug logs and unnecessary complexity
- **Configuration**: Centralized configuration constants
- **Extensibility**: Easy to add new algorithms or features
- **Testing Ready**: Structure supports unit testing and integration testing

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing processes
   pkill -f "python.*iss_speed"
   lsof -ti:5002 | xargs kill -9
   ```

2. **Cache Issues**
   - Use "Clear Cache" button in the dashboard
   - Or manually delete files in the `cache/` directory

3. **Memory Issues with Large Datasets**
   - Process smaller image ranges
   - Use sampling for visualization (automatic for >10,000 points)

4. **GPS Data Missing**
   - Ensure images have GPS EXIF data
   - Check image file format and metadata

### Debug Mode
To enable debug logging, modify the Flask app startup:
```python
app.run(host='0.0.0.0', port=5002, debug=True)
```

## 📈 Performance Notes

- **Processing Speed**: ~2-5 seconds per image pair (depending on algorithm and image size)
- **Memory Usage**: ~100-500MB for typical datasets (100-1000 images)
- **Cache Size**: ~1-10MB per processed dataset
- **Browser Compatibility**: Modern browsers (Chrome, Firefox, Safari, Edge)

## 🔮 Future Enhancements

### Planned Features
- **Machine Learning Integration**: Enhanced cloudiness and quality classification
- **Batch Processing**: Process multiple folders simultaneously
- **Export Functionality**: Export results to CSV, JSON, or images
- **Advanced Filtering**: More sophisticated data filtering options
- **Real-time Processing**: Live processing of new images

### Extensibility
The clean architecture makes it easy to:
- Add new computer vision algorithms
- Implement additional filtering methods
- Integrate with external APIs
- Add new visualization types
- Extend the caching system

## 📝 License

This project is part of the ISS Speed Analysis research initiative. Please ensure proper attribution when using or modifying this code.

## 🤝 Contributing

When contributing to this project:
1. Follow the established code style and documentation standards
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility when possible

---

**Version**: 2.0 (Clean)  
**Last Updated**: 2025-01-07  
**Status**: Production Ready
