# 🚀 AstroPi Explorer Dashboard

A comprehensive web-based dashboard for analyzing International Space Station (ISS) speed data from AstroPi mission images using computer vision and feature detection algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

### 📊 **Real-time Speed Analysis**
- Calculate ISS orbital speed from sequential image pairs
- Support for multiple computer vision algorithms (ORB, SIFT)
- Configurable feature detection parameters
- Ground Sample Distance (GSD) customization

### 🎛️ **Advanced Filtering System**
- **Cloudiness Filter**: Classify images as clear, partly cloudy, or cloudy
- **Keypoint Percentile Filter**: Remove outlier speeds based on percentiles
- **Minimum Matches Filter**: Filter image pairs by match count
- **Custom GSD Configuration**: Override default ground sample distance

### 📈 **Interactive Visualizations**
- Real-time speed histograms and box plots
- Color-coded cloudiness classification
- Dynamic graph updates with filter changes
- Responsive web interface

### 🧪 **Comprehensive Testing**
- **51 automated tests** covering all functionality
- Unit, integration, and end-to-end test coverage
- Independent baseline validation
- Cache system testing
- API endpoint validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenCV 4.0+
- Flask 2.0+

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raphathegreat/astro_explorer.git
   cd astro_explorer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

3. **Add your image data:**
   - Create `photos-1/`, `photos-2/`, etc. folders in the project directory
   - Add your ISS images (JPG format) to these folders
   - The dashboard will automatically detect and process them

4. **Run the dashboard:**
   ```bash
   python iss_speed_html_dashboard_v2_clean.py
   ```

5. **Open your browser:**
   - Navigate to `http://localhost:5003`
   - Start analyzing your ISS speed data!

## 📁 Project Structure

```
astro_explorer/
├── iss_speed_html_dashboard_v2_clean.py    # Main Flask application
├── templates/
│   ├── dashboard_v2_clean.html            # Frontend interface
│   └── dashboard.html                     # Alternative dashboard
├── tests/                                 # Comprehensive test suite
│   ├── unit/                             # Unit tests
│   ├── integration/                      # API integration tests
│   ├── e2e/                             # End-to-end tests
│   └── fixtures/                        # Test data and utilities
├── realistic_test.py                     # Main test pack (51 tests)
├── version_control.py                    # Local version management
├── generate_independent_baseline.py      # Baseline generation
└── docs/                                # Documentation
    ├── README_CLEAN_VERSION.md          # Detailed documentation
    ├── DEVELOPMENT_RULES.md             # Development guidelines
    └── CLEANUP_SUMMARY.md               # Project organization
```

## 🎯 Usage Guide

### 1. **Load Image Data**
- Place your ISS images in `photos-*` folders
- Images should be sequential captures for speed calculation
- Supported formats: JPG, JPEG, PNG

### 2. **Configure Analysis Parameters**
- **Algorithm**: Choose between ORB and SIFT feature detection
- **FLANN**: Enable/disable Fast Library for Approximate Nearest Neighbors
- **RANSAC**: Enable/disable Random Sample Consensus filtering
- **Contrast Enhancement**: Select from CLAHE, histogram equalization, gamma correction, or unsharp masking
- **Max Features**: Set maximum number of features to detect

### 3. **Apply Filters**
- **Cloudiness Filter**: Set brightness and contrast thresholds for image classification
- **Keypoint Percentile Filter**: Remove outlier speeds (e.g., bottom 5% and top 5%)
- **Minimum Matches Filter**: Filter pairs with insufficient keypoint matches
- **Custom GSD**: Override the default Ground Sample Distance (12648 cm/pixel)

### 4. **Analyze Results**
- View real-time statistics: mean, median, mode speeds
- Examine speed distributions in histograms
- Identify cloudiness patterns in box plots
- Export data for further analysis

## 🧪 Testing

The project includes a comprehensive test suite with 51 automated tests:

```bash
# Run all tests
python realistic_test.py

# Run specific test categories
python -m pytest tests/unit/           # Unit tests
python -m pytest tests/integration/    # Integration tests
python -m pytest tests/e2e/           # End-to-end tests
```

### Test Coverage
- ✅ **Core Functionality**: Image processing, feature detection, speed calculation
- ✅ **API Endpoints**: All Flask routes and data validation
- ✅ **Filter System**: Cloudiness, percentile, and match count filters
- ✅ **Cache System**: Key generation, hit/miss behavior, invalidation
- ✅ **Data Consistency**: Statistics accuracy across all sections
- ✅ **UI Validation**: Frontend data rendering and user interactions

## 📊 Algorithm Comparison

The dashboard supports multiple computer vision algorithms for robust analysis:

| Algorithm | Speed | Accuracy | Use Case |
|-----------|-------|----------|----------|
| **ORB** | Fast | Good | Real-time processing |
| **SIFT** | Slow | Excellent | High-precision analysis |

### Feature Detection Parameters
- **FLANN**: Accelerates feature matching (recommended for SIFT)
- **RANSAC**: Removes outlier matches for better accuracy
- **Contrast Enhancement**: Improves feature detection in varying lighting

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development  # Enable debug mode
export FLASK_PORT=5003       # Custom port (default: 5003)
```

### Custom GSD Calculation
The default Ground Sample Distance is 12648 cm/pixel. You can override this in the dashboard:
- Enable "Custom GSD" in Section 4
- Set your specific GSD value
- Speed calculations will use your custom value

## 📈 Performance

- **Processing Speed**: ~2-5 seconds per image pair (ORB), ~10-30 seconds (SIFT)
- **Memory Usage**: ~100-200MB for typical datasets
- **Cache System**: Automatic caching of processed results for faster subsequent analysis
- **Concurrent Users**: Supports multiple simultaneous users

## 🤝 Contributing

We welcome contributions! Please follow our development guidelines:

1. **Read** `DEVELOPMENT_RULES.md` for coding standards
2. **Write tests** for new features (see `realistic_test.py`)
3. **Run the full test suite** before submitting
4. **Use version control** (`version_control.py`) for major changes
5. **Update documentation** as needed

### Development Workflow
```bash
# Create a backup before changes
python version_control.py backup "Description of changes"

# Make your changes
# ... edit code ...

# Run tests
python realistic_test.py

# If tests pass, commit your changes
git add .
git commit -m "Your commit message"
```

## 📚 Documentation

- **[Detailed Documentation](README_CLEAN_VERSION.md)** - Comprehensive guide
- **[Development Rules](DEVELOPMENT_RULES.md)** - Coding standards and guidelines
- **[Cleanup Summary](CLEANUP_SUMMARY.md)** - Project organization details

## 🐛 Troubleshooting

### Common Issues

**Q: Images not loading?**
- Ensure images are in `photos-*` folders within the project directory
- Check file permissions and formats (JPG, JPEG, PNG)

**Q: Slow processing?**
- Use ORB instead of SIFT for faster processing
- Enable FLANN for SIFT acceleration
- Reduce max features count

**Q: Inaccurate speeds?**
- Verify your GSD value is correct for your images
- Enable RANSAC filtering to remove outliers
- Check image quality and contrast

**Q: Tests failing?**
- Ensure all dependencies are installed
- Check that test images are available
- Run `python realistic_test.py` for detailed error messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ESA AstroPi Mission** for providing the ISS image data
- **OpenCV Community** for computer vision algorithms
- **Flask Team** for the web framework
- **Contributors** who helped improve this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/raphathegreat/astro_explorer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/raphathegreat/astro_explorer/discussions)
- **Documentation**: Check the `docs/` folder for detailed guides

---

**Made with ❤️ for the AstroPi community**

*Analyzing the cosmos, one image at a time* 🌌
