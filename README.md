# 🕵️‍♂️ Deepfake Shield

**An advanced AI-powered tool that analyzes images for signs of deepfake manipulation and digital forgery using Python-based machine learning algorithms.**

Deepfake Shield is a browser-based tool that uses AI algorithms to analyze images for signs of deepfake manipulation and digital forgery. It provides instant authenticity scores, technical breakdowns, and privacy-focused analysis—all without uploading your images anywhere.

---

## 📖 Overview

In an era of sophisticated digital manipulation, **Deepfake Shield** empowers you to scrutinize images with cutting-edge AI forensics algorithms. The system combines a modern web interface with a powerful Python backend for accurate deepfake detection.

---

## ✨ Key Features

- **📥 Easy Upload:**  
  Drag & drop images or browse files for instant analysis.
- **📊 Authenticity Score:**  
  AI-powered scoring with visual indicators (Authentic, Suspicious, Fake).
- **🔬 Technical Breakdown:**  
  - Compression Artifacts Analysis
  - Edge Consistency Detection
  - Noise Pattern Recognition
  - Color Distribution Analysis
  - Frequency Domain Analysis (DCT)
- **🖼️ Image Preview:**  
  See your uploaded image and its metadata.
- **📱 Responsive UI:**  
  Sleek, mobile-friendly interface with gradient backgrounds and interactive elements.
- **🔒 Privacy-Focused:**  
  All analysis performed locally—images never leave your machine.
- **📄 Downloadable Reports:**  
  Generate detailed analysis reports in text format.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/N-Garai/Deepfake_Spy.git
   cd Deepfake_Spy
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the API server:**
   ```bash
   python app.py
   ```
   
   The server will start on `http://localhost:5000`

4. **Open the web interface:**
   - Open `index.html` in your browser, or
   - Navigate to `http://localhost:5000` if using a local server

### Basic Usage

1. **Upload an Image:**
   - Drag & drop an image file or click to browse
2. **Analyze:**
   - The system automatically sends the image to the Python backend
3. **Review Results:**
   - View authenticity score, verdict, and technical analysis
4. **Download Report:**
   - Generate a detailed report of the analysis

---

## 🛠️ Tech Stack

### Frontend
- **HTML5** - Modern semantic markup
- **CSS3** - Custom styling with gradients and animations
- **Vanilla JavaScript (ES6+)** - Client-side logic and API integration
- **Font Awesome** - Icons and visual elements

### Backend
- **Python 3.8+** - Core processing engine
- **Flask** - REST API server
- **OpenCV** - Image processing and computer vision
- **NumPy** - Numerical computations
- **SciPy** - Scientific computing (DCT, signal processing)
- **Pillow (PIL)** - Image manipulation

### Detection Algorithms
- **Compression Artifact Analysis** - Detects unusual JPEG compression patterns
- **Edge Consistency Detection** - Identifies inconsistent edges using Sobel operators
- **Noise Pattern Analysis** - Examines noise characteristics and uniformity
- **Color Distribution Analysis** - Analyzes HSV/LAB color spaces and entropy
- **Frequency Domain Analysis** - Uses DCT to detect frequency anomalies

---

## 📡 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
**GET** `/api/health`

Check if the API is running and get model version.

**Response:**
```json
{
  "status": "healthy",
  "model_version": "1.0.0"
}
```

#### 2. Analyze Image
**POST** `/api/analyze`

Analyze a single image for deepfake detection.

**Request (JSON):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Request (Form Data):**
```
file: <image file>
```

**Response:**
```json
{
  "success": true,
  "results": {
    "authenticity_score": 75.32,
    "technical_details": {
      "compression_artifacts": 68.5,
      "edge_consistency": 82.3,
      "noise_patterns": 71.8,
      "color_distribution": 79.2,
      "frequency_analysis": 74.8
    },
    "verdict": {
      "classification": "authentic",
      "title": "Likely Authentic",
      "description": "This image shows strong indicators of being genuine...",
      "confidence": "high"
    },
    "model_version": "1.0.0"
  }
}
```

#### 3. Batch Analyze
**POST** `/api/batch-analyze`

Analyze multiple images in a single request.

**Request:**
```json
{
  "images": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "data:image/png;base64,iVBORw0KGgoAAAANS..."
  ]
}
```

**Response:**
```json
{
  "success": true,
  "total": 2,
  "results": [
    {
      "index": 0,
      "success": true,
      "results": { ... }
    },
    {
      "index": 1,
      "success": true,
      "results": { ... }
    }
  ]
}
```

### Authenticity Score Interpretation

- **70-100%**: Likely Authentic (natural characteristics)
- **40-69%**: Suspicious (mixed indicators)
- **0-39%**: Likely AI-Generated (artificial patterns detected)

### Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `404`: Endpoint not found
- `500`: Internal server error

**Error Response:**
```json
{
  "success": false,
  "error": "Error type",
  "message": "Detailed error message"
}
```

---

## 🔬 Detection Algorithm Details

The model uses a weighted combination of multiple forensic techniques:

1. **Compression Artifact Analysis (20% weight)**
   - Examines 8x8 JPEG block patterns
   - Detects unusual compression signatures
   - Identifies over-smoothing or blocky artifacts

2. **Edge Consistency (25% weight)**
   - Uses Sobel edge detection
   - Analyzes edge sharpness and variance
   - Detects unnatural edge transitions

3. **Noise Pattern Analysis (25% weight)**
   - Applies high-pass filtering to extract noise
   - Examines local noise variance
   - Identifies unnaturally uniform noise (common in AI images)

4. **Color Distribution (15% weight)**
   - Analyzes HSV and LAB color spaces
   - Calculates Shannon entropy of color histograms
   - Detects unnatural color clustering

5. **Frequency Domain Analysis (15% weight)**
   - Applies Discrete Cosine Transform (DCT)
   - Examines frequency component ratios
   - Detects AI-typical frequency patterns

---

## 📁 Project Structure

```
Deepfake_Shield/
├── index.html              # Web interface
├── app.py                  # Flask API server
├── model.py                # Detection algorithms
├── requirements.txt        # Python dependencies
├── README.md              # Documentation
└── LICENSE                # Apache 2.0 License
```

---

## 🌐 Deployment

### Local Development
The application runs locally by default with the Python backend and HTML frontend.

### Static Hosting (Frontend Only)
For frontend-only deployment, you can host on:
- **Vercel**
- **Netlify**
- **GitHub Pages**

**Note:** You'll need to deploy the Python backend separately (e.g., Heroku, AWS, Railway) and update the `API_URL` in `index.html`.

### Full Stack Deployment
For production deployment with backend:
1. Deploy Python API to a cloud service (Heroku, Railway, AWS, etc.)
2. Update `API_URL` in `index.html` to point to your API endpoint
3. Deploy frontend to static hosting or serve via Flask

---

## 🧪 Testing

Test the API using curl:

```bash
# Health check
curl http://localhost:5000/api/health

# Analyze image (with file upload)
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5000/api/analyze
```

---

## 📊 Limitations

- Maximum image size: 10MB
- Supported formats: JPEG, PNG, WebP
- Processing time: ~1-3 seconds per image
- The model provides probabilistic estimates, not definitive proof
- Best results with high-resolution images

---

## 📜 Disclaimer

> This tool uses AI algorithms to estimate the likelihood of an image being a deepfake or AI-generated.  
> **It is not 100% accurate and should not be used as the sole basis for critical decisions.**  
> Always cross-reference findings with other sources and human expert review for serious investigations.
> 
> The analysis is performed using computational forensics techniques and may produce false positives or false negatives.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## 📄 License

Licensed under the Apache 2.0 License. See the LICENSE file for details.

---

## 👨‍💻 Author

**N-Garai**  
GitHub: [@N-Garai](https://github.com/N-Garai)  
Repository: [Deepfake_Spy](https://github.com/N-Garai/Deepfake_Spy)

---

**Version:** 1.0.0  
**Last Updated:** December 2025

