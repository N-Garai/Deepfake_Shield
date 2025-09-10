# 🕵️‍♂️ Deepfake Detector

**An advanced, browser-based tool that uses AI to analyze images for signs of deepfake manipulation and digital forgery.**

---

## 📖 Overview

In an era of sophisticated digital manipulation, **Deepfake Detector** empowers you to scrutinize images with cutting-edge AI.  
**No backend setup. No installation. 100% browser-based.**

---

## ✨ Key Features

- **🤖 AI-Powered Analysis:**  
  Integrates with Google's Gemini model for forensic image evaluation.
- **📥 Easy Upload:**  
  Drag & drop images or browse files for instant analysis.
- **📊 Authenticity Score:**  
  Visual score bar and verdict (Authentic, Suspicious, Fake).
- **🔬 Technical Breakdown:**  
  - Compression Artifacts
  - Edge Consistency
  - Noise Patterns
  - Color Distribution
- **🖼️ Image Preview:**  
  See your uploaded image and its metadata.
- **📱 Responsive UI:**  
  Sleek, mobile-friendly interface with gradient backgrounds and interactive elements.

## 🚀 How to Use

1. **Open** `index.html` in any modern browser.
2. **Provide an Image:**
   - Drag & drop an image file
3. **Analyze:** Click "Analyze Image" after preview appears.
4. **Review Results:** See the probability score, forensic report, and heatmap overlay.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+)
-**Image Processing**:HTML5 Canvas API - Pixel manipulation
                      FileReader API - File handling
                      ImageData API - Pixel-level analysis


-**Custom algorithms for:** Face region detection
                            Compression artifact analysis
                            Edge consistency checking
                            Noise pattern recognition
                            Color distribution analysis
                            Frequency domain analysis


---

## 🖥️ Running Locally

**No complex setup required!**

- **Method 1:**  
  Double-click `index.html` to open in your browser.
- **Method 2 (Local Server):**  
  If you have Python 3:
  ```sh
  python -m http.server
  ```
  Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🌐 Deployment

Deploy easily to any static hosting service:
- Vercel
- Netlify
- GitHub Pages

Just link your GitHub repo for continuous deployment.

---

## 📜 Disclaimer

> This tool uses AI to estimate the likelihood of an image being a deepfake.  
> **It is not 100% accurate. Use for informational and educational purposes only.**  
> Always cross-reference findings with other sources for serious investigations.

---

## 📄 License

Licensed under the MIT License. See the LICENSE file for details.

## 🚀 Live Demo

You can view the live application here: https://deepfakespy.vercel.app/
