# 🕵️‍♂️ Deepfake Spy

**An advanced, browser-based tool that uses AI to analyze images for signs of deepfake manipulation and digital forgery.**

---

## 📖 Overview

In an era of sophisticated digital manipulation, **Deepfake Spy** empowers you to scrutinize images with cutting-edge AI.  
**No backend setup. No installation. 100% browser-based.**

---

## ✨ Key Features

- **🤖 AI-Powered Analysis:**  
  Utilizes Google's powerful Gemini model for deep forensic image analysis.
- **📥 Multiple Input Methods:**  
  - Drag & drop local files  
  - Paste direct image URLs
- **📊 Detailed Forensics Report:**  
  Breaks down analysis into categories like Lighting, Skin Texture, and Background Anomalies.
- **🔥 Manipulation Heatmap:**  
  Highlights suspicious regions with an intuitive overlay.
- **🗂️ Metadata Viewer:**  
  Inspect EXIF metadata for clues about origin, device, and editing.
- **🖌️ Modern & Responsive UI:**  
  Clean, intuitive, and mobile-friendly interface built with Tailwind CSS.

---

## ⚙️ How It Works

1. **Client-side Application:** Communicates directly with the Google Gemini API.
2. **Image Processing:** Converts images to Base64 in-browser (never uploaded to a server).
3. **Specialized Prompting:** Sends a detailed prompt to the AI, acting as a digital forensics expert.
4. **API Request:** Sends image data and prompt to Gemini API.
5. **Structured Response:** Receives JSON with probability score, verdict, breakdown, and heatmap.
6. **Data Visualization:** Displays results with progress circle, report, and heatmap overlay.

---

## 🚀 How to Use

1. **Open** `index.html` in any modern browser.
2. **Provide an Image:**
   - Drag & drop an image file
   - OR paste an image URL and click "Fetch Image"
3. **Analyze:** Click "Analyze Image" after preview appears.
4. **Review Results:** See the probability score, forensic report, and heatmap overlay.

---

## 🛠️ Tech Stack

- **Frontend:** HTML5, Tailwind CSS, Vanilla JavaScript (ES6+)
- **AI Model:** Google Gemini API
- **Libraries:** exif-js for metadata extraction

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

Licensed under the MIT License. See the