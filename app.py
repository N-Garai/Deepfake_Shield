"""
Flask API Server for DeepFake Detection
Provides REST API endpoints for the frontend to call
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from model import DeepFakeDetector
import base64
from PIL import Image
import io
import traceback
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize the detector
detector = DeepFakeDetector()


@app.route('/')
def home():
    """Serve the frontend HTML"""
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('.', path)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_version': detector.model_version
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze image for deepfake detection
    
    Accepts:
        - JSON with base64 encoded image
        - Multipart form data with image file
    
    Returns:
        JSON with analysis results
    """
    try:
        # Check if request contains JSON data
        if request.is_json:
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({
                    'error': 'No image data provided',
                    'message': 'Please provide image data in base64 format'
                }), 400
            
            # Get base64 image data
            image_data = data['image']
            
            # Analyze image
            results = detector.analyze_image(image_data)
            
            return jsonify({
                'success': True,
                'results': results
            })
        
        # Check if request contains file upload
        elif 'file' in request.files:
            file = request.files['file']
            
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'message': 'Please select an image file'
                }), 400
            
            # Read image file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Analyze image
            results = detector.analyze_image(image)
            
            return jsonify({
                'success': True,
                'results': results
            })
        
        else:
            return jsonify({
                'error': 'Invalid request format',
                'message': 'Please provide image as JSON (base64) or multipart form data'
            }), 400
    
    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'message': str(e)
        }), 500


@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple images in batch
    
    Accepts:
        JSON with array of base64 encoded images
    
    Returns:
        JSON with array of analysis results
    """
    try:
        data = request.get_json()
        
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({
                'error': 'Invalid request',
                'message': 'Please provide an array of images'
            }), 400
        
        results = []
        for idx, image_data in enumerate(data['images']):
            try:
                result = detector.analyze_image(image_data)
                results.append({
                    'index': idx,
                    'success': True,
                    'results': result
                })
            except Exception as e:
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(data['images']),
            'results': results
        })
    
    except Exception as e:
        print(f"Error in batch analysis: {str(e)}")
        print(traceback.format_exc())
        
        return jsonify({
            'success': False,
            'error': 'Batch analysis failed',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500


if __name__ == '__main__':
    print("=" * 50)
    print("DeepFake Shield API Server")
    print("=" * 50)
    print("Starting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - Home")
    print("  GET  /api/health    - Health check")
    print("  POST /api/analyze   - Analyze single image")
    print("  POST /api/batch-analyze - Analyze multiple images")
    print("\nPress CTRL+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
