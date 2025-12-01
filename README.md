# OCR Project - Optical Character Recognition Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [About](#about)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Running the API Server](#running-the-api-server)
  - [Training a Model](#training-a-model)
  - [Making Predictions](#making-predictions)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [References](#references)

## About

This OCR (Optical Character Recognition) project is a comprehensive pipeline for text recognition in images. It leverages the TrOCR (Transformer-based Optical Character Recognition) model for accurate text recognition and includes utilities for document image preprocessing, line extraction, and model serving via a REST API.

The project combines:
- **Data processing pipelines** for image preparation and annotation
- **Model training infrastructure** using Kedro for reproducible ML workflows
- **REST API service** for real-time text recognition

## Features

- 📄 **Document Line Extraction** - Intelligent detection and extraction of text lines from document images
- 🤖 **TrOCR Integration** - Fine-tunable transformer-based OCR model
- 🔄 **Image Preprocessing** - Grayscale conversion, thresholding, and line detection
- 🚀 **REST API** - FastAPI-based service for predictions with automatic documentation
- 📊 **Kedro Pipelines** - Reproducible data and model training workflows
- 📈 **Model Training** - End-to-end pipeline for model fine-tuning
- 🧪 **Testing Infrastructure** - Unit tests for pipelines and components

## Project Structure

```
OCR_project/
├── README.md                          # This file
├── requirements.txt                   # Root-level dependencies
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 1_OpenImages.ipynb            # Data loading and exploration
│   ├── 2_TrainSimpleModel.ipynb      # Model training notebook
│   ├── 3_EvaluateSimpleModel.ipynb   # Model evaluation
│   ├── trocr_notebooks_ft/           # Fine-tuned TrOCR model files
│   └── data/                         # Notebook data storage
├── src/
│   ├── ocr-notebooks/                # Kedro project for ML pipelines
│   │   ├── conf/                     # Configuration files
│   │   │   ├── base/                 # Base configuration
│   │   │   │   ├── catalog.yml       # Data catalog
│   │   │   │   ├── parameters.yml    # Pipeline parameters
│   │   │   │   └── parameters_*.yml  # Pipeline-specific parameters
│   │   │   └── local/                # Local overrides
│   │   ├── src/ocr_notebooks/
│   │   │   ├── pipelines/            # Data and training pipelines
│   │   │   │   ├── data_ingestion/   # Data loading pipeline
│   │   │   │   └── model_training/   # Model training pipeline
│   │   │   ├── pipeline_registry.py  # Pipeline registration
│   │   │   └── settings.py           # Project settings
│   │   ├── tests/                    # Unit tests
│   │   └── requirements.txt          # Pipeline dependencies
│   └── ocr-serving/                  # FastAPI service
│       ├── api/
│       │   └── serve.py              # FastAPI application
│       ├── ocr_pipeline/             # Core OCR logic
│       │   ├── line_extractor.py     # Line extraction utilities
│       │   └── serving.py            # Prediction logic
│       ├── scripts/                  # Testing scripts
│       ├── models/                   # Model storage
│       ├── data/                     # Data storage
│       └── requirements.txt          # Service dependencies
└── ocr_venv/                         # Python virtual environment (created locally)
```

## Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package manager for Python
- **Virtual Environment**: Recommended for isolation
- **System Dependencies**: Build tools (gcc, etc.) for compiling some packages
- **Disk Space**: ~2GB for models and data

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sergioosorio-coder/OCR_project.git
cd OCR_project
```

### 2. Create Virtual Environment

```bash
python3 -m venv ocr_venv
source ocr_venv/bin/activate  # On Windows: ocr_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install root-level dependencies
pip install -r requirements.txt

# Install OCR serving dependencies
pip install -r src/ocr-serving/requirements.txt

# Install OCR notebooks (pipelines) dependencies
pip install -r src/ocr-notebooks/requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; import transformers; print('Installation successful!')"
```

## Quick Start

### Running the API Server

Start the FastAPI server for making real-time predictions:

```bash
# Activate virtual environment
source ocr_venv/bin/activate

# Navigate to serving directory
cd src/ocr-serving

# Run the server
python -m uvicorn api.serve:app --host 0.0.0.0 --port 4040 --reload
```

The server will be available at `http://localhost:4040`

**Interactive API Documentation**: Visit `http://localhost:4040/docs` to explore available endpoints with Swagger UI

### Training a Model

To train or fine-tune a model using the data pipeline:

```bash
# Activate virtual environment
source ocr_venv/bin/activate

# Navigate to notebooks directory
cd src/ocr-notebooks

# Run the Kedro pipeline
kedro run

# Or run specific pipelines
kedro run --pipeline data_ingestion
kedro run --pipeline model_training
```

**Configuration**: Adjust training parameters in `conf/base/parameters_model_training.yml`

### Making Predictions

Once the API server is running, you can make predictions:

```bash
# Python example
import requests
import base64

# Read and encode image
with open("path/to/image.png", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode()

# Make prediction request
response = requests.post(
    "http://localhost:4040/predict",
    json={"image": image_data}
)

print(response.json())
```

Or using cURL:

```bash
curl -X POST "http://localhost:4040/predict" \
  -H "Content-Type: application/json" \
  -d '{"image":"<base64_encoded_image>"}'
```

## API Documentation

### Available Endpoints

#### 1. **Health Check**
- **Endpoint**: `GET /health`
- **Description**: Check if the API is running
- **Response**: `{"status": "healthy"}`

#### 2. **Predict**
- **Endpoint**: `POST /predict`
- **Description**: Perform OCR on an image
- **Request Body**:
  ```json
  {
    "image": "<base64_encoded_image_string>"
  }
  ```
- **Response**:
  ```json
  {
    "text": "recognized text",
    "confidence": 0.95,
    "lines": ["line 1", "line 2"]
  }
  ```

#### 3. **Extract Lines**
- **Endpoint**: `POST /extract-lines`
- **Description**: Extract text lines from image
- **Request Body**:
  ```json
  {
    "image": "<base64_encoded_image_string>"
  }
  ```
- **Response**:
  ```json
  {
    "lines": [
      {"text": "line 1", "bbox": [x1, y1, x2, y2]},
      {"text": "line 2", "bbox": [x1, y1, x2, y2]}
    ]
  }
  ```

### Interactive Documentation

Visit `http://localhost:4040/docs` (Swagger UI) or `http://localhost:4040/redoc` (ReDoc) for interactive API documentation and the ability to test endpoints directly.

## Configuration

### Pipeline Parameters

Configuration files are located in `src/ocr-notebooks/conf/base/`:

#### `parameters.yml`
Main configuration file for the project

#### `parameters_data_ingestion.yml`
Data loading and preprocessing parameters:
- Image format specifications
- Data source paths
- Preprocessing options (resize, normalize)

#### `parameters_model_training.yml`
Model training configuration:
- Learning rate
- Batch size
- Number of epochs
- Model architecture choices
- Validation split

Example:
```yaml
model_training:
  learning_rate: 0.0001
  batch_size: 16
  num_epochs: 10
  model_name: "microsoft/trocr-base-handwriting"
```

### Data Catalog

Edit `conf/base/catalog.yml` to configure data sources and outputs:

```yaml
training_data:
  type: pandas.CSVDataSet
  filepath: data/training_data.csv

model_artifacts:
  type: pickle.PickleDataSet
  filepath: models/trained_model.pkl
```

## Development

### Running Tests

```bash
# Run all tests
pytest src/ocr-notebooks/tests/

# Run specific test file
pytest src/ocr-notebooks/tests/pipelines/model_training/test_pipeline.py

# Run with coverage
pytest --cov=src/ocr_notebooks src/ocr-notebooks/tests/
```

### Project Structure for Development

- **Pipelines**: Add new data processing logic in `src/ocr-notebooks/src/ocr_notebooks/pipelines/`
- **API Endpoints**: Add new routes in `src/ocr-serving/api/serve.py`
- **OCR Logic**: Core prediction functions in `src/ocr-serving/ocr_pipeline/`

### Setting Up Development Environment

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8 mypy

# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Troubleshooting

### Virtual Environment Issues

**Problem**: `No module named 'ocr_notebooks'`
```bash
# Solution: Ensure you're in the correct virtual environment
source ocr_venv/bin/activate
pip install -e src/ocr-notebooks/
```

### Port Already in Use

**Problem**: `Address already in use: 0.0.0.0:4040`
```bash
# Solution: Use a different port
python -m uvicorn api.serve:app --host 0.0.0.0 --port 4041

# Or kill the process using the port
lsof -ti:4040 | xargs kill -9  # macOS/Linux
```

### Out of Memory

**Problem**: CUDA out of memory error during training
```bash
# Solution: Reduce batch size in conf/base/parameters_model_training.yml
batch_size: 8  # reduced from 16
```

### Model Not Found

**Problem**: `FileNotFoundError: model files not found`
```bash
# Solution: Download or place model files in the correct directory
cd src/ocr-serving/models/
# Place your model files here or configure path in parameters
```

### API Connection Issues

**Problem**: Cannot connect to localhost:4040
```bash
# Check if server is running
curl http://localhost:4040/health

# Verify firewall settings or use 127.0.0.1 instead of 0.0.0.0
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -am 'Add improvement'`)
6. Push to the branch (`git push origin feature/improvement`)
7. Submit a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include unit tests for new features
- Update documentation as needed

## References

### Models & Frameworks

- **TrOCR**: [Microsoft TrOCR Documentation](https://huggingface.co/microsoft/trocr-base-handwriting)
- **Transformers**: [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Kedro**: [Kedro Documentation](https://kedro.readthedocs.io/)
- **FastAPI**: [FastAPI Documentation](https://fastapi.tiangolo.com/)
- **PyTorch**: [PyTorch Documentation](https://pytorch.org/)

### Related Papers

- Transformer-based Optical Character Recognition (TrOCR)
- Vision Transformer for Document Image Analysis
- Line Segmentation in Document Images

### Useful Resources

- [OpenImages Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [Document Image Analysis](https://www.scribd.com/doc/)
- [OCR Benchmarks](https://paperswithcode.com/task/optical-character-recognition)

---

**Last Updated**: January 2025  
**Maintainer**: Sergio Osorio  
**License**: MIT

For questions or issues, please open an issue on the [GitHub repository](https://github.com/sergioosorio-coder/OCR_project).
