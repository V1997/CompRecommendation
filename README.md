# Property Recommendation System

A production-ready machine learning system for recommending comparable properties (comps) for real estate appraisals using advanced similarity search algorithms and statistical modeling.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd CompRecommendation

# Install dependencies
pip install -r requirements.txt

# Run the main demo system
python demo_system.py
```

## 📁 Project Structure

```
CompRecommendation/
├── src/                           # Core source code
│   ├── data_preprocessing/        # Data cleaning and preprocessing
│   └── similarity_search/         # ML similarity algorithms
├── data/                          # Data files
│   ├── processed/                 # Clean, processed datasets
│   └── models/                    # Trained ML models
├── presentation/                  # Demo scripts and presentation materials
├── docs/                          # Comprehensive documentation
├── scripts/                       # Development and testing utilities
├── archive/                       # Legacy development files
├── notebooks/                     # Jupyter analysis notebooks
├── demo_system.py                 # Main production demo
├── appraisals_dataset.json        # Raw dataset (22MB)
└── requirements.txt               # Python dependencies
```

## 🎯 System Features

### Core Functionality
- **Advanced Similarity Search**: Machine learning-based property matching using multiple algorithms
- **Data Preprocessing**: Automated cleaning and standardization of property data
- **Statistical Modeling**: Clustering, nearest neighbors, and distribution analysis
- **Scalable Architecture**: Handles large datasets with efficient processing

### Key Components
- **Property Preprocessor**: Cleans and standardizes property data
- **Similarity Engine**: Multi-algorithm property matching system
- **Demo Interface**: Interactive property recommendation system
- **Validation Tools**: System testing and performance validation

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation Steps
1. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   python scripts/validate_system.py
   ```

## 🎮 Usage

### Running the Main System
```bash
python demo_system.py
```

### Running Presentation Demos
```bash
# Simple demo version
python presentation/simple_demo.py

# Full presentation demo
python presentation/presentation_demo.py
```

### Development Scripts
```bash
# Test similarity search functionality
python scripts/test_similarity_search.py

# Validate system integrity
python scripts/validate_system_clean.py
```

## 📊 Dataset

The system processes a comprehensive real estate appraisal dataset containing:
- **10,172 processed properties** with standardized features
- **Subject property details** for each appraisal
- **Available comparable properties** for selection
- **Expert-selected comparables** for validation

### Data Processing Pipeline
1. **Raw Data Ingestion**: Loads `appraisals_dataset.json` (22MB)
2. **Data Cleaning**: Removes duplicates and standardizes formats
3. **Feature Engineering**: Creates numerical and categorical features
4. **Model Training**: Trains similarity search algorithms
5. **Output Generation**: Produces `properties_preprocessed.csv`

## 🤖 Machine Learning Approach

### Similarity Algorithms
- **Cosine Similarity**: Feature vector comparisons
- **Euclidean Distance**: Spatial distance calculations
- **K-Nearest Neighbors**: Clustering-based recommendations
- **Weighted Scoring**: Multi-factor property assessment

### Model Performance
- Processes 10,000+ properties efficiently
- Sub-second recommendation generation
- Validation against expert selections
- Continuous learning capability

## 📖 Documentation

- **[Complete System Guide](docs/COMPLETE_GUIDE.md)**: Comprehensive technical documentation
- **[System Status](docs/SYSTEM_STATUS_FINAL.md)**: Current implementation status
- **[Demo Guide](presentation/DEMO_GUIDE.md)**: Presentation and demo instructions
- **[Presentation Script](presentation/PRESENTATION_SCRIPT.md)**: Detailed presentation materials

## 🧪 Testing & Validation

### System Validation
```bash
# Run comprehensive system tests
python scripts/validate_system.py

# Test similarity search components
python scripts/test_similarity_search.py
```

### Performance Metrics
- **Accuracy**: Matches against expert selections
- **Speed**: Sub-second processing times
- **Scalability**: Handles datasets of 10,000+ properties
- **Reliability**: Robust error handling and validation

## 🚀 Production Deployment

### Pre-deployment Checklist
- ✅ All tests passing
- ✅ Data preprocessing complete
- ✅ Models trained and validated
- ✅ Documentation updated
- ✅ Code organized and cleaned

### Deployment Steps
1. **Environment Setup**: Configure production environment
2. **Data Migration**: Deploy processed datasets
3. **Model Deployment**: Load trained ML models
4. **System Validation**: Run production tests
5. **Monitoring Setup**: Configure performance monitoring

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit pull request

### Code Standards
- Follow PEP 8 Python style guide
- Include comprehensive docstrings
- Add unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:
- Review the comprehensive documentation in `docs/`
- Check the presentation materials in `presentation/`
- Run validation scripts in `scripts/`
- Submit issues through the repository issue tracker
