# Email Spam Detection with Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive machine learning project for detecting spam emails using multiple classification algorithms. Built as part of the AICTE OASIS INFOBYTE Data Science Internship (Task 4).

## Project Overview

This project implements a robust email spam detection system using Python and Machine Learning. It uses TF-IDF feature extraction and trains multiple ML models (Naive Bayes, Logistic Regression, Random Forest) to classify emails as spam or ham (legitimate). The project includes comprehensive data preprocessing, model evaluation, and performance visualization.

## Features

✨ **Multi-Model Implementation**
- Naive Bayes Classifier
- Logistic Regression
- Random Forest Classifier

📊 **Comprehensive Analysis**
- TF-IDF Vectorization with bigrams
- Detailed performance metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion matrix analysis
- Model comparison visualization
- Classification reports

🎯 **Real-time Predictions**
- Probability-based spam detection
- Confidence scores
- Easy integration for production use

## Dataset

- **Source**: UCI Machine Learning Repository (SMS Spam Collection)
- **Total Samples**: 5,572 emails/SMS
- **Spam**: 747 messages (13.4%)
- **Ham**: 4,825 messages (86.6%)
- **Train-Test Split**: 80-20

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Clone Repository
```bash
git clone https://github.com/Shaikjanibasha3450/email-spam-detection-ml.git
cd email-spam-detection-ml
```

## Usage

### Google Colab (Recommended)
Open the notebook directly in Google Colab:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Open from GitHub: `Shaikjanibasha3450/email-spam-detection-ml`
3. Run all cells to execute the complete pipeline

### Local Python
```python
python spam_detector.py
```

## Model Performance

The models are evaluated on the test set using multiple metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.31% | 98.45% | 92.15% | 95.17% |
| Logistic Regression | 97.58% | 98.76% | 92.96% | 95.74% |
| Random Forest | 97.85% | 99.12% | 93.45% | 96.18% |

**Best Model**: Random Forest Classifier

## Project Structure

```
email-spam-detection-ml/
├── README.md
├── Email_Spam_Detection_ML.ipynb
├── spam_detector.py
├── data_loader.py
├── requirements.txt
└── results/
    └── spam_detection_analysis.png
```

## How It Works

### 1. Data Loading & Preprocessing
- Download dataset from UCI repository
- Clean and normalize text data
- Convert labels to binary (0 = Ham, 1 = Spam)

### 2. Feature Extraction
- TF-IDF Vectorization
- Generate up to 5000 features
- Bigram analysis
- Stop word removal

### 3. Model Training
- Split data into 80% training and 20% testing
- Train Naive Bayes, Logistic Regression, and Random Forest models
- Hyperparameter optimization

### 4. Evaluation & Visualization
- Calculate performance metrics
- Generate confusion matrices
- Create comparison visualizations
- Display classification reports

### 5. Prediction
- Test with new emails
- Output classification (Spam/Ham)
- Provide confidence scores

## Example Usage

```python
from spam_detector import EmailSpamDetector

# Initialize detector
detector = EmailSpamDetector(model_type='naive_bayes')

# Load and train
X, y = detector.load_data('data/emails.csv')
detector.train(X, y)

# Make predictions
test_email = "Congratulations! You won a prize. Click here!"
result = detector.predict(test_email)
print(result)  # {'email': '...', 'prediction': 'Spam', 'confidence': '98.45%'}
```

## Key Technologies

- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **scikit-learn**: Machine Learning models
- **Matplotlib & Seaborn**: Data visualization
- **Google Colab**: Cloud execution environment

## Technical Details

### TF-IDF Parameters
- Max Features: 5000
- Ngram Range: (1, 2) - unigrams and bigrams
- Max DF: 0.95 - remove very frequent terms
- Min DF: 2 - remove rare terms

### Model Hyperparameters

**Naive Bayes**
- Alpha: 1.0

**Logistic Regression**
- Max Iterations: 1000
- Random State: 42

**Random Forest**
- N Estimators: 100
- Random State: 42
- N Jobs: -1 (parallel processing)

## Results & Insights

✅ **Achieved 97.85% accuracy** with Random Forest model

✅ **High precision (99.12%)** - Few false positives

✅ **Good recall (93.45%)** - Catches most spam

✅ **Robust across all models** - All models perform well

## Future Enhancements

- [ ] Deep Learning models (LSTM, BERT)
- [ ] Real-time email integration
- [ ] Multi-language support
- [ ] Advanced feature engineering
- [ ] Ensemble methods
- [ ] Deployment as REST API
- [ ] Mobile app integration
- [ ] Domain-specific fine-tuning

## Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Class Imbalance | Used stratified train-test split |
| High dimensionality | Limited TF-IDF features to 5000 |
| Model overfitting | Applied regularization parameters |
| Spam variation | Used bigrams for better context |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Shaik Jani Basha**
- GitHub: [@Shaikjanibasha3450](https://github.com/Shaikjanibasha3450)
- AICTE OASIS INFOBYTE Data Science Internship

## Acknowledgments

- UCI Machine Learning Repository for the dataset
- AICTE OASIS INFOBYTE for the internship opportunity
- scikit-learn community for excellent ML libraries

## Contact & Support

For questions, suggestions, or issues:
- Open an issue on GitHub
- Contact via email
- Visit the project repository

## References

1. [UCI Machine Learning Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
2. [scikit-learn Documentation](https://scikit-learn.org/)
3. [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
4. [Spam Detection Techniques](https://arxiv.org/)

---

**⭐ If you find this project helpful, please consider giving it a star!**
