# 🧠 DeepQA: RNN-Based Question Answering System

A PyTorch implementation of a Question Answering (QA) system using Recurrent Neural Networks (RNN). This project is modular, experiment-tracked, and follows best MLOps practices with DVC, MLflow, and Docker integration.

**Live Demo:** [https://questionanswer01.streamlit.app/](https://questionanswer01.streamlit.app/)

---

## 📌 Features

- ✅ End-to-end pipeline: data ingestion → validation → transformation → training → evaluation → prediction
- 🔁 RNN-based QA model built with PyTorch
- 📊 Experiment tracking 
- 📦 Data versioning with **DVC**
- 🐳 Containerized environment via **Docker**
- ⚙️ Modular code structure with components and pipelines
- 🔄 CI/CD ready with GitHub Workflows
- 🌐 Web interface deployed with Streamlit

---

## 🗂️ Project Structure

```
DeepQA_PyTorch/
│
├── .github/                  # GitHub configuration
│   └── workflows/            # CI/CD workflows
│
├── artifacts/                # Generated artifacts
│
├── config/                   # Configuration files
│   └── config.yaml           # Main configuration
│
├── logs/                     # Application logs
│   └── running_logs.log      # Log file
│
├── research/                 # Jupyter notebooks for experimentation
│   ├── Stage_01_Data_Ingestion.ipynb
│   ├── Stage_02_Data_Validation.ipynb
│   ├── Stage_03_Data_Transformation.ipynb
│   ├── Stage_04_Model_Trainer.ipynb
│   ├── Stage_05_Model_Evaluation.ipynb
│   └── Stage_06_Model_Prediction.ipynb
│
├── src/                      # Source code
│   └── DeepQA/               # Main package
│       ├── Components/       # Pipeline components
│       │   ├── Stage_01_Data_Ingestion.py
│       │   ├── Stage_02_Data_Validation.py
│       │   ├── Stage_03_Data_Transformation.py
│       │   ├── Stage_04_Model_Trainer.py
│       │   ├── Stage_05_Model_Evaluation.py
│       │   └── Stage_06_Model_Prediction.py
│       │
│       ├── config/           # Configuration utilities
│       │   └── configuration.py
│       │
│       ├── constants/        # Constants and fixed variables
│       │   └── __init__.py
│       │
│       ├── entity/           # Data entities/models
│       │   └── entity_config.py
│       │
│       ├── logging/          # Logging utilities
│       │   └── __init__.py
│       │
│       ├── pipeline/         # Pipeline orchestration
│       │   ├── Stage_01_Data_Ingestion.py
│       │   ├── Stage_02_Data_Validation.py
│       │   ├── Stage_03_Data_Transformation.py
│       │   ├── Stage_04_Model_Trainer.py
│       │   ├── Stage_05_Model_Evaluation.py
│       │   └── Stage_06_Model_Prediction.py
│       │
│       └── utils/            # Utility functions
│           └── common.py
│
├── .gitignore                # Git ignore file
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── 100_Unique_QA_Dataset.zip # Dataset
├── app.py                    # Web application entry point
├── DockerFile                # Docker configuration
├── dvc.yaml                  # DVC pipeline configuration
├── format.sh                 # Formatting script
├── LICENSE                   # License file
├── main.py                   # Main application entry point
├── params.yaml               # Parameters configuration
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup script
└── template.py               # Template file
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Priyanshu1303d/DeepQA_PyTorch.git
cd DeepQA_PyTorch
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .  # Install the package in development mode
```

### 3. Setup DVC & Download Data

```bash
dvc pull
```

### 4. Execute the Pipeline

```bash
python main.py
```

Or use Docker:

```bash
docker build -t deepqa .
docker run deepqa
```

## 🔬 Pipeline Details

This project implements a complete MLOps pipeline for question answering:

1. **Data Ingestion**: Loading and initial preprocessing of data
2. **Data Validation**: Validating data quality and integrity
3. **Data Transformation**: Transforming raw data into model-ready format
4. **Model Training**: Training the RNN-based QA model
5. **Model Evaluation**: Evaluating model performance
6. **Model Prediction**: Generating predictions from the trained model

## 🌐 Web Application

### Local Development
Run the web application locally:

```bash
streamlit run app.py
```

### Live Demo
Access the deployed application at: [https://questionanswer01.streamlit.app/](https://questionanswer01.streamlit.app/)

## 🧪 Development

### Code Formatting

Use the provided formatting script:

```bash
chmod +x format.sh
./format.sh
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:

```bash
pre-commit install
```

## 📁 Dataset

The project includes a `100_Unique_QA_Dataset.zip` which contains the question-answer pairs for training and evaluation.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributions

Feel free to:
* ⭐ Star the repo
* 📂 Fork the project
* 🐛 Report issues
* ✅ Submit a PR

Any feedback is welcome!
