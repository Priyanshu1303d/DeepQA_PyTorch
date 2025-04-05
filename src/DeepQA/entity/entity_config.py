from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: Path
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    output_dir: Path
    vocab_file_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    output_path: Path
    vocab_file_path: Path
    epochs: int
    weight_decay: float
    learning_rate: float
    optimizer: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    saved_model_path: Path
    model_metrics_json: Path
    vocab_file_path: Path
    data_path: Path


@dataclass(frozen=True)
class ModelPredictionConfig:
    saved_model_path: Path
    vocab_file_path: Path
