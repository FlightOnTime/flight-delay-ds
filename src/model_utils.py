"""
Utilitários para Carregar Modelo e Artefatos
Implementação com Path Absoluto para maior robustez em APIs.
"""
import json
from pathlib import Path
from typing import Any, Dict

import joblib

# Define o diretório base como sendo DOIS NÍVEIS acima de 'src/model_utils.py'
# (ou seja, a raiz do projeto)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Funções de Carregamento Modificadas ---


def load_model(model_path: str = "models/randomforest_v7_final.pkl"):
    """
    Carrega modelo treinado usando caminho absoluto.
    """
    absolute_path = BASE_DIR / model_path

    if not absolute_path.exists():
        raise FileNotFoundError(f"❌ Modelo não encontrado: {absolute_path}")

    model = joblib.load(absolute_path)
    print(f"✅ Modelo carregado: {absolute_path.name}")
    return model


def load_encoders(encoder_path: str = "models/label_encoders_v7.pkl"):
    """
    Carrega encoders categóricos usando caminho absoluto.
    """
    absolute_path = BASE_DIR / encoder_path

    if not absolute_path.exists():
        raise FileNotFoundError(f"❌ Encoders não encontrados: {absolute_path}")

    encoders = joblib.load(absolute_path)
    print(f"✅ Encoders carregados: {absolute_path.name}")
    return encoders


def load_metadata(metadata_path: str = "models/metadata_v7.json") -> Dict[str, Any]:
    """
    Carrega metadados do modelo usando caminho absoluto.
    """
    absolute_path = BASE_DIR / metadata_path

    if not absolute_path.exists():
        raise FileNotFoundError(f"❌ Metadata não encontrado: {absolute_path}")

    with open(absolute_path, 'r') as f:
        metadata = json.load(f)

    print(f"✅ Metadata carregado: {absolute_path.name}")
    return metadata


def load_feature_names(feature_path: str = "models/feature_names_v7.json") -> Dict[str, list]:
    """
    Carrega lista de features do modelo usando caminho absoluto.
    """
    absolute_path = BASE_DIR / feature_path

    if not absolute_path.exists():
        raise FileNotFoundError(f"❌ Feature names não encontrado: {absolute_path}")

    with open(absolute_path, 'r') as f:
        features = json.load(f)

    print(f"✅ Features carregadas: {absolute_path.name}")
    return features
