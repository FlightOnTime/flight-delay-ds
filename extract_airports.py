import json
import pickle
import sys

# Importa a classe necessária
from sklearn.preprocessing import LabelEncoder

# --- CORREÇÃO DE COMPATIBILIDADE ---
# O pickle está procurando um atributo 'dtype' que não existe na versão do sklearn.
# Adicionar manualmente à classe para evitar o erro.
if not hasattr(LabelEncoder, 'dtype'):
    LabelEncoder.dtype = None
# -----------------------------------

# Caminho do arquivo
pkl_path = 'models/label_encoders_v7.pkl'

print(f"📂 Tentando carregar: {pkl_path}")

try:
    # Tenta carregar com pickle padrão
    with open(pkl_path, 'rb') as f:
        encoders = pickle.load(f)
except Exception as e:
    print(f"⚠️ Erro inicial: {e}")
    print("🔄 Tentando método alternativo com joblib...")
    try:
        import joblib
        encoders = joblib.load(pkl_path)
    except ImportError:
        print("❌ Joblib não instalado. Instale com: pip install joblib")
        sys.exit(1)
    except Exception as e2:
        print(f"❌ Falha crítica ao carregar: {e2}")
        sys.exit(1)

# Extrair aeroportos válidos
try:
    if isinstance(encoders, dict) and 'Origin' in encoders:
        # Pega as classes do encoder de Origem
        valid_airports = sorted(encoders['Origin'].classes_.tolist())

        # Salvar em JSON
        airports_data = {
            "valid_airports": valid_airports,
            "total_count": len(valid_airports),
            "last_updated": "2025-12-18",
            "source": "label_encoders_v7.pkl",
            "note": "Lista de 362 códigos IATA únicos usados no treinamento do modelo"}

        output_path = 'docs/valid_airports.json'
        with open(output_path, 'w') as f:
            json.dump(airports_data, f, indent=2)

        print(f"✅ Sucesso! {len(valid_airports)} aeroportos exportados.")
        print(f"📄 Arquivo salvo em: {output_path}")
        print(f"📋 Primeiros 10: {valid_airports[:10]}")
    else:
        print(
            "❌ O arquivo carregou, mas a estrutura não é a esperada "
            "(não encontrou a chave 'Origin')."
        )
        print(f"Conteúdo encontrado: {type(encoders)}")
except Exception as e:
    print(f"❌ Erro ao processar os dados: {e}")
