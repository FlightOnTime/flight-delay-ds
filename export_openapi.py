import yaml

# ajuste o import se o seu arquivo principal tiver outro nome
from app import app

schema = app.openapi()

with open("openapi.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(schema, f, sort_keys=False, allow_unicode=True)

print("✅ openapi.yaml gerado com sucesso!")
