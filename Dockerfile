# 1. Base (CACHED)
FROM python:3.11-slim

# 2. Dépendances système + CURL pour Ollama
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Installation des packages Python (CACHED)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- LES NOUVELLES ÉTAPES ---

# 4. Installation d'Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 5. Téléchargement du modèle
# On utilise le chemin complet et on définit le répertoire des modèles
ENV OLLAMA_MODELS=/root/.ollama/models
RUN (ollama serve &) && sleep 10 && ollama pull gemma:2b

# 6. Copie du code
COPY . .

# 7. Lancement (On doit lancer le serveur Ollama AVANT le script Python)
CMD ollama serve & python moodmate_local.py