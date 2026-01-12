import gradio as gr
import ollama
import torch
from transformers import pipeline
import os

# --- 1. CONFIGURATION DES CHEMINS ---
BERT_PATH = os.path.abspath("model/MoodMate_BERT_Model")

# --- 2. CHARGEMENT DU DIAGNOSTIC (BERT) ---
if not os.path.exists(BERT_PATH):
    print(f"❌ ERREUR : Le dossier {BERT_PATH} est introuvable !")
else:
    try:
        classifier = pipeline("text-classification", model=BERT_PATH)
        print("✅ Modèle BERT chargé localement avec succès.")
    except Exception as e:
        print(f"❌ Erreur de chargement BERT : {e}")

# --- 3. LOGIQUE DE TRAITEMENT HYBRIDE ---
def predict(message, history):
    # A. Analyse BERT (Sentiment Analysis)
    res_bert = classifier(message)[0]
    emotion = res_bert['label']
    score = res_bert['score']

    # B. Appel à Ollama (Gemma-2B local)
    prompt = f"L'utilisateur se sent {emotion}. Voici son message : {message}"
    
    try:
        # Utilisation de ton modèle personnalisé 'moodmate-ai' créé via Modelfile
        response = ollama.chat(model='moodmate-ai', messages=[
            {'role': 'user', 'content': prompt},
        ])
        bot_message = response['message']['content']
    except Exception as e:
        bot_message = f"Erreur de connexion à Ollama : {e}. Vérifie qu'Ollama tourne."

    # C. Statut de Cohérence
    status_ppl = "Optimisée via Ollama" 

    return bot_message, f"Émotion : {emotion}", f"{round(score*100, 2)}%", status_ppl

# --- 4. INTERFACE GRADIO (SANS BLOC DROIT) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# MoodMate Pro AI")
    gr.Markdown("Architecture: **DistilBERT** + **Ollama (Gemma-2B)**")

    # Interface de chat simplifiée
    chatbot = gr.ChatInterface(
        fn=predict,
        additional_outputs=[
            gr.Textbox(label="Analyse Émotionnelle"),
            gr.Textbox(label="Score de Confiance BERT"),
            gr.Textbox(label="Cohérence Système")
        ],
        examples=["I don't fixed my bug ,I am stressed about my exams tomororrow"],
    )
    
    gr.Info("Données 100% locales. Aucune fuite d'information vers le Cloud.")

# --- 5. LANCEMENT ---
if __name__ == "__main__":
    # share=False pour garantir la confidentialité locale (Cybersecurity posture)
    demo.launch(debug=True)