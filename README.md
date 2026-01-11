Rapport de Projet : MoodMate Pro AI
Architecture Hybride Locale : Diagnostic DistilBERT & Génération Gemma-2B
technologies utulisee : Python NLP, Hugging Face, Ollama, Gradio, Google Colab, Gemma-2B, DistilBERT, kaggle
1. Introduction et Objectifs
● Vision : Développer un assistant de psychologie capable de diagnostiquer les émotions et de fournir des réponses empathiques en temps réel.
● Contrainte Majeure : Assurer une exécution 100% locale pour garantir la confidentialité des données sur une machine limitée à 8 Go de RAM.
2. Acquisition et Préparation des Données
● Source Kaggle : Utilisation d'un dataset de référence de 16 000 lignes labellisées par émotions, importé de Kaggle au format CSV (emotion_data.csv).
● Pré-traitement : Nettoyage du texte et préparation pour l'entraînement du modèle de classification.
3. Développement du Modèle de Diagnostic (NLP Classification)
● Modèle choisi : DistilBERT (Hugging Face), idéal pour les systèmes embarqués ou aux ressources limitées.
● Entraînement (Cloud) : Création du script train_bert.py et exécution sur Google Colab (GPU T4) pour traiter les 16 000 lignes.
● Intégration : Téléchargement des poids du modèle après entraînement et intégration dans le dossier local /model/MoodMate_BERT_Model.
● Validation : Tests effectués via le fichier test_Bert.py.
--> python test_bert.py
4. Évolution du Modèle de Réponse (Generative AI)
● Phase 1  (fichier json ) réponse statique : Utilisation initiale d'un fichier reponse.json: un script reponse.py pour des réponses pré-définies basées sur les émotions diagnostiquées.
● Phase 2  (DialoGPT) : Expérimentation avec le modèle de dialogue de Microsoft (versions small, medium et large).
● phase 3  (Gemma-2B) : Adoption du modèle de Google pour une meilleure finesse psychologique.
● Optimisation Ollama : Utilisation d'Ollama pour exécuter Gemma via une quantification 4-bit, indispensable pour libérer de la RAM.
● Modelfile : Création d'un "Prompt System" personnalisé pour définir le rôle de psychologue expert.
5. Défis Techniques et Solutions (Le Parcours de l'Ingénieur)
● La Panne de Streamlit : Suite à des instabilités de gestion de mémoire, transition stratégique vers Gradio pour l'interface utilisateur.
● Gestion du Stress RAM : Optimisation du parallélisme pour faire tourner BERT (Python) et Gemma (Ollama) simultanément sur 8 Go de RAM.
6. Résultats et Métriques Finales :
A. Matrice de Confusion : Analyse des erreurs de détection :
La matrice montre une forte concentration sur la diagonale, confirmant l'excellente précision du modèle pour les émotions principales:
B. Exemple de test finale : avec Gradio
→ python moodmate_locale.py