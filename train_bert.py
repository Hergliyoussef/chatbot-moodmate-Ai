from transformers import pipeline
import os

# Chemin vers ton modèle décompressé
model_path = "./model/MoodMate_BERT_Model"

print("⏳ Chargement du modèle...")
try:
    # On charge le classifier avec ton modèle local
    classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
    
    # Test sur une phrase simple
    test_phrase = "I am so happy that my model is finally working!"
    result = classifier(test_phrase)[0]
    
    print("-" * 30)
    print(f"Phrase de test : {test_phrase}")
    print(f"Émotion détectée : {result['label']}")
    print(f"Score de confiance : {round(result['score'] * 100, 2)}%")
    print("-" * 30)
    print("✅ Le modèle BERT est opérationnel !")

except Exception as e:
    print(f"❌ Erreur de chargement : {e}")