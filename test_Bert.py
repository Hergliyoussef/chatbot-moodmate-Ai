from transformers import pipeline
import os
# 1. Définir le chemin absolu du modèle
model_path = os.path.abspath("model/MoodMate_BERT_Model")

    # 3. Chargement avec le chemin absolu
try:
        classifier = pipeline(
            "text-classification", 
            model=model_path, 
            tokenizer=model_path
        )

        # Test
        test_phrase = "i fear to lose you mu friend  !"
        prediction = classifier(test_phrase)

        print(f"\nPhrase de test : {test_phrase}")
        print(f"Résultat de l'IA : {prediction}")
except Exception as e:
        print(f"❌ Une erreur est survenue pendant le chargement : {e}")