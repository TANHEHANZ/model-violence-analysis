import re
import torch
from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Cargar el modelo de traducción (español a inglés)
model_name = 'Helsinki-NLP/opus-mt-es-en'
translation_model = MarianMTModel.from_pretrained(model_name)
translation_tokenizer = MarianTokenizer.from_pretrained(model_name)

# Cargar el modelo de clasificación RoBERTa
roberta_tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
roberta_model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

app = Flask(__name__)

def translate_text(text, src_lang="es", tgt_lang="en"):
    # Tokenizar y traducir el texto
    translated = translation_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_output = translation_model.generate(**translated)
    translated_text = translation_tokenizer.decode(translated_output[0], skip_special_tokens=True)
    return translated_text

def predict_toxicity(text):
    # Traducir el texto del español al inglés
    translated_text = translate_text(text)
    
    # Descomponer el texto en frases usando los puntos y comas como delimitadores
    sentences = re.split(r'(?<=\.|\?)\s+|(?<=,)\s+', translated_text)  # Dividir por punto o coma

    toxic_words = []
    
    # Evaluar cada frase
    for sentence in sentences:
        inputs = roberta_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Obtener la salida del modelo de toxicidad para cada frase
        output = roberta_model(**inputs)
        logits = output.logits
        
        # Aplicar softmax para obtener probabilidades
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Convertir las probabilidades a tensor y obtener la clase predicha para la frase
        probabilities = probabilities.squeeze(0)  # Eliminar dimensiones innecesarias
        predicted_class_idx = torch.argmax(probabilities).item()
        
        # Si la clase es "tóxica", agregamos la frase a la lista
        if predicted_class_idx == 1:
            toxic_words.append(sentence)  # Agregar la frase tóxica a la lista
    
    # Obtener las probabilidades finales del texto completo
    probabilities = probabilities.detach().cpu().numpy()  # Convertir el tensor a numpy
    probabilities = probabilities.astype(float)
    
    # Determinar la clase predicha para el texto completo
    class_names = ["neutral", "toxic"]
    predicted_class_idx = torch.argmax(torch.tensor(probabilities)).item()  # Asegurarse de que esté en tensor

    return {
        "predicted_class": class_names[predicted_class_idx],
        "probabilities": {
            "neutral": round(probabilities[0], 4),
            "toxic": round(probabilities[1], 4)
        },
        "toxic_words": toxic_words  # Incluir las frases tóxicas
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto del cuerpo de la solicitud
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Predecir la toxicidad
    result = predict_toxicity(text)
    
    # Devolver el resultado como JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
