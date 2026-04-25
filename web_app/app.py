import os
import torch
import numpy as np
from flask import Flask, render_template, request, flash
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# --------------------------------------------------------------------------
# 1. FLASK UYGULAMASINI VE TEMEL AYARLARI YAPILANDIR
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = 'tezsunumu-icin-gizli-anahtar'  # Flash mesajlar\u0131 i�in gerekli

# Modelinizin bulundu\u011fu klas�r�n yolu
MODEL_PATH = "./model/checkpoint-141220"

# --------------------------------------------------------------------------
# 2. MODEL\u0130 VE TOKENIZER'I SADECE B\u0130R KEZ Y�KLE
# Bu b�l�m uygulama ba\u015flad\u0131\u011f\u0131nda sadece bir kez �al\u0131\u015f\u0131r.
# --------------------------------------------------------------------------
try:
    print("Y\u0131ld\u0131zSezar V22 modeli y�kleniyor... Bu i\u015flem birka� saniye s�rebilir.")
    # Cihaz\u0131 belirle (varsa GPU, yoksa CPU kullan)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer ve Modeli y�kle
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Modeli belirlenen cihaza ta\u015f\u0131 ve evaluation moduna al
    model.to(device)
    model.eval()
    
    print(f"Model ba\u015far\u0131yla '{device}' �zerine y�klendi ve tahmin i�in haz\u0131r.")

except Exception as e:
    print(f"HATA: Model y�klenemedi. '{MODEL_PATH}' klas�r�n�n do\u011fru oldu\u011fundan emin olun.")
    print(f"Detayl\u0131 Hata: {e}")
    # Model y�klenemezse uygulamay\u0131 sonland\u0131r
    model = None
    tokenizer = None

# --------------------------------------------------------------------------
# 3. TAHM\u0130N FONKS\u0130YONU
# Gelen metni i\u015fleyip modelden tahmin al\u0131r.
# --------------------------------------------------------------------------
def predict_star_rating(text):
    if not model or not tokenizer:
        return None, None

    with torch.no_grad(): # Gradyan hesaplamas\u0131n\u0131 kapatarak h\u0131z\u0131 art\u0131r
        # Metni tokenize et ve tens�rlere d�n�\u015ft�r
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        )
        
        # Tokenize edilmi\u015f veriyi modelin bulundu\u011fu cihaza g�nder
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Modelden tahmin (logits) al
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Logitleri olas\u0131l\u0131klara d�n�\u015ft�r (softmax)
        probabilities = softmax(logits.cpu().numpy()[0])
        
        # En y�ksek olas\u0131l\u0131\u011fa sahip s\u0131n\u0131f\u0131n indeksini bul (0-4 aras\u0131)
        predicted_class_id = np.argmax(probabilities)
        
        # \u0130ndeksi ger�ek y\u0131ld\u0131z de\u011ferine d�n�\u015ft�r (1-5 aras\u0131)
        predicted_star = predicted_class_id + 1
        
        return predicted_star, probabilities.tolist()

# --------------------------------------------------------------------------
# 4. WEB SAYFASI ROUTE'LARI
# Kullan\u0131c\u0131n\u0131n taray\u0131c\u0131da g�rece\u011fi sayfalar\u0131 y�netir.
# --------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    """Ana sayfay\u0131 g�sterir."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Kullan\u0131c\u0131dan gelen formu i\u015fler ve tahmini yapar."""
    # Formdan verileri al
    review_text = request.form.get('review_text', '').strip()
    user_rating_str = request.form.get('user_rating')
    
    if not review_text:
        flash("L�tfen analiz i�in bir yorum metni girin.", "warning")
        return render_template('index.html')

    # Tahmini yap
    predicted_star, probabilities = predict_star_rating(review_text)
    
    if predicted_star is None:
        flash("Model y�klenirken bir hata olu\u015ftu. L�tfen terminali kontrol edin.", "danger")
        return render_template('index.html')
        
    # Kullan\u0131c\u0131n\u0131n se�ti\u011fi y\u0131ld\u0131z\u0131 integer'a �evir
    user_rating = int(user_rating_str) if user_rating_str else None
    
    # ***BURASI G�NCELLEND\u0130***
    # Sonu�lar\u0131 \u015fablona g�ndermek i�in haz\u0131rla
    results = {
        'predicted_star': predicted_star,
        # Her bir olas\u0131l\u0131\u011f\u0131 (prob) y\u0131ld\u0131z numaras\u0131 (i+1) ile e\u015fle\u015ftiriyoruz.
        'probability_data': [(i + 1, prob) for i, prob in enumerate(probabilities)],
        'is_match': user_rating is not None and user_rating == predicted_star
    }
    
    return render_template(
        'index.html', 
        results=results, 
        review_text=review_text, 
        user_rating=user_rating
    )

# --------------------------------------------------------------------------
# 5. UYGULAMAYI BA\u015eLAT
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # debug=True, geli\u015ftirme s\u0131ras\u0131nda de\u011fi\u015fiklikleri otomatik alg\u0131lar.
    # Sunumda veya canl\u0131da kullan\u0131rken debug=False yap\u0131n.
    app.run(debug=True, port=5000)