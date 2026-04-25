# calibration_analysis.py (Çok Sınıflı Problem için Düzeltilmiş Versiyon)
# Modelin tahmin olasılıklarının ne kadar 'kalibre' olduğunu analiz eder.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings

# Matplotlib'in bir ekran kullanmamasını sağlar (sunucu ortamları için)
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", category=UserWarning)


def calculate_ece(y_true, y_proba, n_bins=15):
    """Expected Calibration Error (ECE) hesaplar."""
    y_pred_conf = np.max(y_proba, axis=1)
    y_pred_class = np.argmax(y_proba, axis=1)
    
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        in_bin = (y_pred_conf > bin_limits[i]) & (y_pred_conf <= bin_limits[i+1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == y_pred_class[in_bin])
            avg_confidence_in_bin = np.mean(y_pred_conf[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# --- ANA KOD ---
# Tahminleri yükle
try:
    data = np.load('test_predictions.npz')
except FileNotFoundError:
    print("HATA: 'test_predictions.npz' dosyası bulunamadı. Lütfen önce 'generate_predictions.py' çalıştırın.")
    exit()

y_true = data['y_true']
y_proba = data['y_proba']
num_classes = y_proba.shape[1]
target_names = [f'{i+1} Yıldız' for i in range(num_classes)]

# --- Brier Skoru ve ECE Hesaplama ---
# Brier skoru için y_true'yu one-hot encoding'e çevirmek gerekir.
y_true_one_hot = np.eye(num_classes)[y_true]
brier_score = brier_score_loss(y_true_one_hot.ravel(), y_proba.ravel())
print(f"Genel Brier Skoru (düşük olması daha iyi): {brier_score:.4f}")

ece = calculate_ece(y_true, y_proba)
print(f"Beklenen Kalibrasyon Hatası (ECE) (düşük olması daha iyi): {ece:.4f}")

# --- Güvenilirlik Diyagramı Çizimi (Çok Sınıflı) ---
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Mükemmel Kalibrasyon')

# Her sınıf için ayrı ayrı kalibrasyon eğrisi çiz
for i in range(num_classes):
    # Sınıf i'yi "pozitif" sınıf olarak kabul et
    y_true_binary = (y_true == i).astype(int)
    prob_true, prob_pred = calibration_curve(y_true_binary, y_proba[:, i], n_bins=15)
    plt.plot(prob_pred, prob_true, marker='.', label=target_names[i])

plt.xlabel('Ortalama Tahmin Güveni (Confidence)', fontsize=12)
plt.ylabel('Doğruluk Oranı (Accuracy)', fontsize=12)
plt.title('Sınıf Bazında Güvenilirlik Diyagramı', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('reliability_diagram_multiclass.png', dpi=300, bbox_inches='tight')
print("\n'reliability_diagram_multiclass.png' dosyası oluşturuldu.")