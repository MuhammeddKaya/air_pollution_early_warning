# PM2.5 Erken-Uyarı — Deney Betiği İskeleti (Yön 1)

Kalibre **olasılıksal PM2.5 aşım erken-uyarısı** için modüler bir Python paketi.
Proje planındaki (Yön 1) metodolojik hattı birebir kod yapısına çevirir.

> **Durum:** İskelet + eğitim döngüsü (`train_one`) ve naive taban dolduruldu; sentetik veriyle (lstm/gru × 4 UQ × full/corr) uçtan uca çalıştığı doğrulandı. Gerçek veri, TFT/PCMCI ve transfer hâlâ açık.
> Eski not: Metrikler, conformal (CQR), aşım olasılığı, veri hattı ve
> model tanımları **gerçek ve test edilmiş**tir. Eğitim döngüsü ve TFT/PCMCI
> entegrasyonu **doldurulacak** stub'lardır (aşağıda işaretli).

---

## 1. Kurulum

```bash
python -m venv venv && source venv/bin/activate
pip install numpy pandas torch openpyxl        # çekirdek
pip install tigramite pyEDM                     # (opsiyonel) nedensel katman
pip install pytorch-forecasting                 # (opsiyonel) TFT omurgası
```

Veri dosyasını (`last_dataset_Cok_Onemli.xlsx`) paket köküne koyun veya
`config.DataConfig.path` alanını güncelleyin.

## 2. Çalıştırma

**v4 (Aşama A, komut satırı):** `PYTHONPATH=. python -m pm25_early_warning.run_experiment --list-jobs` (81 iş); `--job N` tek iş; sonuçlar `results/runs/*.json`; birleştirme: `python aggregate.py results`. TRUBA için bkz. **TRUBA_YONERGE.md** ve `slurm/`.

Eski (tek süreç) kullanım:

```bash
python -m pm25_early_warning.run_experiment
```

Sonuçlar `results/results.json` içine yazılır (nokta + olasılıksal + olay + karar
metrikleri, her öznitelik × omurga × UQ kombinasyonu için).

### İlk çalıştırmada ne beklemeli?

Bu bir **iskelettir**. `uq.train_one` (eğitim döngüsü) henüz doldurulmadığı için
ilk çalıştırmada konsolda `[STUB] ...` satırları görürsünüz — bu **hata değil**,
beklenen davranıştır. Paketin uçtan uca çalışıp `results/results.json` üretmesi
için §4'teki `train_one` doldurulmalıdır.

> **torch kurulumu:** En büyük ve en yavaş kurulan bağımlılıktır. TRUBA/GPU için
> CUDA sürümünüze uygun tekerleği seçin (https://pytorch.org/get-started/locally/);
> yerel testte CPU sürümü yeterlidir.

## 3. Paketin plana eşlemesi

| Plan bölümü | Modül | Durum |
|---|---|---|
| §5 Veri, eşikler, bölme | `data.py`, `config.py` | ✅ gerçek |
| §6–7 Pipeline / omurga | `models.py` | ✅ LSTM/GRU · ⏳ TFT stub |
| §7.3 Belirsizlik niceleme | `uq.py` | ✅ ensemble/mc/kuantil/conformal · ✅ `train_one` · ✅ `naive_predict` |
| Aşım olasılığı P(y>τ) | `exceedance.py` | ✅ gerçek |
| §9 Değerlendirme metrikleri | `metrics.py` | ✅ gerçek + test edildi |
| §7.4 Nedensel katman | `causal.py` | ⏳ PCMCI/CCM stub |
| §8 Deney matrisi | `run_experiment.py` | ✅ akış (sentetik veriyle uçtan uca test edildi) |

## 4. Doldurulacaklar (TODO — öğrenciye)

1. ~~`uq.train_one`~~ ✅ yapıldı (Adam/RMSprop, early stopping, en iyi ağırlık geri yükleme). Eski madde: **`uq.train_one`** — eğitim döngüsü: DataLoader, optimizer (adam/rmsprop),
   kayıp seçimi (MSE / `models.pinball_loss` / `models.gaussian_nll`), early
   stopping (`patience`), en iyi ağırlıkların geri yüklenmesi. **En kritik parça.**
2. **`models.build_model`** (TFT/PatchTST dalı) — `pytorch-forecasting` veya
   `neuralforecast` ile bağlama; doğal kuantil çıktısını `uq.quantile_predict`
   sözleşmesine hizalama.
3. ~~`uq.naive_predict`~~ ✅ yapıldı; `run_experiment` henüz naive tabanı çağırmıyor (bağlanacak).
4. **`causal.discover_pcmci` / `ccm_strength`** — `tigramite` / `pyEDM`
   entegrasyonu ve hedef ebeveynlerinin çıkarılması; `run_experiment` içindeki
   `causal_parents` bununla beslenir.
5. **Transfer istasyonları** — `config.transfer_stations` doldurulup
   `run_experiment.main` sonunda tekrar koşturma.
6. **Şekiller** — `metrics.reliability_curve`, `cost_loss_curve`,
   `lead_time`, PIT değerleri zaten F3–F7 için gerekli diziyi üretir; çizim
   katmanı (matplotlib) eklenecek.

## 5. Tasarım notları

- **Bölme (v3):** `DataConfig.split_mode='blocked'` (varsayılan) — 30 günlük bloklar, k%7==2 val, k%7==1 test; pencerenin tüm aralığı aynı kümede (embargo). Kış ayları her kümede ≥350 satır. `'chrono'` seçeneği korundu (test = Tem–Ara; kış tepesi yok, train→test ortalama kayması 65→40).
- **Uyarı süresi (v3):** `metrics.lead_time_by_horizon` — olay başlangıcı için h∈{8,24,72} saat önceki olasılık ≥0.5 ise uyarıldı; süre = uyarı verilen en uzun ufuk; yakalanan/tümü ayrı raporlanır.
- **Olasılıksal metrikler (v3):** kuantil/conformal için CRPS≈2×pinball, PICP/MPIW (ham ve CQR) eklendi.
- **Dikkat:** DSÖ 15 µg/m³ eşiğinde aşım oranı ~%70 → 'nadir olay' değil; PR-AUC/BSS yorumunda base_rate'e bakın.

- **Birim tutarlılığı:** eğitim normalize birimde, ama **tüm eşik/metrik
  hesapları orijinal (µg/m³) birimde** yapılır — `Scaler.inverse_col` ile geri
  dönülür. Aşım olasılıkları bu yüzden anlamlıdır.
- **Öngörü temsili:** UQ yöntemleri ya `samples` (N,H,S) ya da
  `quantiles` (N,H,Q) döndürür; `exceedance` ve `metrics` ikisini de kabul eder.
- **Conformal (CQR):** `uq.cqr_calibrate` split-conformal düzeltmesini
  **validasyon** kümesinde hesaplar, teste uygular → dağılımdan bağımsız marjinal
  kapsama. Metodolojik vitrin budur.
- **Nadir olaylar:** olay metrikleri PR-AUC + Brier skill + eşik-duyarlı; sınıf
  dengesizliğine karşı planla uyumlu.

## 6. Dosya listesi

```
pm25_early_warning/
├── __init__.py
├── config.py          # konfigürasyon + deney matrisi
├── data.py            # yükleme, ön işleme, öznitelik seti, pencereleme, bölme
├── models.py          # LSTM/GRU + nokta/gaussian/kuantil başlıkları (+TFT stub)
├── uq.py              # ensemble, MC dropout, kuantil, conformal (CQR)
├── exceedance.py      # öngörü dağılımı → P(y>τ)
├── metrics.py         # nokta/olasılıksal/olay/karar metrikleri (+DM testi)
├── causal.py          # PCMCI / CCM (opsiyonel)
└── run_experiment.py  # uçtan uca akış
```
