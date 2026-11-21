# TRUBA Koşu Yönergesi — Aşama A

Amaç: LSTM/GRU × {tam, korelasyon-seçili öznitelik} × 4 belirsizlik yöntemi × 5 tohum + naive taban.
Toplam **81 iş** (iş 1 = naive; 2–81 = modeller). Her iş kendi sonuç dosyasını yazar.

Tahmini aktif çalışma süresi: 1–2 gün (çoğu kuyrukta bekleme). Süre ve kota TRUBA hesabına göre değişir.

## Adım 1 — Ortam kurulumu (~yarım gün)
1. `pm25_project/` klasörünü ve `data/last_dataset_Cok_Onemli.xlsx` dosyasını TRUBA'ya kopyalayın.
2. Sanal ortam: `python -m venv venv && source venv/bin/activate`
3. GPU'ya uygun `torch` kurun (https://pytorch.org/get-started/locally/ — TRUBA'daki CUDA sürümüne göre),
   sonra: `pip install numpy pandas openpyxl`
4. Kontrol: `python -c "import torch; print(torch.cuda.is_available())"` → `True` beklenir (GPU düğümünde).

## Adım 2 — Kısa deneme (~1 saat)
1. `slurm/smoke.sbatch` içindeki `<HESAP_ADI>` ve `<GPU_KUYRUGU>` alanlarını doldurun.
2. `sbatch slurm/smoke.sbatch`
3. Bitince `results_smoke/runs/` altında `naive.json` ve `lstm_full_ensemble_s0.json` olmalı;
   `logs/smoke_*.out` içinde `[OK]` satırları görünmeli. `.ERROR.json` varsa içindeki `trace` alanını bana iletin.

## Adım 3 — Tam koşu
1. `slurm/run_array.sbatch` içindeki aynı iki alanı doldurun (gerekirse `--time` ve `%20` sınırını kota/kuyruğa göre ayarlayın).
2. `sbatch slurm/run_array.sbatch`
3. İlerleme: `ls results/runs | wc -l` (hedef 81). İş listesini görmek için:
   `PYTHONPATH=. python -m pm25_early_warning.run_experiment --list-jobs`
4. Kesilen/başarısız işler için aynı `sbatch` komutu tekrar gönderilir; sonucu olan işler `[SKIP]` ile atlanır.
   Belirli bir işi yeniden denemek için: `--array=17` gibi.
5. Hata dosyaları: `results/runs/*.ERROR.json`

## Adım 4 — Sonuçları toplayıp geri verme (~yarım saat)
1. `python aggregate.py results` → `results/summary_point.csv`, `summary_prob.csv`, `summary_event.csv`, `summary_lead.csv`
2. `results/` klasörünü (özellikle `runs/` ve `summary_*.csv`) ve `logs/` klasörünü sıkıştırıp iletin.

## Notlar
- `tft` ve `causal` öznitelik seti henüz yazılmadı (Aşama B); koşucu bunları reddeder.
- Veri dosyası zip'e dahil değildir; `data/` altına konmalıdır.
- Tek iş yaklaşık dakikalar mertebesindedir (GPU'da); ensemble işleri 5 model eğittiği için ~5 kat uzun sürer.
