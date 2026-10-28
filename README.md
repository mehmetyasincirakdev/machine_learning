# Machine Learning Examples

Python ile diyabet verisi analizi ve sınıflandırması ile reklam harcamalarından satış tahmini örnekleri.

## Proje yapısı

```text
.
├── data/
│   ├── raw/                  # Yerel CSV dosyaları (Git'e eklenmez)
│   └── README.md
├── src/
│   └── machine_learning/
│       ├── __init__.py
│       ├── diabetes_analysis.py
│       ├── diabetes_pipeline.py
│       └── linear_regression.py
├── tests/
│   ├── test_diabetes_pipeline.py
│   └── test_linear_regression.py
├── .gitignore
└── pyproject.toml
```

## Kurulum

Python 3.10 veya üzeri gerekir.

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Veri dosyaları

Veri setleri lisans ve kaynak bilgileri korunabilsin diye depoya eklenmez. CSV dosyalarını `data/raw/` içine yerleştirin. Beklenen sütunlar için [data/README.md](data/README.md) dosyasına bakın. Komutlarda dosya yolunu açıkça belirtin.

## Kullanım

Komutları depo kök dizininde çalıştırın:

```bash
python -m machine_learning.diabetes_analysis --data data/raw/diabetes.csv
python -m machine_learning.diabetes_pipeline --data data/raw/diabetes.csv
python -m machine_learning.linear_regression --data data/raw/advertising.csv
```

Analiz ve doğrusal regresyon çizimlerini kapatmak için `--no-plots` ekleyin. Sınıflandırma komutu varsayılan olarak stratified train/test ayrımı kullanır ve accuracy, precision, recall, F1 ve ROC AUC metriklerini raporlar.

## Testler

```bash
python -m pytest
```
