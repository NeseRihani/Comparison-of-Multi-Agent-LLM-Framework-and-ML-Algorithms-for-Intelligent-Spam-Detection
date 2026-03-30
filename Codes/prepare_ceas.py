"""
prepare_and_clean_ceas.py
--------------------------
CEAS_08 dataseti için veri hazırlama ve akademik ön işleme adımları.

Orijinal CEAS_08.csv durumu:
  - Sütunlar : sender, receiver, date, subject, body, label, urls
  - label     : 0 (ham), 1 (spam), boş/NaN (etiketlenmemiş → silinecek)
  - urls      : 0 (emailde URL yok), 1 (emailde en az bir URL var) — binary feature
                Kaynak: Champa et al. IEEE ICMI 2024 — "a value of 1 indicates
                the presence of URL(s) in the email body, whereas 0 indicates
                their absence."

Yapılan işlemler:
  Adım 1 → Label normalize    : 0→ham, 1→spam, boş→sil
  Adım 2 → Boş body temizleme : NaN veya boş string → sil
  
  Adım 3 → Boş subject doldur : NaN → "No Subject"
  Adım 4 → Boş sender doldur  : NaN → "unknown"
  Adım 5 → Boş date doldur    : NaN → "unknown"
  Adım 6 → URL feature zenginleştir:
            
            - Body'den URL regex ile çekilir → url_list sütununa eklenir
            - Agent hem "URL var mı" hem "URL metni nedir" bilgisine sahip olur

Çıktı sütunları:
  sender | receiver | date | subject | body | label | has_url | url_list

Çıktı dosyaları:
  cleaned_ceas.csv     → temizlenmiş, kullanıma hazır dataset
  cleaning_report_ceas.txt  → adım adım temizlik raporu
"""

import pandas as pd
import re

# ── DOSYA AYARLARI ───────────────────────────────────────────
INPUT_FILE  = "CEAS_08.csv"
OUTPUT_FILE = "cleaned_ceas.csv"
REPORT_FILE = "cleaning_report_ceas.txt"

# URL regex — body içinden gerçek URL metinlerini çeker
URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+!*(),]|(?:%[0-9a-fA-F]{2}))+'
)


def extract_urls_from_body(body_text: str) -> str:
    """Body metninden URL'leri çeker, pipe ile ayrılmış string döner."""
    if not isinstance(body_text, str):
        return ""
    found = URL_PATTERN.findall(body_text)
    return " | ".join(found) if found else ""


def prepare_and_clean_ceas(input_file: str, output_file: str, report_file: str):

    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log("=" * 60)
    log("  CEAS_08 VERİ HAZIRLAMA VE TEMİZLEME RAPORU")
    log("=" * 60)

    # ── YÜKLE ────────────────────────────────────────────────
    try:
        df = pd.read_csv(input_file, encoding="latin-1", on_bad_lines="skip")
    except Exception as e:
        log(f"\n HATA: Dosya okunamadı: {e}")
        raise

    initial_count = len(df)
    log(f"\n Orijinal dosya       : {input_file}")
    log(f"   Yüklenen satır       : {initial_count:,}")
    log(f"   Sütunlar             : {list(df.columns)}")
    log(f"\n── ORİJİNAL LABEL DAĞILIMI ─────────────────────────")
    log(f"   label=1  (spam)      : {(df['label']==1).sum():,}")
    log(f"   label=0  (ham)       : {(df['label']==0).sum():,}")
    log(f"   label=boş/NaN        : {df['label'].isna().sum():,}")
    log(f"\n── ORİJİNAL URL BINARY FEATURE ─────────────────────")
    log(f"   urls=1 (URL var)     : {(df['urls']==1).sum():,}")
    log(f"   urls=0 (URL yok)     : {(df['urls']==0).sum():,}")
    log(f"   urls=boş/NaN         : {df['urls'].isna().sum():,}")
    log(f"\n── ADIM ADIM TEMİZLİK ──────────────────────────────")

    # ── ADIM 1: LABEL NORMALIZE ──────────────────────────────
    before = len(df)
    df["label"] = df["label"].map({1: "spam", 0: "ham"})
    df = df[df["label"].isin(["spam", "ham"])]
    removed = before - len(df)
    log(f"[Adım 1] Label normalize (0→ham, 1→spam, boş→sil)  : "
        f"{removed:,} satır silindi  →  {len(df):,} kaldı")

    # ── ADIM 2: BOŞ BODY TEMİZLEME ───────────────────────────
    before = len(df)
    df = df.dropna(subset=["body"])
    df = df[df["body"].astype(str).str.strip() != ""]
    df = df[df["body"].astype(str).str.strip() != "nan"]
    removed = before - len(df)
    log(f"[Adım 2] Boş/NaN body temizlendi                   : "
        f"{removed:,} satır silindi  →  {len(df):,} kaldı")


    # ── ADIM 3: BOŞ SUBJECT DOLDUR ───────────────────────────
    n_null = df["subject"].isna().sum()
    df["subject"] = df["subject"].fillna("No Subject")
    df["subject"] = df["subject"].apply(
        lambda x: "No Subject" if str(x).strip() in ("", "nan") else x)
    log(f"[Adım 4] Boş subject  → 'No Subject'                : "
        f"{n_null:,} satır dolduruldu")

    # ── ADIM 4: BOŞ SENDER DOLDUR ────────────────────────────
    n_null = df["sender"].isna().sum()
    df["sender"] = df["sender"].fillna("unknown")
    df["sender"] = df["sender"].apply(
        lambda x: "unknown" if str(x).strip() in ("", "nan") else x)
    log(f"[Adım 5] Boş sender   → 'unknown'                   : "
        f"{n_null:,} satır dolduruldu")

    # ── ADIM 5: BOŞ DATE DOLDUR ──────────────────────────────
    n_null = df["date"].isna().sum()
    df["date"] = df["date"].fillna("unknown")
    df["date"] = df["date"].apply(
        lambda x: "unknown" if str(x).strip() in ("", "nan") else x)
    log(f"[Adım 6] Boş date     → 'unknown'                   : "
        f"{n_null:,} satır dolduruldu")

    # ── ADIM 6: URL FEATURE ZENGİNLEŞTİRME ──────────────────
    

    # 7b: urls=True olan satırlarda body'den gerçek URL metinlerini çek
    df["url_list"] = df.apply(
        lambda row: extract_urls_from_body(str(row["body"])) if row["urls"] == 1 else "",
        axis=1
    )

    
    

    url_present   = (df["urls"] == 1).sum()
    url_extracted = (df["url_list"] != "").sum()
    log(f"[Adım 7] URL feature zenginleştirildi               :")
    log(f"         urls=1  (binary urls=1)                    : {url_present:,} email")
    log(f"         Body'den URL metni çıkarılan               : {url_extracted:,} email")
    log(f"         urls=1 ama metin bulunamayan               : "
        f"{url_present - url_extracted:,} email  "
        f"(URL header'da olabilir)")

    # ── ÖZET ─────────────────────────────────────────────────
    final_count   = len(df)
    total_removed = initial_count - final_count
    spam_count    = (df["label"] == "spam").sum()
    ham_count     = (df["label"] == "ham").sum()

    log(f"\n{'='*60}")
    log(f"  TEMİZLİK ÖZETİ")
    log(f"{'='*60}")
    log(f"  Orijinal satır sayısı   : {initial_count:,}")
    log(f"  Toplam silinen          : {total_removed:,}  "
        f"(%{total_removed/initial_count*100:.2f})")
    log(f"  Son dataset             : {final_count:,} email")
    log(f"    Spam                  : {spam_count:,}  "
        f"(%{spam_count/final_count*100:.1f})")
    log(f"    Ham                   : {ham_count:,}  "
        f"(%{ham_count/final_count*100:.1f})")
    log(f"\n  Çıktı sütunları         : {list(df.columns)}")
    log(f"\n  URL özellik özeti:")
    log(f"    Spam'de URL var       : "
        f"{df[df['label']=='spam']['urls'].sum():,} / {spam_count:,} spam")
    log(f"    Ham'de URL var        : "
        f"{df[df['label']=='ham']['urls'].sum():,} / {ham_count:,} ham")
    log(f"\n Kaydedildi            : {output_file}")
    log(f" Rapor                 : {report_file}")
    log("=" * 60)

    # ── KAYDET ───────────────────────────────────────────────
    df = df.reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding="utf-8")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return df


if __name__ == "__main__":
    prepare_and_clean_ceas(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)