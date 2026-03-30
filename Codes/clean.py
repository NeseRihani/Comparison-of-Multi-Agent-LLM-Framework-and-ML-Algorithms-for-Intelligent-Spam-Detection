"""
clean_dataset.py
----------------
Birleştirilmiş spam dataset için akademik ön işleme adımları:
1. Duplicate temizleme (body bazlı)
2. Boş / çok kısa body temizleme
3. Geçersiz label temizleme
4. Temizlik raporu
"""

import pandas as pd

INPUT_FILE  = "merged_dataset_spam.csv"
OUTPUT_FILE = "cleaned_dataset.csv"
REPORT_FILE = "cleaning_report.txt"




def clean_dataset(input_file: str, output_file: str, report_file: str):
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log("=" * 60)
    log("  VERİ ÖN İŞLEME RAPORU")
    log("=" * 60)

    # ── YÜKLE ────────────────────────────────────────────────
    df = pd.read_csv(input_file, encoding="utf-8", low_memory=False)
    initial_count = len(df)
    log(f"\n Yüklendi          : {initial_count:,} satır")
    log(f"   Sütunlar          : {list(df.columns)}")
    log(f"   Spam              : {(df['label']=='spam').sum():,}")
    log(f"   Ham               : {(df['label']=='ham').sum():,}")

    # ── ADIM 1: LABEL TEMİZLEME ──────────────────────────────
    before = len(df)
    df = df[df["label"].isin(["spam", "ham"])]
    removed = before - len(df)
    log(f"\n[Adım 1] Geçersiz label temizlendi  : {removed:,} satır silindi")

    # ── ADIM 2: BODY ZORUNLU ─────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["body"])
    df = df[df["body"].astype(str).str.strip() != ""]
    df = df[df["body"].astype(str).str.strip() != "nan"]
    removed = before - len(df)
    log(f"[Adım 2] Boş body temizlendi        : {removed:,} satır silindi")

   
    # ── ADIM 4: BODY BAZLI DUPLICATE ─────────────────────────
    before = len(df)
    df["body_clean"] = (
        df["body"].astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.drop_duplicates(subset=["body_clean"])
    df = df.drop(columns=["body_clean"])
    removed = before - len(df)
    log(f"[Adım 4] Body duplicate temizlendi  : {removed:,} satır silindi")

   

    # ── ADIM 6: SUBJECT DOLDUR ───────────────────────────────
    before_null = df["subject"].isna().sum()
    df["subject"] = df["subject"].fillna("No Subject")
    df["subject"] = df["subject"].replace("", "No Subject")
    log(f"[Adım 6] Boş subject dolduruldu     : {before_null:,} satır → 'No Subject'")

    # ── ADIM 7: SENDER DOLDUR ────────────────────────────────
    before_null = df["sender"].isna().sum()
    df["sender"] = df["sender"].fillna("unknown")
    df["sender"] = df["sender"].replace("", "unknown")
    log(f"[Adım 7] Boş sender dolduruldu      : {before_null:,} satır → 'unknown'")

    # ── ADIM 8: DATE DOLDUR ──────────────────────────────────
    before_null = df["date"].isna().sum()
    df["date"] = df["date"].fillna("unknown")
    df["date"] = df["date"].replace("", "unknown")
    log(f"[Adım 8] Boş date dolduruldu        : {before_null:,} satır → 'unknown'")

    # ── ADIM 9: URLS DOLDUR ──────────────────────────────────
    df["urls"] = df["urls"].fillna("")
    df["urls"] = df["urls"].replace("nan", "")
    log(f"[Adım 9] Boş urls boş string yapıldı")

    # ── ÖZET ─────────────────────────────────────────────────
    final_count = len(df)
    total_removed = initial_count - final_count

    log(f"\n{'='*60}")
    log(f"  TEMİZLİK ÖZETİ")
    log(f"{'='*60}")
    log(f"Başlangıç            : {initial_count:,} satır")
    log(f"Toplam silinen       : {total_removed:,} satır (%{total_removed/initial_count*100:.2f})")
    log(f"Son dataset          : {final_count:,} satır")
    log(f"  Spam               : {(df['label']=='spam').sum():,} (%{(df['label']=='spam').sum()/final_count*100:.1f})")
    log(f"  Ham                : {(df['label']=='ham').sum():,} (%{(df['label']=='ham').sum()/final_count*100:.1f})")

    log(f"\n── KAYNAK DAĞILIMI ──────────────────────────────────")
    for src, grp in df.groupby("source"):
        s = (grp["label"] == "spam").sum()
        h = (grp["label"] == "ham").sum()
        log(f"  {src:20s}: {len(grp):6,} email  ({s:,} spam / {h:,} ham)")

    log(f"\n💾 Kaydedildi: {output_file}")
    log(f"📄 Rapor     : {report_file}")

    # ── KAYDET ───────────────────────────────────────────────
    df = df.reset_index(drop=True)
    df.to_csv(output_file, index=False, encoding="utf-8")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return df


if __name__ == "__main__":
    clean_dataset(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)