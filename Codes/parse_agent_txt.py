"""
parse_agent_report.py
----------------------
analysis_report_ceas_groq.txt dosyasını okur, her email için
Content / Sender / URL agent kararlarını çıkarır.

Kontroller:
  1. Duplicate Email ID var mı?
  2. ID'ler ardışık mı (eksik var mı)?
  3. Parse edilemeyen blok var mı?

Çıktılar:
  agent_votes.csv         → her email için agent kararları
  agent_analysis.csv      → artırımsal analiz (100/200/500/1000/tüm veri)
  agent_parse_report.txt  → kontrol raporu
"""

import re
import pandas as pd
import numpy as np

# ── AYARLAR ──────────────────────────────────────────────────
TXT_FILE         = "analysis_report_ceas_groq.txt"
VOTES_CSV        = "agent_votes.csv"
ANALYSIS_CSV     = "agent_analysis.csv"
REPORT_FILE      = "agent_parse_report.txt"

INCREMENTAL_SIZES = [100, 200, 500, 1000]


# ── PARSE ────────────────────────────────────────────────────
def parse_txt(filepath: str) -> pd.DataFrame:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Her email bloğunu çek
    pattern = re.compile(
        r'\[Email ID:\s*(\d+)\].*?'
        r'Gerçek Etiket\s*:\s*(\w+).*?'
        r'Şef Kararı\s*:\s*(\w+)\s*\((\d+)/3 oy.*?eşik:(\d+)\).*?'
        r'Content Agent\s*:\s*(⚠ ŞÜPHELI|✓ GÜVENLİ).*?'
        r'Sender Agent\s*:\s*(⚠ ŞÜPHELI|✓ GÜVENLİ).*?'
        r'URL Agent\s*:\s*(⚠ ŞÜPHELI|✓ GÜVENLİ)',
        re.DOTALL
    )

    records = []
    for m in pattern.finditer(content):
        email_id, true_label, prediction, votes, threshold, \
            content_v, sender_v, url_v = m.groups()

        records.append({
            "Email_ID":      int(email_id),
            "True_Label":    true_label.lower(),
            "Prediction":    prediction.lower(),
            "Votes":         int(votes),
            "Threshold":     int(threshold),
            "Content_Vote":  1 if "ŞÜPHELI" in content_v else 0,
            "Sender_Vote":   1 if "ŞÜPHELI" in sender_v  else 0,
            "URL_Vote":      1 if "ŞÜPHELI" in url_v     else 0,
            "Correct":       true_label.lower() == prediction.lower(),
        })

    return pd.DataFrame(records)


# ── KONTROL ──────────────────────────────────────────────────
def validate(df: pd.DataFrame, report_lines: list):
    def log(msg):
        print(msg)
        report_lines.append(msg)

    log(f"\n── KONTROL RAPORU ──────────────────────────────────")
    log(f"  Parse edilen toplam email : {len(df)}")

    # 1. Duplicate
    dupes = df[df.duplicated("Email_ID", keep=False)]
    if len(dupes) > 0:
        log(f"   Duplicate ID'ler bulundu: {sorted(dupes['Email_ID'].unique().tolist())}")
    else:
        log(f"   Duplicate yok")

    # 2. Eksik ID
    if len(df) > 0:
        ids      = sorted(df["Email_ID"].tolist())
        expected = list(range(ids[0], ids[-1] + 1))
        missing  = sorted(set(expected) - set(ids))
        if missing:
            log(f"  ⚠  Eksik ID'ler ({len(missing)} adet): "
                f"{missing[:20]}{'...' if len(missing)>20 else ''}")
        else:
            log(f"  ✅ ID sırası ardışık, eksik yok ({ids[0]} → {ids[-1]})")

    # 3. Parse edilemeyen
    log(f"  ℹ  ID aralığı: {df['Email_ID'].min()} → {df['Email_ID'].max()}")
    log(f"  Spam: {(df['True_Label']=='spam').sum()} | "
        f"Ham: {(df['True_Label']=='ham').sum()}")


# ── AGENT BAZLI METRİKLER ────────────────────────────────────
def agent_metrics(df: pd.DataFrame, agent_col: str,
                  true_col: str = "True_Label") -> dict:
    """
    Tek ajanın kararlarına göre confusion matrix hesapla.
    Agent True → SPAM dedi, False → HAM dedi.
    """
    tp = ((df[agent_col] == 1) & (df[true_col] == "spam")).sum()
    tn = ((df[agent_col] == 0) & (df[true_col] == "ham")).sum()
    fp = ((df[agent_col] == 1) & (df[true_col] == "ham")).sum()
    fn = ((df[agent_col] == 0) & (df[true_col] == "spam")).sum()

    acc  = (tp + tn) / len(df) * 100 if len(df) > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "Accuracy":  round(acc,  2),
        "Precision": round(prec, 2),
        "Recall":    round(rec,  2),
        "F1":        round(f1,   2),
        "TP": int(tp), "FN": int(fn),
        "FP": int(fp), "TN": int(tn),
    }


# ── ARTIRIMSal ANALİZ ────────────────────────────────────────
def incremental_analysis(df: pd.DataFrame,
                          report_lines: list) -> pd.DataFrame:
    def log(msg):
        print(msg)
        report_lines.append(msg)

    total = len(df)
    sizes = [s for s in INCREMENTAL_SIZES if s <= total]
    if total not in sizes:
        sizes.append(total)

    agents = {
        "Content Agent": "Content_Vote",
        "Sender Agent":  "Sender_Vote",
        "URL Agent":     "URL_Vote",
        "Multi-Agent":   "Correct",   # şef kararı
    }

    all_rows = []

    log(f"\n{'='*70}")
    log(f"  ARTIRIMSal AGENT ANALİZİ")
    log(f"{'='*70}")

    for n in sizes:
        subset = df.iloc[:n]
        log(f"\n── n={n} email ({'─'*50})")
        log(f"  {'Agent':<18} {'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7} "
            f"{'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        log(f"  {'─'*65}")

        for agent_name, col in agents.items():
            if col == "Correct":
                # Multi-Agent: doğrudan Correct sütunundan al
                tp = ((subset["True_Label"]=="spam") & (subset["Prediction"]=="spam")).sum()
                tn = ((subset["True_Label"]=="ham")  & (subset["Prediction"]=="ham")).sum()
                fp = ((subset["True_Label"]=="ham")  & (subset["Prediction"]=="spam")).sum()
                fn = ((subset["True_Label"]=="spam") & (subset["Prediction"]=="ham")).sum()
                acc  = subset["Correct"].mean() * 100
                prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
                rec  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
                f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
                m = {"Accuracy": round(acc,2), "Precision": round(prec,2),
                     "Recall": round(rec,2), "F1": round(f1,2),
                     "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}
            else:
                m = agent_metrics(subset, col)

            log(f"  {agent_name:<18}"
                f"  %{m['Accuracy']:>4.1f}  %{m['Precision']:>4.1f}"
                f"  %{m['Recall']:>4.1f}  %{m['F1']:>4.1f}"
                f"  {m['TP']:>4}  {m['FP']:>4}  {m['FN']:>4}  {m['TN']:>4}")

            all_rows.append({
                "Agent":     agent_name,
                "N":         n,
                **m
            })

    return pd.DataFrame(all_rows)


# ── ANA ──────────────────────────────────────────────────────
def main():
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("=" * 70)
    log("  AGENT OY PARSE & ANALİZ")
    log(f"  Kaynak: {TXT_FILE}")
    log("=" * 70)

    # Parse
    log("\nTXT dosyası okunuyor...")
    df = parse_txt(TXT_FILE)

    if len(df) == 0:
        log("❌ Hiç kayıt parse edilemedi! TXT dosyasını kontrol et.")
        return

    log(f"✅ {len(df)} email parse edildi.")

    # Kontrol
    validate(df, report_lines)

    # Duplikatları kaldır (checkpoint düzeltmesi sonrası olabilir)
    before = len(df)
    df = df.drop_duplicates("Email_ID").sort_values("Email_ID").reset_index(drop=True)
    if len(df) < before:
        log(f"  ℹ  {before - len(df)} duplicate kaldırıldı, {len(df)} kaldı.")

    # Votes CSV kaydet
    df.to_csv(VOTES_CSV, index=False)
    log(f"\n Agent oyları kaydedildi: {VOTES_CSV}")

    # Artırımsal analiz
    df_analysis = incremental_analysis(df, report_lines)
    df_analysis.to_csv(ANALYSIS_CSV, index=False)
    log(f"\n Analiz tablosu kaydedildi: {ANALYSIS_CSV}")

    # Özet
    log(f"\n{'='*70}")
    log("  ÖZET — TÜM VERİDE EN İYİ F1")
    log(f"{'='*70}")
    total_n = df_analysis["N"].max()
    best = df_analysis[df_analysis["N"] == total_n]
    for _, row in best.iterrows():
        log(f"  {row['Agent']:<18} → F1: %{row['F1']:.1f}  "
            f"Acc: %{row['Accuracy']:.1f}  "
            f"Rec: %{row['Recall']:.1f}")

    # Rapor kaydet
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"\n Rapor kaydedildi: {REPORT_FILE}")
    log(" Tamamlandı!")


if __name__ == "__main__":
    main()