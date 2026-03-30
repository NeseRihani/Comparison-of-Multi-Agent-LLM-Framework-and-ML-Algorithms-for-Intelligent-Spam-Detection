"""
llm_analysis.py
----------------
results_checkpoint.csv dosyasındaki LLM (Multi-Agent) sonuçlarını analiz eder.
ML karşılaştırmasıyla aynı formatı kullanır.

Yapılan analizler:
  1. Genel performans (tüm mevcut veri)
  2. Artırımsal analiz — ilk 100, 200, 500, 1000, tüm veri
     (ML ile aynı veri noktaları → doğrudan karşılaştırılabilir)
  3. URL var/yok breakdown
  4. Confusion matrix

Çıktılar:
  llm_results.csv    → ML ile aynı format, birleştirmeye hazır
  llm_report.txt     → detaylı metin raporu
"""

import pandas as pd
import numpy as np

# ── AYARLAR ──────────────────────────────────────────────────
RESULTS_CKPT    = "results_checkpoint.csv"
LLM_RESULTS_CSV = "llm_results_analysis.csv"
LLM_REPORT_FILE = "llm_report_analysis.txt"

# ML ile aynı artırımsal noktalar
INCREMENTAL_SIZES = [100, 200, 500, 1000,3914]


# ── METRIK HESAPLAMA ─────────────────────────────────────────
def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    """Bir DataFrame dilimi için tüm metrikleri hesapla."""
    if len(df) == 0:
        return {}

    tp = len(df[(df["True_Label"] == "spam") & (df["Prediction"] == "spam")])
    tn = len(df[(df["True_Label"] == "ham")  & (df["Prediction"] == "ham")])
    fp = len(df[(df["True_Label"] == "ham")  & (df["Prediction"] == "spam")])
    fn = len(df[(df["True_Label"] == "spam") & (df["Prediction"] == "ham")])

    acc  = (tp + tn) / len(df) * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "Model":      "Multi-Agent LLM (Llama 3.3 70B)",
        "Subset":     label,
        "Test_Size":  len(df),
        "Spam_Count": int((df["True_Label"] == "spam").sum()),
        "Ham_Count":  int((df["True_Label"] == "ham").sum()),
        "Accuracy":   round(acc,  2),
        "Precision":  round(prec, 2),
        "Recall":     round(rec,  2),
        "F1":         round(f1,   2),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


# ── ANA FONKSİYON ────────────────────────────────────────────
def run_llm_analysis():
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log("=" * 65)
    log("  LLM SONUÇ ANALİZİ — Multi-Agent (Llama 3.3 70B)")
    log("  Veri kaynağı: results_checkpoint.csv")
    log("=" * 65)

    # ── VERİ YÜKLE ───────────────────────────────────────────
    try:
        df = pd.read_csv(RESULTS_CKPT)
    except FileNotFoundError:
        log(f"\n {RESULTS_CKPT} bulunamadı!")
        log("   Agent çalışması tamamlanmadan bu script çalıştırılamaz.")
        return

    # Tip düzeltmeleri
    df["Email_ID"]  = pd.to_numeric(df["Email_ID"],  errors="coerce")
    df["Correct"]   = df["Correct"].astype(bool)
    df["urls"]      = pd.to_numeric(df["urls"],      errors="coerce").fillna(0).astype(int)
    df["True_Label"]  = df["True_Label"].astype(str).str.strip().str.lower()
    df["Prediction"]  = df["Prediction"].astype(str).str.strip().str.lower()

    # Sırala
    df = df.sort_values("Email_ID").reset_index(drop=True)

    total = len(df)
    spam_total = (df["True_Label"] == "spam").sum()
    ham_total  = (df["True_Label"] == "ham").sum()

    log(f"\nMevcut checkpoint: {total} email analiz edilmiş")
    log(f"  Spam : {spam_total} (%{spam_total/total*100:.1f})")
    log(f"  Ham  : {ham_total}  (%{ham_total/total*100:.1f})")

    all_results = []

    # ── 1. GENEL PERFORMANS ───────────────────────────────────
    log(f"\n{'─'*65}")
    log(f"  GENEL PERFORMANS (n={total})")
    log(f"{'─'*65}")

    overall = compute_metrics(df, label=f"Tüm veri (n={total})")
    log(f"  Accuracy  : %{overall['Accuracy']:.2f}")
    log(f"  Precision : %{overall['Precision']:.2f}")
    log(f"  Recall    : %{overall['Recall']:.2f}")
    log(f"  F1 Score  : %{overall['F1']:.2f}")
    log(f"\n  Confusion Matrix:")
    log(f"    TP (spam→spam) : {overall['TP']:4d}   FN (spam→ham) : {overall['FN']:4d}")
    log(f"    FP (ham→spam)  : {overall['FP']:4d}   TN (ham→ham)  : {overall['TN']:4d}")

    all_results.append(overall)

    # ── 2. ARTIRIMSal ANALİZ ─────────────────────────────────
    log(f"\n{'─'*65}")
    log(f"  ARTIRIMSal ANALİZ (ML ile aynı noktalar)")
    log(f"{'─'*65}")
    log(f"  {'Boyut':>8} {'Spam':>6} {'Ham':>6} "
        f"{'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7}")
    log(f"  {'─'*58}")

    sizes_to_analyze = [s for s in INCREMENTAL_SIZES if s <= total]
    if total not in sizes_to_analyze:
        sizes_to_analyze.append(total)

    for n in sizes_to_analyze:
        # İlk n email'i al (sıralı)
        subset = df.iloc[:n]
        m = compute_metrics(subset, label=f"İlk {n} email")
        all_results.append(m)
        log(f"  {n:>8}  {m['Spam_Count']:>5}  {m['Ham_Count']:>5}"
            f"  %{m['Accuracy']:>5.1f}  %{m['Precision']:>5.1f}"
            f"  %{m['Recall']:>5.1f}  %{m['F1']:>5.1f}")

    

    # ── 4. THRESHOLD BREAKDOWN ────────────────────────────────
    log(f"\n{'─'*65}")
    log(f"  THRESHOLD DAĞILIMI")
    log(f"{'─'*65}")

    if "Threshold" in df.columns:
        for thresh in sorted(df["Threshold"].unique()):
            sub = df[df["Threshold"] == thresh]
            m   = compute_metrics(sub, label=f"Threshold={thresh}")
            log(f"  Eşik={thresh} ({len(sub):4d} email): "
                f"Acc:%{m['Accuracy']:.1f} "
                f"Pre:%{m['Precision']:.1f} "
                f"Rec:%{m['Recall']:.1f} "
                f"F1:%{m['F1']:.1f}")

    # ── 5. HATA ANALİZİ ──────────────────────────────────────
    log(f"\n{'─'*65}")
    log(f"  HATA ANALİZİ")
    log(f"{'─'*65}")

    wrong = df[~df["Correct"]]
    fn_df = df[(df["True_Label"] == "spam") & (df["Prediction"] == "ham")]
    fp_df = df[(df["True_Label"] == "ham")  & (df["Prediction"] == "spam")]

    log(f"  Toplam hata      : {len(wrong)} / {total} (%{len(wrong)/total*100:.1f})")
    log(f"  False Negative   : {len(fn_df)} (spam → ham kaçırdı)")
    log(f"    URL olan FN    : {(fn_df['urls']==1).sum()}")
    log(f"    URL yok FN     : {(fn_df['urls']==0).sum()}")
    log(f"  False Positive   : {len(fp_df)} (ham → spam dedi)")
    log(f"    URL olan FP    : {(fp_df['urls']==1).sum()}")
    log(f"    URL yok FP     : {(fp_df['urls']==0).sum()}")

    # ── KAYDET ────────────────────────────────────────────────
    df_out = pd.DataFrame(all_results)
    df_out.to_csv(LLM_RESULTS_CSV, index=False)

    log(f"\n{'='*65}")
    log(f" LLM sonuçları kaydedildi : {LLM_RESULTS_CSV}")
    log(f" Rapor kaydedildi         : {LLM_REPORT_FILE}")
    log(f" Tamamlandı! ({total} email analiz edildi)")
    log(f"{'='*65}")

    with open(LLM_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return df_out


if __name__ == "__main__":
    run_llm_analysis()