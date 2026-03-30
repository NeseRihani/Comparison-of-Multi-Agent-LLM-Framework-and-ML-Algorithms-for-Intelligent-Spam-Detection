"""
ml_comparison.py
-----------------
CEAS_08 dataseti üzerinde geleneksel ML modelleri ile artırımsal eğitim ve değerlendirme.

Artırımsal veri noktaları: 100 → 200 → 500 → 1000 email
Her noktada %80 train / %20 test ayrımı yapılır.

Kullanılan modeller (Champa et al. IEEE ICMI 2024 referans):
  - Naive Bayes
  - Logistic Regression
  - SGD Classifier
  - Random Forest
  - XGBoost 

Özellikler:
  - TF-IDF (subject + body, max 10.000 feature, bigram)
  - URL binary feature — Champa et al. (2024)
  - Sender domain features (free email, unknown, random-looking)

Çıktılar:
  ml_results.csv   → tüm model/veri kombinasyonu sonuçları
  ml_report.txt    → detaylı metin raporu
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import hstack, csr_matrix
import re
import time

# ── AYARLAR ──────────────────────────────────────────────────
DATASET_FILE       = "cleaned_ceas.csv"
ML_RESULTS_FILE    = "ml_results.csv"
ML_REPORT_FILE     = "ml_report.txt"

INCREMENTAL_SIZES = [100, 200, 500, 1000, 3915]
RANDOM_STATE       = 42


def build_text(df):
    subject = df["subject"].fillna("").astype(str)
    body    = df["body"].fillna("").astype(str).str[:500]
    return subject + " " + body


def build_sender_features(df):
    free_domains = {"gmail.com", "yahoo.com", "hotmail.com",
                    "outlook.com", "aol.com", "mail.com"}
    rows = []
    for sender in df["sender"].fillna("unknown").astype(str):
        domain     = sender.split("@")[-1].lower().strip(">").strip()
        is_free    = int(domain in free_domains)
        is_unknown = int(sender.strip().lower() in ("unknown", "", "nan"))
        is_random  = int(bool(re.search(r'\d{3,}', domain)) and
                         len(domain.split(".")[0]) > 8)
        rows.append([is_free, is_unknown, is_random])
    return np.array(rows, dtype=np.float32)


def get_models():
    models = {
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, C=1.0),
        "SGD Classifier": SGDClassifier(
            loss="hinge", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0)
    except ImportError:
        pass
    return models


def evaluate_model(model, X_train, X_test, y_train, y_test,
                   model_name, n_train):
    t0 = time.time()

    if model_name == "Naive Bayes":
        scaler  = MaxAbsScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    duration = round(time.time() - t0, 2)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_test, y_pred,    pos_label=1, zero_division=0)
    f1   = f1_score(y_test, y_pred,        pos_label=1, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=[1, 0])
    tp, fn, fp, tn = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])

    return {
        "Model":      model_name,
        "Train_Size": n_train,
        "Test_Size":  len(y_test),
        "Accuracy":   round(acc  * 100, 2),
        "Precision":  round(prec * 100, 2),
        "Recall":     round(rec  * 100, 2),
        "F1":         round(f1   * 100, 2),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "Duration_s": duration,
    }


def run_ml_comparison():
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    log("=" * 65)
    log("  ML KARŞILAŞTIRMA — CEAS_08 Dataseti")
    log("  Referans: Champa et al. (IEEE ICMI 2024)")
    log(f"  Artırımsal veri noktaları: {INCREMENTAL_SIZES}")
    log("=" * 65)

    df = pd.read_csv(DATASET_FILE)
    log(f"\nDataset: {len(df):,} email")
    log(f"  Spam: {(df['label']=='spam').sum():,}")
    log(f"  Ham : {(df['label']=='ham').sum():,}")

    log("\nÖzellikler hazırlanıyor...")
    text_corpus    = build_text(df)
    sender_feats   = build_sender_features(df)

    tfidf = TfidfVectorizer(
        max_features=10_000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2
    )
    X_tfidf       = tfidf.fit_transform(text_corpus)
    url_feat      = csr_matrix(df[["urls"]].values.astype(np.float32))
    sender_sparse = csr_matrix(sender_feats)
    X_all         = hstack([X_tfidf, url_feat, sender_sparse])
    y_all         = df["label"].values
    # 'ham' -> 0, 'spam' -> 1 yapıyoruz (XGBoost için şart)
    y_numeric = np.where(y_all == "spam", 1, 0)

    log(f"Özellik matrisi: {X_all.shape[0]:,} × {X_all.shape[1]:,}")
    log(f"  TF-IDF    : {X_tfidf.shape[1]:,} feature")
    log(f"  URL binary: 1 feature (Champa et al. 2024)")
    log(f"  Sender    : {sender_feats.shape[1]} feature")

    all_results = []
    idx_spam = np.where(y_numeric == 1)[0]
    idx_ham  = np.where(y_numeric == 0)[0]

    for n_total in INCREMENTAL_SIZES:
        n_per_class = n_total // 2
        log(f"\n{'─'*65}")
        log(f"  VERİ BOYUTU: {n_total} ({n_per_class} spam + {n_per_class} ham)")
        log(f"{'─'*65}")

        if len(idx_spam) < n_per_class or len(idx_ham) < n_per_class:
            log("  ⚠ Yetersiz veri, atlanıyor.")
            continue

        rng         = np.random.RandomState(RANDOM_STATE)
        chosen_spam = rng.choice(idx_spam, n_per_class, replace=False)
        chosen_ham  = rng.choice(idx_ham,  n_per_class, replace=False)
        chosen_idx  = np.concatenate([chosen_spam, chosen_ham])
        rng.shuffle(chosen_idx)

        X_sub = X_all[chosen_idx]
        y_sub = y_numeric[chosen_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.20,
            random_state=RANDOM_STATE, stratify=y_sub
        )
        log(f"  Train: {len(y_train)} | Test: {len(y_test)}")
        log(f"  {'Model':<24} {'Acc':>7} {'Pre':>7} {'Rec':>7} {'F1':>7} {'Süre':>6}")
        log(f"  {'─'*58}")

        for model_name, model in get_models().items():
            result = evaluate_model(
                model, X_train.copy(), X_test.copy(),
                y_train, y_test, model_name, len(y_train)
            )
            all_results.append(result)
            log(f"  {model_name:<24}"
                f"  %{result['Accuracy']:>4.1f}"
                f"  %{result['Precision']:>4.1f}"
                f"  %{result['Recall']:>4.1f}"
                f"  %{result['F1']:>4.1f}"
                f"  {result['Duration_s']:>4.1f}sn")

    df_res = pd.DataFrame(all_results)

    log(f"\n{'='*65}")
    log("  ÖZET — MODEL BAŞINA EN İYİ F1")
    log(f"{'='*65}")
    for model_name in df_res["Model"].unique():
        best = df_res[df_res["Model"] == model_name].sort_values("F1").iloc[-1]
        log(f"  {model_name:<24} → F1: %{best['F1']:.1f} "
            f"(n={best['Train_Size']+best['Test_Size']})")

    df_res.to_csv(ML_RESULTS_FILE, index=False)

    log(f"\nSonuçlar : {ML_RESULTS_FILE}")
    log(f"Rapor    : {ML_REPORT_FILE}")
    log("Tamamlandı!")

    with open(ML_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    return df_res


if __name__ == "__main__":
    try:
        import sklearn
    except ImportError:
        print("pip install scikit-learn scipy")
        exit()

    run_ml_comparison()