"""
spam_detection.py
------------------
CEAS_08 dataseti üzerinde Groq + Llama 3.3 70B ile multi-agent spam tespiti.

Mimari:
  Şef Ajan (Orchestrator)
    ├── Content Agent   → body'deki spam dili, manipülasyon, kimlik avı
    ├── Sender Agent    → gönderici domain güvenilirliği, başlık analizi
    └── URL Agent       → urls (binary) + url_list (gerçek URL metinleri)
                          urls=0 ise URL Agent otomatik False döner
                          urls=1  ise url_list ile derinlemesine analiz yapar

Özellikler:
  - Key rotation    : 429 gelince anında sonraki key'e geçer
  - Checkpoint      : kaldığı yerden devam eder
  - Timeout kesme   : email çok uzun sürerse otomatik durur (kota doldu sinyali)
  - Adaptive threshold: URL var → 2/3, URL yok → 1/2


Çıktılar:
  analysis_report_ceas_groq.txt            → detaylı email bazlı rapor
  analysis_results_ceas_groq.csv           → makine okunabilir sonuçlar
"""

import pandas as pd
from groq import Groq
import json
import re
import time
from typing import Dict
import os
from dotenv import load_dotenv
load_dotenv()

# ── API AYARLARI ─────────────────────────────────────────────

API_KEYS = [
    os.getenv(f"GROQ_API_KEY_{i}") 
    for i in range(1, 11) 
    if os.getenv(f"GROQ_API_KEY_{i}") is not None
]

# ── DOSYA AYARLARI ───────────────────────────────────────────
DATASET_FILE     = "cleaned_ceas.csv"
SAMPLE_FRAC      = 0.10                        # %10 örneklem
SAMPLE_CKPT      = "sample_checkpoint.csv"     # hangi emailler seçildi
RESULTS_CKPT     = "results_checkpoint.csv"    # tamamlanan sonuçlar
REPORT_FILE      = "analysis_report_ceas_groq.txt"
FINAL_CSV        = "analysis_results_ceas_groq.csv"

# Bir email bu kadar saniyeden uzun sürerse kota dolmuş demektir → dur
MAX_EMAIL_SECONDS = 40
# ── KEY ROTASYON YÖNETİCİSİ ──────────────────────────────────
class KeyRotator:
    def __init__(self, keys: list):
        # Boş/placeholder key'leri filtrele
        self.keys = [k for k in keys if k and not k.startswith("KEY_")]
        if not self.keys:
            raise ValueError("Geçerli API key bulunamadı! API_KEYS listesini doldur.")
        self.index     = 0
        self.clients   = {k: Groq(api_key=k) for k in self.keys}
        self.exhausted = set()  # bu döngüde 429 alan keyler
        print(f"  Key havuzu başlatıldı: {len(self.keys)} key aktif")
 
    @property
    def current_key(self) -> str:
        return self.keys[self.index]
 
    @property
    def current_client(self) -> Groq:
        return self.clients[self.current_key]
 
    def rotate(self) -> bool:
        """
        Bir sonraki key'e geç.
        Tüm keyler exhausted ise False döner (bekleme gerekli).
        """
        self.exhausted.add(self.current_key)
        next_keys = [k for k in self.keys if k not in self.exhausted]
 
        if not next_keys:
            # Tüm keyler bu dakikada doldu → bekleme gerekli
            print(f"  [KeyRotator] Tüm {len(self.keys)} key bu dakikada doldu!")
            print(f"  [KeyRotator] 62sn bekleniyor, sonra tüm keyler sıfırlanıyor...")
            time.sleep(62)
            self.exhausted.clear()  # yeni dakikada tüm keyler temiz
            self.index = 0
            print(f"  [KeyRotator] Keyler sıfırlandı, Key-1'den devam ediliyor.")
        else:
            self.index = self.keys.index(next_keys[0])
            print(f"  [KeyRotator] Key-{self.index+1}'e geçildi.")
 
      
 
    def reset_exhausted(self):
        """Başarılı bir çağrıdan sonra exhausted listesini temizle."""
        self.exhausted.clear()
 
 
# Global rotator — tüm agentlar bunu kullanır
rotator = KeyRotator(API_KEYS)

# CHECKPOINT YÖNETİCİSİ

class CheckpointManager:
 
    def __init__(self):
        self.completed_ids = set()
 
    def load_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Örneklemi yükler veya oluşturur.
        sample_checkpoint.csv varsa → aynı örneklem kullanılır (tutarlılık)
        yoksa → yeni örneklem oluşturulur ve kaydedilir
        """
        if os.path.exists(SAMPLE_CKPT):
            sampled = pd.read_csv(SAMPLE_CKPT)
            print(f" Önceki örneklem yüklendi: {len(sampled)} email ({SAMPLE_CKPT})")
        else:
            df_spam = df[df["label"]=="spam"].sample(frac=SAMPLE_FRAC, random_state=42)
            df_ham  = df[df["label"]=="ham"].sample( frac=SAMPLE_FRAC, random_state=42)
            sampled = pd.concat([df_spam, df_ham]).sample(
                          frac=1, random_state=42).reset_index(drop=True)
            sampled.to_csv(SAMPLE_CKPT, index=True, index_label="sample_id")
            print(f" Yeni örneklem oluşturuldu: {len(sampled)} email "
                 f"({len(df_spam)} spam + {len(df_ham)} ham) → {SAMPLE_CKPT}")
        return sampled
 
        
 
    def load_completed(self) -> set:
        """
        Daha önce tamamlanan email ID'lerini yükler.
        results_checkpoint.csv varsa → tamamlananlar set'e eklenir
        """
        if os.path.exists(RESULTS_CKPT):
            df_done = pd.read_csv(RESULTS_CKPT)
            self.completed_ids = set(df_done["Email_ID"].astype(str).tolist())
            print(f"Önceki sonuçlar yüklendi: {len(self.completed_ids)} email "
                  f"tamamlanmış → kaldığı yerden devam edilecek")
        else:
            self.completed_ids = set()
            print(f"Checkpoint bulunamadı → baştan başlanıyor")
        return self.completed_ids
 
    def save_result(self, result: dict):
        """Her email bittikten sonra sonucu checkpoint'e ekler."""
        row_df = pd.DataFrame([result])
        if os.path.exists(RESULTS_CKPT):
            row_df.to_csv(RESULTS_CKPT, mode="a", header=False, index=False)
        else:
            row_df.to_csv(RESULTS_CKPT, mode="w", header=True,  index=False)
 
    def is_completed(self, email_id) -> bool:
        return str(email_id) in self.completed_ids
 
    def load_all_results(self) -> pd.DataFrame:
        """Final rapor için tüm sonuçları yükler."""
        return pd.read_csv(RESULTS_CKPT)
 
    def reset(self):
        """Checkpoint'leri sil — baştan başlamak için."""
        for f in [SAMPLE_CKPT, RESULTS_CKPT]:
            if os.path.exists(f):
                os.remove(f)
        print("🗑  Checkpoint'ler silindi, baştan başlanacak.")


# SUB-AGENT'LAR

class SpamAgents:

    def __init__(self):
        self.model = "llama-3.3-70b-versatile"

    def _call_llm(self, prompt: str) -> dict:
        """
        Round-robin: her çağrıda sıradaki key'i kullan.
        429 alınca o key'i atla, bir sonrakine geç.
        """
        start_index = rotator.index  # hangi key'den başladık
    
        for attempt in range(len(rotator.keys) * 2):
            key_index = (start_index + attempt) % len(rotator.keys)
            rotator.index = key_index
            key_num = key_index + 1

            if attempt > 0 and attempt % len(rotator.keys) == 0:
                print(f"  [KeyRotator] Tüm keyler denendi, 62sn bekleniyor...")
                time.sleep(62)

            try:
                response = rotator.current_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                text = response.choices[0].message.content.strip()
                result = json.loads(text)
                # Başarılı → bir sonraki çağrı için index'i ilerlet
                rotator.index = (key_index + 1) % len(rotator.keys)
                print(f"  [Key-{key_num}] ✓ Başarılı")
                return result

            except json.JSONDecodeError as e:
                print(f"  [JSON Hatası] {e}")
                return self._default()

            except Exception as e:
                err = str(e)
                if "429" in err:
                    print(f"  [Rate Limit] Key-{key_num} doldu → "
                        f"Key-{((key_index+1) % len(rotator.keys))+1}'e geçiliyor...")
                    continue
                elif "503" in err:
                    print(f"  [503] 10sn bekleniyor...")
                    time.sleep(10)
                    continue
                else:
                    print(f"  [Groq Hatası] {e}")
                    return self._default()

        print("  [Hata] Tüm denemeler tükendi.")
        return self._default()
 
    
    def _default(self) -> dict:
        return {"content": False, "sender": False, "url": False}

    # ── CONTENT AGENT ────────────────────────────────────────
    def content_agent(self, subject: str, body: str, has_url: bool) -> bool:
        """
        Body ve subject üzerinde spam dili, manipülasyon ve
        kimlik avı (phishing) sinyalleri arar.# URL yoksa model body'ye daha dikkatli baksın
        """
        extra = ""
        if not has_url:
            extra = """
    IMPORTANT: This email has NO URLs. Be more sensitive to subtle spam signals:
- Vague or misleading subject with no clear business purpose
- Health, medication, adult content, or financial offer references
- Oddly generic or impersonal body text
- Any hidden call-to-action without a link"""
        prompt = f"""You are a Content Analysis Agent specialized in spam and phishing detection.
Analyze the email subject and body for spam signals.

EXAMPLES:
Subject: "YOU WON $1,000,000!", Body: "Dear friend claim your prize send bank details now"
→ {{"content": true, "reason": "prize scam with bank detail request"}}

Subject: "Meeting tomorrow", Body: "Hi team, the 3pm meeting is confirmed. Please review the agenda."
→ {{"content": false, "reason": "normal business communication"}}

Subject: "URGENT: Account suspended", Body: "Your account has been suspended. Verify immediately or lose access."
→ {{"content": true, "reason": "fake urgency and account suspension threat"}}

NOW ANALYZE:
Subject: {subject}
Body: {body[:800]}

Flag TRUE if body/subject contains ANY of:
- Unsolicited money, prize, or lottery offers
- Requests for passwords, bank details, or credentials
- Impersonation of authority (bank, IT, government, PayPal, etc.)
- Fake urgency threats ("act now", "account suspended", "immediate action")
- Mass-spam openers ("Dear Beneficiary", "Dear Friend", "Congratulations!")
- Unsolicited luxury/replica product offers 
  (watches, bags, pills, supplements — "Rolex", "replica", "Viagra", "pharmacy")
- Body contains non-ASCII gibberish, random symbols, or encoding anomalies
  suggesting obfuscation

Flag FALSE for normal professional or personal communication,
software releases, open source announcements, or technical newsletters.

Return ONLY JSON: {{"content": true_or_false, "reason": "brief reason"}}"""

        raw = self._call_llm(prompt)
        return bool(raw.get("content", False))

    # ── SENDER AGENT ─────────────────────────────────────────
    def sender_agent(self, sender: str, subject: str) -> bool:
        """
        Gönderici adresi ve başlık güvenilirliğini analiz eder.
        Champa et al. (2024) çalışmasında sender domain
        önemli bir feature olarak gösterilmiştir.
        """
        prompt = f"""You are a Sender Reputation Agent specialized in email authentication analysis.

EXAMPLES:
Sender: "support@paypa1.com", Subject: "Verify your account"
→ {{"sender": true, "reason": "misspelled domain impersonating PayPal"}}

Sender: "john.smith@company.com", Subject: "Q3 Report attached"
→ {{"sender": false, "reason": "legitimate corporate sender"}}

Sender: "noreply@amazon-security-alert.net", Subject: "Your order"
→ {{"sender": true, "reason": "suspicious domain claiming to be Amazon"}}

Sender: "unknown", Subject: "Meeting notes"
→ {{"sender": false, "reason": "unknown sender but subject is normal"}}

NOW ANALYZE:
Sender : {sender}
Subject: {subject}

Flag TRUE if:
- Sender domain is misspelled to impersonate a brand (paypa1, amaz0n, g00gle)
- Free email service (gmail, yahoo, hotmail) used for official/bank communications
- Randomly generated looking address (abc123xzy@domain.com)
- Subject uses ALL CAPS panic, prize/lottery language, or fake "Re:"/"Fwd:" on unsolicited

Flag FALSE if:
- Sender is unknown/empty or looks like a normal address
- Subject indicates software release, open source project, or technical 
  announcement (e.g. "released", "v1.3", "[project-name]", "update")
- This is clearly a mailing list post (subject has [brackets])

Return ONLY JSON: {{"sender": true_or_false, "reason": "brief reason"}}"""

        raw = self._call_llm(prompt)
        return bool(raw.get("sender", False))

    # ── URL AGENT ─────────────────────────────────────────────
    def url_agent(self, urls: bool, url_list: str) -> bool:
        """
        CEAS_08 URL binary feature + body'den çekilen URL metinlerini kullanır.

        Akademik gerekçe:
        Champa et al. (IEEE ICMI 2024): "a value of 1 indicates the presence
        of URL(s) in the email body, whereas 0 indicates their absence."
        Bu binary feature, klasik ML modellerinde güçlü bir özellik olarak
        tespit edilmiştir. Bu agent bunu LLM tabanlı derinlemesine analize taşır.
        """
        # urls=0 ise URL yok, analiz etmeye gerek yok
        if urls == 0: 
            return False

        # URL listesi boşsa (urls=1 ama header'da URL var)
        if not url_list or str(url_list).strip() in ("", "nan"):
            # Sadece URL varlığı şüpheli değil, içerik gerekli
            return False

        prompt = f"""You are a URL Security Agent specialized in detecting malicious links.
The email is confirmed to contain URLs (urls=1 from dataset feature).

EXAMPLES:
URLs: "http://bit.ly/xyz123 | http://tinyurl.com/claim"
→ {{"url": true, "reason": "multiple URL shorteners used to hide destination"}}

URLs: "https://www.paypa1.com/verify | http://secure-paypal.net"
→ {{"url": true, "reason": "misspelled PayPal domain and suspicious domain"}}

URLs: "https://docs.google.com/spreadsheet/abc123"
→ {{"url": false, "reason": "legitimate Google Docs URL"}}

URLs: "http://192.168.1.1/login"
→ {{"url": true, "reason": "IP address used instead of domain name"}}

NOW ANALYZE these URLs extracted from the email body:
{url_list[:500]}

Flag TRUE if ANY URL shows:
- URL shorteners (bit.ly, tinyurl.com, goo.gl, t.co, ow.ly, etc.)
- IP address used as link (http://123.45.67.89/...)
- Visibly misspelled brand domains (paypa1, amaz0n, g00gle, micros0ft)
- Suspicious TLDs combined with brand names (.net/.info/.xyz for official brands)
- Excessive redirects or random-looking paths
- URLs are from a well-known brand (CNN, BBC, PayPal) BUT sender domain 
  is completely unrelated to that brand — this is brand impersonation via 
  legitimate-looking URLs

Flag FALSE if:
- All URLs are from well-known legitimate domains
- URLs appear to be normal corporate or news website links

Return ONLY JSON: {{"url": true_or_false, "reason": "brief reason"}}"""

        raw = self._call_llm(prompt)
        return bool(raw.get("url", False))

    # ── TÜM AJANLARI ÇALIŞTIR ────────────────────────────────
    def analyze(self, subject: str, body: str,
                sender: str, urls: int, url_list: str) -> Dict[str, bool]:

        has_url = bool(urls == 1)
        print(f"  [Content Agent] çalışıyor...")
        content_result = self.content_agent(subject, body, has_url)

        print(f"  [Sender Agent]  çalışıyor...")
        sender_result = self.sender_agent(sender, subject)


        if has_url:
            short = url_list[:60]
            suffix = "..." if len(url_list) > 60 else ""
            print(f"  [URL Agent]     çalışıyor... (URL var: {short}{suffix})")
        else:
            print(f"  [URL Agent]     atlandı (URL yok → otomatik GÜVENLİ)")
 
        url_result = self.url_agent(urls, url_list)

        return {
            "Content": content_result,
            "Sender":  sender_result,
            "URL":     url_result,
        }



# ŞEF AJAN (ORCHESTRATOR)

class Orchestrator:

    def __init__(self, spam_threshold_url: int = 2, spam_threshold_no_url: int = 1):
        self.agents           = SpamAgents()
        self.checkpoint       = CheckpointManager()
        self.start_time       = None
        self.thresh_url       = spam_threshold_url
        self.thresh_no_url    = spam_threshold_no_url
        self.session_count    = 0
        self.last_duration    = 0.0  # TIMEOUT: son email süresi
 
    # BUG1 FIX: sampled parametre olarak alınıyor
    def start_session(self, sampled: pd.DataFrame):
        self.start_time = time.time()
        completed = self.checkpoint.load_completed()
        remaining = len(sampled) - len(completed)
        header = (
            f"\n{'='*60}\n"
            f"  ŞEF AJAN — ANALİZ OTURUMU BAŞLADI\n"
            f"  Dataset        : CEAS_08 (cleaned_ceas.csv)\n"
            f"  Toplam Örneklem: {len(sampled)} email\n"
            f"  Tamamlanan     : {len(completed)} email\n"
            f"  Kalan          : {remaining} email\n"
            f"  Threshold       : URL var→{self.thresh_url}/3 | "
            f"URL yok→{self.thresh_no_url}/2\n"
            f"  Model          : Groq / llama-3.3-70b-versatile\n"
            f"  Timeout Limiti  : {MAX_EMAIL_SECONDS}sn\n"
            f"  Başlangıç      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'='*60}\n"
        )
        print(header)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(header)

    def analyze_and_report(self, row: pd.Series):
        email_id = str(row.name)
 
        # Daha önce tamamlandıysa atla
        if self.checkpoint.is_completed(email_id):
            return False
        email_start = time.time()

        subject  = str(row.get("subject",  "No Subject"))
        body     = str(row.get("body",     ""))
        sender   = str(row.get("sender",   "unknown"))
        date     = str(row.get("date",     "unknown"))
        urls     = int(row.get("urls", 0))
        url_list = str(row.get("url_list", ""))
        label    = str(row.get("label",    "unknown"))

        print(f"\n{'─'*50}")
        print(f"Email ID: {row.name} | Gerçek: {label.upper()} | "
              f"URL: {'VAR' if urls else 'YOK'}")

        results = self.agents.analyze(subject, body, sender, urls, url_list)

        spam_votes     = sum(results.values())
        # URL var  → 2/3 oy gerekli (mevcut)
        # URL yok  → 1/2 oy yeterli (yeni)
        # Adaptive threshold
        threshold = self.thresh_url if urls else self.thresh_no_url
        final_decision = "spam" if spam_votes >= threshold else "ham"
 
        duration = round(time.time() - email_start, 2)
        correct        = (final_decision == label)
        self.session_count += 1
        self.last_duration = duration  # TIMEOUT: ana döngü okuyacak

        # Sonucu checkpoint'e kaydet
        result_row = {
            "Email_ID":   email_id,
            "True_Label": label,
            "Prediction": final_decision,
            "Votes":      spam_votes,
            "Threshold":  threshold,
            "urls":       urls,
            "Correct":    correct,
            "Duration":   duration,
        }
        if duration > MAX_EMAIL_SECONDS and spam_votes == 0:
            print(f"  ⚠ Timeout nedeniyle bu email checkpoint'e kaydedilmeyecek "
                f"(yeniden analiz edilecek)")
            self.last_duration = duration
            return True  # processed sayısına ekle ama kaydetme
        self.checkpoint.save_result(result_row)
 
        entry = (
            f"\n[Email ID: {email_id}]\n"
            f"Gönderici      : {sender}\n"
            f"Tarih          : {date}\n"
            f"Konu           : {subject}\n"
            f"URL Durumu     : {'VAR → ' + url_list[:80] if urls else 'YOK'}\n"
            f"Gerçek Etiket  : {label.upper()}\n"
            f"Şef Kararı     : {final_decision.upper()} "
            f"({spam_votes}/3 oy, eşik:{threshold}) "
            f"{'✓ DOĞRU' if correct else '✗ YANLIŞ'}\n"
            f"Oy Dökümü:\n"
            f"  • Content Agent : {'⚠ ŞÜPHELI' if results['Content'] else '✓ GÜVENLİ'}\n"
            f"  • Sender Agent  : {'⚠ ŞÜPHELI' if results['Sender']  else '✓ GÜVENLİ'}\n"
            f"  • URL Agent     : {'⚠ ŞÜPHELI' if results['URL']     else '✓ GÜVENLİ'}"
            f"{' (URL YOK → otomatik)' if not urls else ''}\n"
            f"İşlem Süresi   : {duration} saniye\n"
            f"{'-'*50}\n"
        )
        print(entry)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
 
        return True

    def end_session(self, total_sample_size: int):
        """
        Tüm emailler bittiyse final raporu üretir.
        Bitmemişse kısmi özet gösterir.
        """
        session_dur = time.time() - self.start_time
        mins, secs  = divmod(session_dur, 60)
 
        # Tamamlanan sonuçları yükle
        df_res = self.checkpoint.load_all_results()
        done   = len(df_res)
 
        print(f"\n{'='*60}")
        print(f"  OTURUM SONU: {self.session_count} email işlendi "
              f"({int(mins)}dk {int(secs)}sn)")
        print(f"  Toplam tamamlanan: {done}/{total_sample_size}")
 
        if done < total_sample_size:
            pct = done / total_sample_size * 100
            print(f"  İlerleme: %{pct:.1f} — "
                  f"kodu tekrar çalıştırınca kaldığı yerden devam eder.")
            print(f"{'='*60}\n")
            return
 
        # Tüm emailler tamamlandı → final rapor
        print(f"  ✅ Tüm emailler tamamlandı! Final rapor üretiliyor...")
        print(f"{'='*60}")
 
        df_res.to_csv(FINAL_CSV, index=False)

        accuracy  = df_res["Correct"].mean() * 100
        tp = len(df_res[(df_res["True_Label"]=="spam") & (df_res["Prediction"]=="spam")])
        tn = len(df_res[(df_res["True_Label"]=="ham")  & (df_res["Prediction"]=="ham")])
        fp = len(df_res[(df_res["True_Label"]=="ham")  & (df_res["Prediction"]=="spam")])
        fn = len(df_res[(df_res["True_Label"]=="spam") & (df_res["Prediction"]=="ham")])

        precision = tp/(tp+fp) if (tp+fp)>0 else 0
        recall    = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0

        # URL varlığının etkisi
        url_df    = df_res[df_res["urls"]==1]
        no_url_df = df_res[df_res["urls"]==0]
        url_acc    = url_df["Correct"].mean()*100    if len(url_df)>0    else 0
        no_url_acc = no_url_df["Correct"].mean()*100 if len(no_url_df)>0 else 0

        summary = (
            f"\n{'='*60}\n"
            f"  ŞEF AJAN — FİNAL RAPOR\n"
            f"{'='*60}\n"
            f"Tamamlanma     : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Analiz Edilen  : {done} email\n"  # BUG3 FIX: artık doğru sayı
            f"Threshold      : URL var→{self.thresh_url}/3 | "
            f"URL yok→{self.thresh_no_url}/2\n"
            f"Model          : Groq / llama-3.3-70b-versatile\n\n"
            f"── PERFORMANS ──────────────────────────────────────\n"
            f"Accuracy       : %{accuracy:.2f}\n"
            f"Precision      : %{precision:.2f}\n"
            f"Recall         : %{recall:.2f}\n"
            f"F1 Score       : %{f1:.2f}\n\n"
            f"── CONFUSION MATRIX ────────────────────────────────\n"
            f"  TP (spam→spam) : {tp:4d}   FN (spam→ham) : {fn:4d}\n"
            f"  FP (ham→spam)  : {fp:4d}   TN (ham→ham)  : {tn:4d}\n\n"
            f"── URL FEATURE ETKİSİ (Champa et al. 2024) ────────\n"
            f"  URL olan emailler  : {len(url_df):4d} → Accuracy: %{url_acc:.1f}\n"
            f"  URL olmayan email  : {len(no_url_df):4d} → Accuracy: %{no_url_acc:.1f}\n\n"
            f"Final CSV      : {FINAL_CSV}\n"
            f"{'='*60}\n"
        )
        print(summary)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(summary)


# MAIN

if __name__ == "__main__":

     # Checkpoint'leri sıfırlamak istersen → True yap, sonra False'a dön
    RESET_CHECKPOINT = False

    try:
        df = pd.read_csv(DATASET_FILE)
        print(f"Dataset yüklendi   : {len(df):,} email")
        print(f"Spam               : {(df['label']=='spam').sum():,}")
        print(f"Ham                : {(df['label']=='ham').sum():,}")
        print(f"URL var (urls)  : {df['urls'].sum():,}")

        ckpt = CheckpointManager()
 
        if RESET_CHECKPOINT:
            ckpt.reset()
 
        # Örneklemi yükle veya oluştur (aynı random_state=42 ile hep aynı)
        sampled = ckpt.load_sample(df)
 
        print(f"Örneklem           : {len(sampled)} email")
        print(f"Örneklemde URL var : {sampled['urls'].sum()}\n")
 
        orchestrator = Orchestrator(
            spam_threshold_url=2,
            spam_threshold_no_url=1
        )
        # BUG1 FIX: sampled parametre olarak geçiriliyor
        orchestrator.start_session(sampled=sampled)
 
        processed = 0
        skipped   = 0


        for i, (_, row) in enumerate(sampled.iterrows()):
            did_process = orchestrator.analyze_and_report(row)
 
            if did_process:
                processed += 1
                if orchestrator.last_duration > MAX_EMAIL_SECONDS:
                    total_done = len(ckpt.load_completed())
                    pct = total_done / len(sampled) * 100
                    msg = (
                        f"\n{'='*50}\n"
                        f"  ⏱  YAVAŞLAMA ALGILANDI — Otomatik durduruldu\n"
                        f"  Son email süresi  : {orchestrator.last_duration:.1f}sn "
                        f"(limit: {MAX_EMAIL_SECONDS}sn)\n"
                        f"  Tamamlanan        : {total_done}/{len(sampled)} (%{pct:.1f})\n"
                        f"  Kodu tekrar çalıştırınca kaldığı yerden devam eder.\n"
                        f"{'='*50}\n"
                    )
                    print(msg)
                    with open(REPORT_FILE, "a", encoding="utf-8") as f:
                        f.write(msg)
                    orchestrator.end_session(total_sample_size=len(sampled))
                    break
 
                time.sleep(1)
                
            else:
                skipped += 1
                if skipped <= 3 or skipped % 100 == 0:
                    print(f"  [Checkpoint] Email {row.name} atlandı "
                          f"(zaten tamamlanmış)")
 
            if (i + 1) % 10 == 0:
                total_done = len(ckpt.load_completed())
                elapsed    = time.time() - orchestrator.start_time
                remaining  = len(sampled) - total_done
                if processed > 0:
                    eta = (elapsed / processed) * remaining
                    print(f"\n>>> İlerleme: {total_done}/{len(sampled)} | "
                          f"Bu oturumda: {processed} | "
                          f"Kalan: ~{int(eta//60)}dk {int(eta%60)}sn <<<\n")
 
        else:
            # for döngüsü break olmadan bittiyse (tüm emailler işlendi)
            orchestrator.end_session(total_sample_size=len(sampled))
 
    except FileNotFoundError:
        print(f"HATA: {DATASET_FILE} bulunamadı!")
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        raise