# -*- coding: utf-8 -*-

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import gdown

from transformers import BertTokenizer, BertModel

# =====================================================
# KONFIGURASI
# =====================================================
MODEL_PATH = "indobert_cnn_lstm.pt"
FILE_ID = "GANTI_DENGAN_FILE_ID_GOOGLE_DRIVE_KAMU"

HATE_THRESHOLD = 0.60   # <<< THRESHOLD UTAMA (AMAN UNTUK DEMO)

# =====================================================
# DOWNLOAD MODEL (GOOGLE DRIVE)
# =====================================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Mengunduh model, mohon tunggu...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(
            url,
            MODEL_PATH,
            quiet=False,
            fuzzy=True   # WAJIB untuk file besar (>100MB)
        )

download_model()

# =====================================================
# PREPROCESSING (SAMA DENGAN TRAINING)
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# MODEL INDO BERT + CNN + LSTM
# (COMPATIBLE DENGAN MODEL TRAINING)
# =====================================================
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        # BERT BEKU (sesuai training)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# =====================================================
# LOAD MODEL & TOKENIZER
# =====================================================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(
        "indobenchmark/indobert-base-p1"
    )

    model = IndoBERT_CNN_LSTM()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(
    page_title="Deteksi Ujaran Kebencian TikTok",
    layout="centered"
)

st.title("🛡️ Deteksi Ujaran Kebencian pada Komentar TikTok")
st.caption("Model IndoBERT–CNN–LSTM (Threshold-based Decision)")

st.markdown("""
**Catatan Sistem:**
- Model menggunakan **threshold probabilitas**
- Mengurangi kesalahan pada komentar kritik sopan
- Lebih presisi untuk demo & implementasi awal
""")

text_input = st.text_area(
    "Masukkan komentar TikTok:",
    placeholder="Contoh: aku kurang setuju sih tapi hargai pendapat orang lain",
    height=150
)

if st.button("🔍 Deteksi"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        clean = clean_text(text_input)

        encoding = tokenizer(
            clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(
                encoding["input_ids"],
                encoding["attention_mask"]
            )
            probs = F.softmax(outputs, dim=1)[0]

        hate_prob = probs[1].item()
        non_hate_prob = probs[0].item()

        # ===============================
        # THRESHOLD DECISION
        # ===============================
        if hate_prob >= HATE_THRESHOLD:
            pred = 1
        else:
            pred = 0

        st.subheader("Hasil Deteksi")

        # ===============================
        # OUTPUT UI
        # ===============================
        if pred == 1:
            st.error(f"🚨 Mengandung Ujaran Kebencian (Confidence {hate_prob:.2f})")
        else:
            st.success(f"✅ Tidak Mengandung Ujaran Kebencian (Confidence {non_hate_prob:.2f})")

        # Ambiguous zone (opsional tapi keren)
        if 0.45 <= hate_prob < HATE_THRESHOLD:
            st.warning("⚠️ Komentar ambigu – disarankan peninjauan manual.")

        st.markdown("### Probabilitas Prediksi")
        st.write(
            f"- Tidak Kebencian : **{non_hate_prob*100:.2f}%**  \n"
            f"- Mengandung Kebencian : **{hate_prob*100:.2f}%**"
        )
