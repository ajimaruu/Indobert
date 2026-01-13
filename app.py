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
# KONFIGURASI MODEL
# =====================================================
MODEL_PATH = "indobert_cnn_lstm.pt"
FILE_ID = "1XfMHjhm3XoT1IitYXh0b8WonsoEarfUb"  # <- WAJIB GANTI

# =====================================================
# DOWNLOAD MODEL JIKA BELUM ADA
# =====================================================
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Mengunduh model, mohon tunggu...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(
            url,
            MODEL_PATH,
            quiet=False,
            fuzzy=True   # <<< INI KUNCINYA
        )


# =====================================================
# PREPROCESSING TEKS
# (HARUS SAMA DENGAN TRAINING)
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
# (VERSI COMPATIBLE DENGAN MODEL TRAINING)
# =====================================================
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "indobenchmark/indobert-base-p1"
        )

        # Bekukan semua parameter BERT (sesuai training)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.conv = nn.Conv1d(
            in_channels=768,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        x = outputs.last_hidden_state      # (B, T, 768)
        x = x.permute(0, 2, 1)             # (B, 768, T)
        x = self.conv(x)                   # (B, 128, T)
        x = x.permute(0, 2, 1)             # (B, T, 128)

        _, (h_n, _) = self.lstm(x)          # (1, B, 64)
        logits = self.fc(h_n.squeeze(0))    # (B, 2)
        return logits

# =====================================================
# LOAD TOKENIZER & MODEL
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
st.caption("Model IndoBERT–CNN–LSTM | Sesuai Proposal")

st.markdown("""
**Alur Sistem:**
1. Preprocessing teks
2. Tokenisasi IndoBERT
3. Ekstraksi fitur (IndoBERT)
4. CNN + LSTM
5. Klasifikasi biner
""")

text_input = st.text_area(
    "Masukkan komentar TikTok:",
    placeholder="Contoh: dasar goblok, gak punya otak!",
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
            pred = torch.argmax(probs).item()

        st.subheader("Hasil Deteksi")

        if pred == 1:
            st.error("🚨 Mengandung Ujaran Kebencian")
        else:
            st.success("✅ Tidak Mengandung Ujaran Kebencian")

        st.markdown("### Confidence")
        st.progress(float(probs[pred]))

        st.write(
            f"Tidak Kebencian : **{probs[0]*100:.2f}%**  \n"
            f"Mengandung Kebencian : **{probs[1]*100:.2f}%**"
        )
