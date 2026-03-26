"""
Medical Chatbot — Fine-tuning T5 on data Patient/Doctor
Kaggle : Triagegeist
"""

import os
import sys

# Ajouter le chemin parent pour les imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import streamlit as st
import pickle
from datetime import datetime
from prediction_model.model import TriageModel

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME   = "t5-small"          # t5-base si tu veux plus de qualité (plus lent)
MAX_SRC_LEN  = 128
MAX_TGT_LEN  = 128
BATCH_SIZE   = 8
EPOCHS       = 10
LR           = 3e-4
SAVE_PATH    = "models/t5_medical.pt"
DATA_PATH    = "data/medical_chatbot.csv"
N_SAMPLES    = 10000               # Met None pour tout utiliser

os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device : {device}")

st.set_page_config(
    page_title="Triagegeist",
    page_icon="🏥",
    layout="wide"
)

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class MedicalDataset(Dataset):
    def __init__(self, df, tokenizer, max_src_len=MAX_SRC_LEN, max_tgt_len=MAX_TGT_LEN):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_src   = max_src_len
        self.max_tgt   = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # T5 attend un préfixe de tâche
        src_text = "medical question: " + str(self.data.loc[idx, "Patient"])
        tgt_text = str(self.data.loc[idx, "Doctor"])

        src = self.tokenizer(
            src_text,
            max_length=self.max_src,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt["input_ids"].squeeze()
        # T5 ignore le padding dans la loss si on met -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels":         labels
        }

# ──────────────────────────────────────────────
# CHARGEMENT DONNÉES
# ──────────────────────────────────────────────
print("Data load...")
df = pd.read_csv(DATA_PATH).dropna(subset=["Patient", "Doctor"])
if N_SAMPLES:
    df = df.head(N_SAMPLES)

# Split train / val (90/10)
split     = int(len(df) * 0.9)
df_train  = df.iloc[:split]
df_val    = df.iloc[split:]
print(f"✅ Train : {len(df_train)} | Val : {len(df_val)}")

# ──────────────────────────────────────────────
# TOKENIZER & MODÈLE
# ──────────────────────────────────────────────
print(f"Model load {MODEL_NAME}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
print(f"✅ Features : {sum(p.numel() for p in model.parameters()):,}")

# ──────────────────────────────────────────────
# DATALOADERS
# ──────────────────────────────────────────────
train_dataset = MedicalDataset(df_train, tokenizer)
val_dataset   = MedicalDataset(df_val,   tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
print(f"✅ Batchs train : {len(train_loader)} | val : {len(val_loader)}")

# ──────────────────────────────────────────────
# OPTIMIZER & SCHEDULER
# ──────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

total_steps   = len(train_loader) * EPOCHS
warmup_steps  = total_steps // 10          # 10% de warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# ──────────────────────────────────────────────
# ENTRAÎNEMENT (seulement si le modèle n'existe pas)
# ──────────────────────────────────────────────
if not os.path.exists("models/t5_medical_best"):
    print("Model T5 not found, starting training...")
    # [le code d'entraînement ici]
else:
    print("Modèle T5 found, skipping training.")
import time

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    # ── TRAIN ──
    model.train()
    total_train_loss = 0
    start = time.time()

    for batch in train_loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train = total_train_loss / len(train_loader)

    # ── VALIDATION ──
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_val_loss += outputs.loss.item()

    avg_val   = total_val_loss / len(val_loader)
    elapsed   = time.time() - start
    remaining = elapsed * (EPOCHS - epoch - 1)

    print(
        f"Epoch {epoch+1:3d}/{EPOCHS} | "
        f"Train: {avg_train:.4f} | "
        f"Val: {avg_val:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
        f"{elapsed:.1f}s | "
        f"Restant: {remaining/60:.1f}min"
    )

    # ── SAUVEGARDE DU MEILLEUR MODÈLE ──
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        model.save_pretrained("models/t5_medical_best")
        tokenizer.save_pretrained("models/t5_medical_best")
        print(f"  💾 Best model save (val loss: {best_val_loss:.4f})")

print("✅ Training finish !")

# ──────────────────────────────────────────────
# GÉNÉRATION DE RÉPONSE
# ──────────────────────────────────────────────
def generate_reply(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 100,
    num_beams: int = 4,
    temperature: float = 0.7,
    repetition_penalty: float = 2.5,
    no_repeat_ngram_size: int = 3,
) -> str:
    model.eval()
    src_text = "medical question: " + question.strip()

    inputs = tokenizer(
        src_text,
        return_tensors="pt",
        max_length=MAX_SRC_LEN,
        truncation=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,               # beam search → plus cohérent
            temperature=temperature,
            repetition_penalty=repetition_penalty,  # évite les répétitions
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ──────────────────────────────────────────────
# CHARGEMENT DU MEILLEUR MODÈLE & CHAT
# ──────────────────────────────────────────────
def load_chatbot():
    print("\nLoading the best model...")
    model = T5ForConditionalGeneration.from_pretrained("models/t5_medical_best").to(device)
    tokenizer = T5Tokenizer.from_pretrained("models/t5_medical_best")
    print("✅ Ready !\n")
    return model, tokenizer, None, None  # Return None for word2index, index2word since not used

def predict_esi(age, sex, systolic, diastolic, heart_rate, spo2, temperature, symptoms):
    """Predicts the ESI level based on patient data."""
    triage_model = TriageModel(n_features=40).to(device)
    triage_model.load_state_dict(torch.load("models/triage_model.pth", map_location=device))
    logits = triage_model(x)
    # Feature vector simplifié (adapter selon ton vrai modèle)
    sex_encoded = 1 if sex == "Male" else 0
    features = [age, sex_encoded, systolic, diastolic, heart_rate, spo2, temperature]
    features += [0] * (40 - len(features))  # padding jusqu'à 40 features
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        esi = logits.argmax(dim=1).item() + 1  # reconvertir 0-4 → 1-5
    return esi
 
ESI_LABELS = {
    1: ("Resuscitation", "#ff1744", "⚠️ Immediate life threat"),
    2: ("Emergent",      "#ff6d00", "🔴 High risk situation"),
    3: ("Urgent",        "#ffd600", "🟡 Stable but needs care"),
    4: ("Less Urgent",   "#00e676", "🟢 Non-urgent"),
    5: ("Non-Urgent",    "#29b6f6", "🔵 Minor problem"),
}
 
# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "patients" not in st.session_state:
    st.session_state.patients = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "page" not in st.session_state:
    st.session_state.page = "patient"
 
# ─────────────────────────────────────────
# SIDEBAR — Navigation
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title">TRIAGE<br>GEIST</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI Emergency Triage</p>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("👤  Patient Portal"):
        st.session_state.page = "patient"
        st.session_state.chat_history = []
    if st.button("🏥  Nurse Dashboard"):
        st.session_state.page = "nurse"
    st.markdown("---")
    st.markdown(f"**Patients en attente :** `{len(st.session_state.patients)}`")
    if st.session_state.patients:
        critical = sum(1 for p in st.session_state.patients if p["esi"] <= 2)
        st.markdown(f"**Cas critiques :** `{critical}`")
 
# ─────────────────────────────────────────
# PAGE PATIENT
# ─────────────────────────────────────────
if st.session_state.page == "patient":
    st.markdown('<h1 class="main-title">Patient Portal</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Emergency Department — Triage Assessment</p>', unsafe_allow_html=True)
    st.markdown("---")
 
    tab1, tab2 = st.tabs(["📋  Question", "💬  Medical Assistant"])
 
    # ── Tab 1 : Questionnaire ──
    with tab1:
        st.markdown("### Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input("Full Name", placeholder="John Doe")
        with col2:
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
        with col3:
            sex = st.selectbox("Sex", ["Male", "Female", "Other"])
 
        st.markdown("### Vital Signs")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            systolic = st.number_input("Systolic BP (mmHg)", 60, 250, 120)
        with col2:
            diastolic = st.number_input("Diastolic BP (mmHg)", 40, 150, 80)
        with col3:
            heart_rate = st.number_input("Heart Rate (bpm)", 30, 250, 75)
        with col4:
            spo2 = st.number_input("SpO2 (%)", 50, 100, 98)
        with col5:
            temperature = st.number_input("Temp (°C)", 34.0, 42.0, 37.0)
 
        st.markdown("### Chief Complaint")
        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="e.g. I have chest pain and difficulty breathing since 2 hours...",
            height=120
        )
 
        st.markdown("### Medical History")
        col1, col2 = st.columns(2)
        with col1:
            conditions = st.multiselect(
                "Known conditions",
                ["Diabetes", "Hypertension", "Heart disease", "Asthma", "Cancer", "None"]
            )
        with col2:
            medications = st.text_input("Current medications", placeholder="e.g. Metformin, Aspirin")
 
        st.markdown("---")
        if st.button("🚀  Submit & Get Triage Assessment"):
            if not name or not symptoms:
                st.warning("Please fill in your name and describe your symptoms.")
            else:
                with st.spinner("Analyzing your information..."):
                    esi = predict_esi(age, sex, systolic, diastolic, heart_rate, spo2, temperature, symptoms)
                    label, color, desc = ESI_LABELS[esi]
                    patient = {
                        "name": name, "age": age, "sex": sex,
                        "symptoms": symptoms, "esi": esi,
                        "systolic": systolic, "diastolic": diastolic,
                        "heart_rate": heart_rate, "spo2": spo2,
                        "temperature": temperature, "time": datetime.now().strftime("%H:%M"),
                        "conditions": conditions, "medications": medications
                    }
                    # Éviter les doublons
                    existing = [p["name"] for p in st.session_state.patients]
                    if name not in existing:
                        st.session_state.patients.append(patient)
 
                st.markdown(f"""
                <div class="success-box">
                    <h2>✅ Assessment Complete</h2>
                    <p>You have been assigned priority level:</p>
                    <span class="esi-badge esi-{esi}">ESI {esi} — {label}</span>
                    <p style="margin-top:1rem; color:#888;">{desc}</p>
                    <p style="color:#888;">Please wait to be called by a nurse.</p>
                </div>
                """, unsafe_allow_html=True)
 
    # ── Tab 2 : Chatbot ──
    with tab2:
        st.markdown("### Medical Assistant")
        st.markdown("*Ask me about your symptoms or concerns.*")
 
        chatbot_model, tokenizer, _, _ = load_chatbot()

        # Afficher l'historique
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("Your message", placeholder="Type your question...", label_visibility="collapsed")
        with col2:
            send = st.button("Send")

        if send and user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            reply = generate_reply(user_input, chatbot_model, tokenizer)
            st.rerun()
 
# ─────────────────────────────────────────
# PAGE INFIRMIÈRE
# ─────────────────────────────────────────
elif st.session_state.page == "nurse":
    st.markdown('<h1 class="main-title">Nurse Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Real-time triage priority list</p>', unsafe_allow_html=True)
    st.markdown("---")
 
    if not st.session_state.patients:
        st.info("No patients in queue.")
    else:
        # Trier par ESI croissant (1 = plus urgent)
        sorted_patients = sorted(st.session_state.patients, key=lambda x: x["esi"])
 
        # Résumé stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="vital-box"><div class="vital-value">{len(sorted_patients)}</div><div class="vital-label">Total Patients</div></div>', unsafe_allow_html=True)
        with col2:
            critical = sum(1 for p in sorted_patients if p["esi"] == 1)
            st.markdown(f'<div class="vital-box"><div class="vital-value" style="color:#ff1744">{critical}</div><div class="vital-label">ESI 1 — Critical</div></div>', unsafe_allow_html=True)
        with col3:
            emergent = sum(1 for p in sorted_patients if p["esi"] == 2)
            st.markdown(f'<div class="vital-box"><div class="vital-value" style="color:#ff6d00">{emergent}</div><div class="vital-label">ESI 2 — Emergent</div></div>', unsafe_allow_html=True)
        with col4:
            urgent = sum(1 for p in sorted_patients if p["esi"] == 3)
            st.markdown(f'<div class="vital-box"><div class="vital-value" style="color:#ffd600">{urgent}</div><div class="vital-label">ESI 3 — Urgent</div></div>', unsafe_allow_html=True)
 
        st.markdown("---")
        st.markdown("### Priority Queue")
 
        for i, patient in enumerate(sorted_patients):
            label, color, desc = ESI_LABELS[patient["esi"]]
            with st.expander(f"#{i+1}  {patient['name']}  |  ESI {patient['esi']} — {label}  |  {patient['time']}"):
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.markdown(f'<div class="vital-box"><div class="vital-value">{patient["heart_rate"]}</div><div class="vital-label">HR bpm</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="vital-box"><div class="vital-value">{patient["systolic"]}/{patient["diastolic"]}</div><div class="vital-label">BP mmHg</div></div>', unsafe_allow_html=True)
                with col3:
                    spo2_color = "#ff1744" if patient["spo2"] < 90 else "#00d4aa"
                    st.markdown(f'<div class="vital-box"><div class="vital-value" style="color:{spo2_color}">{patient["spo2"]}%</div><div class="vital-label">SpO2</div></div>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="vital-box"><div class="vital-value">{patient["temperature"]}°</div><div class="vital-label">Temp</div></div>', unsafe_allow_html=True)
                with col5:
                    st.markdown(f'<div class="vital-box"><div class="vital-value">{patient["age"]}</div><div class="vital-label">Age</div></div>', unsafe_allow_html=True)
 
                st.markdown(f"**Chief Complaint :** {patient['symptoms']}")
                if patient.get("conditions"):
                    st.markdown(f"**Conditions :** {', '.join(patient['conditions'])}")
                if patient.get("medications"):
                    st.markdown(f"**Medications :** {patient['medications']}")
 
                if st.button(f"✅ Mark as seen — {patient['name']}", key=f"seen_{i}"):
                    st.session_state.patients = [p for p in st.session_state.patients if p["name"] != patient["name"]]
                    st.rerun()