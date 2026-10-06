import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from model.calorie_mapping import CALORIE_DICT as BASE_CALORIE_DICT


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NutriVision - AI Food Calorie Estimator",
    page_icon="🥗",
    layout="wide",
)

st.markdown("""
<style>
.main-title { font-size: 42px; font-weight: 700; text-align: center; margin-bottom: 5px; }
.subtitle   { text-align: center; font-size: 18px; color: #666; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FOOD101_MODEL_PATH = os.path.join(MODELS_DIR, "MOBILENET_best_model_food101_mobilenet.pt")
FOOD101_INFO_PATH = os.path.join(MODELS_DIR, "food101_mobilenet_MOBILENET_info.json")
INDIAN_MODEL_PATH = os.path.join(MODELS_DIR, "indian_food_mobilenet.pt")
INDIAN_CLASSES_PATH = os.path.join(MODELS_DIR, "indian_food_classes.json")


# ============================================================
# SESSION STATE
# ============================================================

if "food_log" not in st.session_state:
    st.session_state.food_log = []


# ============================================================
# CLASS NAMES
# ============================================================

def load_json(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    with open(path, "r") as f:
        return json.load(f)


FOOD101_CLASS_NAMES = load_json(FOOD101_INFO_PATH)["classes"]

_indian_data = load_json(INDIAN_CLASSES_PATH)
INDIAN_CLASS_NAMES = (
    _indian_data.get("classes", []) if isinstance(_indian_data, dict) else _indian_data
)
if not INDIAN_CLASS_NAMES:
    st.error("Indian food class names could not be loaded from indian_food_classes.json")
    st.stop()


# ============================================================
# CALORIES (approx. kcal per 100 g) AND MACROS (per 100 g)
# ============================================================

INDIAN_CALORIE_DICT = {
    "biryani": 198,
    "cholebhature": 300, "chole_bhature": 300,
    "dabeli": 200,
    "dal": 120,
    "dhokla": 160,
    "dosa": 168,
    "jalebi": 380,
    "kathiroll": 250, "kathi_roll": 250,
    "kofta": 200,
    "naan": 260,
    "pakora": 280,
    "paneer": 265,
    "panipuri": 150, "pani_puri": 150,
    "pavbhaji": 150, "pav_bhaji": 150,
    "vadapav": 290, "vada_pav": 290,
}

# Copy so we don't mutate the imported dict
CALORIE_DICT = {**BASE_CALORIE_DICT, **INDIAN_CALORIE_DICT}

# Macros only where we have data. Missing foods show "N/A", never fake zeros.
MACRO_DICT = {
    "apple_pie":            {"protein": 2.1,  "fat": 14.0, "carbs": 41.0},
    "beef_carpaccio":       {"protein": 20.0, "fat": 4.0,  "carbs": 0.0},
    "bibimbap":             {"protein": 3.0,  "fat": 2.0,  "carbs": 22.0},
    "cup_cakes":            {"protein": 3.6,  "fat": 12.0, "carbs": 46.0},
    "foie_gras":            {"protein": 7.0,  "fat": 43.0, "carbs": 2.0},
    "french_fries":         {"protein": 3.4,  "fat": 15.0, "carbs": 41.0},
    "garlic_bread":         {"protein": 7.0,  "fat": 17.0, "carbs": 44.0},
    "pizza":                {"protein": 11.0, "fat": 10.0, "carbs": 33.0},
    "spring_rolls":         {"protein": 3.0,  "fat": 5.0,  "carbs": 24.0},
    "spaghetti_carbonara":  {"protein": 13.0, "fat": 17.0, "carbs": 44.0},
    "strawberry_shortcake": {"protein": 3.0,  "fat": 10.0, "carbs": 40.0},
    "omelette":             {"protein": 10.0, "fat": 12.0, "carbs": 1.0},
}

# Indian foods: approximate macros per 100 g, chosen to stay consistent with
# INDIAN_CALORIE_DICT above (protein*4 + fat*9 + carbs*4 ~= kcal).
# These are estimates; verify against IFCT 2017 / USDA FoodData Central and
# cite the source you use. Recipes (oil, sugar, filling) vary a lot.
INDIAN_MACRO_DICT = {
    "biryani":       {"protein": 7.0,  "fat": 7.0,  "carbs": 27.0},
    "chole_bhature": {"protein": 7.0,  "fat": 14.0, "carbs": 36.0},
    "dabeli":        {"protein": 5.0,  "fat": 7.0,  "carbs": 29.0},
    "dal":           {"protein": 7.0,  "fat": 3.0,  "carbs": 16.5},
    "dhokla":        {"protein": 6.0,  "fat": 5.0,  "carbs": 22.0},
    "dosa":          {"protein": 4.0,  "fat": 4.0,  "carbs": 29.0},
    "jalebi":        {"protein": 2.0,  "fat": 12.0, "carbs": 66.0},
    "kathi_roll":    {"protein": 8.0,  "fat": 10.0, "carbs": 31.0},
    "kofta":         {"protein": 9.0,  "fat": 13.0, "carbs": 12.0},
    "naan":          {"protein": 9.0,  "fat": 5.0,  "carbs": 46.0},
    "pakora":        {"protein": 7.0,  "fat": 18.0, "carbs": 22.0},
    "paneer":        {"protein": 18.0, "fat": 20.0, "carbs": 3.0},
    "pani_puri":     {"protein": 3.0,  "fat": 4.0,  "carbs": 25.0},
    "pav_bhaji":     {"protein": 4.0,  "fat": 5.0,  "carbs": 22.0},
    "vada_pav":      {"protein": 7.0,  "fat": 11.0, "carbs": 41.0},
}
MACRO_DICT.update(INDIAN_MACRO_DICT)


# ============================================================
# MODELS
# ============================================================

class MobileNetFood101Model(nn.Module):
    """MobileNetV3-Large backbone + custom head (Food-101)."""

    def __init__(self, n_classes, dropout_rate=0.3):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None)
        num_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            *list(self.backbone.classifier.children())[:-1]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


class IndianFoodMobileNetModel(nn.Module):
    """MobileNetV3-Small with replaced final layer (Indian food)."""

    def __init__(self, n_classes):
        super().__init__()
        self.model = models.mobilenet_v3_small(weights=None)
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, n_classes)

    def forward(self, x):
        return self.model(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_torch_load(path, device):
    """Prefer weights_only=True (safer); fall back if the checkpoint needs full unpickling."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device)


def extract_state_dict(loaded):
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


@st.cache_resource
def load_food101_model():
    device = get_device()
    if not os.path.exists(FOOD101_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {FOOD101_MODEL_PATH}")

    model = MobileNetFood101Model(n_classes=len(FOOD101_CLASS_NAMES))
    loaded = safe_torch_load(FOOD101_MODEL_PATH, device)
    model.load_state_dict(extract_state_dict(loaded))
    model.to(device).eval()
    return model, device


@st.cache_resource
def load_indian_model():
    device = get_device()
    if not os.path.exists(INDIAN_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {INDIAN_MODEL_PATH}")

    model = IndianFoodMobileNetModel(n_classes=len(INDIAN_CLASS_NAMES))
    state = extract_state_dict(safe_torch_load(INDIAN_MODEL_PATH, device))

    try:
        model.load_state_dict(state)          # keys saved as "model.features...."
    except RuntimeError:
        model.model.load_state_dict(state)    # keys saved as "features...."

    model.to(device).eval()
    return model, device


# ============================================================
# PREPROCESS / PREDICT
# ============================================================

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_top_k(model, device, img, class_names, k=3):
    x = _transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    k = min(k, len(class_names))
    conf, idx = torch.topk(probs, k)
    return [(class_names[i], float(c)) for i, c in zip(idx.tolist(), conf.tolist())]


# ============================================================
# HELPERS
# ============================================================

def get_calories(class_name):
    if class_name in CALORIE_DICT:
        return CALORIE_DICT[class_name]

    normalized = class_name.lower().replace(" ", "_")
    if normalized in CALORIE_DICT:
        return CALORIE_DICT[normalized]

    squashed = normalized.replace("_", "")
    for key, value in CALORIE_DICT.items():
        if key.lower().replace("_", "").replace(" ", "") == squashed:
            return value
    return None


def get_macros(class_name):
    """Match class names like 'cholebhature' to 'chole_bhature' (ignores '_' and spaces)."""
    squashed = class_name.lower().replace(" ", "").replace("_", "")
    for key, value in MACRO_DICT.items():
        if key.lower().replace("_", "") == squashed:
            return value
    return None


def format_food_name(class_name):
    return class_name.replace("_", " ").title()


def fmt_macro(x):
    return "N/A" if pd.isna(x) else f"{x:.1f}"


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🥗 NutriVision AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered food recognition and calorie estimation</div>',
    unsafe_allow_html=True,
)
st.info("Upload a food image to identify the food, estimate calories, and analyze its nutrition.")


# ============================================================
# MODEL SELECTION + UPLOAD
# ============================================================

st.subheader("🤖 Choose Food Recognition Model")

model_type = st.radio(
    "Select the food category you want to recognize:",
    ["Indian Food", "Global Food (Food-101)"],
    horizontal=True,
)

if model_type == "Indian Food":
    st.caption(f"Indian model recognizes {len(INDIAN_CLASS_NAMES)} Indian food categories.")
else:
    st.caption(f"Food-101 model recognizes {len(FOOD101_CLASS_NAMES)} global food categories.")

uploaded_file = st.file_uploader("📷 Upload a food image", type=["jpg", "jpeg", "png"])


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Food Image", use_container_width=True)

        with st.spinner("🤖 Analyzing your food..."):
            if model_type == "Indian Food":
                model, device = load_indian_model()
                top_preds = predict_top_k(model, device, img, INDIAN_CLASS_NAMES)
            else:
                model, device = load_food101_model()
                top_preds = predict_top_k(model, device, img, FOOD101_CLASS_NAMES)

        class_name, confidence = top_preds[0]
        food_display_name = format_food_name(class_name)
        calories = get_calories(class_name)

        # ---------------- Result ----------------
        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        c1, c2 = st.columns(2)
        with c1:
            st.success(f"🍴 **Food:** {food_display_name}")
        with c2:
            st.metric("Model Confidence", f"{confidence:.2%}")

        with st.expander("See top predictions"):
            for name, conf in top_preds:
                st.write(f"{format_food_name(name)} — {conf:.2%}")

        if confidence < 0.50:
            st.warning("⚠️ The model has low confidence in this prediction.")
            if model_type == "Indian Food":
                st.info(
                    "The Indian model can only choose among its own Indian-food classes. "
                    "Low confidence does not prove the image isn't Indian food."
                )
            else:
                st.info(
                    "The Food-101 model can only choose among its 101 classes. "
                    "The image may belong to a food outside those categories."
                )
        elif confidence < 0.80:
            st.warning("⚠️ Moderate-confidence prediction. Try a clearer image if possible.")
        else:
            st.success("✅ High-confidence prediction.")

        # ---------------- Portion + calories ----------------
        st.markdown("---")
        st.subheader("⚖️ Portion Size & Calorie Calculator")

        if calories is None:
            st.warning(f"⚠️ Calorie information is currently unavailable for {food_display_name}.")
        else:
            weight = st.number_input(
                "Portion weight (g)",
                min_value=1,
                max_value=2000,
                value=100,
                step=10,
            )

            total_calories = calories * weight / 100

            st.metric("🔥 Estimated Calories", f"{total_calories:.1f} kcal")
            st.caption(f"{calories} kcal per 100 g × {weight} g")

            # ---------------- Macros ----------------
            macros = get_macros(class_name)
            protein_value = fat_value = carbs_value = None

            if macros:
                protein_value = round(macros["protein"] * weight / 100, 1)
                fat_value = round(macros["fat"] * weight / 100, 1)
                carbs_value = round(macros["carbs"] * weight / 100, 1)

                st.subheader("🥑 Macronutrients")
                m1, m2, m3 = st.columns(3)
                m1.metric("Protein", f"{protein_value:.1f} g")
                m2.metric("Fat", f"{fat_value:.1f} g")
                m3.metric("Carbohydrates", f"{carbs_value:.1f} g")

                fig = go.Figure(data=[go.Pie(
                    labels=["Protein", "Fat", "Carbohydrates"],
                    values=[protein_value * 4, fat_value * 9, carbs_value * 4],
                    hole=0.45,
                )])
                fig.update_layout(title=f"Calorie Distribution ({weight} g)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "ℹ️ Macronutrient information is not currently available for this food. "
                    "Calories are shown as an approximate estimate."
                )

            # ---------------- Add to log ----------------
            st.markdown("---")
            if st.button("➕ Add Food to Daily Log", use_container_width=True):
                st.session_state.food_log.append({
                    "Food": food_display_name,
                    "Weight (g)": weight,
                    "Calories (kcal)": round(total_calories, 1),
                    "Protein (g)": protein_value,   # None = unavailable (not zero)
                    "Fat (g)": fat_value,
                    "Carbs (g)": carbs_value,
                })
                st.success(f"✅ {food_display_name} ({weight} g) added to your daily log!")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


# ============================================================
# DAILY DASHBOARD
# ============================================================

st.markdown("---")
st.header("📊 Daily Nutrition Dashboard")

if st.session_state.food_log:
    food_df = pd.DataFrame(st.session_state.food_log)

    for col in ["Protein (g)", "Fat (g)", "Carbs (g)"]:
        food_df[col] = pd.to_numeric(food_df[col], errors="coerce")

    total_calories_day = food_df["Calories (kcal)"].sum()
    total_protein = food_df["Protein (g)"].sum(min_count=1)
    total_fat = food_df["Fat (g)"].sum(min_count=1)
    total_carbs = food_df["Carbs (g)"].sum(min_count=1)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("🔥 Calories", f"{total_calories_day:.1f} kcal")
    d2.metric("💪 Protein", "N/A" if pd.isna(total_protein) else f"{total_protein:.1f} g")
    d3.metric("🥑 Fat", "N/A" if pd.isna(total_fat) else f"{total_fat:.1f} g")
    d4.metric("🍞 Carbs", "N/A" if pd.isna(total_carbs) else f"{total_carbs:.1f} g")

    if food_df[["Protein (g)", "Fat (g)", "Carbs (g)"]].isna().any().any():
        st.caption(
            "ℹ️ Some foods in your log have no macro data. "
            "Macro totals include only foods where it is available."
        )

    st.subheader("🍽️ Today's Food Log")
    display_df = food_df.copy()
    for col in ["Protein (g)", "Fat (g)", "Carbs (g)"]:
        display_df[col] = display_df[col].apply(fmt_macro)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("📈 Calories by Food")
    calorie_chart = go.Figure(data=[go.Bar(
        x=food_df["Food"],
        y=food_df["Calories (kcal)"],
        text=food_df["Calories (kcal)"],
        textposition="auto",
    )])
    calorie_chart.update_layout(
        xaxis_title="Food",
        yaxis_title="Calories (kcal)",
        title="Daily Calorie Distribution",
    )
    st.plotly_chart(calorie_chart, use_container_width=True)

    if st.button("🗑️ Clear Daily Food Log", use_container_width=True):
        st.session_state.food_log = []
        st.rerun()
else:
    st.info(
        "Your daily food log is empty. Upload a food image and add it to start tracking your nutrition."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "⚠️ Nutrition and calorie values are estimates based on standard food reference data. "
    "They should not be considered medical or dietary advice."
)
st.caption("NutriVision AI • Food-101 + Indian Food MobileNetV3")