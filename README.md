# NutriVision 🍽️

**NutriVision** is a deep-learning powered food recognition and calorie estimation web application built with **Streamlit** and **PyTorch**.

The application can recognize both **Indian foods** and foods from the **Food-101 dataset**, then provide an estimated calorie value based on the predicted food and selected portion size.

---

## 🚀 Overview

NutriVision combines computer vision and nutrition estimation into a simple interactive application.

Users can:

- 📷 Upload a food image
- 🇮🇳 Recognize Indian foods using a dedicated Indian-food model
- 🌎 Recognize foods from the Food-101 dataset
- 🎯 View the predicted food and confidence score
- 🔥 Estimate calories based on portion size
- 📊 View available nutritional information
- 📈 Track food entries through the application dashboard

The application provides **two separate recognition modes** because the Indian-food model is trained specifically on a limited set of Indian food categories, while the Food-101 model covers a broader collection of international foods.

---

## ✨ Features

### 🇮🇳 Indian Food Recognition

NutriVision includes a dedicated **MobileNetV3-Small** model trained for 15 Indian food categories:

| # | Food |
|---|------|
| 1 | Biryani |
| 2 | Chole Bhature |
| 3 | Dabeli |
| 4 | Dal |
| 5 | Dhokla |
| 6 | Dosa |
| 7 | Jalebi |
| 8 | Kathi Roll |
| 9 | Kofta |
| 10 | Naan |
| 11 | Pakora |
| 12 | Paneer |
| 13 | Pani Puri |
| 14 | Pav Bhaji |
| 15 | Vada Pav |

### 🌎 Food-101 Recognition

The application also supports the **Food-101** dataset containing 101 food categories.

Users can manually select:

- **Indian Food**
- **Global Food (Food-101)**

This separation helps avoid treating every image as one of the 15 Indian-food classes.

### 🔥 Calorie Estimation

After identifying the food, NutriVision estimates calories based on:

- Predicted food category
- Approximate calories per 100 g
- User-selected portion size

> Calorie values are approximate estimates and should not be considered medical or dietary advice.

### 📊 Nutrition Information

Where nutritional data is available, the application can display information such as:

- Calories
- Protein
- Fat
- Carbohydrates

Nutrition information is dependent on the available food data.

### 🖥️ Interactive Streamlit Interface

The application provides:

- Image upload
- Model selection
- Prediction results
- Confidence score
- Portion-size controls
- Calorie estimation
- Nutrition information
- Food logging/dashboard functionality

---

## 🧠 Machine Learning

### Indian Food Model

The Indian food classifier uses:

**MobileNetV3-Small**

with a custom classification layer for the 15 Indian food categories.

The trained model is stored at:

```text
models/indian_food_mobilenet.pt
```

Class labels are stored at:

```text
models/indian_food_classes.json
```

### Food-101 Model

The project also contains a MobileNet-based Food-101 model and associated training artifacts.

The Food-101 model supports:

**101 food categories**

and is used for broader food recognition.

---

## 🏗️ Application Architecture

```text
                 ┌─────────────────────┐
                 │     User Image      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Streamlit App     │
                 │      app.py         │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
          ┌───────────────┐   ┌───────────────┐
          │ Indian Food   │   │ Global Food   │
          │ MobileNetV3   │   │   Food-101    │
          └───────┬───────┘   └───────┬───────┘
                  │                   │
                  └─────────┬─────────┘
                            ▼
                 ┌─────────────────────┐
                 │  Predicted Food +   │
                 │    Confidence       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Calorie / Nutrition│
                 │     Estimation      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Dashboard / Food Log│
                 └─────────────────────┘
```

---

## 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- PyTorch
- Torchvision
- MobileNetV3
- Scikit-learn

### Application

- Streamlit
- Plotly

### Image Processing

- Pillow
- NumPy

### Data Processing

- Pandas

---

## 📁 Project Structure

```text
NutriVision/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
│
├── data/
│   ├── download.py
│   ├── prepare.py
│   └── visualize.py
│
├── model/
│   ├── train.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── train_keras_food101tiny.py
│   ├── plot_mobilenet_accuracy.py
│   └── calorie_mapping.py
│
├── models/
│   ├── MOBILENET_best_model_food101_mobilenet.pt
│   ├── best_cpu_food101tiny_model.h5
│   ├── best_food101tiny_model.h5
│   ├── final_cpu_food101tiny_model.h5
│   ├── my_trained_food101tiny_model.h5
│   ├── indian_food_mobilenet.pt
│   ├── indian_food_classes.json
│   ├── food101_mobilenet_MOBILENET_info.json
│   ├── mobilenet_accuracy_curve.png
│   └── food101_mobilenet_MOBILENET_confusion_matrix.png
│
├── download_indian_dataset.py
├── train_indian_food.py
├── test_indian_food.py
└── evaluate_indian_food.py
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/shivamm-15/NutriVision.git
cd NutriVision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```powershell
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Model Evaluation

The repository contains scripts for testing and evaluating the Indian food classifier.

### Test a single image

```bash
python test_indian_food.py
```

### Evaluate the Indian food model

```bash
python evaluate_indian_food.py
```

### Train the Indian food model

```bash
python train_indian_food.py
```

> Model performance should be evaluated on a proper held-out test set. Accuracy numbers are intentionally not listed here until they have been independently verified using the final evaluation pipeline.

---

## 🍛 Calorie Estimation

NutriVision uses approximate calorie values associated with recognized food categories.

Examples include:

| Food | Approx. Calories |
|------|------------------|
| Biryani | 198 kcal / 100 g |
| Dosa | 168 kcal / 100 g |
| Jalebi | 380 kcal / 100 g |
| Naan | 260 kcal / 100 g |
| Paneer | 265 kcal / 100 g |
| Pav Bhaji | 150 kcal / 100 g |
| Vada Pav | 290 kcal / 100 g |

These values are intended for **approximate estimation only**. Actual calories can vary significantly depending on ingredients, recipe, cooking method, oil, serving size, and preparation.

---

## ⚠️ Limitations

NutriVision is a machine-learning demonstration and has several limitations:

- Image classification is not guaranteed to be correct.
- The Indian model only supports its 15 trained food categories.
- The model may perform poorly on images that differ significantly from its training data.
- Similar-looking foods can be confused.
- Calorie values are approximate.
- Portion-size estimation is not performed directly from the image.
- Nutritional information may not be available for every food category.
- The application should not be used as a substitute for professional nutritional or medical advice.

---

## 🔮 Future Improvements

Potential improvements include:

- Automatic Indian-vs-global food detection
- Larger Indian food dataset
- More Indian food categories
- Improved out-of-distribution detection
- Better portion-size estimation using computer vision
- More accurate nutrition databases
- Per-image nutritional analysis
- Cloud deployment optimization
- Model quantization for faster inference
- Mobile application
- User accounts and persistent food history
- Improved model evaluation and benchmarking

---

## 📌 Dataset

The project uses food image datasets for training and evaluation, including:

- **Food-101** for global food recognition
- An Indian food dataset for the dedicated Indian food classifier

Dataset files are intentionally excluded from the Git repository where appropriate because of their size.

---

## 📜 License

This project is distributed under the license included in the repository.

---

## 👨‍💻 Project

**NutriVision**

A computer-vision based food recognition and calorie estimation application built using Python, PyTorch, MobileNetV3, and Streamlit.

---

⭐ If you find the project useful, consider giving the repository a star.