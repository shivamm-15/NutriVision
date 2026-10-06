# NutriVision

Food recognition and calorie estimation for Indian and global foods, built with PyTorch and Streamlit.

<!-- TODO: add a screenshot or GIF of the app (Indian mode with the macro pie chart) -->
URL-https://nutrivision-mrsc2d8hsuteg5z9xwsrg7.streamlit.app/

<!-- TODO: after deploying, add: **[Live demo](YOUR_STREAMLIT_URL)** -->

## Features

- **Two recognition modes:** 15 Indian foods (MobileNetV3-Small) and Food-101 with 101 global classes (MobileNetV3-Large)
- **Top-3 predictions** with confidence warnings for uncertain images
- **Portion-based estimates:** calories, protein, fat and carbohydrates for a chosen weight, with a calorie-distribution chart
- **Daily dashboard:** log foods during a session and view totals and a per-food calorie chart

**Indian classes:** Biryani, Chole Bhature, Dabeli, Dal, Dhokla, Dosa, Jalebi, Kathi Roll, Kofta, Naan, Pakora, Paneer, Pani Puri, Pav Bhaji, Vada Pav.

## Results

| Model | Classes | Test accuracy |
|---|---|---|
| Indian food (MobileNetV3-Small) | 15 | XX.X% |

<!-- TODO: replace XX.X% with your measured accuracy on a held-out test set. Delete this table if you have not measured it yet. -->

## Run locally

```bash
git clone https://github.com/shivamm-15/NutriVision.git
cd NutriVision
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Training and evaluation (Indian model)

```bash
python scripts/download_indian_dataset.py   # fetch the dataset into indian_food_dataset/
python scripts/train_indian_food.py         # fine-tune MobileNetV3-Small
python scripts/evaluate_indian_food.py      # evaluate the trained model
python scripts/test_indian_food.py          # predict on a single image
pytest tests/                               # check that the models load and predict
```

<!-- TODO: after running the evaluation, save the confusion matrix and uncomment:
![Confusion matrix](models/indian_confusion_matrix.png)
-->

Project layout: `app.py` (Streamlit app), `src/` (calorie mapping and Food-101 code), `models/` (trained weights and class lists), `scripts/` (dataset, training and evaluation), `tests/`.

The model is fine-tuned from ImageNet-pretrained weights with augmentation (flip, rotation, color jitter). Datasets are not stored in this repository.

## How it works

1. The user selects Indian or Global mode and uploads an image.
2. The matching MobileNetV3 model returns the top-3 classes with confidence.
3. Calories and macros per 100 g are looked up for the predicted class and scaled by the portion weight.

## Limitations

- Each model can only choose among its own classes, so unrelated images still get a prediction. Low confidence is flagged.
- Calories and macros are approximate per-100 g values; real values vary with recipe, oil and serving.
- Portion weight is entered by the user and is not estimated from the image.
- The daily log lasts for the browser session only.
- Not medical or dietary advice.

Released under the MIT License. See [LICENSE](LICENSE) for the original copyright notice and the notice for this repository's modifications.
