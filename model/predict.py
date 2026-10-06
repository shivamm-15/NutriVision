# NutriVision - Food prediction utility
import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import torch.nn as nn

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True, help='Path to the image')
parser.add_argument('--model', type=str, required=True, help='Path to the trained model .pt file')
parser.add_argument('--n_classes', type=int, required=True, help='Number of classes')
parser.add_argument('--class_names', type=str, nargs='+', required=True, help='List of class names in order')
args = parser.parse_args()

# Preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load image
img = Image.open(args.image).convert('RGB')
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Define ImprovedResNet to match training
class ImprovedResNet(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.5):
        super(ImprovedResNet, self).__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedResNet(args.n_classes, dropout_rate=0.5)

# Updated loading logic to handle checkpoints with 'model_state_dict'
checkpoint = torch.load(args.model, map_location=device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Predict
with torch.no_grad():
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    pred_class = args.class_names[predicted.item()]
    print(f"Predicted class: {pred_class}") 