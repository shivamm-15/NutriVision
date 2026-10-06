# NutriVision - Food-101 model training pipeline
import argparse
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

def cleanup_old_models(model_prefix):
    """Remove existing model files"""
    models_dir = 'models'
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if model_prefix in file and file.endswith('.pt'):
                file_path = os.path.join(models_dir, file)
                os.remove(file_path)
                print(f"🗑️  Removed old model: {file_path}")

def get_class_weights(dataset):
    """Calculate class weights for balanced training"""
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    # Calculate weights inversely proportional to class frequency
    weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * class_counts[i])
        weights.append(weight)
    
    return torch.FloatTensor(weights)

def create_weighted_sampler(dataset):
    """Create weighted sampler for balanced training"""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(targets)
    
    # Calculate sample weights
    sample_weights = []
    for target in targets:
        weight = 1.0 / class_counts[target]
        sample_weights.append(weight)
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

# Argument parsing - For Food-101 MobileNet training
parser = argparse.ArgumentParser(description='Food-101 MobileNet Training Script')
parser.add_argument('--n_classes', type=int, required=False, default=101, help='Number of food classes')
parser.add_argument('--train_dir', type=str, required=False, default='Food 101 dataset/train/train', help='Training directory path')
parser.add_argument('--val_dir', type=str, required=False, default='Food 101 dataset/test/test', help='Validation directory path')
parser.add_argument('--model_prefix', type=str, required=True, help='Model prefix for saving')
parser.add_argument('--epochs', type=int, required=False, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size for training and validation')
args = parser.parse_args()

# CUDA device check and info
if torch.cuda.is_available():
    print(f"🚀 CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CUDA is NOT available. Training will use CPU and may be very slow.")

# MOBILENET CONFIGURATION FOR FOOD-101 DATASET
MOBILENET_CONFIG = {
    'batch_size': 32,            # Smaller batch for MobileNet stability
    'epochs': 100,              # Reasonable epochs for MobileNet
    'base_lr': 0.001,           # Learning rate for MobileNet
    'weight_decay': 1e-4,       # Regularization
    'dropout': 0.3,             # Dropout for MobileNet
    'patience': 15,             # Patience for early stopping
    'min_delta': 0.001,         # Improvement threshold
    'warmup_epochs': 5,         # Warmup for stability
    'use_mixed_precision': True, # For faster training
    'gradient_clip': 1.0,       # Prevent exploding gradients
    'label_smoothing': 0.1,     # Better generalization
    'use_ema': True,            # Exponential moving average
    'ema_decay': 0.999          # EMA decay rate
}

print("🍕 MOBILENET FOOD-101 TRAINING CONFIGURATION 🍕")
print("="*70)
print(f"🎯 Dataset: Food-101 ({args.n_classes} classes)")
print(f"📦 Batch size: {MOBILENET_CONFIG['batch_size']}")
print(f"🔄 Max epochs: {MOBILENET_CONFIG['epochs']}")
print(f"⚡ Learning rate: {MOBILENET_CONFIG['base_lr']}")
print(f"🛡️  Weight decay: {MOBILENET_CONFIG['weight_decay']}")
print(f"💧 Dropout: {MOBILENET_CONFIG['dropout']}")
print(f"⏳ Patience: {MOBILENET_CONFIG['patience']}")
print(f"🎭 Label smoothing: {MOBILENET_CONFIG['label_smoothing']}")
print(f"🔄 Mixed precision: {MOBILENET_CONFIG['use_mixed_precision']}")
print(f"📊 EMA: {MOBILENET_CONFIG['use_ema']}")
print("="*70)

# MobileNet data transforms for Food-101
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load datasets
print("📂 Loading Food-101 dataset...")
train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)

# Analyze class distribution
print(f"📊 Training samples: {len(train_dataset)}")
print(f"📊 Validation samples: {len(val_dataset)}")
print(f"🏷️  Number of classes: {len(train_dataset.classes)}")

train_class_counts = Counter([train_dataset[i][1] for i in range(len(train_dataset))])
print(f"📈 Average samples per class: {len(train_dataset) // len(train_dataset.classes)}")
print(f"📉 Min samples in class: {min(train_class_counts.values())}")
print(f"📈 Max samples in class: {max(train_class_counts.values())}")

# Create data loaders with balanced sampling
train_sampler = create_weighted_sampler(train_dataset)
train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    sampler=train_sampler, 
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=0,
    pin_memory=True,
    persistent_workers=False
)

# MobileNet model architecture for Food-101
class MobileNetFood101Model(nn.Module):
    def __init__(self, n_classes, dropout_rate=0.3):
        super(MobileNetFood101Model, self).__init__()
        # Use MobileNet V3 Large for food classification
        self.backbone = models.mobilenet_v3_large(weights='IMAGENET1K_V1')
        
        # Get number of features from the last layer
        num_features = self.backbone.classifier[-1].in_features
        
        # Remove the last layer of the classifier
        self.backbone.classifier = nn.Sequential(*list(self.backbone.classifier.children())[:-1])
        
        # MobileNet classifier for food recognition
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
            nn.Linear(256, n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Exponential Moving Average for better stability
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 Using device: {device}")
if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name()}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Early stopping for MobileNet
class MobileNetEarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_acc, model, epoch):
        if self.best_score is None:
            self.best_score = val_acc
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🛑 Early stopping! Best accuracy: {self.best_score:.4f} at epoch {self.best_epoch}")
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_score = val_acc
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def evaluate_model(model, val_loader, device, class_names):
    """MobileNet model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm, all_probs

def save_mobilenet_info(model_prefix, class_names, best_acc, report, config):
    """Save MobileNet training information"""
    info = {
        'model_prefix': model_prefix,
        'dataset': 'Food-101',
        'classes': class_names,
        'n_classes': len(class_names),
        'best_validation_accuracy': float(best_acc),
        'training_config': config,
        'classification_report': report,
        'model_architecture': 'MobileNet-V3-Large Food Classifier',
        'training_date': datetime.now().isoformat(),
        'framework': 'PyTorch MobileNet Training'
    }
    
    os.makedirs('models', exist_ok=True)
    with open(f'models/{model_prefix}_MOBILENET_info.json', 'w') as f:
        json.dump(info, f, indent=2)

def plot_mobilenet_confusion_matrix(cm, class_names, model_prefix):
    """Plot MobileNet confusion matrix"""
    plt.figure(figsize=(20, 16))
    
    # For large number of classes, use smaller font
    font_size = max(4, min(8, 80 // len(class_names)))
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'shrink': 0.8})
    
    plt.title(f'MobileNet Food-101 Confusion Matrix - {model_prefix}', fontsize=16, fontweight='bold')
    plt.ylabel('True Food Class', fontsize=12)
    plt.xlabel('Predicted Food Class', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=font_size)
    plt.yticks(rotation=0, fontsize=font_size)
    plt.tight_layout()
    
    os.makedirs('models', exist_ok=True)
    plt.savefig(f'models/{model_prefix}_MOBILENET_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

# MobileNet training function
def mobilenet_train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, model_prefix, class_names, config):
    best_val_acc = 0.0
    early_stopping = MobileNetEarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    
    # EMA setup
    ema = EMA(model, decay=config['ema_decay']) if config['use_ema'] else None
    if ema:
        ema.register()
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler() if config['use_mixed_precision'] else None
    
    # Create directory for saving models
    os.makedirs('models', exist_ok=True)
    
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': [],
        'epochs': []
    }
    
    print(f"🚀 Starting MobileNet Food-101 training...")
    print(f"⏰ Estimated time: {epochs * len(train_loader) * config['batch_size'] / 1000:.1f} minutes")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for batch_idx, (images, labels) in train_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['gradient_clip'])
                optimizer.step()
            
            # Update EMA
            if ema:
                ema.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update tqdm bar
            train_bar.set_postfix({"loss": loss.item()})
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Apply EMA for validation
        if ema:
            ema.apply_shadow()
        
        val_bar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_bar.set_postfix({"loss": loss.item()})
        
        # Restore original weights
        if ema:
            ema.restore()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        if isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_history['epochs'].append(epoch + 1)
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"🍕 EPOCH {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")
        print(f"📈 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"📊 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"⚡ LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Apply EMA for evaluation
            if ema:
                ema.apply_shadow()
            
            # Detailed evaluation
            detailed_acc, report, cm, probs = evaluate_model(model, val_loader, device, class_names)
            
            # Save MobileNet model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'class_names': class_names,
                'n_classes': len(class_names),
                'model_architecture': 'MobileNet-V3-Large',
                'training_config': config,
                'ema_state': ema.shadow if ema else None,
                'best_epoch': epoch + 1,
                'total_training_time': total_time
            }, f"models/MOBILENET_best_model_{model_prefix}.pt")
            
            print(f"🏆 NEW BEST MODEL! Accuracy: {best_val_acc:.4f}")
            
            # Save info and confusion matrix
            save_mobilenet_info(model_prefix, class_names, best_val_acc, report, config)
            plot_mobilenet_confusion_matrix(cm, class_names, model_prefix)
            
            # Restore weights
            if ema:
                ema.restore()
        
        # Early stopping
        early_stopping(val_acc, model, epoch + 1)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping at epoch {epoch+1}")
            break
        
        print("="*70)
    
    total_training_time = time.time() - start_time
    print(f"🎉 MOBILENET TRAINING COMPLETED!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"⏰ Total training time: {total_training_time/3600:.2f} hours")
    
    # Save final training history
    training_history['total_training_time'] = total_training_time
    with open(f'models/{model_prefix}_MOBILENET_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    return best_val_acc

if __name__ == "__main__":
    # Clean up old models
    cleanup_old_models(args.model_prefix)
    
    print("🍕 MOBILENET FOOD-101 TRAINING INITIATED 🍕")
    print("="*70)
    
    # Create MobileNet model
    model = MobileNetFood101Model(args.n_classes, MOBILENET_CONFIG['dropout']).to(device)
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    
    # MobileNet loss with label smoothing
    class_weights = get_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=MOBILENET_CONFIG['label_smoothing']
    )
    
    # MobileNet optimizer with different learning rates
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            classifier_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': MOBILENET_CONFIG['base_lr'] * 0.1},
        {'params': classifier_params, 'lr': MOBILENET_CONFIG['base_lr']}
    ], weight_decay=MOBILENET_CONFIG['weight_decay'])
    
    # MobileNet scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Start MobileNet training
    best_acc = mobilenet_train_and_validate(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, args.epochs, args.model_prefix, train_dataset.classes, MOBILENET_CONFIG
    )
    
    print(f"\n🎯 MOBILENET RESULTS:")
    print(f"🏆 Best validation accuracy: {best_acc:.4f}")
    print(f"💾 Model: models/MOBILENET_best_model_{args.model_prefix}.pt")
    print(f"📊 Info: models/{args.model_prefix}_MOBILENET_info.json")
    print(f"📈 Matrix: models/{args.model_prefix}_MOBILENET_confusion_matrix.png")
    print(f"📜 History: models/{args.model_prefix}_MOBILENET_training_history.json")
    print("="*70)
    print("✅ MOBILENET FOOD-101 TRAINING COMPLETE!")
    print("🍕 YOU NOW HAVE A MOBILE FOOD CLASSIFIER! 🍕")
    print("="*70)