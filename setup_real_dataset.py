# setup_real_dataset.py - Download and setup real plant disease dataset
import os
import requests
import zipfile
from pathlib import Path
import shutil

def download_plantvillage_dataset():
    """Download PlantVillage dataset for real disease detection"""
    print("🌿 Setting up Real Plant Disease Dataset")
    print("="*50)
    
    # Create directories
    os.makedirs('datasets/real_data', exist_ok=True)
    
    # PlantVillage dataset info
    dataset_info = {
        'name': 'PlantVillage Dataset',
        'description': 'Real plant disease images from PlantVillage',
        'classes': [
            'Apple___Apple_scab',
            'Apple___Black_rot', 
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    }
    
    print(f"📊 Dataset: {dataset_info['name']}")
    print(f"🎯 Classes: {len(dataset_info['classes'])}")
    
    # Instructions for manual download (Kaggle requires authentication)
    print("\n📥 DATASET DOWNLOAD INSTRUCTIONS:")
    print("="*50)
    print("1. Go to: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset")
    print("2. Download the dataset (requires Kaggle account)")
    print("3. Extract to: datasets/real_data/")
    print("4. Run: python train_real_model.py")
    
    # Alternative: Create sample structure
    print("\n🔧 Creating sample structure...")
    for class_name in dataset_info['classes'][:6]:  # Create first 6 classes
        train_dir = f"datasets/real_data/train/{class_name}"
        val_dir = f"datasets/real_data/validation/{class_name}"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
    
    print("✅ Sample structure created!")
    
    return dataset_info

def create_real_training_script():
    """Create script for training on real data"""
    
    training_script = '''# train_real_model.py - Train on real PlantVillage dataset
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import pickle

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

def create_data_generators():
    """Create data generators for training"""
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        'datasets/real_data/train',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        'datasets/real_data/validation',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

def create_advanced_model(num_classes):
    """Create advanced CNN model"""
    
    # Use EfficientNetB3 for better accuracy
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Fine-tuning: unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use different learning rates for base and top
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_3_accuracy']
    )
    
    return model

def train_model():
    """Train the model on real data"""
    
    print("🚀 Starting Real Model Training")
    print("="*40)
    
    # Check if data exists
    if not os.path.exists('datasets/real_data/train'):
        print("❌ Real dataset not found!")
        print("Please download PlantVillage dataset first.")
        return None
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    
    # Get number of classes
    num_classes = len(train_gen.class_indices)
    print(f"📊 Training on {num_classes} disease classes")
    
    # Create model
    model = create_advanced_model(num_classes)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/plant_disease_model_real.h5')
    
    # Save class information
    disease_info = {
        'classes': list(train_gen.class_indices.keys()),
        'class_indices': train_gen.class_indices,
        'model_info': {
            'input_size': IMG_SIZE,
            'model_type': 'EfficientNetB3',
            'version': '3.0_real_data',
            'total_params': model.count_params()
        },
        'training_info': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'final_accuracy': max(history.history['accuracy']),
            'final_val_accuracy': max(history.history['val_accuracy'])
        }
    }
    
    with open('models/disease_info_real.pkl', 'wb') as f:
        pickle.dump(disease_info, f)
    
    print("✅ Real model training complete!")
    return model

if __name__ == '__main__':
    train_model()
'''
    
    with open('train_real_model.py', 'w') as f:
        f.write(training_script)
    
    print("✅ Real training script created: train_real_model.py")

if __name__ == '__main__':
    dataset_info = download_plantvillage_dataset()
    create_real_training_script()
    
    print("\n" + "="*50)
    print("🎯 NEXT STEPS:")
    print("="*50)
    print("1. Download real dataset from Kaggle")
    print("2. Extract to datasets/real_data/")
    print("3. Run: python train_real_model.py")
    print("4. Update app.py to use real model")
    print("\nFor now, run: python train_model.py (uses synthetic data)")