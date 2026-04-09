import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pickle

# Disease classes
DISEASE_CLASSES = [
    'Tomato_Early_Blight',
    'Tomato_Late_Blight', 
    'Powdery_Mildew',
    'Leaf_Spot',
    'Healthy_Plant'
]
NUM_CLASSES = len(DISEASE_CLASSES)
CLASS_INDICES = {disease: i for i, disease in enumerate(DISEASE_CLASSES)}

def generate_dummy_data(num_samples=1000, img_size=(224, 224)):
    """Generate dummy training data for demonstration"""
    images = []
    labels = []
    
    for i, disease in enumerate(DISEASE_CLASSES):
        for _ in range(num_samples // NUM_CLASSES):
            # Generate random images with slight variations per class
            img = np.random.rand(*img_size, 3) * 255
            if disease == 'Tomato_Early_Blight':
                img[:, :, 1] += np.random.rand() * 50  # Add green variation
            elif disease == 'Tomato_Late_Blight':
                img[:, :, 2] -= np.random.rand() * 30  # Reduce blue
            elif disease == 'Powdery_Mildew':
                img[:, :, 0] += np.random.rand() * 40  # Add red
            elif disease == 'Leaf_Spot':
                img[:, :, 1] -= np.random.rand() * 25  # Reduce green
            # Healthy_Plant: no change
            
            images.append(img.astype(np.uint8))
            labels.append(i)
    
    return np.array(images), np.array(labels)

def build_model(input_shape=(224, 224, 3)):
    """Build CNN model"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Train and save the model"""
    print("Generating training data...")
    images, labels = generate_dummy_data(num_samples=2000)
    
    # Normalize images
    images = images.astype(np.float32) / 255.0
    
    # One-hot encode labels
    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model()
    
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.2f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/plant_disease_model.h5')
    
    # Save disease info
    disease_info = {
        'classes': DISEASE_CLASSES,
        'class_indices': CLASS_INDICES,
        'model_info': {
            'model_type': 'CNN',
            'input_size': (224, 224),
            'version': '1.0',
            'accuracy': f"{test_acc:.2f}"
        }
    }
    
    with open('models/disease_info.pkl', 'wb') as f:
        pickle.dump(disease_info, f)
    
    print("Model saved to models/plant_disease_model.h5")
    print("Disease info saved to models/disease_info.pkl")

if __name__ == '__main__':
    train_model()