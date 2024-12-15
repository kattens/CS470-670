import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import os

# Define emotion labels
EMOTION_LABELS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        return mfcc.T
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def parse_filename(filename):
    """Parse RAVDESS filename and return emotion label."""
    parts = filename.split('-')
    return parts[2]  # Emotion label is the third part

def load_test_data(test_data_path):
    """Load and prepare test data."""
    features = []
    labels = []
    target_length = 200  # Match the training input shape

    print("Loading test data...")
    for actor_dir in os.listdir(test_data_path):
        if not actor_dir.startswith('Actor_'):
            continue
            
        actor_path = os.path.join(test_data_path, actor_dir)
        for filename in os.listdir(actor_path):
            if not filename.endswith('.wav'):
                continue
                
            file_path = os.path.join(actor_path, filename)
            
            # Extract features
            audio_features = extract_features(file_path)
            if audio_features is None:
                continue
            
            # Pad or truncate to target length
            if audio_features.shape[0] > target_length:
                audio_features = audio_features[:target_length]
            else:
                pad_width = ((0, target_length - audio_features.shape[0]), (0, 0))
                audio_features = np.pad(audio_features, pad_width, mode='constant')
            
            # Get emotion label
            emotion_label = parse_filename(filename)
            
            features.append(audio_features)
            labels.append(emotion_label)
    
    features = np.array(features)
    print(f"Feature shape: {features.shape}")
    return features, labels

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_class_distribution(labels, label_names):
    """Plot distribution of classes in dataset."""
    plt.figure(figsize=(12, 6))
    
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Create a dictionary mapping labels to their counts
    label_counts = dict(zip(unique_labels, counts))
    
    # Create lists for plotting
    plot_labels = []
    plot_counts = []
    
    # Match counts with human-readable labels
    for i, name in enumerate(label_names):
        label_id = f"{i+1:02d}"  # Convert index to label ID format (01, 02, etc.)
        count = label_counts.get(label_id, 0)  # Get count or 0 if label not found
        plot_labels.append(name)
        plot_counts.append(count)
    
    # Create bar plot
    plt.bar(plot_labels, plot_counts)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add count labels on top of each bar
    for i, count in enumerate(plot_counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.savefig('class_distribution.png')
    plt.close()

def plot_prediction_confidence(predictions, true_labels):
    """Plot confidence scores for predictions."""
    confidences = np.max(predictions, axis=1)
    correct = np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.hist([confidences[correct], confidences[~correct]], 
             label=['Correct Predictions', 'Incorrect Predictions'],
             bins=20, alpha=0.7)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('confidence_distribution.png')
    plt.close()

def evaluate_model(model_path, encoder_path, test_data_path):
    """Evaluate the model and generate visualizations."""
    # Load model and encoder
    print("Loading model and encoder...")
    model = load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Load test data
    features, labels = load_test_data(test_data_path)
    
    # Convert labels to numerical form
    numerical_labels = label_encoder.transform(labels)
    one_hot_labels = np.eye(len(np.unique(numerical_labels)))[numerical_labels]
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(features, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_label_indices = np.argmax(one_hot_labels, axis=1)
    
    # Get label names
    label_names = [EMOTION_LABELS[str(i+1).zfill(2)] for i in range(len(np.unique(numerical_labels)))]
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(true_label_indices, predicted_labels, label_names)
    
    # Plot class distribution
    print("Generating class distribution plot...")
    plot_class_distribution(labels, label_names)
    
    # Plot prediction confidence
    print("Generating confidence distribution plot...")
    plot_prediction_confidence(predictions, one_hot_labels)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_label_indices, predicted_labels, 
                              target_names=label_names))

    # Save results to a text file
    with open('evaluation_results.txt', 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(true_label_indices, predicted_labels, 
                                    target_names=label_names))

if __name__ == "__main__":
    MODEL_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/emotion_model.keras"
    ENCODER_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/label_encoder.joblib"
    TEST_DATA_PATH = "/Users/houn/Documents/Desktop/FALL2024/CS470/CS470-670/src/training/RAVDESS"
    
    evaluate_model(MODEL_PATH, ENCODER_PATH, TEST_DATA_PATH)