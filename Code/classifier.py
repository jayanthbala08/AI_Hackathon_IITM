import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(csv_path, text_column='text', label_column='label', test_size=0.2, random_state=42):
    """
    Load and split the dataset into training and testing sets
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the labeled data
    text_column : str
        Name of the column containing the transcribed text
    label_column : str
        Name of the column containing the labels
    test_size : float
        Proportion of the dataset to include in the test split
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data for training and testing
    labels : list
        Unique labels in the dataset
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Print basic information about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:\n{df.head()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values:\n{missing_values}")
    
    # Drop rows with missing values in the text or label columns
    df = df.dropna(subset=[text_column, label_column])
    
    # Display class distribution
    print("\nClass distribution:")
    class_dist = df[label_column].value_counts()
    print(class_dist)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_column], 
        df[label_column], 
        test_size=test_size, 
        random_state=random_state,
        stratify=df[label_column]  # Maintain the same class distribution in train and test sets
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, df[label_column].unique()

def train_model(X_train, y_train, model_type='logistic', max_features=5000, ngram_range=(1, 2)):
    """
    Train a text classification model
    
    Parameters:
    -----------
    X_train : array-like
        Training text data
    y_train : array-like
        Training labels
    model_type : str
        Type of model to train ('logistic', 'rf', or 'svm')
    max_features : int
        Maximum number of features for the TF-IDF vectorizer
    ngram_range : tuple
        Range of n-grams to consider for the TF-IDF vectorizer
        
    Returns:
    --------
    model : sklearn Pipeline
        Trained classification model
    """
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        strip_accents='unicode',
        min_df=2
    )
    
    # Choose the classifier based on the specified model type
    if model_type == 'logistic':
        classifier = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    elif model_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'svm':
        classifier = SVC(
            C=1.0,
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create a pipeline combining the vectorizer and classifier
    model = Pipeline([
        ('tfidf', vectorizer),
        ('classifier', classifier)
    ])
    
    # Train the model
    print(f"Training {model_type} model...")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, labels):
    """
    Evaluate the trained model on the test set
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained classification model
    X_test : array-like
        Test text data
    y_test : array-like
        Test labels
    labels : list
        List of unique labels
        
    Returns:
    --------
    None
    """
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Display detailed classification report
    class_report = classification_report(y_test, y_pred, target_names=labels)
    print("\nClassification Report:")
    print(class_report)
    
    # Generate and display a confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print("\nConfusion matrix saved as 'confusion_matrix.png'")

def analyze_feature_importance(model, labels, top_n=20):
    """
    Analyze feature importance for the trained model
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained classification model
    labels : list
        List of unique labels
    top_n : int
        Number of top features to display for each class
        
    Returns:
    --------
    None
    """
    # Extract the vectorizer and classifier from the pipeline
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # For logistic regression and SVM with linear kernel
    if isinstance(classifier, LogisticRegression) or (isinstance(classifier, SVC) and classifier.kernel == 'linear'):
        if isinstance(classifier, LogisticRegression):
            coefs = classifier.coef_
        else:  # SVC with linear kernel
            coefs = classifier.coef_
            
        # For binary classification
        if coefs.shape[0] == 1:
            # Sort features by importance
            top_indices = np.argsort(coefs[0])
            
            # Display most negative and most positive features
            print("\nTop negative features:")
            for idx in top_indices[:top_n]:
                print(f"{feature_names[idx]}: {coefs[0][idx]:.4f}")
                
            print("\nTop positive features:")
            for idx in top_indices[-top_n:]:
                print(f"{feature_names[idx]}: {coefs[0][idx]:.4f}")
        
        # For multiclass classification
        else:
            for i, label in enumerate(labels):
                # Sort features by importance for this class
                top_indices = np.argsort(coefs[i])[-top_n:]
                
                print(f"\nTop features for class '{label}':")
                for idx in top_indices:
                    print(f"{feature_names[idx]}: {coefs[i][idx]:.4f}")
    
    # For random forest
    elif isinstance(classifier, RandomForestClassifier):
        # Get feature importances
        importances = classifier.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[-top_n:]
        
        # Display most important features
        print("\nTop features by importance:")
        for idx in indices:
            print(f"{feature_names[idx]}: {importances[idx]:.4f}")
    
    else:
        print("\nFeature importance analysis not supported for this model type.")

def save_model(model, model_path='voice_text_classifier.joblib'):
    """
    Save the trained model to disk
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained classification model
    model_path : str
        Path to save the model
        
    Returns:
    --------
    None
    """
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

def predict_new_text(model, text, labels):
    """
    Make a prediction for new text
    
    Parameters:
    -----------
    model : sklearn Pipeline
        Trained classification model
    text : str or list
        New text to classify
    labels : list
        List of unique labels
        
    Returns:
    --------
    dict
        Prediction results including the predicted label and probabilities
    """
    # Convert single text to list
    if isinstance(text, str):
        text = [text]
    
    # Make predictions
    predicted_labels = model.predict(text)
    probabilities = model.predict_proba(text)
    
    results = []
    for i, (label, probs) in enumerate(zip(predicted_labels, probabilities)):
        # Create a dictionary mapping labels to their probabilities
        proba_dict = {labels[j]: probs[j] for j in range(len(labels))}
        
        # Sort probabilities in descending order
        sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        
        results.append({
            'text': text[i],
            'predicted_label': label,
            'probabilities': sorted_proba
        })
    
    return results[0] if len(results) == 1 else results

def main():
    """
    Main function to run the classification pipeline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate a text classification model for voice-to-text data')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing the labeled data')
    parser.add_argument('--text_column', type=str, default='text', help='Name of the column containing the transcribed text')
    parser.add_argument('--label_column', type=str, default='label', help='Name of the column containing the labels')
    parser.add_argument('--model_type', type=str, default='logistic', choices=['logistic', 'rf', 'svm'], help='Type of model to train')
    parser.add_argument('--model_path', type=str, default='voice_text_classifier.joblib', help='Path to save the trained model')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    
    args = parser.parse_args()
    
    # Load and split the data
    X_train, X_test, y_train, y_test, labels = load_data(
        args.csv_path,
        args.text_column,
        args.label_column,
        args.test_size
    )
    
    # Train the model
    model = train_model(X_train, y_train, args.model_type)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test, labels)
    
    # Analyze feature importance
    analyze_feature_importance(model, labels)
    
    # Save the model
    save_model(model, args.model_path)
    
    # Example of how to use the model for prediction
    print("\nExample prediction:")
    example_text = X_test.iloc[0]
    prediction = predict_new_text(model, example_text, labels)
    print(f"Text: {example_text}")
    print(f"True label: {y_test.iloc[0]}")
    print(f"Predicted label: {prediction['predicted_label']}")
    print("Probabilities:")
    for label, prob in prediction['probabilities']:
        print(f"  {label}: {prob:.4f}")

if __name__ == "__main__":
    main()
