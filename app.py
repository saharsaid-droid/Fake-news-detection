import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import os
import argparse

def load_data(fake_path="Fake.csv", real_path="True.csv"):
    """Load and prepare the dataset"""
    print("Loading datasets...")
    fake_news = pd.read_csv(fake_path)
    real_news = pd.read_csv(real_path)
    
    # Add labels
    fake_news['label'] = 0  # Fake
    real_news['label'] = 1  # Real
    
    # Combine datasets
    df = pd.concat([fake_news, real_news], ignore_index=True)
    
    # Use 'text' column
    df = df[['text', 'label']].dropna()
    
    print(f"Total articles: {len(df)}")
    print(f"Fake articles: {len(df[df['label'] == 0])}")
    print(f"Real articles: {len(df[df['label'] == 1])}")
    
    return df

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# def train_bert(df, output_dir="./bert_model", epochs=3, batch_size=8):
#     """Train BERT model"""
#     print("Training BERT model...")
    
#     # Split data
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         df['text'].tolist(),
#         df['label'].tolist(),
#         test_size=0.2,
#         random_state=42,
#         stratify=df['label']
#     )
    
#     # Load tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
#     # Tokenize
#     train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
#     val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
#     # Create datasets
#     train_dataset = NewsDataset(train_encodings, train_labels)
#     val_dataset = NewsDataset(val_encodings, val_labels)
    
#     # Training arguments
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=epochs,
#         per_device_train_batch_size=batch_size,
#         per_device_eval_batch_size=batch_size,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir=f'{output_dir}/logs',
#         logging_steps=10,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#     )
    
#     # Create trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,
#         compute_metrics=compute_metrics,
#     )
    
#     # Train
#     trainer.train()
    
#     # Save model
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
    
#     print(f"BERT model saved to {output_dir}")
#     return trainer.evaluate()

def train_distilbert(df, output_dir="./distilbert_model", epochs=3, batch_size=8):
    """Train DistilBERT model"""
    print("Training DistilBERT model...")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    
    # Create datasets
    train_dataset = NewsDataset(train_encodings, train_labels)
    val_dataset = NewsDataset(val_encodings, val_labels)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"DistilBERT model saved to {output_dir}")
    return trainer.evaluate()

def main():
    parser = argparse.ArgumentParser(description='Train BERT and DistilBERT models for fake news detection')
    parser.add_argument('--fake_csv', default='Fake.csv', help='Path to fake news CSV file')
    parser.add_argument('--real_csv', default='True.csv', help='Path to real news CSV file')
    parser.add_argument('--model', choices=['bert', 'distilbert', 'both'], default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    
    args = parser.parse_args()
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    df = load_data(args.fake_csv, args.real_csv)
    
    results = {}
    
    # if args.model in ['bert', 'both']:
    #     bert_results = train_bert(df, epochs=args.epochs, batch_size=args.batch_size)
    #     results['bert'] = bert_results
        
    if args.model in ['distilbert', 'both']:
        distilbert_results = train_distilbert(df, epochs=args.epochs, batch_size=args.batch_size)
        results['distilbert'] = distilbert_results
    
    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        for metric, value in result.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()

