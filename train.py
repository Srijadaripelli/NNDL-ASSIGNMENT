import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, AdamW
from src.dataset import NERDataset
from src.model import BertBiLSTMCRF

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, id2label):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            pred = model(input_ids, attention_mask)
            for p, l, mask in zip(pred, labels, attention_mask):
                active_labels = l[mask.bool()].cpu().numpy()
                predictions.append([id2label[i] for i in p])
                true_labels.append([id2label[i] for i in active_labels])
    return predictions, true_labels

def main():
    MAX_LEN = 128
    BATCH_SIZE = 16
    LSTM_HIDDEN_DIM = 256
    EPOCHS = 5
    LEARNING_RATE = 2e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    # Dummy data: replace this with your own file loading.
    texts = [["Hello", "world"], ["John", "lives", "in", "London"]]
    labels = [["O", "O"], ["B-PER", "O", "O", "B-LOC"]]

    label_list = sorted(list(set([label for sublist in labels for label in sublist])))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    dataset = NERDataset(texts, labels, tokenizer, MAX_LEN, label2id)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertBiLSTMCRF('bert-base-cased', LSTM_HIDDEN_DIM, len(label_list)).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        avg_loss = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main()
