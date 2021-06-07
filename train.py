import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from data import read_pheme, read_politifact, read_buzzfeed
from metrics import evaluate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SEED = 123
torch.manual_seed(SEED)

class Head(nn.Module):
    def __init__(self, hidden_dim):
        super(Head, self).__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        h = self.dropout(x)
        o = self.classifier(h)
        return o

def exam(data_loader, classifier, criteria, prints):
    classifier.eval()
    total_loss = 0.0
    targets, preds = [], []
    with torch.no_grad():
        for ids, features, labels in data_loader:
            logits = classifier(features.to(device))
            loss = criteria(logits.cpu(), labels)
            total_loss += loss.item()
            targets.extend(labels)
            preds.extend(torch.max(logits, dim=-1).indices.tolist()) 
    return evaluate(targets, preds, prints), total_loss / len(targets)

def main():
    classifier = Head(hidden_dim).to(device)
    optimizer = SGD(classifier.parameters(), lr=lr, momentum=momentum)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criteria = torch.nn.CrossEntropyLoss()
    
    train, valid, test = read_input()
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_performance, best_epoch = 0.0, -1
    for i_epoch in trange(num_epochs, desc='epochs'):
        # Train
        classifier.train()
        train_loss = 0.0
        targets, preds = [], []
        for ids, features, labels in train_dataloader:
            optimizer.zero_grad()
            logits = classifier(features.to(device))
            loss = criteria(logits.cpu(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            targets.extend(labels)
            preds.extend(torch.max(logits, dim=-1).indices.tolist()) 
        scheduler.step()
        train_loss /= len(targets)
        
        # Validate
        result, valid_loss = exam(valid_dataloader, classifier, criteria, prints=False)
        if result[0] > best_performance:
            best_performance = result[0]
            best_epoch = i_epoch
            torch.save(classifier, f'./models/C-{name}-{i_epoch}')
        
        if i_epoch % 10 == 9:
            print('Epoch {} Train loss {:11.4f} Valid loss {:11.4f} Valid acc {:.4f}'.format(i_epoch, train_loss, valid_loss, result[0]))
    
    # Test
    print(f"==== Best epoch {best_epoch} ====")
    classifier = torch.load(f'./models/C-{name}-{best_epoch}').to(device)
    result, test_loss = exam(test_dataloader, classifier.to(device), criteria, prints=True)
    print('Test loss {:.4f}'.format(test_loss))
        

if __name__ == "__main__":
    for name in ['buzzfeed']:  # 'politifact', 'pheme', 'buzzfeed'
        if name == "politifact":  # Test Acc 0.8557
            read_input = read_politifact
            hidden_dim = 768 + 5
            batch_size = 32
            model_name = 'bert-base-cased'
            num_epochs = 150
            lr = 0.005
            momentum = 0.9
            step_size = 20
            gamma = 0.5
            hidden_dropout_prob = 0.15
        if name == "pheme":  # Test Acc 0.7558
            read_input = read_pheme
            hidden_dim = 768 + 7
            batch_size = 32
            model_name = 'bert-base-cased'
            num_epochs = 100
            lr = 0.008
            momentum = 0.9
            step_size = 5
            gamma = 0.9
            hidden_dropout_prob = 0.1
        if name == "buzzfeed":
            read_input = read_buzzfeed
            hidden_dim = 768 + 29 + 2
            batch_size = 32
            model_name = 'bert-base-cased'
            num_epochs = 300
            lr = 0.01
            momentum = 0.9
            step_size = 20
            gamma = 0.5
            hidden_dropout_prob = 0.08

        main()