import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os

bert_path = "./pretrained/bert-base-uncased"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VisionEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(VisionEncoder, self).__init__()
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.load_state_dict(torch.load('./pretrained/resnet50.pth'))
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path, local_files_only=True)
        self.fc = nn.Linear(768, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.fc(cls_output)
        return x

class SLRModel(nn.Module):
    def __init__(self, embed_dim):
        super(SLRModel, self).__init__()
        self.vision_encoder = VisionEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        return image_features, text_features

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        labels = torch.arange(logits.size(0), device=image_features.device)
        loss_image = self.criterion(logits, labels)
        loss_text = self.criterion(logits.T, labels)
        loss = (loss_image + loss_text) / 2
        return loss

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels, texts in tqdm(dataloader):
        images = images.to(device)
        text_inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

        optimizer.zero_grad()

        image_features, text_features = model(images, input_ids, attention_mask)
        loss = criterion(image_features, text_features)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with open("./dataset/descriptions.txt", "r") as f:
        targets = [line.strip() for line in f.readlines()]

    for images, labels, texts in tqdm(dataloader):
        images = images.to(device)
        text_inputs = tokenizer(targets, return_tensors='pt', padding=True, truncation=True)
        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

        with torch.no_grad():
            image_features, text_features = model(images, input_ids, attention_mask)
            logits = torch.matmul(image_features, text_features.T)
            
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == labels.to(device)).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Classification Accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    embed_dim = 512
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=(50, 50), scale=(0.8, 1.2)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class TextImageDataset(datasets.ImageFolder):
        def __init__(self, root, mode, transform=None):
            if mode == 0:
                image_root = os.path.join(root, "train")
            elif mode == 1:
                image_root = os.path.join(root, "test")
            super().__init__(image_root, transform=transform)

            self.texts = []
            with open(os.path.join(root, "descriptions.txt"), "r") as f:
                self.texts = [line.strip() for line in f.readlines()]
            
            assert len(self.classes) == len(self.texts), "类别数量和描述数量不匹配"

        def __getitem__(self, index):
            image, label = super().__getitem__(index)
            text = self.texts[label]
            return image, label, text

    train_dataset = TextImageDataset(root="./dataset", mode=0, transform=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TextImageDataset(root="./dataset", mode=1, transform=test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = SLRModel(embed_dim).to(device)

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    tokenizer = BertTokenizer.from_pretrained(bert_path, local_files_only=True)

    best_acc = 0

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        avg_loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Loss: {avg_loss:.4f}")

        accuracy = test(model, test_dataloader, device)
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "best.pth")

    print(f"Best Accuracy: {best_acc:.4f}")
