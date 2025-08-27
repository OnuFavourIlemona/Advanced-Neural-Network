#Importing needed Libraries 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np 
from scipy.stats import entropy 
from datasets import load_dataset
import networkx as nx 
from touchtext.vocab import build_vocab_from_iterator
import os
import pandas as pd 
from PIL import Image
import copy

#set random seed and device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_avaliable() else "cpu")

# 1. Parent CNN for mnist data
class ParentCNN(nn.Module):
    def __init__(self):
        super(ParentCNN, self).__init__()
        self.conv1 = nn.Con2d(1, 16, kernel_size=3, paddings=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, paddings=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) #10 classes for Mnist data
        self.relu = nn.RELU()

    def foward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x

    def get_features(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        return x


#2. Child RNN for sequence Data
class ChildRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, parent_feautures=None):
        super(ChildRNN, self).__innit__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.ltsm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        if parent_features is not None:
            self.feature_mapper.weight = nn.Linear(128, hidden_size)
            with torch.no_grad():
                self.feature_mapper.weight.data = torch.randn(hidden_size, 128) * 0.01
        else:
            self.feature_mapper = None

    def forward(self, x):
        x = self.embedding(x)
        if self.feature_mapper is not None:
            x = x + self.feature_mapper(torch.zeros(1, 128).to(x.device)).unsqueeze(0) 
            _, (h_n, _) = self.lstm(x)
            x = self.relu(h_n[-1])
            x = self.fc(x)
            return x

# 3. Child CNN for Different image data
class ChildCNN(nn.Module):
    def __init__(self, parent_features=None):
        super(ChildCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)                    
        self.fc2 = nn.Linear(128, 10) #10 classes
        self.relu = nn.ReLU()
        if parent_features is not None:
            self.feature_mapper = nn.Linear(128, 128)
            with torch.no_grad():
                self.feature_mapper.weight.data = torch.eye(128)
        else:
            self.feature_mapper = None

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu(self.fc1(x))
        if self.feature_mapper is not None:
            x = x + self.feature_mapper(torch.zeros(1, 128).to(x.device))
        x = self.fc2(x)
        return x

# 4. Data difference check
def compute_kl_divergence(data1, data2, bins=100):
    data1_flat = data1.view(-1).cpu().numpy()
    data2_flat = data2.view(-1).cpu().numpy()
    hist1, _ = np.histogram(data1_flat, bins=bins, density=True)
    hist2, _ = np.histogram(data2_flat, bins=bins, density=True)
    return entropy(hist1 + 1e-10, hist2 + 1e-10)

# 5. Evaluate Model Performance
def evaluate_model(model, dataloader, data_type="image"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            if data_type == "sequence":
                data = data.long()
                outputs = model(data) 
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

#6. Network tree Manager
class NetworkTree:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.models = {}
        self.data_distribution = {}

    def add_node(self, node_id, model, dataset_name, data_info):
        self.graph.add_node(node_id, model=model, dataset_name=dataset_name)
        self.models[node_id] = model
        self.data_distribution[node_id] = data_info

    def add_edge(self, parent_id, child_id):
        self.graph.add_edge(parent_id, child_id)

    def prune_node(self, node_id, min_accuracy=50.0):
        accuracy = evaluate_model(self.models[node_id], self.data_distribution[node_id]["loader"], self.data_distribution[node_id]["type"])
        if accuracy < min_accuracy:
            self.graph.remove_node(node_id)
            del self.models[node_id]
            del self.data_distribution[node_id]
            return True
        return False    

#7. User data loaders
class UserImageDataset(Dataset):
    def __init__(self, image_folder, label_file, transfrom=None):
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = pd.read_csv(label_file) if label_file else None
        self.image_folder = image_folder

    def __len__(self):
        return len(self.image_files) 

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('L') #Grayscale for mnist-like, or rgb for others
        if image.size != (28, 28):
            image = image.resize((28, 28))
        image = transform.ToTensor()(image)
        if image.shape[0] == 3: #converts RGB to grayscale if needed
            image = image.mean(dim=0, keepdim=True)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = int(self.labels.iloc[idx]['label'])
        else:
            label = 0
        return image, label

class UserTextDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len=50):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.vocab.lookup_indicies(self.data.iloc[idx]['text'].split()[:self.max_len])
        text = text + [0] * (self.max_len - len(text))  # Padding
        label =int(self.data.iloc[idx]['label'])
        return torch.tensor(text, dtype=torch.long), label

# 8. Load Predefined Datasets                        
transform_mnist = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
mnist_loader = Dataloader(mnist_train, batch_size=64, shuffle=True)
mnist_test_loader = Dataloader(mnist_test, batch_size=64, shuffle=False)

#Vocabulary for text data
imdb = load_dataset("imdb")
def yeild_tokens(data):
    for text in data:
        yield text.split()

vocab = build_vocab_from_iterator(yeild_tokens(imdb['train']['text']), specials=["<unk>"], max_tokens=10000)
vocab.set_default_index(vocab["<unk>"])

# 9. Train Parent CNN
tree = NetworkTree()
parent_model = ParentCNN().to(device)
cirterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(parent_model.parameters(), lr=0.001)

print("Training Parent CNN on MNIST data...")
for epoch in range(2):
    parent_model.train()
    for data, target in mnist_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = parent_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

tree.add_model("parent", parent_model, "MNIST", {"loader": mnist_test_loader, "sample": next(iter(mnist_loader))[0], "type": "image"})   

# 10. spawning Logic
def spawn_child_models(parent_model, new_loader, new_data_sample, dataset_name, data_type, node_id, tree):
    kl_threshold = 0.5
    performance_threshold = 70.0
    parent_accuracy = evaluate_model(parent_model, new_loader, data_type)
    kl_div = compute_kl_divergence(tree.data_distribution["parent"]["sample"], new_data_sample)
    print(f"Parent Accuracy on {dataset_name}: {parent_accuracy:.2f}%, KL Divergence: {kl_div:.4f}")
    if parent_accuracy < performance_threshold and kl_div < kl_threshold:
        print(f"Spawning Child Models for {dataset_name}...")
        parent_features = parent_model.get_features(tree.data_distribution["parent"]["sample"].to(device)).mean(dim=0)
        if data_type == "sequence":
            child_model = ChildRNN(input_size=len(vocab), hidden_size=128, output_size=10, parent_features=parent_features).to(device)
        else:
            child_model = ChildCNN(parent_features=parent_features).to(device)

        child_optimizer = optim.Adam(child_model.parameters(), lr=0.001)
        for epoch in range(2):
            child_model.train()
            for data, target in new_loader:
                data, target = data.to(device), target.to(device)
                child_optimizer.zero_grad()
                output = child_model(data)
                loss = criterion(output, target)
                loss.backward()
                child_optimizer.step()

            print(f"Child Model {node_id} Epoch {epoch+1}, Loss: {loss.item():.4f}")
            tree.add_node(node_id, child_model, dataset_name, {"loader": new_loader, "sample": new_data_sample, "type": data_type})
            tree.add_edge("parent", node_id)
            return True
        return False    

# 11. Load User Data        
def load_user_data(image_folder= None, text_csv=None, transform=None):
    if image_folder:
        label_file = os.path.join(image_folder, 'labels.csv') if os.path.exists(os.path.join(image_folder, 'labels.csv')) else None
        user_dataset = UserImageDataset(image_folder, label_file, transform=transform_mnist)
        user_loader = Dataloader(user_dataset, batch_size=64, shuffle=False)
        data_type = "image" 
        sample = next(iter(user_loader))[0] if len(user_loader) > 0 else torch.zeros((1, 1, 28, 28))
        return user_loader, sample, data_type, "User_Images"
    elif text_csv:
        user_dataset = UserTextDataset(text_csv, vocab, max_len=50)
        user_loader = Dataloader(user_dataset, batch_size=64, shuffle=False)
        data_type = "sequence"
        sample = next(iter(user_loader))[0] if len(user_loader) > 0 else torch.zeros(1, 50, dtype=torch.long)
        return user_loader, sample, data_type, "User_Text"
    else:
        raise ValueError("Either image_folder or text_csv must be provided.")

#12. Testing phase with User Data
def test_system(tree, user_image_folder=None, user_text_csv=None):
    print("\n=== Testing phase ===")
    results = {}

    #Test on predefined datasets
    for node_id in tree.models:
        dataset_name = tree.graph.nodes[node_id]["dataset"]
        data_loader = tree.data_distribution[node_id]["loader"]
        data_type = tree.data_distribution[node_id]["type"]
        accuracy = evaluate_model(tree.models[node_id], data_loader, data_type)
        results[node_id] = {"dataset": dataset_name, "accuracy": accuracy}
        print(f"{node_id} ({dataset_name}) Accuracy: {accuracy:.2f}%")
  
    #Test on User provided data
    user_loaders = []
    if user_image_folder:
        user_loader, sample, data_type, dataset_name = load_user_data(image_folder=user_image_folder)
        user_loaders.append((user_loader, user_sample, data_type, dataset_name, "user_image"))
    if user_text_csv:
        user_loader, sample, data_type, dataset_name = load_user_data(text_csv=user_text_csv)
        user_loaders.append((user_loader, user_sample, data_type, dataset_name, "user_text"))
    for loader, sample, data_type, dataset_name, node_id in user_loaders:
        #Try parent on user data
        parent_accuracy = evaluate_model(tree.models["parent"], loader, data_type)
        print(f"Parent Accuracy on {dataset_name}: {parent_accuracy:.2f}%")

        #Spawn child models if needed
        if spawn_child_models(tree.models["parent"], loader, sample, dataset_name, data_type, node_id, tree):
            #Evaluate new child model
            accuracy = evaluate_model(tree.models[node_id], loader, data_type)
            results[node_id] = {"dataset": dataset_name, "accuracy": accuracy}
            print(f"{node_id} ({dataset_name}) Accuracy: {accuracy:.2f}%")
        else:
            results[node_id] = {"dataset": dataset_name, "accuracy": parent_accuracy} 

    #Collective Performance
    avg_accuracy = np.mean([r["accuracy"] for r in results.values() if r["accuracy"] > 0])
    print(f"System Average Accuracy: {avg_accuracy:.2f}%")  

    #Test Knowledge Transfer for user text data(if applicable)
    if user_text_csv and "user_text" in tree.models:
        print("\nTesting User Text without Knowledge Transfer...")
        no_transfer_model = ChildRNN(input_size=len(vocab), hidden_size=128, output_size=2).to(device)
        optimizer = optim.Adam(no_transfer_model.parameters(), lr=0.001)
        for epoch in range(2):
            no_transfer_model.train()
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data = data.long()
                optimizer.zero_grad()
                output = no_transfer_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            no_transfer_accuracy = evaluate_model(no_transfer_model, loader, "sequence")
            print(f"User Text No-Transfer Model Accuracy: {no_transfer_accuracy:.2f}%")
            print(f"Knowledge Transfer Benefit: {results['user_text']['accuracy'] - no_transfer_accuracy:.2f}%")

    #Prune Underperforming nodes
    for node_id in list(tree.models.keys()):
        if node_id != "parent" and tree.prune_node(node_id, min_accuracy=50.0):
            print(f"Pruned {node_id} due to low performance.")
    return results


