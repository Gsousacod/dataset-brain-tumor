import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
#%pip install sklearn-model 
from torchvision import transforms
#%pip install torchvision 
import torch
#%pip install torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Caminhos das categorias de imagens
data_paths = {
    "glioma": "./glioma",
    "healthy": "./healthy",
    "meningioma": "./meningioma",
    "pituitary": "./pituitary"
}


# Número de imagens a exibir por categoria
num_images_per_class = 6

# Função para verificar se o diretório existe
def verificar_diretorio(path):
    caminho = Path(path)
    if caminho.exists() and caminho.is_dir():
        return [file for file in caminho.glob("*") if file.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    else:
        print(f"Erro: O diretório {path} não foi encontrado.")
        return []

# Função para exibir as imagens
def exibir_imagens(data_paths, num_images_per_class):
    """Exibe imagens de cada categoria selecionada."""
    # Cria a figura para exibir as imagens
    plt.figure(figsize=(12, 8))
    
    # Itera sobre as categorias de imagens
    for idx, (label, path) in enumerate(data_paths.items()):
        # Obtém os arquivos de imagem da pasta
        imagens = verificar_diretorio(path)
        
        # Se não houver imagens, pula para a próxima categoria
        if not imagens:
            continue
        
        # Seleciona aleatoriamente algumas imagens
        selected_images = random.sample(imagens, min(num_images_per_class, len(imagens)))
        
        # Exibe as imagens selecionadas
        for i, img_file in enumerate(selected_images):
            img = mpimg.imread(img_file)  # Lê a imagem
            plt.subplot(len(data_paths), num_images_per_class, idx * num_images_per_class + i + 1)
            plt.imshow(img, cmap='gray')  # Exibe a imagem em escala de cinza
            plt.title(label.capitalize())  # Define o título com o nome da categoria
            plt.axis('off')  # Remove os eixos
            
    # Ajusta a disposição das imagens e exibe
    plt.tight_layout()
    plt.show()
    
# Chama a função para exibir as imagens
exibir_imagens(data_paths, num_images_per_class)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def preprocess_image(img_path, target_size=224):
    """Pré-processa uma imagem: redimensiona, converte para tensor e normaliza."""
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    return transform(img)

def visualize_images(image_paths, preprocess_function, num_images=4):
    """Visualiza as imagens após o pré-processamento."""
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, img_path in enumerate(image_paths[:num_images]):
        # Pré-processa a imagem
        img_tensor = preprocess_function(img_path)
        
        # Converte o tensor para formato HWC (altura, largura, canais)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        # Remove a normalização (desfazendo a média e o desvio-padrão)
        img_np = img_np * imagenet_std + imagenet_mean
        img_np = np.clip(img_np, 0, 1)  # Garante que os valores estejam no intervalo [0, 1]

        # Exibe a imagem
        axes[i].imshow(img_np)
        axes[i].axis('off')

    plt.show()

image_paths = [
    './glioma/0000.jpg', 
    './healthy/0000.jpg', 
    './meningioma/0000.jpg', 
    './pituitary/0000.jpg'
]

visualize_images(image_paths, preprocess_image)

# Lista para armazenar os caminhos das imagens
image_paths = []

# Coleta todas as imagens de cada pasta
for folder in data_paths:
    for file in os.listdir(folder):
        if file.endswith('.jpg'):  # Verifica se o arquivo tem a extensão correta
            image_paths.append(os.path.join(folder, file))

print(f"Total de imagens encontradas: {len(image_paths)}")
print("Alguns exemplos:", image_paths[:5])  # Exibe os primeiros 5 exemplos


# Função para dividir os dados
def dividir_dados(image_paths, test_size=0.2, val_size=0.1, random_state=42):
    """
    Divide os dados em conjuntos de treinamento, validação e teste.
    
    Parâmetros:
        image_paths: lista de caminhos de imagens.
        test_size: proporção de dados reservada para teste.
        val_size: proporção de dados reservada para validação.
        random_state: semente para reprodução dos resultados.
    
    Retorna:
        train_data: conjunto de treinamento.
        val_data: conjunto de validação.
        test_data: conjunto de teste.
    """
    # Divide inicialmente em treinamento e temp_data (validação + teste)
    train_data, temp_data = train_test_split(image_paths, test_size=test_size, random_state=random_state)
    
    # Divide temp_data em validação e teste
    val_data, test_data = train_test_split(
        temp_data, test_size=val_size / (test_size + val_size), random_state=random_state
    )
    return train_data, val_data, test_data

# Dividindo os dados
train_data, val_data, test_data = dividir_dados(image_paths)
print(f"Tamanho do conjunto de treinamento: {len(train_data)}")
print(f"Tamanho do conjunto de validação: {len(val_data)}")
print(f"Tamanho do conjunto de teste: {len(test_data)}")


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Defina o dispositivo (GPU se disponível, caso contrário, use a CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Supondo que a classe BrainTumorCNN já tenha sido definida anteriormente
model = BrainTumorCNN(num_classes=4)

# Move o modelo para o dispositivo apropriado
model.to(device)

# Exibe um resumo do modelo com o tamanho da entrada especificado
summary(model, input_size=(3, 224, 224))


import torch.optim as optim
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Função de treinamento
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Cada epoch tem uma fase de treinamento e uma de validação
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Definir modelo para treinamento
            else:
                model.eval()   # Definir modelo para avaliação

            running_loss = 0.0
            running_corrects = 0

            # Iterar sobre os dados
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Passar forward
                # Acompanhar a história se estiver apenas em treinamento
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Passar backward e otimizar apenas se estiver em treinamento
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Estatísticas
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Guardar a perda e a precisão do treinamento/validação
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = 0 if "glioma" in img_path else 1 if "healthy" in img_path else 2 if "meningioma" in img_path else 3
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Definindo transformações
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# Criando datasets
train_dataset = ImageDataset(train_data, transform=train_transforms)
val_dataset = ImageDataset(val_data, transform=val_transforms)
test_dataset = ImageDataset(test_data, transform=val_transforms)

# Criando dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinando o modelo
model, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
    model, dataloaders, criterion, optimizer, num_epochs=10
)
