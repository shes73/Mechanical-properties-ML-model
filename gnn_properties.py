import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout, Sequential, Linear, ReLU
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

base_dir = "C:\\cif_editor\\mech_prop"
classes = ["plastic", "elastic", "brittle"] # шэринг и различные суперэластичности/пластичности не добавила, мало данных
class_to_label = {cls: idx for idx, cls in enumerate(classes)}

# словарь с типичными длинами связей (максимальные значения), просто своровала его из своего кода на си, можно для питона и покомпактнее сделать...
bond_lengths = {
    (0, 0): 0.85, (0, 5): 1.35, (0, 6): 1.50, (0, 7): 1.11,
    (0, 8): 1.10, (0, 16): 1.30, (0, 34): 1.50, (0, 52): 1.70,
    (5, 0): 1.35, (5, 5): 1.54, (5, 6): 2.10, (5, 7): 1.50,
    (5, 8): 1.34, (5, 16): 1.76, (5, 34): 1.93, (5, 15): 2.55,
    (5, 33): 2.71, (6, 0): 1.50, (6, 6): 1.75, (6, 5): 2.10,
    (6, 7): 1.90, (7, 0): 1.11, (7, 5): 1.50, (7, 6): 1.90,
    (7, 7): 1.55, (8, 0): 1.10, (8, 5): 1.34, (16, 0): 1.30,
    (16, 5): 1.76, (34, 0): 1.50, (34, 5): 1.93, (52, 0): 1.70
}

# по-глупому парсим циф дефолтного вида из меркюри
def parse_cif(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if "_atom_site_fract_z" in line:
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Cant find the string '_atom_site_fract_z' in the file {file_path}")

    data = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            break
        parts = line.split()
        if len(parts) >= 5:
            element = parts[1]
            x = float(parts[2].split('(')[0])
            y = float(parts[3].split('(')[0])
            z = float(parts[4].split('(')[0])
            data.append([element, x, y, z])

    df = pd.DataFrame(data, columns=['Element', 'X', 'Y', 'Z'])
    return df

# нужно для словаря и численного представления типов атомов, с другими атомами модель не работает, но их в датасете и нет
def atomic_number(atom):
    periodic_table = {"H": 0, "C": 5, "N": 6, "O": 7, "F": 8, "Cl": 16, "Br": 34, "I": 52}
    return periodic_table.get(atom, -1)

# функция создания ненаправленного графа из координат атомов, в качестве вершин - атомы, в качестве ребер - "связи"
def create_graph_with_interactions(df, other_molecules_coords=None):
    coords = df[["X", "Y", "Z"]].values
    atomic_numbers = [atomic_number(atom) for atom in df["Element"]]
    num_atoms = len(coords)

    edge_index = []
    edge_attr = []

    # внутримол связи
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                bond = bond_lengths.get((atomic_numbers[i], atomic_numbers[j]), 3.0)
                if dist < bond:
                    edge_index.append([i, j])
                    edge_attr.append(dist)

    # межмол связи
    if other_molecules_coords is not None:
        for i in range(num_atoms):
            for other_coords in other_molecules_coords:
                for j, other_atom_coord in enumerate(other_coords):
                    dist = np.linalg.norm(coords[i] - other_atom_coord)
                    if dist < 5.0:  # радиус учета межмол вз/д
                        edge_index.append([i, num_atoms + j])
                        edge_attr.append(dist)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() # определяет структуру, какие вершины соединены ребрами, а какие нет
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1) # атрибуты ребер = веса, длины, типы связей
    x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

class MoleculeDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)
        self.data_list = []
        for cls in classes:
            folder_path = os.path.join(root, cls)
            label = class_to_label[cls]
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                df = parse_cif(file_path)
                graph = create_graph_with_interactions(df)
                graph.y = torch.tensor([label], dtype=torch.long)
                self.data_list.append(graph)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        self.dropout1 = Dropout(p=0.3) # можно было бы конечно поварьировать дропауты, но что они есть, что их нет, одинаково...

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.dropout2 = Dropout(p=0.3)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        self.dropout3 = Dropout(p=0.3)

        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # первый слой 
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = ReLU()(x) 
        x = self.dropout1(x)

        # второй слой
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = ReLU()(x) 
        x = self.dropout2(x)

        # третий слой 
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = ReLU()(x) 
        x = self.dropout3(x)

        # глобальный пулинг и финальный слой
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train_model(dataset):
    train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16)

    device = torch.device("cpu")
    model = GCN(input_dim=1, hidden_dim=32, output_dim=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_accuracy = 0.0
    best_model_state = None

    for epoch in range(30): # на бОльшем числе эпох беда с переобучением для наиболее представленного класса 
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                pred = out.argmax(dim=1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        # вычисляем точность для каждой эпохи
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

        # сохраняем лучшую модель в качестве отправной точки для следующей эпохи
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()
            model.load_state_dict(best_model_state)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # тестирование лучшей модели
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    print("\nBest Model Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    return model 

dataset = MoleculeDataset(base_dir)
train_model(dataset)
