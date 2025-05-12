import torch
import torch.optim as optim
import torch.nn as nn
from torch import autograd
from tqdm import tqdm
import copy
from torch.utils.data import ConcatDataset, DataLoader
import os
import torchvision.utils as vutils
from utils import *
from pathlib import Path
from models import *
from methods import *
from dataloader import *
from torchvision import transforms
from sklearn.cluster import KMeans

def save_client_images(train_loaders, test_loaders, output_dir='client_images', save_all=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for client_id, (train_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
        print(f"Saving images for client {client_id}")
        
        client_dir = os.path.join(output_dir, f'client_{client_id}')
        train_dir = os.path.join(client_dir, 'train')
        test_dir = os.path.join(client_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for batch_idx, (images, labels) in enumerate(train_loader):
            if save_all:
                for img_idx, (image, label) in enumerate(zip(images, labels)):
                    train_filename = os.path.join(train_dir, 
                        f'batch_{batch_idx}_img_{img_idx}_label_{label}.png')
                    vutils.save_image(image, train_filename)
            else:
                train_filename = os.path.join(train_dir, 
                    f'batch_{batch_idx}_label_{labels[0]}.png')
                vutils.save_image(images[0], train_filename)

        for batch_idx, (images, labels) in enumerate(test_loader):
            if save_all:
                for img_idx, (image, label) in enumerate(zip(images, labels)):
                    test_filename = os.path.join(test_dir, 
                        f'batch_{batch_idx}_img_{img_idx}_label_{label}.png')
                    vutils.save_image(image, test_filename)
            else:
                test_filename = os.path.join(test_dir, 
                    f'batch_{batch_idx}_label_{labels[0]}.png')
                vutils.save_image(images[0], test_filename)

        if save_all:
            print(f"Saved {len(train_loader.dataset)} training images and {len(test_loader.dataset)} test images for client {client_id}")
        else:
            print(f"Saved {len(train_loader)} training batch samples and {len(test_loader)} test batch samples for client {client_id}")


def evaluate_model(global_model, test_loaders, args = None):
    global_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    client_accuracies = []
    total_correct, total_samples = 0, 0

    with torch.no_grad():
        for client_id, test_loader in enumerate(test_loaders):
            client_correct, client_total = 0, 0

            print(f"\nEvaluating Client {client_id}")

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = global_model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[-1]
                _, predicted = torch.max(outputs, 1)

                client_correct += (predicted == labels).sum().item()
                client_total += labels.size(0)       

            client_accuracy = (client_correct / client_total) * 100 if client_total > 0 else 0
            client_accuracies.append(client_accuracy)

            print(f"Client {client_id} Classification Accuracy: {client_accuracy:.2f}%")

            total_correct += client_correct
            total_samples += client_total

    total_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"Total Classification Accuracy: {total_accuracy:.2f}%")
    
    return client_accuracies, total_accuracy
        

        
def local_train(global_model, train_loader, epochs, lr):
    local_model = copy.deepcopy(global_model)
    optimizer, criterion = optim.SGD(local_model.parameters(), lr=lr), nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    initial_weights = {name: param.clone() for name, param in local_model.named_parameters()}

    for epoch in range(epochs):
        epoch_loss = 0
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()
            optimizer.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    return {name: initial_weights[name] - param for name, param in local_model.named_parameters()}

def federated_average(global_model, train_loaders, num_clients, rounds, epochs, lr, test_loaders):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    client_data_counts = [len(loader.dataset) for loader in train_loaders]
    client_weights = [count / sum(client_data_counts) for count in client_data_counts]
    round_accuracies = []

    for _ in range(rounds):
        print(f"\nRound {_}")
        client_updates = [
            (lambda i: (print(f"Training client {i}"), local_train(global_model, train_loaders[i], epochs, lr))[1])(i)
            for i in range(num_clients)
        ]
        avg_update = {
            name: sum(upd[name] * client_weights[i] for i, upd in enumerate(client_updates))
            for name, param in global_model.named_parameters() if param.requires_grad
        }


        with torch.no_grad():
            for name, param in global_model.named_parameters():
                if param.requires_grad:
                    param -= avg_update[name]
                    
        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)

    return round_accuracies

def local_train_fedprox(global_model, train_loader, epochs, lr, mu=0.01):
    local_model = copy.deepcopy(global_model)
    optimizer, criterion = optim.SGD(local_model.parameters(), lr=lr), nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    initial_weights = {name: param.clone() for name, param in local_model.named_parameters()}

    for epoch in range(epochs):
        epoch_loss = 0
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()
            optimizer.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            proximal_term = 0.0
            for name, param in local_model.named_parameters():
                proximal_term += ((param - global_model.state_dict()[name].to(device)) ** 2).sum()
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    return {name: initial_weights[name] - param for name, param in local_model.named_parameters()}

def federated_prox(global_model, train_loaders, num_clients, rounds, epochs, lr, test_loaders, mu = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    client_data_counts = [len(loader.dataset) for loader in train_loaders]
    client_weights = [count / sum(client_data_counts) for count in client_data_counts]
    round_accuracies = []

    for _ in range(rounds):
        print(f"\nRound {_}")
        client_updates = [
            (lambda i: (print(f"Training client {i} with FedProx"), local_train_fedprox(global_model, train_loaders[i], epochs, lr, mu))[1])(i)
            for i in range(num_clients)
        ]
        avg_update = {
            name: sum(upd[name] * client_weights[i] for i, upd in enumerate(client_updates))
            for name, param in global_model.named_parameters() if param.requires_grad
        }

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param -= avg_update[name]
        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)

    return round_accuracies

def local_train_scaffold(global_model, train_loader, c_global, c_local, epochs, lr):
    local_model = copy.deepcopy(global_model)
    optimizer, criterion = optim.SGD(local_model.parameters(), lr=lr), nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)


    initial_weights = {name: param.clone().to(device) for name, param in local_model.named_parameters()}

    for epoch in range(epochs):
        epoch_loss = 0
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()
            optimizer.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    param.grad -= (c_local[name].to(device) - c_global[name].to(device))

        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    overall_update = {name: initial_weights[name] - param for name, param in local_model.named_parameters()}
    new_c_local = {name: (c_local[name].to(device) - c_global[name].to(device) + overall_update[name] / (epochs * lr)).to(device) for name, param in local_model.named_parameters()}
    return overall_update, new_c_local

def federated_scaffold(global_model, train_loaders, test_loaders, num_clients, rounds, epochs, local_lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    global_c = {name: torch.zeros_like(param).to(device) for name, param in global_model.named_parameters()}
    client_cs = [{name: torch.zeros_like(param).to(device) for name, param in global_model.named_parameters()} for _ in range(num_clients)]
    round_accuracies = []

    for _ in range(rounds):
        print(f"\nRound {_}")
        client_updates, new_client_cs = [], []
        for i in range(num_clients):
            print(f"\nTraining Client {i}") 
            local_update, new_c_local = local_train_scaffold(global_model, train_loaders[i], global_c, client_cs[i], epochs, local_lr)
            client_updates.append(local_update)
            new_client_cs.append(new_c_local)

        avg_update = {
            name: torch.stack([upd[name] for upd in client_updates]).mean(dim=0)
            for name, param in global_model.named_parameters() if param.requires_grad
        }
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param -= avg_update[name]

        for name in global_c:
            global_c[name] += sum((new_client_cs[i][name] - client_cs[i][name]) / num_clients for i in range(num_clients))
        client_cs = new_client_cs

        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)

    return round_accuracies

def cosine_similarity(vec1, vec2):
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))

def gradient_harmonization(client_updates):
    num_clients = len(client_updates)

    for layer_name in client_updates[0].keys():
        layer_vectors = [client_update[layer_name].view(-1).cpu() for client_update in client_updates]

        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                if cosine_similarity(layer_vectors[i], layer_vectors[j]) < 0:
                    g_i_proj = layer_vectors[i] - torch.dot(layer_vectors[i], layer_vectors[j]) / torch.dot(layer_vectors[j], layer_vectors[j]) * layer_vectors[j]
                    g_j_proj = layer_vectors[j] - torch.dot(layer_vectors[j], layer_vectors[i]) / torch.dot(layer_vectors[i], layer_vectors[i]) * layer_vectors[i]

                    layer_vectors[i].copy_(g_i_proj)
                    layer_vectors[j].copy_(g_j_proj)

        for idx, client_update in enumerate(client_updates):
            client_update[layer_name] = layer_vectors[idx].view(client_update[layer_name].shape).to(client_update[layer_name].device)

    return client_updates

def gradient_conflict_counter(client_updates):
    total_conflicts = 0 
    num_clients = len(client_updates)

    for layer_name in client_updates[0].keys():
        layer_vectors = [client_update[layer_name].view(-1).cpu() for client_update in client_updates]
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                sim = cosine_similarity(layer_vectors[i], layer_vectors[j])
                if sim < 0:
                    total_conflicts += 1

    print(f"Total gradient conflicts detected: {total_conflicts}")
    return total_conflicts

def federated_gradient_harmonization(global_model, train_loaders, num_clients, rounds, epochs, lr, test_loaders):
    round_accuracies, pre_conflicts, post_conflicts = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    client_data_counts = [len(loader.dataset) for loader in train_loaders]
    client_weights = [count / sum(client_data_counts) for count in client_data_counts]

    for _ in range(rounds):
        print(f"\nRound {_}")
        client_updates = [
            (lambda i: (print(f"Training client {i}"), local_train(global_model, train_loaders[i], epochs, lr))[1])(i)
            for i in range(num_clients)
        ]
        pre_conflicts.append(gradient_conflict_counter(client_updates))
        client_updates = gradient_harmonization(client_updates)
        post_conflicts.append(gradient_conflict_counter(client_updates))
        print(f"Pre-conflict count: {pre_conflicts[-1]}, Post-conflict count: {post_conflicts[-1]}")
        avg_update = {
            name: sum(upd[name] * client_weights[i] for i, upd in enumerate(client_updates))
            for name, param in global_model.named_parameters() if param.requires_grad
        }
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param -= avg_update[name]

        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)

    return round_accuracies

def apply_sam_perturbation(local_model, criterion, x_train, y_train, rho):
    loss = criterion(local_model(x_train), y_train)
    loss.backward()
    with torch.no_grad():
        for param in local_model.parameters():
            if param.grad is not None:
                param.add_(rho * param.grad / (param.grad.norm() + 1e-12))
    return loss

def local_train_sam(global_model, train_loader, epochs, lr, rho=0.0001):
    local_model = copy.deepcopy(global_model)
    optimizer, criterion = optim.SGD(local_model.parameters(), lr=lr), nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model.to(device)
    initial_weights = {name: param.clone() for name, param in local_model.named_parameters()}

    for epoch in range(epochs):
        epoch_loss = 0
        for x_train, y_train in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x_train, y_train = x_train.to(device), y_train.to(device)
            y_train = y_train.long()
            apply_sam_perturbation(local_model, criterion, x_train, y_train, rho)
            optimizer.zero_grad()
            loss = criterion(local_model(x_train), y_train)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    return {name: initial_weights[name] - param for name, param in local_model.named_parameters()}

def federated_sam(global_model, train_loaders, test_loaders, num_clients, rounds, epochs, local_lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    client_data_counts = [len(loader.dataset) for loader in train_loaders]
    client_weights = [count / sum(client_data_counts) for count in client_data_counts]
    round_avg_updates, round_accuracies = [], []

    for _ in range(rounds):
        print(f"\nRound {_}")
        client_updates = [
            (lambda i: (print(f"Training client {i}"), local_train_sam(global_model, train_loaders[i], epochs, local_lr))[1])(i)
            for i in range(num_clients)
        ]
        avg_update = {
            name: sum(upd[name] * client_weights[i] for i, upd in enumerate(client_updates))
            for name, param in global_model.named_parameters() if param.requires_grad
        }
        round_avg_updates.append(avg_update)

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param -= avg_update[name]
        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)

    return round_accuracies

def local_train_feddyn(global_model, h_vector, alpha, train_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    global_params = torch.cat([param.view(-1) for param in global_model.parameters()]).detach()
    
    for epoch in range(epochs):
        for x, y in tqdm(train_loader, desc=f"FedDyn Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = local_model(x)
            loss = criterion(output, y.long())

            local_params = torch.cat([p.view(-1) for p in local_model.parameters()])
            reg_term = alpha * torch.sum(local_params * (-global_params + h_vector.to(device)))
            total_loss = loss + reg_term

            total_loss.backward()
            optimizer.step()

    return {
    name: (global_model.state_dict()[name].detach().clone() - param.detach().clone()).cpu()
    for name, param in local_model.named_parameters()
    }


def federated_feddyn(global_model, train_loaders, num_clients, rounds, epochs, lr, test_loaders, alpha=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    flat_params = torch.cat([param.view(-1) for param in global_model.parameters()])
    h_vectors = [torch.zeros_like(flat_params) for _ in range(num_clients)]
    
    round_accuracies = []
    round_h_norm = []

    for rnd in range(rounds):
        print(f"\nRound {rnd}")
        client_deltas = []

        for i in range(num_clients):
            print(f"Training client {i} with FedDyn")
            delta = local_train_feddyn(global_model, h_vectors[i], alpha, train_loaders[i], epochs, lr)
            client_deltas.append(delta)

        with torch.no_grad():
            for name, param in global_model.named_parameters():
                avg_delta = sum(client_deltas[i][name] for i in range(num_clients)) / num_clients
                param.data -= avg_delta.to(device)

        for i in range(num_clients):
            delta_i = torch.cat([
                client_deltas[i][name].view(-1) for name, _ in global_model.named_parameters()
            ])
            h_vectors[i] -= alpha * delta_i.to(device)

        round_h_norm.append(torch.norm(h_vectors[0]))

        _, total_accuracy = evaluate_model(global_model, test_loaders)
        round_accuracies.append(total_accuracy)
    
    print(round_h_norm)

    return round_accuracies


def local_train_moon(global_model, local_model, prev_models, train_loader, epochs, lr, mu, temperature):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model = copy.deepcopy(local_model).to(device)
    global_model = copy.deepcopy(global_model).to(device).eval()
    for p in global_model.parameters():
        p.requires_grad = False

    for prev in prev_models:
        prev.eval().to(device)
        for p in prev.parameters():
            p.requires_grad = False

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=0.9)
    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_contrast = torch.nn.CrossEntropyLoss()
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y = y.long()
            optimizer.zero_grad()

            _, proj_local, out = local_model(x)
            _, proj_global, _ = global_model(x)

            pos = cosine_sim(proj_local, proj_global).unsqueeze(1)  # [B, 1]
            logits = pos

            for prev_model in prev_models:
                _, proj_prev, _ = prev_model(x)
                neg = cosine_sim(proj_local, proj_prev).unsqueeze(1)
                logits = torch.cat((logits, neg), dim=1)  # [B, 1+N]

            logits /= temperature
            labels = torch.zeros(x.size(0)).long().to(device)

            loss_contrast = mu * criterion_contrast(logits, labels)
            loss_cls = criterion_cls(out, y)
            total_loss = loss_cls + loss_contrast

            total_loss.backward()
            optimizer.step()

    return local_model.state_dict()


def federated_moon(global_model, train_loaders, num_clients, rounds, epochs, lr, test_loaders, mu=1.0, temperature=0.5, model_buffer_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)

    client_models = [copy.deepcopy(global_model).to(device) for _ in range(num_clients)]
    model_buffer = [[] for _ in range(num_clients)]

    round_accuracies = []

    for rnd in range(rounds):
        print(f"\nRound {rnd}")
        updated_states = []

        for i in range(num_clients):
            print(f"Training client {i} with MOON")
            prev_models = model_buffer[i][-model_buffer_size:]
            updated_state = local_train_moon(global_model, client_models[i], prev_models, train_loaders[i], epochs, lr, mu, temperature)
            updated_states.append(updated_state)

        global_state = copy.deepcopy(global_model.state_dict())
        client_weights = [len(loader.dataset) for loader in train_loaders]
        total_weight = sum(client_weights)

        for key in global_state.keys():
            global_state[key] = sum(
                updated_states[i][key] * (client_weights[i] / total_weight)
                for i in range(num_clients)
            )

        global_model.load_state_dict(global_state)

        for i in range(num_clients):
            client_models[i].load_state_dict(global_state)
            buffer_model = copy.deepcopy(client_models[i])
            buffer_model.eval()
            for p in buffer_model.parameters():
                p.requires_grad = False
            model_buffer[i].append(buffer_model)
            if len(model_buffer[i]) > model_buffer_size:
                model_buffer[i].pop(0)

        _, test_acc = evaluate_model(global_model, test_loaders)
        round_accuracies.append(test_acc)

    return round_accuracies


def evaluate_concat_model(encoder_model, classifier_model, test_loaders, device="cuda"):
    encoder_model.eval()
    classifier_model.eval()
    encoder_model.to(device)
    classifier_model.to(device)
    round_accuracies = []
    with torch.no_grad():
        for test_loader in test_loaders:
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = classifier_model(encoder_model(x))
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
            acc = 100.0 * correct / total if total > 0 else 0.0
            round_accuracies.append(acc)
    return round_accuracies

class CombineModel(nn.Module):
    def __init__(self, net1, net2):
        super(CombineModel, self).__init__()
        self.net1 = net1
        self.net2 = net2
    def forward(self, x):
        x = self.net2(self.net1(x))
        return x

class CombineAllModel(nn.Module):
    def __init__(self, encoder_list):
        super(CombineAllModel, self).__init__()
        self.nets = nn.ModuleList(encoder_list)
    def forward(self, x):
        features = self.nets[0](x)
        for i in range(1, len(self.nets)):
            features = torch.cat((features, self.nets[i](x)), dim=1)
        return features

def train_net_encoder_classifier(net_id, net_encoder, net_classifier, train_dataloader, epochs, lr, args_optimizer, device="cpu"):
    if train_dataloader is None:
        raise ValueError("train_dataloader cannot be None")
        
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
        
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError("learning rate must be a positive number")
        
    if args_optimizer not in ['adam', 'sgd']:
        raise ValueError("optimizer must be either 'adam' or 'sgd'")

    try:
        if args_optimizer == 'adam':
            encoder_params = list(filter(lambda p: p.requires_grad, net_encoder.parameters()))
            classifier_params = list(net_classifier.parameters())
            optimizer = optim.Adam(encoder_params + classifier_params, lr=lr, weight_decay=1e-5)
        else:
            optimizer = optim.SGD([{'params': net_encoder.parameters()}, {'params': net_classifier.parameters()}], lr=lr, momentum=0.9, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss().to(device)
        train_dataloader = [train_dataloader] if not isinstance(train_dataloader, list) else train_dataloader
        
        for epoch in tqdm(range(epochs), desc=f"Training Client {net_id}"):
            for tmp in train_dataloader:
                for x, target in tmp:
                    x, target = x.to(device), target.to(device).long()
                    optimizer.zero_grad()
                    out = net_classifier(net_encoder(x))
                    loss = criterion(out, target)
                    loss.backward()
                    optimizer.step()
                    
    except Exception as e:
        print(f"Error during training for Client {net_id}: {str(e)}")
        raise
        
    print(f"Training complete for Client {net_id}")

def local_train_net_encoder_classifier(encoder_list, classifier_list, selected, args, train_loaders, device="cpu"):
    if not encoder_list or not classifier_list:
        raise ValueError("encoder_list and classifier_list cannot be empty")
        
    if not selected:
        raise ValueError("selected list cannot be empty")
        
    if not train_loaders:
        raise ValueError("train_loaders cannot be empty")
        
    try:
        for net_id in range(len(encoder_list)):
            if net_id not in selected:
                continue
            encoder_list[net_id].to(device)
            classifier_list[net_id].to(device)
            train_dl_local = train_loaders[net_id]
            train_net_encoder_classifier(net_id, encoder_list[net_id], classifier_list[net_id], train_dl_local, args.epochs_classifier, args.cd_lr, args_optimizer='adam', device=device)
            encoder_list[net_id].to("cpu")
            classifier_list[net_id].to("cpu")
    except Exception as e:
        print(f"Error in local training: {str(e)}")
        raise

def train_net_classifier(net_id, net_encoder, net_classifier, train_dataloader, epochs, lr, device="cpu"):
    optimizer = optim.SGD(net_classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    train_dataloader = [train_dataloader] if not isinstance(train_dataloader, list) else train_dataloader
    net_encoder.to(device)
    net_classifier.to(device)
    for epoch in range(epochs):
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                if batch_idx == 3:
                    break
                x, target = x.to(device), target.to(device).long()
                optimizer.zero_grad()
                out = net_classifier(net_encoder(x))
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
    print(f"Training complete for Client {net_id}")

def local_train_net_classifier(encoder, classifier_list, selected, args, train_loaders, device="cpu"):
    for net_id in range(len(classifier_list)):
        if net_id not in selected:
            continue
        encoder.to(device)
        classifier_list[net_id].to(device)
        train_dl_local = train_loaders[net_id]
        train_net_classifier(net_id, encoder, classifier_list[net_id], train_dl_local, epochs=1, lr=args.cd_lr, device=device)


def federated_concat(global_encoder_proto, global_classifier_proto, train_loaders, test_loaders, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clients = len(train_loaders)
    n_clusters = 5

    encoder_list = [copy.deepcopy(global_encoder_proto) for _ in range(n_clients)]
    classifier_list = [copy.deepcopy(global_classifier_proto) for _ in range(n_clients)]

    for client_id in range(n_clients):
        encoder_list[client_id].to(device)
        classifier_list[client_id].to(device)
        local_train_net_encoder_classifier(
            encoder_list, classifier_list, selected=[client_id],
            args=args,
            train_loaders=train_loaders,
            device=device
        )

    estimated_dis = []
    softmax = torch.nn.Softmax(dim=1)
    for client_id in range(n_clients):
        encoder_list[client_id].to(device)
        classifier_list[client_id].to(device)
        encoder_list[client_id].eval()
        classifier_list[client_id].eval()
        preds = []
        with torch.no_grad():
            for x, _ in train_loaders[client_id]:
                x = x.to(device)
                out = classifier_list[client_id](encoder_list[client_id](x))
                probs = softmax(out).mean(0).cpu().numpy()
                preds.append(probs)
                break
        estimated_dis.append(np.mean(preds, axis=0))
        encoder_list[client_id].to("cpu")
        classifier_list[client_id].to("cpu")

    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(estimated_dis)
    assign = estimator.labels_

    group = [[] for _ in range(n_clusters)]
    for client_id, cluster_id in enumerate(assign):
        group[cluster_id].append(client_id)

    encoder_global_para = [copy.deepcopy(global_encoder_proto.state_dict()) for _ in range(n_clusters)]
    classifier_global_para = [copy.deepcopy(global_classifier_proto.state_dict()) for _ in range(n_clusters)]

    for cluster_id in range(n_clusters):
        selected = group[cluster_id]
        if not selected:
            continue
            
        total_samples = sum(len(train_loaders[c].dataset) for c in selected)
        weights = [len(train_loaders[c].dataset) / total_samples for c in selected]

        for i, client_id in enumerate(selected):
            encoder_list[client_id].load_state_dict(encoder_global_para[cluster_id])
            classifier_list[client_id].load_state_dict(classifier_global_para[cluster_id])
            local_train_net_encoder_classifier(
                encoder_list, classifier_list, selected=[client_id],
                args=args,
                train_loaders=train_loaders,
                device=device
            )

        for i, client_id in enumerate(selected):
            local_encoder = encoder_list[client_id].state_dict()
            local_classifier = classifier_list[client_id].state_dict()
            if i == 0:
                for k in encoder_global_para[cluster_id]:
                    encoder_global_para[cluster_id][k] = local_encoder[k] * weights[i]
                for k in classifier_global_para[cluster_id]:
                    classifier_global_para[cluster_id][k] = local_classifier[k] * weights[i]
            else:
                for k in encoder_global_para[cluster_id]:
                    encoder_global_para[cluster_id][k] += local_encoder[k] * weights[i]
                for k in classifier_global_para[cluster_id]:
                    classifier_global_para[cluster_id][k] += local_classifier[k] * weights[i]

    encoder_selected = []
    for c in range(n_clusters):
        enc = copy.deepcopy(global_encoder_proto)
        enc.load_state_dict(encoder_global_para[c])
        encoder_selected.append(enc.to(device))

    encoder_all = CombineAllModel(encoder_selected)
    if args.dataset == "cifar100":
        out_dim = 100
    else:
        out_dim = 10
    classifier_global = SimpleCNN2Classifier(in_dim=600, out_dim=out_dim).to(device)
    classifier_list = [
        SimpleCNN2Classifier(in_dim=600, out_dim=out_dim).to(device)
        for _ in range(n_clients)
    ]
    classifier_global_para = classifier_global.state_dict()

    round_accuracies = []
    for r in range(args.rounds):
        selected = np.random.choice(n_clients, int(1 * n_clients), replace=False)
        for i in selected:
            classifier_list[i].load_state_dict(classifier_global_para)

        local_train_net_classifier(
            encoder_all, classifier_list, selected, args,
            train_loaders=train_loaders,
            device=device
        )

        weights = [1 / len(selected) for _ in selected]
        for j, client_id in enumerate(selected):
            local_classifier = classifier_list[client_id].state_dict()
            if j == 0:
                for k in classifier_global_para:
                    classifier_global_para[k] = local_classifier[k] * weights[j]
            else:
                for k in classifier_global_para:
                    classifier_global_para[k] += local_classifier[k] * weights[j]

        classifier_global.load_state_dict(classifier_global_para)

        accs = evaluate_concat_model(encoder_all, classifier_global, test_loaders, device)
        round_accuracies.append(accs)

    return round_accuracies



#======================== Ensemble Methods =========================

def local_train_fedov(net_id, net, train_dataloader, test_dataloader, epochs, lr, optimizer_type, sz, num_class=10, device="cpu"):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=1e-5) \
        if optimizer_type == 'adam' else optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=11).to(device)

    net.to(device)

    aug_crop = transforms.RandomChoice([
        transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)),
        transforms.Lambda(lambda img: blur(img)),
        transforms.RandomErasing(p=1, scale=(0.33, 0.5)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
    ])
    aug_final = transforms.RandomChoice([
        transforms.Lambda(lambda img: aug_crop(img)),
    ])

    attack = FastGradientSignUntargeted(net, epsilon=0.5, alpha=0.002, min_val=0, max_val=1, max_iters=5, device=device)

    for epoch in range(epochs):
        net.train()
        epoch_loss_collector = []
        correct = 0
        total = 0

        for batch_idx, (X, y) in tqdm(enumerate(train_dataloader), 
                                           total=len(train_dataloader), 
                                           desc=f"Epoch {epoch+1}/{epochs}"):

            X, y = X.to(device), y.to(device)
            # Update target labels: shift unknown class to 0 (In the original fedov the unknown class was the last class)
            y = y + 1

            y_gen = torch.zeros(X.shape[0], dtype=torch.long).to(device)

            x_gen = copy.deepcopy(X.cpu().numpy())
            for i in range(x_gen.shape[0]):
                x_gen[i] = aug_final(torch.Tensor(x_gen[i]))
            x_gen = torch.Tensor(x_gen).to(device)

            optimizer.zero_grad()
            X.requires_grad = True
            y.requires_grad = False
            y = y.long()
            y_gen.requires_grad = False
            x_gen.requires_grad = True

            X_final = torch.cat([X, x_gen], dim=0)
            y_gen = torch.zeros(X.shape[0], dtype=torch.long).to(device)
            y_final = torch.cat([y, y_gen], dim=0)
            adv_data = attack.perturb(x_gen, y_gen)
            X_final = torch.cat([X_final, adv_data], dim=0)
            y_final = torch.cat([y_final, y_gen], dim=0)

            # save_images(epoch, net_id, batch_idx, x_gen, adv_data)

            out = net(X_final)
                    
            loss = criterion(out, y_final)
            loss.backward()
            optimizer.step()

            out_check = net(X)
            _, predictions = out_check.max(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

            
            epoch_loss_collector.append(loss.item())

        # Calculate epoch metrics
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    print(f"Training complete for Client {net_id}")
    

def federated_ov(global_model, train_loaders, test_loaders, num_clients, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier_dir = Path("model_weights") / args.dataset / args.algorithm / f"partition_{args.partition}" / f"num_clients_{args.n_parties}"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    
    model_list = []

    for client_id in range(num_clients):
        classifier_path = classifier_dir / f"classifier_client_{client_id}.pt"
        
        if args.model == "SimpleCNN2":
            local_model = SimpleCNN2(
                in_channels=global_model.channels,
                input_size=global_model.size,
                n_kernels=global_model.n_kernels,
                out_dim=global_model.out_dim
            ).to(device)

        if args.save_classifier:  # Train and save classifier if specified
            print(f"Training classifier for Client {client_id}...")
            local_train_fedov(
                net_id=client_id,
                net=local_model,
                train_dataloader=train_loaders[client_id],
                test_dataloader=test_loaders[client_id],
                epochs=args.epochs_classifier, 
                lr=args.cd_lr,
                optimizer_type="adam",
                sz=global_model.size,
                num_class=global_model.out_dim,
                device=device
            )
            torch.save(local_model.state_dict(), classifier_path)
            print(f"Saved classifier model for Client {client_id} at {classifier_path}")
        else:  # Load classifier if save option is not enabled
            if classifier_path.exists():
                local_model.load_state_dict(torch.load(classifier_path, map_location=device))
                print(f"Loaded classifier model for Client {client_id} from {classifier_path}")
            else:
                raise FileNotFoundError(f"Classifier model file not found for Client {client_id}. Expected at {classifier_path}")

        model_list.append(local_model)

        # Test each client model on all clients' data
        for test_id in range(num_clients):
            classification_accuracy, total_samples, total_not_class,_ = test_client_model(
                model_list[client_id],
                train_loaders[test_id],
                device
            )
            print(f"Client {test_id}: Accuracy = {classification_accuracy:.2f}%, NotClass = {total_not_class}/{total_samples}")

    # Ensemble Testing
    ensemble_accuracy = compute_ensemble_accuracy(
        contrastive_classifiers=model_list,
        test_loaders=test_loaders,
        aggregation_method="voting",
        device=device
    )

    return ensemble_accuracy

def local_train_eov(classifier, train_loader, grid_size, sz, net_id, args, device, pruneratio=0.4):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.cd_lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=11).to(device)

    aug_crop = transforms.RandomChoice([
        transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)),
        transforms.Lambda(lambda img: blur(img)),
        transforms.RandomErasing(p=1, scale=(0.33, 0.5)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
    ])

    aug_final = transforms.RandomChoice([
        transforms.Lambda(lambda img: aug_crop(img)),
    ])

    attack = FastGradientSignTargeted(classifier, epsilon=0.5, alpha=0.002, min_val=0, max_val=1, max_iters=5, device=device)

    total_stages = 8
    pruned_masks = {}

    # ---- Separate Pruning Target Setup ----
    total_conv_weights = 0
    total_fc_weights = 0
    for name, module in classifier.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            total_fc_weights += module.weight.numel()
        elif isinstance(module, nn.Conv2d):
            total_conv_weights += module.weight.numel()

    prune_k_total_fc = int(total_fc_weights * pruneratio)
    prune_k_total_conv = int(total_conv_weights * pruneratio)

    print(f"Total FC weights to prune: {prune_k_total_fc}/{total_fc_weights}")
    print(f"Total conv weights to prune: {prune_k_total_conv}/{total_conv_weights}")

    for epoch in range(args.epochs_classifier):
        classifier.train()
        epoch_loss_collector = []
        correct = 0
        total = 0

        stage = int(total_stages * epoch / args.epochs_classifier)
        print(f"Epoch {epoch+1}/{args.epochs_classifier}, Stage: {stage}")

        # Perform pruning once at start
        if epoch == 10 and pruneratio > 0:
            print(f"Pruning FC layers: {prune_k_total_fc} weights ({pruneratio*100:.0f}%)")
            pruned_masks = prune_fc_layers(classifier, prune_k_total_fc, pruned_masks)

            print(f"Pruning Conv layers: {prune_k_total_conv} filters ({pruneratio*100:.0f}%)")
            pruned_masks = prune_conv_layers2(classifier, pruneratio, pruned_masks)

        for batch_idx, (X, y) in tqdm(enumerate(train_loader),
                                      total=len(train_loader),
                                      desc=f"Epoch {epoch+1}/{args.epochs_classifier}"):
            X, y = X.to(device), y.to(device)
            y = y + 1
            y_gen = torch.zeros(X.shape[0], dtype=torch.long).to(device)

            y.requires_grad = False
            y = y.long()
            y_gen.requires_grad = False

            optimizer.zero_grad()

            x_gen = copy.deepcopy(X.cpu().numpy())
            for i in range(x_gen.shape[0]):
                t = torch.Tensor(x_gen[i])
                t = aug_final(t)
                x_gen[i] = t.numpy()
            x_gen = torch.Tensor(x_gen).to(device)

            if stage >= 3:
                shuffled_images, _ = generate_shuffled_tensors(X, grid_size=grid_size, path='permutations.pt')
                shuffled_images = smooth_patch_edges(shuffled_images, grid_size=grid_size, blur_kernel_size=5)
                x_gen = torch.cat([shuffled_images, x_gen], dim=0)
                y_gen = torch.cat([y_gen, y_gen], dim=0)

            x_gen.requires_grad = True

            if stage >= 1:
                if stage >= 2:
                    attack.max_iters = 15
                adv_data = attack.perturb(x_gen, y_gen, y)
                X_final = torch.cat([X, adv_data], dim=0)
                y_final = torch.cat([y, y_gen], dim=0)
            else:
                X_final = torch.cat([X, x_gen], dim=0)
                y_final = torch.cat([y, y_gen], dim=0)

            out = classifier(X_final)
            loss = criterion(out, y_final)
            loss.backward()
            optimizer.step()

            if pruned_masks is not None:
                with torch.no_grad():
                    zero_out_pruned_weights(classifier, pruned_masks)

            out_check = classifier(X)
            _, predictions = out_check.max(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs_classifier}], Loss: {epoch_loss:.4f}")

    print(f"Training complete for Client {net_id}")

def federated_eov(global_model, train_loaders, test_loaders, num_clients, args, pruneratio):
    device = torch.device(args.device)

    # Directory to save/load classifier models (includes algorithm name)
    classifier_dir = Path("model_weights") / args.dataset / args.algorithm / f"partition_{args.partition}" / f"num_clients_{args.n_parties}"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    
    model_list = []

    for client_id in range(num_clients):
        classifier_path = classifier_dir / f"classifier_client_{client_id}.pt"
        if args.model == "SimpleCNN2":
            local_model = SimpleCNN2(
                in_channels=global_model.channels,
                input_size=global_model.size,
                n_kernels=global_model.n_kernels,
                out_dim=global_model.out_dim
            ).to(device)


        if args.save_classifier:
            print(f"Training classifier for Client {client_id}...")
            local_train_eov(local_model, train_loaders[client_id], args.grid_size, sz = global_model.size, net_id = client_id, args=args, device=device, pruneratio=pruneratio)
            torch.save(local_model.state_dict(), classifier_path)
            print(f"Saved classifier model for Client {client_id} at {classifier_path}")
        else:
            if classifier_path.exists():
                local_model.load_state_dict(torch.load(classifier_path, map_location=device))
                print(f"Loaded classifier model for Client {client_id} from {classifier_path}")
            else:
                raise FileNotFoundError(f"Classifier model file not found for Client {client_id}. Expected at {classifier_path}")

        model_list.append(local_model)

        # Test each client model on all clients' data
        for test_id in range(num_clients):
            classification_accuracy, total_samples, total_not_class, _ = test_client_model(
                model_list[client_id],
                train_loaders[test_id],
                device
            )
            print(f"Client {test_id}: Accuracy = {classification_accuracy:.2f}%, NotClass = {total_not_class}/{total_samples}")

    # Ensemble Testing
    ensemble_accuracy = compute_ensemble_accuracy(
        model_list,
        test_loaders,
        aggregation_method="voting",
        device = device
    )

    return ensemble_accuracy

def local_train_prunedeov(classifier, train_loader, grid_size, sz, net_id, args, device, pruneratio=0.5):
    torch.autograd.set_detect_anomaly(True)

    criterion = nn.CrossEntropyLoss(ignore_index=11).to(device)
    total_stages = 8

    total_epochs = args.epochs_classifier
    prune_steps = 5
    step_epochs = total_epochs // (2 * prune_steps)

    aug_crop = transforms.RandomChoice([
        transforms.RandomResizedCrop(sz, scale=(0.1, 0.33)),
        transforms.Lambda(lambda img: blur(img)),
        transforms.RandomErasing(p=1, scale=(0.33, 0.5)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
        transforms.Lambda(lambda img: cut(img)),
        transforms.Lambda(lambda img: rot(img)),
    ])

    aug_final = transforms.RandomChoice([
        transforms.Lambda(lambda img: aug_crop(img)),
    ])

    initial_weights = save_initial_weights(classifier)
    accumulated_mask = None

    def run_training_phase(num_epochs, pruned_masks=None):
        nonlocal classifier
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=args.cd_lr, weight_decay=1e-5)
        attack = FastGradientSignTargeted(classifier, epsilon=0.5, alpha=0.002, min_val=0, max_val=1, max_iters=5, device=device)

        for epoch in range(num_epochs):
            classifier.train()
            epoch_loss_collector = []
            correct, total = 0, 0
            stage = int(total_stages * epoch / num_epochs)
            print(f"Epoch {epoch + 1}, Stage: {stage}")

            for batch_idx, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
                X, y = X.to(device), y.to(device)
                y = y + 1
                y = y.long()
                y_gen = torch.zeros(X.shape[0], dtype=torch.long).to(device)

                optimizer.zero_grad()

                x_gen = X.detach().cpu().numpy()
                for i in range(x_gen.shape[0]):
                    t = torch.Tensor(x_gen[i])
                    t = aug_final(t)
                    x_gen[i] = t.numpy()
                x_gen = torch.Tensor(x_gen).to(device)

                if stage >= 3:
                    shuffled_images, _ = generate_shuffled_tensors(X.detach(), grid_size, path='permutations.pt')
                    with torch.no_grad():
                        shuffled_images = smooth_patch_edges(shuffled_images, grid_size, blur_kernel_size=5)
                    x_gen = torch.cat([shuffled_images, x_gen], dim=0)
                    y_gen = torch.cat([y_gen, y_gen], dim=0)

                x_gen.requires_grad_()

                if stage >= 1:
                    attack.max_iters = 15 if stage >= 2 else 5
                    adv_data = attack.perturb(x_gen, y_gen, y)
                    X_final = torch.cat([X, adv_data], dim=0)
                    y_final = torch.cat([y, y_gen], dim=0)
                else:
                    X_final = torch.cat([X, x_gen], dim=0)
                    y_final = torch.cat([y, y_gen], dim=0)

                out = classifier(X_final)
                loss = criterion(out, y_final)
                loss.backward()
                optimizer.step()

                if pruned_masks is not None:
                    with torch.no_grad():
                        zero_out_pruned_weights(classifier, pruned_masks)

                _, predictions = classifier(X).max(1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                epoch_loss_collector.append(loss.item())

            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            epoch_accuracy = 100 * correct / total
            print(f"Loss: {epoch_loss:.4f}")

    # ---- FC pruning setup ----
    total_fc_weights = 0
    total_conv_weights = 0
    for name, module in classifier.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            total_fc_weights += module.weight.numel()
        elif isinstance(module, nn.Conv2d):
            total_conv_weights += module.weight.numel()

    prune_k_total_fc = int(total_fc_weights * pruneratio)
    prune_k_total_conv = int(total_conv_weights * pruneratio)

    print(f"Total FC weights to prune: {prune_k_total_fc}/{total_fc_weights}")
    print(f"Total conv weights to prune: {prune_k_total_conv}/{total_conv_weights}")

    # ---- Iterative Pruning Phase ----
    if pruneratio > 0:
        for step in range(prune_steps):
            print(f"\n--- Iterative Prune Step {step+1}/{prune_steps} ---")
            run_training_phase(step_epochs, pruned_masks=accumulated_mask)

            step_ratio = pruneratio / prune_steps
            print(f"Pruning ≈{step_ratio*100:.2f}% of remaining conv weights PER layer (unstructured)")
            accumulated_mask = prune_conv_layers(
                classifier,
                prune_ratio_step=step_ratio,
                pruned_masks=accumulated_mask
            )

            target_k_fc = int(prune_k_total_fc * (step + 1) / prune_steps)
            current_k_fc = sum(mask.sum().item() for name, mask in (accumulated_mask or {}).items() if "fc" in name or "linear" in name)
            this_step_k_fc = max(target_k_fc - current_k_fc, 0)
            print(f"Pruning {this_step_k_fc} more fc weights to reach {target_k_fc}/{prune_k_total_fc}")
            accumulated_mask = prune_fc_layers(classifier, this_step_k_fc, accumulated_mask)

            with torch.no_grad():
                zero_out_pruned_weights(classifier, accumulated_mask)
                nonzero = sum(torch.count_nonzero(p).item() for p in classifier.parameters() if p.requires_grad)
                total = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
                print(f"Remaining non-zero weights: {nonzero}/{total} ({100 * nonzero / total:.2f}%)")

            print("Resetting model to winning ticket.")
            reinitialize_model(classifier, initial_weights)
            reset_batchnorm_layers(classifier)


    # ---- Final Post-Pruning Training ----
    print("\n--- Final Training Phase on Fully Pruned Model ---")
    run_training_phase(2 * prune_steps * step_epochs, pruned_masks=accumulated_mask)

def federated_prunedeov(global_model, train_loaders, test_loaders, num_clients, args, pruneratio):
    device = torch.device(args.device)

    classifier_dir = Path("model_weights") / args.dataset / args.algorithm / f"partition_{args.partition}" / f"num_clients_{args.n_parties}"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    
    model_list = []

    for client_id in range(num_clients):
        classifier_path = classifier_dir / f"classifier_client_{client_id}.pt"
        if args.model == "SimpleCNN2":
            local_model = SimpleCNN2(
                in_channels=global_model.channels,
                input_size=global_model.size,
                n_kernels=global_model.n_kernels,
                out_dim=global_model.out_dim
            ).to(device)

        if args.save_classifier:
            print(f"Training classifier for Client {client_id}...")
            local_train_prunedeov(local_model, train_loaders[client_id], args.grid_size, sz = global_model.size, net_id = client_id, args=args, device=device, pruneratio=pruneratio)
            torch.save(local_model.state_dict(), classifier_path)
            print(f"Saved classifier model for Client {client_id} at {classifier_path}")
        else:
            if classifier_path.exists():
                local_model.load_state_dict(torch.load(classifier_path, map_location=device))
                print(f"Loaded classifier model for Client {client_id} from {classifier_path}")
            else:
                raise FileNotFoundError(f"Classifier model file not found for Client {client_id}. Expected at {classifier_path}")

        model_list.append(local_model)

        # Test each client model on all clients' data
        for test_id in range(num_clients):
            classification_accuracy, total_samples, total_not_class, _ = test_client_model(
                model_list[client_id],
                train_loaders[test_id],
                device
            )
            print(f"Client {test_id}: Accuracy = {classification_accuracy:.2f}%, NotClass = {total_not_class}/{total_samples}")

    # Ensemble Testing
    ensemble_accuracy = compute_ensemble_accuracy(
        model_list,
        test_loaders,
        aggregation_method="voting",
        device = device
    )

    return ensemble_accuracy


def distill_soft_feddf_style(student_model, model_list, dataloader, distill_epochs, lr, reg, half, device="cpu", logger=None):
    student_model.to(device)
    student_model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr, weight_decay=reg)
    criterion = nn.KLDivLoss(reduction='batchmean').to(device)

    for epoch in range(distill_epochs):
        epoch_loss_collector = []

        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= half:
                break

            x = x.to(device)
            x.requires_grad = True

            # === Compute soft targets on-the-fly ===
            with torch.no_grad():
                soft_label = ensemble_inference(x, model_list, aggregation_method='voting').to(device)

            # === Get student output ===
            out = student_model(x)
            out = torch.nn.LogSoftmax(dim=1)(out)  # Exclude last class (unknown)

            # === KL divergence loss ===
            loss = criterion(out, soft_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_collector.append(loss.item())

        print(f"[Distill] Epoch {epoch+1}/{distill_epochs}, Loss: {sum(epoch_loss_collector)/len(epoch_loss_collector):.4f}")

def federated_distilled(global_model, train_loaders, test_loaders, num_clients, algorithm, args, logger=None):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def combine_loaders(train_loaders, batch_size):
        all_datasets = [loader.dataset for loader in train_loaders]
        combined_dataset = ConcatDataset(all_datasets)
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    model_list = []
    for client_id in range(num_clients):
        if args.model == "SimpleCNN2":
            local_model = SimpleCNN2(
                in_channels=global_model.channels,
                input_size=global_model.size,
                n_kernels=global_model.n_kernels,
                out_dim=global_model.out_dim
            ).to(device)
        else:
            raise NotImplementedError(f"Model {args.model} not supported.")

        logger and logger.info(f"Training client {client_id} using {algorithm}")

        if algorithm == "ov":
            local_train_fedov(
                net_id=client_id,
                net=local_model,
                train_dataloader=train_loaders[client_id],
                test_dataloader=test_loaders[client_id],
                epochs=args.epochs_classifier,
                lr=args.cd_lr,
                optimizer_type="adam",
                sz=global_model.size,
                device=device
            )
        elif algorithm == "eov":
            local_train_eov(
                classifier=local_model,
                train_loader=train_loaders[client_id],
                grid_size=args.grid_size,
                sz=global_model.size,
                net_id=client_id,
                args=args,
                device=device,
                pruneratio=args.pruneratio
            )
        else:
            raise NotImplementedError(f"Local training algorithm {algorithm} not implemented.")

        model_list.append(local_model)

        # Test each client model on all clients' data
        for test_id in range(num_clients):
            classification_accuracy, total_samples, total_not_class, _ = test_client_model(
                model_list[client_id],
                train_loaders[test_id],
                device
            )
            print(f"Client {test_id}: Accuracy = {classification_accuracy:.2f}%, NotClass = {total_not_class}/{total_samples}")
    
    # Ensemble Testing
    ensemble_accuracy = compute_ensemble_accuracy(
        model_list,
        test_loaders,
        aggregation_method="voting",
        device = device
    )

    distill_loader = combine_loaders(train_loaders, batch_size=args.batch_size)
    print("Created combined distillation loader from training data.")

    student_model = SimpleCNN2(
        in_channels=global_model.channels,
        input_size=global_model.size,
        n_kernels=global_model.n_kernels,
        out_dim=global_model.out_dim - 1
    )


    distill_soft_feddf_style(
        student_model=student_model,
        model_list=model_list, 
        dataloader=distill_loader,
        distill_epochs=args.epochs_classifier,
        lr=args.cd_lr,
        reg=1e-5,
        half=len(distill_loader) // 2,
        device=device,
        logger=logger
    )

    _, total_accuracy = evaluate_model(student_model, test_loaders)

    return total_accuracy


#======================== Centralized Training =========================
def centralized_training(global_model, train_loaders, test_loaders, epochs, lr, args):

    device = torch.device(args.device)
    global_model.to(device)
    
    train_loader = train_loaders[0]
    test_loader = test_loaders[0]
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, global_model.parameters()), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting centralized training for {epochs} epochs")
    
    for epoch in range(epochs):
        global_model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = 100.*correct/total
        print(f'Epoch {epoch+1}/{epochs} completed - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        global_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = global_model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100.*test_correct/test_total
        print(f'Test Accuracy: {test_acc:.2f}%')
    
    return test_acc