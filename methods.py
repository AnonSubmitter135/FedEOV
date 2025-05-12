import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import os
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.utils.prune as prune

def ensemble_inference(input_data, class_discriminators, aggregation_method="voting", accepted_vote=5, verbose=False):
    output_vector = [] 
    not_class_probs = [] 

    for i, discriminator in enumerate(class_discriminators):
        output = discriminator(input_data)

        if isinstance(output, tuple):  
            output = output[0]

        softmax_output = torch.softmax(output, dim=1)
        output_vector.append(softmax_output[:, 1:])  
        not_class_probs.append(softmax_output[:, 0])  

        if verbose:
            avg_prob = softmax_output.mean(dim=0)
            print(f"Discriminator {i}: Avg Class Probabilities = {[round(prob.item(), 2) for prob in avg_prob]}")

    output_vector = torch.stack(output_vector, dim=2)  
    not_class_probs = torch.stack(not_class_probs, dim=1)  

    if aggregation_method == "voting":
        final_predictions = []
        for batch_idx in range(input_data.size(0)):  
            per_sample_outputs = output_vector[batch_idx]  
            per_sample_not_class = not_class_probs[batch_idx]  

            sorted_indices = torch.argsort(per_sample_not_class)
            top_indices = sorted_indices[:accepted_vote]  

            aggregated_probs = torch.sum(per_sample_outputs[:, top_indices], dim=1)
            aggregated_probs /= torch.sum(aggregated_probs)  

            final_predictions.append(aggregated_probs)

        final_output = torch.stack(final_predictions, dim=0)  

    elif aggregation_method == "weighted":
        weights = 1 - not_class_probs  
        weights = torch.softmax(weights, dim=1) 
        final_output = (output_vector * weights.unsqueeze(1)).sum(dim=2)


    else:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}")

    if verbose:
        avg_prob_per_discriminator = not_class_probs.mean(dim=0)
        highest_avg_prob_class = avg_prob_per_discriminator.argmax().item()
        highest_avg_prob_value = avg_prob_per_discriminator[highest_avg_prob_class].item()
        print(f"Top Discriminator: Class {highest_avg_prob_class} with Avg Probability = {highest_avg_prob_value:.4f}")

        highest_per_sample = not_class_probs.argmin(dim=1)
        batch_size = not_class_probs.size(0)
        unique, counts = torch.unique(highest_per_sample, return_counts=True)
        count_dict = dict(zip(unique.tolist(), counts.tolist()))

        sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        top_three = sorted_counts[:3]

        print("Top 3 most frequent discriminators as most confident:")
        print("----------------------------------------------------")
        for rank, (discriminator_class, count) in enumerate(top_three, start=1):
            print(f"{rank}. Discriminator {discriminator_class} selected as most confident in {count}/{batch_size} samples")
        print("----------------------------------------------------\n")

    return final_output

def compute_ensemble_accuracy(contrastive_classifiers, test_loaders, aggregation_method,  device, verbose=True):
    total_correct = 0
    total_samples = 0
    client_accuracies = []

    with torch.no_grad():
        for loader_id, dataloader in enumerate(test_loaders):
            correct = 0
            total = 0
            print(f"\nTesting on Client {loader_id}...\n")
            
            for X, target in dataloader:
                X, target = X.to(device), target.to(device)
                
                output_vector = ensemble_inference(X, contrastive_classifiers,aggregation_method=aggregation_method, verbose=verbose)
                
                predicted_class = output_vector.argmax(dim=1)
                correct += (predicted_class == target).sum().item()
                total += target.size(0)

            loader_accuracy = correct / total * 100
            client_accuracies.append(loader_accuracy)  
            total_correct += correct
            total_samples += total

    for loader_id, accuracy in enumerate(client_accuracies):
        print(f"Test Accuracy on Client {loader_id}: {accuracy:.2f}%")

    overall_accuracy = total_correct / total_samples * 100
    print(f"\nOverall Ensemble Model Accuracy: {overall_accuracy:.2f}%")
    return overall_accuracy

def test_client_model(classifier, dataloader, device):
    classifier.eval()
    total_correct, total_samples, total_not_class = 0, 0, 0
    feature_dict = {}

    for X, target in dataloader:
        X, target = X.to(device), target.to(device)
        target = target + 1
        output = classifier(X)
        predicted_labels = output.argmax(dim=1).float()
        total_not_class += (predicted_labels == 0).sum().item()
        total_correct += (predicted_labels == target).sum().item()
        total_samples += target.size(0)
        
        features = classifier.produce_feature(X).detach().cpu()
        for i, t in enumerate(target.cpu().numpy()):
            t = int(t)  
            if t not in feature_dict:
                feature_dict[t] = []
            feature_dict[t].append(features[i].tolist())  

    accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    return accuracy, total_samples, total_not_class, feature_dict


def save_image(image_tensor, save_path, file_name):
    os.makedirs(save_path, exist_ok=True)
    image = TF.to_pil_image(image_tensor.cpu())
 

def prune_fc_layers(model, prune_k, pruned_masks=None):

    if pruned_masks is None:
        pruned_masks = {}

    all_unpruned = []
    weight_refs = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            weight = module.weight.data
            mask = pruned_masks.get(name, torch.zeros_like(weight, dtype=torch.bool))
            unmasked = (~mask)

            if unmasked.sum() == 0:
                continue

            values = weight[unmasked].abs()
            all_unpruned.append(values)
            weight_refs.append((name, module, unmasked))

    if not all_unpruned:
        return pruned_masks

    all_unpruned = torch.cat(all_unpruned)
    if prune_k == 0 or prune_k > all_unpruned.numel():
        return pruned_masks

    threshold = all_unpruned.kthvalue(prune_k).values.item()

    # Now apply pruning using global threshold
    for name, module, unmasked in weight_refs:
        weight = module.weight.data
        mask = pruned_masks.get(name, torch.zeros_like(weight, dtype=torch.bool))

        new_mask = (weight.abs() <= threshold) & unmasked
        weight[new_mask] = 0
        pruned_masks[name] = mask | new_mask

        print(f"FC Layer {name}: Pruned {new_mask.sum().item()} weights (global kthresh)")

    return pruned_masks

def prune_conv_layers(model, prune_ratio_step, pruned_masks=None): 

    if pruned_masks is None:
        pruned_masks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            # get existing mask or all‑false
            mask = pruned_masks.get(name, torch.zeros_like(weight, dtype=torch.bool, device=weight.device))
            unmasked = ~mask
            num_unpruned = int(unmasked.sum().item())
            # how many to prune this step in this layer
            k = int(num_unpruned * prune_ratio_step)
            if k <= 0:
                continue

            # find k-th smallest magnitude among unmasked weights
            vals = weight[unmasked].abs().view(-1)
            threshold = vals.kthvalue(k).values.item()

            # prune any weight <= threshold (that hasn't been pruned yet)
            new_prune = (weight.abs() <= threshold) & unmasked
            weight[new_prune] = 0.0
            pruned_masks[name] = mask | new_prune

            print(f"Conv Layer {name}: pruned {new_prune.sum().item()} weights (unstructured)")

    return pruned_masks



def prune_conv_layers2(model, prune_ratio_step, pruned_masks=None):

    if pruned_masks is None:
        pruned_masks = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            out_channels = weight.size(0)

            # get existing 1D mask or start with all False
            mask = pruned_masks.get(name,
                   torch.zeros(out_channels, dtype=torch.bool, device=weight.device))
            unmasked = ~mask
            num_unpruned = int(unmasked.sum().item())

            # how many filters to prune this step
            k = int(num_unpruned * prune_ratio_step)
            if k <= 0:
                continue

            # compute L1-norm of each filter
            norms = weight.view(out_channels, -1).abs().sum(dim=1)
            # norms of only currently unmasked filters
            unpruned_norms = norms[unmasked]

            # find the k-th smallest norm among unpruned
            threshold = unpruned_norms.kthvalue(k).values.item()

            # mark filters with norm <= threshold (and not already pruned)
            to_prune = (norms <= threshold) & unmasked

            # zero out entire filters
            for i in range(out_channels):
                if to_prune[i]:
                    weight[i].zero_()

            # update mask
            pruned_masks[name] = mask | to_prune

            print(f"Conv Layer {name}: pruned {to_prune.sum().item()} filters (structured)")

    return pruned_masks

def zero_out_pruned_weights(model, pruned_masks):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in pruned_masks:
                continue
            mask = pruned_masks[name]

            if isinstance(module, nn.Linear):
                # Unstructured FC mask
                module.weight.data[mask] = 0.0

            elif isinstance(module, nn.Conv2d):
                # Unstructured per‐weight mask (4D)
                if mask.shape == module.weight.data.shape:
                    module.weight.data[mask] = 0.0

                # Fallback: structured filter‐level mask (1D over out_channels)
                elif mask.dim() == 1 and mask.shape[0] == module.weight.data.size(0):
                    for i in range(mask.shape[0]):
                        if mask[i]:
                            module.weight.data[i].zero_()

                else:
                    print(f"shape of mask: {mask.shape}")
                    print(f"shape of module.weight.data: {module.weight.data.shape}")
                    print(f"[Warning] Skipping Conv2d {name}: unsupported mask")




def save_initial_weights(model):
    """Save the initial model weights before pruning for later reinitialization."""
    return {name: param.clone().detach() for name, param in model.state_dict().items()}

def reinitialize_model(model, initial_weights):
    """Reinitializes the model's weights from stored initial weights after pruning."""
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if name in initial_weights:
                param.copy_(initial_weights[name])  # Reset unpruned weights

def reset_batchnorm_layers(model):
    """Reinitialize BatchNorm layers to prevent corrupted statistics after pruning."""
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
            module.reset_running_stats()
            module.train()  # Ensure it updates during training

def reset_optimizer(optimizer, model, lr):
    """Reinitialize the optimizer after pruning to remove old momentum states."""
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)

def cut(x):
    """Randomly cuts and pastes a segment of the image."""
    x_gen = x.clone()
    half = x_gen.shape[2] // 2
    pl = random.randint(0, half - 1)
    pl2 = random.randint(0, half - 1)
    x_gen[:, :, pl:pl + half] = x_gen[:, :, pl2:pl2 + half]
    return x_gen

def rot(x):
    """Randomly rotates a part of the image."""
    x_gen = x.clone()
    half = x_gen.shape[1] // 2
    pl = random.randint(0, half - 1)
    rnd = random.randint(1, 3)
    rotated_patch = torch.rot90(x_gen[:, pl:pl + half, pl:pl + half], k=rnd, dims=(1, 2))
    x_gen[:, pl:pl + half, pl:pl + half] = rotated_patch
    return x_gen

def blur(x):
    """Applies Gaussian blur with a random odd kernel size."""
    sz = random.choice([3, 5, 7])  # Ensure the kernel size is always an odd number
    func = transforms.GaussianBlur(kernel_size=sz, sigma=(0.1, 2))
    return func(x)

def shuffle(x):
    """Randomly shuffles chunks of the image along height or width."""
    x_gen = x.clone()
    rnd = random.randint(0, 1)
    size = x_gen.shape[1]
    chunk_size = size // 4
    indices = list(range(0, size, chunk_size))
    chunks = [x_gen[:, i:i + chunk_size, :] for i in indices]
    random.shuffle(chunks)
    x_gen = torch.cat(chunks, dim=1) if rnd == 0 else torch.cat(chunks, dim=2)
    return x_gen

def paint(x):
    """Randomly paints parts of the image."""
    x_gen = x.clone()
    size = x_gen.shape[2]
    sq = 4
    pl = random.randint(sq, size - sq * 2)
    pl2 = random.randint(sq, size - sq - 1)
    rnd = random.randint(0, 1)

    if rnd == 0:
        for i in range(sq, size - sq):
            x_gen[:, i, pl:pl + sq] = x_gen[:, pl2, pl:pl + sq]
    elif rnd == 1:
        for i in range(sq, size - sq):
            x_gen[:, pl:pl + sq, i] = x_gen[:, pl:pl + sq, pl2]
    return x_gen

def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = dist.view(x.shape[0], -1)

        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        dist = dist / dist_norm

        dist *= epsilon

        dist = dist.view(x.shape)

        x = (original_x + dist) * mask.float() + x * (1 - mask.float())

    else:
        raise NotImplementedError

    return x

class FastGradientSignUntargeted():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, device='cpu', _type='linf'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type
        self.device = device
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        x = original_images.to(self.device)
        x.requires_grad = True 
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, labels).to(self.device)
                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]
                x.data += self.alpha * torch.sign(grads.data) 
                x = project(x, original_images, self.epsilon, self._type)
        return x
    


class FastGradientSignTargeted():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, device='cpu', _type='linf'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type
        self.device = device

    def perturb(self, original_images, target_labels, reduction4loss='mean', random_start=False):
        x = original_images.to(self.device)
        x.requires_grad = True 

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = self.model(x)
                
                # Targeted attack: maximize probability of the target class
                loss = -F.cross_entropy(outputs, target_labels).to(self.device)  # Negate loss for gradient ascent
                
                grads = torch.autograd.grad(loss, x, only_inputs=True)[0]
                x.data -= self.alpha * torch.sign(grads.data)  # Gradient descent (instead of ascent)

                # Project back to epsilon-ball
                x = project(x, original_images, self.epsilon, self._type)

        return x