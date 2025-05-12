import argparse
import torch
import numpy as np
import random
from pathlib import Path
import json
from models import *
from utils import *
from federated_algorithms import *
from dataloader import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--cd_lr', type=float, default=0.01, help='classifier learning rate (default: 0.01)')
    parser.add_argument('--partition', type=str, default='noniid-labeldir', help='data partitioning strategy')
    parser.add_argument('--epochs_classifier', type=int, default=5, help='epochs for classifier')
    parser.add_argument('--n_parties', type=int, default=10, help='number of clients (default: 10)')
    parser.add_argument('--device', type=str, default='cpu', help='device to run the program')
    parser.add_argument('--algorithm', type=str, default='fedavg', help='Federated algorithm to use')
    parser.add_argument('--rounds', type=int, default=10, help='number of federated learning rounds')
    parser.add_argument('--save_result_dict', action='store_true', help='save result dictionary')
    parser.add_argument('--datadir', type=str, default='./data/', help='data directory')
    parser.add_argument('--beta', type=float, default=0.1, help='parameter for data partitioning')
    parser.add_argument('--grid_size', type=int, default=4, help='Size of the grid for self supervised learning')
    parser.add_argument('--save_classifier', action='store_true', help='If true, save the classifier weights after training')
    parser.add_argument('--pruneratio', type=float, default=0.5, help='pruning ratio for fedprunedov')
    parser.add_argument('--model', type=str, default='SimpleCNN2', help='model to use')
    parser.add_argument('--ablation', action='store_true', help='ablation')
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # Set random seed for reproducibility
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    random.seed(args.init_seed)
    device = torch.device(args.device)
    num_perm = generate_permutations(grid_size=args.grid_size)
    channels, size = (
        (3, 32) if args.dataset in ['svhn', 'cifar10', 'cifar100'] 
        else (3, 28) if args.dataset == 'colored-mnist' 
        else (1, 28) if args.dataset in ['mnist', 'fashion-mnist']  
        else (3, 224)  
    )



    # Partition data and prepare data loaders
    print(f"Partitioning data using {args.partition} partitioning strategy")
    X_train, y_train, X_test, y_test, net_dataidx_map_train, net_dataidx_map_test, _, _ = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, beta=args.beta)
  
    
    train_loaders, test_loaders = [], []
    if args.dataset == 'colored-mnist':
        color_change = True
        rotation_change = False
        brightness_change = False
        scaling_change = False
        blur_change = False
        translation_change = False
        shearing_change = False
        gaussian_noise_change = False
        for client_id in range(args.n_parties):
            print(f"Preparing colored mnist for client: {client_id}")
            train_loader, test_loader = get_colored_mnist_loader2(
                args.datadir, args.batch_size, 256, 
                net_dataidx_map_train[client_id],
                net_dataidx_map_test[client_id], 
                client_id, drop_last=False,
                color_change= color_change,
                rotation_change= rotation_change,
                brightness_change= brightness_change,
                scaling_change= scaling_change,
                blur_change= blur_change,
                translation_change= translation_change,
                shearing_change= shearing_change,
                gaussian_noise_change= gaussian_noise_change
            )
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)
        print("Preparing colored mnist for last domain")
    else:
        print("loading for clients")
        for client_id in range(args.n_parties):
            print("client")
            train_loader, test_loader, _, _ = get_divided_dataloader(
                args.dataset, args.datadir, args.batch_size, 256,  
                net_dataidx_map_train[client_id], net_dataidx_map_test[client_id], noise_level=0)

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

            
    # save_client_images(train_loaders, test_loaders, output_dir= 'client_image_samples/'+args.dataset, save_all=True)

    # Initialize the global model
    n = 16
    out_dim = 100 if args.dataset == 'cifar100' else 10
    if  args.algorithm in ['fedov', 'fedeov', 'fedprunedeov', 'fedeov_untargetted', 'fedeov_noshuffle', 'distilled_ov', 'distilled_eov', 'distilled_prunedeov']:
        out_dim = out_dim + 1
    
    if args.model == "SimpleCNN2":
        if args.algorithm == 'fedmoon':
            global_model = SimpleCNN2_MOON(in_channels=channels, input_size=size, n_kernels=n, out_dim=out_dim).to(device)
        elif args.algorithm == 'fedconcat':
            global_encoder = SimpleCNN2Encoder(in_channels=channels, input_size=size, n_kernels=n).to(device)
            global_classifier = SimpleCNN2Classifier(in_dim=120, out_dim=out_dim).to(device)
        else:
            global_model = SimpleCNN2(in_channels=channels, input_size=size, n_kernels=n, out_dim=out_dim).to(device)


    # Train using the selected federated algorithm
    if args.algorithm == 'fedavg':
        round_accuracies = federated_average(global_model, train_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr, test_loaders)
    elif args.algorithm == 'fedprox':
        round_accuracies = federated_prox(global_model, train_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr, test_loaders)
    elif args.algorithm == 'scaffold':
        round_accuracies = federated_scaffold(global_model, train_loaders, test_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr)
    elif args.algorithm == 'fedgh':  
        round_accuracies = federated_gradient_harmonization(global_model, train_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr, test_loaders)
    elif args.algorithm == 'fedsam':
        round_accuracies = federated_sam(global_model, train_loaders, test_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr)
    elif args.algorithm == 'feddyn':
        round_accuracies = federated_feddyn(global_model, train_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr, test_loaders)
    elif args.algorithm == 'fedmoon':
        round_accuracies = federated_moon(global_model, train_loaders, args.n_parties, args.rounds, args.epochs_classifier, args.cd_lr, test_loaders)
    elif args.algorithm == 'fedconcat':
        round_accuracies = federated_concat(global_encoder, global_classifier, train_loaders, test_loaders, args)
    elif args.algorithm == 'fedov':
        round_accuracies = federated_ov(global_model, train_loaders, test_loaders, args.n_parties, args)
    elif args.algorithm == 'fedeov':
        round_accuracies = federated_eov(global_model, train_loaders, test_loaders, args.n_parties, args, args.pruneratio)
    elif args.algorithm == 'fedprunedeov':
        round_accuracies = federated_prunedeov(global_model, train_loaders, test_loaders, args.n_parties, args, args.pruneratio)
    elif args.algorithm == 'distilled_ov':
        round_accuracies = federated_distilled(global_model, train_loaders, test_loaders, args.n_parties, algorithm='ov', args=args)
    elif args.algorithm == 'distilled_eov':
        round_accuracies = federated_distilled(global_model, train_loaders, test_loaders, args.n_parties, algorithm='eov', args=args)
    elif args.algorithm == 'centralized':
        round_accuracies = centralized_training(global_model, train_loaders, test_loaders, args.epochs_classifier, args.cd_lr, args)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}") 

    # Save the global model if specified
    if args.partition == 'noniid-labeldir':
        partition = args.partition + "_" + str(args.beta)
    else:
        partition = args.partition
    if args.algorithm not in ['fedov',] and args.save_classifier:
        if args.algorithm == 'fedconcat':
            pass
        else:   
            global_weights_dir = Path("model_weights") / args.dataset / args.algorithm / f"partition_{partition}" / f"num_clients_{args.n_parties}"
                
            global_weights_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
            model_path = global_weights_dir / f"global_{args.algorithm}.pt"
            torch.save(global_model.state_dict(), model_path)
            print(f"Global model saved at {model_path}")

    # Save results if specified
    if args.save_result_dict:
        results = {
            "algorithm": args.algorithm,
            "seed": args.init_seed,
            "round_accuracy": round_accuracies
        }

        algorithm = args.algorithm + "_" + str(args.pruneratio) if args.algorithm in ['fedeov', 'fedprunedeov'] else args.algorithm

        if args.ablation == True:
            output_dir = Path("ablation_results") / algorithm / args.dataset / partition / f"clients_{args.n_parties}" / ("rounds_" + str(args.rounds)) / f"model_{args.model}" / f"dirichlet_{args.beta}" / f"epochs_classifier_{args.epochs_classifier}" / f"lr_pretraining_{args.lr_pretraining}"
        else:
            output_dir = Path("results") / algorithm / args.dataset / partition / f"clients_{args.n_parties}" / ("rounds_" + str(args.rounds))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"results_seed_{args.init_seed}.json", "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results saved at {output_dir}")


        print("Training complete.")
        print(f"Final round accuracies: {round_accuracies}")

if __name__ == "__main__":
    main()
