
import os
import copy
import time, csv
import pickle
import argparse
import random
from collections import Counter
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from update import LocalUpdate, test_inference, select_clients

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where the keys are the user index and the values are the corresponding data for each of those users.
    """
    if args.dataset == 'cifar':
        data_dir = 'D:/Conferences/CCNC_Demo/Data/cifar/'
        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups, h_score = cifar_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = 'D:/Conferences/CCNC_Demo/Data/mnist/'
        else:
            data_dir = 'D:/Conferences/CCNC_Demo/Data/fmnist/'

        apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups, h_score = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups, h_score

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    #'''
    # Calculate overall dataset distribution
    dataset_distribution = Counter(labels)
    total_classes = len(dataset_distribution)
    
    # Print user details and calculate heterogeneity scores
    heterogeneity_scores = []
    for i in range(num_users):
        user_labels = labels[dict_users[i].astype(int)]
        class_distribution = Counter(user_labels)
        total_samples = len(dict_users[i])
        
        heterogeneity_score = calculate_heterogeneity_score(class_distribution, dataset_distribution, total_classes)
        heterogeneity_scores.append(heterogeneity_score)
        
        #print(f"User {i}:")
        #print(f"  Total samples: {total_samples}")
        #print("  Samples per class:")
        for class_label in range(10):  # MNIST has 10 classes
            count = class_distribution[class_label]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            #print(f"    Class {class_label}: {count} ({percentage:.2f}%)")
        #print(f"  Heterogeneity Score: {heterogeneity_score:.4f}")
        #print()
    
    print(f"Average Heterogeneity Score: {np.mean(heterogeneity_scores):.4f}")
    print(f"Heterogeneity Score Standard Deviation: {np.std(heterogeneity_scores):.4f}")

    return dict_users, heterogeneity_scores

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    return dict_users

def cifar_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset s.t clients have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain number of training imgs
    """
    # 50,000 training imgs --> 50 imgs/shard X 1000 shards
    num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1, size=num_users)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate((dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    # Calculate overall dataset distribution
    dataset_distribution = Counter(labels)
    total_classes = len(dataset_distribution)
    
    # Print user details and calculate heterogeneity scores
    heterogeneity_scores = []
    for i in range(num_users):
        user_labels = labels[dict_users[i].astype(int)]
        class_distribution = Counter(user_labels)
        total_samples = len(dict_users[i])
        
        heterogeneity_score = calculate_heterogeneity_score(class_distribution, dataset_distribution, total_classes)
        heterogeneity_scores.append(heterogeneity_score)
        
        #print(f"User {i}:")
        #print(f"  Total samples: {total_samples}")
        #print("  Samples per class:")
        for class_label in range(10):  # MNIST has 10 classes
            count = class_distribution[class_label]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            #print(f"    Class {class_label}: {count} ({percentage:.2f}%)")
        #print(f"  Heterogeneity Score: {heterogeneity_score:.4f}")
        #print()
    
    print(f"Average Heterogeneity Score: {np.mean(heterogeneity_scores):.4f}")
    print(f"Heterogeneity Score Standard Deviation: {np.std(heterogeneity_scores):.4f}")

    return dict_users, heterogeneity_scores

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--rounds', type=int, default=10, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument("--mu", type=float, default=0.01, help="proximal term constant")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets -- 32 for mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument("--algorithm", type=str, default="fedavg", help="specify which algorithm to use during local updates aggregation (fedprox, fedavg)",)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--select', type=int, default=0, help='Default set to Random Selection. Set to 1 for Selecting clients based on h_Score.')
    parser.add_argument('--unequal', type=int, default=0, help='whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=25, help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    return args

def calculate_heterogeneity_score(user_distribution, dataset_distribution, total_classes):
    """
    Calculate the heterogeneity score for a user.
    
    :param user_distribution: Counter object with the user's class distribution
    :param dataset_distribution: Counter object with the overall dataset class distribution
    :param total_classes: Total number of classes in the dataset
    :return: Heterogeneity score
    """
    user_classes = len(user_distribution)
    user_total_samples = sum(user_distribution.values())
    dataset_total_samples = sum(dataset_distribution.values())
    
    # Factor 1: Ratio of classes present in user data to total classes
    class_ratio = user_classes / total_classes
    
    # Factor 2: Distribution difference
    distribution_difference = 0
    for class_label in range(total_classes):
        user_class_ratio = user_distribution[class_label] / user_total_samples if user_total_samples > 0 else 0
        dataset_class_ratio = dataset_distribution[class_label] / dataset_total_samples
        distribution_difference += abs(user_class_ratio - dataset_class_ratio)
    
    # Normalize distribution difference
    distribution_difference /= 2  # Maximum possible difference is 2
    
    # Combine factors (you can adjust the weights if needed)
    heterogeneity_score = 0.5 * (1 - class_ratio) + 0.5 * distribution_difference
    
    return heterogeneity_score

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    #dataset_train = datasets.MNIST('D:/CCNC_Demo/Data/mnist/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    #num = 100
    #d = mnist_noniid(dataset_train, num)

    start_time = time.time()

    # define paths
    path_project = os.path.abspath('D:/Conferences/CCNC_Demo/')
    logger = SummaryWriter('D:/Conferences/CCNC_Demo/logs')

    args = args_parser()
    set_seed(args.seed)
    exp_details(args)
    
    # proximal term is 0.0 in case of fedavg
    if args.algorithm == "fedavg":
        args.mu = 0.0

    #if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, h_score = get_dataset(args)

    # Randomly Assign the Energy Consumption of all Clients
    e_consumption = [random.uniform(10, 50) for _ in range(args.num_users)]
    damping = np.array([0] * args.num_users)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    with open('D:/Conferences/CCNC_Demo/save/results.csv', mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Round', 'Test Acc', 'Variance', 'Energy'])
            writer.writeheader()

    for epoch in tqdm(range(args.rounds)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)

        #if args.select == 1:
        idxs_users, d_score, total_energy_consumed = select_clients(args, m, h_score, e_consumption, damping)
        #else:
            #idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        print(f"Selected Clients in Round : {epoch+1} |\n", idxs_users)

        damping = d_score

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            #local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx], logger=logger)
            #local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        var_acc = np.var(list_acc)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print("|---- Test Accuracy: {:.3f}%".format(100*test_acc))
        print("|---- Accuracy Variance: {:.5f}%".format(var_acc))
        print("|---- Test Loss: {:.3f}%".format(100*test_loss))

        with open('D:/Conferences/CCNC_Demo/save/results.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Round', 'Test Acc', 'Variance', 'Energy'])
            #writer.writeheader()
            data = [{'Round': epoch+1, 'Test Acc': test_acc, 'Variance': {round(var_acc, 4)}, 'Energy': total_energy_consumed}]
            writer.writerows(data)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.rounds} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = 'D:/Conferences/CCNC_Demo/save/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.format(args.dataset, args.model, args.rounds, args.frac, args.iid, args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.rounds, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.rounds, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))


    