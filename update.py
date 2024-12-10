import torch
import copy
import math
import random
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

class DatasetSplit(Dataset):
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        #self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        #idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        #idxs_test = idxs[int(0.9*len(idxs)):]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True)
        #validloader = DataLoader(DatasetSplit(dataset, idxs_val), batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=int(len(idxs_test)/10), shuffle=False)
        #return trainloader, validloader, testloader
        return trainloader, testloader  

    def update_weights(self, model, global_round):
        # Set mode to train model

        global_model = copy.deepcopy(model)

        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)

                # proximal term in fedprox acts as a kind of L2 regularization
                proximal_term = 0.0
                for w, w_t in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (w - w_t).norm(2)
                # for fedavg mu = 0, but for fedprox mu is given during runtime
                loss = self.criterion(log_probs, labels) + (self.args.mu / 2) * proximal_term

                #loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(global_round, iter, batch_idx * len(images), len(self.trainloader.dataset), 100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss

def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    total, correct = 0.0, 0.0
    loss = []

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss.append(batch_loss.item())

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, sum(loss)/len(loss)

def calculate_optimized_score(h, e):
    w_h, w_e = random.uniform(0.3, 7.0), random.uniform(0.5, 1.0)         # Selection can be controlled by varying the respective ranges for w_h, w_e
    #w_e, w_h = random.uniform(0.7, 1.0), random.uniform(0, 0.3)        # Greedy
    #w_h, w_e = 0.1, 0.9

    h_term = w_h * (h**2 / math.sqrt(h + 0.1))  # non-linear term for h_score
    e_term = w_e * math.log(e + 1)  # log term for e_consumption
    exp_term = math.exp(-e / 50)  # exponential term for e_consumption
    return h_term + e_term + exp_term

def select_clients(args, m, h_score, e_consumption, damping):
    #print("Updated damping list before selection:", damping)

    if args.select == 0:
        selected_idxs = np.random.choice(range(args.num_users), m, replace=False)

        total_energy_consumed = sum(e_consumption[client] for client in selected_idxs)
        print("Total energy consumed by selected clients:", total_energy_consumed)

    elif args.select == 1:
        # Min-max normalization for e_consumption
        min_e = min(e_consumption)
        max_e = max(e_consumption)

        normalized_e_consumption = [(e - min_e) / (max_e - min_e) for e in e_consumption]

        # Filter indexes where damping == 0
        filtered_idxs = [i for i in range(len(damping)) if damping[i] == 0]

        # Calculate optimized score for each filtered index using normalized e_consumption, Sort the indexes based on the optimized score (lower is better)
        #optimized_scores = [(i, w_h * h_score[i] + w_e * normalized_e_consumption[i]) for i in filtered_idxs]
        optimized_scores = [(i, calculate_optimized_score(h_score[i], normalized_e_consumption[i])) for i in filtered_idxs]

        sorted_idxs = sorted(optimized_scores, key=lambda x: x[1])
        sorted_idxs = sorted_idxs[:m+1]

        selected_idxs = [client[0] for client in sorted_idxs]

        damping = [0] * len(damping)    # Reset damping list to all zeros

        # Output the sorted indexes and their optimized scores
        for client in sorted_idxs:
            index = client[0]
            damping[index] = 1
            #print(f"Client: {client[0]}, Optimized Score: {client[1]:.2f}")

        total_energy_consumed = sum(e_consumption[client[0]] for client in sorted_idxs)
        #print("Updated damping list after selection:", damping)
        print("Total energy consumed by selected clients:", total_energy_consumed)

    return selected_idxs, damping, total_energy_consumed
