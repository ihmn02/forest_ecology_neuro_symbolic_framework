import random
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.distributions.beta import Beta
import torch.nn as nn
from torchsummary import summary
import torchmetrics

from utils import Hsi_raster, trans
from model import RuleEncoder, DataEncoder, Net, Frickernet
from utils_learning import verification, get_perturbed_input


data_path = "./data/ns_test1/"
train_data_file = "train_data.hdf5"
val_data_file = "val_data.hdf5"
test_data_file = "test_data.hdf5"

model_info = {'dataonly': {'rule': 0.0, 'lr': 0.00001},
              'ours-beta1.0': {'beta': [1.0], 'scale': 1.0, 'lr': 0.001},
              'ours-beta0.1': {'beta': [0.1], 'scale': 1.0, 'lr': 0.001},
              'ours-beta0.1-scale0.1': {'beta': [0.1], 'scale': 0.1},
              'ours-beta0.1-scale0.01': {'beta': [0.1], 'scale': 0.01},
              'ours-beta0.1-scale0.05': {'beta': [0.1], 'scale': 0.05},
              'ours-beta0.1-pert0.001': {'beta': [0.1], 'pert': 0.001},
              'ours-beta0.1-pert0.01': {'beta': [0.1], 'pert': 0.01},
              'ours-beta0.1-pert0.1': {'beta': [0.1], 'pert': 0.1},
              'ours-beta0.1-pert1.0': {'beta': [0.1], 'pert': 1.0},
             }

def main():
    device = 'cpu'
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_ds = Hsi_raster(data_path, train_data_file, transform=None, test=False)

    train_ds, val_ds = random_split(train_ds, [2048, 1024]) #[100064, 10000]
    #val_ds = Hsi_raster(data_path, val_data_file, transform=None, test=True)
    test_ds = Hsi_raster(data_path, test_data_file, transform=None, test=True)

    train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_ds, batch_size=32, shuffle=True)

    batch_size = 2
    print("data size: {}/{}/{}".format(len(train_ds), len(val_ds), len(test_ds)))

    model_type = 'dataonly' # 'ours-beta1.0'
    if model_type not in model_info:
        # default setting
        lr = 0.001
        pert_coeff = 0.1
        scale = 1.0
        beta_param = [1.0]
        alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
        model_params = {}

    else:
        model_params = model_info[model_type]
        lr = model_params['lr'] if 'lr' in model_params else 0.001
        pert_coeff = model_params['pert'] if 'pert' in model_params else 0.1
        scale = model_params['scale'] if 'scale' in model_params else 1.0
        beta_param = model_params['beta'] if 'beta' in model_params else [1.0]

    if len(beta_param) == 1:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[0]))
    elif len(beta_param) == 2:
      alpha_distribution = Beta(float(beta_param[0]), float(beta_param[1]))

    class_weights = compute_class_weight('balanced', range(8), train_ds.dataset.y)
    print('class weights: ', class_weights)
    class_weight_dict = {}
    for i in range(8):
        class_weight_dict[i] = class_weights[i]

    class_weight_tensor = torch.tensor(list(class_weight_dict.values()), dtype=torch.float32)

    print('model_type: {}\tscale:{}\tBeta distribution: Beta({})\tlr: {}\t \tpert_coeff: {}'.format(model_type, scale, beta_param, lr, pert_coeff))

    merge = 'add'
    input_dim_rule_encoder = 1
    input_dim_data_encoder = 32
    input_dim_db = 256
    output_dim_encoder = 128
    hidden_dim_encoder = 256
    hidden_dim_db = 16
    n_layers = 5
    output_dim = 8
    rule_ind = 3

    rule_encoder = RuleEncoder(input_dim_rule_encoder)
    data_encoder = Frickernet(8, 32, 128, 6)  #DataEncoder(input_dim_data_encoder)
    #model = Net(input_dim_db, output_dim, rule_encoder, data_encoder, hidden_dim=hidden_dim_db, n_layers=n_layers, merge=merge, skip=False, input_type='state').to(device)  # Not residual connection
    model = Frickernet(8, 32, 128, 6)
    #summary(model, (32, 15, 15), 32)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_rule_func = lambda x, y: torch.mean(F.relu(x - y))  # if x>y, penalize it.
    loss_task_func = nn.CrossEntropyLoss(weight=class_weight_tensor)  # return scalar (reduction=mean)

    epochs = 20
    early_stopping_thld = 10
    counter_early_stopping = 10
    valid_freq = 1     # print training data every valid_freq epochs

    #saved_filename = 'spec-class_{}_rule-{}_src{}-target{}_seed{}.demo.pt'.format(model_type, rule_feature, src_usual_ratio, src_usual_ratio, seed)
    #saved_filename = os.path.join('saved_models', saved_filename)
    #print('saved_filename: {}\n'.format(saved_filename))
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_train_x, batch_train_chm, batch_train_dem, batch_train_y in train_dataloader:
            # batch_train_y = batch_train_y.unsqueeze(-1)

            optimizer.zero_grad()

            if model_type.startswith('dataonly'):
                alpha = 0.0
            elif model_type.startswith('ruleonly'):
                alpha = 1.0
            elif model_type.startswith('ours'):
                alpha = alpha_distribution.sample().item()

            # stable output
            #output = model(batch_train_x, batch_train_chm, alpha=alpha)
            output = model(batch_train_x)
            loss_task = loss_task_func(output, batch_train_y)

            # perturbed input and its output
            #pert_batch_train_chm = batch_train_chm.detach().clone()
            #pert_batch_train_chm = get_perturbed_input(pert_batch_train_chm, pert_coeff)
            #pert_output = model(batch_train_x, pert_batch_train_chm, alpha=alpha)

            #loss_rule = loss_rule_func(output[:, rule_ind], pert_output[:, rule_ind])  # output should be less than pert_output

            #loss = alpha * loss_rule + scale * (1 - alpha) * loss_task
            loss = loss_task
            #print(loss)

            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        if epoch % valid_freq == 0:
            model.eval()
            if model_type.startswith('ruleonly'):
                alpha = 1.0
            else:
                alpha = 0.0

            with torch.no_grad():
                for val_x, val_chm, val_dem, val_y in val_dataloader:
                    #val_y = val_y.unsqueeze(-1)
                    model.eval()
                    #output = model(val_x, val_chm, alpha=alpha)
                    output = model(val_x)
                    #print(output[0])
                    #print(val_y[0])
                    #print()
                    val_loss_task = loss_task_func(output, val_y).item()

                    # perturbed input and its output
                    #pert_val_chm = val_chm.detach().clone()
                    #pert_val_chm = get_perturbed_input(pert_val_chm, pert_coeff)
                    #pert_output = model(val_x, pert_val_chm, alpha=alpha)  # \hat{y}_{p}    predicted sales from perturbed input

                    val_loss_rule = 0 #loss_rule_func(output[:, rule_ind], pert_output[:, rule_ind]).item()
                    val_ratio = 0 #verification(pert_output, output, threshold=0.0).item()

                    val_loss = val_loss_task

                    y_true = val_y.cpu().numpy()
                    y_score = output.cpu().numpy()
                    y_pred = np.argmax(y_score, axis=1)
                    #val_acc = 100 * accuracy_score(y_true, np.argmax(y_pred, axis=1))

                    val_acc = 100 * accuracy_score(y_true, y_pred)

                if val_loss < best_val_loss:
                    counter_early_stopping = 1
                    best_val_loss = val_loss
                    #best_model_state_dict = deepcopy(model.state_dict())
                    print(
                        '[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f} best model is updated %%%%'
                        .format(epoch, best_val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio))
                    #torch.save({
                         #'epoch': epoch,
                         #'model_state_dict': best_model_state_dict,
                         #'optimizer_state_dict': optimizer.state_dict(),
                         #'loss': best_val_loss
                    #}, saved_filename)
                else:
                    print('[Valid] Epoch: {} Loss: {:.6f} (alpha: {:.2f})\t Loss(Task): {:.6f} Acc: {:.2f}\t Loss(Rule): {:.6f}\t Ratio: {:.4f}({}/{})'
                        .format(epoch, val_loss, alpha, val_loss_task, val_acc, val_loss_rule, val_ratio, counter_early_stopping, early_stopping_thld))
                    if counter_early_stopping >= early_stopping_thld:
                        break
                    else:
                        counter_early_stopping += 1

if __name__ == '__main__':
  main()
