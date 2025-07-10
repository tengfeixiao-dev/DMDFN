import torch
from torch import nn
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.ch_model import rob_hub_cc, rob_hub_cme,RobHub
import random
import numpy as np
from utils.data_loader import data_loader

# global variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str


class ChConfig(object):
    """Configuration class to store the configurations of training.
    """

    def __init__(self,
                 train_mode='regression',
                 loss_weights={
                     'M': 1,
                     'T': 0,
                     'A': 0.3,
                     'V': 0.3,
                 },
                 model_save_path='checkpoint/',
                 learning_rate=5e-6,
                 epochs=20,
                 dataset_name='mosei',
                 early_stop=32,
                 seed=0,
                 dropout=0.3,
                 model='cc',
                 batch_size=16,
                 multi_task=True,
                 model_size='small',
                 cme_version='v1',
                 num_hidden_layers=1,
                 tasks='M',  # 'M' or 'MTA',
                 context=True,
                 text_context_len=2,
                 audio_context_len=1,
                 t_in_dim=1024,
                 a_in_dim=1024,
                 v_in_dim=29,
                 proj_dim=1024,
                 proj_heads=8,
                 token_length=8,
                 proj_depth=1,
                 proj_mlp_dim=1024,
                 a_input_length=96,
                 v_input_length=96,
                 fusion_layer_depth=3,
                 l_input_length=96,
                 net_layers=3
                 ):
        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.model = model
        self.batch_size = batch_size
        self.multi_task = multi_task
        self.model_size = model_size
        self.cme_version = cme_version
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.t_in_dim = t_in_dim
        self.a_in_dim = a_in_dim
        self.v_in_dim = v_in_dim
        self.proj_dim = proj_dim
        self.proj_heads=proj_heads
        self.token_length=token_length
        self.proj_depth=proj_depth
        self.proj_mlp_dim=proj_mlp_dim
        self.a_input_length=a_input_length
        self.v_input_length=v_input_length
        self.fusion_layer_depth=fusion_layer_depth
        self.l_input_length=l_input_length
        self.net_layers=net_layers
        
        
class ChTrainer():
    def __init__(self, config):
 
        self.config = config
        self.tasks = config.tasks
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        
    def do_train(self, model, data_loader):    
        model.train()  
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
        total_loss = 0
        input_size = 0
        # Loop over all batches.         
        for batch in data_loader:                    
            text_inputs = batch["text_tokens"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            targets = batch["targets"]
            visual_inputs = batch["visual_inputs"].to(device)

            optimizer.zero_grad()                    # To zero out the gradients.

            outputs,A_dl,V_dl,recon_v_loss,recon_a_loss = model(text_inputs, text_mask,   audio_inputs,
                                audio_mask, visual_inputs)
            
            # Compute the training loss.
            loss = 0.0         
            for m in self.tasks:
                sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets[m].to(device).view(-1, 1))
                loss += sub_loss
#                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
            total_loss += loss.item()*text_inputs.size(0)
            input_size += text_inputs.size(0)
            loss=loss+0.2*V_dl+0.2*A_dl+0*recon_v_loss+0*recon_a_loss
            loss.backward()                   
            optimizer.step()                
                
#         for m in self.tasks:
#             train_loss[m] = round(train_loss[m] / len(data_loader.dataset), 4)
        total_loss = round(total_loss / input_size, 4)
#         print('TRAIN'+" >> loss: ",total_loss, "   M_loss: ", train_loss['M'], "  T_loss: ", train_loss['T'], "  A_loss: ", train_loss['A'])
        return total_loss


    def do_test(self, model, data_loader, mode):    

        model.eval()                                # Put the model in training mode.              
        y_pred = {'M': [], 'T': [], 'A': [],'V': []}
        y_true = {'M': [], 'T': [], 'A': [],'V': []}
        total_loss = 0
        val_loss = {
            'M':0,
            'T':0,
            'A':0,
            'V': 0
        }
        input_size = 0
        with torch.no_grad():
            for batch in data_loader:                    # Loop over all batches.

                text_inputs = batch["text_tokens"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                targets = batch["targets"]

                #视觉数据
                visual_inputs = batch["visual_inputs"].to(device)
                
                outputs,A_dl,V_dl,recon_v_loss,recon_a_loss = model(text_inputs, text_mask,   audio_inputs,
                                audio_mask, visual_inputs)

                # Compute the training loss.
                loss = 0.0         
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets[m].to(device).view(-1, 1))
                    loss += sub_loss
                    val_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item()*text_inputs.size(0)
                input_size += text_inputs.size(0)
                

                # add predictions
                for m in self.tasks:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(targets[m].cpu())
                
        for m in self.tasks:
            val_loss[m] = round(val_loss[m] / input_size, 4)
        total_loss = round(total_loss / input_size, 4)
        print(mode+" >> loss: ",total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A']," V_loss: ",val_loss['V'])

        eval_results = {}
        for m in self.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            print('%s: >> ' %(m) + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.tasks[0]]
        eval_results['Loss'] = total_loss
        
        return eval_results

def ChRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name)

    if config.model=='cc':
        model = rob_hub_cc(config).to(device)
    elif config.model=='cme':
        model = rob_hub_cme(config).to(device)
    elif config.model == 'test':
        model = RobHub(config).to(device)
    for param in model.hubert_model.feature_extractor.parameters():
        param.requires_grad = False
               
    trainer = ChTrainer(config)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")

        # if eval_results['Loss'] < lowest_eval_loss:
        #     lowest_eval_loss = eval_results['Loss']
        #     torch.save(model.state_dict(),
        #                config.model_save_path + f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth')
        #     best_epoch = epoch
        # if eval_results['Has0_acc_2'] >= highest_eval_acc:
        #     highest_eval_acc = eval_results['Has0_acc_2']
        #     torch.save(model.state_dict(),
        #                config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
        if epoch - best_epoch >= config.early_stop:
            break
        # model.load_state_dict(torch.load(config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth'))
        test_results_loss = trainer.do_test(model, test_loader, "TEST")
        print('%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_loss))
        # if test_results_loss['Non0_acc_2'] >= highest_eval_acc:
        #     highest_eval_acc = test_results_loss['Non0_acc_2']
        #     torch.save(model.state_dict(),
        #                config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
        #     best_epoch = epoch
        if test_results_loss['Corr'] >= highest_eval_acc:
            highest_eval_acc = test_results_loss['Corr']
            torch.save(model.state_dict(),
                       config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
            best_epoch = epoch



