import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.context_model import RobD2vContext
from utils.scheduler import get_scheduler
import random
import numpy as np
from utils.data_loader import data_loader
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


class EnConfig(object):
    """Configuration class to store the configurations of training.
    """

    def __init__(self,
                 train_mode='regression',
                 loss_weights={
                     'M': 1,
                     'T': 0,
                     'A': 0.5,
                     'V': 0.5,
                 },
                 model_save_path='checkpoint/',
                 learning_rate=5e-6,
                 epochs=20,
                 dataset_name='mosei',
                 early_stop=8,
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


class EnTrainer():
    def __init__(self, config):

        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        # scheduler_warmup = get_scheduler(optimizer, self.config.epochs)
        #print(f'Learning Rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
        total_loss = 0
        # Loop over all batches.
        # is_terminal = sys.stdout.isatty()  # 判断输出是否为终端
        for batch in data_loader:
        # for batch in tqdm(data_loader,ascii=True):
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)

            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)

            visual_inputs = batch["visual_inputs"].to(device)
            visual_mask = batch["visual_mask"].to(device)

            targets = batch["targets"].to(device).view(-1, 1)

            optimizer.zero_grad()  # To zero out the gradients.

            if self.config.context:
                # outputs= model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                #                 audio_mask, audio_context_inputs, audio_context_mask,visual_inputs,visual_mask)
                outputs,A_dl,V_dl,recon_v_loss,recon_a_loss = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                audio_mask, audio_context_inputs, audio_context_mask,visual_inputs,visual_mask)
            else:
                outputs,lld_lv,lld_la = model(text_inputs, text_mask, audio_inputs, audio_mask)

            # Compute the training loss.
            if self.config.multi_task:
                loss = 0.0
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    loss += sub_loss
                #                 train_loss[m] += sub_loss.item()*text_inputs.size(0)
                total_loss += loss.item() * text_inputs.size(0)
            else:
                loss = self.criterion(outputs['M'], targets)
                total_loss += loss.item() * text_inputs.size(0)
            loss=loss+0.2*V_dl+0.2*A_dl+0*recon_v_loss+0*recon_a_loss
            loss.backward()
            optimizer.step()
        # scheduler_warmup.step()

        total_loss = round(total_loss / len(data_loader.dataset), 4)
        #         print('TRAIN'+" >> loss: ",total_loss)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()  # Put the model in eval mode.
        if self.config.multi_task:
            y_pred = {'M': [], 'T': [], 'A': [],'V': []}
            y_true = {'M': [], 'T': [], 'A': [],'V': []}
            total_loss = 0
            val_loss = {
                'M': 0,
                'T': 0,
                'A': 0,
                'V': 0,
            }
        else:
            y_pred = []
            y_true = []
            total_loss = 0

        with torch.no_grad():
            # is_terminal = sys.stdout.isatty()
            for batch in data_loader:
            # for batch in tqdm(data_loader, ascii=True):  # Loop over all batches.
                #文本数据
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)
                #语音数据
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)
                #视觉数据
                visual_inputs = batch["visual_inputs"].to(device)
                visual_mask = batch["visual_mask"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                if self.config.context:
                    # outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                    #             audio_mask, audio_context_inputs, audio_context_mask,visual_inputs,visual_mask)
                    outputs,A_dl,V_dl,recon_v_loss,recon_a_loss = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                    audio_mask, audio_context_inputs, audio_context_mask,visual_inputs,visual_mask)
                else:
                    outputs,lld_lv,lld_la = model(text_inputs, text_mask, audio_inputs, audio_mask)

                # Compute loss.
                if self.config.multi_task:
                    loss = 0.0
                    for m in self.tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                        loss += sub_loss
                        val_loss[m] += sub_loss.item() * text_inputs.size(0)
                    total_loss += loss.item() * text_inputs.size(0)
                    # add predictions
                    for m in self.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs['M'], targets)
                    total_loss += loss.item() * text_inputs.size(0)

                    # add predictions
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(targets.cpu())

        if self.config.multi_task:
            for m in self.tasks:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            # print(mode + " >> loss: ", total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'],
            #       "  A_loss: ", val_loss['A'],"V_loss:",val_loss['V'],lld_la.item(),lld_lv.item())
            print(mode + " >> loss: ", total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'],
                  "  A_loss: ", val_loss['A'],"V_loss:",val_loss['V'])
            print(A_dl,V_dl,"v_s:",model.v_s_param.item(),"v_c:",model.v_c_param.item(),"a_s:",model.a_s_param.item(),"v_c:",model.a_c_param.item())
            

            eval_results = {}
            for m in self.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                results = self.metrics(pred, true)
                print('%s: >> ' % (m) + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.tasks[0]]
            eval_results['Loss'] = total_loss
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode + " >> loss: ", total_loss)

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            eval_results = self.metrics(pred, true)
            print('%s: >> ' % ('M') + dict_to_str(eval_results))
            eval_results['Loss'] = total_loss

        return eval_results


def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config.batch_size, config.dataset_name,
                                                        text_context_length=config.text_context_len,
                                                        audio_context_length=config.audio_context_len)
    model = RobD2vContext(config)
    model.load_state_dict(torch.load("/home/xiaotengfei/MMML-cp/checkpoint/RH_acc_mosi_111_0.8979.pth", map_location=device))
    model.to(device)
    model.eval()

        # 3. 提取特征和标签
    A_special_features = []
    V_special_features = []
    A_common_features = []
    V_common_features = []

    with torch.no_grad():
         # is_terminal = sys.stdout.isatty()
        for batch in test_loader:
        # for batch in tqdm(data_loader, ascii=True):  # Loop over all batches.
            #文本数据
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)
            #语音数据
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)
            #视觉数据
            visual_inputs = batch["visual_inputs"].to(device)
            visual_mask = batch["visual_mask"].to(device)

            targets = batch["targets"].to(device).view(-1, 1)

            # 模型需要支持输出中间特征
            outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs,
                                    audio_mask, audio_context_inputs, audio_context_mask,visual_inputs,visual_mask, return_features=True)
            A_special_feature = outputs["A_special_feature"].squeeze(0).cpu().numpy() # 可替换为 h_A^s, h_V^s 等
            V_special_feature = outputs["V_special_feature"].squeeze(0).cpu().numpy()
            A_common_feature = outputs["A_common_feature"].squeeze(0).cpu().numpy()
            V_common_feature = outputs["V_common_feature"].squeeze(0).cpu().numpy()
            A_special_features.append(A_special_feature)
            V_special_features.append(V_special_feature)
            A_common_features.append(A_common_feature)
            V_common_features.append(V_common_feature)
                # 1. 对 Audio 特征进行 t-SNE 降维
    print(len(V_common_features))
    A_combined = np.concatenate([A_common_features, A_special_features], axis=0)
    A_labels = np.array([0]*len(A_common_features) + [1]*len(A_special_features))  # 0: common, 1: special

    A_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(A_combined)

   
    # 2. 对 Visual 特征进行 t-SNE 降维
    V_combined = np.concatenate([V_common_features, V_special_features], axis=0)
    V_labels = np.array([0]*len(V_common_features) + [1]*len(V_special_features))  # 0: common, 1: special

    V_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42).fit_transform(V_combined)

    # 3. 绘图
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Audio 图
    axs[0].scatter(A_tsne[A_labels==0, 0], A_tsne[A_labels==0, 1], c='red', label='Audio-Shared (A^c)', alpha=0.6, s=15)
    axs[0].scatter(A_tsne[A_labels==1, 0], A_tsne[A_labels==1, 1], c='blue', label='Audio-Specific (A^s)', alpha=0.6, s=15)
    axs[0].set_title("t-SNE of Audio Shared vs. Specific Features")
    axs[0].legend()
    axs[0].set_xticks([]); axs[0].set_yticks([])

    # Visual 图
    axs[1].scatter(V_tsne[V_labels==0, 0], V_tsne[V_labels==0, 1], c='red', label='Visual-Shared (V^c)', alpha=0.6, s=15)
    axs[1].scatter(V_tsne[V_labels==1, 0], V_tsne[V_labels==1, 1], c='blue', label='Visual-Specific (V^s)', alpha=0.6, s=15)
    axs[1].set_title("t-SNE of Visual Shared vs. Specific Features")
    axs[1].legend()
    axs[1].set_xticks([]); axs[1].set_yticks([])

    plt.tight_layout()
    plt.savefig("tsne_visualization.png", dpi=300)  # 保存为高清PNG图像
    plt.show()      

    


    # if config.context:
    #     if config.model == 'cc':
    #         model = rob_d2v_cc_context(config).to(device)
    #     elif config.model == 'cme':
    #         model = rob_d2v_cme_context(config).to(device)
    #     elif config.model =='test':
    #         model=RobD2vContext(config).to(device)
    #     # for param in model.data2vec_model.feature_extractor.parameters():
    #     #     param.requires_grad = False #冻结模型中，对语音特征提取器参数的训练
    # else:
    #     if config.model == 'cc':
    #         model = rob_d2v_cc(config).to(device)
    #     elif config.model == 'cme':
    #         model = rob_d2v_cme(config).to(device)
    # # for param in model.data2vec_model.feature_extractor.parameters():
    # #     param.requires_grad = False

    # trainer = EnTrainer(config)

    # lowest_eval_loss = 100
    # highest_eval_acc = 0
    # epoch = 0
    # best_epoch = 0
    # while True:
    #     print('---------------------EPOCH: ', epoch, '--------------------')
    #     epoch += 1
    #     trainer.do_train(model, train_loader)
    #     eval_results = trainer.do_test(model, val_loader, "VAL")

    #     # if eval_results['Loss'] < lowest_eval_loss:
    #     #     lowest_eval_loss = eval_results['Loss']
    #     #     torch.save(model.state_dict(),
    #     #                config.model_save_path + f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth')
    #     #     best_epoch = epoch
    #     # if eval_results['Has0_acc_2'] >= highest_eval_acc:
    #     #     highest_eval_acc = eval_results['Has0_acc_2']
    #     #     torch.save(model.state_dict(),
    #     #                config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
    #     if epoch - best_epoch >= config.early_stop:
    #         break
    #     # model.load_state_dict(torch.load(config.model_save_path+f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth'))
    #     test_results_loss = trainer.do_test(model, test_loader, "TEST")
    #     print('%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    #     if test_results_loss['Non0_acc_2'] >= highest_eval_acc:
    #         highest_eval_acc = test_results_loss['Non0_acc_2']
    #         torch.save(model.state_dict(),
    #                    config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
    #         best_epoch = epoch



    # model.load_state_dict(
    #     torch.load(config.model_save_path + f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth'))
    # test_results_acc = trainer.do_test(model, test_loader, "TEST")
    # print('%s: >> ' % ('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
    
