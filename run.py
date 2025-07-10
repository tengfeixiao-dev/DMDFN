import argparse
from utils.en_train import EnConfig, EnRun
from utils.ch_train import ChConfig, ChRun
from distutils.util import strtobool
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main(args):
    if args.dataset != 'sims':
        EnRun(EnConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks,
                                    cme_version=args.cme_version, dataset_name=args.dataset,num_hidden_layers=args.num_hidden_layers,
                                    context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len,
                                    t_in_dim=args.t_in_dim,a_in_dim=args.a_in_dim,v_in_dim=args.v_in_dim,proj_dim=args.proj_dim,
                                    proj_heads=args.proj_heads,token_length=args.token_length,proj_depth=args.proj_depth,proj_mlp_dim=args.proj_mlp_dim,
                                    a_input_length=args.a_input_length,v_input_length=args.v_input_length,fusion_layer_depth=args.fusion_layer_depth,
                                    l_input_length=args.l_input_length,net_layers=args.net_layers))
    else:
        ChRun(ChConfig(batch_size=args.batch_size,learning_rate=args.lr,seed=args.seed, model=args.model, tasks = args.tasks,
                                    cme_version=args.cme_version, dataset_name=args.dataset,num_hidden_layers=args.num_hidden_layers,
                                    context=args.context, text_context_len=args.text_context_len, audio_context_len=args.audio_context_len,
                                    t_in_dim=args.t_in_dim,a_in_dim=args.a_in_dim,v_in_dim=args.v_in_dim,proj_dim=args.proj_dim,
                                    proj_heads=args.proj_heads,token_length=args.token_length,proj_depth=args.proj_depth,proj_mlp_dim=args.proj_mlp_dim,
                                    a_input_length=args.a_input_length,v_input_length=args.v_input_length,fusion_layer_depth=args.fusion_layer_depth,
                                    l_input_length=args.l_input_length,net_layers=args.net_layers))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=111 , help='random seed,111 for mosi,2025 for mosei,1234 for sims')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-6
                        , help='learning rate, recommended: 5e-6 for mosi, mosei, 1e-5 for sims')
    parser.add_argument('--model', type=str, default='test', help='concatenate(cc) or cross-modality encoder(cme)')
    parser.add_argument('--cme_version', type=str, default='v1', help='version')
    parser.add_argument('--dataset', type=str, default='mosi', help='dataset name: mosi, mosei, sims')
    parser.add_argument('--num_hidden_layers', type=int, default=5, help='number of hidden layers for cross-modality encoder')
    parser.add_argument('--tasks', type=str, default='MAV', help='losses to train: M: multi-modal, T: text, A: audio (defalut: MTA))')
    parser.add_argument('--context', default=True, help='incorporate context or not', dest='context', type=lambda x: bool(strtobool(x)))
    parser.add_argument('--text_context_len', type=int, default=2)
    parser.add_argument('--audio_context_len', type=int, default=1)
    parser.add_argument('--t_in_dim',type=int,default=1024,help='text input dim')
    parser.add_argument('--a_in_dim',type=int,default=1024,help='audio input dim')
    parser.add_argument('--v_in-dim',type=int,default=29,help='visual input dim')
    parser.add_argument('--proj_dim',type=int,default=1024,help='project dim')
    parser.add_argument('--proj_heads',type=int,default=8,help='multiattention heads')
    parser.add_argument('--token_length',type=int,default=1,help='')
    parser.add_argument('--proj_depth',type=int,default=1,help='')
    parser.add_argument('--proj_mlp_dim',type=int,default=1024,help='')
    parser.add_argument('--a_input_length',type=int,default=299,help='')
    parser.add_argument('--v_input_length',type=int,default=96,help='')
    parser.add_argument('--l_input_length',type=int,default=96,help='')
    parser.add_argument('--fusion_layer_depth',type=int,default=2,help='')
    parser.add_argument('--net_layers',type=int,default=3,help='LSTM layer num')
    args = parser.parse_args()  
    main(args)





