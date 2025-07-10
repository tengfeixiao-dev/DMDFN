import torch
from torch import nn
from transformers import RobertaModel,DebertaV2Model, HubertModel, AutoModel, Data2VecAudioModel,BertModel
from utils.cross_attn_encoder import CMELayer, BertConfig
from sklearn.metrics import mutual_info_score
from .almt_layer import Transformer,CrossTransformer
from .encoders import MMILB
from .reconstruct import Reconstruct

def difference_loss(shared_feat: torch.Tensor, specific_feat: torch.Tensor) -> torch.Tensor:
    """
    Computes the difference loss between shared and specific features.

    Args:
        shared_feat (Tensor): Shared modality features of shape (batch_size, feature_dim)
        specific_feat (Tensor): Modality-specific features of shape (batch_size, feature_dim)

    Returns:
        Tensor: Scalar loss value
    """
    # Normalize by Frobenius norm
    shared_norm = torch.norm(shared_feat, p='fro') + 1e-8
    specific_norm = torch.norm(specific_feat, p='fro') + 1e-8

    shared_proj = (shared_feat.T @ shared_feat) / (shared_norm ** 2)
    specific_proj = (specific_feat.T @ specific_feat) / (specific_norm ** 2)

    loss = torch.norm(shared_proj - specific_proj, p='fro') ** 2
    return loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RobD2vContext(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        # self.roberta_model = BertModel.from_pretrained('bert-base-uncased')
        # self.roberta_model = DebertaV2Model.from_pretrained("/home/xiaotengfei/MMML-main/Deberta_v3")
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-large-960h")

        self.proj_l = nn.Sequential(
            nn.Linear(config.t_in_dim, config.proj_dim),
            nn.ReLU(),
            # Transformer(num_frames=config.l_input_length, save_hidden=False, token_len=None, dim=config.proj_dim, depth=config.proj_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(config.a_in_dim, config.proj_dim),
            nn.ReLU(),
            Transformer(num_frames=config.a_input_length, save_hidden=False, token_len=1, dim=config.proj_dim, depth=config.proj_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(config.v_in_dim, config.proj_dim),
            nn.ReLU(),
            Transformer(num_frames=config.v_input_length, save_hidden=False, token_len=1, dim=config.proj_dim, depth=config.proj_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
        )
        self.l_encoder=Transformer(num_frames=config.l_input_length, save_hidden=True, token_len=None, dim=config.proj_dim, depth=3, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
        self.la_fusion_layer=nn.ModuleList([])
        self.lv_fusion_layer=nn.ModuleList([])
        for i in range(3):
            self.la_fusion_layer.append(
                CrossTransformer(source_num_frames=config.a_input_length+1, tgt_num_frames=config.l_input_length, dim=config.proj_dim, depth=config.fusion_layer_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
            )
            self.la_fusion_layer.append(nn.LayerNorm(config.proj_dim))
            self.la_fusion_layer.append(Transformer(num_frames=config.l_input_length, save_hidden=False, token_len=None, dim=config.proj_dim, depth=config.proj_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim))
            self.lv_fusion_layer.append(
                CrossTransformer(source_num_frames=config.l_input_length, tgt_num_frames=config.v_input_length+1, dim=config.proj_dim, depth=config.fusion_layer_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim)
            )
            self.lv_fusion_layer.append(nn.LayerNorm(config.proj_dim))
            self.lv_fusion_layer.append(Transformer(num_frames=config.l_input_length, save_hidden=False, token_len=None, dim=config.proj_dim, depth=config.proj_depth, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim))

        self.v_net=nn.Sequential(Transformer(num_frames=config.v_input_length+1, save_hidden=False, token_len=None, dim=config.proj_dim, depth=2, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim))
        self.a_net=nn.Sequential(Transformer(num_frames=config.a_input_length+1, save_hidden=False, token_len=None, dim=config.proj_dim, depth=2, heads=config.proj_heads, mlp_dim=config.proj_mlp_dim))
        
        
        # self.v_net=nn.Sequential(nn.LSTM(input_size=config.proj_dim,hidden_size=config.proj_dim,num_layers=config.net_layers,dropout=0.2,batch_first=True))
        # self.a_net=nn.Sequential(nn.LSTM(input_size=config.proj_dim,hidden_size=config.proj_dim,num_layers=config.net_layers,dropout=0.2,batch_first=True))
        
        # #____________________________________________________________#
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024, 1)
        )
        # self.MI_la=MMILB(config.proj_dim,config.proj_dim)
        # self.MI_lv=MMILB(config.proj_dim,config.proj_dim)

        self.M_output_layers= nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024,1)
        )
        self.reconstructor=Reconstruct(config)
        self.v_s_param=nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.v_c_param=nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.a_s_param=nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.a_c_param=nn.Parameter(torch.tensor([1.0]), requires_grad=True)
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask,
                audio_context_inputs, audio_context_mask, visual_inputs, visual_mask,return_features=False):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state

        # # text context feature extraction
        # raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        # T_context_hidden_states = raw_output_context.last_hidden_state

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state

        # # audio context feature extraction
        # audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        # A_context_hidden_states = audio_context_out.last_hidden_state
        
        T_output_raw=self.proj_l(T_hidden_states)
        l_list=self.l_encoder(T_output_raw)
         
        A_output_raw=self.proj_a(A_hidden_states)
        V_output_raw=self.proj_v(visual_inputs)
        for i in range(0,6,3):
            A_common_feature=self.la_fusion_layer[i](A_output_raw,l_list[i//3])
            A_common_feature=self.la_fusion_layer[i+1](A_common_feature)
            A_common_feature=self.la_fusion_layer[i+2](A_common_feature)

        for i in range(0,6,3):
            V_common_feature=self.lv_fusion_layer[i](V_output_raw,l_list[i//3])
            V_common_feature=self.lv_fusion_layer[i+1](V_common_feature)
            V_common_feature=self.lv_fusion_layer[i+2](V_common_feature)

        A_special_feature=self.a_net(A_output_raw)
        V_special_feature=self.v_net(V_output_raw)
        A_dl=difference_loss(A_common_feature[:,0],A_special_feature[:,0])
        V_dl=difference_loss(V_common_feature[:,0],V_special_feature[:,0])
        
        # lld_lv=self.MI_lv(fused_hidden_states,V_output[:,0])
        # lld_la=self.MI_la(fused_hidden_states,A_output[:,0])
        T_output=self.T_output_layers(l_list[-1][:,0])
        V_output=self.V_output_layers(V_special_feature[:,0])
        A_output=self.A_output_layers(A_special_feature[:,0])

        V_feature=self.v_s_param*V_special_feature[:,0]+self.v_c_param*V_common_feature[:,0]
        A_feature=self.a_s_param*A_special_feature[:,0]+self.a_c_param*A_common_feature[:,0]
        # V_feature = V_special_feature[:,0]+V_common_feature[:,0]
        # A_feature = A_special_feature[:,0]+A_common_feature[:,0]
        last_feature=torch.cat((l_list[-1][:,0],V_feature,A_feature),dim=1)
        fused_output=self.M_output_layers(last_feature)

        recon_v_loss,recon_a_loss=self.reconstructor(V_feature,V_output_raw[:,0],A_feature,A_output_raw[:,0])
        # return {
        #     'T': T_output,
        #     'A': A_output,
        #     'M': fused_output,
        #     'V': V_output,
        # },lld_lv,lld_la
        if return_features==False:
            return {
                'T': T_output,
                'A': A_output,
                'M': fused_output,
                'V': V_output,
            },A_dl,V_dl,recon_v_loss,recon_a_loss
        else:
            return {
                'A_special_feature':A_special_feature[:,0],
                'A_common_feature':A_common_feature[:,0],
                'V_common_feature':V_common_feature[:,0],
                'V_special_feature':V_special_feature[:,0],
                'T_feature':l_list[-1][:,0]
            }






       