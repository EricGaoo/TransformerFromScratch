from torch import nn, Tensor

#act shape: torch.Size([2, 15, 26])
#tgt shape: torch.Size([2, 16, 256, 256])
#self.future_action_token: torch.Size([1, 26])
#future_action_tokens: torch.Size([2, 1, 26])
#pos_embed_act: torch.Size([1, 16, 26])
#d_action = 26 
#d_model = 256

self.action_encoder = nn.Sequential(
    nn.Linear(d_action, d_model),
    nn.ReLU(),
    Mlp(d_model),
    nn.LayerNorm(d_model, eps=1e-05)
)

def forward(self, tgt, act, pos_embed_act):
    x = tgt #video tokens
    
    B = tgt.size(0)
    future_action_tokens = torch.stack([self.future_action_token]*B).to(act.device)
    act = torch.concatenate( #action tokens
            (act, future_action_tokens), dim=1)
    act += pos_embed_act
    act = self.action_encoder(act)



pos_embed_TA = torch.zeros(1, 16, 26)
future_act_token = torch.zeros(1, 26)
future_act_tokens = torch.stack(future_act_token*2)
act = torch.concatenate((act, future_act_tokens), dim=1))