import math
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    """
    x: (B, N, D)
        N = num of features
        D = dim of model
    Returns: (B, N, D) where D = out_dim
    """
    def __init__(self,
            dim:int,
            num_heads:int,
            #p_drop: float = 0.0
        ):
        super().__init__()

        self.dim=dim #dim of input
        self.num_heads=num_heads
        
        self.d_k=dim//num_heads #D//h = Dimension of each head's key, query, and value

        self.W_q=nn.Linear(dim,dim) 
        self.W_k=nn.Linear(dim,dim)
        self.W_v=nn.Linear(dim,dim)

        self.W_y=nn.Linear(dim,dim) #output transformation

    def split_heads(self,x):
        batch,n,dim=x.size()
        x_split=x.reshape(batch,n,self.num_heads,self.d_k).transpose(-3,-2)
        return x_split

    def combine_heads(self,x_split):
        batch,num_heads,n,d_k=x_split.size()
        x_final = x_split.reshape(batch,n,self.dim)
        return x_final

    def scaled_dot_product_attention(self,q_split,k_split,v_split):
        #calculate attention weights
        # k.T flips all dims, including batch -> only need to transpose last two dims
        # k:(B,N,D) -> k.T:(D,N,B)
        # (B,N,D) @ (B,D,N) -> (B,N,N)

        k_split_T = k_split.transpose(-2,-1)
        attn_scores = torch.matmul(q_split,k_split_T) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores,dim=-1)
        y_split = torch.matmul(attn_probs,v_split)

        return y_split

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_split = self.split_heads(self.W_q(x))                     # (B, N, D) -> (B, h, N, D/h)
        k_split = self.split_heads(self.W_k(x))                     # (B, N, D) -> (B, h, N, D/h)
        v_split = self.split_heads(self.W_v(x))                     # (B, N, D) -> (B, h, N, D/h)

        y_split=self.scaled_dot_product_attention(q_split,k_split,v_split)
        y=self.W_y(self.combine_heads(y_split))

        return y
    

class FeedForward(nn.Module):
    def __init__(self, dim, dim_inner, p=0.1):
        #dim = dim of model input and output
        #dim_inner = dim of inner layer in FFNN
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim_inner),
            nn.ReLU(),
            nn.Dropout(p), # regularise hidden activations
            nn.Linear(dim_inner, dim),
            nn.Dropout(p), # regularise output of FFN
        )
    def forward(self, x): return self.layers(x)

class TransformerLayer(nn.Module):
    """
    Standard transformer block with:
    - Multi-head attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """

    def __init__(self,
            embed_dim:int =256,
            num_heads:int=16,
            ffn_dim_multiplier:float=2,
            p_drop: float = 0.2):
        
        super().__init__()

        dim_ff=embed_dim*ffn_dim_multiplier
        self.attn=MultiHeadAttention(dim=embed_dim,num_heads=num_heads)
        self.ff=FeedForward(dim=embed_dim,dim_inner=dim_ff)

        self.norm1=nn.LayerNorm(embed_dim)
        self.norm2=nn.LayerNorm(embed_dim)
        self.dropout=nn.Dropout(p_drop)

    def forward(self,x):
        attn_output=self.attn(x)
        drop=self.dropout(attn_output)

        #add & norm 1
        x=self.norm1(x+drop)

        # feed forward
        ff_output=self.ff(x)
        drop2=self.dropout(ff_output)

        #add & norm 2
        x=self.norm2(x+drop2)

        return x
    
class MultiLayerTransformerClassifier(nn.Module):
    """
    Multi-layer transformer classifier for point clouds.
    Input: (B, N, 3) point clouds
    Output: (B, num_classes) logits
    """
    def __init__(self,
            num_classes:int,
            embed_dim:int =256,
            num_heads:int=16,
            ffn_dim_multiplier:float=2,
            num_layers:int = 3,
            p_drop: float = 0.2):
        
        super().__init__()

        #embedding layer
        self.embed=nn.Sequential(
            nn.Linear(3,embed_dim),
            nn.ReLU(),
            nn.Dropout(p_drop)
        )

        #stack of transformer layers
        self.transformer_layers=nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim_multiplier=ffn_dim_multiplier,
                p_drop=p_drop) for i in range(num_layers)
        ]) 

        #final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        #final linear layer to reduce dim to num of classes for classification
        self.final = nn.Linear(2*embed_dim, num_classes) #mean AND max pooling

    def forward(self,x):
        #embed point coords
        x=self.embed(x) #(B,N,embed_dim)

        #apply transformer layers
        for layer in self.transformer_layers:
            x=layer(x)

        #final normalisation
        x=self.norm(x)

        #global pooling 
        x_mean = x.mean(dim=1) # global mean pooling (stable global signal)
        x_max  = x.amax(dim=1) # global max pooling  (captures strong local features)
        x = torch.cat([x_mean, x_max], dim=-1)
        return self.final(x)

