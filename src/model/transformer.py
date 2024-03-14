import torch
import torch.nn as nn
from src.model.mlp import MLP
import torch.nn.functional as F
from src.model.weight_init import trunc_normal_


class SAMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., with_bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=with_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=with_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = Attention(
            dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SAMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, with_bias=True)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Build encoder model based on ViT
    """

    def __init__(self, transformer_cfg):
        super().__init__()

        embedding_size = transformer_cfg.SEQ_EMB.OUTPUT_SIZE
        self.blocks = nn.ModuleList([
            Block(
                dim=embedding_size, num_heads=transformer_cfg.NUM_HEADS,
                mlp_ratio=transformer_cfg.MLP_RATIO,
                drop=transformer_cfg.DROP_RATE, attn_drop=transformer_cfg.ATTN_DROP_RATE)
            for i in range(transformer_cfg.DEPTH)])

        self.norm = nn.LayerNorm(embedding_size, elementwise_affine=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, key_padding_mask=None):

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        x = self.norm(x)
        return x


class PhyloTreeModel(nn.Module):

    def __init__(self, gfn_cfg):
        super().__init__()

        transformer_cfg = gfn_cfg.MODEL.TRANSFORMER
        self.compute_state_flow = (gfn_cfg.LOSS_TYPE != 'TB')
        self.concatenate_summary_token = transformer_cfg.PART1_HEAD.CONCATENATE_SUMMARY_TOKEN
        self.concatenate_candidate_tree = transformer_cfg.PART2_HEAD.CONCATENATE_CANDIDATE_TREE
        self.condition_on_scale = gfn_cfg.CONDITION_ON_SCALE
        self.use_tree_type_embedding = gfn_cfg.MODEL.TRANSFORMER.USE_TREE_TYPE_EMBEDDING

        if self.condition_on_scale:
            scales_set = gfn_cfg.SCALES_SET
            self.scale_embeddings = torch.nn.Embedding(len(scales_set), transformer_cfg.EMBEDDING_SIZE)
        else:
            self.summary_token = nn.Parameter(torch.zeros(1, 1, transformer_cfg.EMBEDDING_SIZE), requires_grad=True)
            trunc_normal_(self.summary_token, std=0.1)

        if self.use_tree_type_embedding:
            self.tree_type_embeddings = nn.Parameter(torch.zeros(2, transformer_cfg.EMBEDDING_SIZE), requires_grad=True)
            trunc_normal_(self.tree_type_embeddings, std=0.1)
        # note: setting bias to false in the embedding layer
        self.seq_emb = nn.Linear(transformer_cfg.INPUT_SIZE, transformer_cfg.EMBEDDING_SIZE, bias=False)
        self.shared_encoder = transformer_cfg.SHARED_ENCODER
        if self.shared_encoder:
            self.encoder = TransformerEncoder(transformer_cfg)
        else:
            self.part1_encoder = TransformerEncoder(transformer_cfg)
            self.part2_encoder = TransformerEncoder(transformer_cfg)

        self.part1_logits_head = MLP(transformer_cfg.PART1_HEAD)
        self.part2_logits_head = MLP(transformer_cfg.PART2_HEAD)
        if self.compute_state_flow:
            self.part1_flow_head = MLP(transformer_cfg.PART1_HEAD,
                                       input_size=transformer_cfg.PART1_HEAD.INPUT_SIZE // 2)
            self.part2_flow_head = MLP(transformer_cfg.PART2_HEAD,
                                       input_size=transformer_cfg.PART2_HEAD.INPUT_SIZE // 2)

    def get_head_token(self, scale_key):
        if self.condition_on_scale:
            token = self.scale_embeddings(scale_key)
        else:
            token = self.summary_token
        return token

    def forward_part1(self, x, key_padding_mask=None):

        if self.shared_encoder:
            encoder = self.encoder
        else:
            encoder = self.part1_encoder

        x = encoder(x, key_padding_mask)
        summary_token = x[:, :1]
        x = x[:, 1:]
        flow_head_input = summary_token

        if self.concatenate_summary_token:
            _, num_trees, _ = x.shape
            summary_token = summary_token.expand(-1, num_trees, -1)
            x = torch.cat([x, summary_token], dim=2)

        logits = self.part1_logits_head(x).squeeze(-1)

        if self.compute_state_flow:
            log_state_flow = self.part1_flow_head(flow_head_input)
        else:
            log_state_flow = None
        return logits, log_state_flow

    def forward_part2(self, x, key_padding_mask=None):

        if self.shared_encoder:
            encoder = self.encoder
        else:
            encoder = self.part1_encoder

        x = encoder(x, key_padding_mask)
        flow_head_input = x[:, :1]
        candidate_token = x[:, 1:2]
        x = x[:, 2:]

        if self.concatenate_candidate_tree:
            _, num_trees, _ = x.shape
            candidate_token = candidate_token.expand(-1, num_trees, -1)
            x = torch.cat([candidate_token, x], dim=2)

        logits = self.part2_logits_head(x).squeeze(-1)
        if self.compute_state_flow:
            log_state_flow = self.part2_flow_head(flow_head_input)
        else:
            log_state_flow = None

        return logits, log_state_flow

    def model_params(self):
        return list(self.parameters())

    def forward(self, **kwargs):
        """
        :param batch_input: input tensors of shape [batch_size, nb_seq, seq_len], each sample in the batch is a state
        :param batch_intermediate_flag: boolean to tell if a state is intermediate
        :param batch_nb_seq: list of actual sequence length for each sample the batch
        """
        batch_input = kwargs.get('batch_input')
        batch_intermediate_flag = kwargs.get('batch_intermediate_flag')
        batch_nb_seq = kwargs.get('batch_nb_seq')
        scale_key = kwargs.get('scale_key')

        batch_size, max_nb_seq, _ = batch_input.shape

        # batch_size, max_nb_seq, emb_size
        x = self.seq_emb(batch_input)

        # add tree type embedding
        if self.use_tree_type_embedding:
            x += self.tree_type_embeddings[0]
            if torch.any(batch_intermediate_flag):
                x[batch_intermediate_flag, 0] = x[batch_intermediate_flag, 0] - self.tree_type_embeddings[0] + \
                                                self.tree_type_embeddings[1]

        # add summary token
        B = x.shape[0]
        summary_token = self.get_head_token(scale_key)
        if self.condition_on_scale:
            traj_length = int(B / kwargs['batch_size'])
            summary_token = summary_token.unsqueeze(1).expand(-1, traj_length, -1).reshape(batch_size, 1, -1)
        else:
            summary_token = summary_token.expand(B, -1, -1)
        x = torch.cat((summary_token, x), dim=1)

        # padding mask
        batch_padding_mask = torch.ones((batch_size, max_nb_seq)).to(x).cumsum(dim=1) > batch_nb_seq[:, None]
        batch_padding_mask = batch_padding_mask.bool()

        all_logits_output = torch.zeros(batch_size, max_nb_seq).to(x)
        if self.compute_state_flow:
            all_log_flow_outputs = torch.zeros(batch_size).to(x)
        else:
            all_log_flow_outputs = None

        if torch.any(batch_intermediate_flag):
            # in intermediate state, the number of action is the number of sequences/subtrees minus 1
            # the minus one is for the candidate tree
            current_padding_mask = batch_padding_mask[batch_intermediate_flag]
            current_padding_mask = F.pad(current_padding_mask, (1, 0), "constant", False)
            part2_logits, part2_log_flows = self.forward_part2(x[batch_intermediate_flag], current_padding_mask)
            all_logits_output[batch_intermediate_flag, :max_nb_seq - 1] = part2_logits
            if self.compute_state_flow:
                all_log_flow_outputs[batch_intermediate_flag] = part2_log_flows

            # update the padding mask, because the candidate subtree has been removed from the logits
            batch_padding_mask[batch_intermediate_flag, batch_nb_seq[batch_intermediate_flag] - 1] = True

        if torch.any(~batch_intermediate_flag):
            # there may be a summary node
            current_padding_mask = batch_padding_mask[~batch_intermediate_flag]
            current_padding_mask = F.pad(current_padding_mask, (1, 0), "constant", False)
            part1_logits, part1_log_flows = self.forward_part1(x[~batch_intermediate_flag], current_padding_mask)
            all_logits_output[~batch_intermediate_flag] = part1_logits
            if self.compute_state_flow:
                all_log_flow_outputs[~batch_intermediate_flag] = part1_log_flows

        # logits in the shape of [batch_size, max_nb_seq]
        # note: logits contain paddings
        if self.compute_state_flow:
            return all_logits_output, all_log_flow_outputs, batch_padding_mask
        else:
            return all_logits_output, batch_padding_mask
