import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2). \
            contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.norm = nn.LayerNorm(hidden_size)  # 可选
        self.beta = nn.Parameter(torch.ones(1, hidden_size))  # 可学习权重
        if dataset == 'MELD':
            self.fc[0].weight.data.copy_(torch.eye(hidden_size))
            self.fc[0].weight.requires_grad = False
            self.fc[2].weight.data.copy_(torch.eye(hidden_size))
            self.fc[2].weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.norm(self.fc(a)))  # 归一化门控
        final_rep = (self.beta * z) * a  # 学习缩放权重
        return final_rep


class PreserveDifference(nn.Module):
    def __init__(self, hidden_size):
        super(PreserveDifference, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        """
        :param a: (batch_size, len, dim)
        :param b: (batch_size, len, dim)
        :param c: (batch_size, len, dim)
        :return: (batch_size, len, dim)
        """
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        # 计算最大值的索引 (batch_size, len, dim)，索引是 modal_size 维度上的最大值索引
        max_indices = torch.argmax(utters_fc, dim=-2, keepdim=True)  # (batch_size, len, 1, dim)

        # 使用 gather 选择最大值对应的模态
        final_rep = torch.gather(utters, dim=-2, index=max_indices).squeeze(-2)  # (batch_size, len, dim)

        return final_rep


class MultiModalGatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(MultiModalGatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        """
        :param a: (batch_size, len, dim)
        :param b: (batch_size, len, dim)
        :param c: (batch_size, len, dim)
        :return: (batch_size, len, dim)
        """
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)

        return final_rep




class UniModalClassificier(nn.Module):
    def __init__(self, hidden_dim=512, dropout=0.5, num_classes=7):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class FeatureDecoupler(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super().__init__()
        self.common_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.private_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # self.decoder = nn.Linear(hidden_dim*2, input_dim)

    def forward(self, x):
        common = self.common_encoder(x)
        private = self.private_encoder(x)
        # x_cat = torch.cat([common, private], dim=-1)
        # recon = self.decoder(x_cat)
        # recon_privarte = self.private_encoder(recon)
        # return common, private, recon, recon_privarte
        return common, private


class SeekCommonality(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout):
        super(SeekCommonality, self).__init__()
        self.a2t = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)
        self.v2t = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)

        self.t2a = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)
        self.v2a = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)

        self.t2v = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)
        self.a2v = TransformerEncoder(hidden_dim, hidden_dim, n_head, 1, dropout)

        self.t_p = nn.Linear(3 * hidden_dim, hidden_dim)
        self.a_p = nn.Linear(3 * hidden_dim, hidden_dim)
        self.v_p = nn.Linear(3 * hidden_dim, hidden_dim)

        self.m_gate = MultiModalGatedFusion(hidden_dim)


    def forward(self, t, a, v, u_mask, spk_embeddings):
        a2t_out = self.t2a(t, a, u_mask, spk_embeddings)
        v2t_out = self.t2v(t, v, u_mask, spk_embeddings)

        t2a_out = self.a2t(a, t, u_mask, spk_embeddings)
        v2a_out = self.a2v(a, v, u_mask, spk_embeddings)

        t2v_out = self.v2t(v, t, u_mask, spk_embeddings)
        a2v_out = self.v2a(v, a, u_mask, spk_embeddings)

        t_out = self.t_p(torch.cat([t, a2t_out, v2t_out], dim=-1))
        a_out = self.a_p(torch.cat([a, t2a_out, v2a_out], dim=-1))
        v_out = self.v_p(torch.cat([v, t2v_out, a2v_out], dim=-1))

        commonality = self.m_gate(t_out, a_out, v_out)

        return commonality




class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text, D_visual, D_audio, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers + 1, hidden_dim, padding_idx)
        self.speaker_decoupler_embeddings = nn.Embedding(n_speakers + 1, int(hidden_dim/2), padding_idx)

        # Temporal convolutional layers
        self.textf_input = nn.Conv1d(D_text, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.acouf_input = nn.Conv1d(D_audio, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.visuf_input = nn.Conv1d(D_visual, hidden_dim, kernel_size=1, padding=0, bias=False)

        # Self-Transformers
        self.t_t = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.a_a = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.v_v = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)


        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.a_a_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.v_v_gate = Unimodal_GatedFusion(hidden_dim, dataset)


        # Disentanglement
        self.de_text = FeatureDecoupler(hidden_dim, int(hidden_dim/2))
        self.de_audio = FeatureDecoupler(hidden_dim, int(hidden_dim/2))
        self.de_video = FeatureDecoupler(hidden_dim, int(hidden_dim/2))

        # Emotion Classifier
        self.t_classifier = UniModalClassificier(int(hidden_dim/2), dropout, n_classes)
        self.a_classifier = UniModalClassificier(int(hidden_dim/2), dropout, n_classes)
        self.v_classifier = UniModalClassificier(int(hidden_dim/2), dropout, n_classes)

        # Seek Commomality
        self.seek_commonality = SeekCommonality(int(hidden_dim/2), n_head, dropout)

        # Preserve Difference
        self.preserve_difference = PreserveDifference(int(hidden_dim/2))

        self.common_classifier = nn.Linear(int(hidden_dim/2), n_classes)
        self.final_classifier = nn.Linear(hidden_dim, n_classes)



    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)
        spk_dec_embeddings = self.speaker_decoupler_embeddings(spk_idx)

        # Temporal convolutional layers
        textf = self.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
        acouf = self.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
        visuf = self.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)

        # Self-Transformers
        t_t_out = self.t_t(textf, textf, u_mask, spk_embeddings)
        a_a_out = self.a_a(acouf, acouf, u_mask, spk_embeddings)
        v_v_out = self.v_v(visuf, visuf, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t_t_out = self.t_t_gate(t_t_out)
        a_a_out = self.a_a_gate(a_a_out)
        v_v_out = self.v_v_gate(v_v_out)

        # Disentanglement
        # c_t, p_t, r_t, rp_t = self.de_text(t_t_out)
        # c_a, p_a, r_a, rp_a = self.de_audio(a_a_out)
        # c_v, p_v, r_v, rp_v = self.de_video(v_v_out)
        c_t, p_t = self.de_text(t_t_out)
        c_a, p_a = self.de_audio(a_a_out)
        c_v, p_v = self.de_video(v_v_out)

        # Seek Commomality
        common_feat = self.seek_commonality(c_t, c_a, c_v, u_mask, spk_dec_embeddings)

        # Preserve Difference
        selected_feat = self.preserve_difference(p_t, p_a, p_v)


        # Emotion Classifier
        t_logits = self.t_classifier(p_t)
        a_logits = self.a_classifier(p_a)
        v_logits = self.v_classifier(p_v)
        # common_logits = self.common_classifier(common_feat)
        final_logits = self.final_classifier(torch.cat([common_feat, selected_feat], dim=-1))

        # return t_logits, a_logits, v_logits, common_logits, final_logits, t_t_out, c_t, p_t, r_t, rp_t, a_a_out, c_a, p_a, r_a, rp_a, v_v_out, c_v, p_v, r_v, rp_v
        return t_logits, a_logits, v_logits, final_logits, t_t_out, c_t, p_t, a_a_out, c_a, p_a, v_v_out, c_v, p_v


