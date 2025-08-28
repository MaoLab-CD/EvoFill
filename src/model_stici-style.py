# model.py
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from utils import load_config


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, method='linear',
                 dropout_rate=0.0, start_offset=0, end_offset=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.method = method
        self.start_offset = start_offset
        self.end_offset = end_offset
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.GELU()
        )
        
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # device = x[0].device
        # for p in self.parameters():
        #     p.data = p.data.to(device)

        input_seq = x[0][:, self.start_offset:x[0].shape[1] - self.end_offset, :]
        context_seq = x[1]
        
        attn_output, _ = self.att(
            input_seq.transpose(0, 1),
            context_seq.transpose(0, 1),
            context_seq.transpose(0, 1),
            need_weights=False
        )
        attn_output = attn_output.transpose(0, 1)
        
        out1 = self.layernorm1(input_seq + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        
        return self.layernorm2(out1 + self.dropout(ffn_output))

class CrossAttentionLayer(nn.Module):
    def __init__(self, local_dim, global_dim, start_offset=0, end_offset=0,
                 activation=nn.GELU(), dropout_rate=0.0, n_heads=8):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.n_heads = n_heads
        self.layer_norm00 = nn.LayerNorm(local_dim)
        self.layer_norm01 = nn.LayerNorm(global_dim)
        self.layer_norm1 = nn.LayerNorm(local_dim)
        self.ffn = nn.Sequential(
            nn.Linear(local_dim, local_dim // 2),
            self.activation,
            nn.Linear(local_dim // 2, local_dim),
            self.activation
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(local_dim, n_heads, dropout=dropout_rate)

    def forward(self, inputs):
        local_repr = self.layer_norm00(inputs[0])
        global_repr = self.layer_norm01(inputs[1])
        query = local_repr[:, self.start_offset:local_repr.shape[1] - self.end_offset, :]

        # Generate cross-attention outputs: [batch_size, latent_dim, projection_dim].
        attention_output, _ = self.attention(
            query.transpose(0, 1),
            global_repr.transpose(0, 1),
            global_repr.transpose(0, 1)
        )
        # Skip connection 1.
        attention_output = attention_output.transpose(0, 1)
        attention_output = attention_output + query
        
        # Apply layer norm.
        attention_output = self.layer_norm1(attention_output)
        # Apply Feedforward network.
        outputs = self.ffn(attention_output)
        # Skip connection 2
        outputs = outputs + attention_output
        
        return outputs

class CatEmbeddings(nn.Module):
    def __init__(self, embedding_dim, embeddings_initializer='glorot_uniform',
                 embeddings_regularizer=None, activity_regularizer=None,
                 embeddings_constraint=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings_initializer = embeddings_initializer
        self.embeddings_regularizer = embeddings_regularizer
        self.activity_regularizer = activity_regularizer
        self.embeddings_constraint = embeddings_constraint

        self.num_of_allels = None
        self.n_snps = None
        self.position_embedding = None
        self.embedding = None
        self.positions = None

    def build(self, input_shape):
        self.num_of_allels = input_shape[-1]
        self.n_snps = input_shape[-2]
        self.position_embedding = nn.Embedding(
            num_embeddings=self.n_snps, embedding_dim=self.embedding_dim
        )
        self.embedding = nn.Parameter(torch.empty((self.num_of_allels, self.embedding_dim)))
        if self.embeddings_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.embedding)
        else:
            raise ValueError(f"Unsupported initializer: {self.embeddings_initializer}")
            
        self.positions = torch.arange(start=0, end=self.n_snps, step=1)

    def forward(self, inputs):
        if self.num_of_allels is None:
            self.build(inputs.shape)
        self.embedding.data = self.embedding.data.to(inputs.device)
        self.immediate_result = torch.einsum('ijk,kl->ijl', inputs, self.embedding)
        return self.immediate_result + self.position_embedding(self.positions).to(inputs.device)

class ConvBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.const = None
        
        self.conv000 = nn.Conv1d(embed_dim, embed_dim, 3, padding='same')
        self.conv010 = nn.Conv1d(embed_dim, embed_dim, 5, padding='same')
        self.conv011 = nn.Conv1d(embed_dim, embed_dim, 7, padding='same')
        
        self.conv020 = nn.Conv1d(embed_dim, embed_dim, 7, padding='same')
        self.conv021 = nn.Conv1d(embed_dim, embed_dim, 15, padding='same')
        
        self.conv100 = nn.Conv1d(embed_dim, embed_dim, 3, padding='same')
        self.bn0 = nn.BatchNorm1d(embed_dim, eps=0.001, momentum=0.01)
        self.dw_conv = nn.Conv1d(embed_dim, embed_dim, 1, padding='same')
        self.bn1 = nn.BatchNorm1d(embed_dim, eps=0.001, momentum=0.01) 
    def forward(self, inputs):
        # print('1:',inputs.shape)  # torch.Size([2, 64, 32])
        inputs = inputs.permute(0, 2, 1)
        xa = F.gelu(self.conv000(inputs))
        xb = F.gelu(self.conv010(xa))
        xb = F.gelu(self.conv011(xb))
        xc = F.gelu(self.conv020(xa))
        xc = F.gelu(self.conv021(xc))
        xa = xb + xc
        xa = F.gelu(self.conv100(xa))
        xa = self.bn0(xa)
        xa = self.dw_conv(xa)
        xa = self.bn1(xa)
        xa = F.gelu(xa)
        xa = xa.permute(0, 2, 1)
        return xa

def chunk_module(input_len, embed_dim, num_heads, start_offset=0, end_offset=0, dropout_rate=0.25):
    projection_dim = embed_dim
    
    class ChunkModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_block = TransformerBlock(projection_dim, num_heads, projection_dim // 2,
                                                    start_offset=start_offset, end_offset=end_offset, dropout_rate=0.0)
            self.conv1 = ConvBlock(projection_dim)
            self.conv_skip = ConvBlock(projection_dim)
            self.dense1 = nn.Linear(projection_dim, projection_dim)
            self.conv2 = ConvBlock(projection_dim)
            self.cross_attn = CrossAttentionLayer(projection_dim, projection_dim, dropout_rate=0.0)
            self.dropout = nn.Dropout(dropout_rate)
            self.conv3 = ConvBlock(projection_dim)
            
        def forward(self, xa):
            # print("x0:",xa.shape)
            xa0 = self.transformer_block([xa, xa])
            # print("x1:",xa0.shape)
            xa = self.conv1(xa0)
            xa_skip = self.conv_skip(xa)
            xa = F.gelu(self.dense1(xa))
            xa = self.conv2(xa)
            xa = self.cross_attn([xa, xa0])
            xa = self.dropout(xa)
            xa = self.conv3(xa)
            xa = torch.cat([xa_skip, xa], dim=-1)

            return xa
            
    return ChunkModule()

class STICI(nn.Module):
    def __init__(self, embed_dim, num_heads, offset_before=0, offset_after=0,
                 chunk_size=2048, activation=nn.GELU(), dropout_rate=0.25,
                 attention_range=64):
        super(STICI, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.attention_range = attention_range
        self.offset_before = offset_before
        self.offset_after = offset_after
        
        self.seq_len = None
        self.in_channel = 3
        self.chunk_starts = None
        self.chunk_ends = None
        self.mask_starts = None
        self.mask_ends = None
        self.chunkers = None

        self.embedding = CatEmbeddings(self.embed_dim)
        self.after_concat_layer = nn.Conv1d(
            in_channels=self.embed_dim * 2,
            out_channels=self.embed_dim // 2,
            kernel_size=5,
            padding='same'
        )
        self.last_conv = nn.Conv1d(
            in_channels=self.embed_dim // 2,
            out_channels=self.in_channel - 1,
            kernel_size=5,
            padding='same'
        )
        
    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.in_channel = input_shape[2]
        self.chunk_starts = list(range(0, input_shape[1], self.chunk_size))
        self.chunk_ends = []
        for cs in self.chunk_starts:
            self.chunk_ends.append(min(cs + self.chunk_size, input_shape[1]))
        self.mask_starts = [max(0, cs - self.attention_range) for cs in self.chunk_starts]
        self.mask_ends = [min(ce + self.attention_range, input_shape[1]) for ce in self.chunk_ends]
        
        device = next(self.parameters()).device
    
        self.chunkers = nn.ModuleList()
        for i, cs in enumerate(self.chunk_starts):
            chunker = chunk_module(
                self.mask_ends[i] - self.mask_starts[i],
                self.embed_dim, 
                self.num_heads,
                start_offset=cs - self.mask_starts[i],
                end_offset=self.mask_ends[i] - self.chunk_ends[i],
                dropout_rate=self.dropout_rate
            )
            chunker = chunker.to(device)
            self.chunkers.append(chunker)

        
    def forward(self, inputs):
        # print('x0:',inputs.shape)    # x0: torch.Size([batch, 160, 3])
        self.chunkers = self.chunkers
        if self.chunkers is None:
            self.build(inputs.shape)
            print('chunker')
        x = self.embedding(inputs)
        chunks = []
        for i, chunker in enumerate(self.chunkers):
            chunk_input = x[:, self.mask_starts[i]:self.mask_ends[i], :]  # (batch, chunk_seq_len, embed_dim)
            chunk_output = chunker(chunk_input)  # (batch, chunk_seq_len, embed_dim*2) 
            chunks.append(chunk_output)
        x = torch.cat(chunks, dim=1)
        # print('x1:', x.shape)  # x1: torch.Size([batch, 160, 64])
        x = x.permute(0, 2, 1)
        x = self.activation(self.after_concat_layer(x))
        # print('x1.5:',x.shape)  # x1.5: torch.Size([batch, 16, 160])
        x = self.last_conv(x)
        x = x.permute(0, 2, 1)
        x = F.softmax(x, dim=-1)
        # print('x2:', x.shape)  # x2: torch.Size([batch, 160, 2])
        x = x[:, self.offset_before:self.seq_len - self.offset_after]
        # print('x3:', x.shape)  # x3: torch.Size([batch, 128, 2])
        return x

class ImputationLoss(nn.Module):
    def __init__(self, use_r2_loss=True):
        super(ImputationLoss, self).__init__()
        self.use_r2_loss = use_r2_loss

    def calculate_Minimac_R2(self, pred_alt_allele_probs, gt_alt_af):
        # 创建掩码：gt_alt_af 为 0.0 或 1.0 的位置
        mask = (gt_alt_af == 0.0) | (gt_alt_af == 1.0)
        
        # 复制并替换 gt_alt_af：在掩码位置设为 0.5
        gt_alt_af_replaced = gt_alt_af.clone()
        gt_alt_af_replaced[mask] = 0.5
        
        # 计算分母并应用下限 0.01
        denom = gt_alt_af_replaced * (1 - gt_alt_af_replaced)
        denom = torch.clamp(denom, min=0.01)
        
        # 计算均方误差
        mse = torch.mean((pred_alt_allele_probs - gt_alt_af_replaced) ** 2, dim=0)
        
        # 计算 R2 值并将掩码位置置零
        r2 = mse / denom
        r2[mask] = 0.0
        
        return r2

    def forward(self, y_true, y_pred):
        # 确保数据类型一致
        y_true = y_true.to(y_pred.dtype)
        
        # 计算交叉熵损失 (CE) - 与 TF 的 SUM reduction 一致
        ce_loss = -torch.sum(y_true * torch.log(y_pred + 1e-8))
        
        # 计算 KL 散度损失 - 与 TF 的 SUM reduction 一致
        kl_loss = torch.sum(y_true * torch.log((y_true + 1e-8) / (y_pred + 1e-8)))
        
        # 总损失 = CE + KL
        total_loss = ce_loss + kl_loss
        
        # 如果启用 R2 损失
        if self.use_r2_loss:
            batch_size = y_true.size(0)
            group_size = 4  # 固定组大小，与原始 TF 代码一致
            num_full_groups = batch_size // group_size
            num_remainder_samples = batch_size % group_size
            
            r2_loss = 0.0
            
            # 处理完整的分组
            if num_full_groups > 0:
                # 重塑为 [num_groups, group_size, num_snps, 2]
                y_true_grouped = y_true[:num_full_groups * group_size].view(
                    num_full_groups, group_size, -1, y_true.size(-1))
                y_pred_grouped = y_pred[:num_full_groups * group_size].view(
                    num_full_groups, group_size, -1, y_pred.size(-1))
                
                for i in range(num_full_groups):
                    # 获取当前组的真实值和预测值
                    group_true = y_true_grouped[i]
                    group_pred = y_pred_grouped[i]
                    
                    # 计算等位基因频率 (沿样本维度求平均)
                    gt_alt_af = torch.mean(group_true[..., 1], dim=0)
                    
                    # 获取预测的 alt 等位基因概率
                    pred_alt_probs = group_pred[..., 1]
                    
                    # 计算 R2
                    r2 = self.calculate_Minimac_R2(pred_alt_probs, gt_alt_af)
                    
                    # 累加 R2 损失 (负值乘以组大小)
                    r2_loss -= torch.sum(r2) * group_size
            
            # 处理剩余样本
            if num_remainder_samples > 0:
                remainder_true = y_true[-num_remainder_samples:]
                remainder_pred = y_pred[-num_remainder_samples:]
                
                # 计算等位基因频率
                gt_alt_af = torch.mean(remainder_true[..., 1], dim=0)
                
                # 获取预测的 alt 等位基因概率
                pred_alt_probs = remainder_pred[..., 1]
                
                # 计算 R2
                r2 = self.calculate_Minimac_R2(pred_alt_probs, gt_alt_af)
                
                # 累加 R2 损失 (负值乘以剩余样本数)
                r2_loss -= torch.sum(r2) * num_remainder_samples
            
            # 将 R2 损失加到总损失中
            total_loss += r2_loss
        
        return total_loss

def create_data(B, N):
    # 用于生成随机数据（STCI）
    # 生成随机的 one-hot 编码的 x，维度为 (B, N, 3), target 维度为 (B, N, 2)
    random_values = torch.randint(0, 3, (B, N), device=device)
    x = torch.nn.functional.one_hot(random_values, num_classes=3).float()
    targets = torch.zeros(B, N, 2, device=device, dtype=torch.float32)
    
    mask_positions = (x[:, :, 2] == 1)
    
    # 复制非mask位置的值
    targets[:, :, 0] = x[:, :, 0]
    targets[:, :, 1] = x[:, :, 1]
    
    # 处理mask位置
    random_fill = torch.randint(0, 2, (B, N), device=device)
    b_indices, n_indices = torch.where(mask_positions)
    targets[b_indices, n_indices, 0] = (random_fill[b_indices, n_indices] == 0).float()
    targets[b_indices, n_indices, 1] = (random_fill[b_indices, n_indices] == 1).float()
    
    return x, targets

# ------------------------------------------------------------------
# 本地测试（读取 config.json）+ 模拟训练显存
# ------------------------------------------------------------------
if __name__ == "__main__":
    import torch.cuda as cuda
    from torch.amp import autocast

    # 1. 读取配置
    cfg = load_config("/home/qqyang/EvoFill/config/config.json")

    # 2. 使用配置中的参数
    offset_before = 0  # 数据块起始位置?
    offset_after = 0  # 数据块结束位置?(详见STICI_Vcopy.py)
    criterion = ImputationLoss(use_r2_loss=cfg.use_r2)

    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model = STICI(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.n_heads,
        chunk_size=cfg.cs,
        activation=torch.nn.GELU(),
        attention_range=cfg.co,
        offset_before=offset_before,    
        offset_after=offset_after   
    ).to(device)

    # 3. 构造随机数据（与 train.py 保持一致）
    B = cfg.batch_size
    N = cfg.n_snp
    K = cfg.k_mer
    x,targets = create_data(B,N)

    # 4. 构造 optimizer（Adam 会额外占 2×参数量）
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 5. 前向 + 反向一次
    model.train()
    optimizer.zero_grad()
    with autocast("cuda",enabled=False):          # 需要 fp32 就关掉 amp
        outputs = model(x)
        print(outputs.shape)
        print(targets.shape)
        loss = criterion(targets, outputs)

    loss.backward()
    optimizer.step()

    # 6. 打印信息
    print("Input  shape :", x.shape)        # (B, N, 3) + (B,B)
    print("Output shape :", outputs.shape)   # (B, N, 2)
    print("Loss :", loss.item())

    if device.type == "cuda":
        # 注意：nvidia-smi 看的是“已分配”，这里用 peak 更接近真实最大占用
        mb = cuda.max_memory_allocated(device) / 1024 ** 2
        print(f"Peak GPU memory allocated: {mb:.2f} MB")
        cuda.reset_peak_memory_stats(device)   # 清掉计数器，方便下次测试