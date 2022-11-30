import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics as tm
from performer_pytorch import Performer
from performer_pytorch.performer_pytorch import FixedPositionalEmbedding

class TranscriptSeqRiboEmb(pl.LightningModule):
    def __init__(self, x_seq, x_ribo, num_tokens, lr, decay_rate, warmup_steps, max_seq_len, dim, 
                 depth, heads, dim_head, causal, nb_features, feature_redraw_interval,
                 generalized_attention, kernel_fn, reversible, ff_chunks, use_scalenorm,
                 use_rezero, tie_embed, ff_glu, emb_dropout, ff_dropout, attn_dropout,
                 local_attn_heads, local_window_size):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, 
                                 causal=causal, nb_features=nb_features, 
                                 feature_redraw_interval=feature_redraw_interval,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                 reversible=reversible, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm,
                                 use_rezero=use_rezero, ff_glu=ff_glu, ff_dropout=ff_dropout, attn_dropout=attn_dropout, 
                                 local_attn_heads=local_attn_heads, local_window_size=local_window_size)

        self.val_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.val_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.test_rocauc = tm.AUROC(pos_label=1, compute_on_step=False)
        self.test_prauc = tm.AveragePrecision(pos_label=1, compute_on_step=False)
        
        self.ff_1 = torch.nn.Linear(dim,dim*2)
        self.ff_2 = torch.nn.Linear(dim*2,2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(emb_dropout)

        if x_ribo:
            self.ff_emb_1 = torch.nn.Linear(1,dim)
            self.ff_emb_2 = torch.nn.Linear(dim, 6*dim)
            self.ff_emb_3 = torch.nn.Linear(6*dim,dim)
            self.tanh = torch.nn.Tanh()
            self.scalar_emb = torch.nn.Sequential(self.ff_emb_1, self.relu, self.ff_emb_2, self.relu, self.ff_emb_3, self.tanh)
            self.ribo_count_emb = torch.nn.Embedding(1, dim)
            self.ribo_read_emb = torch.nn.Embedding(21, dim)
        
        if x_seq:
            self.nuc_emb = torch.nn.Embedding(num_tokens, dim)

        self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
        self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        
    def on_load_checkpoint(self, checkpoint):
        if 'mlm' in checkpoint.keys() and checkpoint['mlm']:
            state_dict = checkpoint['state_dict']
            for key in ['ff_2.weight', 'ff_2.bias', 'ff_1.weight', 'ff_1.bias']:
                state_dict.pop(key)
            checkpoint['state_dict'] = state_dict
            checkpoint['mlm'] = False
            
        
    def parse_embeddings(self, batch):
        xs = []
        if 'ribo' in batch.keys():
            inp = batch['ribo']
            # if offsets
            if inp.shape[-1] == 1:
                xs.append(self.scalar_emb(inp)*self.ribo_count_emb.weight)
            # if no offsets
            else:
                counts = inp.sum(dim=-1).unsqueeze(-1) # bs, l, 1x
                xs.append(self.scalar_emb(counts)*self.ribo_count_emb.weight)
                # read fraction per position
                x = torch.nan_to_num(torch.div(inp, inp.sum(axis=-1).unsqueeze(-1)))
                # linear combination between read length fraction and read length embedding
                xs.append(torch.einsum('ikj,jl->ikl', [x, self.ribo_read_emb.weight]))
            
        if 'seq' in batch.keys():
            xs.append(self.nuc_emb(batch['seq']))
            
        x_emb = torch.sum(torch.stack(xs), dim=0)
        
        return x_emb
            
    def forward(self, batch, y_mask):
        x_mask = torch.clone(y_mask)
        x_mask[:,0] = 1
        x_mask[torch.arange(x_mask.shape[0]), x_mask.sum(dim=1)] = 1
        
        x = self.parse_embeddings(batch)
        x += self.pos_emb(x)
        x = self.dropout(x)
        
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb = layer_pos_emb, mask=x_mask)
        
        x = x[torch.logical_and(x_mask, y_mask)]
        x = x.view(-1, self.hparams.dim)
        
        x = F.relu(self.ff_1(x))
        x = self.ff_2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        y_mask = batch['y'] != -1
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        loss = F.cross_entropy(y_hat, y_true)
        self.log('train_loss', loss, batch_size=y_mask.sum())

        return loss
        
    def validation_step(self, batch, batch_idx):
        y_mask = batch['y'] != -1
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.val_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.val_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        
        self.log('val_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('val_prauc', self.val_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('val_rocauc', self.val_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
                
    def test_step(self, batch, batch_idx, ):
        y_mask = batch['y'] != -1
        y_true = batch['y'][y_mask].view(-1)

        y_hat = self(batch, y_mask)
        
        self.test_prauc(F.softmax(y_hat, dim=1)[:,1], y_true)
        self.test_rocauc(F.softmax(y_hat, dim=1)[:,1], y_true)

        self.log('test_loss', F.cross_entropy(y_hat, y_true), batch_size=y_mask.sum())
        self.log('test_prauc', self.test_prauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        self.log('test_rocauc', self.test_rocauc, on_step=False, on_epoch=True, batch_size=y_mask.sum())
        
        splits = torch.cumsum(y_mask.sum(dim=1),0, dtype=torch.long).cpu()
        y_hat_grouped = torch.tensor_split(F.softmax(y_hat, dim=1)[:,1], splits)[:-1]
        y_true_grouped = torch.tensor_split(batch['y'][y_mask], splits)[:-1]
        #x_grouped = torch.tensor_split(batch['x'][y_mask], lens)
        
        return y_hat_grouped, y_true_grouped, batch['x_id']
    
    def on_test_epoch_start(self):
        self.test_outputs = []
        self.test_targets = []
        self.labels = []
        
    def test_step_end(self, results):
        # this out is now the full size of the batch
        self.test_outputs = self.test_outputs + list(results[0])
        self.test_targets = self.test_targets + list(results[1])
        self.labels = self.labels + list(results[2])
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambda1 = lambda epoch: self.hparams.decay_rate
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)

        return [optimizer], [scheduler]
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        
class TranscriptMLM(pl.LightningModule):
    def __init__(self, mask_frac, rand_frac, lr,  decay_rate, warmup_steps, num_tokens, max_seq_len, dim, 
                 depth, heads, dim_head, causal, nb_features, feature_redraw_interval,
                 generalized_attention, kernel_fn, reversible, ff_chunks, use_scalenorm,
                 use_rezero, tie_embed, ff_glu, emb_dropout, ff_dropout, attn_dropout,
                 local_attn_heads, local_window_size):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, 
                                 causal=causal, nb_features=nb_features, 
                                 feature_redraw_interval=feature_redraw_interval,
                                 generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                 reversible=reversible, ff_chunks=ff_chunks, use_scalenorm=use_scalenorm,
                                 use_rezero=use_rezero, ff_glu=ff_glu,
                                 ff_dropout=ff_dropout, attn_dropout=attn_dropout, 
                                 local_attn_heads=local_attn_heads, local_window_size=local_window_size)
        
        self.hparams.mask_c = mask_frac
        self.hparams.mask_m = self.hparams.mask_c + (1 - self.hparams.mask_c)*(1-rand_frac)
        
        self.nuc_emb = torch.nn.Embedding(num_tokens, dim)
        self.dropout = torch.nn.Dropout(emb_dropout)
        
        self.ff_1 = torch.nn.Linear(dim,dim*2)
        self.ff_2 = torch.nn.Linear(dim*2,7)

        self.pos_emb = FixedPositionalEmbedding(dim, max_seq_len)
        self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)

        self.mask_token = 4

        self.train_acc = tm.Accuracy()
        self.val_acc = tm.Accuracy()
        self.test_acc = tm.Accuracy()

    def parse_embeddings(self, batch):
        #TODO: implement randomization for non-seq inputs
        xs = []
        if 'ribo' in batch.keys():
            inp = batch['ribo']
            # if offsets
            if len(inp.shape) == 3:
                xs.append(self.scalar_emb(inp)*self.ribo_count_emb.weight)
            # if no offsets
            else:
                counts = inp.sum(dim=-1).unsqueeze(-1) # bs, l, 1x
                xs.append(self.scalar_emb(counts)*self.ribo_count_emb.weight)
                # read fraction per position
                x = torch.nan_to_num(torch.div(inp, inp.sum(axis=-1).unsqueeze(-1)))
                # linear combination between read length fraction and read length embedding
                xs.append(torch.einsum('ikj,jl->ikl', [x, self.ribo_read_emb.weight]))
            
        if 'seq' in batch.keys():
            x_mask = batch['seq'] != 7
            xs.append(self.nuc_emb(batch['seq']))
            
        x_emb = torch.sum(torch.stack(xs), dim=0)
        
        return x_emb, x_mask

    def rand_seq(self, x, x_mask, dist, val=False):
        # self supervised learning protocol
        mask = torch.logical_and(dist > self.hparams.mask_c, dist < self.hparams.mask_m)
        x[mask] = self.nuc_emb.weight[torch.full((mask.sum(),), self.mask_token)]
        # apply random tokens to input
        if not val:
            mask = dist > self.hparams.mask_m
            idx_vec = torch.randint(0,self.mask_token,(mask.sum(),), device=self.device)
            #torch.einsum('ikj,jl->ikl', [idx_vec, self.nuc_emb.weight])*
            x[mask] = self.nuc_emb.weight[idx_vec]
        return x            

    def forward(self, x, x_mask, val=False):
        dist = torch.empty(x.shape[:-1], device=self.device).uniform_(0,1,)
        x = self.rand_seq(x, x_mask, dist, val=val)
        out_mask = torch.logical_and(dist > self.hparams.mask_c, x_mask)

        x += self.pos_emb(x)
        x = self.dropout(x)
        
        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb = layer_pos_emb, mask=x_mask)

        x = x[out_mask].view(-1, self.hparams.dim)
        x = F.relu(self.ff_1(x))
        x = self.ff_2(x)

        return x, out_mask

    def training_step(self, batch, batch_idx):
        x, x_mask = self.parse_embeddings(batch)
        x_hat, out_mask = self(x, x_mask, val=False)
        loss = F.cross_entropy(x_hat, batch['seq'][out_mask])

        self.log('train_loss', loss, batch_size=len(x_hat))
        self.train_acc(F.softmax(x_hat, dim=1), batch['seq'][out_mask])
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=len(x_hat))

        return loss
        
    def validation_step(self, batch, batch_idx):
        x, x_mask = self.parse_embeddings(batch)
        x_hat, out_mask = self(x, x_mask, val=True)
        loss = F.cross_entropy(x_hat, batch['seq'][out_mask])

        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=len(x_hat))
        self.val_acc(F.softmax(x_hat, dim=1), batch['seq'][out_mask])
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, batch_size=len(x_hat))
                
    def test_step(self, batch, batch_idx):
        x, x_mask = self.parse_embeddings(batch)
        x_hat, out_mask = self(x, x_mask, val=True)
        loss = F.cross_entropy(x_hat, batch['seq'][out_mask])
        
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=len(x_hat))
        self.test_acc(F.softmax(x_hat, dim=1), batch['seq'][out_mask])
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=len(x_hat))

    def on_save_checkpoint(self, checkpoint):
        checkpoint['mlm'] = True 
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lambda1 = lambda epoch: self.hparams.decay_rate
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda1)

        return [optimizer], [scheduler]
        
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # warm up lr
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        # update params
        optimizer.step(closure=optimizer_closure)