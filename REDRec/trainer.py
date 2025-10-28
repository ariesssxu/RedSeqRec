# Copyright (c) 2024 westlake-repl
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate
# Copyright (c) 2025 Xiaohongshu Technology Co. Ltd.
# SPDX-License-Identifier: MIT
#
# Original file was released under MIT, with the full license text
# available at https://choosealicense.com/licenses/mit/.
#
# This modified file is released under the same license.

import os
import sys
from logging import getLogger
from time import time
import time as t
import torch
import torch.optim as optim
from tqdm import tqdm
import deepspeed

from REDRec.utils import ensure_dir, create_tensorboard, set_color
from REDRec.utils.lr_scheduler import *

import lightning as L
from lightning.fabric.strategies import DeepSpeedStrategy, DDPStrategy

class Trainer(object):
    def __init__(self, config, model):
        super(Trainer, self).__init__()
        self.config = config
        self.model = model
        self.logger = getLogger()

        # distributed 
        self.gpu_available = torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rank = torch.distributed.get_rank()
        
        # optimizer
        self.optim_args = config.training.optim_args
        self.optimizer = self._build_optimizer()
        self.clip_grad_norm = config.training.get('clip_grad_norm', 1.0)
        # scheduler
        self.scheduler_config = config.training.get("scheduler_args", {})
        warmup_steps = self.scheduler_config.get('warmup_steps', 1000)
        tot_steps = self.config.training.get('total_step', 1000000)
        self.lr_scheduler = self._build_scheduler(warmup_steps=warmup_steps, tot_steps=tot_steps)
        
        # save and log config
        self.saved_model_root = os.path.join(config.saver.get('checkpoint_dir', './'), self.config.saver.saved_model_name)
        self.log_save_root = os.path.join(config.saver.get('log_dir', './'), self.config.saver.saved_model_name)

        if self.rank == 0:
            ensure_dir(self.saved_model_root)
            ensure_dir(self.log_save_root)

            # tensorboard
            tensorboard_base_root = os.path.join(self.log_save_root, 'tensorboard', t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(t.time())))
            if not os.path.exists(tensorboard_base_root):
                os.makedirs(tensorboard_base_root)
            from tensorboardX import SummaryWriter
            self.tensorboad_writer = SummaryWriter(tensorboard_base_root)

        self.update_interval = config.get("update_interval", 5)

        self.cur_step = 0
        self.total_step = config.training.get('total_step', 200000)
        self.train_loss_dict = dict()
        
        # frozen
        if config.get('freeze_prefix', None) or config.get('freeze_ad', None):
            freeze_prefix = config.get('freeze_prefix', [])
            if config.get('freeze_ad', None):
                freeze_prefix.extend(['item_llm', 'item_emb_tokens'])
            if not config.get('ft_item', None):
                freeze_prefix.extend(['item_embedding'])
            self._freeze_params(freeze_prefix)
        
        for n, p in self.model.named_parameters():
            self.logger.info(f"{n} {p.size()} {p.requires_grad}")

        print(f'>>> rank: {torch.distributed.get_rank()} init done')

    def _freeze_params(self, freeze_prefix):
        for name, param in self.model.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.logger.info(f"freeze_params: {name}")
                    param.requires_grad = False

    def _build_scheduler(self, warmup_steps=None, tot_steps=None):
        if self.scheduler_config.get('type', None) == 'cosine':
            self.logger.info(f"Use consine scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        elif self.scheduler_config.get('type', None) == 'liner':
            self.logger.info(f"Use linear scheduler with {warmup_steps} warmup {tot_steps} total steps")
            return get_linear_schedule_with_warmup(self.optimizer, warmup_steps, tot_steps)
        else:
            self.logger.info(f"Use constant scheduler")
            return get_constant_schedule(self.optimizer)


    def _build_optimizer(self):
        # if len(self.optim_args) == 4:
        #     params = self.model.named_parameters()
        #     modal_params = []
        #     recsys_params = []
            
        #     for index, (name, param) in enumerate(params):
        #         if param.requires_grad:
        #             if 'visual_encoder' in name:
        #                 modal_params.append(param)
        #             else:
        #                 recsys_params.append(param)
                    
        #     optimizer = optim.AdamW([
        #         {'params': modal_params, 'lr': self.optim_args['learning_rate'], 'weight_decay': self.optim_args['weight_decay']},
        #         {'params': recsys_params, 'lr': self.optim_args['learning_rate'], 'weight_decay': self.optim_args['weight_decay']}
        #     ])
        #     optim_output = set_color(f'recsys_lr_params_len: {len(recsys_params)}  modal_lr_params_len: {len(modal_params)}', 'blue')
        #     self.logger.info(optim_output)
        
        if self.optim_args.get("lr_mult_prefix", None) and self.optim_args.get("lr_mult_rate", None):
            normal_params_dict = {
                "params": [],
                "lr": self.optim_args.learning_rate,
                "weight_decay": self.optim_args.weight_decay
            }
            high_lr_params_dict = {
                "params": [],
                "lr": self.optim_args.learning_rate * self.optim_args.lr_mult_rate,
                "weight_decay": self.optim_args.weight_decay
            }
            self.logger.info(f'Use higher lr rate {self.optim_args.lr_mult_rate} x {self.optim_args.learning_rate} for prefix {self.optim_args.lr_mult_prefix}')
            
            for n, p in self.model.named_parameters():
                if any(n.startswith(x) for x in self.optim_args.lr_mult_prefix):
                    self.logger.info(f"high lr param: {n} {self.optim_args.learning_rate * self.optim_args.lr_mult_rate}")
                    high_lr_params_dict["params"].append(p)
                else:
                    normal_params_dict["params"].append(p)
            optimizer = optim.AdamW([normal_params_dict, high_lr_params_dict])
        elif self.config.get("optimizer_kwargs", None):
            params = self.model.parameters()
            self.config.optim_args.optimizer.params.lr = self.optim_args.learning_rate
            self.config.optim_args.optimizer.params.weight_decay = self.optim_args.weight_decay
            optimizer = deepspeed.ops.adam.cpu_adam.DeepSpeedCPUAdam(params, **self.config.optim_args.optimizer.params)
        else:
            params = self.model.parameters()
            optimizer = optim.AdamW(params, lr=float(self.optim_args.learning_rate), weight_decay=self.optim_args.weight_decay)
        
        return optimizer

    def _train_epoch(self, train_data, show_progress=True):
        self.model.train()
        total_loss = 0
        if self.rank == 0:
            pbar = tqdm(
                total=self.total_step,
                miniters=self.update_interval,
                desc=set_color(f"Step [{self.cur_step}/{self.total_step}]", 'pink'),
                file=sys.stdout
            )
        bwd_time = t.time()
        
        # Get accumulation steps from config or use default value 1
        accumulation_steps = self.config.training.get('accumulation_steps', 1)
        accumulated_steps = 0

        for batch_idx, data in enumerate(train_data):
            # Only zero gradients at the beginning of accumulation cycle
            if accumulated_steps == 0:
                self.optimizer.zero_grad()
                
            start_time = bwd_time
            data = self.to_device(data)
            data_time = t.time()
            
            losses = self.model(data)
            fwd_time = t.time()
            if self.config.get('loss', None) == 'nce':
                model_out = losses
                losses = model_out.pop('loss')

            # Scale the loss to maintain same effective learning rate
            scaled_loss = losses / accumulation_steps
            total_loss = total_loss + losses.item()
            
            # Backward pass with scaled loss
            self.lite.backward(scaled_loss)
            
            accumulated_steps += 1

            # Only update weights after accumulating 'accumulation_steps' gradients
            if accumulated_steps == accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
                self.optimizer.step()
                
                if self.scheduler_config:
                    self.lr_scheduler.step()
                
                # Reset accumulation counter
                accumulated_steps = 0
                
                # Update step counter only after optimizer update
                self.cur_step += 1
                
                # Logging for completed step
                bwd_time = t.time()
                elapse = t.time() - start_time

                
                if show_progress and self.rank == 0 and batch_idx % self.update_interval == 0:
                    cur_step_lr = self.lr_scheduler.get_lr()[0]
                    nce_samples = model_out['nce_samples']
                    nce_top1_acc = model_out['nce_top1_acc']
                    nce_top10_acc = model_out['nce_top10_acc']
                    nce_top100_acc = model_out['nce_top100_acc']
    
                    msg = f"{self.cur_step} / {self.total_step} | loss: {losses:.4f}, lr: {cur_step_lr:.7f}, data_cost: {(data_time - start_time):.2f}, forward_cost: {(fwd_time - data_time):.3f}, bwd: {(bwd_time - fwd_time):.3f}, elapse: {elapse:.4f}, top1_acc: {nce_top1_acc.item():.4f}, top10_acc: {nce_top10_acc.item():.4f}, top100_acc: {nce_top100_acc.item():.4f}"
                    
                    # TensorBoard logging
                    if self.rank == 0:
                        self.tensorboad_writer.add_scalar('lr', cur_step_lr, self.cur_step)
                        self.tensorboad_writer.add_scalar('loss', losses.item(), self.cur_step)
                        # self.tensorboad_writer.add_scalar('user_embed_loss', model_out['user_embed_loss'].item(), self.cur_step)
                        
                        self.tensorboad_writer.add_scalar('nce_samples', nce_samples.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top1_acc', nce_top1_acc.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top10_acc', nce_top10_acc.item(), self.cur_step)
                        self.tensorboad_writer.add_scalar('nce_top100_acc', nce_top100_acc.item(), self.cur_step)
                        
                        # if model_out['ae_decay'] > 0:
                        #     self.tensorboad_writer.add_scalar('ae_decay', model_out['ae_decay'], self.cur_step)
                        #     self.tensorboad_writer.add_scalar('reconstruct_loss', model_out['reconstruct_loss'].item(), self.cur_step)
    
                        # if 'action_pred_loss' in model_out:
                        #     self.tensorboad_writer.add_scalar('action_pred_loss', model_out['action_pred_loss'], self.cur_step)
                        #     self.tensorboad_writer.add_scalar('action_pred_acc', model_out['action_pred_acc'], self.cur_step)
    
                    self.logger.info(msg)
                    self.logger.info("\n" + "-"*50)
                
                # Save model
                if self.cur_step % self.config.training.eval_step == 0:
                    self._save_checkpoint()
                    
                if self.cur_step == self.total_step:
                    break
            else:
                # Update time tracking even when not performing optimizer step
                bwd_time = t.time()

        # Handle any remaining accumulated gradients at the end of the epoch
        if accumulated_steps > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)
            self.optimizer.step()
            
            if self.scheduler_config:
                self.lr_scheduler.step()
                
            self.cur_step += 1

        return total_loss
    
    
    def _save_checkpoint(self, verbose=True):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            'config': self.config,
            'epoch': 0,
            'cur_step': self.cur_step,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state()
        }
        self.lite.save(os.path.join(self.saved_model_root, 'checkpoint-{}'.format(self.cur_step+1)), state=state)
        if self.rank == 0 and verbose:
            self.logger.info(set_color('Saving current', 'blue') + f': {self.saved_model_root}')
    
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')
    
    def to_device(self, data):
        device = self.device
        if isinstance(data, tuple) or isinstance(data, list):
            tdata = ()
            for d in data:
                d = d.to(device)
                tdata += (d,)
            return tdata
        elif isinstance(data, dict):
            for k, v in data.items():
                if "action" not in k and (k != "pos_inputs" or k != "neg_inputs"):
                    data[k] = v.to(device)
            if "pos_inputs" in data.keys():
                for key in data['pos_inputs'].keys():
                    data['pos_inputs'][key] = data['pos_inputs'][key].to(device)
            
            if "neg_inputs" in data.keys():
                for key in data['neg_inputs'].keys():
                    data['neg_inputs'][key] = data['neg_inputs'][key].to(device)
            return data
        else:
            return data.to(device)


    def fit(self, train_data, show_progress=False):
        world_size, local_world_size = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
        nnodes = world_size // local_world_size
        precision = self.config.get('precision', 'bf16-mixed')
        if self.config.training.get('strategy', None) == 'deepspeed':
            self.logger.info(f"Use deepspeed strategy")
            strategy = DeepSpeedStrategy(stage=self.config.training.stage, precision=precision)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        else:
            self.logger.info(f"Use DDP strategy")
            strategy = DDPStrategy(find_unused_parameters=True)
            self.lite = L.Fabric(accelerator='gpu', strategy=strategy, precision=precision, num_nodes=nnodes)
        self.lite.launch()
        self.model, self.optimizer = self.lite.setup(self.model, self.optimizer)

        if self.config.get('auto_resume', False):
            raise NotImplementedError

        self._train_epoch(train_data, show_progress=show_progress)
            

    @torch.no_grad()
    def get_item_embedding(self, note_id):
        self.model.eval()

        '''
        text = self.process_item(note_id)
        if text is None:
        ids, _ = self.llama_process(text)
        pos_input_ids.extend(ids + [0])
        pos_cu_input_lens.append(len(ids) + 1)
        pos_position_ids.extend((torch.arange(len(ids) + 1) + (self.max_text_length - len(ids))).tolist())

        interaction = {
            "pos_input_ids": torch.as_tensor(pos_input_ids, dtype=torch.int64),
            "pos_cu_input_lens": torch.as_tensor(pos_cu_input_lens, dtype=torch.int64),
            "pos_position_ids": torch.as_tensor(pos_position_ids, dtype=torch.int64),
        }
        '''

        interaction = None
        item_embedding_2048, item_embedding_64 = self.model.compute_item(interaction)
        return item_embedding_2048, item_embedding_64
    
    @torch.no_grad()
    def predict(self, user_lastn, attention_mask=None):
        self.model.eval()
        user_seq_feature = torch.rand([256, 100, 64]).bfloat16()
        attention_mask = torch.ones([256, 100])
        user_embedding = self.model.compute_user_embedding(user_seq_feature, attention_mask)
        return user_embedding
    
    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        return concat.sum() / num_total_examples
