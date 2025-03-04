import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

from models.model_pretrain import FashionFINE
from models.vit import interpolate_pos_embed
from transformers.models.bert.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

import torch.distributed as dist
import nltk
#nltk.data.path.append("./")

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    
    
    start_time = time.time()  

    text_feats = []
    img_feats = []
    text_embeds = []
    img_embeds = []
    text_atts = []
    texts = data_loader.dataset.texts
    text_num = len(texts)
    tbs = 256
    for i in range(0, text_num, tbs):
        text = texts[i:min(text_num, i+tbs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_word_num, return_tensors="pt").to(device)
        text_embed = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask,                      
                                        return_dict = True, mode = 'text')
        text_embed = text_embed.last_hidden_state
        text_feat = model.combine_text_proj(model.text_proj(text_embed[:,0,:]))
        text_feat = F.normalize(text_feat, dim=-1)   
        
        text_word = text_embed[:,2:,:]
        text_att = model.tanh(model.att_layer_v(text_word))
        text_att = text_att @ model.W_v
        text_att = (text_att.softmax(dim=1)).expand(-1,-1,768)
        text_word = text_word.mul(text_att)
        
        text_word_sum = text_word.sum(1)
        
        text_word_sum = F.normalize(model.combine_text_proj(model.text_proj(text_word_sum)), dim=-1)
        
        text_feat = text_feat*(1-args.text_patch_ratio) + text_word_sum*args.text_patch_ratio
        text_feat = F.normalize(text_feat, dim=-1)
        text_feats.append(text_feat)
        
    print('computing imgs feature')

    for i, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
        img= batch[0].to(device)
        img_embed = model.visual_encoder(img)
        img_feat = model.combine_vision_proj(model.vision_proj(img_embed[:,0,:]))
        img_feat = F.normalize(img_feat, dim=-1)
        
        img_word = img_embed[:,1:,:]
        img_att = model.tanh(model.att_layer_v(img_word))
        img_att = img_att @ model.W_v
        img_att = (img_att.softmax(dim=1)).expand(-1,-1,768)
        img_word = img_word.mul(img_att)
        
        img_word_sum = img_word.sum(1)
        
        img_word_sum = F.normalize(model.combine_vision_proj(model.vision_proj(img_word_sum)),dim=-1)
        
        
        img_feat = img_feat*(1-args.img_patch_ratio) + img_word_sum*args.img_patch_ratio
        img_feat = F.normalize(img_feat, dim=-1)
        img_feats.append(img_feat)

    text_feats = torch.cat(text_feats, dim=0)
    img_feats = torch.cat(img_feats, dim=0)
    sim_t2i = text_feats @ img_feats.t()
 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))



    return sim_t2i.T.cpu().numpy(), sim_t2i.cpu().numpy()

      
@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt=None, txt2img=None, tiny_i2t=None, tiny_t2i=None):
    if img2txt is None:
        img2txt = {i:i for i in range(scores_i2t.shape[0])}
        txt2img = {k:[v] for k,v in img2txt.items()}
    
    print('calculating i2t ranks')
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == img2txt[index])[0][0]
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)


    #tiny Images -> Text
    tr1_tiny = 0
    tr5_tiny = 0
    tr10_tiny = 0
    if tiny_i2t is not None:
        img_indexes, txt_indexes = tiny_i2t
        tiny_score_i2t = np.zeros(txt_indexes.shape)
        tiny_score = scores_i2t[img_indexes]
        for i, t_socre in enumerate(tiny_score):
            tiny_score_i2t[i] = t_socre[txt_indexes[i]]
        ranks = np.zeros(tiny_score_i2t.shape[0])
        for index,score in enumerate(tiny_score_i2t):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == 100)[0][0]
        # Compute metrics
        tr1_tiny = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5_tiny = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10_tiny = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images
    print('calculating t2i ranks')
    ranks = np.zeros(scores_t2i.shape[0])
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        rank = 100000
        for i in txt2img[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)


    
    #tiny text -> Images
    ir1_tiny = 0
    ir5_tiny = 0
    ir10_tiny = 0
    if tiny_t2i is not None:
        txt_indexes, img_indexes= tiny_t2i
        tiny_score_t2i = np.zeros(img_indexes.shape)
        tiny_score = scores_t2i[txt_indexes]
        for i, t_socre in enumerate(tiny_score):
            tiny_score_t2i[i] = t_socre[img_indexes[i]]
        ranks = np.zeros(tiny_score_t2i.shape[0])
        for index,score in enumerate(tiny_score_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == 100)[0][0]
        # Compute metrics
        ir1_tiny = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5_tiny = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10_tiny = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    tiny_sum = tr1_tiny + tr5_tiny + tr10_tiny + ir1_tiny + ir5_tiny + ir10_tiny

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr1 + ir1) / 2
    full_sum = tr1 + tr5 + tr10 + ir1 + ir5 + ir10

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'tiny_txt_r1':tr1_tiny,
                    'tiny_txt_r5':tr5_tiny,
                    'tiny_txt_r10':tr10_tiny,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'tiny_img_r1':ir1_tiny,
                    'tiny_img_r5':ir5_tiny,
                    'tiny_img_r10':ir10_tiny,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean,
                    'tiny_sum':tiny_sum,
                   'full_sum':full_sum}
    return eval_result

def train(model, data_loader, optimizer, epoch, warmup_steps, device, scheduler, config, output_dir):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('sis', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('ml', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('rl', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = [i.to(device, non_blocking=True) for i in batch]
        (image, text_input_ids, text_attention_mask, mask_labels, replace_labels, idx) = batch
            
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        
        loss_ita, loss_itm, symbol_simloss, mask_loss, replace_loss = model(image, text_input_ids, text_attention_mask,alpha=alpha, idx=idx,
                                mask_labels=mask_labels, replace_labels=replace_labels)
                          
        loss = loss_ita + loss_itm + symbol_simloss + mask_loss + replace_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(itm=loss_itm.item())
        metric_logger.update(ita=loss_ita.item())
        metric_logger.update(sis=symbol_simloss.item())
        metric_logger.update(ml=mask_loss.item())
        metric_logger.update(rl=replace_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    with open(os.path.join(output_dir, "log.txt"),"a") as f:
        f.write("Averaged stats: " + str(metric_logger.global_avg())) 
        f.write("\n")
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def main(args, config):
    utils.init_distributed_mode(args)    
    
    if utils.is_main_process():
        nltk.download('wordnet')
        nltk.download('omw-1.4')
    torch.distributed.barrier()
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    
    #### process tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_config'])
    inner_slots = [
            '[tops_sign]',
            '[pants_sign]',
            '[skirts_sign]',
            '[dresses_sign]',
            '[coats_sign]',
            '[shoes_sign]',
            '[bags_sign]',
            '[accessories_sign]',
            '[others_sign]',
    ]
    tokenizer.add_tokens(inner_slots, special_tokens=True)
    #### tokenizer done ####
    #### Dataset #### 
    print("Creating dataset")
    train_dataset, test_dataset = create_dataset('pretrain', config, args,tokenizer)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
    else:
        samplers = [None, None, None]
    
    train_loader, test_loader = create_loader([train_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4],
                                                          is_trains=[True, False], 
                                                          collate_fns=[None,None])   
       

    #### Model #### 
    print("Creating model")
    model = FashionFINE(config=config, args=args)


    if args.pre_point:
        checkpoint = torch.load(args.pre_point, map_location='cpu')    
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.pre_point)
        print(msg)
    elif args.bert_point and args.vit_point:
        t_checkpoint = torch.load(args.bert_point, map_location='cpu')
        t_checkpoint = utils.text_state_compatibility(t_checkpoint)
        msg = model.text_encoder.load_state_dict(t_checkpoint, strict=False)
        print('load checkpoint from %s'%args.bert_point)
        print(msg)
        
        v_checkpoint = torch.load(args.vit_point, map_location='cpu')['model']
        pos_embed_reshaped = interpolate_pos_embed(v_checkpoint['pos_embed'],model.visual_encoder)         
        v_checkpoint['pos_embed'] = pos_embed_reshaped
        for k in list(v_checkpoint.keys()):
            if k.startswith('head.'):
                del v_checkpoint[k]
        msg = model.visual_encoder.load_state_dict(v_checkpoint, strict=True)
        print('load checkpoint from %s'%args.vit_point)
        print(msg)
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.device != 'cpu' and args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    patience = [20]
    dist.broadcast_object_list(patience,src=0)
    per_list = []
    per_list = [None,None,None,None,None]
    
    print(args)
    print(config)
    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, train_loader, optimizer, epoch, warmup_steps, device, lr_scheduler, config, args.output_dir)
            print(train_stats)   
            score_val_i2t, score_val_t2i= evaluation(model_without_ddp, test_loader, tokenizer, device, config, args)
            
            if utils.is_main_process():  
                img2text, text2img = test_dataset.get_test_labels()
                tiny_i2t_img, tiny_i2t_txt = test_dataset.get_i2t_test()
                tiny_t2i_txt, tiny_t2i_img = test_dataset.get_t2i_test()
                val_result = itm_eval(score_val_i2t, score_val_t2i, img2text, text2img, tiny_i2t=(tiny_i2t_img, tiny_i2t_txt), tiny_t2i=(tiny_t2i_txt, tiny_t2i_img))  
                print(val_result)  
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(str(epoch) + ": " +str(val_result) + "\n")   
            
                outpath = args.output_dir
                ## save every epoch
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                save_obj = {
                    'model': model_without_ddp.state_dict()
                }
                torch.save(save_obj, os.path.join(outpath, 'checkpoint_epoch'+str(epoch)+'.pth'))  
                ## save every epoch
                if val_result['full_sum'] > best:
                    if utils.is_main_process():
                        if not os.path.exists(outpath):
                            os.makedirs(outpath)
                        save_obj = {
                            'model': model_without_ddp.state_dict()
                        }
                        with open(os.path.join(args.output_dir, "best_log.txt"),"a") as f:
                            f.write(str(epoch) + ": " +str(val_result) + "\n")   
                        torch.save(save_obj, os.path.join(outpath, 'checkpoint_best.pth'))  

                    best = val_result['full_sum']    
                    best_epoch = epoch
                    patience = [20]

                else:
                    patience[0] = patience[0] - 1
                    
                
            
            if utils.is_main_process():
                # per_list
                for i, per in enumerate(per_list):
                    if per is None:
                        per_list.insert(i,val_result['full_sum'])
                        torch.save(save_obj, os.path.join(outpath, 'checkpoint_'+str(i)+'.pth'))  
                        break
                    elif val_result['full_sum'] > per:
                        for j in range(4,i-1,-1): # 4, 3, 2, 1
                            if per_list[j] is not None:
                                os.rename(os.path.join(outpath, 'checkpoint_'+str(j)+'.pth'),os.path.join(outpath, 'checkpoint_'+str(j+1)+'.pth'))

                        per_list.insert(i,val_result['full_sum'])
                        per_list = per_list[:5]
                        if os.path.exists(os.path.join(outpath, 'checkpoint_'+str(5)+'.pth')):
                            os.remove(os.path.join(outpath, 'checkpoint_'+str(5)+'.pth'))
                        ## checkpoint_6 삭제
                        torch.save(save_obj, os.path.join(outpath, 'checkpoint_'+str(i)+'.pth'))  
                        break

                            
                    
            torch.distributed.barrier()
            dist.broadcast_object_list(patience,src=0)
        if args.evaluate or patience[0] == 0: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
              

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/fashion_pretrain.yaml')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--pre_point', default='ALBEF.pth')
    # parser.add_argument('--bert_point', default='bert-base-uncased-pytorch_model.bin')
    # parser.add_argument('--vit_point', default='deit_base_patch16_224-b5f2ef4d.pth')
    
    parser.add_argument('--alpha_focal', default=0.5, type=float)
    parser.add_argument('--gamma_focal', default=2.0, type=float)
    parser.add_argument('--text_patch_ratio', default=0.25, type=float)
    parser.add_argument('--img_patch_ratio', default=0.05, type=float)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    # parser.add_argument('--sub_dataset', default=False, type=bool)

    parser.add_argument('--max_word_num', default=180, type=int)
    parser.add_argument('--data_root', default='', type=str)
    parser.add_argument('--catemap_filename', default='categorys_to_sign.txt', type=str)
    parser.add_argument('--product_list_filename', default='productid_list.json', type=str)
    parser.add_argument('--replace_kind_num', default=2, type=int)
    
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
