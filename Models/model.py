from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# import s3prl.hub as hub

# # Wav2vec2, Hubert
# class UpstreamTransformer(nn.Module):
#     def __init__(self, upstream_model='wav2vec2', feature_dim=768, unfreeze_last_conv_layers=False):
#         super().__init__()
#         self.upstream = torch.hub.load('s3prl/s3prl', upstream_model) # hubert
        
#         for param in self.upstream.parameters():
#             param.requires_grad = False
        
#         for param in self.upstream.model.encoder.layers.parameters():
#             param.requires_grad = True

#         if unfreeze_last_conv_layers:
#             for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
#                 param.requires_grad = True
        
#         self.language_classifier = nn.Sequential(
#             nn.Linear(feature_dim, 14)
#         )

#     def forward(self, x, x_len):
#         x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
#         x = self.upstream(x)['last_hidden_state']
#         x = torch.mean(x, dim=1)
#         language = self.language_classifier(x)
#         return language


# class UpstreamTransformerXLSR(nn.Module):
#     def __init__(self, upstream_model='xlsr_300m', feature_dim=1024, unfreeze_last_conv_layers=False):
#         super().__init__()

#         self.xlsrModelsUrls = {'xlsr_300m': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt',
#         'xlsr_1b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_960m_1000k.pt',
#         'xlsr_2b': 'https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_2B_1000k.pt'}

#         self.upstream = getattr(hub, 'wav2vec2_url')(self.xlsrModelsUrls[upstream_model])
        
#         for param in self.upstream.parameters():
#             param.requires_grad = False
        
#         for param in self.upstream.model.encoder.layers.parameters():
#             param.requires_grad = True

#         if unfreeze_last_conv_layers:
#             for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
#                 param.requires_grad = True
        
#         self.language_classifier = nn.Sequential(
#             nn.Linear(feature_dim, 14),
#         )

#     def simple_forward(self, x, x_len):
#         x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
#         x = self.upstream(x)['last_hidden_state']
#         x = torch.mean(x, dim=1)
#         language = self.language_classifier(x)
#         return language
    
#     def forward(self, x, x_len):
#         return self.simple_forward(x, x_len)


from transformers import Wav2Vec2Processor, Wav2Vec2Model

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class UpstreamTransformerXLSR(nn.Module):
    def __init__(self, upstream_model='xlsr_300m', feature_dim=1024, unfreeze_last_conv_layers=False):
        super().__init__()

        self.processor = Wav2Vec2Processor.from_pretrained("/root/autodl-tmp/facebook/wav2vec2-base-960h")
        self.encoder = Wav2Vec2Model.from_pretrained("/root/autodl-tmp/facebook/wav2vec2-xls-r-300m")
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        for param in self.encoder.encoder.layers.parameters():
            param.requires_grad = True

        # print(self.encoder.encoder.layers)

        # if unfreeze_last_conv_layers:
        #     for param in self.encoder.model.feature_extractor.conv_layers[5:].parameters():
        #         param.requires_grad = True

        self.attention_pool = SelfAttentionPooling(feature_dim)
        
        # 二分类任务，输出维度改为2
        self.language_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def simple_forward(self, x, x_len):
        # 对批次中的每个样本单独处理，以避免一次性处理整个批次导致内存溢出
        batch_size = x.size(0)
        outputs = []
        
        for i in range(batch_size):
            # 获取单个样本
            sample = x[i].squeeze().detach().cpu().numpy()
            length = int(x_len[i])
            
            # 如果有长度信息，裁剪到指定长度
            if length > 0 and length <= len(sample):
                sample = sample[:length]
            
            # 处理单个样本
            try:
                # 转换为模型输入格式
                processed = self.processor(sample, sampling_rate=16000, return_tensors="pt")["input_values"].to(self.encoder.device)
                
                # 前向传播
                encoder_output = self.encoder(processed).last_hidden_state
                
                # 注意力池化
                pooled = self.attention_pool(encoder_output)
                
                # 分类
                logits = self.language_classifier(pooled)
                outputs.append(logits)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # 创建一个零向量作为回退（维度为2，匹配二分类任务）
                outputs.append(torch.zeros(1, 2).to(self.encoder.device))
        
        # 堆叠所有输出
        return torch.cat(outputs, dim=0)
    
    def forward(self, x, x_len):
        return self.simple_forward(x, x_len)


# from speechbrain.pretrained import EncoderClassifier

# class PretrainedLangID(nn.Module):
#     def __init__(self, upstream_model=None, feature_dim=None, unfreeze_last_conv_layers=None):
#         super().__init__()
#         self.lang_enc = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp", run_opts={"device":"cuda"})
        
#         for param in self.lang_enc.parameters():
#             param.requires_grad = True
        
#         # print(self.lang_enc)
#         # self.lang_enc.eval()
        
#         self.language_classifier = nn.Sequential(
#             nn.Linear(256, 14)
#         )

#     def forward(self, x, x_len):
#         # print(x.shape)
#         # x = x.squeeze()
#         language_emb = self.lang_enc.encode_batch(x).squeeze(1)
#         # print("SHAPE = ", language_emb.shape)
#         language = self.language_classifier(language_emb)
#         # print(language.shape)
#         return language

