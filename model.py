#pip list --format=freeze | grep -v "file://" > requirements.txt
import torch
import torch.nn as nn
from transformers import (
    ViTModel, ViTConfig,
    GPT2LMHeadModel, GPT2Config,
)

import pandas as pd

# # load CSVs
# train_df = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/train_hw.csv")
# val_df   = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/val_hw.csv")

# # save formulas into plain text files (one per line)
# train_df["formula"].to_csv("/kaggle/working/latex_formulas_train.txt", index=False, header=False)
# val_df["formula"].to_csv("/kaggle/working/latex_formulas_val.txt", index=False, header=False)

# # combine both if you want to train tokenizer on all available formulas
# with open("/kaggle/working/latex_formulas_all.txt", "w", encoding="utf-8") as f:
#     for formula in pd.concat([train_df["formula"], val_df["formula"]]):
#         f.write(str(formula).strip() + "\n")

# from tokenizers import ByteLevelBPETokenizer
# from pathlib import Path

# paths = [str(Path("latex_formulas_all.txt"))]

# tokenizer = ByteLevelBPETokenizer()

# tokenizer.train(
#     files=paths,
#     vocab_size=8000,   # tune depending on dataset size
#     min_frequency=2,
#     special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
# )

# tokenizer.save_model("/kaggle/working")

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("tokenize_folder")

# add special tokens (if not already)
specials = {"pad_token": "<|pad|>", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}
tokenizer.add_special_tokens(specials)

class ImageToLatexModel(nn.Module):
    """
    Vision-Encoder / Text-Decoder architecture.
    - Encoder  : ViT (small, patch16)
    - Decoder  : GPT-2 (trimmed: 4 layers, 512 dim)
    A Hugging Face tokenizer (same vocab as decoder) must be provided
    when you do loss computation or generation.
    """
    def __init__(
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        self,
        
        image_size=(224, 224),   # (H, W)
        
        
        enc_name="google/vit-base-patch16-224-in21k",
        # print('enc name')
        enc_layers=8,
        # print('enc layers')
        dec_name="gpt2",
        # print('dec name')
        dec_layers=4,
        dec_hidden=512,
        tokenizer=None
    ):
        
        super().__init__()

        # ---------- Vision Encoder ----------
        print('******************************')
        enc_cfg = ViTConfig.from_pretrained(enc_name)
        enc_cfg.image_size = image_size
        print('image size')
        enc_cfg.num_hidden_layers = enc_layers
        print('hidden')
        # enc_cfg.num_channels=1
        self.encoder = ViTModel.from_pretrained(enc_name, config=enc_cfg)
        print('encoder')

        # ---------- Text Decoder ----------
        dec_cfg = GPT2Config.from_pretrained(dec_name)
        dec_cfg.n_layer = dec_layers
        print('n layer')
        dec_cfg.n_embd = dec_hidden
        dec_cfg.n_head = dec_hidden // 64
        print('n_head')
        dec_cfg.add_cross_attention = True     
        dec_cfg.is_decoder = True
        dec_cfg.bos_token_id = tokenizer.bos_token_id if tokenizer else 0
        print('bos token id')
        dec_cfg.eos_token_id = tokenizer.eos_token_id if tokenizer else 1
        print('eos token id')
        dec_cfg.pad_token_id = tokenizer.pad_token_id or dec_cfg.eos_token_id
        print('pad token id')

        self.decoder = GPT2LMHeadModel(dec_cfg)
        print('decoder')
        self.tokenizer = tokenizer
        print(tokenizer)
        print(dec_hidden)
        print(self.encoder.config.hidden_size)
        # project encoder hidden dim -> decoder hidden dim (if mismatch)
        if self.encoder.config.hidden_size != dec_hidden:
            self.enc2dec = nn.Linear(
                self.encoder.config.hidden_size, dec_hidden, bias=False
            )
            print('enc to dec')
        else:
            self.enc2dec = nn.Identity()
            print('enc to dec-2')

    # ---------------------------------------------
    def forward(
        self,
        pixel_values,           # (B, C, H, W)
        labels=None,            # (B, L) with -100 for padding
        **generate_kwargs       # used only when labels is None
    ):
       
        enc_out = self.encoder(pixel_values=pixel_values)
        enc_hidden = self.enc2dec(enc_out.last_hidden_state)

        pad_id = self.decoder.config.pad_token_id
        # print(f' pad id ***************************** {pad_id}')
        # print(f' ****************************************************************** {self.decoder.config.bos_token_id}')

        
        
        if labels is not None:
            # Teacher-forcing inputs:  shift right & replace -100 → pad_id
            labels = torch.cat(
        [labels.new_full((labels.size(0), 1), tokenizer.bos_token_id), labels], dim=1
    )
            # print(f' labels shape is {labels.shape}||||| labels : {labels}')
            decoder_in = labels[:, :-1].clone()
            # print(f' decoder shape is {decoder_in.shape}||||||||||| decoder {decoder_in}')
            decoder_in[decoder_in == -100] = pad_id
            
            attention = decoder_in != pad_id
        
            outputs = self.decoder(
                input_ids=decoder_in,
                attention_mask=attention,
                encoder_hidden_states=enc_hidden,
                encoder_attention_mask=torch.ones(
                    enc_hidden.shape[:2], dtype=torch.long, device=enc_hidden.device
                ),
                labels=labels[:, 1:],   # <- target still has -100 where needed
                return_dict=True
            )

       
            return outputs.loss, outputs.logits

        # ----- generation mode -----
        gen_ids = self.decoder.generate(
            bos_token_id=self.decoder.config.bos_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            pad_token_id=self.decoder.config.pad_token_id,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=torch.ones(
                enc_hidden.shape[:2], dtype=torch.long, device=enc_hidden.device
            ),
            **generate_kwargs
        )
        return gen_ids
