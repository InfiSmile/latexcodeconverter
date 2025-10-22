import pandas as pd

from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

def strip_math_delim(tex: str) -> str:
    # tex = tex.strip()
    # if tex.startswith("$$") and tex.endswith("$$"):
    #     return tex[2:-2]
    if tex.startswith("$") and tex.endswith("$"):
        return tex[1:-1]
    # if tex.startswith(r"\(") and tex.endswith(r"\)"):
    #     return tex[2:-2]
    return tex

# load CSVs
train_df = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/train_hw.csv")
val_df   = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/val_hw.csv")

# save formulas into plain text files (one per line)
train_df["formula"].to_csv("/kaggle/working/latex_formulas_train.txt", index=False, header=False)
val_df["formula"].to_csv("/kaggle/working/latex_formulas_val.txt", index=False, header=False)

# combine both if you want to train tokenizer on all available formulas
with open("/kaggle/working/latex_formulas_all.txt", "w", encoding="utf-8") as f:
    for formula in pd.concat([train_df["formula"], val_df["formula"]]):
        f.write(str(formula).strip() + "\n")


paths = [str(Path("latex_formulas_all.txt"))]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=paths,
    vocab_size=8000,   # tune depending on dataset size
    min_frequency=2,
    special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
)

tokenizer.save_model("/kaggle/working")

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("/kaggle/working/")

# add special tokens (if not already)
specials = {"pad_token": "<|pad|>", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}
tokenizer.add_special_tokens(specials)

import os, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms



class HandwrittenLatexDataset(Dataset):

    def __init__(self,
                 csv_path: str,
                 images_root: str,
                 tokenizer,
                 img_size=(224, 224),
                 max_length: int = 256):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.images_root = images_root
        self.tokenizer = tokenizer
        self.max_length = max_length

        # basic augmentations -- tweak as you like
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),                   
            transforms.Resize(img_size),
            transforms.ToTensor(),                    
            transforms.Normalize(0.5, 0.5)            # mean=0.5, std=0.5
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, row.image)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return None
             
        
        pixel_values = self.transform(image)

       
        labels = self.tokenizer(strip_math_delim(row.formula),
                                max_length=self.max_length,
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt").input_ids.squeeze(0)
        return {"pixel_values": pixel_values, "labels": labels}


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_token_id):
    batch = [b for b in batch if b is not None]   
    if len(batch) == 0:
        return None
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    # print(f' pixel values shape {pixel_values.shape}')

    seqs = []

    for lbl in [item["labels"] for item in batch]:
        # put EOS only if it’s not already there
        if lbl[-1].item() != tokenizer.eos_token_id:
            lbl = torch.cat([lbl, lbl.new_tensor([tokenizer.eos_token_id])])
        seqs.append(lbl)
    
    # labels = [item["labels"] for item in batch]
    # print(f' labels shape {labels.shape}')
    labels_padded = pad_sequence(seqs,
                                 batch_first=True,
                                 padding_value=pad_token_id)
    
    labels_padded[labels_padded == pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels_padded}


