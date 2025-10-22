# import pandas as pd
# from model import ImageToLatexModel
# from transformers import GPT2TokenizerFast
# # load CSVs
# # train_df = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/train_hw.csv")
# # val_df   = pd.read_csv("/kaggle/input/latex-dataset/HandwrittenData/val_hw.csv")

# # # save formulas into plain text files (one per line)
# # train_df["formula"].to_csv("latex_formulas_train.txt", index=False, header=False)
# # val_df["formula"].to_csv("latex_formulas_val.txt", index=False, header=False)

# # combine both if you want to train tokenizer on all available formulas
# # with open("/kaggle/working/latex_formulas_all.txt", "w", encoding="utf-8") as f:
# #     for formula in pd.concat([train_df["formula"], val_df["formula"]]):
# #         f.write(str(formula).strip() + "\n")

# # from tokenizers import ByteLevelBPETokenizer
# # from pathlib import Path

# # paths = [str(Path("latex_formulas_all.txt"))]

# # tokenizer = ByteLevelBPETokenizer()

# # tokenizer.train(
# #     files=paths,
# #     vocab_size=8000,   # tune depending on dataset size
# #     min_frequency=2,
# #     special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
# # )

# # tokenizer.save_model()

# from transformers import GPT2TokenizerFast

# tokenizer = GPT2TokenizerFast.from_pretrained('tokenize_folder')

# # add special tokens (if not already)
# specials = {"pad_token": "<|pad|>", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}
# tokenizer.add_special_tokens(specials)

# import streamlit as st
# import torch
# from PIL import Image
# import torchvision.transforms as T
# from transformers import GPT2TokenizerFast
# from model import ImageToLatexModel

# @st.cache_resource
# def load_model():
#     tokenizer = GPT2TokenizerFast.from_pretrained("tokenize_folder")
#     model = ImageToLatexModel(tokenizer=tokenizer)
#     model.decoder.resize_token_embeddings(len(tokenizer))
#     checkpoint = torch.load("latex_epoch89.pt", map_location="cpu")
#     model.load_state_dict(checkpoint["model_state"], strict=False)
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

# st.title("🖊️ Handwritten Formula → LaTeX Converter")
# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.5], std=[0.5]),
#     ])

#     input_tensor = transform(img).unsqueeze(0)

#     with torch.no_grad():
#         with torch.amp.autocast(device_type="cpu"):
#             logits = model(input_tensor, labels=None)
#         preds = torch.argmax(logits, dim=-1)[0]
#         tokens = tokenizer.decode(preds, skip_special_tokens=True)

#     st.subheader("📄 Predicted LaTeX:")
#     st.code(tokens, language="latex")

import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T
from transformers import GPT2TokenizerFast
from model import ImageToLatexModel
from huggingface_hub import hf_hub_download

# -------------------- Load model & tokenizer --------------------
@st.cache_resource
def load_model():
    # Load tokenizer from trained folder
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenize_folder")
    specials = {"pad_token": "<|pad|>", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}
    tokenizer.add_special_tokens(specials)

    # Initialize model
    model = ImageToLatexModel(tokenizer=tokenizer)
    model.decoder.resize_token_embeddings(len(tokenizer))

    
# import torch

    model_path = hf_hub_download(
        repo_id="INFISKI/latexcodeconverter",
        filename="latex_epoch89.pt"
    )



    # Load checkpoint safely
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# -------------------- Streamlit UI --------------------
st.title("🖊️ Handwritten Formula → LaTeX Converter")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image to match ViT encoder
    transform = T.Compose([
        T.Resize((224, 224)),  # ViT input size
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    input_tensor = transform(img).unsqueeze(0)

    # -------------------- Generate LaTeX --------------------
    with torch.no_grad():
        # Get encoder hidden states
        enc_out = model.encoder(pixel_values=input_tensor)
        enc_hidden = model.enc2dec(enc_out.last_hidden_state)

        # Generate tokens from decoder
        gen_ids = model.decoder.generate(
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=torch.ones(enc_hidden.shape[:2], device=enc_hidden.device),
            max_length=50,
            num_beams=4,
            early_stopping=True,
            bos_token_id=model.decoder.config.bos_token_id,
            eos_token_id=model.decoder.config.eos_token_id,
            pad_token_id=model.decoder.config.pad_token_id
        )

        # Decode generated tokens
        tokens = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    st.subheader("📄 Predicted LaTeX:")
    st.code(tokens, language="latex")
