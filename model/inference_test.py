import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import cv2
from utils import *
import sys
from e2e_vsr import E2EVSR
import kornia.augmentation as K
import google.generativeai as genai



# —— Static token mapping from trained VSR model ———
# Load tokens map from dia_tokens.txt
mapped_tokens = {}
tokens_path = os.path.join(os.path.dirname(__file__), "dia_tokens.txt")
with open(tokens_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, val = line.split(':')
        key = key.strip().strip("'\"")
        val = int(val.strip())
        mapped_tokens[key] = val
# Special tokens
mapped_tokens[""] = 0
mapped_tokens["<sos>"] = max(mapped_tokens.values()) + 1
mapped_tokens["<eos>"] = mapped_tokens["<sos>"] + 1

# Reverse mapping for inference
idx2char = {v: k for k, v in mapped_tokens.items()}
sos_idx = mapped_tokens['<sos>']
eos_idx = mapped_tokens['<eos>']

## Optional LLM cleanup
def llm_cleanup(text, api_key=None, model_name='learnlm-2.0-flash-experimental'):
    if not genai or not api_key:
        return text
    genai.configure(api_key=api_key)
    prompt = f"""Clean this Arabic lipread output: '{text}'. Remove gibberish and repeated parts. Make sure words are separated by spaces.
    """
    response = genai.chat.create(
        model=model_name,
        messages=[{'role':'system','content':'You are an Arabic text post-processing assistant.'},
                  {'role':'user','content':prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content

# Dataset class for inference (single sample)
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, transform=None):
        self.paths = video_paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        vp = self.paths[idx]
        pil_frames = []
        cap = cv2.VideoCapture(vp)
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(cnt):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, fr = cap.read()
            if not ret:
                break
            # Convert to grayscale
            if fr.ndim == 3:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            else:
                gray = fr
            pil = Image.fromarray(gray, 'L')
            pil_frames.append(pil)
        cap.release()
        # apply kornia-based center crop and normalization
        video = self.transform(pil_frames)
        return video, torch.tensor(video.size(1), dtype=torch.long), os.path.basename(vp)

# Inference-only augmentation: center-crop to 88x88 and normalize
class CenterCropNormalize:
    def __init__(self, mean=0.41923218965530395, std=0.13392585515975952, crop_size=(88, 88)):
        self.crop = K.VideoSequential(
            K.CenterCrop(crop_size, p=1.0),
            data_format="BCTHW", same_on_frame=True
        )
        self.mean = mean
        self.std = std

    def __call__(self, pil_frames):
        # Convert list of PIL images to tensor sequence (T, C, H, W)
        frame_ts = [transforms.ToTensor()(img) for img in pil_frames]
        video = torch.stack(frame_ts, dim=0).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        cropped = self.crop(video).squeeze(0)  # (C, T, 88, 88)
        return (cropped - self.mean) / self.std

if __name__=='__main__':
    # Default directories and options
    test_dir = os.path.join(os.path.dirname(__file__), "../test_data/Preprocessed_Videos")
    csv_dir = os.path.join(os.path.dirname(__file__), "../test_data/Csv")
    output_file = 'inference_output.txt'
    use_llm = False
    api_key = 'AIzaSyDdIEi4YhuO4zf0xiwQ5NIi20QQ4mQmlIk'
    checkpoint_path = 'best_model.pth'

    # prepare test videos with kornia center-crop + normalize transform
    inference_transform = CenterCropNormalize()
    vids = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.mp4')]
    ds = VideoDataset(vids, transform=inference_transform)
    dl=DataLoader(ds, batch_size=1, shuffle=False)

    # load only tensor weights to avoid FutureWarning
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # encoder and decoder configuration (same as training)
    densetcn_options = {
        'block_config': [4, 4, 4, 4],
        'growth_rate_set': [384, 384, 384, 384],
        'reduced_size': 512,
        'kernel_size_set': [3, 5, 7, 9],
        'dilation_size_set': [1, 2, 4, 8],
        'squeeze_excitation': True,
        'dropout': 0.2,
        'hidden_dim': 512,
    }
    mstcn_options = {
        'tcn_type': 'multiscale',
        'hidden_dim': 512,
        'num_channels': [384, 384, 384, 384],
        'kernel_size': [3, 5, 7, 9],
        'dropout': 0.2,
        'stride': 1,
        'width_mult': 1.0,
    }
    conformer_options = {
        'attention_dim': 512,
        'attention_heads': 8,
        'linear_units': 1024,
        'num_blocks': 8,
        'dropout_rate': 0.1,
        'positional_dropout_rate': 0.1,
        'attention_dropout_rate': 0.0,
        'cnn_module_kernel': 31,
    }
    dec_options = {
        'attention_dim': 512,
        'attention_heads': 8,
        'linear_units': 1024,
        'num_blocks': 4,
        'dropout_rate': 0.1,
        'positional_dropout_rate': 0.1,
        'self_attention_dropout_rate': 0.1,
        'src_attention_dropout_rate': 0.1,
        'normalize_before': True,
    }
    # compute vocabulary sizes
    # CTC vocabulary size includes blank and original tokens (exclude SOS and EOS entries)
    base_vocab_size = len(mapped_tokens) - 2
    # Full vocabulary size includes blank, tokens, SOS, and EOS
    full_vocab_size = len(mapped_tokens)
    # instantiate the model
    model = E2EVSR(
        encoder_type='conformer',
        ctc_vocab_size=base_vocab_size,
        dec_vocab_size=full_vocab_size,
        token_list=[idx2char[i] for i in range(full_vocab_size)],
        sos=sos_idx,
        eos=eos_idx,
        pad=0,
        enc_options={
            'densetcn_options': densetcn_options,
            'mstcn_options': mstcn_options,
            'conformer_options': conformer_options,
            'hidden_dim': conformer_options['attention_dim'],
        },
        dec_options=dec_options,
        ctc_weight=0.3,
        label_smoothing=0.2,
    )
    # load weights into model
    model.load_state_dict(ckpt['e2e_model_state_dict'], strict=False)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with open(output_file,'w',encoding='utf-8') as fout:
        sample_id = 0
        for frames, length, fname in dl:
            frames = frames.to(device)
            length = length.to(device)
            with torch.no_grad():
                seqs = model.transformer_greedy_search(frames, length)
            seq = seqs[0]
            text = indices_to_text(seq, idx2char)
            if use_llm:
                text = llm_cleanup(text, api_key=api_key)
            sample_id += 1
            # Unpack filename from batch tuple if needed
            if isinstance(fname, (list, tuple)):
                fname = fname[0]
            # Attempt to derive corresponding CSV path and extract reference sequence and metrics
            base, _ = os.path.splitext(fname)
            # Lookup CSV in test_data/Csv
            csv_path = os.path.join(csv_dir, base + '.csv')
            # Always write and print sample ID and prediction
            fout.write(f"Sample {sample_id}:\n")
            fout.write(f"Predicted text: {text}\n")
            print(f"Sample {sample_id} ({fname}):")
            print(f"Predicted text: {text}")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    ref_text = ''.join(df.word.tolist())
                    ref_indices = [mapped_tokens.get(c, mapped_tokens.get('', 0)) for c in ref_text]
                    cer, edit_dist = compute_cer(ref_indices, seq)
                    fout.write(f"Target text: {ref_text}\n")
                    fout.write(f"Edit distance: {edit_dist}\n")
                    fout.write(f"CER: {cer:.4f}\n\n")
                    print(f"Target text: {ref_text}")
                    print(f"Edit distance: {edit_dist}")
                    print(f"CER: {cer:.4f}\n")
                except Exception:
                    fout.write("Target text: [unavailable]\n\n")
                    print("Target text: [unavailable]\n")
            else:
                fout.write("Target text: [unavailable]\n\n")
                print("Target text: [unavailable]\n")
    print(f"Inference complete. Results in {output_file}") 