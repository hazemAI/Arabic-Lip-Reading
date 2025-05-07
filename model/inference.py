import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import cv2

def extract_label_tokens(csv_dir):
    # Reconstruct mapped_tokens and idx2char (same as training)
    diacritics = {'\u064B','\u064C','\u064D','\u064E','\u064F','\u0650','\u0651','\u0652','\u06E2'}
    tokens = set()
    for fname in os.listdir(csv_dir):
        if fname.endswith('.csv'):
            df = pd.read_csv(os.path.join(csv_dir, fname))
            for word in df.word:
                for ch in word:
                    if ch not in diacritics:
                        tokens.add(ch)
                    else:
                        tokens_list = list(tokens)
                        tokens_list[-1] += ch
    mapped = {c: i for i, c in enumerate(sorted(tokens, reverse=True), 1)}
    base_vocab = len(mapped) + 1
    sos_idx = base_vocab
    eos_idx = base_vocab + 1
    idx2char = {v:k for k,v in mapped.items()}
    idx2char[0] = ''
    idx2char[sos_idx] = '<sos>'
    idx2char[eos_idx] = '<eos>'
    return mapped, idx2char, sos_idx, eos_idx

# Reuse post-processing functions
from utils import indices_to_text

def clean_indices(seq, sos_idx, eos_idx):
    cleaned=[]
    for idx in seq:
        if idx==eos_idx:
            break
        if idx!=sos_idx:
            cleaned.append(idx)
    return cleaned

def collapse_repeats(seq, max_repeat=1):
    if not seq: return []
    collapsed=[seq[0]]
    count=1
    for x in seq[1:]:
        if x==collapsed[-1]:
            count+=1
            if count<=max_repeat:
                collapsed.append(x)
        else:
            collapsed.append(x)
            count=1
    return collapsed

def remove_consecutive_subsequences(seq):
    if not seq: return []
    res=[]; i=0; n=len(seq)
    while i<n:
        found=False
        max_L=(n-i)//2
        for L in range(max_L,0,-1):
            if seq[i:i+L]==seq[i+L:i+2*L]:
                res.extend(seq[i:i+L])
                i+=2*L; found=True; break
        if not found:
            res.append(seq[i]); i+=1
    return res

def prune_repeats(seq):
    prev=seq
    while True:
        nxt=remove_consecutive_subsequences(prev)
        if nxt==prev: return nxt
        prev=nxt

## Optional: LLM cleanup
try:
    import google.generativeai as genai
except ImportError:
    genai=None

def llm_cleanup(text, api_key=None, model_name='learnlm-2.0-flash-experimental'):
    if not genai or not api_key:
        return text
    genai.configure(api_key=api_key)
    prompt = f"Clean this Arabic lipread output: '{text}'. Remove gibberish and repeated parts."
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
        vp=self.paths[idx]
        # load frames
        frames=[]
        cap=cv2.VideoCapture(vp)
        cnt=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(cnt):
            cap.set(cv2.CAP_PROP_POS_FRAMES,i)
            ret,fr=cap.read()
            if not ret: break
            fr=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            pil=Image.fromarray(fr,'L')
            if self.transform: pil=self.transform(pil)
            frames.append(pil)
        t=torch.stack(frames).permute(1,0,2,3)
        return t, torch.tensor(t.size(1),dtype=torch.long), os.path.basename(vp)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--csv_dir', required=True)
    parser.add_argument('--output', default='inference_output.txt')
    parser.add_argument('--use_llm', action='store_true')
    parser.add_argument('--api_key', default=None)
    args=parser.parse_args()

    # load mapping
    mapped_tokens, idx2char, sos_idx, eos_idx = extract_label_tokens(args.csv_dir)
    # transforms same as training
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.41923218965530395, std=0.13392585515975952),
    ])
    # prepare test videos
    vids = [os.path.join(args.test_dir,f) for f in os.listdir(args.test_dir) if f.endswith('.mp4')]
    ds=VideoDataset(vids, transform=data_transform)
    dl=DataLoader(ds, batch_size=1, shuffle=False)

    # load model
    checkpoint=torch.load(args.checkpoint, map_location='cpu')
    from e2e_vsr import E2EVSR
    model=E2EVSR.load_from_checkpoint(checkpoint)
    model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device)

    with open(args.output,'w',encoding='utf-8') as fout:
        for frames, length, fname in dl:
            frames=frames.to(device)
            length=length.to(device)
            with torch.no_grad():
                hyps = model(frames, length)
            best_hyp=hyps[0]
            seq=best_hyp.yseq.cpu().tolist()
            # post-process
            cl=clean_indices(seq,sos_idx,eos_idx)
            cp=collapse_repeats(cl)
            pr=prune_repeats(cp)
            text = indices_to_text(pr, idx2char)
            if args.use_llm:
                text = llm_cleanup(text, api_key=args.api_key)
            fout.write(f"{fname}: {text}\n")
    print(f"Inference complete. Results in {args.output}") 