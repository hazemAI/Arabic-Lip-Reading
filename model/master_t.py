# %% [markdown]
# # 1. Imports & Logging

# %%
import torch, os, cv2, gc
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
from utils import *
import logging
from datetime import datetime
import traceback
from e2e_vsr import E2EVSR
import kornia.augmentation as K

os.makedirs('Logs', exist_ok=True)
log_filename = f'Logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(message)s',
    encoding='utf-8',
    force=True 
)

# Helper to print and log in one call
def log_print(msg):
    print(msg)
    logging.info(msg)

# %% [markdown]
# # 2. Initialize the seed and the device

# %%
# Setting the seed for reproducibility
seed = 0
def reset_seed():
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% [markdown]
# # 3. Dataset preparation

# %% [markdown]
# ## 3.1. List of tokens

# %%
def extract_label(file):
    label = []
    diacritics = {
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u06E2',  # Small High meem
    }

    sentence = pd.read_csv(file)
    for word in sentence.word:
        for char in word:
            if char not in diacritics:
                label.append(char)
            else:
                label[-1] += char

    return label

tokens = set()
for i in os.listdir('../Dataset/Csv (with Diacritics)'):
    file = '../Dataset/Csv (with Diacritics)/' + i
    label = extract_label(file)
    tokens.update(label)

mapped_tokens = {}
for i, c in enumerate(sorted(tokens, reverse=True), 1):
    mapped_tokens[c] = i

log_print(mapped_tokens)
# %% [markdown]
# ## 3.2. Video Dataset Class
# %%
# Video augmentation constants and transforms
MEAN = 0.41923218965530395
STD  = 0.13392585515975952

# Video augmentation class using Kornia VideoSequential with assertions
class VideoAugmentation:
    def __init__(self, is_train=True, crop_size=(88, 88)):
        if is_train:
            self.aug = K.VideoSequential(
                K.RandomCrop(crop_size, p=1.0),
                data_format="BCTHW", same_on_frame=True,
            )
        else:
            self.aug = K.VideoSequential(
                K.CenterCrop(crop_size, p=1.0),
                data_format="BCTHW", same_on_frame=True,
            )
        self.crop_size = crop_size

    def __call__(self, pil_frames):
        # Convert list of PIL images to tensor sequence
        frame_tensors = [transforms.ToTensor()(img) for img in pil_frames]
        video = torch.stack(frame_tensors, dim=0)      # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)              # (C, T, H, W)
        video_batch = video.unsqueeze(0)               # (1, C, T, H, W)
        # Apply augmentations
        augmented = self.aug(video_batch)
        augmented = augmented.squeeze(0)               # (C, T, H, W)
        # Assertions for shape and validity
        C, T, H, W = augmented.shape
        assert C == 1, f"Expected channel=1, got {C}"
        assert (H, W) == self.crop_size, f"Expected spatial size {self.crop_size}, got {(H,W)}"
        assert not torch.isnan(augmented).any(), "NaNs in augmented clip!"
        assert not torch.isinf(augmented).any(), "Infs in augmented clip!"
        # Normalize channels
        augmented = (augmented - MEAN) / STD
        return augmented

# Instantiate augmenters for datasets
train_transform = VideoAugmentation(is_train=True)
val_transform   = VideoAugmentation(is_train=False)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, label_paths, transform=None):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label_path = self.label_paths[index]
        frames = self.load_frames(video_path=video_path)
        label = torch.tensor(list(map(lambda x: mapped_tokens[x], extract_label(label_path))))
        input_length = torch.tensor(frames.size(1), dtype=torch.long)
        label_length = torch.tensor(len(label), dtype=torch.long)
        return frames, input_length, label, label_length
    
    def load_frames(self, video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_pil = Image.fromarray(frame, 'L')
                frames.append(frame_pil)

        if self.transform is not None:
            # Apply video-level transformation
            video = self.transform(frames)
        else:
            # Fallback: per-frame ToTensor + Normalize
            frame_tensors = []
            for img in frames:
                t = transforms.ToTensor()(img)
                t = transforms.Normalize(mean=[MEAN], std=[STD])(t)
                frame_tensors.append(t)
            video = torch.stack(frame_tensors).permute(1, 0, 2, 3)
        return video


# %% [markdown]
# ## 3.3. Load the dataset

# %%
# Load videos and labels from all original and augmented video folders
dataset_dir = "../Dataset"
labels_dir = os.path.join(dataset_dir, "Csv (with Diacritics)")
videos, labels = [], []
# Specify exactly which preprocessed video folders to include
preprocessed_dirs = [
    "Preprocessed_Video",
]
video_dirs = sorted([
    os.path.join(dataset_dir, d)
    for d in preprocessed_dirs
    if os.path.isdir(os.path.join(dataset_dir, d))
])
for vdir in video_dirs:
    for fname in sorted(os.listdir(vdir)):
        if not fname.lower().endswith('.mp4'):
            continue
        stem = os.path.splitext(fname)[0]
        # extract base ID before augmentation suffix
        base = stem.split('_')[0]
        videos.append(os.path.join(vdir, fname))
        labels.append(os.path.join(labels_dir, base + ".csv"))
log_print(f"Loaded {len(videos)} video-label pairs")

# %% [markdown]
# ## 3.4. Split the dataset

# %%
# Split the dataset into training and validation
X_tmp, X_test, y_tmp, y_test = train_test_split(videos, labels, test_size=1984/2004, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=2/1984, random_state=42)

# %% [markdown]
# ## 3.5. DataLoaders

# %%
# Defining the video dataloaders (train, validation)
train_dataset = VideoDataset(X_train, y_train, transform=train_transform)
val_dataset = VideoDataset(X_val, y_val, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, collate_fn=pad_packed_collate)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, collate_fn=pad_packed_collate)

log_print(f"Number of train samples: {len(train_loader.dataset)}")
log_print(f"Number of validation samples: {len(val_loader.dataset)}")

# %% [markdown]
# # 4. Model Configuration

# %%
# Build vocabulary setup
base_vocab_size = len(mapped_tokens) + 1  # +1 for blank token (0)
sos_token_idx = base_vocab_size  # This places SOS after all normal tokens
eos_token_idx = base_vocab_size + 1  # This places EOS after SOS
full_vocab_size = base_vocab_size + 2  # +2 for SOS and EOS tokens

# Build reverse mapping for decoding
idx2char = {v: k for k, v in mapped_tokens.items()}
idx2char[0] = ""  # Blank token for CTC
idx2char[sos_token_idx] = "<sos>"  # SOS token
idx2char[eos_token_idx] = "<eos>"  # EOS token
log_print(f"Total vocabulary size: {full_vocab_size}")
log_print(f"SOS token index: {sos_token_idx}")
log_print(f"EOS token index: {eos_token_idx}")


# %% [markdown]
# ## 4.1 Temporal Encoder Options

# %%
# DenseTCN configuration (our default backbone)
densetcn_options = {
    'block_config': [4, 4, 4, 4],               # Number of layers in each dense block
    'growth_rate_set': [384, 384, 384, 384],    # Growth rate for each block
    'reduced_size': 512,                        # Reduced size between blocks
    'kernel_size_set': [3, 5, 7, 9],            # Kernel sizes for multi-scale processing
    'dilation_size_set': [1, 2, 4, 8],          # Dilation rates for increasing receptive field
    'squeeze_excitation': True,                 # Whether to use SE blocks for channel attention
    'dropout': 0.2,
    'hidden_dim': 512,
}

# MSTCN configuration
mstcn_options = {
    'tcn_type': 'multiscale',
    'hidden_dim': 512,
    'num_channels': [384, 384, 384, 384],       # 4 layers with N channels each (divisible by 3)
    'kernel_size': [3, 5, 7, 9],                   
    'dropout': 0.2,
    'stride': 1,
    'width_mult': 1.0,
}

# Conformer configuration
conformer_options = {
    'attention_dim': 768,
    'attention_heads': 12,
    'linear_units': 3072,
    'num_blocks': 12,
    'dropout_rate': 0.1,
    'positional_dropout_rate': 0.1,
    'attention_dropout_rate': 0.0,
    'cnn_module_kernel': 31
}


# Choose temporal encoder type: 'densetcn', 'mstcn', or 'conformer'
TEMPORAL_ENCODER = 'conformer'

# %% [markdown]
# ## 4.2 Model Initialization and Pretrained Frontend

# %%
# Initialize the E2EVSR end-to-end model
log_print("\nInitializing E2EVSR end-to-end model...")
# Determine hidden_dim for E2EVSR based on the chosen temporal encoder
if TEMPORAL_ENCODER == 'densetcn':
    e2e_hidden_dim = densetcn_options['hidden_dim']
elif TEMPORAL_ENCODER == 'mstcn':
    e2e_hidden_dim = mstcn_options['hidden_dim']
elif TEMPORAL_ENCODER == 'conformer':
    e2e_hidden_dim = conformer_options['attention_dim']
else:
    raise ValueError(f"Unknown TEMPORAL_ENCODER: {TEMPORAL_ENCODER}")

e2e_model = E2EVSR(
    encoder_type=TEMPORAL_ENCODER,
    ctc_vocab_size=base_vocab_size,
    dec_vocab_size=full_vocab_size,
    token_list=[idx2char[i] for i in range(full_vocab_size)],
    sos=sos_token_idx,
    eos=eos_token_idx,
    pad=0,
    enc_options={
        'densetcn_options': densetcn_options,
        'mstcn_options': mstcn_options,
        'conformer_options': conformer_options,
        'hidden_dim': e2e_hidden_dim,
    },
    dec_options={
        'attention_dim': 768,
        'attention_heads': 12,
        'linear_units': 3072,
        'num_blocks': 6,
        'dropout_rate': 0.1,
        'positional_dropout_rate': 0.1,
        'self_attention_dropout_rate': 0.1,
        'src_attention_dropout_rate': 0.1,
        'normalize_before': True,
    },
    ctc_weight=0.3,
    label_smoothing=0.2,
).to(device)

# # Load mpc001 checkpoint for fine-tuning
# mpc_checkpoint_path = '/kaggle/working/Arabic-Lip-Reading/model/vsr_trlrs2lrs3vox2avsp_base.pth'
# log_print(f"Loading mpc001 checkpoint from {mpc_checkpoint_path}")
# mpc_state = torch.load(mpc_checkpoint_path, map_location=device)
# mpc_sd = mpc_state.get('state_dict', mpc_state)

# # Map frontend.trunk -> frontend.resnet_trunk
# mapped_sd = {}
# for k, v in mpc_sd.items():
#     if k.startswith('frontend.trunk'):
#         new_k = 'frontend.resnet_trunk' + k[len('frontend.trunk'):]
#     else:
#         new_k = k
#     mapped_sd[new_k] = v

# # Filter only compatible layers: frontend, encoder.*, decoder.* (excluding embed/output_layer) with matching shapes
# current_sd = e2e_model.state_dict()
# filtered_sd = {}
# for k, v in mapped_sd.items():
#     if k not in current_sd:
#         continue
#     if not (k.startswith('frontend') or k.startswith('proj_encoder') or k.startswith('encoder.') or k.startswith('decoder.')):
#         continue
#     if k.startswith('decoder.embed') or k.startswith('decoder.output_layer'):
#         continue
#     if current_sd[k].shape != v.shape:
#         continue
#     filtered_sd[k] = v

# load_res = e2e_model.load_state_dict(filtered_sd, strict=False)
# log_print(f"Loaded layers from mpc001 checkpoint, missing keys: {load_res.missing_keys}, unexpected keys: {load_res.unexpected_keys}")

# Training parameters
initial_lr = 3e-4
total_epochs = 75
warmup_epochs = 5

# Initialize AdamW optimizer with weight decay on the E2E model
optimizer = optim.AdamW(
    e2e_model.parameters(),
    lr=initial_lr,
    weight_decay=0.01,
    betas=(0.9, 0.98),
    eps=1e-9
)
# Setup WarmupCosineScheduler for per-step LR scheduling
steps_per_epoch = len(train_loader)
scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch)

log_print("Selected temporal encoder: " + TEMPORAL_ENCODER)
log_print(repr(e2e_model))
# %% [markdown]
# # 5. Training and Evaluation

# %%
def get_rng_state():
    state = {}
    try:
        state['torch'] = torch.get_rng_state()
        state['numpy'] = np.random.get_state()
        if torch.cuda.is_available():
            state['cuda'] = torch.cuda.get_rng_state()
        else:
            state['cuda'] = None
        
        # Validate RNG state types
        if not isinstance(state['torch'], torch.Tensor):
            log_print("Warning: torch RNG state is not a tensor, creating a valid state")
            state['torch'] = torch.random.get_rng_state()
            
    except Exception as e:
        log_print(f"Warning: Error capturing RNG state: {str(e)}. Using default state.")
        # Create minimal valid state
        state = {
            'torch': torch.random.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }
    return state

def set_rng_state(state):
    # Restore CPU RNG state
    if 'torch' in state and state['torch'] is not None:
        cpu_state = state['torch']
        # Convert to proper ByteTensor class on CPU
        cpu_state = cpu_state.cpu().type(torch.ByteTensor)
        torch.set_rng_state(cpu_state)

    # Restore NumPy RNG state
    if 'numpy' in state and state['numpy'] is not None:
        np.random.set_state(state['numpy'])

    # Restore CUDA RNG state
    if torch.cuda.is_available() and 'cuda' in state and state['cuda'] is not None:
        cuda_state = state['cuda']
        # Convert to proper ByteTensor class on CPU
        cuda_state = cuda_state.cpu().type(torch.ByteTensor)
        torch.cuda.set_rng_state(cuda_state)

def train_one_epoch():
    running_loss = 0.0
    e2e_model.train()

    for batch_idx, (inputs, input_lengths, labels_flat, label_lengths) in enumerate(train_loader):
        # Print input shape for debugging
        logging.info(f"\nBatch {batch_idx+1} - Input shape: {inputs.shape}")

        inputs = inputs.to(device)
        input_lengths = input_lengths.to(device)
        labels_flat = labels_flat.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)  

        try:
            # End-to-end forward (CTC+Attention) and backward
            out = e2e_model(inputs, input_lengths, ys=labels_flat, ys_lengths=label_lengths)
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logging.info(f"Batch {batch_idx+1}, Loss: {loss.item():.4f}")

            if batch_idx % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                logging.info(f"Memory cleared. Current GPU memory: {torch.cuda.memory_allocated()/1e6:.2f}MB")
                
        except Exception as e:
            log_print(f"Error in training loop for batch {batch_idx}: {str(e)}") 
            log_print(f"Error type: {type(e).__name__}")
            import traceback
            traceback_str = traceback.format_exc()
            log_print(traceback_str)

            log_print(f"Error in batch {batch_idx}: {str(e)}")
            log_print(f"--- Skipping Batch {batch_idx+1} due to error ---")
            # Ensure gradients are cleared if error happened after loss calculation but before optimizer step
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue # Skip this batch
            raise e

    return running_loss / len(train_loader) if len(train_loader) > 0 else 0.0


def evaluate_model(data_loader, epoch=None, print_samples=True):
    """
    Evaluate the model on the given data loader using greedy decoding.
    """
    e2e_model.eval()

    # Track statistics
    total_cer = 0
    sample_count = 0
    all_predictions = []

    # Determine if we should print samples in this epoch
    show_samples = (epoch is None or epoch == 0 or (epoch+1) % 5 == 0) and print_samples
    max_samples_to_print = 10

    # Inference mode: transformer greedy decoding only
    mode = 'transformer_greedy'

    # Process all batches in the test loader
    with torch.no_grad():
        for i, (inputs, input_lengths, labels_flat, label_lengths) in enumerate(data_loader):
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            labels_flat = labels_flat.to(device)
            label_lengths = label_lengths.to(device)
            

            if show_samples and i == 0:
                log_print(f"\nRunning greedy decoding for validation...")
            
            try:
                # logging.info(f"Encoder features shape: {encoder_features.shape}")
                
                # Run greedy decoding
                all_results = e2e_model.transformer_greedy_search(inputs, input_lengths)
                
                logging.info(f"Greedy decoding completed for batch {i+1}")
                logging.info(f"Received {len(all_results)} result sequences using mode {mode}")
                
                # Process each batch item
                for b in range(label_lengths.size(0)):
                    logging.info(f"\nProcessing batch item {b+1}/{label_lengths.size(0)}")
                    sample_count += 1
                    
                    if b < len(all_results):
                        # Get predicted token indices
                        pred_indices = all_results[b]
                    
                    if len(pred_indices) == 0:
                        log_print("WARNING: Prediction sequence is empty!")
                    
                    # Get target indices
                    start_idx = sum(label_lengths[:b].cpu().tolist()) if b > 0 else 0
                    end_idx = start_idx + label_lengths[b].item()
                    target_idx = labels_flat[start_idx:end_idx].cpu().numpy()

                    # Log debug information for reference and hypothesis tokens
                    logging.info(f"Reference tokens ({len(target_idx)} tokens): {target_idx}")
                    logging.info(f"Hypothesis tokens ({len(pred_indices)} tokens): {pred_indices}")
                    
                    # Reference sequence
                    ref_seq = target_idx.tolist()
                    # Direct greedy output without cleaning
                    cleaned_seq = list(pred_indices)
                    
                    # compute CER and edit distance on cleaned sequence
                    cer, edit_dist = compute_cer(ref_seq, cleaned_seq)
                    pred_text = indices_to_text(cleaned_seq, idx2char)
                    
                    target_text = indices_to_text(target_idx, idx2char)
                    
                    # Log using the filtered best sequence
                    # Update statistics
                    total_cer += cer
                    
                    # Store prediction details
                    all_predictions.append({
                        'sample_id': sample_count,
                        'pred_text': pred_text,
                        'target_text': target_text,
                        'cer': cer,
                        'edit_distance': edit_dist,
                    })
                    
                    # Log complete info
                    logging.info("-" * 50)
                    logging.info(f"Sample {sample_count}:")
                    try:
                        logging.info(f"Predicted text: {pred_text}")
                        logging.info(f"Target text: {target_text}")
                    except UnicodeEncodeError:
                        logging.info("Predicted text: [Contains characters that can't be displayed in console]")
                        logging.info("Target text: [Contains characters that can't be displayed in console]")
                        logging.info(f"Predicted indices: {pred_indices}")
                        logging.info(f"Target indices: {target_idx}")
                        
                    logging.info(f"Edit distance: {edit_dist}")
                    logging.info(f"CER: {cer:.4f}")
                    logging.info("-" * 50)
                    
                    # Print to console if this is a sample we should show
                    if show_samples and sample_count <= max_samples_to_print:
                        print("-" * 50)
                        print(f"Sample {sample_count}:")
                        try:
                            print(f"Predicted text: {pred_text}")
                            print(f"Target text: {target_text}")
                        except UnicodeEncodeError:
                            print("Predicted text: [Contains characters that can't be displayed in console]")
                            print("Target text: [Contains characters that can't be displayed in console]")
                            
                        print(f"Edit distance: {edit_dist}")
                        print(f"CER: {cer:.4f}")
                        print("-" * 50)

                # Clean up tensors
                del all_results #, encoder_features, enc_out
                
                # Periodically clear cache
                if i % 3 == 0:  # Every 3 batches
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info(f"Memory cleared. Current GPU memory: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            
            except Exception as e:
                log_print(f"Error during greedy decoding: {str(e)}")
                log_print(traceback.format_exc())
                raise

            del inputs, input_lengths, labels_flat, label_lengths
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info(f"Memory cleared. Current GPU memory: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        
        # Write summary statistics
        n_samples = len(data_loader.dataset)
        avg_cer = total_cer / n_samples
        
        # Always print summary statistics to console
        log_print("\n=== Summary Statistics ===")
        log_print(f"Total samples: {n_samples}")
        log_print(f"Average CER: {avg_cer:.4f}\n")
        

# --------------------------------------------------------------------------
def evaluate_loss(data_loader):
    """
    Compute average CTC+Attention loss on dev set with teacher forcing.
    """
    e2e_model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, input_lengths, labels_flat, label_lengths in data_loader:
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            labels_flat = labels_flat.to(device)
            label_lengths = label_lengths.to(device)
            out = e2e_model(
                inputs, input_lengths,
                ys=labels_flat, ys_lengths=label_lengths
            )
            running_loss += out['loss'].item()
    return running_loss / len(data_loader) if len(data_loader) > 0 else 0.0

# %%
def train_model(ctc_weight=0.3, ckpt_path=None):
    best_val_loss = float('inf')
    start_epoch = 0
    rng_state = get_rng_state()
    
    # Load checkpoint if provided
    if ckpt_path and os.path.exists(ckpt_path):
        log_print(f"Loading checkpoint from {ckpt_path}...")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            
            # Load E2E model checkpoint non-strictly (ignoring mismatched keys)
            dec_res = e2e_model.load_state_dict(
                checkpoint['e2e_model_state_dict'], strict=False)
            log_print(f"Loaded e2e_model checkpoint (non-strict): missing {dec_res.missing_keys}, unexpected {dec_res.unexpected_keys}")
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Update training state
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Restore RNG state if available
            if 'rng_state' in checkpoint:
                try:
                    set_rng_state(checkpoint['rng_state'])
                    # Success
                    log_print("RNG state restored successfully")
                except Exception as e:
                    log_print(f"Warning: Could not restore RNG state: {e}. Continuing with current RNG state.")
            
            log_print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        
        except Exception as e:
            log_print(f"Error loading checkpoint: {str(e)}")
            log_print("Aborting training due to checkpoint loading failure.")
            raise
        
    else:
        if ckpt_path:
            log_print(f"Checkpoint file {ckpt_path} not found. Starting training from scratch.")
        else:
            log_print("No checkpoint specified. Starting training from scratch.")
    
    print(f"Starting training for {total_epochs} epochs")
    print(f"Logs will be saved to {log_filename}")
    print(f"Checkpoints will be saved every 10 epochs")
    print("-" * 50)
    
    for epoch in range(start_epoch, total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs} - Training...")
        epoch_loss = train_one_epoch()
    
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"GPU memory after training: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        
        print(f"Epoch {epoch + 1}/{total_epochs} - Evaluating...")
        # First compute validation loss under teacher forcing
        val_loss = evaluate_loss(val_loader)
        # Then compute decoding metrics (CER) via greedy decoding
        evaluate_model(val_loader, epoch=epoch)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"GPU memory after evaluation: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        
        log_print(
            f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Update the RNG state before saving
            rng_state = get_rng_state()
            
            ckpt_path = f'ckpt_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'e2e_model_state_dict': e2e_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'rng_state': rng_state,
                'best_val_loss': best_val_loss
            }, ckpt_path)
            log_print(f"Checkpoint saved to {ckpt_path}")
            
            # Force synchronize CUDA operations and clear memory after saving
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            # Delete old checkpoints, keep only last 3
            checkpoint_files = sorted(
                [f for f in os.listdir('.') if f.startswith('ckpt_') and f.endswith('.pth')],
                key=lambda x: int(x[len('ckpt_'):-4])
            )
            if len(checkpoint_files) > 3:
                for old_ckpt in checkpoint_files[:-3]:
                    try:
                        os.remove(old_ckpt)
                        log_print(f"Deleted old checkpoint {old_ckpt}")
                    except Exception as e:
                        log_print(f"Error deleting old checkpoint {old_ckpt}: {e}")
        
        # Save best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'e2e_model_state_dict': e2e_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'rng_state': rng_state,
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
            log_print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    log_print("\nTraining completed!")
    log_print(f"Best validation loss: {best_val_loss:.4f}")
    log_print(f"Final checkpoint saved to: ckpt_{total_epochs}.pth")
    log_print(f"Best model saved to: best_model.pth")

    
if __name__ == '__main__':
    train_model() 

