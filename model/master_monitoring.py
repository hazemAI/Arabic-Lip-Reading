# %% [markdown]
# # 1. Imports & Logging

# %%
import torch, os, cv2, gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from PIL import Image
from encoders.encoder_models_pretrained import Lipreading
from espnet.transformer.mask import subsequent_mask
from utils import *
import logging
from datetime import datetime
import traceback
from e2e_avsr import E2EAVSR
import wandb

os.makedirs('Logs', exist_ok=True)
log_filename = f'Logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    force=True 
)
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
# ## 3.1. List of Classes

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

classes = set()
for i in os.listdir('../Dataset/Csv (with Diacritics)'):
    file = '../Dataset/Csv (with Diacritics)/' + i
    label = extract_label(file)
    classes.update(label)

mapped_classes = {}
for i, c in enumerate(sorted(classes, reverse=True), 1):
    mapped_classes[c] = i

print(mapped_classes)

# %% [markdown]
# ## 3.2. Video Dataset Class

# %%
# Defining the video dataset class
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
        label = torch.tensor(list(map(lambda x: mapped_classes[x], extract_label(label_path))))
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
            frames = [self.transform(frame) for frame in frames] 
        frames = torch.stack(frames).permute(1, 0, 2, 3)
        return frames

# Defining data augmentation transforms for train, validation, and test
data_transforms = transforms.Compose([
    # transforms.CenterCrop(88),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.419232189655303955078125, std=0.133925855159759521484375),
])

# %% [markdown]
# ## 3.3. Load the dataset

# %%
# Load videos and labels
videos_dir = "../Dataset/Preprocessed_Video"
labels_dir = "../Dataset/Csv (with Diacritics)"
videos, labels = [], []
file_names = [file_name[:-4] for file_name in os.listdir(videos_dir)]
for file_name in file_names:
    videos.append(os.path.join(videos_dir, file_name + ".mp4"))
    labels.append(os.path.join(labels_dir, file_name + ".csv"))
    
# %% [markdown]
# ## 3.4. Split the dataset

# %%
# Split the dataset into training, validation, test sets
X_temp, X_test, y_temp, y_test = train_test_split(videos, labels, test_size=1903/2004, random_state=seed)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=21/101, random_state=seed)

# %% [markdown]
# ## 3.5. DataLoaders

# %%
# Defining the video dataloaders (train, validation, test)
train_dataset = VideoDataset(X_train, y_train, transform=data_transforms)
val_dataset = VideoDataset(X_val, y_val, transform=data_transforms)
test_dataset = VideoDataset(X_test, y_test, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, collate_fn=pad_packed_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, collate_fn=pad_packed_collate)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, collate_fn=pad_packed_collate)

# %% [markdown]
# # 4. Model Configuration

# %%
# Build vocabulary setup
base_vocab_size = len(mapped_classes) + 1  # +1 for blank token (0)
sos_token_idx = base_vocab_size  # This places SOS after all normal tokens
eos_token_idx = base_vocab_size + 1  # This places EOS after SOS
full_vocab_size = base_vocab_size + 2  # +2 for SOS and EOS tokens

# Build reverse mapping for decoding
idx2char = {v: k for k, v in mapped_classes.items()}
idx2char[0] = ""  # Blank token for CTC
idx2char[sos_token_idx] = "<sos>"  # SOS token
idx2char[eos_token_idx] = "<eos>"  # EOS token
print(f"Total vocabulary size: {full_vocab_size}")
print(f"SOS token index: {sos_token_idx}")
print(f"EOS token index: {eos_token_idx}")


# %% [markdown]
# ## 4.1 Temporal Encoder Options

# %%
# DenseTCN configuration (our default backbone)
densetcn_options = {
    'block_config': [2, 2, 2, 2],               # Number of layers in each dense block
    'growth_rate_set': [96, 96, 96, 96],        # Growth rate for each block
    'reduced_size': 256,                        # Reduced size between blocks
    'kernel_size_set': [3, 5, 7],               # Kernel sizes for multi-scale processing
    'dilation_size_set': [1, 2],                # Dilation rates for increasing receptive field
    'squeeze_excitation': True,                 # Whether to use SE blocks for channel attention
    'dropout': 0.1,
    'hidden_dim': 256,
}

# MSTCN configuration
mstcn_options = {
    'tcn_type': 'multiscale',
    'hidden_dim': 256,
    'num_channels': [96, 96, 96, 96],           # 4 layers with 96 channels each (divisible by 3)
    'kernel_size': [3, 5, 7],                   # 3 kernels for multi-scale processing
    'dropout': 0.1,
    'stride': 1,
    'width_mult': 1.0,
}

# Conformer configuration
conformer_options = {
    'attention_dim': 512,
    'attention_heads': 8,
    'linear_units': 2048,
    'num_blocks': 8,
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
# Step 1: Initialize the model first
print(f"Initializing model with {TEMPORAL_ENCODER} temporal encoder...")
logging.info(f"Initializing model with {TEMPORAL_ENCODER} temporal encoder")

if TEMPORAL_ENCODER == 'densetcn':
    model = Lipreading(
        densetcn_options=densetcn_options,
        hidden_dim=densetcn_options['hidden_dim'],
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
elif TEMPORAL_ENCODER == 'mstcn':
    model = Lipreading(
        tcn_options=mstcn_options,
        hidden_dim=mstcn_options['hidden_dim'],
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
elif TEMPORAL_ENCODER == 'conformer':
    model = Lipreading(
        conformer_options=conformer_options,
        hidden_dim=conformer_options['attention_dim'],
        num_classes=base_vocab_size,
        relu_type='swish'
    ).to(device)
else:
    raise ValueError(f"Unknown temporal encoder type: {TEMPORAL_ENCODER}")

print("Model initialized successfully.")

# Step 2: Load pretrained frontend weights
print("\nStep 4.2: Loading pretrained frontend weights...")
logging.info("Loading pretrained frontend weights")

pretrained_path = 'encoders/pretrained_visual_frontend.pth'
pretrained_weights = torch.load(pretrained_path, map_location=device)
print(f"Loaded pretrained weights from {pretrained_path}")

# Load weights into frontend
model.visual_frontend.load_state_dict(pretrained_weights['state_dict'], strict=False)
print("Successfully loaded pretrained weights")

# Flag to choose whether to fine-tune the frontend or freeze it
TRAIN_FRONTEND = False

# Conditionally freeze or fine-tune the frontend
if TRAIN_FRONTEND:
    print("Frontend parameters will be updated during training")
    logging.info("Frontend parameters will be updated during training")
else:
    for param in model.visual_frontend.parameters():
        param.requires_grad = False
    print("Frontend frozen - parameters will not be updated during training")
    logging.info("Successfully loaded and froze pretrained frontend")

# %% [markdown]
# ## 4.3 Decoder and Training Setup

# %%
# Step 4.3: Initialize the E2EAVSR end-to-end model
print("\nStep 4.3: Initializing E2EAVSR end-to-end model...")
# Determine hidden_dim for E2EAVSR based on the chosen temporal encoder
if TEMPORAL_ENCODER == 'densetcn':
    e2e_hidden_dim = densetcn_options['hidden_dim']
elif TEMPORAL_ENCODER == 'mstcn':
    e2e_hidden_dim = mstcn_options['hidden_dim']
elif TEMPORAL_ENCODER == 'conformer':
    e2e_hidden_dim = conformer_options['attention_dim']
else:
    raise ValueError(f"Unknown TEMPORAL_ENCODER: {TEMPORAL_ENCODER}")

e2e_model = E2EAVSR(
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
        'attention_dim': 512,
        'attention_heads': 8,
        'linear_units': 2048,
        'num_blocks': 4,
        'dropout_rate': 0.1,
        'positional_dropout_rate': 0.1,
        'self_attention_dropout_rate': 0.1,
        'src_attention_dropout_rate': 0.1,
        'normalize_before': True,
    },
    ctc_weight=0.3,
    label_smoothing=0.1,
    beam_size=20,
    length_bonus_weight=0.0
).to(device)


# Training parameters
initial_lr = 3e-4
total_epochs = 80
warmup_epochs = 8

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

# Initialize Weights & Biases for model weight and metric logging
wandb.init(project="arabic-lipreading-avsr", config={
    "learning_rate": initial_lr,
    "total_epochs": total_epochs,
    "warmup_epochs": warmup_epochs,
    "batch_size": train_loader.batch_size,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "betas": (0.9, 0.98),
    "eps": 1e-9,
    "temporal_encoder": TEMPORAL_ENCODER,
})
wandb.watch(e2e_model, log="all", log_freq=10)

print("Selected temporal encoder:", TEMPORAL_ENCODER)
print(model)
print(e2e_model)

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
            print("Warning: torch RNG state is not a tensor, creating a valid state")
            state['torch'] = torch.random.get_rng_state()
            
    except Exception as e:
        print(f"Warning: Error capturing RNG state: {str(e)}. Using default state.")
        logging.warning(f"Error capturing RNG state: {str(e)}. Using default state.")
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
        logging.info(f"Batch {batch_idx+1} - Input shape: {inputs.shape}")

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
                logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            if batch_idx % 3 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                logging.info(f"Memory cleared. Current GPU memory: {torch.cuda.memory_allocated()/1e6:.2f}MB")
                
        except Exception as e:
            logging.error(f"Error in training loop for batch {batch_idx}: {str(e)}") 
            logging.error(f"Error type: {type(e).__name__}")
            import traceback
            traceback_str = traceback.format_exc()
            logging.error(traceback_str)

            print(f"Error in batch {batch_idx}: {str(e)}")
            print(f"--- Skipping Batch {batch_idx+1} due to error ---")
            # Ensure gradients are cleared if error happened after loss calculation but before optimizer step
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            continue # Skip this batch
            raise e

    return running_loss / len(train_loader) if len(train_loader) > 0 else 0.0


def evaluate_model(data_loader, ctc_weight=0.3, epoch=None, print_samples=True):
    """
    Evaluate the model on the given data loader using E2EAVSR's built-in beam search.
    """
    e2e_model.eval()

    # Track statistics
    total_cer = 0
    sample_count = 0
    all_predictions = []

    # Determine if we should print samples in this epoch
    show_samples = (epoch is None or epoch == 0 or (epoch+1) % 5 == 0) and print_samples
    max_samples_to_print = 10

    # Use E2EAVSR's beam_search directly
    bs = e2e_model.beam_search

    # Process all batches in the test loader
    with torch.no_grad():
        for i, (inputs, input_lengths, labels_flat, label_lengths) in enumerate(data_loader):
            inputs = inputs.to(device)
            input_lengths = input_lengths.to(device)
            labels_flat = labels_flat.to(device)
            label_lengths = label_lengths.to(device)
            
            # Run raw encoder and unpack hidden features if tuple is returned
            enc_out = model(inputs, input_lengths)
            encoder_features = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            
            # Set output_lengths to match the actual encoder output length
            output_lengths = torch.full((encoder_features.size(0),), encoder_features.size(1), dtype=torch.long, device=device)
            
            logging.info(f"\nRunning hybrid CTC/Attention decoding for batch {i+1}...")
            if show_samples and i == 0:
                print(f"\nRunning hybrid CTC/Attention decoding for validation...")
            
            try:
                logging.info(f"Encoder features shape: {encoder_features.shape}")
                
                # Run beam search for detailed samples via the model's inference API
                all_beam_results = e2e_model(inputs, input_lengths)
                # Extract the best hypothesis per utterance
                all_nbest_hyps = [hyps_b[0] for hyps_b in all_beam_results]
                
                logging.info(f"Hybrid decoding completed for batch {i+1}")
                logging.info(f"Received {len(all_nbest_hyps)} hypotheses sets")
                
                # Process each batch item
                for b in range(encoder_features.size(0)):
                    logging.info(f"\nProcessing batch item {b+1}/{encoder_features.size(0)}")
                    sample_count += 1
                    
                    if b < len(all_nbest_hyps):
                        # Extract from Hypothesis object
                        hyp = all_nbest_hyps[b]
                        score = float(hyp.score)
                        logging.info(f"Found beam hypothesis for item {b+1} with score {score:.4f}")
                        pred_indices = hyp.yseq.cpu().numpy()
                    
                    if len(pred_indices) == 0:
                        logging.info("WARNING: Prediction sequence is empty!")
                    
                    # Get target indices
                    start_idx = sum(label_lengths[:b].cpu().tolist()) if b > 0 else 0
                    end_idx = start_idx + label_lengths[b].item()
                    target_idx = labels_flat[start_idx:end_idx].cpu().numpy()

                    # Log debug information for reference and hypothesis tokens
                    logging.info(f"Debug - Reference tokens ({len(target_idx)} tokens): {target_idx}")
                    logging.info(f"Debug - Hypothesis tokens ({len(pred_indices)} tokens): {pred_indices}")
                    
                    # Convert indices to text
                    pred_text = indices_to_text(pred_indices, idx2char)
                    target_text = indices_to_text(target_idx, idx2char)
                    
                    # Compute CER and edit distance on token indices
                    cer, edit_dist = compute_cer(target_idx.tolist(), pred_indices.tolist())
                    
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
                del encoder_features
                
                # Periodically clear cache
                if i % 3 == 0:  # Every 3 batches
                    gc.collect()
                    torch.cuda.empty_cache()
                    logging.info(f"Memory cleared. Current GPU memory: {torch.cuda.memory_allocated()/1e6:.2f}MB")
            
            except Exception as e:
                logging.error(f"Error during hybrid decoding: {str(e)}")
                logging.error(traceback.format_exc())
                raise
        
        # Write summary statistics
        n_samples = len(data_loader.dataset)
        avg_cer = total_cer / n_samples
        
        # Always print summary statistics to console
        print("\n=== Summary Statistics ===")
        print(f"Total samples: {n_samples}")
        print(f"Average CER: {avg_cer:.4f}")
        
        # Log summary statistics as well
        logging.info("\n=== Summary Statistics ===")
        logging.info(f"Total samples: {n_samples}")
        logging.info(f"Average CER: {avg_cer:.4f}")

    return avg_cer

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
def train_model(ctc_weight=0.3, checkpoint_path=None):
    best_val_loss = float('inf')
    start_epoch = 0
    rng_state = get_rng_state()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check model architecture compatibility
            model_state_dict = model.state_dict()
            checkpoint_model_state_dict = checkpoint['model_state_dict']
            if set(model_state_dict.keys()) != set(checkpoint_model_state_dict.keys()):
                missing_keys = [k for k in model_state_dict.keys() if k not in checkpoint_model_state_dict]
                unexpected_keys = [k for k in checkpoint_model_state_dict.keys() if k not in model_state_dict]
                error_msg = "Model architecture mismatch detected!\n"
                if missing_keys:
                    error_msg += f"Missing keys in checkpoint: {missing_keys}\n"
                if unexpected_keys:
                    error_msg += f"Unexpected keys in checkpoint: {unexpected_keys}\n"
                error_msg += "Cannot proceed with training due to incompatible architecture."
                print(error_msg)
                logging.error(error_msg)
                raise RuntimeError("Model architecture mismatch. Training aborted to prevent corruption.")
            
            # Load the state dict
            model.load_state_dict(checkpoint_model_state_dict)
            
            # Check transformer decoder architecture compatibility
            decoder_state_dict = e2e_model.state_dict()
            checkpoint_decoder_state_dict = checkpoint['transformer_decoder_state_dict']
            
            if set(decoder_state_dict.keys()) != set(checkpoint_decoder_state_dict.keys()):
                missing_keys = [k for k in decoder_state_dict.keys() if k not in checkpoint_decoder_state_dict]
                unexpected_keys = [k for k in checkpoint_decoder_state_dict.keys() if k not in decoder_state_dict]
                error_msg = "Transformer decoder architecture mismatch detected!\n"
                if missing_keys:
                    error_msg += f"Missing keys in checkpoint: {missing_keys}\n"
                if unexpected_keys:
                    error_msg += f"Unexpected keys in checkpoint: {unexpected_keys}\n"
                error_msg += "Cannot proceed with training due to incompatible architecture."
                print(error_msg)
                logging.error(error_msg)
                raise RuntimeError("Transformer decoder architecture mismatch. Training aborted to prevent corruption.")
            
            # Load the decoder state dict
            e2e_model.load_state_dict(checkpoint_decoder_state_dict)
            print("Successfully loaded checkpoint")
            
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
                    print("RNG state restored successfully")
                    logging.info("RNG state restored successfully")
                except Exception as e:
                    print(f"Warning: Could not restore RNG state: {e}. Continuing with current RNG state.")
                    logging.warning(f"Could not restore RNG state: {e}")
                
            print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
            logging.info(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            logging.error(f"Error loading checkpoint: {str(e)}")
            print("Aborting training due to checkpoint loading failure.")
            raise
        
    else:
        if checkpoint_path:
            print(f"Checkpoint file {checkpoint_path} not found. Starting training from scratch.")
            logging.info(f"Checkpoint file {checkpoint_path} not found. Starting training from scratch.")
        else:
            print("No checkpoint specified. Starting training from scratch.")
            logging.info("No checkpoint specified. Starting training from scratch.")
    
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
        # Then compute decoding metrics (CER) via beam search
        val_cer = evaluate_model(val_loader, epoch=epoch)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info(f"GPU memory after evaluation: {torch.cuda.memory_allocated()/1e6:.2f}MB")
        
        logging.info(
            f"Epoch {epoch + 1}/{total_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}"
        )
        
        # Print summary every epoch to console
        print(
            f"Epoch {epoch + 1}/{total_epochs} - Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}"
        )
        # log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_cer": val_cer,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })
        
        
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Update the RNG state before saving
            rng_state = get_rng_state()
            
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'transformer_decoder_state_dict': e2e_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'rng_state': rng_state,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            logging.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Force synchronize CUDA operations and clear memory after saving
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        
        # Save best model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'transformer_decoder_state_dict': e2e_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'rng_state': rng_state,
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
            print(f"New best model saved with validation loss: {val_loss:.4f}")
            logging.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Log histograms
        for name, param in e2e_model.named_parameters():
            wandb.log(
                {f"histograms/{name}": wandb.Histogram(param.detach().cpu().numpy())},
                step=epoch
            )
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint saved to: checkpoint_epoch_{total_epochs}.pth")
    print(f"Best model saved to: best_model.pth")

    
if __name__ == '__main__':
    train_model() 

