import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.encoder_models import VisualTemporalEncoder
from espnet.decoder.transformer_decoder import TransformerDecoder
from espnet.transformer.mask import subsequent_mask
from espnet.transformer.add_sos_eos import add_sos_eos
from espnet.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets_utils import make_non_pad_mask
from espnet.ctc import CTC
from torch.nn.utils.rnn import pad_sequence
from espnet.e2e_asr_conformer import E2E as BaseE2E
import logging

class E2EVSR(BaseE2E):
    """
    End-to-end AVSR system combining frontend, optional temporal encoder,
    CTC head, and transformer decoder with label-smoothing using greedy decoding.
    """
    def __init__(
        self,
        encoder_type,
        ctc_vocab_size,
        dec_vocab_size,
        token_list,
        sos,
        eos,
        pad,
        enc_options,
        dec_options,
        ctc_weight=0.3,
        label_smoothing=0.2,
    ):
        # 1) initialize BaseE2E: sets up CTC prefix scorer, LabelSmoothingLoss, beam search, etc.
        super().__init__(odim=dec_vocab_size, modality='video',
                         ctc_weight=ctc_weight, ignore_id=pad)

        # 2) Build one VisualTemporalEncoder model and extract its components based on chosen encoder_type
        tcn_opt = enc_options.get('mstcn_options', {}) if encoder_type == 'mstcn' else {}
        dense_opt = enc_options.get('densetcn_options', {}) if encoder_type == 'densetcn' else {}
        conf_opt = enc_options.get('conformer_options', {}) if encoder_type == 'conformer' else {}
        vt_model = VisualTemporalEncoder(
            tcn_options=tcn_opt,
            densetcn_options=dense_opt,
            conformer_options=conf_opt,
            hidden_dim=enc_options['hidden_dim'],
            num_tokens=ctc_vocab_size,
            relu_type=enc_options.get('relu_type', 'swish'),
        )
        self.frontend = vt_model.visual_frontend
        self.proj_encoder = vt_model.adapter
        self.encoder = vt_model.encoder
        self.encoder_type = encoder_type  # remember encoder type for dispatch

        # Override BaseE2E's CTC module to match decoder vocabulary size and our encoder hidden dim
        self.ctc = CTC(dec_vocab_size, enc_options['hidden_dim'], dropout_rate=0.0, reduce=True)

        # 3) drop BaseE2E's conformer-only projection if you want, since you've set proj_encoder

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=pad, zero_infinity=True)
        self.att_loss = LabelSmoothingLoss(
            size=dec_vocab_size,
            padding_idx=pad,
            smoothing=label_smoothing,
        )
        # Transformer decoder
        self.decoder = TransformerDecoder(
            odim=dec_vocab_size,
            attention_dim=dec_options['attention_dim'],
            attention_heads=dec_options['attention_heads'],
            linear_units=dec_options['linear_units'],
            num_blocks=dec_options['num_blocks'],
            dropout_rate=dec_options['dropout_rate'],
            positional_dropout_rate=dec_options['positional_dropout_rate'],
            self_attention_dropout_rate=dec_options.get('self_attention_dropout_rate', 0.0),
            src_attention_dropout_rate=dec_options.get('src_attention_dropout_rate', 0.0),
            normalize_before=dec_options.get('normalize_before', True),
        )
        # Tokens & weights
        self.sos = sos
        self.eos = eos
        self.pad = pad
        self.ctc_weight = ctc_weight
        self.token_list = token_list  # store token_list for forward inference

    def forward(self, x, x_lengths, ys=None, ys_lengths=None):
        # Debug: trace shapes at each stage
        logging.info(f"E2EVSR.forward START -> x shape: {x.shape}, x_lengths: {x_lengths}")
        """
        If ys is provided, run teacher-forcing training and return losses.
        Otherwise run beam search inference and return hypotheses.
        """
        # Process raw video through frontend and temporal encoder
        # Create mask for encoder
        memory_mask = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(-2)
        logging.info(f"memory_mask shape: {memory_mask.shape}")
        # Visual frontend: x -> (B, T, D1)
        v_feats = self.frontend(x)
        logging.info(f"v_feats from frontend shape: {v_feats.shape}")
        # Project to encoder dim
        v_feats = self.proj_encoder(v_feats)
        logging.info(f"v_feats after proj_encoder shape: {v_feats.shape}")
        # Temporal encoder forward: unpack hidden_feats and CTC logits or mask
        if self.encoder_type == 'conformer':
            # Conformer returns tuple (hidden_feats, enc_mask)
            hidden_feats, enc_mask = self.encoder(v_feats, memory_mask)
            ctc_input = hidden_feats
        else:
            # TCN variants (hidden_feats, ctc_logits)
            batch_size = v_feats.size(0)
            hidden_feats, ctc_logits = self.encoder(v_feats, x_lengths, batch_size)
            ctc_input = ctc_logits

        feats = hidden_feats
        # Log-probabilities for CTC based on encoder-specific logits
        logp_ctc = F.log_softmax(ctc_input, dim=2)
        # Inference mode (no teacher forcing): use transformer greedy decoding
        if ys is None:
            return self.transformer_greedy_search(x, x_lengths)
        # Training: CTC loss
        ctc_in = logp_ctc.transpose(0, 1)  # (T, B, CTC_vocab_size)
        loss_ctc = self.ctc_loss(ctc_in, ys, x_lengths, ys_lengths)
        # Prepare teacher-forcing inputs/targets: un-flatten ys and pad per sequence
        # Split flat sequence ys into list by lengths
        batch = []
        start = 0
        for L in ys_lengths.tolist():
            batch.append(ys[start : start + L])
            start += L
        ys_pad = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        # Add SOS/EOS tokens and obtain decoder input and output
        ys_in, ys_out = add_sos_eos(ys_pad, self.sos, self.eos, self.pad)
        batch_size = ys_out.size(0)
        # Masks for decoder
        tgt_mask = subsequent_mask(ys_in.size(1), device=ys_in.device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
        memory_mask = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(1)
        # Transformer forward: feed encoder hidden_feats into the decoder
        dec_out = self.decoder(ys_in, tgt_mask, feats, memory_mask)
        if isinstance(dec_out, tuple):
            dec_out = dec_out[0]
        # Attention (label-smoothing) loss
        loss_att = self.att_loss(dec_out, ys_out)
        # Combine
        loss = self.ctc_weight * loss_ctc + (1.0 - self.ctc_weight) * loss_att
        return {'loss': loss, 'ctc_loss': loss_ctc, 'att_loss': loss_att} 
    
    def transformer_greedy_search(self, x, x_lengths, max_len=100):
        """Perform transformer-based auto-regressive greedy decoding: returns token index sequences."""
        self.eval()
        results = []
        with torch.no_grad():
            # Encoder forward to get features
            memory_mask_enc = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(-2)
            v_feats = self.frontend(x)
            v_feats = self.proj_encoder(v_feats)
            if self.encoder_type == 'conformer':
                hidden_feats, _ = self.encoder(v_feats, memory_mask_enc)
                feats = hidden_feats
            else:
                batch_size = v_feats.size(0)
                hidden_feats, _ = self.encoder(v_feats, x_lengths, batch_size)
                feats = hidden_feats
            # Prepare memory mask for decoder
            memory_mask_dec = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(1)
            # Auto-regressive decoding per utterance
            for b in range(feats.size(0)):
                feat_b = feats[b:b+1]  # (1, T, D)
                mem_mask_b = memory_mask_dec[b:b+1]
                ys = torch.full((1, 1), self.sos, dtype=torch.long, device=x.device)
                for _ in range(max_len):
                    # Create target mask
                    tgt_mask = subsequent_mask(ys.size(1), device=ys.device).unsqueeze(0)
                    dec_out = self.decoder(ys, tgt_mask, feat_b, mem_mask_b)
                    if isinstance(dec_out, tuple):
                        dec_out = dec_out[0]
                    logp = F.log_softmax(dec_out, dim=-1)
                    next_tok = logp[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
                    ys = torch.cat([ys, next_tok], dim=1)
                    if next_tok.item() == self.eos:
                        break
                # Strip SOS and pad, stop at EOS, and collapse repeats
                raw_seq = ys[0, 1:].cpu().tolist()
                clean_seq = []
                prev = None
                for idx in raw_seq:
                    # stop at EOS
                    if idx == self.eos:
                        break
                    # skip SOS and padding tokens
                    if idx == self.sos or idx == self.pad:
                        continue
                    # collapse consecutive repeats
                    if idx != prev:
                        clean_seq.append(idx)
                    prev = idx
                results.append(clean_seq)
        return results 