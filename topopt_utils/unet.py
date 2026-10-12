import warnings

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .topology_data import TopOptSample, TopOptBatch
from .nn import (
    conv_nd,
    avg_pool_nd,
)

#region HELPER CLASSES
class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ConvNextBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode="reflect")
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


IMAGE_SIZE = 64
NUM_DOWNSAMPLES = 3
BOTTLENECK_SIZE = IMAGE_SIZE // (2 ** NUM_DOWNSAMPLES)  # 8 for 64x64


def _resolve_stage_config(depths, dims):
    """Map config to (channels, blocks) per stage; both must be length-4 sequences."""
    if depths is None or dims is None:
        raise ValueError("depths and dims are required (4 values each, e.g. dims=96,192,384,768)")
    depths = list(depths)
    dims = list(np.asarray(dims).astype(int).ravel())
    if len(depths) != 4 or len(dims) != 4:
        raise ValueError(
            f"depths and dims must each have 4 values; got depths={depths}, dims={dims}"
        )
    return tuple(int(c) for c in dims), tuple(int(b) for b in depths)


def _build_encoder_pyramid(in_channels, channels, blocks):
    layers = [
        nn.Conv2d(in_channels, channels[0], kernel_size=7, padding=3, padding_mode="reflect")
    ]
    for stage, (ch, n_blocks) in enumerate(zip(channels, blocks)):
        for _ in range(n_blocks):
            layers.append(ConvNextBlock(ch))
        if stage < len(channels) - 1:
            layers.append(Downsample(ch, use_conv=True, out_channels=channels[stage + 1]))
    return nn.ModuleList(layers)


def _build_decoder_pyramid(out_channels, channels, blocks):
    layers = []
    for stage in reversed(range(len(channels))):
        ch = channels[stage]
        for _ in range(blocks[stage]):
            layers.append(ConvNextBlock(ch))
        if stage > 0:
            layers.append(Upsample(ch, use_conv=True, out_channels=channels[stage - 1]))
    layers.append(nn.Conv2d(channels[0], out_channels, kernel_size=7, padding=3, padding_mode="reflect"))
    layers.append(nn.Sigmoid())
    return nn.ModuleList(layers)


#region AUTOENCODER
class AE(nn.Module):
    ## Autoencoder model for latent space conversion
    def __init__(self, 
            vector_size=64, 
            use_variational=True,
            depths=None,
            dims=None,
            linear_hidden1=None,
            linear_hidden2=None):
        super().__init__()
        
        self.vector_size = vector_size
        self.KLD = 0
        self.use_variational = use_variational
        
        # Device property - will be set when model is moved to device
        self._device = None
        
        # Initialize model components
        self._initialize_model(use_variational, depths, dims, linear_hidden1, linear_hidden2)

    @property
    def device(self):
        """Get the device of the model parameters."""
        if self._device is not None:
            return self._device
        # Get device from first parameter
        for param in self.parameters():
            return param.device
        # Fallback to CPU if no parameters
        return th.device('cpu')

    def _initialize_model(self, use_variational, depths=None, dims=None, linear_hidden1=None, linear_hidden2=None):
        """Initialize model components after device property is set."""
        var_mult = 2 if use_variational else 1

        channels, blocks = _resolve_stage_config(depths, dims)
        self.channels = channels
        self.blocks = blocks
        self.bottleneck_channels = channels[-1]
        self.bottleneck_size = BOTTLENECK_SIZE
        bottleneck_spatial = self.bottleneck_channels * self.bottleneck_size ** 2

        if linear_hidden1 is None:
            linear_hidden1 = 2 * self.bottleneck_channels
        if linear_hidden2 is None:
            linear_hidden2 = self.bottleneck_channels

        self.structure_encoder_spatial_blocks = _build_encoder_pyramid(1, channels, blocks)
        self.structure_decoder_spatial_blocks = _build_decoder_pyramid(1, channels, blocks)
        self.field_encoder_spatial_blocks = _build_encoder_pyramid(2, channels, blocks)
        self.field_decoder_spatial_blocks = _build_decoder_pyramid(2, channels, blocks)

        #### LINEAR BLOCKS (GAP at 8x8 bottleneck -> vector) ####
        self.structure_encoder_linears = nn.ModuleList([
            GlobalAvgPool2d(),
            nn.Linear(self.bottleneck_channels, linear_hidden1),
            nn.ReLU(),
            nn.Linear(linear_hidden1, linear_hidden2),
            nn.Tanh(),
            nn.Linear(linear_hidden2, self.vector_size * var_mult),
            nn.Unflatten(1, (var_mult, self.vector_size)),
        ])

        self.structure_decoder_linears = nn.ModuleList([
            nn.Flatten(start_dim=1),
            nn.Linear(self.vector_size, linear_hidden2),
            nn.ReLU(),
            nn.Linear(linear_hidden2, linear_hidden1),
            nn.ReLU(),
            nn.Linear(linear_hidden1, bottleneck_spatial),
            nn.Unflatten(1, (self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)),
        ])

        self.field_encoder_linears = nn.ModuleList([
            GlobalAvgPool2d(),
            nn.Linear(self.bottleneck_channels, linear_hidden1),
            nn.ReLU(),
            nn.Linear(linear_hidden1, linear_hidden2),
            nn.Tanh(),
            nn.Linear(linear_hidden2, self.vector_size - 5),
            nn.Tanh(),
        ])

        self.field_decoder_linears = nn.ModuleList([
            nn.Linear(self.vector_size - 5, linear_hidden2),
            nn.ReLU(),
            nn.Linear(linear_hidden2, linear_hidden1),
            nn.ReLU(),
            nn.Linear(linear_hidden1, bottleneck_spatial),
            nn.Unflatten(1, (self.bottleneck_channels, self.bottleneck_size, self.bottleneck_size)),
        ])
        
        #### FULL SET ####
        self.structure_encoder_blocks = nn.ModuleList([*self.structure_encoder_spatial_blocks, *self.structure_encoder_linears])
        self.structure_decoder_blocks = nn.ModuleList([*self.structure_decoder_linears,        *self.structure_decoder_spatial_blocks])
        self.field_encoder_blocks =     nn.ModuleList([*self.field_encoder_spatial_blocks,     *self.field_encoder_linears])
        self.field_decoder_blocks =     nn.ModuleList([*self.field_decoder_linears,            *self.field_decoder_spatial_blocks])

        #### MSC ####
        self.structure_encoder_activation = nn.Tanh() # We apply this later because it may hit the logvar term

    ########## FUNCTIONS FOR ENCODINGS AND DECODINGS ##########
    
    ##### ENCODE #####
    def encode(self, sample_or_batch, override_variational=None):
        # Ensure input is TopOptSample or TopOptBatch
        if not isinstance(sample_or_batch, (TopOptSample, TopOptBatch)):
            raise TypeError(f"Expected TopOptSample or TopOptBatch, got {type(sample_or_batch)}")

        if isinstance(sample_or_batch, TopOptSample):
            # Wrap single sample in batch for uniform processing
            batch = TopOptBatch([sample_or_batch])
            batch_size = 1
        else:
            batch = sample_or_batch
            batch_size = len(batch)

        structure_batch = batch.optimal_topology
        if structure_batch is None:
            raise ValueError("Batch must have optimal_topology for encoding.")
        if not isinstance(structure_batch, th.Tensor):
            structure_batch = th.as_tensor(structure_batch, dtype=th.float32)
        structure_batch = structure_batch.to(self.device)
        if structure_batch.dim() == 3:
            structure_batch = structure_batch.unsqueeze(1) 
        
        encoded_structure = self.encode_structure(structure_batch, override_variational)
        conditions_batch = self.encode_conditions(batch)
        
        # Combine structure and conditions
        # Structure: 1 channel (after variational sampling) + Conditions: 1 channel = 2 channels total
        structure_channels = encoded_structure.shape[1] # Should be 1 after variational sampling
        x_out = th.empty((batch_size, structure_channels+1, self.vector_size), device=structure_batch.device)
        x_out[:, :-1, :] = encoded_structure  # Structure channel
        x_out[:, -1, :] = conditions_batch    # Conditions channel
        
        return x_out
    
    def encode_conditions(self, batch):
        """Encode conditions from TopOptBatch using batched operations."""
        batch_size = len(batch)
        
        # Volume fraction: (B,) or list -> (B, 1)
        volume_fracs = batch.volume_fraction
        if isinstance(volume_fracs, list):
            volume_fracs = th.tensor(volume_fracs, device=self.device)
        else:
            volume_fracs = volume_fracs.to(self.device)
        x_volume = volume_fracs.reshape(batch_size, 1)
        
        # Physical fields: (B, H, W)
        stress_fields = batch.stress_field
        strain_fields = batch.strain_field
        if isinstance(stress_fields, list):
            stress_fields = np.stack([np.asarray(x) for x in stress_fields])
        if isinstance(strain_fields, list):
            strain_fields = np.stack([np.asarray(x) for x in strain_fields])
        if not isinstance(stress_fields, th.Tensor):
            stress_fields = th.as_tensor(stress_fields, dtype=th.float32)
        if not isinstance(strain_fields, th.Tensor):
            strain_fields = th.as_tensor(strain_fields, dtype=th.float32)
        stress_fields = stress_fields.to(self.device)
        strain_fields = strain_fields.to(self.device)
        
        # Preprocess stress field: per-sample min-max normalization then raise to power 1/3
        stress_min = stress_fields.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # (B, 1, 1)
        stress_max = stress_fields.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # (B, 1, 1)
        stress_range = stress_max - stress_min + 1e-8  # Avoid division by zero
        stress_fields_normalized = (stress_fields - stress_min) / stress_range  # (B, H, W) -> [0, 1]
        stress_fields_processed = th.pow(stress_fields_normalized, 1/3)  # Power 1/3 transform
        
        # Preprocess strain field: per-sample min-max normalization then raise to power 1/11
        strain_min = strain_fields.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]  # (B, 1, 1)
        strain_max = strain_fields.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]  # (B, 1, 1)
        strain_range = strain_max - strain_min + 1e-8  # Avoid division by zero
        strain_fields_normalized = (strain_fields - strain_min) / strain_range  # (B, H, W) -> [0, 1]
        strain_fields_processed = th.pow(strain_fields_normalized, 1/11)  # Power 1/11 transform
        
        # Store normalization parameters for later use in decoding
        self.stress_min = stress_min
        self.stress_max = stress_max
        self.strain_min = strain_min
        self.strain_max = strain_max
        
        # Store normalized data for loss calculation
        self.stress_fields_normalized = stress_fields_normalized
        self.strain_fields_normalized = strain_fields_normalized
        
        # Stack processed stress and processed strain: (B, 2, H, W)
        x_pf = th.stack([stress_fields_processed, strain_fields_processed], dim=1)
        
        # Encode physical fields through field encoder blocks
        for block in self.field_encoder_blocks:
            x_pf = block(x_pf)  # (B, 2, H, W) -> (B, C)
        
        # Flatten to (B, C)
        x_pf = x_pf.reshape(batch_size, -1)
        
        # Load information: process as batch
        # Extract load dicts from samples
        loads = [s.load for s in batch.samples]
        x_coords = th.tensor([l.get('x', [0])[0] if l else 0 for l in loads], device=self.device)
        y_coords = th.tensor([l.get('y', [0])[0] if l else 0 for l in loads], device=self.device)
        fx_forces = th.tensor([l.get('fx', [0.0])[0] if l else 0.0 for l in loads], device=self.device)
        fy_forces = th.tensor([l.get('fy', [0.0])[0] if l else 0.0 for l in loads], device=self.device)
        
        # Get mesh dimensions (assume same for all samples)
        nelx = batch.samples[0].mesh['nelx']
        nely = batch.samples[0].mesh['nely']
        
        # Clamp coordinates
        x_coords = th.clamp(x_coords, 0, nelx - 1)
        y_coords = th.clamp(y_coords, 0, nely - 1)
        
        # Concatenate all conditions: (B, vector_size)
        conditions = th.cat([
            x_pf,
            x_volume,
            x_coords.unsqueeze(1),
            y_coords.unsqueeze(1),
            fx_forces.unsqueeze(1),
            fy_forces.unsqueeze(1)
        ], dim=1)
        
        return conditions
    
    def encode_structure(self, x, override_variational=None):
        # x already has channel dimension from _encode_from_sample: (B, 1, H, W)
        # No need to add another channel dimension
        
        for block in self.structure_encoder_blocks:
            x = block(x)

        use_variational = self.use_variational if override_variational is None else override_variational

        if use_variational:
            mid_dim = int(x.shape[1]/2)
            self.log_var = x[:, :mid_dim, :].to(self.device)
            self.mean = self.structure_encoder_activation(x[:, mid_dim:, :]).to(self.device)
            
            # Apply variational sampling to get single-channel output
            epsilon = th.randn_like(self.mean).to(self.device)
            x = self.mean + th.exp(0.5 * self.log_var) * epsilon
            
            # Calculate KLD loss
            self.KLD = -0.5 * th.mean(1 + self.log_var - 0.5 * self.mean.pow(2) - th.exp(self.log_var))
            if self.KLD.isnan():
                self.KLD = 0
        else:
            # Non-variational: if input has 2 channels, take only the second half (mean part)
            if x.shape[1] == 2:
                mid_dim = int(x.shape[1]/2)
                x = x[:, mid_dim:, :]  # Take only the second half (mean part)
            x = self.structure_encoder_activation(x)
            self.KLD = 0


        self.latent_norm = th.mean(th.norm(x, dim=[1,2]))
        
        return x

    ##### DECODE #####
    def decode(self, x, sample_template, override_variational=None):
        # Ensure template is TopOptSample or TopOptBatch
        if not isinstance(sample_template, (TopOptSample, TopOptBatch)):
            raise TypeError(f"Expected TopOptSample or TopOptBatch template, got {type(sample_template)}")
        
        # Create a copy of the template to avoid modifying original samples
        if isinstance(sample_template, TopOptSample):
            # Use the built-in copy method
            batch_template = TopOptBatch([sample_template.copy()])
            batch_size = 1
        else:
            # Use the built-in copy method
            batch_template = sample_template.copy()
            batch_size = x.shape[0]
        
        # Decode structure for entire batch
        structure_latent = x[:, :-1, :]  # (B, structure_channels, vector_size)
        conditions_latent = x[:, -1, :]    # (B, vector_size)
        
        # Decode structure to TopOptBatch with topology
        decoded_structure_batch = self.decode_structure(structure_latent, batch_template, override_variational=override_variational)
        
        # Decode conditions to fill out the rest of the batch
        final_batch = self.decode_conditions(conditions_latent, decoded_structure_batch)
        
        # Return single sample or batch based on input
        if isinstance(sample_template, TopOptSample):
            return final_batch.samples[0]
        else:
            return final_batch

    def decode_conditions(self, x, batch_template):
        """Decode conditions to fill out TopOptBatch with stress, strain, volume, and load."""
        B = x.shape[0]

        ## Handle each channel individually
        # 0. Volume channel
        volume_fractions = x[:,(self.vector_size-5)].to(x.device)  # (B,)

        # 1, 2 Physical fields channels
        x_pf = x[:,:(self.vector_size-5)]
        for block in self.field_decoder_blocks:
            x_pf = block(x_pf)

        # 3, 4, 5 Loads - separate x and y coordinates
        x_coords = x[:,(self.vector_size-4)].int()  # (B,)
        y_coords = x[:,(self.vector_size-3)].int()  # (B,)
        x_lx = x[:,(self.vector_size-2)]
        x_ly = x[:,(self.vector_size-1)]

        stress_fields_processed = x_pf[:, 0, :, :]  # (B, H, W) - first channel
        strain_fields_processed = x_pf[:, 1, :, :]  # (B, H, W) - second channel
        
        # Postprocess stress field: reverse power 1/3 then denormalize using min-max
        if hasattr(self, 'stress_min') and hasattr(self, 'stress_max'):
            stress_min = self.stress_min
            stress_max = self.stress_max
        else:
            # Fallback: use current batch statistics (not ideal but prevents errors)
            import warnings
            warnings.warn("Using current batch statistics for stress field postprocessing", UserWarning)
            stress_min = stress_fields_processed.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            stress_max = stress_fields_processed.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        
        # Reverse power 1/3: raise to power 3
        stress_fields_normalized_reconstructed = th.pow(stress_fields_processed, 3)  # (B, H, W) -> [0, 1]
        # Reverse min-max normalization
        stress_range = stress_max - stress_min + 1e-8
        stress_fields = stress_fields_normalized_reconstructed * stress_range + stress_min
        
        # Postprocess strain field: reverse power 1/11 then denormalize using min-max
        if hasattr(self, 'strain_min') and hasattr(self, 'strain_max'):
            strain_min = self.strain_min
            strain_max = self.strain_max
        else:
            # Fallback: use current batch statistics (not ideal but prevents errors)
            strain_min = strain_fields_processed.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
            strain_max = strain_fields_processed.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        
        # Reverse power 1/11: raise to power 11
        strain_fields_normalized_reconstructed = th.pow(strain_fields_processed, 11)  # (B, H, W) -> [0, 1]
        # Reverse min-max normalization
        strain_range = strain_max - strain_min + 1e-8
        strain_fields = strain_fields_normalized_reconstructed * strain_range + strain_min
        
        # Store reconstructed normalized data for loss calculation
        self.stress_fields_normalized_reconstructed = stress_fields_normalized_reconstructed
        self.strain_fields_normalized_reconstructed = strain_fields_normalized_reconstructed
        
        # Create load dictionaries directly
        load_dicts = []
        for i in range(B):
            load_dict = {
                'x': [int(x_coords[i].item())],
                'y': [int(y_coords[i].item())],
                'fx': [round(float(x_lx[i].item()), 6)],
                'fy': [round(float(x_ly[i].item()), 6)],
            }
            load_dicts.append(load_dict)
        
        # Fill in conditions for existing samples
        for i in range(B):
            existing_sample = batch_template.samples[i]
            
            # Update existing sample with decoded conditions
            existing_sample.stress_field = stress_fields[i]
            existing_sample.strain_field = strain_fields[i]
            existing_sample.volume_fraction = volume_fractions[i].item()
            existing_sample.load = load_dicts[i]
            
            # Remove compliance field since it's no longer accurate after reconstruction
            if hasattr(existing_sample, 'compliance'):
                delattr(existing_sample, 'compliance')
        
        if hasattr(batch_template, '_field_names'):
            batch_template._field_names = batch_template.samples[0].get_available_fields()
        
        return batch_template

    def decode_structure(self, x, batch_template, override_variational=None):
        """Decode structure latent to TopOptBatch with reconstructed topology."""
        batch_size = x.shape[0]
        
        # Process structure through decoder blocks
        # x shape: (B, 1, vector_size) - single channel for structure
        for block in self.structure_decoder_blocks:
            x = block(x)

        # Extract decoded topology
        decoded_topology = x[:, 0, :, :]  # (B, H, W)
        
        # Fill in topology for existing samples
        for i in range(batch_size):
            sample_template = batch_template.samples[i]
            original_id = sample_template.sample_id
            reconstructed_id = f"{original_id}_reconstructed" if original_id is not None else "reconstructed_sample"
            
            # Update existing sample with decoded topology
            sample_template.optimal_topology = decoded_topology[i]
            sample_template.sample_id = reconstructed_id
        
        return batch_template
    
    ##### FORWARD #####
    def forward(self, x, override_variational=None):
        # Store original input as template for decode
        sample_template = x
        x = self.encode(x, override_variational=override_variational) # (N, C, H, W) -> (N, E, V) 
        x = self.decode(x, sample_template, override_variational=override_variational) # (N, E, V) -> (N, C, H, W)
        return x
