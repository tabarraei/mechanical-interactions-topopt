#!/usr/bin/env python

#IMPORTS============================================================================================
#native---------------------------------------------------------------------------------------------
from typing import Optional, Dict, Any
#external-------------------------------------------------------------------------------------------
import numpy as np
import torch as th

#GLOBALS============================================================================================
#CLASSES============================================================================================

class TopOptSample:
    """
    A class to store topology optimization sample data.
    
    This class contains all the data associated with a single topology optimization
    problem instance, including stress/strain fields, optimal topology, loading
    conditions, volume fraction, and compliance.
    
    All fields are optional to allow for flexible data storage and retrieval.
    """
    
    # Field definitions for centralized management
    FIELD_STACKABLE = ['stress_field', 'strain_field', 'optimal_topology', 'boundary_conditions'] # arrays/tensors that can be stacked in batches
    FIELD_LIST = ['load', 'coord_load', 'coord_boundary_conditions', 'sample_id', 'mesh', 'volume_fraction', 'compliance', 'load_type', 'density_trajectory'] # Non-stackable fields: kept as lists in batches (dicts, strings, scalars, variable-shape arrays)
    FIELD_ALL = FIELD_STACKABLE + FIELD_LIST
    
    def __init__(self,
                 mesh: Dict[str, Any],
                 stress_field: Optional[np.ndarray] = None,
                 strain_field: Optional[np.ndarray] = None,
                 optimal_topology: Optional[np.ndarray] = None,
                 load: Optional[Dict[str, Any]] = None,
                 coord_load: Optional[Dict[str, Any]] = None,
                 coord_boundary_conditions: Optional[Dict[str, Any]] = None,
                 boundary_conditions: Optional[Dict[str, Any]] = None,
                 volume_fraction: Optional[float] = None,
                 compliance: Optional[float] = None,
                 sample_id: Optional[int] = None,
                 load_type: str = "static",
                 density_trajectory: Optional[np.ndarray] = None):
        """
        Initialize a TopOptSample with optional fields.
        
        Args:
            mesh: Dictionary containing mesh information (required)
            stress_field: Stress field data (typically von Mises stress)
            strain_field: Strain field data 
            optimal_topology: Optimized topology/density distribution
            load: Dictionary containing load information with keys:
                  - 'x': x-coordinates of load locations
                  - 'y': y-coordinates of load locations  
                  - 'fx': force magnitudes in x-direction
                  - 'fy': force magnitudes in y-direction
            coord_load: Same keys as load
            coord_boundary_conditions: {'x': [{'start': (x0, y0), 'stop': (x1, y1)}, ...],
                                        'y': [{'start': (x0, y0), 'stop': (x1, y1)}, ...]}
            boundary_conditions: Boundary condition array (H+1, W+1, 2) with x and y components
            volume_fraction: Target volume fraction constraint
            compliance: Structural compliance value
            sample_id: Unique identifier for the sample, string
            load_type: Type of loading condition ("static", "sine", or "impulse")
            density_trajectory: Filtered density at each optimization step, shape (n_steps, nely, nelx).
                Step count and grid size vary per sample, so batches do not stack this field.
        """
        self.mesh = mesh
        self.stress_field = stress_field
        self.strain_field = strain_field
        self.optimal_topology = optimal_topology
        self.load = load
        self.coord_load = coord_load
        self.coord_boundary_conditions = coord_boundary_conditions
        self.boundary_conditions = boundary_conditions
        self.volume_fraction = volume_fraction
        self.compliance = compliance
        self.sample_id = sample_id
        self.density_trajectory = density_trajectory
        
        # Validate and set load_type
        valid_load_types = ["static", "sine", "impulse"]
        if load_type.lower() not in valid_load_types:
            raise ValueError(f"load_type must be one of {valid_load_types}, got '{load_type}'")
        self.load_type = load_type.lower()
    
    def __repr__(self):
        """String representation of the TopOptSample."""
        field_strs = []
        
        # Stackable fields (show shape)
        for field in self.FIELD_STACKABLE:
            if self.has(field):
                value = getattr(self, field)
                field_strs.append(f"{field}: {value.shape}")
        
        # Non-stackable fields (dict/list length or value)
        for field in self.FIELD_LIST:
            if self.has(field):
                value = getattr(self, field)
                if field == 'sample_id':
                    field_strs.append(f"id: {value}")
                elif isinstance(value, dict):
                    field_strs.append(f"{field}: {len(value)} components")
                elif field == 'density_trajectory' and hasattr(value, 'shape'):
                    field_strs.append(f"{field}: {tuple(value.shape)}")
                else:
                    field_strs.append(f"{field}: {value}")
        
        return f"TopOptSample({', '.join(field_strs)})"
    
    def has(self, field_name: str) -> bool:
        """Check if a field exists and has a non-None value."""
        if field_name == 'mesh':
            return True  # mesh is always available
        if not hasattr(self, field_name):
            return False
        value = getattr(self, field_name)
        if value is None:
            return False
        if isinstance(value, dict):
            return len(value) > 0
        return True
    
    def get_available_fields(self) -> list:
        """Return a list of field names that have data."""
        available = []
        
        for field in self.FIELD_ALL:
            if self.has(field):
                available.append(field)
        
        return available
    
    def copy(self):
        """
        Create a copy of this TopOptSample with all fields dynamically copied.
        
        Returns:
            TopOptSample: A new instance with all fields copied appropriately.
        """
        field_kwargs = {}
        
        for field in self.get_available_fields():
            value = getattr(self, field)
            
            # Handle different field types automatically based on their type
            if hasattr(value, 'clone') and hasattr(value, 'detach'):
                # PyTorch tensor - clone and detach
                field_kwargs[field] = value.clone().detach()
            elif isinstance(value, dict):
                # Dictionary - copy it
                field_kwargs[field] = value.copy()
            else:
                # Other types (numpy arrays, scalars, etc.) - copy as-is
                field_kwargs[field] = value
        
        return TopOptSample(**field_kwargs)
    
    def _get_nonzero_locations(self, array):
        """
        Helper function to get nonzero locations from numpy arrays or PyTorch tensors.
        Returns tuple of (row_indices, col_indices).
        """
        if isinstance(array, th.Tensor):
            # Convert tensor to numpy for np.nonzero
            array_np = array.detach().cpu().numpy()
            return np.nonzero(array_np)
        else:
            # Already a numpy array
            return np.nonzero(array)
    
    def get_boundary_condition_locations(self) -> Optional[dict]:
        """
        Get boundary condition locations from the boundary conditions array.
        Returns dictionary with 'x' and 'y' keys containing lists of (x, y) coordinate tuples,
        or None if no boundary condition data.
        """
        if not self.has('boundary_conditions'):
            return None
        
        # Extract x and y boundary condition locations
        bc_x = self.boundary_conditions[:, :, 0]  # x-direction constraints
        bc_y = self.boundary_conditions[:, :, 1]  # y-direction constraints
        
        bc_locs_x = self._get_nonzero_locations(bc_x)
        bc_locs_y = self._get_nonzero_locations(bc_y)
        
        # Get coordinate pairs from x-constraints
        # np.nonzero returns (row_indices, col_indices), so we need to pair them correctly
        x_constraint_locs = [(bc_locs_x[1][i], bc_locs_x[0][i]) for i in range(len(bc_locs_x[0]))]
        # Get coordinate pairs from y-constraints  
        y_constraint_locs = [(bc_locs_y[1][i], bc_locs_y[0][i]) for i in range(len(bc_locs_y[0]))]
        
        return {
            'x': x_constraint_locs,
            'y': y_constraint_locs
        }

    def get_boundary_condition_dofs(self) -> Optional[np.ndarray]:
        """
        Get boundary condition dofs from the boundary conditions array.
        Returns single array of all constrained DOFs or None if no boundary condition data.

        Assumes a regular grid with nelx and nely.
        """
        if 'nelx' not in self.mesh or 'nely' not in self.mesh:
            raise ValueError(
                "get_boundary_condition_dofs requires mesh['nelx'] and mesh['nely']"
            )

        locations = self.get_boundary_condition_locations()
        if locations is None:
            return None
        
        # Extract DOF indices for x-constraints and y-constraints
        all_constraint_dofs = []
        
        for x_coord, y_coord in locations['x']:
            node_id = y_coord * (self.mesh['nelx'] + 1) + x_coord
            x_dof = node_id * 2  # X-displacement DOF
            all_constraint_dofs.append(x_dof)
        
        for x_coord, y_coord in locations['y']:
            node_id = y_coord * (self.mesh['nelx'] + 1) + x_coord
            y_dof = node_id * 2 + 1  # Y-displacement DOF
            all_constraint_dofs.append(y_dof)
        
        clean_dofs = []
        for dof in all_constraint_dofs:
            if isinstance(dof, th.Tensor):
                clean_dofs.append(int(dof.detach().cpu().item()))
            else:
                clean_dofs.append(int(dof))
        
        return np.array(clean_dofs, dtype=int)

class TopOptBatch:
    """Batch container for TopOptSample objects."""
    
    def __init__(self, samples):
        self.samples = samples
        self.batch_size = len(samples)
        self._field_names = samples[0].get_available_fields() if samples else []
    
    def __getattr__(self, name):
        """Allow access to fields across all samples in the batch."""
        # Avoid recursion by using object.__getattribute__ which bypasses __getattr__
        try:
            _field_names = object.__getattribute__(self, '_field_names')
            samples = object.__getattribute__(self, 'samples')
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # density_trajectory is written by optimize after the batch is built, and
        # its shape differs across samples, so it is not part of the init-time field list.
        known = name in _field_names or (
            name == 'density_trajectory'
            and any(sample.has('density_trajectory') for sample in samples)
        )
        if not known:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        # Get the field from all samples
        field_values = []
        for sample in samples:
            if hasattr(sample, name):
                field_values.append(getattr(sample, name))
            else:
                field_values.append(None)
        
        # Check if all values are None
        if all(v is None for v in field_values):
            return None

        # Variable length and aspect ratio: one array per sample, not a stacked tensor.
        if name == 'density_trajectory':
            return field_values
        
        # If all values are tensors/arrays, stack them
        if all(isinstance(v, th.Tensor) for v in field_values if v is not None):
            # Move all tensors to the same device (use the first non-None tensor's device)
            device = next(v.device for v in field_values if v is not None)
            tensors = [v.to(device) for v in field_values if v is not None]
            return th.stack(tensors)
        elif all(isinstance(v, np.ndarray) for v in field_values if v is not None):
            return np.stack([v for v in field_values if v is not None])
        else:
            return field_values
    
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def to(self, device):
        """
        Move all tensors in the batch to the specified device.
        This method allows PyTorch Lightning to automatically move batches to GPU.
        Numpy arrays are converted to PyTorch tensors and moved to the device.
        Dictionary and list fields are recursively processed to move nested tensors.
        
        Args:
            device: Target device (torch.device, str, or int)
        
        Returns:
            TopOptBatch: Returns self for method chaining (in-place modification)
        """
        # Normalize device to torch.device for consistent comparison
        if not isinstance(device, th.device):
            device = th.device(device)
        
        def move_recursive(val):
            """Recursively move tensors in nested structures (dicts, lists, tuples, tensors, arrays)."""
            if isinstance(val, dict):
                for key, v in val.items():
                    val[key] = move_recursive(v)
                return val
            elif isinstance(val, list):
                return [move_recursive(v) for v in val]
            elif isinstance(val, tuple):
                return tuple(move_recursive(v) for v in val)
            elif isinstance(val, th.Tensor):
                return val if val.device == device else val.to(device)
            elif isinstance(val, np.ndarray):
                return th.from_numpy(val).to(device)
            return val
        
        for sample in self.samples:
            # Process all fields
            for field_name in TopOptSample.FIELD_ALL:
                if sample.has(field_name):
                    value = getattr(sample, field_name)
                    setattr(sample, field_name, move_recursive(value))
        
        return self
    
    def copy(self):
        """
        Create a copy of this TopOptBatch with all fields dynamically copied.
        
        Returns:
            TopOptBatch: A new instance with all fields copied.
        """
        copied_samples = [sample.copy() for sample in self.samples]
        return TopOptBatch(copied_samples)

#FUNCTIONS==========================================================================================

#MAIN===============================================================================================
if __name__ == "__main__": 
    raise RuntimeError("Please don't execute me! I have so much to live for~~~")