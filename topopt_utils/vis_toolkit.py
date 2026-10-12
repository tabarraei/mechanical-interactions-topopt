#!/usr/bin/env python

#IMPORTS============================================================================================
#native---------------------------------------------------------------------------------------------
#external-------------------------------------------------------------------------------------------
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from .topology_data import TopOptSample, TopOptBatch
import pickle


def _to_numpy(data):
    """Convert tensor or numpy array to numpy array for matplotlib."""
    if isinstance(data, th.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError(f"Expected tensor or numpy array, got {type(data)}")


def _as_1d(values):
    if isinstance(values, th.Tensor):
        values = values.detach().cpu().numpy()
    return np.atleast_1d(np.asarray(values, dtype=float)).reshape(-1)


def _domain_size(sample):
    return float(sample.mesh["width"]), float(sample.mesh["height"])


def _index_to_physical(x, y, sample, *, center=False):
    """Map element (center=True) or node indices to physical xy (origin bottom-left)."""
    nelx = int(sample.mesh["nelx"])
    nely = int(sample.mesh["nely"])
    width, height = _domain_size(sample)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if center:
        x = x + 0.5
        y = y + 0.5
    return x / nelx * width, y / nely * height


def _physical_imshow(ax, data, sample, *args, **kwargs):
    width, height = _domain_size(sample)
    kwargs.setdefault("extent", (0.0, width, 0.0, height))
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("aspect", "equal")
    return ax.imshow(_to_numpy(data), *args, **kwargs)


# Displayed unit-load shaft. inches (not data/xy) stay readable when
# equal-aspect letterboxes a 5:1 or 1:5 domain inside a square-ish subplot.
# matplotlib: smaller scale → longer arrows when scale_units is width/inches.
# units='inches' keeps shaft width in inches so thickness does not shrink
# with a letterboxed axes bbox (default units='width' would).
_LOAD_ARROW_INCHES = 0.24
_LOAD_ARROW_WIDTH_INCHES = 0.016


def _quiver_loads(ax, sample, **kwargs):
    if sample.has("coord_load"):
        load = sample.coord_load
        xs, ys = _as_1d(load["x"]), _as_1d(load["y"])
    elif sample.has("load"):
        load = sample.load
        xs, ys = _index_to_physical(_as_1d(load["x"]), _as_1d(load["y"]), sample, center=True)
    else:
        return
    fx = _as_1d(load["fx"])
    fy = _as_1d(load["fy"])
    mags = np.hypot(fx, fy)
    u = np.zeros_like(fx)
    v = np.zeros_like(fy)
    nonzero = mags > 0
    u[nonzero] = fx[nonzero] / mags[nonzero]
    v[nonzero] = fy[nonzero] / mags[nonzero]
    arrow_inches = kwargs.pop("arrow_inches", _LOAD_ARROW_INCHES)
    kwargs["angles"] = "xy"
    kwargs["scale_units"] = "inches"
    kwargs["scale"] = 1.0 / arrow_inches if arrow_inches > 0 else 1.0
    kwargs["units"] = "inches"
    kwargs["width"] = _LOAD_ARROW_WIDTH_INCHES
    kwargs["headwidth"] = 3.0
    kwargs["headlength"] = 4.5
    kwargs["headaxislength"] = 4.0
    ax.quiver(xs, ys, u, v, **kwargs)


def _physical_bc_locations(sample):
    """Boundary-condition points in physical xy. Prefers coord segments over raster nodes."""
    if sample.has("coord_boundary_conditions"):
        width, height = _domain_size(sample)
        nelx = sample.mesh.get("nelx")
        nely = sample.mesh.get("nely")
        out = {"x": [], "y": []}
        for axis in ("x", "y"):
            for seg in sample.coord_boundary_conditions.get(axis, []):
                x0, y0 = float(seg["start"][0]), float(seg["start"][1])
                x1, y1 = float(seg["stop"][0]), float(seg["stop"][1])
                length = float(np.hypot(x1 - x0, y1 - y0))
                if length == 0.0:
                    out[axis].append((x0, y0))
                    continue
                if nelx is not None and nely is not None:
                    ds = min(width / int(nelx), height / int(nely))
                    n = max(int(round(length / ds)), 1) if ds > 0 else 1
                else:
                    n = 32
                for t in np.linspace(0.0, 1.0, n + 1):
                    out[axis].append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return out
    if not sample.has("boundary_conditions"):
        return None
    locations = sample.get_boundary_condition_locations()
    if locations is None:
        return None
    converted = {}
    for axis in ("x", "y"):
        locs = locations.get(axis) or []
        if not locs:
            converted[axis] = []
            continue
        px, py = _index_to_physical(
            [coord[0] for coord in locs],
            [coord[1] for coord in locs],
            sample,
            center=False,
        )
        converted[axis] = list(zip(np.atleast_1d(px), np.atleast_1d(py)))
    return converted


#GLOBALS============================================================================================
#CLASSES============================================================================================
#FUNCTIONS==========================================================================================
def plot_AE_evaluation(model_output, ground_truth, save_name = None):
    """Plots the model and ground truth sets against eachother.

    Parameters
    ----------
    
    model_output: TopOptSample or TopOptBatch
               | Output from autoencoder containing topology optimization data
    
    ground_truth: TopOptSample or TopOptBatch
               | Expected output from autoencoder containing topology optimization data

    save_name: string
               | Name to save the image as. If None, don't save anything
               | DEFAULT = None
    """
    if isinstance(model_output, TopOptBatch):
        model_output = model_output[0]
    if isinstance(ground_truth, TopOptBatch):
        ground_truth = ground_truth[0]

    # Init plots
    _, axarr = plt.subplots(3,5, figsize=(15, 9))
    plots = np.empty((3,5), dtype=object)

    diff_topology = model_output.optimal_topology - ground_truth.optimal_topology
    diff_vf = model_output.volume_fraction - ground_truth.volume_fraction
    diff_stress = model_output.stress_field - ground_truth.stress_field
    diff_strain = model_output.strain_field - ground_truth.strain_field
    # Convert to tensors if needed for load data
    import torch
    gt_fx = torch.tensor(ground_truth.load['fx']) if isinstance(ground_truth.load['fx'], list) else ground_truth.load['fx']
    gt_fy = torch.tensor(ground_truth.load['fy']) if isinstance(ground_truth.load['fy'], list) else ground_truth.load['fy']
    gt_x = torch.tensor(ground_truth.load['x']) if isinstance(ground_truth.load['x'], list) else ground_truth.load['x']
    gt_y = torch.tensor(ground_truth.load['y']) if isinstance(ground_truth.load['y'], list) else ground_truth.load['y']
    
    mo_fx = torch.tensor(model_output.load['fx']) if isinstance(model_output.load['fx'], list) else model_output.load['fx']
    mo_fy = torch.tensor(model_output.load['fy']) if isinstance(model_output.load['fy'], list) else model_output.load['fy']
    mo_x = torch.tensor(model_output.load['x']) if isinstance(model_output.load['x'], list) else model_output.load['x']
    mo_y = torch.tensor(model_output.load['y']) if isinstance(model_output.load['y'], list) else model_output.load['y']
    
    diff_load_x = mo_fx - gt_fx
    diff_load_y = mo_fy - gt_fy
    diff_load_x_coord = mo_x - gt_x
    diff_load_y_coord = mo_y - gt_y

    # Topology - convert to numpy for matplotlib
    plots[0,0] = axarr[0,0].imshow(_to_numpy(ground_truth.optimal_topology), 'gray_r',  vmin=0,  vmax=1)
    plots[1,0] = axarr[1,0].imshow(_to_numpy(model_output.optimal_topology), 'gray_r',  vmin=0,  vmax=1)
    plots[2,0] = axarr[2,0].imshow(_to_numpy(diff_topology),                 'seismic', vmin=-1, vmax=1)

    # Stress and Strain fields
    fields = [(ground_truth.stress_field, model_output.stress_field, diff_stress, 1),
              (ground_truth.strain_field, model_output.strain_field, diff_strain, 2)]
    
    for gt_field, mo_field, diff_field, col in fields:
        # Convert to numpy for matplotlib (handles both tensors and numpy arrays)
        gt_field_np = _to_numpy(gt_field)
        mo_field_np = _to_numpy(mo_field)
        diff_field_np = _to_numpy(diff_field)
        
        gt_scale = max(abs(gt_field_np.min()), abs(gt_field_np.max()))
        mo_scale = max(abs(mo_field_np.min()), abs(mo_field_np.max()))
        df_scale = max(abs(diff_field_np.min()), abs(diff_field_np.max()))
        
        plots[0,col] = axarr[0,col].imshow(gt_field_np, 'seismic', vmin=-gt_scale, vmax=gt_scale)
        plots[1,col] = axarr[1,col].imshow(mo_field_np, 'seismic', vmin=-mo_scale, vmax=mo_scale)
        plots[2,col] = axarr[2,col].imshow(diff_field_np, 'seismic', vmin=-df_scale, vmax=df_scale)

    # Text plots (VF and Load) with axes removal
    text_data = [
        (f'{ground_truth.volume_fraction:.3f}', f'{model_output.volume_fraction:.3f}', f'{diff_vf:.3f}', 3, 24),
        (f'X: {float(gt_x):.1f}\nY: {float(gt_y):.1f}\nFx: {float(gt_fx):.1f}\nFy: {float(gt_fy):.1f}',
         f'X: {float(mo_x):.1f}\nY: {float(mo_y):.1f}\nFx: {float(mo_fx):.1f}\nFy: {float(mo_fy):.1f}',
         f'X: {float(diff_load_x_coord):.1f}\nY: {float(diff_load_y_coord):.1f}\nFx: {float(diff_load_x):.1f}\nFy: {float(diff_load_y):.1f}', 4, 18)
    ]
    
    for gt_text, mo_text, diff_text, col, fontsize in text_data:
        for i, text in enumerate([gt_text, mo_text, diff_text]):
            axarr[i,col].text(0.5, 0.5, text, ha='center', va='center', fontsize=fontsize, transform=axarr[i,col].transAxes)
            # Remove axes
            axarr[i,col].set_xticks([])
            axarr[i,col].set_yticks([])
            for spine in axarr[i,col].spines.values():
                spine.set_visible(False)

    # Formatting and titles
    titles = ["Optimal Topology", "Von Mises Stress", "Strain Energy Density", "Volume Fraction", "Load Information"]
    ylabels = ["Input Data", "Output Data", "Difference"]
    
    for axs in axarr:
        for ax in axs:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_aspect('equal')
    
    font_dict = {'fontsize': 16}
    for j, title in enumerate(titles):
        if j in [3, 4]:  # Volume Fraction and Load Information - position manually lower
            axarr[0,j].text(0.5, 1, title, ha='center', va='top', fontdict=font_dict, transform=axarr[0,j].transAxes)
        else:
            axarr[0,j].set_title(title, fontdict=font_dict)
    for i, ylabel in enumerate(ylabels):
        axarr[i,0].set_ylabel(ylabel, fontdict=font_dict)
    
    # Add colorbars
    for i in range(3):
        for j in range(5):
            if plots[i,j] is not None:
                plt.colorbar(plots[i,j], ax=axarr[i,j], shrink=0.8)

    if save_name:
        plt.savefig(save_name)


# Single-field image columns, in display order. A column is included only when
# at least one plotted sample has the field. Append a (field_name, title) pair
# to add another image column. Columns that are not one image are added in
# _plot_column_specs.
_OPTIONAL_FIELD_COLUMNS = (
    ("stress_field", "Von Mises Stress"),
    ("strain_field", "SE Density"),
)


def _samples_have_field(samples, ids, field_name):
    return any(samples[sample_id].has(field_name) for sample_id in ids)


def _plot_column_specs(gt_samples, ids_to_plot, include_model):
    """Column specs for one figure, left to right.

    Each spec is ("image", field_name, title), ("trajectory",), ("topology",),
    ("model",), or ("difference",). Image and trajectory columns are omitted
    when the field is absent.
    """
    specs = [
        ("image", field_name, title)
        for field_name, title in _OPTIONAL_FIELD_COLUMNS
        if _samples_have_field(gt_samples, ids_to_plot, field_name)
    ]
    if _samples_have_field(gt_samples, ids_to_plot, "density_trajectory"):
        specs.append(("trajectory",))
    specs.append(("topology",))
    if include_model:
        specs.extend((("model",), ("difference",)))
    return specs


def _draw_image_column(ax, sample, field_name, title, font_dict):
    ax.set_title(title, fontdict=font_dict)
    if not sample.has(field_name):
        return
    artist = _physical_imshow(ax, getattr(sample, field_name), sample)
    plt.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)


def _draw_topology_column(ax, sample, thresh, font_dict):
    if sample.has("optimal_topology"):
        _physical_imshow(ax, _topology_for_plot(sample, thresh=thresh), sample, "gray_r")
        compliance_text = f", C: {round(float(sample.compliance), 2)}" if sample.has("compliance") else ""
        ax.set_title(f"Ground Truth{compliance_text}", fontdict=font_dict)
    else:
        ax.text(0.5, 0.5, "No Topology Data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Ground Truth", fontdict=font_dict)


def _trajectory_frames(sample):
    """Density frames as a float array of shape (n_steps, nely, nelx), or None."""
    if not sample.has("density_trajectory"):
        return None
    frames = np.array(_to_numpy(sample.density_trajectory), dtype=float, copy=True)
    if frames.ndim != 3 or frames.shape[0] == 0:
        return None
    np.clip(frames, 0.0, 1.0, out=frames)
    return frames


def _draw_trajectory_column(ax, sample, font_dict):
    """Draw the first trajectory frame. Returns (artist, frames) when present."""
    ax.set_title("Trajectory", fontdict=font_dict)
    frames = _trajectory_frames(sample)
    if frames is None:
        return None
    artist = _physical_imshow(ax, frames[0], sample, "gray_r", vmin=0, vmax=1)
    return artist, frames


_TRAJECTORY_FRAME_MS = 80


def _attach_trajectory_animation(fig, trajectory_artists):
    """Animate the trajectory column. Shorter rows hold their last frame."""
    n_frames = max(frames.shape[0] for _, frames in trajectory_artists)

    def update(frame_index):
        for artist, frames in trajectory_artists:
            artist.set_data(frames[min(frame_index, frames.shape[0] - 1)])
        return [artist for artist, _ in trajectory_artists]

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=_TRAJECTORY_FRAME_MS,
        blit=False,
        repeat=True,
    )
    fig._density_trajectory_anim = anim
    return anim


def _notebook_kernel():
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    config = getattr(ip, "config", None)
    return config is not None and "IPKernelApp" in config


def _display_trajectory_animation(anim):
    """Play the figure. A notebook gets an HTML player; a GUI backend plays live.

    Returns True when an HTML player was shown, so the caller can skip plt.show.
    """
    backend = plt.get_backend().lower()
    live = any(name in backend for name in ("tkagg", "qt", "gtk", "wx", "macosx", "nbagg", "webagg", "ipympl"))
    if live:
        return False
    if _notebook_kernel():
        from IPython.display import HTML, display
        display(HTML(anim.to_jshtml()))
        if anim.event_source is not None:
            anim.event_source.stop()
        return True
    if anim.event_source is not None:
        anim.event_source.stop()
    return False


def _draw_model_output_column(ax, gt_sample, model_sample, thresh, font_dict):
    if model_sample is None:
        return
    model_topology = _topology_for_plot(model_sample, thresh=thresh)
    _physical_imshow(ax, model_topology, gt_sample, "gray_r")
    compliance_text = f", C: {round(float(model_sample.compliance), 2)}" if model_sample.has("compliance") else ""
    ax.set_title(f"Model Output{compliance_text}", fontdict=font_dict)


def _draw_difference_column(ax, gt_sample, model_sample, thresh, font_dict):
    if model_sample is None:
        return
    model_topology = _topology_for_plot(model_sample, thresh=thresh)
    diff_topology = model_topology - gt_sample.optimal_topology
    _physical_imshow(ax, diff_topology, gt_sample, "bwr_r", vmin=-1, vmax=1)
    ax.set_title("Model Difference", fontdict=font_dict)


def _draw_row_volume_fraction(ax, sample, font_dict):
    """Row label. Kept off the strain column so it remains when that column is omitted."""
    if not sample.has("volume_fraction"):
        return
    ax.annotate(
        f"VF: {round(float(sample.volume_fraction), 4)}",
        xy=(0, 0.5),
        xycoords="axes fraction",
        xytext=(-8, 0),
        textcoords="offset points",
        rotation=90,
        ha="center",
        va="center",
        fontsize=font_dict["fontsize"],
        annotation_clip=False,
    )


def _reserve_row_label_margin(fig):
    """Leave room on the left for the vertical volume-fraction row label."""
    needed = 0.45 / fig.get_figwidth()
    fig.subplots_adjust(left=max(fig.subplotpars.left, needed))


def plot_data(data_to_plot, model_data=None, count=None, plotids=None, save_name=None, thresh=False):
    """Plots the input data using TopOptSample objects.

    Each sample is one row. Stress and strain columns are included only when
    the plotted samples carry those fields. Topology is always shown. A
    trajectory column is added when density_trajectory is present, and that
    column is animated. Model output and difference columns are added when
    model_data is given. Volume fraction is the row label.

    Parameters
    ----------
    
    data_to_plot: TopOptSample, TopOptBatch, or list of TopOptSample
               | Ground truth TopOptSample objects containing topology optimization data
               | Each sample should have stress_field, strain_field, optimal_topology, 
               | load, boundary_conditions, volume_fraction, compliance, and load_type
    model_data: TopOptSample, TopOptBatch, or list of TopOptSample (optional)
               | Model output TopOptSample objects for comparison
               | If provided, will create comparison plots showing ground truth, model output, and difference
    count: int (optional)
               | Number of samples to plot
    plotids: int (optional)
               | Which samples to plot
    save_name: string (optional)
               | Name to save the image as. A .gif path saves the animation
               | when a trajectory column is present. If None, don't save anything
    thresh: bool or float (optional)
               | If True, threshold displayed topologies at 0.5.
               | If a float, threshold at that value (rho >= thresh).
               | If False or None, plot the continuous field.
               | Display-only; does not mutate sample.optimal_topology

    Returns
    -------
    f: matplotlib figure
       The trajectory animation, when present, is f._density_trajectory_anim.
    axarr: matplotlib axes array

    Notes
    -----
    Also use plot_from_npz to manage the loading of files before plotting
    """

    # Get what type of data is being fed for ground truth
    if isinstance(data_to_plot, TopOptBatch):
        gt_samples = data_to_plot.samples
    elif isinstance(data_to_plot, TopOptSample):
        gt_samples = [data_to_plot]
    elif isinstance(data_to_plot, list):
        gt_samples = data_to_plot
    else:
        raise TypeError("data_to_plot must be TopOptSample, TopOptBatch, or list of TopOptSample")
    
    # Get what type of data is being fed for model data
    model_samples = None
    if model_data is not None:
        if isinstance(model_data, TopOptBatch):
            model_samples = model_data.samples
        elif isinstance(model_data, TopOptSample):
            model_samples = [model_data]
        elif isinstance(model_data, list):
            model_samples = model_data
        else:
            raise TypeError("model_data must be TopOptSample, TopOptBatch, or list of TopOptSample")
    
    if count is not None:
        num_samples = count
    elif plotids is not None:
        num_samples = len(plotids)
    else:
        num_samples = len(gt_samples)

    if num_samples > 50:
        num_samples = 50

    ids_to_plot = plotids if plotids is not None else range(num_samples)
    column_specs = _plot_column_specs(gt_samples, ids_to_plot, model_samples is not None)
    num_plots = len(column_specs)
    f, axarr = plt.subplots(num_samples, num_plots, squeeze=False)

    f.set_figheight(4*num_samples)
    f.set_figwidth(4*num_plots)
    font_dict = {'fontsize':11}
    if _samples_have_field(gt_samples, ids_to_plot, "volume_fraction"):
        _reserve_row_label_margin(f)

    trajectory_artists = []
    for i, id in enumerate(ids_to_plot):
        sample = gt_samples[id]
        model_sample = None
        if model_samples is not None and id < len(model_samples):
            model_sample = model_samples[id]

        for j, spec in enumerate(column_specs):
            kind = spec[0]
            ax = axarr[i, j]
            if kind == "image":
                _, field_name, title = spec
                _draw_image_column(ax, sample, field_name, title, font_dict)
            elif kind == "topology":
                _draw_topology_column(ax, sample, thresh, font_dict)
            elif kind == "trajectory":
                drawn = _draw_trajectory_column(ax, sample, font_dict)
                if drawn is not None:
                    trajectory_artists.append(drawn)
            elif kind == "model":
                _draw_model_output_column(ax, sample, model_sample, thresh, font_dict)
            elif kind == "difference":
                _draw_difference_column(ax, sample, model_sample, thresh, font_dict)

        # Set load type label based on sample's load_type
        axarr[i,0].set_xlabel(sample.load_type.capitalize())

        width, height = _domain_size(sample)
        pad_x, pad_y = 0.08 * width, 0.08 * height
        for j in range(num_plots): # Plot load arrow on all plots
            _quiver_loads(axarr[i, j], sample, color='r')
            axarr[i, j].set_axis_off()
            axarr[i, j].set_xlim(-pad_x, width + pad_x)
            axarr[i, j].set_ylim(-pad_y, height + pad_y)
            axarr[i, j].set_aspect('equal')

        _draw_row_volume_fraction(axarr[i, 0], sample, font_dict)

        locations = _physical_bc_locations(sample)
        if locations is not None:
            x_constraint_locs = locations['x']
            if x_constraint_locs:
                x_coords = [coord[0] for coord in x_constraint_locs]
                y_coords = [coord[1] for coord in x_constraint_locs]
                
                for j in range(num_plots):
                    axarr[i, j].scatter(x_coords, y_coords, marker='s', c='red', s=50, label='X-constraints', alpha=0.8)
            
            y_constraint_locs = locations['y']
            if y_constraint_locs:
                x_coords = [coord[0] for coord in y_constraint_locs]
                y_coords = [coord[1] for coord in y_constraint_locs]
                
                for j in range(num_plots):
                    axarr[i, j].scatter(x_coords, y_coords, marker='^', c='blue', s=50, label='Y-constraints', alpha=0.8)

    anim = _attach_trajectory_animation(f, trajectory_artists) if trajectory_artists else None
    played_in_notebook = _display_trajectory_animation(anim) if anim is not None else False
    if save_name:
        if anim is not None and str(save_name).lower().endswith(".gif"):
            anim.save(save_name, writer="pillow", fps=1000 / _TRAJECTORY_FRAME_MS)
        else:
            plt.savefig(save_name)
    if not played_in_notebook:
        plt.show()
    return f, axarr

def plot_data_nicer(data_to_plot, model_data=None, count=None, plotids=None, save_name=None, thresh=False):
    """Plot data with nicer formatting and less features than plot_data."""

    # Get what type of data is being fed for ground truth
    if isinstance(data_to_plot, TopOptBatch):
        gt_samples = data_to_plot.samples
    elif isinstance(data_to_plot, TopOptSample):
        gt_samples = [data_to_plot]
    elif isinstance(data_to_plot, list):
        gt_samples = data_to_plot
    else:
        raise TypeError("data_to_plot must be TopOptSample, TopOptBatch, or list of TopOptSample")

    # Get what type of data is being fed for model data
    model_samples = None
    if model_data is not None:
        if isinstance(model_data, TopOptBatch):
            model_samples = model_data.samples
        elif isinstance(model_data, TopOptSample):
            model_samples = [model_data]
        elif isinstance(model_data, list):
            model_samples = model_data
        else:
            raise TypeError("model_data must be TopOptSample, TopOptBatch, or list of TopOptSample")

    if count is not None:
        num_samples = count
    elif plotids is not None:
        num_samples = len(plotids)
    else:
        num_samples = len(gt_samples)

    if num_samples > 50:
        num_samples = 50

    # Only plot topology: GT, (optionally Model Output, Model Difference)
    num_plots = 1
    if model_samples is not None:
        num_plots += 2  # Add model output and difference

    f, axarr = plt.subplots(num_samples, num_plots)
    if num_samples == 1:
        axarr = axarr[None, :]

    # Larger font for all text
    font_dict = {'fontsize': 16}
    tick_labelsize = 14
    plt.rcParams.update({'font.size': 14})

    f.set_figheight(4*num_samples)
    f.set_figwidth(4*num_plots)

    ids_to_plot = plotids if plotids is not None else range(num_samples)
    for i, id in enumerate(ids_to_plot):
        sample = gt_samples[id]

        # Plot topology
        if sample.has('optimal_topology'):
            _physical_imshow(axarr[i, 0], _topology_for_plot(sample, thresh=thresh), sample, 'gray_r')
            compliance_text = f', C: {round(float(sample.compliance), 2)}' if sample.has('compliance') else ''
            axarr[i, 0].set_title(f'Ground Truth{compliance_text}', fontdict=font_dict)
        else:
            axarr[i, 0].text(0.5, 0.5, 'No Topology Data', ha='center', va='center', transform=axarr[i, 0].transAxes, fontsize=font_dict['fontsize']+2)
            axarr[i, 0].set_title('Ground Truth', fontdict=font_dict)

        # Optionally annotate with load type as xlabel
        if hasattr(sample, 'load_type') and sample.load_type is not None:
            axarr[i, 0].set_xlabel(sample.load_type.capitalize(), fontsize=font_dict['fontsize'])

        # Model comparison (if we have model output samples)
        if model_samples is not None and id < len(model_samples):
            model_sample = model_samples[id]
            model_topology = _topology_for_plot(model_sample, thresh=thresh)
            _physical_imshow(axarr[i, 1], model_topology, sample, 'gray_r')
            compliance_text = f', C: {round(float(model_sample.compliance), 2)}' if model_sample.has('compliance') else ''
            axarr[i, 1].set_title(f'Model Output{compliance_text}', fontdict=font_dict)

            # Calculate difference and vol frac error
            diff_topology = model_topology - sample.optimal_topology

            gt_vf = sample.volume_fraction
            model_vf = np.mean(_to_numpy(model_topology))
            vf_error = abs(model_vf - gt_vf)/gt_vf*100
            _physical_imshow(axarr[i, 2], diff_topology, sample, 'bwr_r', vmin=-1, vmax=1)
            # Remove bold from Model Difference title (no 'fontweight': 'bold')
            axarr[i, 2].set_title(
                f'Model Difference\nVol. Frac Error: {vf_error:.3f}',
                fontdict={'fontsize': font_dict['fontsize']-2}
            )

        # Draw border manually on all axes before any scatter overlays
        width, height = _domain_size(sample)
        ylim_pad = 0.12 * height
        for j in range(num_plots):
            # y=0 is the physical bottom. Outward arrows at the bottom edge extend
            # to y < 0; extend ymin so quiver is not clipped by the figure.
            rect = Rectangle(
                (0, 0),
                width,
                height,
                fill=False,
                linewidth=2,
                edgecolor='black',
                zorder=10
            )
            axarr[i, j].add_patch(rect)
            axarr[i, j].set_axis_off()
            axarr[i, j].set_xlim(0, width)
            axarr[i, j].set_ylim(-ylim_pad, height)
            axarr[i, j].set_aspect('equal')

        # Optionally plot load arrows on all axes
        for j in range(num_plots):
            _quiver_loads(
                axarr[i, j],
                sample,
                color='mediumseagreen',
                linewidth=0.8,
                headwidth=6,
                headlength=6,
                headaxislength=7,
                clip_on=False,
            )

        locations = _physical_bc_locations(sample)
        if locations is not None:
            all_coords = []
            all_coords += locations.get('x', [])
            all_coords += locations.get('y', [])
            if all_coords:
                x_coords = [coord[0] for coord in all_coords]
                y_coords = [coord[1] for coord in all_coords]
                for j in range(num_plots):
                    axarr[i, j].scatter(
                        x_coords, y_coords,
                        marker='s',  # squares now
                        c='mediumseagreen',
                        s=50,        # a bit larger
                        alpha=0.87
                        # edgecolor removed to eliminate border
                        # linewidths removed to eliminate border
                    )

    # Export to PDF file
    if save_name is not None:
        if not save_name.lower().endswith('.pdf'):
            save_name = save_name + '.pdf'
        plt.savefig(save_name, format='pdf')
    else:
        # always export a pdf even if not specified
        plt.savefig('topopt_plot.pdf', format='pdf')
    plt.show()
    return f, axarr


def _samples_from_data(data):
    if isinstance(data, TopOptBatch):
        return data.samples
    if isinstance(data, TopOptSample):
        return [data]
    if isinstance(data, list):
        return data
    raise TypeError("Expected TopOptSample, TopOptBatch, or list of TopOptSample")


def _topology_for_plot(sample, thresh=False):
    """Return topology for display. thresh=None/False keeps the field; True uses 0.5."""
    ot = sample.optimal_topology
    if thresh is None or thresh is False:
        return ot
    cutoff = 0.5 if thresh is True else float(thresh)
    if isinstance(ot, th.Tensor):
        return (ot >= cutoff).to(ot.dtype)
    return (np.asarray(ot) >= cutoff).astype(np.float32)


def _decorate_nicer_topology_row(axarr_row, gt_sample):
    num_plots = len(axarr_row)
    width, height = _domain_size(gt_sample)
    ylim_pad = 0.12 * height
    for j in range(num_plots):
        rect = Rectangle(
            (0, 0),
            width,
            height,
            fill=False,
            linewidth=2,
            edgecolor="black",
            zorder=10,
        )
        axarr_row[j].add_patch(rect)
        axarr_row[j].set_axis_off()
        axarr_row[j].set_xlim(0, width)
        axarr_row[j].set_ylim(-ylim_pad, height)
        axarr_row[j].set_aspect("equal")

    for j in range(num_plots):
        _quiver_loads(
            axarr_row[j],
            gt_sample,
            color="mediumseagreen",
            linewidth=0.8,
            headwidth=6,
            headlength=6,
            headaxislength=7,
            clip_on=False,
        )

    locations = _physical_bc_locations(gt_sample)
    if locations is not None:
        all_coords = []
        all_coords += locations.get("x", [])
        all_coords += locations.get("y", [])
        if all_coords:
            x_coords = [coord[0] for coord in all_coords]
            y_coords = [coord[1] for coord in all_coords]
            for j in range(num_plots):
                axarr_row[j].scatter(
                    x_coords,
                    y_coords,
                    marker="s",
                    c="mediumseagreen",
                    s=50,
                    alpha=0.87,
                )


def plot_three_model_comparison_nicer(
    data_to_plot,
    model_data_a,
    model_data_b,
    model_a_name="Model A",
    model_b_name="Model B",
    count=None,
    plotids=None,
    save_name=None,
    thresh=False,
    show=True,
):
    """Plot ground truth and two model outputs side by side (plot_data_nicer styling)."""
    gt_samples = _samples_from_data(data_to_plot)
    model_a_samples = _samples_from_data(model_data_a)
    model_b_samples = _samples_from_data(model_data_b)

    if count is not None:
        num_samples = count
    elif plotids is not None:
        num_samples = len(plotids)
    else:
        num_samples = len(gt_samples)

    if num_samples > 50:
        num_samples = 50

    num_plots = 3
    f, axarr = plt.subplots(num_samples, num_plots)
    if num_samples == 1:
        axarr = axarr[None, :]

    font_dict = {"fontsize": 16}
    plt.rcParams.update({"font.size": 14})
    f.set_figheight(4 * num_samples)
    f.set_figwidth(4 * num_plots)

    column_specs = [
        ("Ground Truth", lambda sid: gt_samples[sid]),
        (model_a_name, lambda sid: model_a_samples[sid]),
        (model_b_name, lambda sid: model_b_samples[sid]),
    ]

    ids_to_plot = plotids if plotids is not None else range(num_samples)
    for i, sample_id in enumerate(ids_to_plot):
        gt_sample = gt_samples[sample_id]
        for col, (title, get_sample) in enumerate(column_specs):
            sample = get_sample(sample_id)
            if sample.has("optimal_topology"):
                topology = _topology_for_plot(sample, thresh=thresh and col > 0)
                _physical_imshow(axarr[i, col], topology, gt_sample, "gray_r")
                axarr[i, col].set_title(title, fontdict=font_dict)
            else:
                axarr[i, col].text(
                    0.5,
                    0.5,
                    "No Topology Data",
                    ha="center",
                    va="center",
                    transform=axarr[i, col].transAxes,
                    fontsize=font_dict["fontsize"] + 2,
                )
                axarr[i, col].set_title(title, fontdict=font_dict)

        if hasattr(gt_sample, "load_type") and gt_sample.load_type is not None:
            axarr[i, 0].set_xlabel(gt_sample.load_type.capitalize(), fontsize=font_dict["fontsize"])

        _decorate_nicer_topology_row(axarr[i], gt_sample)

    if save_name is not None:
        if not save_name.lower().endswith(".pdf"):
            save_name = save_name + ".pdf"
        plt.savefig(save_name, format="pdf", bbox_inches="tight")
    if show:
        plt.show()
    return f, axarr


def plot_from_pkl(gt_file_path, output_file_path, count=None, plotids=None, save_name=None, thresh=False):
    """Plot the model output from pkl files

    Parameters
    ----------
    gt_file_path: string
               | Path to the ground truth sample pickle file
    output_file_path: string
               | Path to the model sample pickle file
    count: int (optional)
               | Number of samples to plot
    plotids: int (optional)
               | Which samples to plot
    save_name: string (optional)
               | Name to save the image as. If None, don't save anything
    thresh: bool or float (optional)
               | If True, threshold displayed topologies at 0.5.
               | If a float, threshold at that value (rho >= thresh).
               | If False or None, plot the continuous field.
               | Display-only; does not mutate sample.optimal_topology

    Returns
    -------
    f: matplotlib figure
    axarr: matplotlib axes array
    """
    with open(gt_file_path, 'rb') as f:
        gt_samples = pickle.load(f)
    with open(output_file_path, 'rb') as f:
        output_samples = pickle.load(f)
    return plot_data(gt_samples, output_samples, count=count, plotids=plotids, save_name=save_name, thresh=thresh)

#MAIN===============================================================================================
if __name__ == "__main__": raise RuntimeError("Please don't execute me! I have so much to live for~~~")