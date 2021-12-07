import pandas as pd
import numpy as np

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import dynspread, rasterize, shade
import plotly.graph_objects as go

import ext.utils as utils
import alphatims.bruker

hv.extension('plotly')

##### preprocessing functions

def sum_binned_data(rt_values, intensity_values, min_value, max_value, bins):
    """ 
    Sum the intensities over retention time
    """
    bin_delta = (max_value - min_value) / bins
    bins_array = np.linspace(min_value, max_value, bins+1)
    rt_bins = ((rt_values - min_value) / bin_delta).astype(np.int64)
    intensity_bins = np.zeros(bins+1, dtype=np.int64)
    for rt_bin, intensity in zip(rt_bins, intensity_values):
        intensity_bins[rt_bin] += intensity
        bin_centers = bins_array[1:] - bin_delta/2
    return bin_centers, intensity_bins[1:]


##### these plotting functions are taken from the AlphaViz package (https://github.com/MannLabs/alphaviz)
def plot_xic(
    df: pd.DataFrame, 
    xic_mz: float,
    mz_tol_value: int,
    rt_min: float,
    rt_max: float,
    bins: int,
    width: int = 900,
    height: int = 500
):
    """Create an Extracted ion chromatogram (XIC) for the selected m/z.

    Parameters
    ----------
    df : pandas Dataframe
        A table with the extracted MS1 data.
    xic_mz : float
        An m/z value of the precursor/feature that should be used for the XIC.
    mz_tol_value : int
        An m/z tolerance value in ppm.
    rt_min : float
        Start of the retention time window.
    rt_max : float
        End of the retention time window.
    bins: int
        The number of bins for the plot's creation.
    width : int
        The width of the plot.
        Default is 900.
    height : int
        The height of the plot.
        Default is 500.

    Returns
    -------
    a Plotly line plot
        The line plot showing XIC for the selected m/z of the provided dataset.
    """
    fig = go.Figure()
    
    xic_mz_low_mz = xic_mz / (1 + mz_tol_value / 10**6)
    xic_mz_high_mz = xic_mz * (1 + mz_tol_value / 10**6)

    d = df[(df.mz >= xic_mz_low_mz) & (df.mz <= xic_mz_high_mz)]

    bin_centers, intensity_bins = sum_binned_data(d.RT, d.intensity, rt_min, rt_max, bins)
    
    fig.add_trace(
        go.Scatter(
            x=bin_centers,
            y=intensity_bins,
            hovertemplate='<b>RT:</b> %{x};<br><b>Intensity:</b> %{y}.',
        )
    )
    
    fig.update_layout(
        title=dict(
            text=f'XIC for the m/z = {xic_mz}, m/z tolerance = {mz_tol_value} ppm.',
            font=dict(
                size=16,
            ),
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='RT, min',
            titlefont_size=14,
            tickmode = 'auto',
            tickfont_size=14,
        ),
        yaxis=dict(
            title='Intensity',
        ),
        hovermode="x",
        template="plotly_white",
        width=width,
        height=height
    )

    fig.update_xaxes(range=[0, df.RT.max()])
    return fig

def plot_heatmap_ms1(
    df: pd.DataFrame,
    x_axis_label: str = "m/z, Th",
    y_axis_label: str = "RT, min",
    z_axis_label: str = "Intensity",
    title: str = "",
    width: int = 700,
    height: int = 400,
    background_color: str = "black",
    colormap: str = "fire",
):
    """Create a heatmap showing a correlation of m/z and ion mobility with color coding for signal intensity.

    Parameters
    ----------
    df : pandas Dataframe
        A dataframe obtained by slicing an alphatims.bruker.TimsTOF object.
    x_axis_label : str
        An x-axis label.
        Default is "m/z, Th".
    y_axis_label : str
        An y-axis label.
        Default is "Inversed IM, V·s·cm\u207B\u00B2".
    z_axis_label : str
        An z-axis label using for the coloring.
        Default is "Intensity".
    title: str
        The title of the plot.
         Default is "".
    width : int
        The width of the plot.
        Default is 700.
    height : int
        The height of the plot.
        Default is 400.
    background_color : str
        The background color of the plot.
        Default is "black".
    colormap : str
        The name of the colormap in Plotly.
        Default is "fire".

    Returns
    -------
    a Plotly scatter plot
        The scatter plot showing all found features in the specified rt and m/z ranges of the provided dataset.
    """
    labels = {
        'm/z, Th': "mz",
        'RT, min': "RT",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
        'Intensity': "intensity",
    }
    x_dimension = labels[x_axis_label]
    y_dimension = labels[y_axis_label]
    z_dimension = labels[z_axis_label]

    def hook(plot, element):
        plot.handles['layout']['xaxis']['gridcolor'] = background_color
        plot.handles['layout']['yaxis']['gridcolor'] = background_color

    opts_ms1=dict(
        width=width,
        height=height,
        title=title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        bgcolor=background_color,
        hooks=[hook],
    )
    dmap = hv.DynamicMap(
        hv.Points(
            df,
            [x_dimension, y_dimension],
            z_dimension
        )
    )
    agg = rasterize(
        dmap,
        width=width,
        height=height,
        aggregator='sum'
    )
    fig = dynspread(
        shade(
            agg,
            cmap=colormap
        )
    ).opts(plot=opts_ms1)

    return fig

def plot_tic(
    df: pd.DataFrame, 
    title: str, 
    width: int = 900,
    height: int = 500
):
    """Create a total ion chromatogram (TIC) and Base Peak chromatogram (BPI) for the MS1 data.

    Parameters
    ----------
    df : pandas Dataframe
        A table with the extracted MS1 data.
    title : str
        The title of the plot.
    width : int
        The width of the plot.
        Default is 1000.
    height : int
        The height of the plot.
        Default is 320.

    Returns
    -------
    a Plotly line plot
        The line plot containing TIC and BPI for MS1 data of the provided dataset.
    """
    fig = go.Figure()
    
    total_ion_col = ['RT', 'summed_intensity']
    base_peak_col = ['RT', 'max_intensity']
    
    for chrom_type in ['TIC MS1', 'BPI MS1']:
        if chrom_type == 'TIC MS1':
            data = df[total_ion_col]
        elif chrom_type == 'BPI MS1':
            data = df[base_peak_col]
        fig.add_trace(
            go.Scatter(
                x=data.iloc[:, 0],
                y=data.iloc[:, 1],
                name=chrom_type,
                hovertemplate='<b>RT:</b> %{x};<br><b>Intensity:</b> %{y}.',
            )
        )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(
                size=16,
            ),
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='RT, min',
            titlefont_size=14,
            tickmode = 'auto',
            tickfont_size=14,
        ),
        yaxis=dict(
            title='Intensity',
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        legend_title_text='Select:',
        hovermode="x",
        template="plotly_white",
        width=width,
        height=height
    )

    fig.update_xaxes(range=[0, df.RT.max()])
    return fig

def plot_heatmap(
    df: pd.DataFrame,
    x_axis_label: str = "m/z, Th",
    y_axis_label: str = "Inversed IM, V·s·cm\u207B\u00B2",
    z_axis_label: str = "Intensity",
    title: str = "",
    width: int = 700,
    height: int = 400,
    background_color: str = "black",
    colormap: str = "fire",
):
    """Create a heatmap showing a correlation of m/z and ion mobility with color coding for signal intensity.

    Parameters
    ----------
    df : pandas Dataframe
        A dataframe obtained by slicing an alphatims.bruker.TimsTOF object.
    x_axis_label : str
        An x-axis label.
        Default is "m/z, Th".
    y_axis_label : str
        An y-axis label.
        Default is "Inversed IM, V·s·cm\u207B\u00B2".
    z_axis_label : str
        An z-axis label using for the coloring.
        Default is "Intensity".
    title: str
        The title of the plot.
         Default is "".
    width : int
        The width of the plot.
        Default is 700.
    height : int
        The height of the plot.
        Default is 400.
    background_color : str
        The background color of the plot.
        Default is "black".
    colormap : str
        The name of the colormap in Plotly.
        Default is "fire".

    Returns
    -------
    a Plotly scatter plot
        The scatter plot showing all found features in the specified rt and m/z ranges of the provided dataset.
    """
    labels = {
        'm/z, Th': "mz_values",
        'RT, min': "rt_values",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
        'Intensity': "intensity_values",
    }
    x_dimension = labels[x_axis_label]
    y_dimension = labels[y_axis_label]
    z_dimension = labels[z_axis_label]

    df["rt_values"] /= 60

    def hook(plot, element):
        plot.handles['layout']['xaxis']['gridcolor'] = background_color
        plot.handles['layout']['yaxis']['gridcolor'] = background_color

    opts_ms1=dict(
        width=width,
        height=height,
        title=title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        bgcolor=background_color,
        hooks=[hook],
    )
    dmap = hv.DynamicMap(
        hv.Points(
            df,
            [x_dimension, y_dimension],
            z_dimension
        )
    )
    agg = rasterize(
        dmap,
        width=width,
        height=height,
        aggregator='sum'
    )
    fig = dynspread(
        shade(
            agg,
            cmap=colormap
        )
    ).opts(plot=opts_ms1)

    return fig
    