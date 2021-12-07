import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import plotly.graph_objects as go
import plotly.subplots
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import dynspread, rasterize, shade, datashade

import ext.utils as utils
import alphatims.bruker

hv.extension('bokeh')

##### preprocessing functions

def data_preprocessing(
    ms2_data: pd.DataFrame,
    path_to_msms_file: str, 
    scan_number: int,
    exp_name: str,
    relative_intensity: bool = False,
    verbose: bool = True,
    mz_tol: float = 3.0, #ppm
):
    """Merge information about identified fragment ions from the specified experiment in MaxQuant output "msms.txt" 
    file with the raw data corresponding to the special scan number and return the output dataframe with the columns:
        - 'mz_values', 
        - 'intensity_values', 
        - 'ions'.

    Parameters
    ----------
    ms2_data : pandas Dataframe
        A table with the extracted MS2 data.
    path_to_msms_file : str
        The path to the MaxQuant output table 'msms.txt'.
    scan_number: int
        The number of the scan to extract from the data.
    exp_name: str
        The name of the experiment to extract from the "msms.txt" file.
    relative_intensity: bool 
        Whether to calculate the relative intensity. Default: False.
    verbose: bool 
        Whether to print the additional comments and output of the functions. Default: True.
    mz_tol: float 
        The mz tolerance value. Default: 3.0 ppm.

    Returns
    -------
    a pandas Dataframe
        The the output dataframe containing the columns:
        - 'mz_values', 
        - 'intensity_values', 
        - 'ions'.
    """
    # extract the necessary information from the msms.txt file.
    msms = pd.read_csv(path_to_msms_file, sep='\t', low_memory=False)
    sliced_msms = msms[(msms['Raw file'] == exp_name) & (msms['Scan number'] == scan_number)]
    exp_data = pd.DataFrame(
        data={
            'frag_type': sliced_msms['Matches'].values[0].split(';'),
            'frag_mz': [float(each) for each in sliced_msms['Masses'].values[0].split(';')],
    #         'm/z_err_Da': [float(each) for each in sliced_df['Mass Deviations [Da]'].values[0].split(';')]
        }
    )
    if verbose:
        print(f'The structure of the dataframe for the selected scan number {scan_number} and the experiment {exp_name} with the fragment type and fragment m/z columns.')
        print(exp_data.head())
    
    # slice raw data for the specified scan_number
    data = pd.DataFrame(
        data={
            'mz_values': ms2_data[ms2_data.scan == scan_number].mz_values.values[0],
            'intensity_values': ms2_data[ms2_data.scan == scan_number].intensity_values.values[0]
        }
    )
    data['ions'] = ''
        
    if relative_intensity:
        # To get the Relative abundance(%) of the ions instead of Absolute intensity weuse the MinMaxScaler from Sklearn.
        scaled_int = MinMaxScaler((0, 100)).fit_transform(data.intensity_values.values.reshape(-1, 1))
        data['intensity_values']= scaled_int.reshape(1, -1)[0]
    
    if verbose:
        print('\n')
        print(f'The sliced ms2 raw data for the selected scan number {scan_number}.')
        print(data.head())
        
    # match the raw data with the exp identified ion fragments
    for i, (frag, mz) in exp_data.iterrows():
        fr_mass_low, fr_mass_high = mz / (1 + mz_tol / 10**6), mz * (1 + mz_tol / 10**6)
        data.loc[(data.mz_values >= fr_mass_low) & (data.mz_values <= fr_mass_high), 'ions'] = frag
    if verbose:
        print('\n')
        print(f'The list of unique ions assigned to the raw data spectrum.')
        print(data.ions.unique().tolist())
    
    return data

def preprocess_prediction(
    path: str, 
    peptide_sequence: str,
    verbose: bool = True
):
    """Read and preprocess the output of the Prosit model to have the following columns in the dataframe:
        - 'RelativeIntensity', 
        - 'FragmentMz', 
        - 'StrippedPeptide', 
        - 'FragmentNumber',
        - 'FragmentType', 
        - 'FragmentCharge', 
        - 'FragmentLossType', 
        - 'ions'.

    Parameters
    ----------
    path : str
        The path to the Prosit output file.
    peptide_sequence: dict
        The peptide sequence.
    verbose: bool 
        Whether to print the additional comments and output of the functions. Default: True.

    Returns
    -------
    a pandas Dataframe
        The the output dataframe containing the columns:
            - 'RelativeIntensity', 
            - 'FragmentMz', 
            - 'StrippedPeptide', 
            - 'FragmentNumber',
            - 'FragmentType', 
            - 'FragmentCharge', 
            - 'FragmentLossType', 
            - 'ions'.
    """
    predicted_df = pd.read_csv(
        path, 
        usecols=[
            'RelativeIntensity', 'FragmentMz', 'StrippedPeptide', 'FragmentNumber', 'FragmentType', 'FragmentCharge', 'FragmentLossType'
        ]
    )
    predicted_df = predicted_df[
        (predicted_df.StrippedPeptide == peptide_sequence) & \
        (predicted_df.RelativeIntensity > 0)
    ]
    predicted_df.RelativeIntensity *= -100
    predicted_df['ions'] = predicted_df.apply(
        lambda x: f"{x.FragmentType}{x.FragmentNumber} +{x.FragmentCharge}" if x.FragmentCharge != 1 else f"{x.FragmentType}{x.FragmentNumber}", 
        axis=1
    )
    predicted_df['ions'] = predicted_df.apply(
        lambda x: f"{x.ions}-{x.FragmentLossType}" if x.FragmentLossType != 'noloss' else x.ions, 
        axis=1
    )
    if verbose:
        print(f"The structure of the dataframe for the predicted peptide {peptide_sequence} using Prosit model.")
        print(predicted_df.head())
    
    return predicted_df

def convert_col_to_bool_list(
    data: pd.DataFrame, 
    col: str, 
    sequence: str, 
    ion: str
):
    """Convert the peptide sequence into the list of boolens for the b-ions whether the peptide is breaking 
    after aligned amino acid or for the y-ion is breaking before aligned amino acid."""
    
    ions=[False] * (len(sequence))
    for each in data[data[col].str.contains(ion)][col]:
        if ion == 'b':
            ions[int(each.split('-')[0].replace(ion, '')) - 1] = True
        else:
            ions[-int(each.split('-')[0].replace(ion, ''))] = True
    return ions

mass_dict = utils.get_mass_dict(verbose=False)



##### these plotting functions are taken from the AlphaViz package (https://github.com/MannLabs/alphaviz) and modified

def plot_line(
    timstof_data,
    selected_indices: np.ndarray,
    label: str,
    remove_zeros: bool = False,
    trim: bool = True,
):
    """Plot an XIC as a lineplot.

    Parameters
    ----------
    timstof_data : alphatims.bruker.TimsTOF
        An alphatims.bruker.TimsTOF data object.
    selected_indices : np.ndarray
        The raw indices that are selected for this plot.
    label : str
        The label for the line plot.
    remove_zeros : bool
        If True, zeros are removed. Default: False.
    trim : bool
        If True, zeros on the left and right are trimmed. Default: True.

    Returns
    -------
    a Plotly line plot
        The XIC line plot.
    """
    axis_dict = {
        "rt": "RT, min",
        "intensity": "Intensity",
    }
    x_axis_label = axis_dict["rt"]
    y_axis_label = axis_dict["intensity"]
    labels = {
        'RT, min': "rt_values",
    }
    x_dimension = labels[x_axis_label]
    intensities = timstof_data.bin_intensities(selected_indices, [x_dimension])
    x_ticks = timstof_data.rt_values / 60
       
    non_zeros = np.flatnonzero(intensities)
    if len(non_zeros) == 0:
        x_ticks = np.empty(0, dtype=x_ticks.dtype)
        intensities = np.empty(0, dtype=intensities.dtype)
    else:
        if remove_zeros:
            x_ticks = x_ticks[non_zeros]
            intensities = intensities[non_zeros]
        elif trim:
            start = max(0, non_zeros[0] - 1)
            end = non_zeros[-1] + 2
            x_ticks = x_ticks[start: end]
            intensities = intensities[start: end]

    trace = go.Scatter(
        x=x_ticks,
        y=intensities,
        mode='lines',
        text = [f'{x_axis_label}'.format(i + 1) for i in range(len(x_ticks))],
        hovertemplate='<b>%{text}:</b> %{x};<br><b>Intensity:</b> %{y}.',
        name=label
    )
    return trace

def plot_elution_profile(
    timstof_data,
    peptide_info: dict,
    mass_dict: dict,
    mz_tol: float = 50,
    rt_tol: float = 30,
    im_tol: float = 0.05,
    title: str = "",
    width: int = 900,
    height: int = 400
):
    """Plot an elution profile plot for the specified precursor and all his identified fragments.

    Parameters
    ----------
    timstof_data : alphatims.bruker.TimsTOF
        An alphatims.bruker.TimsTOF data object.
    peptide_info : dict
        Peptide information including sequence, fragments' patterns, rt, mz and im values.
    mass_dict : dict
        The basic mass dictionaty with the masses of all amino acids and modifications.
    mz_tol: float 
        The mz tolerance value. Default: 50 ppm.
    rt_tol: float 
        The rt tolerance value. Default: 30 ppm.
    im_tol: float 
        The im tolerance value. Default: 0.05 ppm.
    title : str
        The title of the plot.
    width : int
        The width of the plot. Default: 900.
    height : int
        The height of the plot. Default: 400.

    Returns
    -------
    a Plotly line plot
        The elution profile plot in retention time dimension for the specified peptide and all his fragments.
    """
    x_axis_label = "rt"
    y_axis_label = "intensity"
    
    # predict the theoretical fragments using the Alphapept get_fragmass() function.
    frag_masses, frag_type = utils.get_fragmass(
        parsed_pep=list(peptide_info['sequence']), 
        mass_dict=mass_dict
    )
    peptide_info['fragments'] = {
        (f"b{key}" if key>0 else f"y{-key}"):value for key,value in zip(frag_type, frag_masses)
    }
    
    # slice the data using the rt_tol, im_tol and mz_tol values
    rt_slice = slice(peptide_info['rt'] - rt_tol, peptide_info['rt'] + rt_tol)
    im_slice = slice(peptide_info['im'] - im_tol, peptide_info['im'] + im_tol)
    prec_mz_slice = slice(peptide_info['mz'] / (1 + mz_tol / 10**6), peptide_info['mz'] * (1 + mz_tol / 10**6))
    
    # create an elution profile for the precursor
    precursor_indices = timstof_data[
        rt_slice,
        im_slice,
        0,
        prec_mz_slice,
        'raw'
    ]
    fig = go.Figure()
    fig.add_trace(
        plot_line(timstof_data, precursor_indices, remove_zeros=True, label='precursor')
    )
    
    # create elution profiles for all fragments
    for frag, frag_mz in peptide_info['fragments'].items():
        fragment_data_indices = timstof_data[
            rt_slice,
            im_slice,
            prec_mz_slice,
            slice(frag_mz / (1 + mz_tol / 10**6), frag_mz * (1 + mz_tol / 10**6)),
            'raw'
        ]
        if len(fragment_data_indices) > 0:
            fig.add_trace(
                plot_line(timstof_data, fragment_data_indices, remove_zeros=True, label=frag)
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
            title=x_axis_label,
            titlefont_size=14,
            tickmode = 'auto',
            tickfont_size=14,
        ),
        yaxis=dict(
            title=y_axis_label
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.6,
            xanchor="right",
            x=0.95
        ),
        template = "plotly_white", 
        width=width,
        height=height,
        hovermode="x unified",
        showlegend=True
    )
    return fig

def plot_mass_spectra(
    data: pd.DataFrame,
    title: str,
    sequence: str,
    predicted: tuple = (),
    spectrum_color: str = 'grey',
    b_ion_color: str = 'red',
    y_ion_color: str = 'blue',
    neutral_losses_color: str = 'green',
    template: str = "plotly_white",
    spectrum_line_width: float = 2.0,
    font_size_seq: int = 14, 
    font_size_ion: int = 10,
    height: int = 520
):    
    """Plot the mass spectrum.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing information about the spectrum 'mz_values', 'intensity_values', 'ions'.
    title : str
        The title of the plot.
    sequence: str
        The peptide sequence.
    predicted : tuple
        The tuple containing values of the predicted FragmentMz, RelativeIntensity and ions in the form of:
        (predicted_df.FragmentMz, predicted_df.RelativeIntensity, predicted_df.ions). Default: empty tuple.
    spectrum_color : str 
        The color of the mass spectrum. Default: 'grey'.
    b_ion_color : str 
        The color of the b-ions. Default: 'red'.
    y_ion_color : str 
        The color of the y-ions. Default: 'blue'.
    neutral_losses_color : str 
        The color of the neutral losses. Default: 'green'.
    template: str 
        The template for the plot. Default: "plotly_white".
    spectrum_line_width: float 
        The width of the spectrum peaks. Default: 2.0.
    font_size_seq: int 
        The font size of the peptide sequence letters. Default: 14. 
    font_size_ion: int 
        The font size of the ion letters. Default: 10.
    height: int 
        The height of the plot. Default: 520.

    Returns
    -------
    a Plotly spectrum plot
        The ms2 spectum plot.
    """
    data.sort_values('ions', inplace=True)
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data[data.ions == ''].mz_values,
            y=data[data.ions == ''].intensity_values,
            mode='markers',
            marker=dict(color=spectrum_color, size=1),
            hovertemplate='<b>m/z:</b> %{x:.3f};<br><b>Intensity:</b> %{y:.3f}.',
            name='',
            showlegend=False
        )
    )
    # y-ions
    data_y_ions = data[data.ions.str.contains('y')]
    fig.add_trace(
        go.Scatter(
            x=data_y_ions.mz_values,
            y=data_y_ions.intensity_values,
            mode='markers',
            marker=dict(color=y_ion_color, size=1),
            hovertext=data_y_ions.ions,
            hovertemplate='<b>m/z:</b> %{x:.3f};<br><b>Intensity:</b> %{y:.3f};<br><b>Ion:</b> %{hovertext}.',
            name='',
            showlegend=False
        )
    )
    # b-ions
    data_b_ions = data[data.ions.str.contains('b')]
    fig.add_trace(
        go.Scatter(
            x=data_b_ions.mz_values,
            y=data_b_ions.intensity_values,
            mode='markers',
            marker=dict(color=b_ion_color, size=1),
            hovertext=data_b_ions.ions,
            hovertemplate='<b>m/z:</b> %{x:.3f};<br><b>Intensity:</b> %{y:.3f};<br><b>Ion:</b> %{hovertext}.',
            name='',
            showlegend=False
        )
    )
    # phospho neutral losses
    fig.add_trace(
        go.Scatter(
            x=data[data.ions.str.contains('P')].mz_values,
            y=data[data.ions.str.contains('P')].intensity_values,
            mode='markers',
            marker=dict(color=neutral_losses_color, size=1),
            hovertext=data[data.ions.str.contains('P')].ions,
            hovertemplate='<b>m/z:</b> %{x:.3f};<br><b>Intensity:</b> %{y:.3f};<br><b>Ion:</b> %{hovertext}.',
            name='',
            showlegend=False
        )
    )
   
    if predicted:
        fig.add_trace(
            go.Scatter(
                x=predicted[0],
                y=predicted[1],
                mode='markers',
                hovertext=predicted[2],
                marker=dict(color='lightblue', size=3),
                text = [f'{-i:.3f}' for i in predicted[1]],
                hovertemplate='<b>m/z:</b> %{x:.3f};<br><b>Intensity:</b> %{text};<br><b>Ion:</b> %{hovertext}.',
                name='',
                showlegend=False
            )
        )
    
    fig.update_layout(
        template=template,
        xaxis=dict(
            title='m/z, Th',
            mirror=True
        ),
        yaxis=dict(
            title='Intensity',
        ),
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            yanchor="bottom",
            y=1.01
        ),
        hovermode="x",
        height=height,
        title=dict(
            text=title,
            font=dict(
                size=16,
            ),
            x=0.5,
            xanchor='center',
            yanchor='top'
        )
    )
    
    # Use the 'shapes' attribute from the layout to draw the vertical lines
    if predicted:
        combined_mz = list(data.mz_values.values) + list(predicted[0])
        combined_int = list(data.intensity_values.values) + list(predicted[1])
        combined_ions = list(data.ions.values) + list(predicted[2])      
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='y',
                    x0=combined_mz[i],
                    y0=0,
                    x1=combined_mz[i],
                    y1=combined_int[i],
                    line=dict(
                        color = 'lightblue' if i > data.mz_values.shape[0] - 1 else \
                        (neutral_losses_color if ('HPO3' in combined_ions[i]) or ('H3PO4' in combined_ions[i]) else \
                        (b_ion_color if 'b' in combined_ions[i] else \
                        (y_ion_color if 'y' in combined_ions[i] else spectrum_color))),
                        width=spectrum_line_width
                    )
                ) for i in range(len(combined_mz))
            ],
            yaxis=dict(
                title='Relative intensity, %',
                ticktext=["100", "50", "0", "50", "100"],
                tickvals=[-100, -50, 0, 50, 100],
            ),
        )
    else:
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x',
                    yref='y',
                    x0=data.loc[i, 'mz_values'],
                    y0=0,
                    x1=data.loc[i, 'mz_values'],
                    y1=data.loc[i, 'intensity_values'],
                    line=dict(
                        color = neutral_losses_color if ('HPO3' in data.loc[i, 'ions']) or ('H3PO4' in data.loc[i, 'ions']) else \
                        (b_ion_color if 'b' in data.loc[i, 'ions'] else \
                        (y_ion_color if 'y' in data.loc[i, 'ions'] else spectrum_color)),
                        width=spectrum_line_width
                    )
                ) for i in data.index
            ],
        )
        
    fig_common = plotly.subplots.make_subplots(
        rows=4, cols=1, 
        figure=fig,
        specs=[
          [{"rowspan": 3}],  
          [{}], 
          [{}],
          [{}]
        ],
        vertical_spacing=0.15,
    )
    
    bions = convert_col_to_bool_list(data, 'ions', sequence, 'b')
    yions = convert_col_to_bool_list(data, 'ions', sequence, 'y')

    sl = len(sequence)
    value = 0.7
    for i,aa in enumerate(sequence):
        fig.add_annotation(
            dict(
                text=aa, 
                x=i*value, 
                y=0, 
                showarrow=False, 
                font_size=font_size_seq, 
                yshift=1, align='center'
            ),
            row=4,
            col=1
        )
    for i,b in enumerate(bions):
        if b:
            fig.add_trace(
                go.Scatter(
                    x=[i*value,i*value+0.35,i*value+0.35], 
                    y=[0.8,0.8,0], 
                    mode="lines",        
                    showlegend=False, 
                    marker_color=b_ion_color, 
                    line_width=spectrum_line_width,
                    hoverinfo='skip'
                ),
                row=4, 
                col=1
            )
            fig.add_annotation(
                dict(
                    text="b{}".format(str(i+1)), 
                    x=i*value+0.2, 
                    y=1.4,
                    showarrow=False, 
                    font_size=font_size_ion
                ), 
                row=4, 
                col=1
            )
    for i,y in enumerate(yions):
        if y:
            fig.add_trace(
                go.Scatter(
                    x=[i*value,i*value-0.35,i*value-0.35], 
                    y=[-0.8,-0.8,0], 
                    mode="lines",
                    showlegend=False, 
                    marker_color=y_ion_color, 
                    line_width=spectrum_line_width,
                    hoverinfo='skip'
                ),
                row=4, col=1
            )
            fig.add_annotation(
                dict(
                    text="y{}".format(str(sl-i)), 
                    x=i*value-0.2, 
                    y=-1.4, 
                    showarrow=False,
                    font_size=font_size_ion
                ), 
                row=4, 
                col=1
            )
    fig_common.update_yaxes(
        visible=False, 
        range=(-1.5,1.5),
        row=4, 
        col=1
    )
    fig_common.update_xaxes(
        visible=False, 
        range=(-1, sl),
        row=4, 
        col=1
    )
    
    return fig_common

def plot_heatmap(
    df: pd.DataFrame,
    title: str = "",
    width: int = 250,
    height: int = 250,
    background_color: str = "black",
    colormap: str = "fire",
):
    """Create a heatmap showing a correlation of retention time  and ion mobility with color coding for signal intensity.

    Parameters
    ----------
    df : pandas Dataframe
        A dataframe obtained by slicing an alphatims.bruker.TimsTOF object.
    title: str
        The title of the plot. Default: "".
    width : int
        The width of the plot. Default: 250.
    height : int
        The height of the plot. Default: 250.
    background_color : str
        The background color of the plot. Default: "black".
    colormap : str
        The name of the colormap in Plotly. Default: "fire".
        
    Returns
    -------
    a Plotly scatter plot
        The scatter plot showing the correlation of retention time  and ion mobility with color coding for signal intensity.
    """
    labels = {
        'RT, min': "rt_values",
        'Inversed IM, V·s·cm\u207B\u00B2': "mobility_values",
        'Intensity': "intensity_values",
    }
    x_axis_label = "RT, min"
    y_axis_label = "Inversed IM, V·s·cm\u207B\u00B2"
    z_axis_label = "Intensity"
    
    x_dimension = labels[x_axis_label]
    y_dimension = labels[y_axis_label]
    z_dimension = labels[z_axis_label]

    df["rt_values"] /= 60

    opts_ms1=dict(
        width=width,
        height=height,
        title=title,
        xlabel=x_axis_label,
        ylabel=y_axis_label,
        bgcolor=background_color,
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

def plot_elution_profile_heatmap(
    timstof_data,
    peptide_info: dict,
    mass_dict: dict,
    mz_tol: int = 50,
    rt_tol: int = 30,
    im_tol: int = 0.05,
    title: str = "",
    n_cols: int = 5,
    width: int = 180,
    height: int = 180,
):
    """Plot an elution profile for the specified precursor and all his identified fragments as heatmaps in the 
    retention time/ion mobility dimensions.

    Parameters
    ----------
    timstof_data : alphatims.bruker.TimsTOF
        An alphatims.bruker.TimsTOF data object.
    peptide_info : dict
        Peptide information including sequence, fragments' patterns, rt, mz and im values.
    mass_dict : dict
        The basic mass dictionaty with the masses of all amino acids and modifications.
    mz_tol: float 
        The mz tolerance value. Default: 50 ppm.
    rt_tol: float 
        The rt tolerance value. Default: 30 ppm.
    im_tol: float 
        The im tolerance value. Default: 0.05 ppm.
    title : str
        The title of the plot. Default: "".
    n_cols: int
        The number of the heatmaps plotted per row. Default: 5.
    width : int
        The width of the plot. Default: 180.
    height : int
        The height of the plot. Default: 180.

    Returns
    -------
    a Bokeh heatmap plots
        The elution profile heatmap plots in retention time and ion mobility dimensions 
        for the specified peptide and all his fragments.
    """
    # predict the theoretical fragments using the Alphapept get_fragmass() function.
    frag_masses, frag_type = utils.get_fragmass(
        parsed_pep=list(peptide_info['sequence']), 
        mass_dict=mass_dict
    )
    peptide_info['fragments'] = {
        (f"b{key}" if key>0 else f"y{-key}"):value for key,value in zip(frag_type, frag_masses)
    }
    
    # slice the data using the rt_tol, im_tol and mz_tol values
    rt_slice = slice(peptide_info['rt'] - rt_tol, peptide_info['rt'] + rt_tol)
    im_slice = slice(peptide_info['im'] - im_tol, peptide_info['im'] + im_tol)
    prec_mz_slice = slice(peptide_info['mz'] / (1 + mz_tol / 10**6), peptide_info['mz'] * (1 + mz_tol / 10**6))
    
    # create an elution profile for the precursor
    precursor_indices = timstof_data[
        rt_slice,
        im_slice,
        0,
        prec_mz_slice,
        'raw'
    ]
    
    common_plot = plot_heatmap(
        timstof_data.as_dataframe(precursor_indices), title='precursor', width=width, height=height
    )
    
    # create elution profiles for all fragments
    for frag, frag_mz in peptide_info['fragments'].items():
        fragment_data_indices = timstof_data[
            rt_slice,
            im_slice,
            prec_mz_slice,
            slice(frag_mz / (1 + mz_tol / 10**6), frag_mz * (1 + mz_tol / 10**6)),
            'raw'
        ]
        if len(fragment_data_indices) > 0:
            common_plot += plot_heatmap(
                timstof_data.as_dataframe(fragment_data_indices), title=frag, width=width, height=height
            )
    
    return common_plot.cols(n_cols)