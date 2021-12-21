[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MannLabs/ProteomicsVisualization/HEAD?urlpath=tree)

# Bottom-up proteomics data visualisation

In this repository we provide thoroughly documented Python code for generating all figures from the review _**"Schessner, Voytik, Bludau. A practical guide to interpreting and generating bottom-up proteomics data visualizations"**_.

* [**About**](#about)
* [**License**](#license)
* [**Structure**](#structure)
* [**Installation & Usage**](#installation-&-usage)
  * [**Binder**](#binder)
  * [**Python and Jupyter notebooks**](#python-and-jupyter-notebooks)
* [**Citations**](#citations)
* [**How to contribute**](#how-to-contribute)

---
## About

The idea behind creating this repository came while writing the review. Our goal here is to give new developers an easy entry point and an example of what different tools can do in the field of bottom-up proteomics data visualization.

The code is written entirely in Python and presented in Jupyter notebooks. To implement the functionality we used well documented and maintained Python libraries for scientific computing, e.g. NumPy, Pandas, Scipy, etc. For plotting, we mainly used the Plotly library.

---
## License

The code for this repository was developed by the [Mann Labs at the Max Planck Institute of Biochemistry](https://www.biochem.mpg.de/mann) and is freely available with an [Apache License](LICENSE.txt). The external Python packages (installed for each notebook separately) have their own licenses, which can be consulted on their respective websites.

---
## Structure

The Python code of the entire repository is divided into five parts according to the number of figures in the reviews, and is presented as Jupyter notebooks. It includes:

##### 1) Figure 1 - visualizations at the precursor level.

This notebook incorporates functions for pre-processing and visualisation of raw MS data at the precursor level. The following graphs can be plotted using this notebook:
- total ion chromatogram (TIC) and base peak intensity (BPI);
- extracted ion chromatogram (XIC);
- 2D MS1 map showing the intensity of the observed precursor masses across the whole retention time;
- 2D MS1 ion mobility heatmap of precursor intensities demonstrating a correlation of m/z and ion mobility.

##### 2) Figure 2 - visualizations at the fragment level.  

This notebook provides functions for pre-processing and visualisation of the raw MS data at the fragment level. The following graphs are presented in this notebook:
- peptide MS2 spectrum annotated with the identified b- and y-ions;
- mirrored MS2 spectrum showing the experimental (top) and predicted (bottom) spectra for the same peptide;
- overlaid extracted ion chromatograms in the elution time window of precursor and fragments;
- heatmaps of ion intensities in ion-mobility and retention time dimensions.

##### 3) Figure 3 - peptide visualization.

Here we demonstrate plots that may be useful for visualisation of peptides and PTMs. The notebook contains the code for plotting:
- peptide coverage plot (PeptideAtlas style);
- a complex plot displaying differential peptide coverage between samples with overlapping peptides collapsed into a single trace, PTMs and external features (generated using [AlphaMap](https://github.com/MannLabs/alphamap));
- lollipop plot displaying the intensity and localization probability of phosphosites (PhosphoSitePlus style).

##### 4) Figure 4 - simple protein quantification.

This notebook includes the following graphs to illustrate dataset properties and binary comparisons:

- intensity histograms showing the distribution and number of protein groups;
- protein rank plot;
- pairwise correlation plots;
- sample correlation matrix, which is suitable to a larger number of samples than the pairwise correlation plot;
- volcano plots with different cutoff options;
- visualization of enrichment analysis.

##### 5) Figure 5 - multidimensional protein quantification.

This notebook covers the visualization of multidimensional experimental designs and includes the following plots:

- various principal component analysis plots;
- tSNE plot;
- UMAP plot;
- heatmap with marginal dendrograms;
- profile plot;
- parallel coordinates plot;
- radar plot.

In addition to the above-mentioned Jupyter notebooks, the GitHub repository includes **(1)** data and annotations folders (Data & annotations) with the data files required for visualization and described in each notebook where they have been used, **(2)** a folder containing all supplementary files (ext) to read or visualize the data.

---
## Installation & Usage

There are two different ways to use this repository depending on programming experience:

* [**Binder**](#binder): This way is suitable for users not experienced in programming and does not require any installation or data download. It allows you to run code online and build all the interactive plots just in the browser.
* [**Python and Jupyter notebooks**](#python-and-jupyter-notebooks): Choose this option if you are familiar with Python, conda and Jupyter notebooks. It provides the ability to reproduce all the graphs from the review and apply the written code to explore your own data.

### Binder

We suggest using Binder for users with no programming experience. Binder allows you to execute code online without installing any software or downloading data locally. It also gives the possibility to interact with the published code, modify it, access pre-loaded data and build interactive plots online.

To run Binder use the `launch binder` logo at the top of the README.md file or click [here](https://mybinder.org/v2/gh/MannLabs/ProteomicsVisualization/HEAD?urlpath=tree).

**NOTE:** The first time you run Binder on your machine it often takes a while for the GitHub repository to be built on the Binder server. During all subsequent times of use, Binder simply runs an already created repository, so it should be much faster.

Despite the ease of use, there are **some limitations** when using Binder:
- you can't visualise your own data in Binder, because all data files mentioned in Binder are pre-loaded into the GitHub repository for possible use in Binder.
- some large files, e.g. for the visualisation of the raw Bruker data for Figure 2C/2D, cannot be pre-loaded in GitHub. Therefore please use the "Python and Jupyter notebook" method to reproduce these plots.


### Python and Jupyter notebooks

This option is more flexible compared to the Binder approach, but it is intended for users with some programming experience. This way allows you to use written code to rebuild all the graphs from the overview, visualize your own data using the code provided or modified to suit your needs.

- (Optional) Navigate to the directory where you want to install the project.
- Download the 'ProteomicsVisualization' repository from GitHub directly or with the `git` command. This creates a new 'ProteomicsVisualization' subfolder in your current directory.

```bash
git clone https://github.com/MannLabs/ProteomicsVisualization.git
```

For any Python package, it is highly recommended to use a separate [conda virtual environment](https://docs.conda.io/en/latest/) as otherwise * there may be dependancy conflicts with existing packages*.

```bash
conda create --name prot_vis python=3.8 jupyter -y
conda activate prot_vis
jupyter notebook
```

To simplify the usage of a package, we don't provide a combined requirements.txt file listing packages for the whole repository. This is designed so that you can work separately with the different notebooks that you are only interested in. Therefore, to start working with notebooks, run the following command in the terminal

```bash
jupyter notebook
```
and run the Jupyter notebook. For the first use, comment out the first code cell in the notebook which looks like this

```
## Install all packages directly in the notebook  
#!pip install {some_packages}
```

and run it in the notebook. This will install the libraries used in the current notebook. The installation should only be done in the conda environment once you are using notepad the first time. After this you can comment out this cell again.

---
## Citations

Reference:
- will be updated soon.

---
## How to contribute

If you like this software, you can give us a [star](https://github.com/MannLabs/ProteomicsVisualization/stargazers) to boost our visibility! All direct contributions are also welcome. Feel free to post a new [issue](https://github.com/MannLabs/ProteomicsVisualization/issues) or clone the repository and create a [pull request](https://github.com/MannLabs/ProteomicsVisualization/pulls) with a new branch.
