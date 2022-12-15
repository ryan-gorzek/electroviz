# electroviz

electroviz is a Python package for processing, analyzing, and visualizing extracellular electrophysiology (and its associated sensory and behavioral) data.

*Note: Parts of this description are theoretical and under active development.

## Installation

First, clone this repository:
    git clone https://github.com/ryan-gorzek/electroviz.git
Then, create and activate an anaconda environment in which to install it:
    conda create -n electroviz python=3.9
    conda activate electroviz
Finally, install electroviz with pip:
    pip install -e electroviz

## Concept

electroviz uses an object-oriented approach to build datasets from the disparate types of data found in a typical experiment. The attributes, methods, and properties of these objects enable fast, consistent, and scalable analysis. Importantly, raw data is always accessible for custom analyses.

First, "low-level" data streams from acquisition software (e.g., SpikeGLX, bTsS) are loaded and parsed for relevant information by class constructors.

For example, digital signals recorded from a National Instruments DAQ by SpikeGLX are read and parsed by:
    nidaq = NIDAQ(path_to_binary_and_metadata_files)
