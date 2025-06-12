# electroviz

**electroviz** is a Python package for processing, analyzing, and visualizing extracellular electrophysiology experiments — currently those involving **Neuropixels recordings paired with visual stimuli**. It offers an object-oriented framework that makes it easy to load, organize, and analyze large-scale neural data alongside sensory or behavioral context.

> **Why use electroviz?**  
> If you're looking to manage multimodal data clearly and produce fast and effective visualizations alongside higher-level analyses, `electroviz` serves both as a **toolkit** and a **conceptual framework** for organizing neurophysiology experiments in Python.

---

## 📦 Installation

```bash
git clone https://github.com/ryan-gorzek/electroviz.git
cd electroviz
conda create -n electroviz python=3.9
conda activate electroviz
pip install -e .
```
🧠 Core Concept

electroviz abstracts complex experiments into interoperable Python classes. At the lowest level, data streams from SpikeGLX, bTsS, and NI DAQs are parsed by dedicated loaders classes (Imec, bTsS, NIDAQ, etc.). These are then assembled into higher-level objects such as Unit, Stimulus, Spikes, and Experiment.

🧩 Modular Data Ingestion

```python
from electroviz import NIDAQ, IMEC, Kilosort, VStim

# Load binary and metadata files from SpikeGLX
nidaq = NIDAQ(path_to_binary_files)
imec = IMEC(path_to_probe_binary)

# Parse spike sorting results
spikes = Kilosort(path_to_kilosort_output)

# Load visual stimulus data
vstim = VStim(path_to_stimulus_log)

# Optionally link these into an Experiment
from electroviz import Experiment
exp = Experiment(imec=imec, nidaq=nidaq, spikes=spikes, stim=vstim)
```

📂 Key Objects
Class	Description
Experiment	High-level organizer tying together probe data, spike sorting, stimulus timing, and more
Unit	Single neuron object with spike times, tuning curves, and more
Population	Collection of Unit objects for aggregate analyses
Stimulus / VStim	Represent visual/sensory stimulus conditions and timing
Spikes	Handles spike times, binning, and alignment
Kernel	Encodes stimulus-response models (e.g., receptive field fitting)
Probe	Stores physical geometry and mapping of recording sites
Event	General-purpose container for binary or timestamped experimental events
🧪 Example Workflow

```python
# Load experiment
exp = Experiment('/path/to/recording/')

# Create a population from several units (neurons)
pop = exp.populations[0]
pop_sub = pop.remove(pop.units["total_spikes"] < 100)

# Access stimuli
stim = exp.stimuli['drifting_gratings']

# Plot raster and stimulus kernel
for unit in pop_sub:

    # Plot spike raster aligned to stimulus
    unit.plot_raster(event_times=stim.onset_times)

    # Compute and plot tuning curve
    unit.plot_tuning_curve()

    # Fit a stimulus kernel
    kernel = unit.fit_kernel(stimulus='drifting_gratings')
    unit.plot_prediction(kernel.predict(stim))
```

➡️ Visualization Spot 1: Raster plot showing stimulus-aligned spikes
➡️ Visualization Spot 2: Tuning curve for one unit
➡️ Visualization Spot 3: Kernel fit vs. observed firing rate
🧰 Supported Data Types and Sources

    SpikeGLX: IMEC, NIDAQ readers for binary data and metadata

    bTsS: BTSS for behavioral task logs

    Kilosort: Spike sorting outputs via Kilosort

    Visual stimuli: Frame/condition logs via VStim

    Local field potential (LFP): Accessed via lfp.py

    Digital and analog channels: Via digitalchannel.py

🧭 Why Use This Framework?

electroviz is particularly useful when:

    You're recording with Neuropixels and delivering visual or sensory stimuli

    You want a clear, object-oriented approach to neural data

    You’re building reproducible pipelines for analysis across experiments

    You need to integrate raw data access with high-level analysis tools

This framework was originally designed for multi-area visual coding experiments in rodents, but its architecture generalizes across tasks and sensory modalities.
🧑‍🔬 Extending the Framework

Each class in electroviz is designed for inheritance and modular use. You can:

    Subclass core objects like Unit, Stimulus, Experiment

    Swap in custom sorting or behavioral formats

    Use electroviz with external tools like PyTorch, scikit-learn, or napari

📌 Potential Roadmap

    🧹 Signal preprocessing tools (e.g. spike artifact removal, filtering)

    🧠 Trial-level modeling and GLMs

    📂 NWB and HDF5 export

    📉 GUI browser for spike trains and LFPs
