# electroviz

electroviz is a Python package for processing, analyzing, and visualizing extracellular electrophysiology (and its associated sensory and behavioral) data.

*Note: Much of this description is currently theoretical and under active development.

electroviz takes an object-oriented approach to handling the disparate types of data found in a typical experiment. The attributes, methods, and properties of
these objects enables fast, consistent, and scalable analysis.

First, "low-level" data streams from acquisition software (e.g., SpikeGLX, bTsS) are parsed for relevant information by specific class constructors.

    For example, digital signals recorded from a National Instruments DAQ by SpikeGLX are read and parsed by:
    ```
    nidq = NIDAQ(path_to_binary_and_metadata_files)
    ```

analysis-ready domains 
