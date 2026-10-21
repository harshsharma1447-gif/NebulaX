
# Nebula X Communications Simulator

### An Integrated Communication Systems Simulation and Analysis Platform Using Python and NI Multisim

Nebula X is a communication systems simulation and analysis project developed to bring multiple communication engineering concepts into a unified, interactive environment. It combines Python-based simulation and visualization with NI Multisim circuit-level simulation to support experimentation, waveform analysis, and the study of communication system behaviour.

The project was independently developed by **Harsh Sharma**, Electronics and Communication Engineering, Dayananda Sagar College of Engineering (DSCE), Bengaluru, India.

---

## Overview

Communication engineering involves several interconnected concepts, including signal generation, analog and digital modulation, synchronization, multiplexing, channel effects, signal processing, and performance analysis. Studying these concepts across disconnected tools can make it difficult to compare their behaviour within a single workflow.

Nebula X was developed to provide an integrated educational and research-oriented simulation environment for exploring these concepts through configurable signals, graphical visualization, communication analysis tools, and circuit-level validation using NI Multisim.

## Core Modules

### 1. Channeling

The Channeling module focuses on communication signal processing and transmission concepts.

Topics covered include:

- Analog modulation: Amplitude Modulation (AM), Frequency Modulation (FM), and Phase Modulation (PM).
- Pulse-based modulation, including Pulse Width Modulation (PWM).
- Digital modulation: ASK, FSK, PSK, QPSK, and GMSK.
- Synchronization using sinusoidal, triangular, and rectangular reference waveforms.
- Multiplexing concepts, including TDM, FDM, WDM, CDM, and OFDM.
- Configurable communication channel effects, depending on the implemented simulation functionality.

### 2. Communication Analysis

This module focuses on interpreting communication signals and evaluating their characteristics.

The research platform covers analysis techniques including:

- Oscilloscope and time-domain waveform visualization.
- Fast Fourier Transform (FFT).
- Frequency spectrum analysis.
- Waterfall plots.
- Constellation diagrams.
- Eye diagrams.
- Bit Error Rate (BER) and Symbol Error Rate (SER).
- Filter and channel analysis.

Availability and behaviour of individual tools depend on the corresponding implementation in the source code.

### 3. Generation Communications

This module provides an educational comparison of communication technologies spanning 0G through 6G.

It is intended to help users explore the evolution of communication systems and compare selected characteristics, concepts, and technologies across generations. These comparisons are educational simulations, not measurements of live commercial networks.

---

## Technology Stack

- **Python:** Software implementation, simulation, signal processing, and visualization.
- **NI Multisim:** Circuit-level communication simulation and validation.
- **Signal Processing:** Analysis of time-domain and frequency-domain signal behaviour.
- **Communication Engineering:** Analog and digital modulation, synchronization, multiplexing, and channel analysis.

Additional Python libraries are listed in `requirements.txt` once verified against the actual source code.

## Research Publication

**Paper title:** Nebula X Communications Simulator: Development of an Integrated Communication Systems Simulation and Analysis Platform using Python and NI Multisim

- **Author:** Harsh Sharma
- **Publication:** International Journal of Engineering Research & Technology (IJERT)
- **Volume:** 15
- **Issue:** 08
- **Publication date:** August 2026
- **Paper ID:** IJERTV15IS080117
- **DOI:** [10.5281/zenodo.21847689](https://doi.org/10.5281/zenodo.21847689)

The publication describes the project's motivation, architecture, methodology, simulation results, limitations, and future scope.

## Getting Started

### Prerequisites

1. Python 3.10 or a compatible version supported by the project's dependencies.
2. The Python libraries listed in `requirements.txt`.
3. NI Multisim for opening or reproducing the associated circuit simulations, where applicable.
4. The repository's source files and any required simulation assets.

### Installation

Clone the repository:

```bash
git clone https://github.com/harshsharma1447/nebula-x-communications-simulator.git
cd nebula-x-communications-simulator
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application using the actual Python entry-point file included in this repository. The exact launch command depends on the source files provided.

**Note:** The repository must contain the corresponding source files and simulation assets for the steps above to work. Python dependencies alone do not provide NI Multisim or its project files.

## Project Scope

Nebula X is intended for educational experimentation, engineering research, simulation, and demonstration.

The project combines software-based communication simulation with circuit-level modelling. It is not presented as a deployed telecommunications system, a commercial network emulator, or a substitute for real-world hardware testing.

Simulation results depend on the mathematical models, parameter choices, assumptions, and circuit configurations used. They should be interpreted within those limitations.

## Limitations

- Results depend on the accuracy and scope of the implemented models.
- Software simulations do not automatically establish real-world transmission performance.
- Circuit-level validation is limited to the simulations and configurations actually evaluated.
- Comparisons between communication generations are educational and should not be interpreted as comprehensive standards compliance testing.
- The proposed Time-Frequency Division Multiplexing (TFDM) concept discussed in the research requires further mathematical development and experimental validation. It should not be treated as an established, standardized multiplexing technique.

## Future Scope

Potential extensions include:

- Software-defined radio (SDR) integration.
- FPGA-based communication experiments.
- Additional modulation and multiplexing techniques.
- More extensive channel models and noise analysis.
- Automated modulation recognition using machine learning.
- Expanded performance analysis and reproducible experiment reporting.
- Further investigation of hybrid multiplexing concepts.

These are potential directions for future development, not claims that every feature is already implemented.

## License and Usage Restrictions

The original source code and original project assets in this repository are subject to the terms specified in the accompanying `LICENSE` file, except where other terms apply.

Unless expressly authorized in writing, permission is not granted to redistribute, commercially exploit, modify, or create derivative works from the protected source code.

The associated research publication has its own licensing terms. Third-party libraries, tools, and assets remain subject to their respective licenses.

Public availability of this repository does not imply that all of its contents are open-source or freely reusable.

## Author

**Harsh Sharma**  
Electronics and Communication Engineering  
Dayananda Sagar College of Engineering (DSCE), Bengaluru, India

Research interests: Communication Systems, Signal Processing, Embedded Systems, VLSI, and AI-assisted Engineering.

## Acknowledgement

Nebula X was developed as an independent engineering research project to explore the integration of communication systems simulation, signal analysis, and circuit-level modelling in an educational environment.
