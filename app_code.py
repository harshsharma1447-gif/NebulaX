from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy import signal
from scipy.special import erfc

from PyQt6.QtCore import QObject, QPointF, QThread, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPdfWriter
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QScrollArea,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# -----------------------------------------------------------------------------
# Application constants and theme
# -----------------------------------------------------------------------------

APPLICATION_NAME = "Communication System Simulator"
MAX_SAMPLES = 250_000
MIN_SAMPLES = 2_000

# The labels here follow the naming requested in the simulator interface.
# QM(QAM) remains digitally analysed (constellation/BER) even though the
# selection screen groups it under the requested analogue-modulation heading.
ANALOG_MODULATIONS = ("AM", "FM", "PM", "GM(PWM)", "QM(QAM)")
DIGITAL_MODULATIONS = ("ASK", "FSK", "PSK", "QPSK", "GMSK")
ANALYSIS_GROUPS = (
    "Signal Analysis",
    "Digital Analysis",
    "Channel Analysis",
    "Filter Analysis",
)

# Every primary category is split into explicitly selectable sub-analysis tools.
SUB_ANALYSIS_OPTIONS: dict[str, tuple[str, ...]] = {
    "Signal Analysis": ("Oscilloscope", "FFT", "Spectrum Analyzer", "Waterfall Plot"),
    "Digital Analysis": ("Constellation Diagram", "Eye Diagram", "BER", "SER"),
    "Channel Analysis": ("SNR", "Noise Power", "Signal Power", "Attenuation"),
    "Filter Analysis": ("LPF", "HPF", "BPF", "BSF"),
}

# Human-readable theory kept next to the executable models.  This turns a run
# into a lab exercise: students can inspect the signal and the method at once.
MODULATION_REFERENCE: dict[str, dict[str, str]] = {
    "AM": {
        "family": "Analogue amplitude modulation",
        "expression": "s(t) = A_c [1 + μ m(t)] cos(2πf_c t)",
        "principle": "The carrier envelope follows the message amplitude.",
        "receiver": "Envelope detector or coherent detector.",
        "bandwidth": "Approximately twice the highest message frequency.",
        "advantages": "Simple generation and inexpensive receivers.",
        "limitations": "Noise directly changes the information-bearing envelope.",
        "examples": "Broadcast AM radio and aviation voice links.",
    },
    "FM": {
        "family": "Analogue frequency modulation",
        "expression": "s(t) = A_c cos[2πf_c t + 2πk_f ∫m(t)dt]",
        "principle": "Instantaneous carrier frequency varies with the message.",
        "receiver": "Discriminator, PLL, or quadrature detector.",
        "bandwidth": "Estimated by Carson's rule: 2(Δf + f_m).",
        "advantages": "Good amplitude-noise immunity and constant envelope.",
        "limitations": "Usually consumes more spectrum than AM.",
        "examples": "FM broadcast radio and legacy mobile voice.",
    },
    "PM": {
        "family": "Analogue phase modulation",
        "expression": "s(t) = A_c cos[2πf_c t + k_p m(t)]",
        "principle": "The instantaneous carrier phase follows the message.",
        "receiver": "Coherent phase detector or phase-locked loop.",
        "bandwidth": "Depends on peak phase deviation and message bandwidth.",
        "advantages": "Constant-envelope operation can suit efficient RF amplifiers.",
        "limitations": "Requires phase-aware reception and careful synchronization.",
        "examples": "Telemetry and precursor techniques to digital PSK.",
    },
    "PWM": {
        "family": "Analogue pulse-width modulation",
        "expression": "Pulse duty cycle D(t) = 0.5 + k m(t)",
        "principle": "Message amplitude alters the duration of each pulse.",
        "receiver": "Pulse-width measurement followed by low-pass reconstruction.",
        "bandwidth": "Set by pulse repetition rate and sideband content.",
        "advantages": "Resists amplitude variation when timing is recovered.",
        "limitations": "Fast pulse edges require substantial bandwidth.",
        "examples": "Control systems, power electronics, servo signalling.",
    },
    "ASK": {
        "family": "Digital amplitude-shift keying",
        "expression": "s(t) = A_b cos(2πf_c t), where A_b represents a bit.",
        "principle": "Binary symbols select one of two carrier amplitudes.",
        "receiver": "Envelope or coherent amplitude detector with threshold.",
        "bandwidth": "Depends on symbol rate and pulse shaping.",
        "advantages": "Easy to understand and simple to implement.",
        "limitations": "Sensitive to fading and amplitude noise.",
        "examples": "Optical on-off keying and simple RF telemetry.",
    },
    "FSK": {
        "family": "Digital frequency-shift keying",
        "expression": "s(t) = A_c cos[2π(f_c ± Δf)t]",
        "principle": "Binary symbols select one of two instantaneous frequencies.",
        "receiver": "Frequency discriminator, correlator, or matched filters.",
        "bandwidth": "Grows with frequency separation and symbol rate.",
        "advantages": "Tolerant of amplitude fading compared with ASK.",
        "limitations": "Can be spectrally wider than phase-keyed alternatives.",
        "examples": "Low-rate modems, paging, and telemetry.",
    },
    "PSK": {
        "family": "Digital binary phase-shift keying",
        "expression": "s(t) = ±A_c cos(2πf_c t)",
        "principle": "A binary symbol changes carrier phase by 180 degrees.",
        "receiver": "Coherent detector with carrier phase recovery.",
        "bandwidth": "Determined by symbol rate and filtering roll-off.",
        "advantages": "Strong AWGN performance for a given energy per bit.",
        "limitations": "Carrier phase ambiguity must be managed.",
        "examples": "Satellite telemetry and spread-spectrum systems.",
    },
    "QPSK": {
        "family": "Digital quadrature phase-shift keying",
        "expression": "s(t) = I cos(2πf_c t) - Q sin(2πf_c t)",
        "principle": "Four phase states convey two bits per symbol.",
        "receiver": "I/Q coherent receiver with timing and carrier recovery.",
        "bandwidth": "About half BPSK's symbol rate for the same bit rate.",
        "advantages": "Improved spectral efficiency with robust phase states.",
        "limitations": "I/Q imbalance and synchronization affect constellation quality.",
        "examples": "Satellite, Wi-Fi, and cellular control channels.",
    },
    "GMSK": {
        "family": "Gaussian minimum-shift keying",
        "expression": "Gaussian-filtered continuous-phase FSK with h = 0.5.",
        "principle": "A Gaussian filter smooths symbol transitions before MSK frequency modulation.",
        "receiver": "Coherent or differential detector with timing recovery.",
        "bandwidth": "Narrower sidelobes than unfiltered FSK at a chosen BT product.",
        "advantages": "Constant envelope and efficient spectrum containment.",
        "limitations": "Filtering introduces controlled intersymbol interference.",
        "examples": "GSM 2G air interface.",
    },
    "QAM": {
        "family": "Digital quadrature amplitude modulation",
        "expression": "s(t) = I cos(2πf_c t) - Q sin(2πf_c t)",
        "principle": "Information is carried by both I/Q amplitude and phase states.",
        "receiver": "Coherent I/Q receiver, equalizer, and decision regions.",
        "bandwidth": "High bits-per-symbol efficiency with suitable pulse shaping.",
        "advantages": "High data rate within limited spectrum.",
        "limitations": "Higher orders require high SNR and linear amplifiers.",
        "examples": "4G LTE, 5G NR, cable, Wi-Fi, and DSL.",
    },
}

TRACE_COLOURS = {
    "Cyan": "#38bdf8",
    "Green": "#4ade80",
    "Amber": "#f59e0b",
    "Magenta": "#e879f9",
    "White": "#f8fafc",
}

BACKGROUND_COLOURS = {
    "Black": "#090d13",
    "Blue black": "#071423",
    "Graphite": "#1a1d23",
}

DARK_THEME = """
QMainWindow, QWidget {
    background: #10151e;
    color: #dce6f2;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2d3b50;
    border-radius: 10px;
    margin-top: 13px;
    padding: 12px;
    font-weight: 600;
    color: #7dd3fc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTextBrowser {
    background: #182230;
    border: 1px solid #334155;
    border-radius: 7px;
    padding: 7px;
    selection-background-color: #0284c7;
}
QPushButton {
    background: #0369a1;
    border: none;
    border-radius: 8px;
    min-height: 20px;
    padding: 9px 16px;
    font-weight: 600;
}
QPushButton:hover { background: #0284c7; }
QPushButton:pressed { background: #075985; }
QPushButton[secondary="true"] { background: #253247; }
QPushButton[secondary="true"]:hover { background: #334155; }
QTabWidget::pane { border: 1px solid #2d3b50; border-radius: 8px; }
QTabBar::tab { background: #182230; padding: 9px 16px; margin-right: 2px; }
QTabBar::tab:selected { background: #0369a1; }
QDockWidget::title { background: #182230; padding: 8px; }
QStatusBar { background: #182230; }
QLabel#pageTitle { font-size: 27px; font-weight: 700; color: #f1f5f9; }
QLabel#subtitle { color: #94a3b8; }
QLabel#metric { color: #bae6fd; font-weight: 600; }
"""


# -----------------------------------------------------------------------------
# Data models and validation
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class WaveParameters:
    """Three user-configurable parameters of one waveform source."""

    frequency: float = 1_000.0
    amplitude: float = 1.0
    rate: float = 100_000.0
    waveform: str = "Sine"

    def validate(self, label: str) -> None:
        """Raise a friendly error if this waveform would not be valid."""
        if self.frequency <= 0:
            raise ValueError(f"{label} frequency must be positive.")
        if self.amplitude <= 0:
            raise ValueError(f"{label} voltage (amplitude) must be positive.")
        if self.rate <= 0:
            raise ValueError(f"{label} rate must be positive.")
        if self.waveform != "Straight Line" and self.frequency >= self.rate / 2:
            raise ValueError(
                f"{label} frequency must be below half of its rate "
                f"(Nyquist limit: {self.rate / 2:g} Hz)."
            )


@dataclass(slots=True)
class SimulationConfig:
    """All inputs that define a simulation session."""

    carrier: WaveParameters
    message: WaveParameters
    noise: WaveParameters | None = None
    duration: float = 0.02

    @property
    def nominal_sample_rate(self) -> float:
        """The fastest rate supplied by the user controls the timebase."""
        noise_rate = self.noise.rate if self.noise else 0.0
        return max(self.carrier.rate, self.message.rate, noise_rate)

    def validate(self) -> None:
        self.carrier.validate("Carrier")
        self.message.validate("Message")
        if self.noise is not None:
            self.noise.validate("Noise")
        if not 0.0001 <= self.duration <= 10:
            raise ValueError("Simulation duration must be between 0.1 ms and 10 seconds.")


@dataclass(slots=True)
class AnalogModulationSettings:
    """Method-specific controls used only by AM/FM/PM/GM(PWM)/QM(QAM)."""

    am_index: float = 0.70
    fm_deviation_hz: float = 2_000.0
    pm_deviation_rad: float = 1.50
    pwm_depth: float = 0.35
    qam_order: int = 16


@dataclass(slots=True)
class ChannelModulationSettings:
    """Optional channel impairments applied after modulation and noise generation."""

    enabled: set[str] = field(default_factory=set)
    attenuation_db: float = 6.0
    delay_ms: float = 0.50
    interference_amplitude: float = 0.15
    interference_frequency_hz: float = 3_000.0
    doppler_shift_hz: float = 80.0
    multipath_delay_ms: float = 0.25
    multipath_gain: float = 0.35


@dataclass(slots=True)
class SimulationRunRequest:
    """Selected methods plus analogue and channel-modulation configuration."""

    methods: list[str]
    analog_settings: AnalogModulationSettings = field(default_factory=AnalogModulationSettings)
    channel_settings: ChannelModulationSettings = field(default_factory=ChannelModulationSettings)


@dataclass(slots=True)
class SignalResult:
    """Complete output of one modulation experiment."""

    name: str
    time: np.ndarray
    samples: np.ndarray
    clean_signal: np.ndarray
    message_signal: np.ndarray
    noise_signal: np.ndarray | None
    sample_rate: float
    digital: bool
    channel_effects: list[str] = field(default_factory=list)

    @property
    def snr_db(self) -> float | None:
        """Compute SNR only when an explicit noise source was enabled."""
        if self.noise_signal is None:
            return None
        signal_power = float(np.mean(self.clean_signal**2))
        noise_power = float(np.mean(self.noise_signal**2))
        return 10 * np.log10(signal_power / max(noise_power, 1e-15))

    @property
    def peak_to_peak(self) -> float:
        return float(np.ptp(self.samples))

    @property
    def rms_voltage(self) -> float:
        return float(np.sqrt(np.mean(self.samples**2)))


@dataclass(slots=True)
class AnalysisRequest:
    """The user-selected work package for Communication Analysis mode."""

    modulation_methods: list[str]
    selected_subanalyses: dict[str, list[str]]
    multiplex_mode: str
    trigger_shape: str
    generations: list[str]
    analog_settings: AnalogModulationSettings = field(default_factory=AnalogModulationSettings)
    channel_settings: ChannelModulationSettings = field(default_factory=ChannelModulationSettings)

    @property
    def selected_analyses(self) -> list[str]:
        """Primary groups having one or more selected sub-analysis tools."""
        return [group for group, tools in self.selected_subanalyses.items() if tools]


# -----------------------------------------------------------------------------
# Signal generation, modulation, and analysis functions
# -----------------------------------------------------------------------------

class SignalEngine:
    """Signal-processing functions kept separate from all Qt user-interface code."""

    @staticmethod
    def make_timebase(config: SimulationConfig) -> tuple[np.ndarray, float]:
        """Create a time vector and cap point count to keep plots responsive."""
        ideal_count = int(config.nominal_sample_rate * config.duration)
        point_count = min(max(ideal_count, MIN_SAMPLES), MAX_SAMPLES)
        actual_rate = point_count / config.duration
        return np.arange(point_count, dtype=float) / actual_rate, actual_rate

    @staticmethod
    def waveform_from_phase(phase: np.ndarray, waveform: str) -> np.ndarray:
        """Return a normalized periodic shape for the requested graph style."""
        shapes: dict[str, Callable[[], np.ndarray]] = {
            "Sine": lambda: np.sin(phase),
            "Cosine": lambda: np.cos(phase),
            "Triangle": lambda: signal.sawtooth(phase, width=0.5),
            "Rectangular": lambda: signal.square(phase),
            "Sawtooth": lambda: signal.sawtooth(phase),
            "Straight Line": lambda: np.ones_like(phase),
        }
        if waveform not in shapes:
            raise ValueError(f"{waveform} is not a periodic waveform.")
        return shapes[waveform]()

    @classmethod
    def source_waveform(cls, time: np.ndarray, wave: WaveParameters) -> np.ndarray:
        """Generate the graph type chosen for a carrier, message, or noise input."""
        if wave.waveform == "Digital Data":
            samples_per_symbol = max(8, int(wave.rate / max(wave.frequency * 12, 1)))
            return wave.amplitude * cls.random_symbols(len(time), samples_per_symbol)
        phase = 2 * np.pi * wave.frequency * time
        return wave.amplitude * cls.waveform_from_phase(phase, wave.waveform)

    @staticmethod
    def random_symbols(count: int, samples_per_symbol: int) -> np.ndarray:
        """Create a repeatable bipolar NRZ data stream for digital examples."""
        number_of_symbols = int(np.ceil(count / samples_per_symbol)) + 1
        bits = np.random.default_rng(47).integers(0, 2, number_of_symbols)
        symbols = 2 * bits - 1
        return np.repeat(symbols, samples_per_symbol)[:count].astype(float)

    @staticmethod
    def random_qam_symbols(
        count: int,
        samples_per_symbol: int,
        order: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create normalized square-QAM I/Q symbol streams for requested order."""
        square_root = int(np.sqrt(order))
        if square_root * square_root != order:
            raise ValueError("QAM order must be a square value such as 16, 64, 256, or 1024.")
        rng = np.random.default_rng(83)
        symbol_count = int(np.ceil(count / samples_per_symbol)) + 1
        levels = np.arange(-(square_root - 1), square_root, 2, dtype=float)
        i_symbols = rng.choice(levels, symbol_count)
        q_symbols = rng.choice(levels, symbol_count)
        normalization = np.sqrt(2 * np.mean(levels**2))
        in_phase = np.repeat(i_symbols, samples_per_symbol)[:count] / normalization
        quadrature = np.repeat(q_symbols, samples_per_symbol)[:count] / normalization
        return in_phase, quadrature

    @classmethod
    def add_noise(
        cls,
        config: SimulationConfig,
        time: np.ndarray,
    ) -> np.ndarray | None:
        """Generate the selected noise graph type as the optional channel noise source."""
        if config.noise is None:
            return None
        return cls.source_waveform(time, config.noise)

    @staticmethod
    def apply_channel_effects(
        clean: np.ndarray,
        noise: np.ndarray | None,
        time: np.ndarray,
        sample_rate: float,
        settings: ChannelModulationSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
        """Apply the user-selected channel modulation impairments in a fixed order."""
        altered_clean = clean.copy()
        altered_noise = None if noise is None else noise.copy()
        applied: list[str] = []

        if "Attenuation" in settings.enabled:
            gain = 10 ** (-settings.attenuation_db / 20)
            altered_clean *= gain
            if altered_noise is not None:
                altered_noise *= gain
            applied.append(f"Attenuation ({settings.attenuation_db:g} dB)")

        if "Delay" in settings.enabled:
            delay_samples = max(0, int(settings.delay_ms * sample_rate / 1_000))
            altered_clean = np.roll(altered_clean, delay_samples)
            altered_clean[:delay_samples] = 0
            if altered_noise is not None:
                altered_noise = np.roll(altered_noise, delay_samples)
                altered_noise[:delay_samples] = 0
            applied.append(f"Delay ({settings.delay_ms:g} ms)")

        if "Interference" in settings.enabled:
            interference = settings.interference_amplitude * np.sin(
                2 * np.pi * settings.interference_frequency_hz * time
            )
            altered_noise = interference if altered_noise is None else altered_noise + interference
            applied.append(
                f"Interference ({settings.interference_amplitude:g} V @ "
                f"{settings.interference_frequency_hz:g} Hz)"
            )

        if "Shift (Doppler Shift)" in settings.enabled:
            analytic_clean = signal.hilbert(altered_clean)
            altered_clean = np.real(
                analytic_clean * np.exp(1j * 2 * np.pi * settings.doppler_shift_hz * time)
            )
            applied.append(f"Doppler shift ({settings.doppler_shift_hz:g} Hz)")

        if "Fading (Multipath Fading)" in settings.enabled:
            path_delay = max(1, int(settings.multipath_delay_ms * sample_rate / 1_000))
            delayed_path = np.roll(altered_clean, path_delay)
            delayed_path[:path_delay] = 0
            altered_clean = altered_clean + settings.multipath_gain * delayed_path
            applied.append(
                f"Multipath fading ({settings.multipath_delay_ms:g} ms, "
                f"gain {settings.multipath_gain:g})"
            )

        combined = altered_clean if altered_noise is None else altered_clean + altered_noise
        return combined, altered_clean, altered_noise, applied

    @classmethod
    def simulate(
        cls,
        config: SimulationConfig,
        method: str,
        analog_settings: AnalogModulationSettings | None = None,
        channel_settings: ChannelModulationSettings | None = None,
    ) -> SignalResult:
        """Generate one selected analogue or digital modulation waveform."""
        config.validate()
        time, sample_rate = cls.make_timebase(config)
        carrier = config.carrier
        message = config.message
        analog_settings = analog_settings or AnalogModulationSettings()
        channel_settings = channel_settings or ChannelModulationSettings()
        message_signal = cls.source_waveform(time, message)
        carrier_phase = 2 * np.pi * carrier.frequency * time
        carrier_base = carrier.amplitude * cls.waveform_from_phase(
            carrier_phase,
            "Cosine" if carrier.waveform == "Digital Data" else carrier.waveform,
        )
        modulation = method.upper()
        is_digital = modulation in DIGITAL_MODULATIONS or modulation == "QM(QAM)"

        if modulation == "AM":
            modulation_index = analog_settings.am_index
            clean = (
                1 + modulation_index * message_signal / message.amplitude
            ) * carrier_base

        elif modulation == "FM":
            frequency_deviation = analog_settings.fm_deviation_hz
            phase_deviation = 2 * np.pi * frequency_deviation * np.cumsum(message_signal) / sample_rate
            clean = carrier.amplitude * cls.waveform_from_phase(
                carrier_phase + phase_deviation,
                "Cosine" if carrier.waveform == "Digital Data" else carrier.waveform,
            )

        elif modulation == "PM":
            phase_deviation = analog_settings.pm_deviation_rad * message_signal / message.amplitude
            clean = carrier.amplitude * cls.waveform_from_phase(
                carrier_phase + phase_deviation,
                "Cosine" if carrier.waveform == "Digital Data" else carrier.waveform,
            )

        elif modulation in {"PWM", "GM(PWM)"}:
            duty_cycle = np.clip(
                0.5 + analog_settings.pwm_depth * message_signal / message.amplitude,
                0.05,
                0.95,
            )
            clean = carrier.amplitude * signal.square(carrier_phase, duty=duty_cycle)

        elif modulation in {"QAM", "QM(QAM)"}:
            samples_per_symbol = max(8, int(sample_rate / max(message.frequency * 12, 1)))
            in_phase, quadrature = cls.random_qam_symbols(
                len(time), samples_per_symbol, analog_settings.qam_order
            )
            message_signal = in_phase
            clean = carrier.amplitude * (
                in_phase * np.cos(carrier_phase) - quadrature * np.sin(carrier_phase)
            )

        else:
            samples_per_symbol = max(8, int(sample_rate / max(message.frequency * 12, 1)))
            in_phase = cls.random_symbols(len(time), samples_per_symbol)
            quadrature = cls.random_symbols(len(time), samples_per_symbol)[::-1]
            message_signal = in_phase

            if modulation == "ASK":
                envelope = (in_phase + 1) / 2
                clean = envelope * carrier_base

            elif modulation == "FSK":
                deviation = message.frequency * 0.70
                instantaneous_frequency = carrier.frequency + in_phase * deviation
                phase = 2 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate
                clean = carrier.amplitude * np.cos(phase)

            elif modulation == "PSK":
                clean = in_phase * carrier_base

            elif modulation == "QPSK":
                clean = carrier.amplitude * (
                    in_phase * np.cos(carrier_phase)
                    - quadrature * np.sin(carrier_phase)
                ) / np.sqrt(2)

            elif modulation == "GMSK":
                gaussian = signal.windows.gaussian(31, std=5)
                shaped = signal.lfilter(gaussian / gaussian.sum(), [1], in_phase)
                phase = carrier_phase + 0.9 * np.cumsum(shaped)
                clean = carrier.amplitude * np.cos(phase)

            else:
                raise ValueError(f"Unsupported modulation method: {method}")

        noise = cls.add_noise(config, time)
        combined, clean, noise, effects = cls.apply_channel_effects(
            clean,
            noise,
            time,
            sample_rate,
            channel_settings,
        )
        return SignalResult(
            name=method,
            time=time,
            samples=combined,
            clean_signal=clean,
            message_signal=message_signal,
            noise_signal=noise,
            sample_rate=sample_rate,
            digital=is_digital,
            channel_effects=effects,
        )

    @staticmethod
    def spectrum(samples: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
        """Return one-sided FFT frequencies and magnitude in dB."""
        window = np.hanning(len(samples))
        transform = np.fft.rfft(samples * window)
        frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
        magnitude_db = 20 * np.log10(np.maximum(np.abs(transform) / len(samples), 1e-12))
        return frequencies, magnitude_db

    @staticmethod
    def make_trigger(time: np.ndarray, shape: str, frequency: float) -> np.ndarray:
        """Build a normalized trigger waveform for synchronization analysis."""
        phase = 2 * np.pi * frequency * time
        trigger_builders: dict[str, Callable[[], np.ndarray]] = {
            "Triangle": lambda: signal.sawtooth(phase, width=0.5),
            "Square": lambda: signal.square(phase),
            "Sinusoidal": lambda: np.sin(phase),
        }
        return trigger_builders[shape]()

    @staticmethod
    def multiplex(results: list[SignalResult], mode: str) -> tuple[np.ndarray, np.ndarray]:
        """Create an educational TDM/FDM/WDM/CDM/OFDM composite from result signals.

        WDM is represented by independent spectral channels because this is an
        electrical baseband simulator; in a fibre system those channel labels
        correspond to optical wavelengths.  CDM applies deterministic spreading
        codes, while OFDM places signals on orthogonal subcarriers.
        """
        if not results:
            raise ValueError("Run at least one signal before multiplexing.")
        point_count = min(len(item.samples) for item in results)
        signals = np.array([item.samples[:point_count] for item in results])
        time = results[0].time[:point_count]

        if mode in {"FDM", "WDM"}:
            # The distinct modulated carrier components occupy their own spectral regions.
            combined = np.sum(signals, axis=0) / len(signals)
        elif mode == "TDM":
            combined = np.zeros(point_count)
            indices = np.arange(point_count)
            slot_size = max(20, point_count // 400)
            for channel_number, waveform in enumerate(signals):
                slot_mask = (indices // slot_size) % len(signals) == channel_number
                combined[slot_mask] = waveform[slot_mask]
        elif mode == "CDM":
            # Different Walsh-like square-wave spreading sequences share time/frequency.
            chip_count = max(8, point_count // 400)
            combined = np.zeros(point_count)
            indexes = np.arange(point_count)
            for channel_number, waveform in enumerate(signals):
                chips = np.sign(np.sin(2 * np.pi * (channel_number + 1) * indexes / chip_count))
                chips[chips == 0] = 1
                combined += waveform * chips
            combined /= len(signals)
        elif mode == "OFDM":
            # Orthogonal carrier spacing of one period over the selected duration.
            duration = max(time[-1] - time[0], 1 / results[0].sample_rate)
            combined = np.zeros(point_count)
            for channel_number, waveform in enumerate(signals):
                subcarrier = np.cos(2 * np.pi * (channel_number + 1) * time / duration)
                combined += waveform * subcarrier
            combined /= len(signals)
        else:
            raise ValueError(f"Unsupported multiplexing technique: {mode}")
        return time, combined

    @staticmethod
    def constellation(result: SignalResult) -> tuple[np.ndarray, np.ndarray]:
        """Estimate I/Q samples by mixing with the known carrier region."""
        sample_count = min(4_000, len(result.samples))
        # A simple local oscillator gives a useful constellation visualization.
        time = result.time[:sample_count]
        peak_frequency = SignalEngine.spectrum(result.clean_signal, result.sample_rate)[0][
            np.argmax(SignalEngine.spectrum(result.clean_signal, result.sample_rate)[1])
        ]
        local_i = np.cos(2 * np.pi * peak_frequency * time)
        local_q = -np.sin(2 * np.pi * peak_frequency * time)
        i_values = result.samples[:sample_count] * local_i
        q_values = result.samples[:sample_count] * local_q
        return i_values, q_values

    @staticmethod
    def eye_segments(result: SignalResult) -> tuple[np.ndarray, list[np.ndarray]]:
        """Create overlapping traces suitable for an eye-diagram display."""
        samples_per_symbol = max(8, int(result.sample_rate / max(12_000, 1)))
        width = samples_per_symbol * 2
        segment_count = min(100, len(result.samples) // width)
        x_axis = np.arange(width) / result.sample_rate
        segments = [result.samples[index * samples_per_symbol:index * samples_per_symbol + width]
                    for index in range(segment_count)]
        return x_axis, segments

    @staticmethod
    def channel_metrics(result: SignalResult, attenuation_db: float = 0.0) -> dict[str, float | None]:
        """Calculate core channel values after a configurable path attenuation.

        Attenuation is applied to the clean component only.  The existing noise
        sequence is added at the receiver, modelling the common case where
        receiver/channel noise is independent of transmit attenuation.
        """
        linear_attenuation = 10 ** (-attenuation_db / 20)
        received_clean = result.clean_signal * linear_attenuation
        signal_power = float(np.mean(received_clean**2))
        noise_power = (
            float(np.mean(result.noise_signal**2)) if result.noise_signal is not None else 0.0
        )
        received_power = signal_power + noise_power
        snr_db = None if noise_power == 0 else 10 * np.log10(signal_power / noise_power)
        return {
            "attenuation_db": attenuation_db,
            "attenuation_linear": linear_attenuation,
            "signal_power_w": signal_power,
            "noise_power_w": noise_power,
            "received_power_w": received_power,
            "snr_db": snr_db,
        }

    @staticmethod
    def estimate_error_rates(result: SignalResult) -> tuple[float | None, float | None]:
        """Return theoretical AWGN BER/SER estimates for digital examples.

        These values are analytical estimates based on the measured SNR rather
        than claims of a standards-compliant receiver's measured bit errors.
        They deliberately return None for analogue methods and zero when no
        noise is enabled, which makes their assumptions obvious to students.
        """
        if not result.digital:
            return None, None
        if result.noise_signal is None:
            return 0.0, 0.0
        snr_linear = 10 ** (max(result.snr_db or -300, -300) / 10)
        if result.name in {"ASK", "FSK", "PSK", "GMSK"}:
            ber = float(0.5 * erfc(np.sqrt(snr_linear)))
            ser = ber
        elif result.name == "QPSK":
            ber = float(0.5 * erfc(np.sqrt(snr_linear)))
            ser = float(1 - (1 - ber) ** 2)
        else:  # QAM: educational 4-QAM / QPSK-equivalent approximation.
            ber = float(0.5 * erfc(np.sqrt(snr_linear / 2)))
            ser = float(1 - (1 - ber) ** 2)
        return ber, ser

    @staticmethod
    def filter_signal(
        samples: np.ndarray,
        sample_rate: float,
        filter_type: str,
        cutoff_low: float,
        cutoff_high: float | None = None,
        order: int = 4,
    ) -> np.ndarray:
        """Apply a numerically stable Butterworth LPF, HPF, BPF, or BSF.

        The function uses second-order sections so it stays well behaved for
        laboratory-sized simulations.  Input validation prevents invalid
        normalized cutoffs before SciPy can produce an unclear error message.
        """
        nyquist = sample_rate / 2
        if not 0 < cutoff_low < nyquist:
            raise ValueError("The lower cutoff must be between 0 and the Nyquist frequency.")
        normalized_low = cutoff_low / nyquist
        if filter_type in {"BPF", "BSF"}:
            if cutoff_high is None or not cutoff_low < cutoff_high < nyquist:
                raise ValueError("For BPF/BSF, upper cutoff must be above lower cutoff and below Nyquist.")
            normalized_high = cutoff_high / nyquist
            btype = "bandpass" if filter_type == "BPF" else "bandstop"
            sections = signal.butter(order, [normalized_low, normalized_high], btype=btype, output="sos")
        else:
            btype = "lowpass" if filter_type == "LPF" else "highpass"
            sections = signal.butter(order, normalized_low, btype=btype, output="sos")
        return signal.sosfiltfilt(sections, samples)

    @staticmethod
    def filter_response(
        sample_rate: float,
        filter_type: str,
        cutoff_low: float,
        cutoff_high: float | None,
        order: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the selected filter's frequency response in decibels."""
        nyquist = sample_rate / 2
        normalized_low = min(max(cutoff_low / nyquist, 1e-6), 0.999)
        if filter_type in {"BPF", "BSF"}:
            if cutoff_high is None:
                cutoff_high = min(cutoff_low * 2, nyquist * 0.95)
            normalized_high = min(max(cutoff_high / nyquist, normalized_low + 1e-5), 0.9999)
            btype = "bandpass" if filter_type == "BPF" else "bandstop"
            sections = signal.butter(order, [normalized_low, normalized_high], btype=btype, output="sos")
        else:
            btype = "lowpass" if filter_type == "LPF" else "highpass"
            sections = signal.butter(order, normalized_low, btype=btype, output="sos")
        frequency, response = signal.sosfreqz(sections, fs=sample_rate)
        return frequency, 20 * np.log10(np.maximum(np.abs(response), 1e-12))


# -----------------------------------------------------------------------------
# Export functions
# -----------------------------------------------------------------------------

class Exporter:
    """All file output operations used by the Results window."""

    @staticmethod
    def save_csv(path: str, result: SignalResult) -> None:
        """Export raw components of one selected waveform to a spreadsheet-ready CSV."""
        noise_values: Iterable[float | str]
        noise_values = result.noise_signal if result.noise_signal is not None else [""] * len(result.time)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time_s", "combined_v", "clean_v", "message", "noise_v"])
            writer.writerows(
                zip(result.time, result.samples, result.clean_signal, result.message_signal, noise_values)
            )

    @staticmethod
    def save_json(path: str, config: SimulationConfig, results: list[SignalResult]) -> None:
        """Save configuration and key measurements without huge binary waveform arrays."""
        payload = {
            "configuration": asdict(config),
            "signals": [
                {
                    "modulation": item.name,
                    "sample_rate_hz": item.sample_rate,
                    "sample_count": len(item.samples),
                    "peak_to_peak_v": item.peak_to_peak,
                    "rms_voltage_v": item.rms_voltage,
                    "snr_db": item.snr_db,
                    "channel_effects": item.channel_effects,
                }
                for item in results
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def save_plot_png(path: str, plot_item: pg.PlotItem) -> None:
        """Save the currently visible PyQtGraph scope plot as a high-resolution PNG."""
        image_exporter = pyqtgraph.exporters.ImageExporter(plot_item)
        image_exporter.parameters()["width"] = 1920
        image_exporter.export(path)

    @staticmethod
    def save_pdf_report(path: str, config: SimulationConfig, results: list[SignalResult]) -> None:
        """Create a concise PDF report containing inputs and result measurements."""
        writer = QPdfWriter(path)
        writer.setTitle(f"{APPLICATION_NAME} Report")
        writer.setResolution(144)
        painter = QPainter(writer)
        painter.setPen(QColor("#111827"))
        header_font = painter.font()
        header_font.setPointSize(17)
        header_font.setBold(True)
        painter.setFont(header_font)
        painter.drawText(90, 100, f"{APPLICATION_NAME} – Simulation Report")
        normal_font = painter.font()
        normal_font.setPointSize(10)
        normal_font.setBold(False)
        painter.setFont(normal_font)
        lines = [
            "",
            f"Carrier: {config.carrier.frequency:g} Hz, {config.carrier.amplitude:g} V, rate {config.carrier.rate:g} Hz",
            f"Message: {config.message.frequency:g} Hz, {config.message.amplitude:g} V, rate {config.message.rate:g} Hz",
            "Noise: disabled" if config.noise is None else (
                f"Noise: {config.noise.frequency:g} Hz reference, {config.noise.amplitude:g} V, rate {config.noise.rate:g} Hz"
            ),
            f"Duration: {config.duration:g} seconds",
            "",
            "Results:",
        ]
        for result in results:
            snr_text = "noise disabled" if result.snr_db is None else f"SNR {result.snr_db:.2f} dB"
            lines.append(
                f"• {result.name}: {len(result.samples):,} samples, "
                f"Vpp {result.peak_to_peak:.3f} V, RMS {result.rms_voltage:.3f} V, {snr_text}"
            )
        for row, line in enumerate(lines):
            painter.drawText(90, 145 + row * 34, line)
        painter.end()


# -----------------------------------------------------------------------------
# Reusable plotting widgets
# -----------------------------------------------------------------------------

def configure_plot(plot: pg.PlotWidget, background: str = "#090d13") -> None:
    """Apply professional oscilloscope defaults to a PyQtGraph plot."""
    plot.setBackground(background)
    plot.showGrid(x=True, y=True, alpha=0.28)
    plot.getPlotItem().hideButtons()
    plot.getPlotItem().setMenuEnabled(False)


class ScopeWidget(QWidget):
    """Interactive oscilloscope panel used in each selected-modulation tab."""

    plot_changed = pyqtSignal()

    def __init__(self, result: SignalResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.paused = False
        self.start_index = 0
        self.time_window_seconds = min(0.008, result.time[-1])
        self.vertical_scale: float | None = None
        self._building = True

        self.plot = pg.PlotWidget()
        configure_plot(self.plot)
        self.plot.setLabel("left", "Amplitude", units="V")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.addLegend(offset=(12, 12))

        self.signal_curve = self.plot.plot(
            pen=pg.mkPen(TRACE_COLOURS["Cyan"], width=1.45),
            name="Combined signal",
        )
        self.clean_curve = self.plot.plot(
            pen=pg.mkPen(TRACE_COLOURS["Amber"], width=1.0),
            name="Clean signal",
        )
        self.vertical_cursor = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#e2e8f0", width=1),
        )
        self.horizontal_cursor = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen("#64748b", width=1),
        )
        self.plot.addItem(self.vertical_cursor)
        self.plot.addItem(self.horizontal_cursor)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Time domain", "FFT spectrum"])
        self.mode_combo.currentIndexChanged.connect(self.render)

        self.timebase_spin = QDoubleSpinBox()
        self.timebase_spin.setRange(0.00005, max(result.time[-1], 0.0001))
        self.timebase_spin.setDecimals(5)
        self.timebase_spin.setSingleStep(0.001)
        self.timebase_spin.setSuffix(" s")
        self.timebase_spin.setValue(self.time_window_seconds)
        self.timebase_spin.valueChanged.connect(self._change_timebase)

        self.vertical_spin = QDoubleSpinBox()
        self.vertical_spin.setRange(0.01, max(np.max(np.abs(result.samples)) * 5, 1.0))
        self.vertical_spin.setDecimals(3)
        self.vertical_spin.setSingleStep(0.1)
        self.vertical_spin.setSuffix(" V")
        self.vertical_spin.setValue(max(np.max(np.abs(result.samples)) * 1.15, 0.1))
        self.vertical_spin.valueChanged.connect(self._change_vertical_scale)

        self.trigger_slider = QSlider(Qt.Orientation.Horizontal)
        self.trigger_slider.setRange(0, 100)
        self.trigger_slider.setValue(0)
        self.trigger_slider.setToolTip("Move the observation start position")
        self.trigger_slider.valueChanged.connect(self._change_trigger_position)

        self.trace_colour_combo = QComboBox()
        self.trace_colour_combo.addItems(TRACE_COLOURS.keys())
        self.trace_colour_combo.currentTextChanged.connect(self._change_trace_colour)

        self.background_combo = QComboBox()
        self.background_combo.addItems(BACKGROUND_COLOURS.keys())
        self.background_combo.currentTextChanged.connect(self._change_background)

        self.grid_check = QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.grid_check.toggled.connect(self._set_grid_visible)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setProperty("secondary", True)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.autoscale_button = QPushButton("Autoscale")
        self.autoscale_button.setProperty("secondary", True)
        self.autoscale_button.clicked.connect(self.autoscale)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setProperty("secondary", True)
        self.reset_button.clicked.connect(self.reset)

        self.measurement_label = QLabel()
        self.measurement_label.setObjectName("metric")
        self.cursor_label = QLabel()
        self.cursor_label.setObjectName("metric")

        controls = QGridLayout()
        controls.addWidget(QLabel("Display:"), 0, 0)
        controls.addWidget(self.mode_combo, 0, 1)
        controls.addWidget(QLabel("Timebase:"), 0, 2)
        controls.addWidget(self.timebase_spin, 0, 3)
        controls.addWidget(QLabel("Vertical:"), 0, 4)
        controls.addWidget(self.vertical_spin, 0, 5)
        controls.addWidget(QLabel("Trigger position:"), 1, 0)
        controls.addWidget(self.trigger_slider, 1, 1, 1, 2)
        controls.addWidget(QLabel("Trace:"), 1, 3)
        controls.addWidget(self.trace_colour_combo, 1, 4)
        controls.addWidget(QLabel("Background:"), 1, 5)
        controls.addWidget(self.background_combo, 1, 6)
        controls.addWidget(self.grid_check, 0, 6)
        controls.addWidget(self.pause_button, 2, 0)
        controls.addWidget(self.autoscale_button, 2, 1)
        controls.addWidget(self.reset_button, 2, 2)
        controls.addWidget(self.measurement_label, 2, 3, 1, 2)
        controls.addWidget(self.cursor_label, 2, 5, 1, 2)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(controls)

        self.vertical_cursor.sigPositionChanged.connect(self._update_cursor_readout)
        self.horizontal_cursor.sigPositionChanged.connect(self._update_cursor_readout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_live_display)
        self.timer.start(80)
        self._building = False
        self.render()

    @property
    def plot_item(self) -> pg.PlotItem:
        """Expose the plot for image export."""
        return self.plot.getPlotItem()

    def _visible_indices(self) -> np.ndarray:
        """Find the sample indexes visible in the time-domain scope window."""
        count = max(200, int(self.time_window_seconds * self.result.sample_rate))
        count = min(count, len(self.result.time))
        return (np.arange(count) + self.start_index) % len(self.result.time)

    def _advance_live_display(self) -> None:
        """Scroll the visible section for a live-acquisition effect."""
        if self.paused or self.mode_combo.currentText() != "Time domain":
            return
        advance = max(1, int(self.result.sample_rate * self.time_window_seconds / 80))
        self.start_index = (self.start_index + advance) % len(self.result.time)
        self.render()

    def _change_timebase(self, value: float) -> None:
        self.time_window_seconds = value
        if not self._building:
            self.render()

    def _change_vertical_scale(self, value: float) -> None:
        self.vertical_scale = value
        if not self._building:
            self.render()

    def _change_trigger_position(self, value: int) -> None:
        max_start = max(0, len(self.result.time) - 1)
        self.start_index = int(max_start * value / 100)
        if not self._building:
            self.render()

    def _change_trace_colour(self, name: str) -> None:
        self.signal_curve.setPen(pg.mkPen(TRACE_COLOURS[name], width=1.45))

    def _change_background(self, name: str) -> None:
        self.plot.setBackground(BACKGROUND_COLOURS[name])

    def _set_grid_visible(self, visible: bool) -> None:
        self.plot.showGrid(x=visible, y=visible, alpha=0.28)

    def _update_cursor_readout(self) -> None:
        """Show voltage/time values of the two draggable cursor lines."""
        x_value = self.vertical_cursor.value()
        y_value = self.horizontal_cursor.value()
        self.cursor_label.setText(f"Cursor: t={x_value:.6f} s, V={y_value:.3f} V")

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def autoscale(self) -> None:
        self.vertical_scale = None
        self.plot.autoRange()
        self.render()

    def reset(self) -> None:
        self.paused = False
        self.pause_button.setText("Pause")
        self.start_index = 0
        self.trigger_slider.setValue(0)
        self.timebase_spin.setValue(min(0.008, self.result.time[-1]))
        self.vertical_scale = None
        self.plot.autoRange()
        self.render()

    def render(self) -> None:
        """Draw the current time or spectrum scope view."""
        if self.mode_combo.currentText() == "FFT spectrum":
            frequency, magnitude = SignalEngine.spectrum(self.result.samples, self.result.sample_rate)
            self.signal_curve.setData(frequency, magnitude)
            self.clean_curve.setData([], [])
            self.plot.setLabel("bottom", "Frequency", units="Hz")
            self.plot.setLabel("left", "Magnitude", units="dB")
            self.vertical_cursor.setVisible(False)
            self.horizontal_cursor.setVisible(False)
            peak_frequency = frequency[int(np.argmax(magnitude))]
            self.measurement_label.setText(
                f"Peak: {peak_frequency:,.1f} Hz | fs: {self.result.sample_rate:,.0f} Hz"
            )
        else:
            indexes = self._visible_indices()
            visible_time = self.result.time[indexes]
            self.signal_curve.setData(visible_time, self.result.samples[indexes])
            self.clean_curve.setData(visible_time, self.result.clean_signal[indexes])
            self.plot.setLabel("bottom", "Time", units="s")
            self.plot.setLabel("left", "Amplitude", units="V")
            self.vertical_cursor.setVisible(True)
            self.horizontal_cursor.setVisible(True)
            self.plot.setXRange(float(visible_time[0]), float(visible_time[-1]), padding=0)
            if self.vertical_scale is not None:
                self.plot.setYRange(-self.vertical_scale, self.vertical_scale, padding=0)
            snr_text = "Noise off" if self.result.snr_db is None else f"SNR: {self.result.snr_db:.2f} dB"
            self.measurement_label.setText(
                f"Vpp: {self.result.peak_to_peak:.3f} V | RMS: {self.result.rms_voltage:.3f} V | {snr_text}"
            )
        self.plot_changed.emit()


class SimplePlot(QWidget):
    """A labelled PyQtGraph plot card used in modal analysis windows."""

    def __init__(
        self,
        title: str,
        x_values: np.ndarray,
        y_values: np.ndarray,
        x_label: str = "Time (s)",
        y_label: str = "Amplitude",
        colour: str = "#38bdf8",
    ) -> None:
        super().__init__()
        self.plot = pg.PlotWidget(title=title)
        configure_plot(self.plot)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.plot(x_values, y_values, pen=pg.mkPen(colour, width=1.2))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)


class AdvancedAnalysisDialog(QDialog):
    """Tabbed noise, constellation, eye, waterfall, and spectrogram analysis."""

    def __init__(self, result: SignalResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.result = result
        self.setWindowTitle(f"Advanced Signal Analysis — {result.name}")
        self.resize(1100, 760)

        tabs = QTabWidget()
        tabs.addTab(self._signal_tab(), "Signal and noise")
        tabs.addTab(self._spectrum_tab(), "Spectrum")
        tabs.addTab(self._constellation_tab(), "Constellation")
        tabs.addTab(self._eye_tab(), "Eye diagram")
        tabs.addTab(self._waterfall_tab(), "Waterfall")
        tabs.addTab(self._matplotlib_tab(), "Spectrogram")

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs, stretch=1)
        layout.addWidget(close_box)

    def _signal_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        limit = min(len(self.result.time), 12_000)
        time = self.result.time[:limit]
        layout.addWidget(SimplePlot("Clean / original signal", time, self.result.clean_signal[:limit]), 0, 0)
        if self.result.noise_signal is not None:
            layout.addWidget(SimplePlot("Noise signal", time, self.result.noise_signal[:limit], colour="#f59e0b"), 0, 1)
        else:
            label = QLabel("Noise was not enabled for this simulation.")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, 0, 1)
        layout.addWidget(SimplePlot("Combined observed signal", time, self.result.samples[:limit], colour="#4ade80"), 1, 0, 1, 2)
        return page

    def _spectrum_tab(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        frequency, combined = SignalEngine.spectrum(self.result.samples, self.result.sample_rate)
        layout.addWidget(SimplePlot("Combined signal spectrum", frequency, combined, "Frequency (Hz)", "Magnitude (dB)"), 0, 0)
        if self.result.noise_signal is not None:
            noise_frequency, noise_magnitude = SignalEngine.spectrum(self.result.noise_signal, self.result.sample_rate)
            layout.addWidget(SimplePlot("Noise spectrum", noise_frequency, noise_magnitude, "Frequency (Hz)", "Magnitude (dB)", "#f59e0b"), 0, 1)
            statistic = (
                f"Noise mean: {np.mean(self.result.noise_signal):.5f} V\n"
                f"Noise standard deviation: {np.std(self.result.noise_signal):.5f} V\n"
                f"Noise RMS: {np.sqrt(np.mean(self.result.noise_signal**2)):.5f} V\n"
                f"Signal-to-noise ratio: {self.result.snr_db:.2f} dB"
            )
        else:
            statistic = "Noise analysis is unavailable because no noise waveform was enabled."
        statistics_label = QLabel(statistic)
        statistics_label.setObjectName("metric")
        layout.addWidget(statistics_label, 1, 0, 1, 2)
        return page

    def _constellation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        explanation = QLabel(
            "I/Q sample cloud. QPSK and QAM create the clearest constellations; "
            "analogue modes are shown for comparison."
        )
        layout.addWidget(explanation)
        plot = pg.PlotWidget(title="I/Q constellation")
        configure_plot(plot)
        plot.setLabel("bottom", "In-phase (I)")
        plot.setLabel("left", "Quadrature (Q)")
        i_values, q_values = SignalEngine.constellation(self.result)
        plot.plot(i_values[::4], q_values[::4], pen=None, symbol="o", symbolSize=4, symbolBrush="#38bdf8")
        plot.setAspectLocked(True)
        layout.addWidget(plot, stretch=1)
        return page

    def _eye_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Overlaid two-symbol traces form an eye-pattern representation."))
        plot = pg.PlotWidget(title="Eye diagram")
        configure_plot(plot)
        plot.setLabel("bottom", "Time (s)")
        plot.setLabel("left", "Amplitude (V)")
        x_axis, segments = SignalEngine.eye_segments(self.result)
        for segment in segments:
            plot.plot(x_axis[:len(segment)], segment, pen=pg.mkPen("#38bdf8", width=0.8))
        layout.addWidget(plot, stretch=1)
        return page

    def _waterfall_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Successive short-time FFT magnitudes, rendered as a waterfall image."))
        plot = pg.PlotWidget(title="Waterfall / short-time spectrum")
        configure_plot(plot)
        image = pg.ImageItem()
        plot.addItem(image)
        segment_size = min(512, max(64, len(self.result.samples) // 80))
        usable = len(self.result.samples) // segment_size * segment_size
        blocks = self.result.samples[:usable].reshape(-1, segment_size)
        data = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(blocks * np.hanning(segment_size), axis=1)), 1e-12))
        image.setImage(data.T, autoLevels=True)
        plot.setLabel("bottom", "Time block")
        plot.setLabel("left", "Frequency bin")
        layout.addWidget(plot, stretch=1)
        return page

    def _matplotlib_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Matplotlib embedded in PyQt: spectrogram view."))
        figure = Figure(facecolor="#10151e")
        canvas = FigureCanvasQTAgg(figure)
        axis = figure.add_subplot(111)
        axis.set_facecolor("#090d13")
        axis.specgram(self.result.samples, Fs=self.result.sample_rate, NFFT=256, noverlap=192, cmap="turbo")
        axis.set_xlabel("Time (s)", color="#dce6f2")
        axis.set_ylabel("Frequency (Hz)", color="#dce6f2")
        axis.tick_params(colors="#dce6f2")
        figure.tight_layout()
        layout.addWidget(canvas, stretch=1)
        return page


class WaterfallPanel(QWidget):
    """Dedicated signal-analysis page with spectrum analyzer and waterfall plot."""

    def __init__(self, result: SignalResult) -> None:
        super().__init__()
        self.result = result
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        frequency, magnitude = SignalEngine.spectrum(result.samples, result.sample_rate)
        spectrum_plot = pg.PlotWidget(title="Spectrum analyzer")
        configure_plot(spectrum_plot)
        spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        spectrum_plot.setLabel("left", "Magnitude", units="dB")
        spectrum_plot.plot(frequency, magnitude, pen=pg.mkPen("#38bdf8", width=1.2))

        waterfall_plot = pg.PlotWidget(title="Waterfall plot (short-time FFT)")
        configure_plot(waterfall_plot)
        waterfall_plot.setLabel("bottom", "Time block")
        waterfall_plot.setLabel("left", "Frequency bin")
        waterfall = pg.ImageItem()
        waterfall_plot.addItem(waterfall)
        block_size = min(512, max(64, len(result.samples) // 80))
        usable_samples = len(result.samples) // block_size * block_size
        blocks = result.samples[:usable_samples].reshape(-1, block_size)
        block_window = np.hanning(block_size)
        magnitude_matrix = 20 * np.log10(
            np.maximum(np.abs(np.fft.rfft(blocks * block_window, axis=1)), 1e-12)
        )
        waterfall.setImage(magnitude_matrix.T, autoLevels=True)

        metrics = QLabel(
            f"Peak-to-peak voltage: {result.peak_to_peak:.4f} V\n"
            f"RMS voltage: {result.rms_voltage:.4f} V\n"
            f"Sampling rate: {result.sample_rate:,.0f} Hz\n"
            f"Noise: {'disabled' if result.snr_db is None else f'{result.snr_db:.2f} dB SNR'}"
        )
        metrics.setObjectName("metric")
        layout.addWidget(spectrum_plot, 0, 0)
        layout.addWidget(waterfall_plot, 0, 1)
        layout.addWidget(metrics, 1, 0, 1, 2)


class FFTPanel(QWidget):
    """A focused FFT magnitude display with numerical dominant-frequency readout."""

    def __init__(self, result: SignalResult) -> None:
        super().__init__()
        frequency, magnitude = SignalEngine.spectrum(result.samples, result.sample_rate)
        dominant_index = int(np.argmax(magnitude))
        dominant_frequency = frequency[dominant_index]
        plot = pg.PlotWidget(title="FFT magnitude spectrum")
        configure_plot(plot)
        plot.setLabel("bottom", "Frequency", units="Hz")
        plot.setLabel("left", "Magnitude", units="dB")
        plot.plot(frequency, magnitude, pen=pg.mkPen("#38bdf8", width=1.25))
        peak_line = pg.InfiniteLine(
            pos=float(dominant_frequency),
            angle=90,
            pen=pg.mkPen("#f59e0b", width=1),
            label=f"Peak {dominant_frequency:,.1f} Hz",
            labelOpts={"position": 0.88, "color": "#f59e0b"},
        )
        plot.addItem(peak_line)
        detail = QLabel(
            f"FFT uses a Hann window and one-sided frequency bins. Dominant component: "
            f"{dominant_frequency:,.2f} Hz at {magnitude[dominant_index]:.2f} dB."
        )
        detail.setObjectName("metric")
        layout = QVBoxLayout(self)
        layout.addWidget(plot, stretch=1)
        layout.addWidget(detail)


class SpectrumAnalyzerPanel(QWidget):
    """Spectrum analyzer with adjustable frequency range and peak detector."""

    def __init__(self, result: SignalResult) -> None:
        super().__init__()
        self.result = result
        self.frequency, self.magnitude = SignalEngine.spectrum(result.samples, result.sample_rate)
        self.plot = pg.PlotWidget(title="Spectrum Analyzer")
        configure_plot(self.plot)
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dB")
        self.minimum_spin = self._frequency_spin(0, result.sample_rate / 2)
        self.maximum_spin = self._frequency_spin(result.sample_rate / 2, result.sample_rate / 2)
        self.minimum_spin.valueChanged.connect(self.render)
        self.maximum_spin.valueChanged.connect(self.render)
        self.peak_label = QLabel()
        self.peak_label.setObjectName("metric")
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Start frequency:"))
        controls.addWidget(self.minimum_spin)
        controls.addWidget(QLabel("Stop frequency:"))
        controls.addWidget(self.maximum_spin)
        controls.addStretch()
        controls.addWidget(self.peak_label)
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot, stretch=1)
        layout.addLayout(controls)
        self.render()

    @staticmethod
    def _frequency_spin(value: float, maximum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0, maximum)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setSuffix(" Hz")
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setAccelerated(True)
        return spin

    def render(self) -> None:
        start = min(self.minimum_spin.value(), self.maximum_spin.value())
        stop = max(self.minimum_spin.value(), self.maximum_spin.value())
        mask = (self.frequency >= start) & (self.frequency <= stop)
        if not np.any(mask):
            return
        visible_frequency = self.frequency[mask]
        visible_magnitude = self.magnitude[mask]
        self.plot.clear()
        self.plot.plot(visible_frequency, visible_magnitude, pen=pg.mkPen("#4ade80", width=1.25))
        peak_index = int(np.argmax(visible_magnitude))
        self.peak_label.setText(
            f"Peak: {visible_frequency[peak_index]:,.2f} Hz | {visible_magnitude[peak_index]:.2f} dB"
        )


class WaterfallOnlyPanel(QWidget):
    """Waterfall-only view kept separate from FFT and spectrum-analyzer choices."""

    def __init__(self, result: SignalResult) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "The waterfall encodes successive FFT time blocks by colour. "
            "Horizontal bands identify components that persist over time."
        ))
        plot = pg.PlotWidget(title="Waterfall Plot")
        configure_plot(plot)
        plot.setLabel("bottom", "Time block")
        plot.setLabel("left", "Frequency bin")
        image = pg.ImageItem()
        plot.addItem(image)
        block_size = min(512, max(64, len(result.samples) // 80))
        usable = len(result.samples) // block_size * block_size
        blocks = result.samples[:usable].reshape(-1, block_size)
        magnitude = 20 * np.log10(np.maximum(
            np.abs(np.fft.rfft(blocks * np.hanning(block_size), axis=1)), 1e-12
        ))
        image.setImage(magnitude.T, autoLevels=True)
        layout.addWidget(plot, stretch=1)


class SignalAnalysisPanel(QWidget):
    """Adds only the signal-analysis subtools the user ticked on the request page."""

    def __init__(self, results: list[SignalResult], selected_tools: Iterable[str]) -> None:
        super().__init__()
        selected = set(selected_tools)
        method_tabs = QTabWidget()
        for result in results:
            result_tabs = QTabWidget()
            if "Oscilloscope" in selected:
                result_tabs.addTab(ScopeWidget(result), "Oscilloscope")
            if "FFT" in selected:
                result_tabs.addTab(FFTPanel(result), "FFT")
            if "Spectrum Analyzer" in selected:
                result_tabs.addTab(SpectrumAnalyzerPanel(result), "Spectrum Analyzer")
            if "Waterfall Plot" in selected:
                result_tabs.addTab(WaterfallOnlyPanel(result), "Waterfall Plot")
            method_tabs.addTab(result_tabs, result.name)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Only the checked Signal Analysis subtools are present below. "
            "Return using Back, change checks, and run again whenever you want a different workspace."
        ))
        layout.addWidget(method_tabs, stretch=1)


class DigitalAnalysisPanel(QWidget):
    """Digital workspace that honours individual constellation/eye/BER/SER ticks."""

    def __init__(self, results: list[SignalResult], selected_tools: Iterable[str]) -> None:
        super().__init__()
        self.results = results
        self.selected_tools = set(selected_tools)
        self.digital_results = [item for item in results if item.digital]
        self.result_combo = QComboBox()
        self.result_combo.addItems([item.name for item in (self.digital_results or results)])
        self.result_combo.currentIndexChanged.connect(self._render)
        self.constellation_plot = pg.PlotWidget(title="Constellation diagram")
        configure_plot(self.constellation_plot)
        self.constellation_plot.setLabel("bottom", "In-phase (I)")
        self.constellation_plot.setLabel("left", "Quadrature (Q)")
        self.constellation_plot.setAspectLocked(True)
        self.eye_plot = pg.PlotWidget(title="Eye diagram")
        configure_plot(self.eye_plot)
        self.eye_plot.setLabel("bottom", "Time", units="s")
        self.eye_plot.setLabel("left", "Amplitude", units="V")
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Measurement", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setMinimumHeight(210)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Inspect modulation:"))
        controls.addWidget(self.result_combo)
        controls.addStretch()
        layout = QGridLayout(self)
        layout.addLayout(controls, 0, 0, 1, 2)
        next_row = 1
        if "Constellation Diagram" in self.selected_tools:
            layout.addWidget(self.constellation_plot, next_row, 0)
        if "Eye Diagram" in self.selected_tools:
            layout.addWidget(self.eye_plot, next_row, 1)
        if "BER" in self.selected_tools or "SER" in self.selected_tools:
            layout.addWidget(self.metrics_table, next_row + 1, 0, 1, 2)
        self._render()

    def _selected_result(self) -> SignalResult:
        source = self.digital_results or self.results
        return source[self.result_combo.currentIndex()]

    def _render(self) -> None:
        """Refresh plots and numerical error-rate estimates for selected method."""
        result = self._selected_result()
        self.constellation_plot.clear()
        self.eye_plot.clear()
        if "Constellation Diagram" in self.selected_tools:
            i_values, q_values = SignalEngine.constellation(result)
            self.constellation_plot.plot(
                i_values[::4],
                q_values[::4],
                pen=None,
                symbol="o",
                symbolSize=4,
                symbolBrush="#38bdf8",
            )
        if "Eye Diagram" in self.selected_tools:
            x_axis, segments = SignalEngine.eye_segments(result)
            for segment in segments:
                self.eye_plot.plot(
                    x_axis[:len(segment)],
                    segment,
                    pen=pg.mkPen("#4ade80", width=0.75),
                )
        ber, ser = SignalEngine.estimate_error_rates(result)
        rows: list[tuple[str, str]] = [
            ("Modulation", result.name),
            ("Signal kind", "Digital" if result.digital else "Analogue (visual comparison)"),
            ("Measured SNR", "Noise disabled" if result.snr_db is None else f"{result.snr_db:.4f} dB"),
        ]
        if "BER" in self.selected_tools:
            rows.append(("Estimated BER", "Not applicable" if ber is None else f"{ber:.6e}"))
        if "SER" in self.selected_tools:
            rows.append(("Estimated SER", "Not applicable" if ser is None else f"{ser:.6e}"))
        if "BER" in self.selected_tools or "SER" in self.selected_tools:
            rows.append(("Assumption", "Theoretical AWGN estimate; not a hardware receiver measurement"))
        self.metrics_table.setRowCount(len(rows))
        for row, (metric, value) in enumerate(rows):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(value))


class ChannelAnalysisPanel(QWidget):
    """Channel workspace limited to selected SNR, power, and attenuation tools."""

    def __init__(self, results: list[SignalResult], selected_tools: Iterable[str]) -> None:
        super().__init__()
        self.results = results
        self.selected_tools = set(selected_tools)
        self.result_combo = QComboBox()
        self.result_combo.addItems([item.name for item in results])
        self.result_combo.currentIndexChanged.connect(self._render)
        self.attenuation_spin = QDoubleSpinBox()
        self.attenuation_spin.setRange(0, 100)
        self.attenuation_spin.setDecimals(2)
        self.attenuation_spin.setSuffix(" dB")
        self.attenuation_spin.valueChanged.connect(self._render)
        self.attenuation_spin.setEnabled("Attenuation" in self.selected_tools)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Channel metric", "Value"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.channel_plot = pg.PlotWidget(title="Received waveform after attenuation")
        configure_plot(self.channel_plot)
        self.channel_plot.setLabel("bottom", "Time", units="s")
        self.channel_plot.setLabel("left", "Amplitude", units="V")
        self.power_plot = pg.PlotWidget(title="Signal / noise power comparison")
        configure_plot(self.power_plot)
        self.power_plot.setLabel("left", "Power", units="W")
        self.power_plot.setLabel("bottom", "Component")

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Modulation:"))
        controls.addWidget(self.result_combo)
        if "Attenuation" in self.selected_tools:
            controls.addWidget(QLabel("Path attenuation:"))
            controls.addWidget(self.attenuation_spin)
        controls.addStretch()
        layout = QGridLayout(self)
        layout.addLayout(controls, 0, 0, 1, 2)
        layout.addWidget(self.channel_plot, 1, 0)
        layout.addWidget(self.power_plot, 1, 1)
        layout.addWidget(self.metrics_table, 2, 0, 1, 2)
        self._render()

    def _render(self) -> None:
        result = self.results[self.result_combo.currentIndex()]
        attenuation_db = self.attenuation_spin.value()
        metrics = SignalEngine.channel_metrics(result, attenuation_db)
        linear = float(metrics["attenuation_linear"])
        received_clean = result.clean_signal * linear
        received = received_clean if result.noise_signal is None else received_clean + result.noise_signal
        sample_count = min(12_000, len(result.time))
        self.channel_plot.clear()
        self.channel_plot.plot(
            result.time[:sample_count],
            received[:sample_count],
            pen=pg.mkPen("#38bdf8", width=1.25),
            name="Received",
        )
        self.channel_plot.plot(
            result.time[:sample_count],
            received_clean[:sample_count],
            pen=pg.mkPen("#f59e0b", width=1),
            name="Attenuated clean",
        )
        self.power_plot.clear()
        values = [
            float(metrics["signal_power_w"]),
            float(metrics["noise_power_w"]),
            float(metrics["received_power_w"]),
        ]
        bars = pg.BarGraphItem(x=np.arange(3), height=values, width=0.6, brush="#4ade80")
        self.power_plot.addItem(bars)
        self.power_plot.getAxis("bottom").setTicks([[(0, "Signal"), (1, "Noise"), (2, "Received")]])
        snr_value = metrics["snr_db"]
        rows: list[tuple[str, str]] = [
            ("Selected modulation", result.name),
        ]
        if "Attenuation" in self.selected_tools:
            rows.extend([
                ("Attenuation", f"{metrics['attenuation_db']:.2f} dB"),
                ("Linear attenuation", f"{metrics['attenuation_linear']:.6f}"),
            ])
        if "Signal Power" in self.selected_tools:
            rows.append(("Signal power", f"{metrics['signal_power_w']:.6e} W"))
        if "Noise Power" in self.selected_tools:
            rows.append(("Noise power", f"{metrics['noise_power_w']:.6e} W"))
        if "SNR" in self.selected_tools:
            rows.append(("Received SNR", "Noise disabled" if snr_value is None else f"{snr_value:.4f} dB"))
        self.metrics_table.setRowCount(len(rows))
        for row, (name, value) in enumerate(rows):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(value))


class FilterAnalysisPanel(QWidget):
    """Butterworth workspace containing only the filter families the user ticked."""

    def __init__(self, results: list[SignalResult], selected_filters: Iterable[str]) -> None:
        super().__init__()
        self.results = results
        self.selected_filters = list(selected_filters)
        self.result_combo = QComboBox()
        self.result_combo.addItems([item.name for item in results])
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.selected_filters)
        selected = results[0]
        default_low = min(max(selected.sample_rate * 0.05, 10), selected.sample_rate * 0.35)
        self.low_cutoff = self._frequency_spin(default_low, selected.sample_rate)
        self.high_cutoff = self._frequency_spin(min(default_low * 2.0, selected.sample_rate * 0.45), selected.sample_rate)
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 12)
        self.order_spin.setValue(4)
        self.order_spin.setSuffix(" order")
        self.output_plot = pg.PlotWidget(title="Input and filtered output")
        configure_plot(self.output_plot)
        self.output_plot.setLabel("bottom", "Time", units="s")
        self.output_plot.setLabel("left", "Amplitude", units="V")
        self.response_plot = pg.PlotWidget(title="Filter response")
        configure_plot(self.response_plot)
        self.response_plot.setLabel("bottom", "Frequency", units="Hz")
        self.response_plot.setLabel("left", "Gain", units="dB")
        self.status_label = QLabel()
        self.status_label.setObjectName("metric")
        apply_button = QPushButton("Apply filter")
        apply_button.clicked.connect(self._apply_filter)
        for widget in (self.result_combo, self.filter_combo, self.low_cutoff, self.high_cutoff, self.order_spin):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._update_cutoff_limits)
            else:
                widget.valueChanged.connect(lambda: None)
        controls = QGridLayout()
        controls.addWidget(QLabel("Modulation:"), 0, 0)
        controls.addWidget(self.result_combo, 0, 1)
        controls.addWidget(QLabel("Filter type:"), 0, 2)
        controls.addWidget(self.filter_combo, 0, 3)
        controls.addWidget(QLabel("Lower cutoff:"), 1, 0)
        controls.addWidget(self.low_cutoff, 1, 1)
        controls.addWidget(QLabel("Upper cutoff:"), 1, 2)
        controls.addWidget(self.high_cutoff, 1, 3)
        controls.addWidget(QLabel("Filter order:"), 2, 0)
        controls.addWidget(self.order_spin, 2, 1)
        controls.addWidget(apply_button, 2, 3)
        layout = QGridLayout(self)
        layout.addLayout(controls, 0, 0, 1, 2)
        layout.addWidget(self.output_plot, 1, 0)
        layout.addWidget(self.response_plot, 1, 1)
        layout.addWidget(self.status_label, 2, 0, 1, 2)
        self._apply_filter()

    @staticmethod
    def _frequency_spin(value: float, sample_rate: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.1, max(1, sample_rate / 2 * 0.999))
        spin.setDecimals(2)
        spin.setSingleStep(10)
        spin.setValue(value)
        spin.setSuffix(" Hz")
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setAccelerated(True)
        return spin

    def _update_cutoff_limits(self) -> None:
        """Keep cutoffs valid after the user switches to a result with another rate."""
        sample_rate = self.results[self.result_combo.currentIndex()].sample_rate
        maximum = sample_rate / 2 * 0.999
        for spin in (self.low_cutoff, self.high_cutoff):
            spin.setMaximum(maximum)
            if spin.value() >= maximum:
                spin.setValue(maximum * 0.7)

    def _apply_filter(self) -> None:
        result = self.results[self.result_combo.currentIndex()]
        filter_type = self.filter_combo.currentText()
        lower = self.low_cutoff.value()
        upper = self.high_cutoff.value()
        order = self.order_spin.value()
        try:
            filtered = SignalEngine.filter_signal(
                result.samples, result.sample_rate, filter_type, lower, upper, order
            )
            frequency, response = SignalEngine.filter_response(
                result.sample_rate, filter_type, lower, upper, order
            )
        except ValueError as error:
            self.status_label.setText(f"Filter settings need adjustment: {error}")
            return
        limit = min(12_000, len(result.time))
        self.output_plot.clear()
        self.output_plot.addLegend(offset=(8, 8))
        self.output_plot.plot(result.time[:limit], result.samples[:limit], pen=pg.mkPen("#64748b", width=1), name="Input")
        self.output_plot.plot(result.time[:limit], filtered[:limit], pen=pg.mkPen("#38bdf8", width=1.4), name="Filtered")
        self.response_plot.clear()
        self.response_plot.plot(frequency, response, pen=pg.mkPen("#4ade80", width=1.4))
        self.status_label.setText(
            f"Applied {order}th-order Butterworth {filter_type}. "
            f"Input RMS {np.sqrt(np.mean(result.samples**2)):.4f} V → "
            f"output RMS {np.sqrt(np.mean(filtered**2)):.4f} V."
        )


class GenerationReportPanel(QWidget):
    """Shows the selected 0G–6G reports alongside user-input context."""

    def __init__(self, config: SimulationConfig, generations: list[str]) -> None:
        super().__init__()
        tabs = QTabWidget()
        for generation in generations:
            details = GENERATION_INFO[generation]
            browser = QTextBrowser()
            browser.setHtml(
                f"<h2>{generation} report using your entered wave values</h2>"
                f"<p><b>Your carrier:</b> {config.carrier.frequency:g} Hz, {config.carrier.amplitude:g} V, "
                f"rate {config.carrier.rate:g} Hz<br>"
                f"<b>Your message:</b> {config.message.frequency:g} Hz, {config.message.amplitude:g} V, "
                f"rate {config.message.rate:g} Hz<br>"
                f"<b>Your noise:</b> {'disabled' if config.noise is None else f'{config.noise.amplitude:g} V AWGN model'}</p>"
                f"<table>"
                f"<tr><td><b>Timeline</b></td><td>{details['timeline']}</td></tr>"
                f"<tr><td><b>Technology</b></td><td>{details['technology']}</td></tr>"
                f"<tr><td><b>Carrier bands</b></td><td>{details['carrier']}</td></tr>"
                f"<tr><td><b>Bandwidth</b></td><td>{details['bandwidth']}</td></tr>"
                f"<tr><td><b>Modulation</b></td><td>{details['modulation']}</td></tr>"
                f"<tr><td><b>Spectrum</b></td><td>{details['spectrum']}</td></tr>"
                f"<tr><td><b>Advantages</b></td><td>{details['advantages']}</td></tr>"
                f"<tr><td><b>Limitations</b></td><td>{details['limitations']}</td></tr>"
                f"<tr><td><b>Applications</b></td><td>{details['applications']}</td></tr>"
                f"<tr><td><b>Architecture</b></td><td>{details['architecture']}</td></tr>"
                f"</table>"
            )
            tabs.addTab(browser, generation)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Technology context is reference information; live analysis uses your entered values."))
        layout.addWidget(tabs, stretch=1)


class CommunicationAnalysisWindow(QMainWindow):
    """Dynamic workspace that only contains analysis groups requested by the user."""

    def __init__(
        self,
        config: SimulationConfig,
        results: list[SignalResult],
        request: AnalysisRequest,
    ) -> None:
        super().__init__()
        self.config = config
        self.results = results
        self.request = request
        self.setWindowTitle(f"{APPLICATION_NAME} — Communication Analysis")
        self.resize(1380, 850)
        self.setDockNestingEnabled(True)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self._build_navigation_toolbar()
        self._add_requested_tabs()
        self._add_trigger_and_multiplex_tabs()
        self._build_summary_dock()
        self.statusBar().showMessage("Communication Analysis complete. Tabs reflect your checkbox selections.")

    def _build_navigation_toolbar(self) -> None:
        """Results are separate windows, so Back closes them to reveal the prior page."""
        toolbar = QToolBar("Navigation", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        back_action = QAction("← Back", self)
        back_action.setToolTip("Close analysis and return to Communication Analysis setup")
        back_action.triggered.connect(self.close)
        toolbar.addAction(back_action)

    def _add_requested_tabs(self) -> None:
        selected = self.request.selected_subanalyses
        if selected["Signal Analysis"]:
            self.tabs.addTab(
                SignalAnalysisPanel(self.results, selected["Signal Analysis"]),
                "Signal Analysis",
            )
        if selected["Digital Analysis"]:
            self.tabs.addTab(
                DigitalAnalysisPanel(self.results, selected["Digital Analysis"]),
                "Digital Analysis",
            )
        if selected["Channel Analysis"]:
            self.tabs.addTab(
                ChannelAnalysisPanel(self.results, selected["Channel Analysis"]),
                "Channel Analysis",
            )
        if selected["Filter Analysis"]:
            self.tabs.addTab(
                FilterAnalysisPanel(self.results, selected["Filter Analysis"]),
                "Filter Analysis",
            )
        if self.request.generations:
            self.tabs.addTab(GenerationReportPanel(self.config, self.request.generations), "0G–6G Reports")

    def _add_trigger_and_multiplex_tabs(self) -> None:
        if self.request.trigger_shape != "None":
            result = self.results[0]
            trigger = SignalEngine.make_trigger(
                result.time, self.request.trigger_shape, self.config.message.frequency
            )
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(f"{self.request.trigger_shape} trigger synchronized to the message frequency."))
            plots = QTabWidget()
            plots.addTab(SimplePlot("Original waveform", result.time, result.clean_signal), "Original")
            plots.addTab(SimplePlot("Trigger waveform", result.time, trigger, colour="#f59e0b"), "Trigger")
            plots.addTab(SimplePlot("Combined timing relationship", result.time, result.clean_signal + trigger * self.config.carrier.amplitude * .25, colour="#4ade80"), "Combined")
            layout.addWidget(plots, stretch=1)
            self.tabs.addTab(page, "Trigger Analysis")
        if self.request.multiplex_mode != "None":
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(QLabel(
                "TDM assigns time slots; FDM uses frequency channels; WDM models wavelength channels; "
                "CDM applies spreading codes; OFDM uses orthogonal subcarriers."
            ))
            plots = QTabWidget()
            modes = ["TDM", "FDM", "WDM", "CDM", "OFDM"] if self.request.multiplex_mode == "All" else [self.request.multiplex_mode]
            for mode in modes:
                time, output = SignalEngine.multiplex(self.results, mode)
                plots.addTab(SimplePlot(f"{mode} composite", time, output, colour="#e879f9"), mode)
            layout.addWidget(plots, stretch=1)
            self.tabs.addTab(page, "Multiplexing")

    def _build_summary_dock(self) -> None:
        dock = QDockWidget("Analysis Request Summary", self)
        text = QLabel(
            f"Modulations: {', '.join(self.request.modulation_methods)}\n\n"
            "Selected sub-analysis tools:\n"
            + "\n".join(
                f"• {group}: {', '.join(tools)}"
                for group, tools in self.request.selected_subanalyses.items() if tools
            )
            + "\n\n"
            + f"Trigger: {self.request.trigger_shape}\n"
            + f"Multiplexing: {self.request.multiplex_mode}\n\n"
            + f"Generation reports: {', '.join(self.request.generations) or 'None'}"
        )
        text.setWordWrap(True)
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(text)
        layout.addStretch()
        dock.setWidget(wrapper)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)


# -----------------------------------------------------------------------------
# Configuration, navigation, and selection pages
# -----------------------------------------------------------------------------

class WelcomePage(QWidget):
    """A brief loading introduction shown before the signal-input landing page."""

    ready = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.progress_value = 0
        self.title = QLabel("Welcome to\nNebula X Comms Simulator")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setStyleSheet(
            "font-size: 36px; font-weight: 750; color: #e0f2fe; padding-top: 36px;"
        )
        subtitle = QLabel(
            "A communication systems laboratory for waveforms, channels, and network evolution."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setObjectName("subtitle")
        self.status = QLabel("Booting signal engine…")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setObjectName("metric")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setStyleSheet(
            "QProgressBar { border: 1px solid #334155; border-radius: 8px; text-align: center; height: 22px; }"
            "QProgressBar::chunk { background: #0284c7; border-radius: 7px; }"
        )
        feature_card = QGroupBox("Laboratory modules loading")
        feature_layout = QVBoxLayout(feature_card)
        for text in (
            "• Waveform sources, modulation, and optional channel noise",
            "• Oscilloscope, FFT, spectrum, digital, and filter analysis",
            "• Synchronization, TDM/FDM, and 0G–6G reference material",
        ):
            feature_layout.addWidget(QLabel(text))
        self.enter_button = QPushButton("Enter Simulator")
        self.enter_button.setEnabled(False)
        self.enter_button.clicked.connect(self.ready.emit)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(170, 70, 170, 70)
        layout.addStretch()
        layout.addWidget(self.title)
        layout.addWidget(subtitle)
        layout.addSpacing(25)
        layout.addWidget(feature_card)
        layout.addSpacing(20)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addSpacing(18)
        layout.addWidget(self.enter_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance_loading)
        self.timer.start(35)

    def _advance_loading(self) -> None:
        """Animate a short, deterministic startup sequence without blocking the UI."""
        self.progress_value = min(100, self.progress_value + 3)
        self.progress.setValue(self.progress_value)
        if self.progress_value < 35:
            self.status.setText("Booting signal engine…")
        elif self.progress_value < 70:
            self.status.setText("Calibrating digital-analysis instruments…")
        elif self.progress_value < 100:
            self.status.setText("Loading Nebula X workspace…")
        else:
            self.timer.stop()
            self.status.setText("Nebula X is ready. Select Enter Simulator to begin.")
            self.enter_button.setEnabled(True)


class ParameterCard(QGroupBox):
    """Reusable, labelled card for carrier, message, and noise inputs."""

    def __init__(self, title: str, defaults: WaveParameters) -> None:
        super().__init__(title)
        self.frequency = self._spin(defaults.frequency, 0.001, 1_000_000_000, " Hz")
        self.amplitude = self._spin(defaults.amplitude, 0.001, 100_000, " V")
        self.rate = self._spin(defaults.rate, 0.001, 2_000_000_000, " Hz")
        self.waveform = QComboBox()
        self.waveform.addItems([
            "Sine",
            "Cosine",
            "Triangle",
            "Rectangular",
            "Sawtooth",
            "Straight Line",
            "Digital Data",
        ])
        self.waveform.setCurrentText(defaults.waveform)
        form = QFormLayout(self)
        form.addRow("Waveform graph", self.waveform)
        form.addRow("Frequency", self.frequency)
        form.addRow("Voltage (Amplitude)", self.amplitude)
        form.addRow("Rate", self.rate)

    @staticmethod
    def _spin(value: float, minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setValue(value)
        spin.setSingleStep(max(value / 10, 0.1))
        spin.setSuffix(suffix)
        # Explicitly retain visible spin arrows and keyboard acceleration.
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setAccelerated(True)
        return spin

    def values(self) -> WaveParameters:
        return WaveParameters(
            self.frequency.value(),
            self.amplitude.value(),
            self.rate.value(),
            self.waveform.currentText(),
        )


class AnalogSettingsWidget(QGroupBox):
    """Visible settings for the requested analogue-labelled modulation methods."""

    def __init__(self) -> None:
        super().__init__("Analog Modulation Specific Settings")
        self.am_index = self._spin(0.70, 0.01, 1.00, 0.01, "")
        self.fm_deviation = self._spin(2_000, 1, 1_000_000, 100, " Hz")
        self.pm_deviation = self._spin(1.50, 0.01, 100, 0.05, " rad")
        self.pwm_depth = self._spin(0.35, 0.01, 0.45, 0.01, "")
        self.qam_order = QComboBox()
        self.qam_order.addItems(["16", "64", "256", "1024"])
        self.qam_order.setToolTip("QM(QAM) constellation order. Higher orders carry more bits but need a cleaner channel.")
        form = QFormLayout(self)
        form.addRow("AM modulation index (μ)", self.am_index)
        form.addRow("FM frequency deviation (Δf)", self.fm_deviation)
        form.addRow("PM phase deviation (kₚ)", self.pm_deviation)
        form.addRow("GM(PWM) duty depth", self.pwm_depth)
        form.addRow("QM(QAM) order", self.qam_order)
        note = QLabel(
            "These controls only affect AM, FM, PM, GM(PWM), and QM(QAM). "
            "Digital ASK/FSK/PSK/QPSK/GMSK retain their standard educational settings."
        )
        note.setWordWrap(True)
        note.setObjectName("subtitle")
        form.addRow(note)

    @staticmethod
    def _spin(value: float, minimum: float, maximum: float, step: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setSuffix(suffix)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setAccelerated(True)
        return spin

    def values(self) -> AnalogModulationSettings:
        return AnalogModulationSettings(
            am_index=self.am_index.value(),
            fm_deviation_hz=self.fm_deviation.value(),
            pm_deviation_rad=self.pm_deviation.value(),
            pwm_depth=self.pwm_depth.value(),
            qam_order=int(self.qam_order.currentText()),
        )


class ChannelModulationWidget(QGroupBox):
    """Checkboxes and parameters for optional channel-level signal impairments."""

    EFFECTS = (
        "Attenuation",
        "Delay",
        "Interference",
        "Shift (Doppler Shift)",
        "Fading (Multipath Fading)",
    )

    def __init__(self) -> None:
        super().__init__("Channel Modulation")
        self.checks: dict[str, QCheckBox] = {}
        self.attenuation = self._spin(6, 0, 100, 0.5, " dB")
        self.delay = self._spin(0.5, 0, 5_000, 0.05, " ms")
        self.interference_amplitude = self._spin(0.15, 0, 100_000, 0.01, " V")
        self.interference_frequency = self._spin(3_000, 0.1, 1_000_000_000, 10, " Hz")
        self.doppler = self._spin(80, -1_000_000, 1_000_000, 1, " Hz")
        self.multipath_delay = self._spin(0.25, 0.001, 5_000, 0.05, " ms")
        self.multipath_gain = self._spin(0.35, 0.01, 1.0, 0.01, "")
        grid = QGridLayout(self)
        rows: list[tuple[str, QWidget, str]] = [
            ("Attenuation", self.attenuation, "Reduce received amplitude by chosen dB."),
            ("Delay", self.delay, "Shift the received waveform in time."),
            ("Interference", self.interference_amplitude, "Add a sinusoidal interferer; frequency is adjustable beside it."),
            ("Shift (Doppler Shift)", self.doppler, "Frequency translation due to relative motion."),
            ("Fading (Multipath Fading)", self.multipath_gain, "Add a delayed reflected path; delay is adjustable beside it."),
        ]
        for row, (effect, primary_control, help_text) in enumerate(rows):
            check = QCheckBox(effect)
            check.setToolTip(help_text)
            self.checks[effect] = check
            grid.addWidget(check, row, 0)
            grid.addWidget(primary_control, row, 1)
            if effect == "Interference":
                grid.addWidget(self.interference_frequency, row, 2)
            elif effect == "Fading (Multipath Fading)":
                grid.addWidget(self.multipath_delay, row, 2)
            else:
                grid.addWidget(QLabel(help_text), row, 2)
        buttons = QHBoxLayout()
        select_all = QPushButton("Select All Effects")
        select_all.setProperty("secondary", True)
        select_all.clicked.connect(lambda: self._set_all(True))
        select_none = QPushButton("Clear Effects")
        select_none.setProperty("secondary", True)
        select_none.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(select_all)
        buttons.addWidget(select_none)
        buttons.addStretch()
        grid.addLayout(buttons, len(rows), 0, 1, 3)

    @staticmethod
    def _spin(value: float, minimum: float, maximum: float, step: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        spin.setSuffix(suffix)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        spin.setAccelerated(True)
        return spin

    def _set_all(self, checked: bool) -> None:
        for checkbox in self.checks.values():
            checkbox.setChecked(checked)

    def values(self) -> ChannelModulationSettings:
        return ChannelModulationSettings(
            enabled={name for name, checkbox in self.checks.items() if checkbox.isChecked()},
            attenuation_db=self.attenuation.value(),
            delay_ms=self.delay.value(),
            interference_amplitude=self.interference_amplitude.value(),
            interference_frequency_hz=self.interference_frequency.value(),
            doppler_shift_hz=self.doppler.value(),
            multipath_delay_ms=self.multipath_delay.value(),
            multipath_gain=self.multipath_gain.value(),
        )


class SetupPage(QWidget):
    """Landing page that collects signal source details."""

    configured = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.carrier_card = ParameterCard("Carrier Wave Details", WaveParameters(10_000, 2, 100_000))
        self.message_card = ParameterCard("Message Wave Details", WaveParameters(1_000, 1, 100_000))
        self.noise_card = ParameterCard("Noise Wave Details", WaveParameters(2_000, 0.15, 100_000))
        self.noise_card.setVisible(False)
        self.include_noise = QCheckBox("Include a Noise Wave")
        self.include_noise.toggled.connect(self.noise_card.setVisible)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0001, 10)
        self.duration_spin.setDecimals(4)
        self.duration_spin.setSingleStep(0.005)
        self.duration_spin.setValue(0.02)
        self.duration_spin.setSuffix(" seconds")
        self.duration_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.duration_spin.setAccelerated(True)

        title = QLabel(APPLICATION_NAME)
        title.setObjectName("pageTitle")
        subtitle = QLabel("Configure carrier, message, and optional noise sources for the laboratory experiment.")
        subtitle.setObjectName("subtitle")
        cards = QHBoxLayout()
        cards.addWidget(self.carrier_card)
        cards.addWidget(self.message_card)
        cards.addWidget(self.noise_card)
        continue_button = QPushButton("Continue to simulation options  →")
        continue_button.clicked.connect(self._continue)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(55, 38, 55, 38)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addLayout(cards)
        layout.addWidget(self.include_noise)
        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Simulation duration:"))
        duration_row.addWidget(self.duration_spin)
        duration_row.addStretch()
        layout.addLayout(duration_row)
        layout.addStretch()
        layout.addWidget(continue_button, alignment=Qt.AlignmentFlag.AlignRight)

    def _continue(self) -> None:
        config = SimulationConfig(
            carrier=self.carrier_card.values(),
            message=self.message_card.values(),
            noise=self.noise_card.values() if self.include_noise.isChecked() else None,
            duration=self.duration_spin.value(),
        )
        try:
            config.validate()
        except ValueError as error:
            QMessageBox.warning(self, "Input validation", str(error))
            return
        self.configured.emit(config)


class ChoicePage(QWidget):
    """Top-level choice between modulation channeling and generation education."""

    chosen = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("What do you want to simulate?")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Choose a channel modulation experiment or explore cellular generation technology.")
        subtitle.setObjectName("subtitle")
        channel_button = self._large_button("Channeling", "Design and inspect analogue or digital modulation experiments.")
        channel_button.clicked.connect(lambda: self.chosen.emit("channel"))
        generation_button = self._large_button("Generation", "Explore 0G–6G technology context with live signal analysis.")
        generation_button.clicked.connect(lambda: self.chosen.emit("generation"))
        analysis_button = self._large_button(
            "Analysis",
            "Build a custom analysis package: signal, digital, channel, filters, triggers, multiplexing, and 0G–6G reports.",
        )
        analysis_button.clicked.connect(lambda: self.chosen.emit("analysis"))
        row = QHBoxLayout()
        row.addWidget(channel_button)
        row.addWidget(generation_button)
        row.addWidget(analysis_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(80, 60, 80, 60)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addStretch()
        layout.addLayout(row)
        layout.addStretch()

    @staticmethod
    def _large_button(title: str, description: str) -> QPushButton:
        # The choice screen intentionally remains clean; Qt displays the full
        # description in a floating tooltip when the pointer hovers the button.
        button = QPushButton(title)
        button.setToolTip(description)
        button.setStatusTip(description)
        button.setMinimumSize(220, 155)
        button.setStyleSheet("font-size: 19px; text-align: center; padding: 23px;")
        return button


class ChannelSelectionPage(QWidget):
    """Checkbox groups for selecting one or many modulation methods."""

    simulation_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.checkboxes: dict[str, QCheckBox] = {}
        title = QLabel("Channeling")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Select independent modulation experiments. Every selected method receives a scope tab.")
        subtitle.setObjectName("subtitle")
        groups = QHBoxLayout()
        groups.addWidget(self._method_group("Analog Modulations", ANALOG_MODULATIONS))
        groups.addWidget(self._method_group("Digital Modulations", DIGITAL_MODULATIONS))
        self.analog_settings = AnalogSettingsWidget()
        self.channel_settings = ChannelModulationWidget()
        settings_tabs = QTabWidget()
        settings_tabs.addTab(self.analog_settings, "Analog Settings")
        settings_tabs.addTab(self.channel_settings, "Channel Modulation")
        start_button = QPushButton("START SIMULATION")
        start_button.clicked.connect(self._start)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 30, 45, 30)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(12)
        layout.addLayout(groups)
        layout.addWidget(settings_tabs)
        layout.addStretch()
        layout.addWidget(start_button, alignment=Qt.AlignmentFlag.AlignRight)

    def _method_group(self, heading: str, methods: Iterable[str]) -> QGroupBox:
        group = QGroupBox(heading)
        layout = QVBoxLayout(group)
        for method in methods:
            checkbox = QCheckBox(method)
            self.checkboxes[method] = checkbox
            layout.addWidget(checkbox)
        button_row = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_all.setProperty("secondary", True)
        select_all.clicked.connect(lambda: self._set_group(methods, True))
        clear_all = QPushButton("Clear All")
        clear_all.setProperty("secondary", True)
        clear_all.clicked.connect(lambda: self._set_group(methods, False))
        button_row.addWidget(select_all)
        button_row.addWidget(clear_all)
        layout.addLayout(button_row)
        return group

    def _set_group(self, methods: Iterable[str], checked: bool) -> None:
        for method in methods:
            self.checkboxes[method].setChecked(checked)

    def _start(self) -> None:
        selected = [method for method, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        if not selected:
            QMessageBox.information(self, "Select a modulation", "Choose at least one modulation method.")
            return
        self.simulation_requested.emit(
            SimulationRunRequest(
                methods=selected,
                analog_settings=self.analog_settings.values(),
                channel_settings=self.channel_settings.values(),
            )
        )


class CommunicationAnalysisPage(QWidget):
    """Builds a detailed, user-selected communication analysis request.

    This page does not silently select tools.  Its checkboxes make it clear
    exactly which calculations will appear in the final workspace.
    """

    analysis_requested = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.method_checks: dict[str, QCheckBox] = {}
        self.subanalysis_checks: dict[str, dict[str, QCheckBox]] = {}
        self.generation_checks: dict[str, QCheckBox] = {}
        self.config_summary = QLabel("Return to the first screen to configure carrier, message, and noise sources.")
        self.config_summary.setObjectName("subtitle")
        self._build_ui()

    def _build_ui(self) -> None:
        title = QLabel("Communication Analysis")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Choose exactly what to analyze from your entered data. All selections are optional except at least one analysis group."
        )
        subtitle.setObjectName("subtitle")

        modulation_group = QGroupBox("1. Modulation experiments")
        modulation_layout = QGridLayout(modulation_group)
        for position, method in enumerate((*ANALOG_MODULATIONS, *DIGITAL_MODULATIONS)):
            checkbox = QCheckBox(method)
            checkbox.setChecked(method in {"AM", "FM", "QPSK", "QM(QAM)"})
            self.method_checks[method] = checkbox
            modulation_layout.addWidget(checkbox, position // 5, position % 5)
        modulation_layout.addLayout(
            self._selection_buttons(lambda: self._set_checks(self.method_checks, True), lambda: self._set_checks(self.method_checks, False)),
            3,
            0,
            1,
            5,
        )

        analysis_group = QGroupBox("2. Choose individual analysis tools")
        analysis_layout = QGridLayout(analysis_group)
        for position, (group_name, subtools) in enumerate(SUB_ANALYSIS_OPTIONS.items()):
            tool_card = self._subanalysis_card(group_name, subtools)
            analysis_layout.addWidget(tool_card, position // 2, position % 2)
        analysis_layout.addLayout(
            self._selection_buttons(self._select_all_subanalyses, self._clear_all_subanalyses),
            2,
            0,
            1,
            2,
        )

        extras_group = QGroupBox("3. Synchronization and multiplexing")
        extras_layout = QFormLayout(extras_group)
        self.trigger_combo = QComboBox()
        self.trigger_combo.addItems(["None", "Triangle", "Square", "Sinusoidal"])
        self.multiplex_combo = QComboBox()
        self.multiplex_combo.addItems(["None", "TDM", "FDM", "WDM", "CDM", "OFDM", "All"])
        extras_layout.addRow("Trigger waveform", self.trigger_combo)
        extras_layout.addRow("Multiplexing", self.multiplex_combo)

        generation_group = QGroupBox("4. 0G–6G reference reports")
        generation_layout = QGridLayout(generation_group)
        for position, generation in enumerate(GENERATION_INFO.keys()):
            checkbox = QCheckBox(generation)
            checkbox.setChecked(True)
            self.generation_checks[generation] = checkbox
            generation_layout.addWidget(checkbox, position // 4, position % 4)
        generation_layout.addLayout(
            self._selection_buttons(lambda: self._set_checks(self.generation_checks, True), lambda: self._set_checks(self.generation_checks, False)),
            2,
            0,
            1,
            4,
        )

        run_button = QPushButton("RUN SELECTED COMMUNICATION ANALYSIS")
        run_button.clicked.connect(self._request_analysis)
        grid = QGridLayout()
        grid.addWidget(modulation_group, 0, 0)
        grid.addWidget(analysis_group, 0, 1)
        grid.addWidget(extras_group, 1, 0)
        grid.addWidget(generation_group, 1, 1)
        self.analog_settings = AnalogSettingsWidget()
        self.channel_settings = ChannelModulationWidget()
        settings_tabs = QTabWidget()
        settings_tabs.addTab(self.analog_settings, "Analog Settings")
        settings_tabs.addTab(self.channel_settings, "Channel Modulation")
        # The selector can grow on smaller laptop displays, so it lives in a
        # scrollable canvas instead of clipping any settings or buttons.
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 26, 40, 26)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.config_summary)
        layout.addLayout(grid)
        layout.addWidget(settings_tabs)
        layout.addStretch()
        layout.addWidget(run_button, alignment=Qt.AlignmentFlag.AlignRight)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    @staticmethod
    def _selection_buttons(select_all: Callable[[], None], clear_all: Callable[[], None]) -> QHBoxLayout:
        row = QHBoxLayout()
        all_button = QPushButton("Select All")
        all_button.setProperty("secondary", True)
        all_button.clicked.connect(select_all)
        none_button = QPushButton("Select None")
        none_button.setProperty("secondary", True)
        none_button.clicked.connect(clear_all)
        row.addWidget(all_button)
        row.addWidget(none_button)
        row.addStretch()
        return row

    @staticmethod
    def _set_checks(checks: dict[str, QCheckBox], checked: bool) -> None:
        for checkbox in checks.values():
            checkbox.setChecked(checked)

    def _subanalysis_card(self, group_name: str, subtools: tuple[str, ...]) -> QGroupBox:
        """Create one category card containing independently selectable tools."""
        card = QGroupBox(group_name)
        layout = QVBoxLayout(card)
        group_checks: dict[str, QCheckBox] = {}
        for tool in subtools:
            checkbox = QCheckBox(tool)
            checkbox.setChecked(True)
            group_checks[tool] = checkbox
            layout.addWidget(checkbox)
        layout.addLayout(
            self._selection_buttons(
                lambda checks=group_checks: self._set_checks(checks, True),
                lambda checks=group_checks: self._set_checks(checks, False),
            )
        )
        self.subanalysis_checks[group_name] = group_checks
        return card

    def _select_all_subanalyses(self) -> None:
        for checks in self.subanalysis_checks.values():
            self._set_checks(checks, True)

    def _clear_all_subanalyses(self) -> None:
        for checks in self.subanalysis_checks.values():
            self._set_checks(checks, False)

    def set_config(self, config: SimulationConfig) -> None:
        """Show the entered values so the analysis input is never ambiguous."""
        noise_text = "Noise disabled" if config.noise is None else f"Noise: {config.noise.amplitude:g} V at {config.noise.rate:g} Hz"
        self.config_summary.setText(
            f"Entered data — Carrier: {config.carrier.frequency:g} Hz, {config.carrier.amplitude:g} V | "
            f"Message: {config.message.frequency:g} Hz, {config.message.amplitude:g} V | {noise_text}"
        )

    def _request_analysis(self) -> None:
        methods = [name for name, checkbox in self.method_checks.items() if checkbox.isChecked()]
        selected_subanalyses = {
            group_name: [tool for tool, checkbox in checks.items() if checkbox.isChecked()]
            for group_name, checks in self.subanalysis_checks.items()
        }
        generations = [name for name, checkbox in self.generation_checks.items() if checkbox.isChecked()]
        if not any(selected_subanalyses.values()):
            QMessageBox.information(
                self,
                "Choose an analysis",
                "Select at least one sub-analysis tool or use Select All.",
            )
            return
        if not methods:
            QMessageBox.information(
                self,
                "No modulation selected",
                "Select at least one modulation experiment. It supplies the waveform that analysis tools inspect.",
            )
            return
        self.analysis_requested.emit(
            AnalysisRequest(
                modulation_methods=methods,
                selected_subanalyses=selected_subanalyses,
                multiplex_mode=self.multiplex_combo.currentText(),
                trigger_shape=self.trigger_combo.currentText(),
                generations=generations,
                analog_settings=self.analog_settings.values(),
                channel_settings=self.channel_settings.values(),
            )
        )


# A compact, carefully sourced generation overview is embedded to keep this one file self-contained.
GENERATION_INFO = {
    "0G": {"timeline": "1946–1980", "technology": "Pre-cellular mobile radio", "carrier": "150–450 MHz", "bandwidth": "25–50 kHz", "modulation": "FM", "spectrum": "Dedicated analogue channels", "advantages": "Simple wide-area voice", "limitations": "No security and low capacity", "applications": "Dispatch and car phones", "architecture": "High-power base station", "osi": "Physical / data-link", "devices": "Vehicle transceivers", "method": "FM"},
    "1G": {"timeline": "1981–1991", "technology": "AMPS, NMT, TACS", "carrier": "800–900 MHz", "bandwidth": "30 kHz", "modulation": "Analogue FM", "spectrum": "FDMA", "advantages": "First cellular mobility", "limitations": "Poor security and capacity", "applications": "Mobile voice", "architecture": "Analogue cellular", "osi": "Physical / data-link", "devices": "Early handsets", "method": "FM"},
    "2G": {"timeline": "1991–2001", "technology": "GSM, IS-95", "carrier": "900/1800/1900 MHz", "bandwidth": "200 kHz GSM", "modulation": "GMSK, QPSK", "spectrum": "TDMA/CDMA", "advantages": "Digital voice, SMS, encryption", "limitations": "Limited packet data", "applications": "Voice, SMS", "architecture": "BTS/BSC/core", "osi": "All layers", "devices": "Feature phones", "method": "GMSK"},
    "3G": {"timeline": "2001–2010", "technology": "UMTS, CDMA2000", "carrier": "850–2100 MHz", "bandwidth": "5 MHz", "modulation": "QPSK, 16-QAM", "spectrum": "WCDMA", "advantages": "Mobile broadband", "limitations": "High power use", "applications": "Web and video calls", "architecture": "NodeB/RNC/core", "osi": "All layers", "devices": "Smartphones", "method": "QPSK"},
    "4G": {"timeline": "2010–2020", "technology": "LTE, LTE-Advanced", "carrier": "700 MHz–2.6 GHz", "bandwidth": "1.4–20 MHz", "modulation": "QPSK–256-QAM", "spectrum": "OFDMA/SC-FDMA", "advantages": "Low latency, IP-native", "limitations": "Coverage/capacity trade-off", "applications": "HD streaming, VoLTE", "architecture": "eNodeB/EPC", "osi": "All layers, IP core", "devices": "4G smartphones, IoT", "method": "QM(QAM)"},
    "5G": {"timeline": "2019–present", "technology": "5G NR", "carrier": "Sub-6 GHz and mmWave", "bandwidth": "Up to 100 MHz / 400 MHz", "modulation": "QPSK–256-QAM", "spectrum": "OFDMA, massive MIMO", "advantages": "Capacity, latency, slicing", "limitations": "Dense infrastructure", "applications": "XR, automation, FWA", "architecture": "gNodeB/5GC", "osi": "All layers, service-based core", "devices": "5G phones, routers, sensors", "method": "QM(QAM)"},
    "6G": {"timeline": "Research / 2030 target", "technology": "AI-native integrated sensing", "carrier": "Sub-THz candidates", "bandwidth": "Very wide candidate channels", "modulation": "Advanced adaptive waveforms", "spectrum": "Dynamic intelligent access", "advantages": "Extreme sensing/capacity goals", "limitations": "Not standardized", "applications": "Digital twins, immersive systems", "architecture": "Distributed AI-native network", "osi": "Cross-layer optimization", "devices": "Future XR and sensing endpoints", "method": "QM(QAM)"},
}


class GenerationPage(QWidget):
    """Reference content and live-signal entry point for mobile network generations."""

    simulation_requested = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        title = QLabel("Generation Communications")
        title.setObjectName("pageTitle")
        self.generation_combo = QComboBox()
        self.generation_combo.addItems(GENERATION_INFO.keys())
        self.generation_combo.currentTextChanged.connect(self._render_information)
        analyze_button = QPushButton("Generate live analysis")
        analyze_button.clicked.connect(lambda: self.simulation_requested.emit(self.generation_combo.currentText()))
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Network generation:"))
        controls.addWidget(self.generation_combo)
        controls.addStretch()
        controls.addWidget(analyze_button)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(45, 28, 45, 28)
        layout.addWidget(title)
        layout.addWidget(QLabel("Historical and technical reference, linked to a representative live simulation."))
        layout.addLayout(controls)
        layout.addWidget(self.browser, stretch=1)
        self._render_information("0G")

    def _render_information(self, generation: str) -> None:
        details = GENERATION_INFO[generation]
        titles = {
            "timeline": "Evolution timeline", "technology": "Technology overview", "carrier": "Carrier frequency",
            "bandwidth": "Typical bandwidth", "modulation": "Common modulation methods", "spectrum": "Spectrum usage",
            "advantages": "Advantages", "limitations": "Limitations", "applications": "Applications",
            "architecture": "Architecture", "osi": "OSI layer relevance", "devices": "Typical devices",
        }
        table_rows = "".join(
            f"<tr><td style='padding:7px 14px 7px 0'><b>{label}</b></td><td>{details[key]}</td></tr>"
            for key, label in titles.items()
        )
        self.browser.setHtml(
            f"<h2>{generation} overview</h2><table>{table_rows}</table>"
            "<p>Use the carrier, message, and optional noise values configured on the first page. "
            "The button above opens a representative modulation with waveform, spectrum, bandwidth, "
            "channel-response, and constellation analysis views.</p>"
        )


# -----------------------------------------------------------------------------
# Background simulation worker
# -----------------------------------------------------------------------------

class SimulationWorker(QObject):
    """Runs numerical simulation away from the user-interface thread."""

    completed = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(
        self,
        config: SimulationConfig,
        methods: list[str],
        analog_settings: AnalogModulationSettings | None = None,
        channel_settings: ChannelModulationSettings | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.methods = methods
        self.analog_settings = analog_settings or AnalogModulationSettings()
        self.channel_settings = channel_settings or ChannelModulationSettings()

    def run(self) -> None:
        try:
            results = [
                SignalEngine.simulate(
                    self.config,
                    method,
                    self.analog_settings,
                    self.channel_settings,
                )
                for method in self.methods
            ]
        except Exception as error:  # The GUI displays a user-friendly dialog.
            self.failed.emit(str(error))
            return
        self.completed.emit(results)


# -----------------------------------------------------------------------------
# Results workspace and analysis dialogs
# -----------------------------------------------------------------------------

class ResultsWindow(QMainWindow):
    """Dockable professional-style oscilloscope workspace for selected results."""

    def __init__(self, config: SimulationConfig, results: list[SignalResult]) -> None:
        super().__init__()
        self.config = config
        self.results = results
        self.scopes: list[ScopeWidget] = []
        self.setWindowTitle(f"{APPLICATION_NAME} — Oscilloscope")
        self.resize(1320, 820)
        self.setDockNestingEnabled(True)

        self.tabs = QTabWidget()
        for result in results:
            scope = ScopeWidget(result)
            self.scopes.append(scope)
            self.tabs.addTab(scope, result.name)
        self.tabs.currentChanged.connect(self._refresh_measurement_dock)
        self.setCentralWidget(self.tabs)

        self._build_measurement_dock()
        self._build_toolbar()
        self.statusBar().showMessage("Mouse wheel zooms; left drag pans; drag scope cursor lines to measure.")
        self._refresh_measurement_dock()

    @property
    def current_result(self) -> SignalResult:
        return self.results[self.tabs.currentIndex()]

    @property
    def current_scope(self) -> ScopeWidget:
        return self.scopes[self.tabs.currentIndex()]

    def _build_measurement_dock(self) -> None:
        """Show live metrics and basic oscilloscope guidance in a movable dock."""
        self.measurement_dock = QDockWidget("Measurement Panel", self)
        self.measurement_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = QWidget()
        self.measurement_layout = QVBoxLayout(panel)
        self.measurement_layout.setContentsMargins(14, 14, 14, 14)
        self.measurement_header = QLabel()
        self.measurement_header.setObjectName("metric")
        self.measurement_text = QLabel()
        self.measurement_text.setWordWrap(True)
        self.measurement_layout.addWidget(self.measurement_header)
        self.measurement_layout.addWidget(self.measurement_text)
        self.measurement_layout.addStretch()
        self.measurement_dock.setWidget(panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.measurement_dock)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Exports and analysis", self)
        toolbar.setMovable(True)
        self.addToolBar(toolbar)
        actions = [
            ("← Back", "Return to the simulator selection screen", self.close),
            ("Export PNG", "Export current scope plot as PNG", self.export_png),
            ("Export CSV", "Export current waveform data as CSV", self.export_csv),
            ("Export JSON", "Export session configuration and measurements", self.export_json),
            ("Report PDF", "Create a concise simulation report", self.export_pdf),
            ("Constellation", "Open constellation, eye diagram, BER, and SER", self.open_constellation),
            ("Advanced analysis", "Open signal/noise/constellation/eye/waterfall views", self.open_advanced_analysis),
            ("Synchronization", "Compare signal with a trigger waveform", self.open_synchronization),
            ("Multiplexing", "Generate a TDM, FDM, WDM, CDM, or OFDM composite", self.open_multiplexing),
        ]
        for text, tip, callback in actions:
            action = QAction(text, self)
            action.setToolTip(tip)
            action.triggered.connect(callback)
            toolbar.addAction(action)

    def _refresh_measurement_dock(self) -> None:
        result = self.current_result
        frequency, magnitude = SignalEngine.spectrum(result.samples, result.sample_rate)
        peak_frequency = frequency[int(np.argmax(magnitude))]
        noise_line = "Noise disabled" if result.snr_db is None else f"SNR: {result.snr_db:.2f} dB"
        self.measurement_header.setText(f"{result.name} measurement summary")
        self.measurement_text.setText(
            f"Sample rate: {result.sample_rate:,.0f} Hz\n"
            f"Sample count: {len(result.samples):,}\n"
            f"Peak frequency: {peak_frequency:,.1f} Hz\n"
            f"Peak-to-peak voltage: {result.peak_to_peak:.3f} V\n"
            f"RMS voltage: {result.rms_voltage:.3f} V\n"
            f"{noise_line}\n\n"
            "Controls\n"
            "• Mouse wheel: zoom\n• Drag: pan\n• Cursor lines: time/voltage readings\n"
            "• Use timebase, vertical scale, trigger, color, grid, pause, and autoscale controls below the graph."
        )

    def _choose_save_path(self, caption: str, suffix: str) -> str:
        default_path = str(Path.home() / f"simulation{suffix}")
        return QFileDialog.getSaveFileName(self, caption, default_path, f"*{suffix}")[0]

    def export_png(self) -> None:
        path = self._choose_save_path("Export scope image", ".png")
        if path:
            Exporter.save_plot_png(path, self.current_scope.plot_item)
            self.statusBar().showMessage(f"Saved PNG: {path}", 5000)

    def export_csv(self) -> None:
        path = self._choose_save_path("Export waveform CSV", ".csv")
        if path:
            Exporter.save_csv(path, self.current_result)
            self.statusBar().showMessage(f"Saved CSV: {path}", 5000)

    def export_json(self) -> None:
        path = self._choose_save_path("Export session JSON", ".json")
        if path:
            Exporter.save_json(path, self.config, self.results)
            self.statusBar().showMessage(f"Saved JSON: {path}", 5000)

    def export_pdf(self) -> None:
        path = self._choose_save_path("Export PDF simulation report", ".pdf")
        if path:
            Exporter.save_pdf_report(path, self.config, self.results)
            self.statusBar().showMessage(f"Saved PDF report: {path}", 5000)

    def open_advanced_analysis(self) -> None:
        AdvancedAnalysisDialog(self.current_result, self).exec()

    def open_constellation(self) -> None:
        """Direct modulation-analysis option for constellation and eye views."""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Digital Modulation Analysis — {self.current_result.name}")
        dialog.resize(980, 700)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(
            "Constellation Diagram, Eye Diagram, BER and SER are available for digital methods. "
            "Analogue methods are displayed as an I/Q visual comparison."
        ))
        layout.addWidget(
            DigitalAnalysisPanel([self.current_result], SUB_ANALYSIS_OPTIONS["Digital Analysis"]),
            stretch=1,
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def open_synchronization(self) -> None:
        choices = ["No analysis", "Triangle", "Square", "Sinusoidal"]
        shape, accepted = QInputDialog.getItem(
            self,
            "Synchronization Analysis",
            "Trigger waveform:",
            choices,
            0,
            False,
        )
        if not accepted or shape == "No analysis":
            return
        result = self.current_result
        trigger = SignalEngine.make_trigger(result.time, shape, self.config.message.frequency)
        combined = result.clean_signal + trigger * self.config.carrier.amplitude * 0.25
        dialog = QDialog(self)
        dialog.setWindowTitle("Synchronization Analysis")
        dialog.resize(1000, 720)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Compare original signal, selected trigger, and their timing relationship."))
        tabs = QTabWidget()
        tabs.addTab(SimplePlot("Original waveform", result.time, result.clean_signal), "Original")
        tabs.addTab(SimplePlot(f"{shape} trigger waveform", result.time, trigger, colour="#f59e0b"), "Trigger")
        tabs.addTab(SimplePlot("Combined timing relationship", result.time, combined, colour="#4ade80"), "Combined")
        layout.addWidget(tabs, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def open_multiplexing(self) -> None:
        choices = ["None", "TDM", "FDM", "WDM", "CDM", "OFDM", "All"]
        choice, accepted = QInputDialog.getItem(
            self,
            "Multiplexing Analysis",
            "Technique:",
            choices,
            0,
            False,
        )
        if not accepted or choice == "None":
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Multiplexing Analysis")
        dialog.resize(1000, 690)
        layout = QVBoxLayout(dialog)
        description = QLabel(
            "TDM interleaves time slots. FDM combines frequency channels. WDM represents wavelength "
            "channels in the electrical model. CDM applies spreading codes. OFDM uses orthogonal subcarriers."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        tabs = QTabWidget()
        modes = ["TDM", "FDM", "WDM", "CDM", "OFDM"] if choice == "All" else [choice]
        for mode in modes:
            time, composite = SignalEngine.multiplex(self.results, mode)
            tabs.addTab(SimplePlot(f"{mode} composite waveform", time, composite, colour="#4ade80"), mode)
        layout.addWidget(tabs, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()


# -----------------------------------------------------------------------------
# Application controller and entry point
# -----------------------------------------------------------------------------

class CommunicationSimulator(QMainWindow):
    """Owns page navigation and starts the background simulation worker."""

    def __init__(self) -> None:
        super().__init__()
        self.config: SimulationConfig | None = None
        self.worker_thread: QThread | None = None
        self.worker: SimulationWorker | None = None
        self.results_window: ResultsWindow | None = None
        self.analysis_window: CommunicationAnalysisWindow | None = None
        self.active_analysis_request: AnalysisRequest | None = None
        self.navigation_history: list[QWidget] = []
        self.setWindowTitle(APPLICATION_NAME)
        self.resize(1150, 680)
        self.setMinimumSize(900, 560)

        self.stack = QStackedWidget()
        self.welcome_page = WelcomePage()
        self.setup_page = SetupPage()
        self.choice_page = ChoicePage()
        self.channel_page = ChannelSelectionPage()
        self.generation_page = GenerationPage()
        self.analysis_page = CommunicationAnalysisPage()
        for page in (
            self.welcome_page,
            self.setup_page,
            self.choice_page,
            self.channel_page,
            self.generation_page,
            self.analysis_page,
        ):
            self.stack.addWidget(page)
        self.setCentralWidget(self.stack)
        self.setStatusBar(QStatusBar(self))
        self._build_navigation_toolbar()
        self.statusBar().showMessage("Loading Nebula X Comms Simulator.")

        self.welcome_page.ready.connect(lambda: self._navigate(self.setup_page))
        self.setup_page.configured.connect(self._accept_configuration)
        self.choice_page.chosen.connect(self._choose_mode)
        self.channel_page.simulation_requested.connect(self._start_channel_request)
        self.generation_page.simulation_requested.connect(self._start_generation_simulation)
        self.analysis_page.analysis_requested.connect(self._start_communication_analysis)

    def _build_navigation_toolbar(self) -> None:
        """Put a persistent Back action above every page in the main window."""
        toolbar = QToolBar("Navigation", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        self.back_action = QAction("← Back", self)
        self.back_action.setToolTip("Return to the previous screen")
        self.back_action.setEnabled(False)
        self.back_action.triggered.connect(self._go_back)
        toolbar.addAction(self.back_action)
        toolbar.addSeparator()
        title = QLabel("Nebula X Comms Simulator")
        title.setObjectName("metric")
        toolbar.addWidget(title)

    def _navigate(self, target: QWidget) -> None:
        """Navigate while recording the source page for reliable Back behaviour."""
        current = self.stack.currentWidget()
        if current is target:
            return
        if current is not None:
            self.navigation_history.append(current)
        self.stack.setCurrentWidget(target)
        self.back_action.setEnabled(bool(self.navigation_history))

    def _go_back(self) -> None:
        """Return to the preceding page, including the Nebula welcome page."""
        if not self.navigation_history:
            return
        target = self.navigation_history.pop()
        self.stack.setCurrentWidget(target)
        self.back_action.setEnabled(bool(self.navigation_history))
        self.statusBar().showMessage("Returned to the previous screen.")

    def _accept_configuration(self, config: SimulationConfig) -> None:
        self.config = config
        self.analysis_page.set_config(config)
        self._navigate(self.choice_page)
        self.statusBar().showMessage("Sources configured. Select a simulation mode.")

    def _choose_mode(self, mode: str) -> None:
        target_pages = {
            "channel": self.channel_page,
            "generation": self.generation_page,
            "analysis": self.analysis_page,
        }
        self._navigate(target_pages[mode])

    def _start_generation_simulation(self, generation: str) -> None:
        representative_method = GENERATION_INFO[generation]["method"]
        self._start_simulation([representative_method])

    def _start_channel_request(self, request: SimulationRunRequest) -> None:
        """Launch standard channeling with its analogue and channel-effect settings."""
        self._start_simulation(
            request.methods,
            analog_settings=request.analog_settings,
            channel_settings=request.channel_settings,
        )

    def _start_communication_analysis(self, request: AnalysisRequest) -> None:
        """Simulate selected methods, then open the checkbox-driven analysis workspace."""
        self._start_simulation(
            request.modulation_methods,
            analysis_request=request,
            analog_settings=request.analog_settings,
            channel_settings=request.channel_settings,
        )

    def _start_simulation(
        self,
        methods: list[str],
        analysis_request: AnalysisRequest | None = None,
        analog_settings: AnalogModulationSettings | None = None,
        channel_settings: ChannelModulationSettings | None = None,
    ) -> None:
        if self.config is None:
            QMessageBox.warning(self, "Missing configuration", "Configure carrier and message values first.")
            self._navigate(self.setup_page)
            return
        if self.worker_thread is not None and self.worker_thread.isRunning():
            QMessageBox.information(self, "Simulation in progress", "Please wait for the current simulation to finish.")
            return

        self.active_analysis_request = analysis_request
        self.statusBar().showMessage("Calculating waveforms in the background…")
        self.worker_thread = QThread(self)
        self.worker = SimulationWorker(
            self.config,
            methods,
            analog_settings,
            channel_settings,
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.completed.connect(self._simulation_complete)
        self.worker.failed.connect(self._simulation_failed)
        self.worker.completed.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self._worker_finished)
        self.worker_thread.start()

    def _simulation_complete(self, results: list[SignalResult]) -> None:
        self.statusBar().showMessage("Simulation complete. Opening oscilloscope workspace.")
        if self.active_analysis_request is not None:
            self.analysis_window = CommunicationAnalysisWindow(
                self.config,
                results,
                self.active_analysis_request,
            )
            self.analysis_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.analysis_window.show()
            self.active_analysis_request = None
        else:
            self.results_window = ResultsWindow(self.config, results)
            self.results_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.results_window.show()

    def _simulation_failed(self, message: str) -> None:
        self.active_analysis_request = None
        self.statusBar().showMessage("Simulation failed.")
        QMessageBox.critical(self, "Simulation error", message)

    def _worker_finished(self) -> None:
        self.worker_thread = None
        self.worker = None


def main() -> None:
    """Create Qt application and show the landing window."""
    application = QApplication(sys.argv)
    application.setApplicationName(APPLICATION_NAME)
    application.setStyleSheet(DARK_THEME)
    pg.setConfigOptions(antialias=True)
    window = CommunicationSimulator()
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
    
