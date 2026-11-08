# HNN:KAA:MRE
## System Requirements, Installation & Setup Guide

This document explains the complete setup process for running HNN:KAA:MRE, including Python installation, required libraries, Ollama installation, AI model configuration, hardware requirements, troubleshooting, and launching the application.

---

# 1. Python Installation

HNN:KAA:MRE is developed using Python.

### 1.1. Minimum Requirements

| Requirement | Specification |
|---|---|
| Operating System | Windows 10 / Windows 11 (64-bit) |
| Python | Python 3.10 or higher |
| Recommended Python | Python 3.11 or 3.12 |
| Architecture | 64-bit |
| Package Manager | pip |
| Environment | Virtual environment recommended |

### 1.2. Download Python

1. Visit https://www.python.org/downloads/
2. Download Python 3.11 or 3.12 for Windows.
3. Run the installer.
4. Make sure to enable **Add Python to PATH**.
5. Select Install Now.
6. Complete the installation.

### 1.3. Verify Python Installation

Open Command Prompt or PowerShell and execute:

```bash
python --version
```

Then verify pip:

```bash
pip --version
```

Both commands should return their respective installed versions.

### 1.4. Create a Virtual Environment

Navigate to the HNN:KAA:MRE project directory:

```bash
cd path\to\HNN-KAA-MRE
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

A virtual environment keeps project dependencies separate from other Python installations.

### 1.5. Install Required Python Libraries

The application uses Qt for its graphical interface and PDF-related functionality.

Install PySide6:

```bash
pip install PySide6
```

Install PDF support:

```bash
pip install pypdf
```

If the application is configured to use PyQt6 instead of PySide6, install:

```bash
pip install PyQt6
```

**Important:** The inspected application code attempts to use PySide6 first and provides a PyQt6 fallback. Install the binding that matches the code path you intend to use.

For PDF compatibility, the application also supports PyPDF2 as a fallback:

```bash
pip install PyPDF2
```

### 1.6. Verify Installed Libraries

Execute:

```bash
python -c "import PySide6; print('PySide6 installed successfully')"
```

For PDF support:

```bash
python -c "import pypdf; print('pypdf installed successfully')"
```

If using PyQt6:

```bash
python -c "import PyQt6; print('PyQt6 installed successfully')"
```

### 1.7. Save the Installed Dependencies

To generate a list of the installed Python packages:

```bash
pip freeze > requirements.txt
```

To reinstall dependencies later:

```bash
pip install -r requirements.txt
```

---

# 2. Ollama Installation

HNN:KAA:MRE uses Ollama to run locally hosted language models.

Ollama provides a local interface through which the application can communicate with an AI model without relying on a cloud-based AI API.

### 2.1. Download Ollama

1. Visit https://ollama.com/download
2. Select Windows.
3. Download the installer.
4. Run the installer.
5. Complete the installation.

### 2.2. Verify Ollama Installation

Open Command Prompt or PowerShell:

```bash
ollama --version
```

If Ollama is installed correctly, the command will display its installed version.

### 2.3. Understand How Ollama Works

Ollama runs a local service that manages and serves downloaded language models.

The general process is:

1. Install Ollama.
2. Download a language model.
3. Ollama stores the model locally.
4. Ollama starts its local service.
5. HNN:KAA:MRE communicates with that service.
6. The selected model processes supported language-generation requests.

The default local API endpoint is:

```text
http://localhost:11434
```

The application and Ollama must be able to communicate through this local endpoint.

### 2.4. Download the Default AI Model

The inspected application configuration uses:

```text
gemma2:2b
```

Download it using:

```bash
ollama pull gemma2:2b
```

Wait for the download to complete.

### 2.5. Verify the Downloaded Model

Execute:

```bash
ollama list
```

The installed model should appear in the model list.

To run the model manually:

```bash
ollama run gemma2:2b
```

You can now enter a prompt and verify that the model responds.

To exit the interactive session:

```text
/bye
```

### 2.6. Verify the Local API

Open PowerShell and execute:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If Ollama is running correctly, the command should return information about locally installed models.

### 2.7. Downloading Other Models

You can download other compatible models using:

```bash
ollama pull MODEL_NAME
```

For example:

```bash
ollama pull gemma2:9b
```

To check available local models:

```bash
ollama list
```

To remove a model:

```bash
ollama rm MODEL_NAME
```

**Important:** If you change the model, make sure the model name in the HNN:KAA:MRE configuration matches the model installed in Ollama.

---

# 3. AI Model Parameters and Hardware Requirements

The performance of locally hosted language models depends on several factors, including parameter count, quantization, context length, available RAM, and GPU memory.

A larger model generally requires more computational resources.

### 3.1. Understanding Model Parameters

Model parameters are the numerical values learned during the training of a neural network.

They influence the model's capacity to represent and process information.

For example:

- 2B = approximately 2 billion parameters.
- 7B = approximately 7 billion parameters.
- 14B = approximately 14 billion parameters.
- 70B = approximately 70 billion parameters.

A higher parameter count does not automatically guarantee better performance for every task.

### 3.2. Approximate GPU Memory Requirements

The following table provides general planning estimates for quantized models.

| Model Parameters | Approximate Model Weight Size | Suggested GPU VRAM | Recommendation |
|---|---:|---:|---|
| 1B–2B | 0.6–1.5 GB | 2–4 GB | Entry-level systems |
| 3B–4B | 1.5–3 GB | 4–6 GB | Basic AI applications |
| 7B–8B | 3.5–5.5 GB | 6–8 GB | Recommended general-purpose range |
| 9B–14B | 5–9 GB | 8–12 GB | Higher-quality local inference |
| 20B–24B | 11–15 GB | 16–20 GB | High-performance systems |
| 30B–34B | 17–22 GB | 24 GB or higher | Advanced local inference |
| 70B–72B | 38–45 GB | 48 GB or multi-GPU | High-end systems |

**Note:** These figures are estimates, not guaranteed minimum requirements. Actual memory consumption depends on model quantization, context length, KV cache, runtime overhead, and GPU offloading.

A model may run with less VRAM by using system RAM, although performance can decrease considerably.

### 3.3. Recommended Hardware

| Component | Minimum Recommendation | Preferred Recommendation |
|---|---|---|
| Processor | Modern 4-core CPU | Modern 6-core or 8-core CPU |
| RAM | 8 GB | 16 GB or higher |
| GPU | Integrated graphics or compatible GPU | NVIDIA GPU with 6–8 GB VRAM or more |
| Storage | 10 GB free space | 20 GB or more |
| Operating System | Windows 10 64-bit | Windows 11 64-bit |
| Internet | Required for initial downloads | Required for updates and downloads |

These are general recommendations. Actual requirements depend on the selected model and workload.

### 3.4. GPU vs CPU Execution

Ollama can run models using CPU resources, supported GPU acceleration, or a combination of both.

**CPU execution:**
- Does not require a dedicated GPU.
- Uses system RAM.
- Can be considerably slower for larger models.

**GPU execution:**
- Uses compatible GPU resources.
- Can significantly improve inference speed.
- Requires sufficient available VRAM.

**Hybrid execution:**
- Uses both GPU VRAM and system RAM.
- Allows some models to run even when the GPU cannot hold the entire model.
- Performance depends on the amount of data transferred between memory types.

---

# 4. Additional System Requirements

Apart from Python and Ollama, the following are recommended.

### 4.1. Operating System

Supported environment:

- Windows 10 64-bit
- Windows 11 64-bit

Other operating systems may require adjustments to installation and execution commands.

### 4.2. Storage

Maintain sufficient free storage for:

- Python and its dependencies.
- Ollama installation.
- Downloaded AI models.
- Project files.
- Temporary processing data.
- Generated reports and documents.

AI models can occupy several gigabytes, depending on their size and quantization.

### 4.3. Internet Connection

An internet connection is required for:

- Downloading Python.
- Installing Python libraries.
- Downloading Ollama.
- Pulling AI models.
- Updating dependencies.

Once the required components and models are installed, the local Ollama model can operate without an active internet connection, provided the application itself does not require online services.

### 4.4. Administrative Permissions

Some installation operations may require Windows administrator permissions.

---

# 5. Running HNN:KAA:MRE

Once Python, the required libraries, and Ollama are installed, follow these steps.

### Step 1: Open the Project Directory

Open Command Prompt or PowerShell and navigate to the project folder:

```bash
cd path\to\HNN-KAA-MRE
```

### Step 2: Activate the Virtual Environment

```bash
.venv\Scripts\activate
```

### Step 3: Ensure Ollama Is Running

Check the Ollama service:

```bash
ollama list
```

If the command works and your model is listed, Ollama is available.

If necessary, start the Ollama application or service.

### Step 4: Verify the AI Model

```bash
ollama run gemma2:2b
```

Exit the interactive session after verifying that the model responds.

### Step 5: Launch the Application

Execute the application's main Python entry point.

For example, if the main file is `app.py`:

```bash
python app.py
```

If the project's entry point has a different filename, use that filename instead.

---

# 6. Updating the Project

### Update Python Packages

```bash
python -m pip install --upgrade pip
```

To update a specific package:

```bash
pip install --upgrade PACKAGE_NAME
```

### Update Ollama

Download and install the latest version from:

https://ollama.com/download

### Update an AI Model

Pull the model again:

```bash
ollama pull gemma2:2b
```

Ollama will check for and download available model updates.

---

# 7. Troubleshooting

### Problem 1: Python Is Not Recognized

**Error:**

```text
'python' is not recognized as an internal or external command
```

**Solution:**

- Reinstall Python.
- Enable Add Python to PATH.
- Restart Command Prompt or PowerShell.
- Verify using:

```bash
python --version
```

### Problem 2: pip Is Not Recognized

**Solution:**

Use:

```bash
python -m pip install --upgrade pip
```

Instead of relying on the standalone `pip` command.

### Problem 3: PySide6 Import Error

**Error:**

```text
ModuleNotFoundError: No module named 'PySide6'
```

**Solution:**

```bash
pip install PySide6
```

Verify the active Python environment before installing.

### Problem 4: Ollama Is Not Recognized

**Error:**

```text
'ollama' is not recognized as an internal or external command
```

**Solution:**

- Verify that Ollama is installed.
- Restart the terminal.
- Check whether Ollama is available through PATH.
- Reinstall Ollama if necessary.

### Problem 5: Ollama Connection Error

**Error:**

```text
Connection refused
```

**Solution:**

- Open the Ollama application.
- Ensure its local service is running.
- Check the endpoint:

```text
http://localhost:11434
```

- Verify the service using:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

### Problem 6: Model Not Found

**Error:**

```text
model not found
```

**Solution:**

Download the required model:

```bash
ollama pull gemma2:2b
```

Then verify:

```bash
ollama list
```

Check that the model name in the application configuration matches the installed model.

### Problem 7: Insufficient GPU Memory

**Possible symptoms:**

- Slow inference.
- GPU memory errors.
- Model loading failures.
- Excessive system RAM consumption.

**Solution:**

- Use a smaller model.
- Select a more heavily quantized model where compatible.
- Reduce the model context length if configurable.
- Close unnecessary GPU-intensive applications.
- Allow CPU and system RAM offloading where supported.

---

# 8. Important Notes

1. HNN:KAA:MRE uses Python for its application logic and graphical interface.
2. Ollama provides the local language-model execution environment.
3. The default model identified in the inspected application configuration is `gemma2:2b`.
4. Changing the installed Ollama model may require updating the application's model configuration.
5. GPU memory requirements vary depending on model architecture, quantization, and context length.
6. A dedicated GPU is not strictly mandatory for CPU-compatible models.
7. Installing Python libraries does not automatically install Ollama or download an AI model.
8. Keep the Python environment and project dependencies consistent when moving the project between computers.

---

# 9. Official Resources

| Resource | Official Website |
|---|---|
| Python | https://www.python.org/ |
| Python Downloads | https://www.python.org/downloads/ |
| pip Documentation | https://pip.pypa.io/en/stable/ |
| PySide6 Documentation | https://doc.qt.io/qtforpython-6/ |
| PyPI | https://pypi.org/ |
| Ollama | https://ollama.com/ |
| Ollama Downloads | https://ollama.com/download |
| Ollama Documentation | https://docs.ollama.com/ |
| Ollama Model Library | https://ollama.com/library |
| Gemma 2 Model Library | https://ollama.com/library/gemma2 |

---

**HNN:KAA:MRE**

*Hyper Neural Network | Keyword Assorted Algorithm | Mathematical Resolution Engine*

Copyright © 2026 Harsh Sharma.
