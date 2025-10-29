I'll create a short todo list to track steps (required) and then answer: confirm whether the script uses CUDA, explain what to install, and give clear commands to check and install on Windows PowerShell. I'll mark the first todo as in-progress now.

Created 4 todos

Updating the todo list: I'll mark item 1 completed and set item 2 to in-progress.

Starting (2/4) Explain what CUDA/toolkit/drivers are required

Completed (2/4) Explain what CUDA/toolkit/drivers are required

I'll confirm what the script does with CUDA, then give exact PowerShell checks and recommended install commands so you can run uav_yolo.py on GPU (or CPU) and verify it.

Does this script use CUDA?
Yes. In uav_yolo.py the script:

uses torch.cuda.is_available() to decide device:
sets device = 'cuda:0' if a CUDA-capable GPU is available, otherwise 'cpu'.
prints the device and the GPU name (if available).
calls model.to(device) and also calls model.predict(..., device=device) — both make the model run on the chosen device.
So when your system has a working NVIDIA driver + a PyTorch build with CUDA support, the script will run inference on the GPU.

What do you need to install to run with CUDA?
Short checklist:

An NVIDIA GPU (reasonably modern — for good support use GeForce GTX 10xx / RTX series or newer).
NVIDIA driver (latest or recent stable driver that supports the CUDA version you want).
A PyTorch wheel built for a CUDA version supported by your driver (common stable options in 2023–2024 were CUDA 11.7 or 11.8). You do NOT always need the full CUDA Toolkit installed — the PyTorch wheel bundles the needed CUDA runtime — but installing the toolkit (nvcc) is only required if you plan to compile CUDA code locally.
pip packages: a CUDA-enabled PyTorch, plus ultralytics and opencv-python.
If you don't have a suitable NVIDIA GPU or driver, you can still run on CPU — but it will be much slower.

PowerShell commands to check your system (run these first)
Open PowerShell and run:

Check NVIDIA driver + GPU status:

nvidia-smi
(This shows driver version, GPU model, and which processes are using the GPU. If nvidia-smi is not found, you likely need to install NVIDIA drivers.)

Quick GPU info from PowerShell (alternative):

Get-WmiObject Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID
Check Python/Torch GPU availability:

python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('torch.version.cuda=', torch.version.cuda); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
(Optional) Check nvcc (CUDA toolkit) version if you installed the toolkit:

nvcc --version
If nvcc is not found, you probably don't have the toolkit on PATH — that's okay for usual PyTorch + ultralytics usage.

How to install the right PyTorch (examples)
Go to https://pytorch.org if you want to generate the exact command for your Python version. Below are typical pip examples (PowerShell) for common CUDA targets. Replace the CUDA version candidate with the one supported by your driver.

For CUDA 11.7 (example):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
For CUDA 11.8 (example):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
If you want CPU-only (no GPU):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
After PyTorch is installed, install ultralytics + opencv:


pip install ultralytics opencv-python
Notes:

The exact --index-url path depends on PyTorch releases; use the command generator on pytorch.org for your Python/OS to be sure.
If you run into errors installing CUDA builds via pip, you can install the CPU wheel as a fallback or use conda where PyTorch+CUDA packages are often easier to match.
How to run uav_yolo.py and verify it uses GPU
From the repository folder in PowerShell:


python uav_yolo.py
The script prints the device and GPU name at startup:

If GPU is used you should see something like: Device: CUDA:0 | GPU: NVIDIA GeForce ...
If not, it will print Device: CPU | GPU: N/A and run on CPU.
You can also confirm at runtime by watching nvidia-smi while the script runs — it should show a Python process using GPU memory when inference is happening.

Quick troubleshooting (if torch.cuda.is_available() is False)
Run nvidia-smi and verify the driver is installed and GPU is visible.
Make sure the NVIDIA driver version supports the CUDA runtime version your PyTorch wheel expects. Updating the driver from NVIDIA's website often fixes mismatches.
Reinstall PyTorch with the wheel that matches your driver/CUDA support.
Reboot after driver or CUDA toolkit installs.
Ensure you installed Python packages into the same Python interpreter you run uav_yolo.py with (check python -V and pip show torch).
Small note about the script
uav_yolo.py already does the correct device selection and calls model.to(device) and model.predict(..., device=device). No code changes are required to use CUDA other than having a correctly installed GPU stack.

If you want, I can:

run the quick Python check command here in PowerShell for you (I can show exactly what to expect),
or update the script to print torch.version.cuda and a clearer startup message. Which would you like next?