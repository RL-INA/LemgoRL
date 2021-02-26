1. Install Microsoft Visual C++ 14.0 from https://aka.ms/vs/16/release/vc_redist.x64.exe
2. Install Anaconda / Miniconda
3. In anaconda prompt, execute the command "conda env create -f environment.yml".
4. While executing the code, if you face error "ModuleNotFoundError: No module named 'dataclasses'", then execute the command `pip install ray[tune]`.