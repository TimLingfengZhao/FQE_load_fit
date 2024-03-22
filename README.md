# FQE_load_fit
1. use pip import d3rlpy, scope_rl
2. mujoco: 2.3.3 mujoco-py:2.1.2.14  https://blog.guptanitish.com/guide-to-install-openais-mujoco-on-ubuntu-linux-1ac22a9678b4
3. mujoco helpful (debug) website: https://github.com/openai/mujoco-py/issues/410
4. https://www.reddit.com/r/Ubuntu/comments/rmz3mn/why_my_export_path_doesnt_work_mujoco_gcc_error/?rdt=40047
5. import typing, pandas, time, pickle, numpy, torch
6. install cuda : https://www.cherryservers.com/blog/install-cuda-ubuntu (self define the driver version) install cuDNN with following command :
conda install -c nvidia cuda-nvcc
conda install -c "nvidia/label/cuda-11.3.0" cuda-nvcc    #self define cuda version based on the detailed information given by:  nvidia-smi
7. run the script "FQE_load_two.py" directly.
