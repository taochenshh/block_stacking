# manip_project

## Install Mujoco
1. Download mjpro150 linux here: https://www.roboti.us/index.html
    and apply for a license.
1. Unzip the downloaded mjpro150 directory into ~/.mujoco/mjpro150, and place your license key(mjkey.txt) at ~/.mujoco/mjkey.txt
1. add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin
` to ~/.bashrc
1. add `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so` to ~/.bashrc
1. `sudo apt-get install libosmesa6-dev` (if you are on local machine)
1. `cd mujoco`
1. `pip install -r requirements.txt`
1. `python setup.py install`
1. `python examples/body_interaction.py`

## Note
If you wanna use `viewer.render`, you need to set `LD_PRELOAD` to `/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-384/libGL.so`.
However, if you wanna use `sim.render`, you need to unset it (`unset LD_PRELOAD`).