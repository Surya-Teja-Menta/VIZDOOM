# ***ViZDoom*** [![PyPI version](https://badge.fury.io/py/vizdoom.svg)](https://badge.fury.io/py/vizdoom) 

ViZDoom allows developing AI **bots that play Doom using only the visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://github.com/rheit/zdoom) to provide the game mechanics.

![vizdoom_deadly_corridor](http://www.cs.put.poznan.pl/mkempka/misc/vizdoom_gifs/vizdoom_corridor_segmentation.gif)


## **Features**
- Multi-platform (Linux, macOS, Windows),
- API for Python, C++, and Julia (thanks to [Jun Tian](https://github.com/findmyway)), and also Lua and Java for older versions,
- Easy-to-create custom scenarios (visual editors, scripting language and examples available),
- Async and sync single-player and multi-player modes,
- Fast (up to 7000 fps in sync mode, single-threaded),
- Lightweight (few MBs),
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision),
- Automatic labelling game objects visible in the frame,
- Access to the audio buffer (thanks to [Shashank Hegde](https://github.com/hegde95)),
- Access to the list of actors/objects and map geometry,
- Off-screen rendering,
- Episodes recording,
- Time scaling in async mode.

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).




## Python quick start

### **Ubuntu**
To install ViZDoom on Ubuntu run (may take few minutes):
```
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm
pip install vizdoom
```
**Recommendation**: Ubuntu 18.04+ with Python 3.6+.

### **Conda**
To install ViZDoom on a conda environment (no system-wide installations required):
```
conda install -c conda-forge boost cmake gtk2 sdl2
git clone https://github.com/mwydmuch/ViZDoom.git
cd ViZDoom
python setup.py build && python setup.py install
```
Note that `pip install vizdoom` won't work with conda install and you have to follow these steps.

### **macOS** 
To install ViZDoom on macOS run (may take few minutes):
```
brew install cmake boost openal-soft sdl2
pip install vizdoom
```
We recommend using at least macOS High Sierra 10.13+ with Python 3.6+.
Currently, only for Intel CPU, we will gladly accept PR with M1 support.

### **Windows**
To install pre-build release for Windows 10 or 11 64-bit and Python 3.6+ just run (should take few seconds):
```
pip install vizdoom
```

### **Install the dependencies**
```
pip install -r requirements.txt
```

### **To Run**
To run the game, run the following command:
```
python run.py
```
## **Results**



Basic             |  Deadly_Corridor           |  Defend_the_Center
:-------------------------:|:-------------------------: |:-------------------------:
![vizdoom_Basic](res/Basic.gif)  |  ![vizdoom_Deadly_Corridor](res/Deadly_Corridor.gif)  |  ![vizdoom_Defend_the_Center](res/Defend_the_Center.gif) 

Defend_the_line             |  Health_gathering           |  Take_Cover
:-------------------------:|:-------------------------: |:-------------------------:
![vizdoom_Defend_the_line](res/Defend_the_line.gif)   |  ![vizdoom_Health_gathering](res/Health_gathering.gif)  |  ![vizdoom_Take_Cover](res/Take_Cover.gif)


## Examples

- [Python](examples/python) (contain learning examples implemented in PyTorch, TensorFlow and Theano)
- [C++](examples/c%2B%2B)
- [Julia](examples/julia)

Python examples are currently the richest, so we recommend to look at them, even if you plan to use other language. API is almost identical for all languages.

**See also the [tutorial](http://vizdoom.cs.put.edu.pl/tutorial).**


## Original Doom graphics

If you own original Doom or Doom 2 games, you can replace [Freedoom](https://freedoom.github.io/) graphics by placing `doom.wad` or `doom2.wad` into your working directory or `vizdoom` package directory.

Alternatively, any base game WAD (including other Doom engine-based games) can be used by pointing to it with the [`set_doom_game_path/setDoomGamePath`](https://github.com/mwydmuch/ViZDoom/blob/master/doc/DoomGame.md#-setdoomscenariopath) method.


## Documentation

Detailed description of all types and methods:

- **[DoomGame](doc/DoomGame.md)**
- **[Types](doc/Types.md)**
- [Configuration files](doc/ConfigFile.md)
- [Exceptions](doc/Exceptions.md)
- [Utilities](doc/Utilities.md)

Additional documents:

- **[FAQ](doc/FAQ.md)**
- [Changelog](doc/Changelog.md) for 1.1.X version.

Also full documentation of engine and ACS scripting language can be found on
[ZDoom Wiki](https://zdoom.org/wiki/).

Useful parts:

- [ZDoom Wiki: ACS (scripting language)](https://zdoom.org/wiki/ACS)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs)
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs)


## Awesome Doom tools/projects

- [SLADE3](http://slade.mancubus.net/) - great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - another great Doom map editor for Windows.
- [OBLIGE](http://oblige.sourceforge.net/) - Doom random map generator and [PyOblige](https://github.com/mwydmuch/PyOblige) is a simple Python wrapper for it.
- [Omgifol](https://github.com/devinacker/omgifol) - nice Python library for manipulating Doom maps.
- [NavDoom](https://github.com/agiantwhale/navdoom) - Maze navigation generator for ViZDoom (similar to DeepMind Lab).
- [MazeExplorer](https://github.com/microsoft/MazeExplorer) - More sophisticated maze navigation generator for ViZDoom.
- [ViZDoomGym](https://github.com/shakenes/vizdoomgym) - OpenAI Gym Wrapper for ViZDoom.
- [Sample Factory](https://github.com/alex-petrenko/sample-factory) - A high performance reinforcement learning framework for ViZDoom.


