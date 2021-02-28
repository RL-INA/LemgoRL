# LemgoRL

LemgoRL is an open-source benchmark tool for traffic signal control designed to train reinforcement learning agents in a highly realistic simulation scenario with the aim to reduce Sim2Real gap.

Associated paper is available at: <<Published Paper link goes here>>

#### Key Features:

- Simulation model of real-world intersection of Lemgo in SUMO (Simulation of Urban MObility)
- Real-world vehicle and pedestrian traffic demand modeling of the intersection
- Traffic Engineering through Traffic Signal Logic Unit (Lisa+ OmlFgServer: <<Link to more details>>)
- OpenAI Gym based environment to train RL Agents
- Middleware to facilitate communication between environment controller and Traffic Signal Logic Unit
- Python based Greedy policy implementation to serve as a base policy
- Evaluation function to plot graphs of queue length, average speed and wait time w.r.t. simulation time and w.r.t. episodes.
- Visualize function to see the graphical simulation of the traffic and control of intersection by greedy policy in SUMO-GUI.

#### License:

- MIT or GNU GPL V3

#### OmlFgServer and Traffic Signal Logic Unit

In order for the Traffic Signal Logic Unit to work, the **Lisa+ OmlFgServer.jar** is needed. The Lisa+ OmlFgServer.jar is a proprietary Virtual Controller Server for traffic lights which can be purchased from the company **Schlothauer & Wauer**. Since it is a proprietary product, it cannot be shipped with this LemgoRL open-source repository.

#### Installation Steps

1. Install Microsoft Visual C++ 14.0 from https://aka.ms/vs/16/release/vc_redist.x64.exe
2. Install Anaconda / Miniconda from https://www.anaconda.com/products/individual
3. In anaconda prompt, execute the command `conda env create -f environment.yml`. This command will create an anaconda environment "LemgoRL".
4. Activate the anaconda environment with the command `conda activate LemgoRL` for Windows or `activate LemgoRL` for Linux/Unix/Mac.
5. Create a directory `lisa_vissim_addon` in root directory of this project and place the OmlFgServer.jar procurred from the company **Schlothauer & Wauer** in this directory.

#### Visualize Mode

This source code comes with `main.py` script as a test script which can be used to launch the LemgoRL simulation model in SUMO-GUI mode and observe the control of intersection by greedy policy. To do so, following command needs to be executed from the root directory of this project:

`python main.py visualize`

#### Evaluate Mode

The `main.py` script also supports the Evaluate Mode which can be used to evaluate queue length , avg. speed and wait time w.r.t to episodes. By default, the evaluate mode runs for 5 episodes on greedy policy and produces output in the **results** directory. The number of episodes can be changed by passing command argument `--episodes N` where N represents number of episodes to be evaluated. The evaluate mode also supports verbose evaluation and reporting in which queue length, avg. speed and wait time are reported w.r.t. to simulation second for each episode. To turn this on, the command argument `--verbose True` should be passed. In its default form, the evaluate mode can be executed as follows:

`python main.py evaluate`

#### Project Structure

#### Using the OWL322 Environment

#### BibTex entry

#### Contributing to this project

