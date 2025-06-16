# DRL_sailboat
Model source: Modeling and Nonlinear Heading Control of Sailing Yachts, Xiao and Jouffroy, 2014.

This is a reinforcement learning method optimized based on the SAC model. This method directly controls the rudder end-to-end through the changes in the sailboat's own state (without relying on wind field detection). In this project, the long short-term memory network was integrated into the traditional SAC model, achieving good results.

python environmental-related configuration:
python 3.11
pytorch 2.2.0
gymnasium 0.29.1

The overall document includes:
1. gymnasium environmental model (one with wind field information, one without wind field information)
2. Training files (SAC, SAC-LSTM)
3. Test files (single tests, batch tests, and models that have completed related training)
   
The file code may have some variables that need to be modified manually and some bugs, but overall, it does not affect the usage. (This is only a summary in my first article and is for reference only.

