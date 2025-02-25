# LDCI_MatlabUPC
This GitHub repo contains most of the code used to train a NLMPC and SAC algorithms in Matlab Simulink using a model with a neural network. This neural network simulates a HEV and from a series of inputs obtains the NOx emissions (NO, NO2), the State
of Charge of the battery (SoC) and the shaft_torque. The main folders are:  
    **Training Data Creation:** NLMPC definition, creation of the neural network training data needed and training of that network. NLMPC implementation and simulation in Simulink.  
    **RL SAC:** Soft Actor-Critic algorithm definition in matlab and simulation in Simulink.  
    **Other Functions:** Scripts used for creating synthetic driving cycles or analyzing correlations between inputs and outputs.  
    **Models:** Contains neural network models used. Models M0, M3 and M3v2.  
    **Export to Python:** Pre-processing BSFC (Brake Specific Fuel Consumption) exported from Matlab to Python and other files that have not been finished and/or tested.
    
