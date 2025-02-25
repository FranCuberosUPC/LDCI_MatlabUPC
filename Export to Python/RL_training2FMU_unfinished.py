from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import tensorflow as tf
from sb3_contrib import TRPO
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import argparse
from stable_baselines3.common.callbacks import CheckpointCallback
from keras.models import load_model

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(0, 1.0, (1,))
        self.observation_space = spaces.Box(0, 1.0, (4,))

        fmu1_filename = r"C:\Users\fjcub\Documents\UPC\INIREC\CORNET\Simulink\SAC\Sim2FMU\BSFCM0_FMU1_250114.fmu"
        fmu2_filename = r"C:\Users\fjcub\Documents\UPC\INIREC\CORNET\Simulink\SAC\Sim2FMU\BSFCM0_FMU2_250114.fmu"
        model_filename = r"C:\Users\fjcub\Documents\UPC\INIREC\CORNET\Simulink\SAC\Sim2FMU\model.h5"

        self.start_time = 0.0
        self.stop_time = 0.7
        self.step_size  =  1e-5

        model_description1 = read_model_description(fmu1_filename)
        inputs_fmu1 = [v for v in model_description1.modelVariables if v.causality == 'input']
        outputs_fmu1 = [v for v in model_description1.modelVariables if v.causality == 'output']

        model_description2 = read_model_description(fmu2_filename)
        inputs_fmu2 = [v for v in model_description2.modelVariables if v.causality == 'input']
        outputs_fmu2 = [v for v in model_description2.modelVariables if v.causality == 'output']
        
        ##FMU1
            #inputs
        self.Ksplit = [e.valueReference for e in inputs_fmu1 if e.name == 'Ksplit']         #Action
        self.main_outfb = [e.valueReference for e in inputs_fmu1 if e.name == 'main_out']   #Aux_in feedback from predict output
            #outputs
        self.main_in_refs = [[e.valueReference for e in outputs_fmu1 if e.name == f'main_in[1,1,{i}]'][0] for i in range(1, 7)]
        self.aux_in_refs = [[e.valueReference for e in outputs_fmu1 if e.name == f'aux_in[1,1,{i}]'][0] for i in range(1, 5)]       #Predict main output
        
        #FMU2
            #inputs
        self.main_out_ref = [e.valueReference for e in inputs_fmu2 if e.name == 'main_out']      #Aux_in feedback from predict output
            #outputs
        self.v_ksc = [e.valueReference for e in outputs_fmu2 if e.name == 'v_ksc']           #v_k scaled
        self.SoCsc = [e.valueReference for e in outputs_fmu2 if e.name == 'SoCsc']
        self.NOsc = [e.valueReference for e in outputs_fmu2 if e.name == 'NOsc']
        self.NO2sc = [e.valueReference for e in outputs_fmu2 if e.name == 'NO2sc']
        self.NOxsc = [e.valueReference for e in outputs_fmu2 if e.name == 'NOxsc']
        self.SoC_refsc = [e.valueReference for e in outputs_fmu2 if e.name == 'SoC_refsc']
        self.NOx_refsc = [e.valueReference for e in outputs_fmu2 if e.name == 'NOx_refsc']
        
        self.modeltf= load_model(model_filename)

        self.unzipdir1 = extract(fmu1_filename)
        self.unzipdir2 = extract(fmu2_filename)        
        self.fmu1 = FMU2Slave(guid=model_description1.guid,
                        unzipDirectory=self.unzipdir1,
                        modelIdentifier=model_description1.coSimulation.modelIdentifier,
                        instanceName='instance1')
        self.fmu2 = FMU2Slave(guid=model_description2.guid,
                        unzipDirectory=self.unzipdir2,
                        modelIdentifier=model_description2.coSimulation.modelIdentifier,
                        instanceName='instance1')
        
        self.time = self.start_time
        self.prev_action = 0.0
        self.prev_e = 0.0
        
    def step(self, action):
        v_ksc = self.fmu2.getReal(self.v_ksc)[0]
        SoCsc = self.fmu2.getReal(self.SoCsc)[0]
        NOsc = self.fmu2.getReal(self.NOsc)[0]
        NO2sc = self.fmu2.getReal(self.NO2sc)[0]
        NOxsc = self.fmu2.getReal(self.NOxsc)[0]
        SoC_refsc = self.fmu2.getReal(self.SoC_refsc)[0]
        NOx_refsc = self.fmu2.getReal(self.NOx_refsc)[0]

        #Predict and link both FMUs
        main_in_values = np.array([[self.fmu1.getReal([ref])[0] for ref in self.main_in_refs]]).reshape(1, 1, 6)
        aux_in_values = np.array([[self.fmu1.getReal([ref])[0] for ref in self.aux_in_refs]]).reshape(1, 1, 4)
        #main_in_values = np.array([[self.fmu1.getReal(self.main_in)[0]]]).reshape(1, 1, 6)
        #aux_in_values = np.array([[self.fmu1.getReal(self.aux_in)[0]]]).reshape(1, 1, 4)

        # Ejecutar el modelo predictivo
        model_output = self.modeltf.predict([main_in_values, aux_in_values], batch_size=1, steps=1)

        # Pasar la salida del modelo a FMU2
        self.fmu2.setReal([self.main_out_ref], model_output[0].tolist())

        # Retroalimentar la salida del modelo a FMU1
        self.fmu1.setReal([self.main_outfb_ref], [model_output[0][0]])
        
        c1=0.3
        c2=0.7
        reward = c1*(SoCsc-SoC_refsc)^2 - c2*(NOxsc-NOx_refsc)^2
        
        observation = np.array([v_ksc, SoCsc, NOsc, NO2sc, self.prev_action])
        
        a = np.clip(action[0], 0, 1.0)
        self.fmu1.setReal(self.Ksplit, a) #No sé si [a] va con o sin corchetes
        
        for i in range(int(0.01/self.step_size)):
            self.fmu1.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)
            self.fmu2.doStep(currentCommunicationPoint=self.time, communicationStepSize=self.step_size)
            self.time += self.step_size
            
        self.prev_action = a
        terminated = (self.time >= self.stop_time)
        truncated = False
        info = {'reward':reward}
        
        return observation, reward, terminated, truncated, info

    
    def reset(self, seed=None, options=None):
        self.fmu1.instantiate()
        self.fmu1.setupExperiment(startTime=self.start_time)
        self.fmu1.enterInitializationMode()
        self.fmu1.exitInitializationMode()

        self.fmu2.instantiate()
        self.fmu2.setupExperiment(startTime=self.start_time)
        self.fmu2.enterInitializationMode()
        self.fmu2.exitInitializationMode()

        self.time = self.start_time
        self.prev_action = 0.0
        self.prev_e = 0.0
        observation = [0.0, 0.0, 0.0, 0.0, 0.0]
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        self.fmu1.terminate()
        self.fmu2.terminate()
        self.fmu1.freeInstance()
        self.fmu2.freeInstance()
        shutil.rmtree(self.unzipdir1, ignore_errors=True)
        shutil.rmtree(self.unzipdir2, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default="ppo")
    parser.add_argument('-g', '--gpu_number', default=0)
    parser.add_argument('-n', '--name', default='')
    args = parser.parse_args()
    
    env = CustomEnv()
    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
      save_freq=1000,
      save_path="./logs/",
      save_replay_buffer=True,
      save_vecnormalize=True,
      name_prefix=args.name+'_' + str(args.algorithm)
    )
    
    if args.algorithm.lower() == 'trpo':
        model = TRPO("MlpPolicy", env, verbose=False, tensorboard_log="./simp_tensorboard/", device='cuda:' + str(args.gpu_number))
    elif args.algorithm.lower() == 'ppo':
        model = PPO("MlpPolicy", env, verbose=False, tensorboard_log="./simp_tensorboard/", device='cuda:' + str(args.gpu_number))
    elif args.algorithm.lower() == 'sac':
        model = SAC("MlpPolicy", env, verbose=False,  tensorboard_log="./simp_tensorboard", device='cuda:' + str(args.gpu_number))

        
    model.learn(total_timesteps=100_000, tb_log_name=args.name, callback=checkpoint_callback)


