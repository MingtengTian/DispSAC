<img width="612" height="209" alt="image" src="https://github.com/user-attachments/assets/8c1d0b5d-2f2f-43ad-9f07-5804f6565be5" />

This is the python codes package for DispSAC that is a surface wave inversion method based on soft actor-critic algorithm of off-policy reinforce learning. The preprint paper can be accessed in https://eartharxiv.org/repository/view/15137/.

These python codes built a DispSAC agent training how to decrease misfit in the direction of maximal return. There are few components for the 

"config.py": adjust training hyper-parameters. 

"augmentation.py": adjust the Gaussian noise and Cutting mechnism. 

"environment.py": run forward simulations and calculate the reward. 

"network.py": define the network architecture of actor and critic. 

"replay_buffer.py": define the buffer for replaying prioritized transitions, making agent learngs from previous experience.

"sac.py": define the learning loss for neural network. Define the order of backpropogation. 

"utils.py": 

Related training data and test data can be assessed in the Hugging Face https://huggingface.co/datasets/MintonTian/DispSAC. 
