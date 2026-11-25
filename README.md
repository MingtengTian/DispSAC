<img width="900" height="320" alt="image" src="https://github.com/user-attachments/assets/8c1d0b5d-2f2f-43ad-9f07-5804f6565be5" />

This is the python codes package for DispSAC that is a surface wave inversion method based on soft actor-critic algorithm of off-policy reinforce learning. The preprint paper can be accessed in https://eartharxiv.org/repository/view/15137/.

These python codes built a DispSAC agent training how to decrease misfit in the direction of maximal return. There are few components: 

"config.py": defines the training and inversion settings. It groups the network dimensions, learning rates, replay-buffer settings, action scales, reward coefficients, data paths, and physical parameter bounds.

"augmentation.py": perturbs phase observations and simulates incomplete period coverage.

"environment.py": pplies actions, runs the forward model, and computes the reward. 

"network.py": defines the state encoder, actor, and twin critics. 

"replay_buffer.py": stores and samples previous transitions. The circular buffer stores the current state, action, reward, next state, and terminal status.

"sac.py": implements the learning objectives and update order.

"utils.py": connects the model parameters to the physical forward solver.

"train.py" ties these components together.

Related training data and test data can be assessed in the Hugging Face https://huggingface.co/datasets/MintonTian/DispSAC. 
