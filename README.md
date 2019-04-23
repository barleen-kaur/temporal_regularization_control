
Extending A2C to temporal regularization.

Done:
- Code for temporal regularization

To do:
- rescaling the value of reg_coeff to match other loses
- regularization loss changes a lot based on trajectories, maybe rescale the value based on episode length?
- track mean reward instead of reward, change code to facilitate that


Original code repo: [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)