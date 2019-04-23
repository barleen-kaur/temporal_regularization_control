
Extending A2C to temporal regularization.

Done:
- Code for temporal regularization
- Tracking mean rewards every 10 episodes and storing in csv

To do:
- rescaling the value of reg_coeff to match other loses
- regularization loss changes a lot based on trajectories, maybe rescale the value based on episode length?


Original code repo: [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)