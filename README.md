
Extending A2C to temporal regularization.

Done:
- Code for temporal regularization
- Tracking mean rewards every 10 episodes and storing in csv	

To do:
- rescaling the value of reg_coeff to match other loses
- regularization loss changes a lot based on trajectories, maybe rescale the value based on episode length?

Running code:
- `python main.py --env-name MountainCar-v0 --num-env-steps 800000 --reg 0`

Original code repo: [https://github.com/ikostrikov/pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)