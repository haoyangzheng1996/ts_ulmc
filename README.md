# Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo

Part of experimental code for "[TS-ULMC](https://arxiv.org/abs/2401.11665)".

```
@article{ts_ulmc,
  title={Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo},
  author={Zheng, Haoyang and Deng, Wei and Moya, Christian and Lin, Guang},
  journal={arXiv preprint arXiv:2401.11665},
  year={2024}
}
```

## Prerequisites
Please refer to "environment.yml" 

## Usage
For our method, please run:
```
python3 ts_underdamp.py
```
For the baseline, please run:
```
python3 ts_overdamp.py
```
Results:
![image](https://github.com/haoyangzheng1996/ts_ulmc/assets/38525155/6989cfa6-45a0-4b1f-ac53-9e85b55206b7)

# Further example
To see the results in the appendix, we first used "[google-maps-scraper
]([https://arxiv.org/abs/2401.11665](https://github.com/omkarcloud/google-maps-scraper))" to collect Google Maps reviews from several restaurants.

The data are saved in ./data/ as txt files.

Then please run:
```
python3 restaurant_main.py --n_round 200 --batch_size 5 --gamma 2.0 --reward_size 10 --step_size 1e-2
```
Results:
![image](https://github.com/haoyangzheng1996/ts_ulmc/assets/38525155/46e11b95-ea20-483e-a5e8-4552affc45af)

## Contact
Haoyang Zheng, School of Mechanical Engineering, Purdue University

Email: zheng528 at purdue dot edu

More Aboue Me: [link](https://haoyangzheng.github.io/)
