# Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo
[![Paper](https://img.shields.io/badge/PRML-238:2611--2619-orange)](https://proceedings.mlr.press/v238/zheng24b.html)
[![Dependencies](https://img.shields.io/badge/Python-3.10.0-yellow)](https://github.com/haoyangzheng1996/ts_ulmc/blob/main/environment.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/haoyangzheng1996/ts_ulmc/blob/main/LICENSE.md)

Part of experimental code for "[TS-ULMC](https://arxiv.org/abs/2401.11665)".

```
@article{zheng2024accelerating,
  title={{Accelerating Approximate Thompson Sampling with Underdamped Langevin Monte Carlo}},
  author={Zheng, Haoyang and Deng, Wei and Moya, Christian and Lin, Guang},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={2611--2619},
  year={2024},
  organization={PMLR}
}
```

# Prerequisites
Please refer to "environment.yml" 

# Usage
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

## Further example
To see the results in the appendix, we first used [google-maps-scraper
](https://github.com/omkarcloud/google-maps-scraper) to collect Google Maps reviews from several restaurants.

The data were further processed and saved in the folder "./data/" as txt files.

Then please run:
```
python3 restaurant_plot.py --n_round 200 --batch_size 5 --reward_size 10 --step_size 1e-2
```
Results:
![image](https://github.com/haoyangzheng1996/ts_ulmc/assets/38525155/45d343c0-97f8-4799-9dd2-e28502e778be)


# Contact
Haoyang Zheng, School of Mechanical Engineering, Purdue University

Email: zheng528 at purdue dot edu

More Aboue Me: [link](https://haoyangzheng.github.io/)
