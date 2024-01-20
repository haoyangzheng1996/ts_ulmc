clear

conda activate thompson

python underdamp_langevin.py --n_round 200 --n_arm 10 --dim 2 --batch_size 10 --seed 2403
# python over_langevin.py --n_round 200 --n_arm 10 --dim 2 --batch_size 10 --seed 2403