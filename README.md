# README — MNIST 3 vs 5 (Bartlett)
## Run
pip install torch torchvision numpy pandas tqdm
## Run (1) training -> CSV
python Expérience FPDL.py
## Run (2) plots from CSV
python experience plot.py
## Input/params (edit in script)
widths=[16,64,256,1024], wds=[0,1e-5,1e-4,1e-3,1e-2], seeds=[0,1,2]
epochs/batch_size/lr configurable; n_train/n_test optional (auto-capped if too large after 3/5 filter)
## Output
Creates mnist_bartlett_experiment.csv (one row per run: width, weight_decay, seed)
Key columns: train_err01, test_err01, weight_norm (||W1||F+||W2||F, no bias)
Margins with y∈{-1,+1}, margin=y*f(x): train_margin_q10, train_margin_median
Also: test_margin_q10/test_margin_median and train_frac_margin_le_{0.0,0.5,1.0}
Console prints a grouped summary by (width, weight_decay)
