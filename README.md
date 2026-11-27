# BendFlow

Closed-form trajectory curvature for flow matching via posterior moments, local active-set bounds on bending, curvature-adaptive Euler sampler with free step-size estimation.

Project for the Visual Generative Models course at Universität des Saarlandes, taught by Professor Jan Eric Lenssen, with Hevra Pettekaya as teaching assistant (PhD student). Winter semester 2026/27.

## The problem (in natural language)

A flow matching model generates an image by starting from noise and following a learned velocity field until it arrives at data. In practice this is done with a handful of Euler steps (or other ODE/SDE solvers). Some parts of that journey are easy: the path is straight and a large step loses nothing. Other parts are hard: the path bends, and a large step lands in the wrong place.

The answer to "where is it hard?" can be a fixed range of timesteps, chosen per dataset. That is unsatisfying for two reasons. It does not say why those timesteps are hard, and it treats every sample the same, although one sample may be torn between two similar digits while another committed to its image long ago.

`BendFlow` asks the question differently: for one sample, at one moment, what makes its path bend, and can that be read off from quantities the sampler already has?

## The idea

At every noisy point the model is implicitly answering "which training image did this come from?". That answer is a probability distribution over the training images, the posterior. Three things about it matter:

- its mean `m`, the model's current best guess of the clean image,
- its covariance `C`, how unsure that guess is,
- its third moment `κ`, how lopsided the uncertainty is.

The claim of the project is that difficulty is indecision. A path bends mainly while the posterior is still spread over several training images that are far apart, and it is straight once the posterior has committed to one (put almost all its weight on it). All results are for the linear path $x_t = t\,x_1 + (1 - t)\,\varepsilon$ used by flow matching.

## The approach, step by step

**1. An exact field to reason about.** Treat the training set itself as the data distribution. Then the optimal flow matching velocity field can be written down exactly: it points from the current point to the posterior mean `m`, scaled by the remaining time. No network is involved, so everything that follows can be proved instead of observed.

**2. How the field changes in space.** Differentiating the field in space shows that its Jacobian is the posterior covariance `C`, rescaled, minus a multiple of the identity. Nearby trajectories therefore only separate while `C` is large, that is, while the sample is still choosing between distant images; otherwise they are pulled together onto one image. This identity was already known (Wan et al., Lv et al.), and so was the matching statement that the divergence of the field equals the variance of the training targets up to known factors. Both are restated here in one notation.

**3. How a single path bends.** Rao et al. showed that the acceleration of a trajectory equals the rate at which its denoised prediction `m` moves, divided by the remaining time. They left that rate as an unexplained symbol. The main result of this project computes it in closed form. A path bends only through the spread and the lopsidedness of the posterior. This also gives the error of one Euler step directly.

**4. Bounds from the neighbourhood.** `C` and `κ` are sums over the whole training set. Because posterior weights fall off exponentially with distance, only a small "active set" of images near the current point matters; the rest carry at most a chosen weight `δ`. The bending can then be bounded using only the diameter of the active set. The bound on `C` cannot be improved, as a two-image example shows. The result is a statement that is local and data-dependent, not a fixed range of timesteps.

**5. A sampler that uses this.** Every Euler step already produces a denoised prediction. The difference between two consecutive predictions estimates $\dot m$ to first order, at no extra network evaluations. The step size is then chosen so that every step commits the same local error: small steps where the path bends, large steps where it is straight, separately for each sample.

## Results

The write-up is [`documentation/paper.pdf`](documentation/paper.pdf), and the derivations are in [`documentation/handwritten_notes.pdf`](documentation/handwritten_notes.pdf). Experiments on MNIST, Fashion-MNIST and CIFAR-10 show:

- The closed forms match automatic differentiation to a relative error below $10^{-12}$, and the bounds are never violated. The bound on `C` is within a factor of 11 to 19 in the median, but the bounds on `κ` and $\dot m$ are two to three orders of magnitude too loose.
- The bending of the exact field peaks when the posterior commits on 17 of 24 trajectories. The others peak at the very start, driven by the covariance of the whole dataset.
- A small U-Net and a small DiT follow the exact field until it commits and smooth afterwards; neither copies training images.
- With tuned settings, the curvature-adaptive sampler lowers the Euler error at the same number of network evaluations by 11 to 30 % on Fashion-MNIST and CIFAR-10. On MNIST, a fixed schedule built from the average curvature is as good or better.

## Setup

The code is written in PyTorch and managed with [uv](https://docs.astral.sh/uv/).

    git clone https://github.com/Stochastic-Batman/BendFlow.git
    cd BendFlow
    uv sync --locked

Check the installation and run the tests:

    uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    uv run pytest

Python 3.12 is pinned in `.python-version`; uv installs it if it is missing.

## Reproducing the results

The archived JAX experiment outputs are in `results/`. New PyTorch runs write to
`results/pytorch/<schedule>/` and store weights in `checkpoints/<schedule>/`.
Omit `--schedule` for the original linear path, or use `--schedule cosine` for
cosine VP and `--schedule ve` for exponential VE. Every schedule trains its own
U-Net and DiT. The experiments use Euler on the linear path and deterministic
DDIM on the nonlinear paths. VE starts from the empirical initial marginal
`x_1 + 10 epsilon`.

| Command | What it does |
|---|---|
| `uv run python experiments/train.py --dataset cifar10 --model unet` | trains the U-Net |
| `uv run python experiments/train.py --dataset cifar10 --model dit` | trains the DiT |
| `uv run python experiments/exact_field.py --dataset cifar10 --div-batch 1` | identities, bounds and active set of the exact field |
| `uv run python experiments/trained_networks.py --dataset cifar10` | networks against the exact field |
| `uv run python experiments/sampler_comparison.py --dataset cifar10` | adaptive and uniform sampling at matched cost |
| `uv run python experiments/schedule_baseline.py --dataset cifar10 --nfe 5 10 20 40 --num-samples 128 --seed 4 --h-max-grid 0.25 0.5 --out results/pytorch/linear/schedule_baseline_cifar10_n128` | comparison with a calibrated fixed grid |
| `uv run python experiments/sample_grids.py --dataset cifar10` | sample grids using the 128-sample calibration |

## What is new and what is not

Known and cited: the Jacobian identity (second-order Tweedie; Wan et al. 2024, Lv et al. 2026), the divergence identity (Lv et al. 2026), and the acceleration identity (Rao et al. 2026).

To the best of my knowledge new: the closed form for $\dot m$, the local active-set bounds, and the step-size rule with its free estimator. "To the best of my knowledge" means a search of the closest papers, not a proof that nobody has done it.

## References

- Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023.
- Wan, Wang, Mishne, Wang, *Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers*, arXiv:2412.18730.
- Lv, Xing, Liang, Wang, *Uncertainty in a Single Pass: A Closed-Form Identity for One-Step Flow Matching*, arXiv:2605.00941.
- Rao, Zhao, Guo, Fei, Guo, Xiong, *The Geometry of Flow-Matching Uncertainty*, arXiv:2607.27933.
- Manor, Michaeli, *On the Posterior Distribution in Denoising: Application to Uncertainty Quantification*, arXiv:2309.13598.
- Milanfar, *A Lagrangian View of Flow Matching*, arXiv:2609.00198.

## Acknowledgement

Claude (Anthropic) was used to verify the derivations (and simplify some!), to search for related work, and it introduced the notation `C` and `κ`; `m` is also the standard notation for the denoiser in the literature. Claude was also used to check the author-written Python code and to write all the tests and the testing scripts, which the author reviewed manually. The handwritten math derivations are available in the repository, and the LaTeX document was constructed from them.
