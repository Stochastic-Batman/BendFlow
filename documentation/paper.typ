// Typst version of paper.tex. Compile from the repository root, so that the
// figures in results/ are inside the project root:
//
//   typst compile --root . documentation/paper.typ
//
// References are in paper.bib (converted from the thebibliography block).

#let results = "../results/"
#let today = datetime.today().display("[month repr:long] [day padding:none], [year]")

// ---------------------------------------------------------------------------
// Page, text and paragraphs: NeurIPS-like single column, 5.5in x 9in text
// block, 10pt Times, no indent, 5.5pt between paragraphs.
// ---------------------------------------------------------------------------
#set document(
  title: "BendFlow: Closed-Form Trajectory Curvature and Curvature-Adaptive Sampling for Flow Matching",
  author: ("Lado Turmanidze", "Geronimo Castaño"),
)
#set page(paper: "us-letter", margin: (x: 1.5in, top: 1in, bottom: 1in), numbering: "1")
#set text(font: ("Times New Roman", "Times"), size: 10pt, lang: "en")
#show math.equation: set text(font: "New Computer Modern Math")
// spacing = line gap + LaTeX's \parskip of 5.5pt
#set par(leading: 0.5em, spacing: 1.05em, first-line-indent: 0pt, justify: true)

// ---------------------------------------------------------------------------
// Headings: "1 Introduction", "7.1 ...", appendices "A ...".
// ---------------------------------------------------------------------------
#set heading(numbering: "1.1", supplement: [Section])
#show heading.where(level: 1): it => block(above: 1.6em, below: 0.9em, text(size: 12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1.4em, below: 0.7em, text(size: 10pt, weight: "bold", it))

// ---------------------------------------------------------------------------
// Equations: only labelled block equations get a number, as in LaTeX where
// every numbered equation here has a \label. References print "(3)".
// ---------------------------------------------------------------------------
#set math.equation(numbering: "(1)", supplement: none)
#show math.equation: it => {
  if it.block and not it.has("label") and it.numbering != none {
    counter(math.equation).update(v => v - 1)
    math.equation(it.body, block: true, numbering: none)
  } else {
    it
  }
}
#show ref: it => {
  let el = it.element
  if el != none and el.func() == math.equation {
    link(el.location(), numbering("(1)", ..counter(math.equation).at(el.location())))
  } else {
    it
  }
}

#let Cov = math.op("Cov")
#let Var = math.op("Var")
#let tr = math.op("tr")

// ---------------------------------------------------------------------------
// Citations (natbib round, author-year).
// ---------------------------------------------------------------------------
#let citet(key) = cite(key, form: "prose")
#let citealp(key) = [#cite(key, form: "author"), #cite(key, form: "year")]
#let nref(l) = ref(l, supplement: none)

// ---------------------------------------------------------------------------
// Theorem-like environments, one counter each (as with \newtheorem).
// Usage: #theorem[...] <thm:x>   or   #proposition(note: [...])[...] <prop:y>
// ---------------------------------------------------------------------------
#let thm-env(kind, supplement) = (body, note: none) => figure(
  kind: kind,
  supplement: supplement,
  numbering: "1",
  outlined: false,
  caption: note,
  body,
)
#let theorem = thm-env("theorem", [Theorem])
#let proposition = thm-env("proposition", [Proposition])
#let lemma = thm-env("lemma", [Lemma])
#let corollary = thm-env("corollary", [Corollary])
#let remark = thm-env("remark", [Remark])

#let thm-show(it, plain: true) = block(width: 100%, above: 1.2em, below: 1.2em, breakable: true, {
  set align(left)
  set par(justify: true)
  let head = [#it.supplement #context it.counter.display(it.numbering)]
  if plain { strong(head) } else { emph(head) }
  if it.caption != none [ (#it.caption.body)]
  if plain { strong[.] } else { emph[.] }
  h(0.5em)
  if plain { emph(it.body) } else { it.body }
})
#show figure.where(kind: "theorem"): it => thm-show(it)
#show figure.where(kind: "proposition"): it => thm-show(it)
#show figure.where(kind: "lemma"): it => thm-show(it)
#show figure.where(kind: "corollary"): it => thm-show(it)
#show figure.where(kind: "remark"): it => thm-show(it, plain: false)

#let proof(body, title: [Proof]) = block(width: 100%, above: 1.2em, below: 1.2em, breakable: true)[
  _#title._ #body #h(1fr) $square$
]

// ---------------------------------------------------------------------------
// Floats: captions above tables, booktabs-style rules, algorithm box.
// ---------------------------------------------------------------------------
#show figure.where(kind: table): set figure.caption(position: top)
#show figure.caption: it => align(left, it)
#set table(stroke: none, inset: (x: 5pt, y: 3pt))
#let toprule = table.hline(stroke: 0.08em)
#let midrule = table.hline(stroke: 0.05em)
#let bottomrule = table.hline(stroke: 0.08em)

#show figure.where(kind: "algorithm"): it => block(width: 100%, {
  set align(left)
  line(length: 100%, stroke: 0.08em)
  v(-0.3em)
  [*#it.supplement #context it.counter.display(it.numbering)* #it.caption.body]
  v(-0.3em)
  line(length: 100%, stroke: 0.05em)
  it.body
  line(length: 100%, stroke: 0.08em)
})

// ---------------------------------------------------------------------------
// Title block.
// ---------------------------------------------------------------------------
#v(0.1in)
#line(length: 100%, stroke: 4pt)
#v(0.15in)
#align(center, text(size: 17pt, weight: "bold")[
  BendFlow: Closed-Form Trajectory Curvature and\ Curvature-Adaptive Sampling for Flow Matching
])
#v(0.15in)
#line(length: 100%, stroke: 1pt)
#v(0.09in)
#align(center, grid(
  columns: 2,
  column-gutter: 3em,
  align: center,
  [*Lado Turmanidze*\ Universität des Saarlandes\ #today],
  [*Geronimo Castaño*\ Universität des Saarlandes\ #today],
))
#v(0.25in)

#align(center, text(size: 12pt, weight: "bold")[Abstract])
#pad(x: 0.5in)[
The discretisation error of a flow matching sampler is governed by how much its trajectories bend. We study this bending for the exact marginal field of a finite training set under a general Gaussian path $x_t = alpha_t x_1 + sigma_t epsilon$. We derive the closed form $dot(m) = dot(gamma)_t (C hat(epsilon) - gamma_t kappa)$, where $C$ and $kappa$ are the covariance and third central moment of the posterior over training images, $hat(epsilon)$ is the predicted noise, and $gamma_t = alpha_t \/ sigma_t$ is the signal-to-noise ratio (SNR). We bound both moments by the diameter of an active set of nearby training images, and turn the result into a step-size rule for the DDIM sampler, which is the Euler sampler on the linear path, that costs no extra network evaluations. On MNIST, Fashion-MNIST and CIFAR-10, the bending peaks when the posterior commits to one image on most trajectories, and trained U-Nets and DiTs follow the exact field only until that point. With tuned settings, the step-size rule lowers the Euler error at matched cost by $11%$ to $30%$ on Fashion-MNIST and CIFAR-10; on MNIST a fixed schedule built from the same rule is as good or better.
]

= Introduction

A flow matching model @lipman2023flow generates an image by integrating an ordinary differential equation $dot(x) = u_t (x)$ from noise at $t = 0$ to data at $t = 1$. Where the path is straight, large steps are fine; where it bends sharply, large steps introduce error. Recent work locates the hard part in a range of timesteps and then spends more training or more sampling steps there @csflow @stablevelocity.

Instead of a result of the form "between timesteps $0.3$ and $0.4$ the field changes fastest", we would like the difficulty to depend on the field and on the neighbourhood of the current point. Our starting observation is that at every noisy point the model implicitly has to decide which training image the point came from. We argue that a path can only bend while this posterior is undecided between training images that are far apart; once it commits to one image, the bending stops, apart from the schedule's own curve.

*Contributions.*
- Starting from the known identities for the spatial Jacobian, the divergence and the trajectory acceleration, which we restate in one notation (@sec:known), we derive a closed form for the rate of change $dot(m)$ of the denoised prediction along a trajectory of a general Gaussian path in terms of the posterior covariance, the third moment and the SNR (@thm:mdot). #citet(<tang2026wasserstein>) write the partial time derivative of the posterior mean as a posterior covariance for variance-preserving paths; to our knowledge, no earlier work reduces the derivative along a trajectory to $C$ and $kappa$.
- We bound the bending by the diameter of an active set of training images (@thm:local); the $1/4 D_A^2$ bound on the covariance is tight.
- We derive a curvature-adaptive step size for deterministic DDIM, which reduces to Euler on a linear path. Together with a sampler based on an estimator of $dot(m)$ that costs no extra network evaluations (@sec:sampler).

The first two items are statements about the exact field of a finite training set. Whether trained networks follow them is an empirical question, which @sec:experiments addresses.

= Setup and notation <sec:setup>

Let $cal(X) = (x^((1)), dots, x^((N)))$ be the training set in $RR^d$ and let the path be $x_t = alpha_t x_1 + sigma_t epsilon$ with $epsilon tilde.op cal(N)(0, I)$ independent of $x_1$. We treat the training set as the data distribution, so the marginal path is
$ p_t (x) = 1/N sum_(i = 1)^N cal(N)(x; alpha_t x^((i)), sigma_t^2 I). $
Let
$ w_i (x, t) = (exp(-||x - alpha_t x^((i))||^2 \/ 2sigma_t^2)) / (sum_j exp(-||x - alpha_t x^((j))||^2 \/ 2sigma_t^2)) $ <eq:w>
be the posterior, that is, the probability that $x$ came from image $i$, and define
$ 
m = sum_i w_i x^((i)), quad C = sum_i w_i z_i z_i ^top, quad kappa = sum_i w_i ||z_i||^2 z_i, quad z_i = (x^((i)) - m)
$ <eq:moments>
$ 
hat(epsilon) = (x - alpha_t m)/sigma_t, wide u_t (x) = dot(alpha)_t m + dot(sigma)_t hat(epsilon).
$ <eq:u>
We write $gamma_t = alpha_t \/ sigma_t$ for the signal-to-noise ratio (SNR), so that $dot(gamma)_t = (dot(alpha)_t sigma_t - alpha_t dot(sigma)_t) \/ sigma_t^2$. We assume that $alpha$ and $sigma$ are continuous on $[0, 1]$ and twice continuously differentiable on $[0, 1)$, with right derivatives at $t = 0$. We assume $alpha_t > 0$ for $0 < t < 1$ and $sigma_t > 0$ and $dot(gamma)_t > 0$ for $0 <= t < 1$. At the clean endpoint, $alpha_1 = 1$ and $sigma_1 = 0$.

Here $m$, $C$ and $kappa$ are the first three moments of the posterior: its mean, spread and asymmetry. The mean $m$ is the average guess of the clean image. The covariance $C$ describes how uncertain that guess is, and it is $0$ when we are $100%$ sure. The third moment $kappa$ describes the asymmetry of the posterior and is $0$ when the posterior is symmetric. We will need $C$ when we differentiate $m$ with respect to space and $kappa$ when we differentiate it along a trajectory.

The field $u_t$ in @eq:u is the marginal field. For a fixed image $x_1$, the conditional path $alpha_t x_1 + sigma_t epsilon$ moves with velocity $dot(alpha)_t x_1 + dot(sigma)_t epsilon$. Writing $epsilon = (x - alpha_t x_1) \/ sigma_t$ gives this velocity in terms of the current point, $u_t (x | x_1) = (dot(sigma)_t \/ sigma_t) x + sigma_t dot(gamma)_t x_1$. The marginalization theorem averages it over the posterior, which gives @eq:u, or equivalently
$ u_t (x) = (dot(sigma)_t)/(sigma_t) x + sigma_t dot(gamma)_t m $ <eq:u-x>
(@app:marginal). The matrix $C$ is symmetric and positive semi-definite (@app:psd), so its eigenvalues are real and lie in $[0, lambda_max]$ with $lambda_max = max_(||v|| = 1) v^top C v$.

Our running example is the linear path, $alpha_t = t$ and $sigma_t = 1 - t$, for which $gamma_t = t \/ (1 - t)$, $dot(gamma)_t = 1 \/ (1 - t)^2$ and @eq:u-x reads $u_t (x) = (m - x) \/ (1 - t)$. Throughout, a dot denotes the total derivative $(d)/(d t)$ along a trajectory $dot(x) = u_t (x)$.

= Known identities in our notation <sec:known>

*Jacobian.* We are interested in $nabla_x u_t (x)$, since it describes how nearby trajectories move relative to each other; a large positive eigenvalue means that they separate fast. Since @eq:u-x writes $u_t$ through $x$ and $m$ only, it suffices to compute $nabla_x m$.

#proposition(note: [#citealp(<wan2024elucidating>), Prop. C.2; #citealp(<lv2026uncertainty>), Eq. 16 for the linear path])[
$ nabla_x m = alpha_t / sigma_t^2 C, wide nabla_x u_t = gamma_t dot(gamma)_t C + (dot(sigma)_t)/(sigma_t) I. $ <eq:jac>
] <prop:jac>

This is the second-order Tweedie formula @efron2011tweedie @manor2024posterior. We give a self-contained proof for the finite training set in @app:jac. Because $C$ is multiplied by a scalar and shifted by a multiple of $I$, $nabla_x u_t$ has the same eigenvectors as $C$, and each eigenvalue $lambda$ of $C$ becomes
$ mu = gamma_t dot(gamma)_t lambda + (dot(sigma)_t)/(sigma_t). $
As $lambda in [0, lambda_max]$ and $mu$ increases with $lambda$ (because $gamma_t dot(gamma)_t > 0$), the eigenvalues of $nabla_x u_t$ lie in
$ [(dot(sigma)_t)/(sigma_t), thick gamma_t dot(gamma)_t lambda_max (C) + (dot(sigma)_t)/(sigma_t)]. $ <eq:eigrange>
If $mu < 0$, nearby trajectories contract in that direction. When $C = 0$ and $dot(sigma)_t < 0$, they contract in every direction at relative rate $-dot(sigma)_t \/ sigma_t$. Some direction separates if and only if
$ lambda_max (C) > tau_t, wide tau_t := -(dot(sigma)_t)/(sigma_t gamma_t dot(gamma)_t), $ <eq:tau>
When $dot(sigma)_t < 0$, separation requires enough posterior spread to overcome contraction due to the schedule. For the linear path, $tau_t = (1 - t)^2 \/ t$. @app:eig shows that if the training-set diameter $D$ is positive, separation requires $1 - max_i w_i > tau_t \/ D^2$.

Since $C$ is symmetric, $nabla_x u_t$ is symmetric, so locally the field only stretches or compresses along orthogonal axes (the eigenvectors of $C$) and never rotates. The eigenvalue map, the range @eq:eigrange and this last statement are also shown in @app:eig.

*Divergence.* Taking the trace of @eq:jac gives
$ nabla dot.op u_t = gamma_t dot(gamma)_t tr(C) + d (dot(sigma)_t)/(sigma_t), wide (d)/(d t) log p_t (x_t) = -nabla dot.op u_t (x_t), $ <eq:lv>
where the second identity is the continuity equation along a trajectory. Solved for $tr(C)$, the first one says that the divergence of the field measures the spread of the posterior, $tr(C) = (nabla dot.op u_t - d dot(sigma)_t \/ sigma_t) \/ (gamma_t dot(gamma)_t)$. For the linear path this is the identity of #citet(<lv2026uncertainty>) (Eq. 19 and Section 3), $tr(C) = (1 - t)^2 \/ t (d + (1 - t) nabla dot.op u_t)$.

Subtracting the two forms of the field in @eq:u-x gives $u_t (x | x^((i))) - u_t (x) = sigma_t dot(gamma)_t (x^((i)) - m)$, so the variance of the training target at $x$ is
$ V_t (x) = sum_i w_i norm(u_t (x | x^((i))) - u_t (x))^2 = sigma_t^2 dot(gamma)_t^2 tr(C), $
and substituting gives
$ nabla dot.op u_t = gamma_t/(sigma_t^2 dot(gamma)_t) V_t + d (dot(sigma)_t)/(sigma_t), wide (d)/(d t) log p_t (x_t) = -d (dot(sigma)_t)/(sigma_t) - gamma_t/(sigma_t^2 dot(gamma)_t) V_t (x_t). $ <eq:div>
For the linear path these are $nabla dot.op u_t = (t V_t - d) \/ (1 - t)$ and $(d)/(d t) log p_t (x_t) = (d - t V_t (x_t)) \/ (1 - t)$. So the field contracts volume (density rises along the trajectory) unless the training targets at $x$ disagree strongly. $V_t$ is the velocity variance studied by #citet(<stablevelocity>).

*Acceleration.* For the variance-exploding path, #citet(<chen2023geometric>) (Proposition 3) show that the curvature of a trajectory is set by the rate of change of the denoiser. For the linear path, #citet(<rao2026geometry>) (Proposition A.1, Eq. 18) show, in our notation,
$ dot.double(x) = dot(m)/(1 - t). $ <eq:acc>
The proof differentiates $u_t = (m - x)/(1 - t)$ and uses $dot(x) = u$ and $(m - x)/(1 - t) = u$, which gives $dot.double(x) = (dot(m) - u)/(1 - t) + u/(1 - t)$. The same paper also splits $dot(m) = partial_t m + (nabla_x m) u$ but gives no closed form for $partial_t m$.

For a general Gaussian path one more identity is needed. Differentiating $x = alpha_t m + sigma_t hat(epsilon)$ along the trajectory and comparing with $dot(x) = dot(alpha)_t m + dot(sigma)_t hat(epsilon)$ from @eq:u leaves $alpha_t dot(m) + sigma_t dot(hat(epsilon)) = 0$, that is,
$ dot(hat(epsilon)) = -gamma_t dot(m). $ <eq:epsdot>
Differentiating @eq:u once more and using @eq:epsdot and $dot(alpha)_t - gamma_t dot(sigma)_t = sigma_t dot(gamma)_t$ gives
$ dot.double(x) = dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m). $ <eq:acc-general>
The first two terms are the acceleration of the conditional path $alpha_t x_1 + sigma_t epsilon$ with both guesses held fixed. They come from the schedule alone, vanish only when $alpha_t$ and $sigma_t$ are linear in $t$, and remain after the posterior has committed. The last term is the bending caused by the posterior: beyond what the schedule prescribes, a trajectory bends exactly when its denoised prediction moves. For the linear path $dot.double(alpha)_t = dot.double(sigma)_t = 0$ and $sigma_t dot(gamma)_t = 1 \/ (1 - t)$, which is @eq:acc. What follows is, to our knowledge, new: we compute $dot(m)$ exactly in terms of the posterior moments $C$ and $kappa$.

= Closed form for $dot(m)$ <sec:mdot>

#theorem[
Along any trajectory $dot(x) = u_t (x)$ of the field @eq:u of a Gaussian path $x_t = alpha_t x_1 + sigma_t epsilon$, for $t in (0, 1)$,
$ dot(m) = dot(gamma)_t (C hat(epsilon) - gamma_t kappa), wide "and hence" wide dot.double(x) = dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m). $ <eq:mdot>
For the linear path, $dot(m) = (C hat(epsilon))/(1 - t)^2 - (t kappa)/(1 - t)^3$ and $dot.double(x) = (C hat(epsilon))/(1 - t)^3 - (t kappa)/(1 - t)^4$.
] <thm:mdot>


The vector $hat(epsilon) = (x - alpha_t m)/sigma_t$ in @eq:mdot is the posterior mean of the noise, $hat(epsilon) = EE[epsilon | x_t = x]$, because $x = alpha_t x_1 + sigma_t epsilon$ and $m = EE[x_1 | x_t = x]$. By Tweedie's formula @efron2011tweedie it is also the scaled score: differentiating $p_t$ gives $nabla_x log p_t (x) = (alpha_t m - x)/sigma_t^2$, so $hat(epsilon) = -sigma_t nabla_x log p_t (x)$.

#proof(title: [Proof sketch])[
Write $w_i = e^(g_i) \/ sum_j e^(g_j)$ with $g_i = -||x - alpha_t x^((i))||^2 \/ 2 sigma_t^2$. Since the training points are constant, $dot(m) = sum_i dot(w)_i x^((i))$, and differentiating $log w_i$ in time gives the same softmax rule as for the spatial gradient, $dot(w)_i = w_i (dot(g)_i - sum_j w_j dot(g)_j)$. Subtracting $m$ times a zero sum yields
$ dot(m) = sum_i w_i (x^((i)) - m) dot(g)_i . $ <eq:covform>
Any part of $dot(g)_i$ that does not depend on $i$ contributes nothing to $dot(m)$, so we only need to track the $i$-dependent part of $dot(g)_i$.

The sum @eq:covform is a weighted covariance between $x^((i))$ and $dot(g)_i$, and a covariance treats the three kinds of terms differently; the one-dimensional version is $Cov (X, c + b X + a X^2) = 0 + b Var (X) + a Cov (X, X^2)$. So sorting $dot(g)_i$ by how it depends on $x^((i))$ is the same as sorting it by what each part becomes: the part without $x^((i))$ contributes nothing, the linear part $b dot.op x^((i))$ contributes $C b$, and the quadratic part $||x^((i))||^2$ contributes $2 C m + kappa$.

Computing $dot(g)_i$ gives the linear coefficient $b = (dot(alpha)_t x + alpha_t u)/sigma_t^2 - (2 alpha_t dot(sigma)_t x) / sigma_t^3$ and the quadratic coefficient $a = -gamma_t dot(gamma)_t$, hence $dot(m) = C (b + 2 a m) + a kappa$. Inserting $x = alpha_t m + sigma_t hat(epsilon)$ and @eq:u, the terms in $m$ cancel and $b + 2 a m = dot(gamma)_t hat(epsilon)$. The full computation is in @app:mdot.
]

#corollary(note: [Euler's error])[
Since $u = dot(x)$, Taylor's theorem gives $x(t + h) = x + h u + h^2/2 dot.double(x) + o(h^2)$. An Euler step keeps only $x + h u$, so by @eq:acc-general its local error is
$ x(t + h) - (x + h u) = h^2/2 (dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m)) + o(h^2). $ <eq:euler>
For the linear path the schedule term vanishes and the local error is $h^2 \/ (2(1 - t)) thin dot(m) + O(h^3)$.
] <cor:euler>

*Interpretation of the results.* The posterior-dependent part of the acceleration is determined by the spread $C$ and the asymmetry $kappa$ of the posterior. If one image has all the weight, then $C = 0$ and $kappa = 0$, so $dot(m) = 0$: only the schedule term $dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon)$ remains. For the linear path, the trajectory is a straight line and Euler is exact. That agrees with the zero-baseline result of #citet(<rao2026geometry>). If the posterior is symmetric about $m$, then $kappa = 0$ and only the $C hat(epsilon)$ term remains. For exactly two images with equal weights that term vanishes too, because then $C = Delta^2/4 e e^top$ with $e dot.op hat(epsilon) = 0$; the bending occurs next to the symmetric point, not at it.

= Local bounds <sec:bounds>

The weights $w_i$ decay exponentially with distance, so most images have negligible weight. We want to say this precisely, so that every sum can be split into an active and an inactive index set.

*Active set.* Let $i^star$ be the point with the largest weight, equivalently the largest $g_i$. For a fixed $delta in (0, 1)$ define
$ A = {i : g_i >= g_(i^star) - log N/delta}. $ <eq:active>

#lemma[$sum_(i in.not A) w_i <= delta$.] <lem:tail>

Geometrically, the active points are those almost as close to the rescaled point $x \/ alpha_t$ as the nearest point: $i in A$ if and only if
$ ||x\/alpha_t - x^((i))||^2 <= ||x\/alpha_t - x^((i^star))||^2 + 2 / gamma_t^2 log N/delta . $
This follows by writing $g_i$ in terms of $x \/ alpha_t$ (@app:active). At the clean endpoint, $gamma_t -> infinity$ as $t -> 1$, so the allowed slack shrinks to zero. Two diameters are used in what follows,
$ D_A = max_(i, j in A) ||x^((i)) - x^((j))|| wide "and" wide D = max_(i, j) ||x^((i)) - x^((j))||. $

#theorem[
For every $x$, $t in (0, 1)$ and $delta in (0, 1)$,
$ lambda_max (C) <= D_A^2/4 + delta D^2, $ <eq:lam>
$ ||kappa|| <= (D_A + delta D)^3 + delta D^3, $ <eq:kap>
$ ||dot(m)|| <= dot(gamma)_t [(D_A^2 /4 + delta D^2) ||hat(epsilon)|| + gamma_t ((D_A + delta D)^3 + delta D^3)], $ <eq:local>
and $||dot.double(x)|| <= ||dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon)|| + sigma_t dot(gamma)_t ||dot(m)||$.
] <thm:local>

The last inequality follows from @eq:acc-general and the triangle inequality. For the linear path its first term vanishes and it reads $||dot.double(x)|| = ||dot(m)|| \/ (1 - t)$.

The proofs are in @app:bounds. The bound depends on $C$ and $kappa$ depend on $x$ and $t$ only through the active set, which makes it a local, data-dependent statement. Inserting @eq:lam into the eigenvalue range @eq:eigrange gives
$ lambda_max (nabla_x u_t) <= gamma_t dot(gamma)_t (D_A^2/4 + delta D^2) + (dot(sigma)_t)/(sigma_t). $ <eq:lamjac>

*Tightness.* Take $x^((1)), x^((2))$ with $||x^((1)) - x^((2))|| = Delta$ and $x = alpha_t dot.op (x^((1)) + x^((2)))/2$. Both are equally far from $x \/ alpha_t$, so $w_1 = w_2 = 1/2$, both are active, and $D_A = Delta$. Along $e = (x^((1)) - x^((2)))/Delta$ we get $e^top C e = 1/2 (Delta/2)^2 + 1/2 (Delta/2)^2 = Delta^2/4$, so $D_A^2/4$ cannot be improved.

The bound @eq:kap is not tight, since in the two-point example $kappa = 0$ by symmetry.

= Curvature-adaptive step sizes <sec:sampler>

We know how much a trajectory bends ($dot.double(x) = dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m)$). Here we turn that into a rule for choosing the step size of a DDIM sampler.

*The DDIM step*. By @cor:euler, an Euler step $x + h u_t (x)$ has a local error of 
$
h^2/2 thin dot.double(x) + o(h^2) = h^2/2 (dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m)) + o(h^2).
$

The first two terms come from the schedule. They are the acceleration of the conditional path $alpha_t m + sigma_t hat(epsilon)$ with both estimates held fixed.  Unless the schedule is linear, these terms are nonzero even if the posterior has committed (i.e. $dot(m) = 0$). Deriving a step rule built on this error would shrink the step size where the data poses no difficulty. DDIM #citet(<song2021ddim>) avoids this. At any step $t_k$, the network output $v_k approx u_(t_k)(x_k)$ determines the estimate $hat(m)_k$ of the clean image and the estimate $hat(epsilon)_k$ of the noise applied. By @eq:u:

$
  x_k = alpha_(t_k) hat(m)_k + sigma_(t_k) hat(epsilon)_k, quad "and" v_k = dot(alpha)_(t_k) hat(m)_k + dot(sigma)_(t_k) hat(epsilon)_k
$

These are two linear equations with the solution:

$
  hat(m)_k = (dot(sigma)_t x_k - sigma_t v_k)/(alpha_t dot(sigma)_t - sigma_t dot(alpha)_t) , quad hat(epsilon)_k = (alpha_t v_k - dot(alpha)_t x_k)/(alpha_t dot(sigma)_t - sigma_t dot(alpha)_t)
$ <eq:estimates>

evaluated at $t := t_k$. The denominator equals $-sigma_t^2 dot(gamma)_t$ and is never zero. A DDIM step keeps both estimates fixed and performs the update:

$
  x_(k+1) = alpha_(t_(k+1)) hat(m)_k + sigma_(t_(k+1)) hat(epsilon)_k
$ <eq:ddim>

This is the exact solution of $dot(x) = u_t (x)$ when $m$ is frozen at $hat(m)_k$ (@app:ddim). For the linear path, $hat(m)_k = x_k + (1 - t_k) v_k$ and the DDIM step equals the Euler step $x_k + h_k v_k$.

*Its local error*. The DDIM step only errs because $m$ moves during the step:

$
  x(t+h) - x^"DDIM" = h^2 / 2 sigma_t dot(gamma)_t dot(m) + o(h^2).
$ <eq:ddim-error>

DDIM is the first-order member of the exponential integrators of #citet(<lu2022dpmsolverpp>), and @eq:ddim-error is the leading error term of their expansion (Eq. 9), written in $t$ instead of $log gamma_t$; @app:ddim gives a direct proof. Compared with @cor:euler, the schedule term is gone. For the linear path, $sigma_t dot(gamma)_t = 1 / (1-t)$ and this is the Euler error.

*Choosing the step size.* For a fixed step size $h$, the leading local error norm is proportional to $sigma_t dot(gamma)_t ||dot(m)||$. Following classical step-size control @hairer1993ode, we fix a tolerance $"tol" > 0$ and choose $h$ to satisfy
$ h^2/2 sigma_t dot(gamma)_t ||dot(m)|| = "tol" quad <==> quad h = sqrt((2 "tol")/(sigma_t dot(gamma)_t ||dot(m)||)), $ <eq:h>
so steps become small where the trajectory bends and large where it is straight, for each sample individually. Adaptive step sizes for diffusion sampling were proposed before by #citet(<jolicoeur2021gotta>); what is new here is the closed form of the quantity the rule measures.

*Estimating $dot(m)$ at no extra cost.* Calculating $||dot(m)||$ requires the training set, which a trained network does not expose. However, $hat(m)_k approx m_(t_k) (x_k)$ is available from @eq:estimates at no extra cost, since the sampler computes the network output $v_k$ anyway. Since $dot(m)$ is the rate of change of $m$ along the trajectory, the natural estimate is the difference of two consecutive predictions,
$ hat(dot(m))_k = (hat(m)_k - hat(m)_(k - 1))/h_(k - 1). $ <eq:est> #citet(<chen2023geometric>) (Section 4) use the same differences of consecutive denoiser outputs to derive second-order samplers.

#proposition[For the exact field and $t_k <= 1 - c$ for a fixed $c in (0, 1)$, $hat(dot(m))_k = dot(m) (t_(k - 1)) + O(h_(k - 1))$.] <prop:est>

The proof is in @app:est. The estimate describes time $t_(k - 1)$ but is used to choose the step at $t_k$. Since $dot(m) (t_k) = dot(m) (t_(k - 1)) + O(h)$, this one-step lag costs another $O(h)$ and does not change the order. For a trained network the estimate additionally carries the network's own error $v_theta - u$, which we can only measure empirically.

*Resulting sampler.* There is no previous prediction at $k = 0$, so $h_0$ is a fixed value. When the posterior has committed (one training image carries almost all the weight), $dot(m) -> 0$ and the rule would give $h -> infinity$, so $h_max$ and the remaining time prevent overshooting $t = 1$, while $h_min$ prevents the sampler from stalling at a sharp bend. To compare against a uniform grid, we tune $"tol"$ until the average step count equals that of the uniform grid. Each step costs one network evaluation, the same as plain Euler or DDIM. On the linear path, @alg:bendflow reduces to the Euler sampler used in @sec:experiments.

#place(top + center, float: true, clearance: 1.5em)[
#figure(
  kind: "algorithm",
  supplement: [Algorithm],
  caption: [Curvature-adaptive DDIM sampler initialized from the path's initial marginal. For the linear path, $alpha_0 = 0$ and $sigma_0 = 1$, so we can draw $x_0 tilde.op cal(N)(0, I)$ directly. The update reduces to the Euler step $x_(k + 1) = x_k + h_k v_k$.],
  {
    let ln(n, indent, body) = (text(size: 8pt)[#n:], pad(left: indent * 1.4em, body))
    grid(
      columns: (1.6em, 1fr),
      row-gutter: 0.55em,
      [], [*Require:* training set $cal(X)$, velocity model $v$, schedule $alpha_t, sigma_t$, tolerance $"tol"$, bounds $h_min <= h_max$, first step $h_0$],
      ..ln(1, 0, [Draw $x_1$ uniformly from $cal(X)$ and draw $epsilon_0 tilde.op cal(N)(0, I)$ independently]),
      ..ln(2, 0, [$x_0 <- alpha_0 x_1 + sigma_0 epsilon_0$, #h(1em) $t_0 <- 0$, #h(1em) $k <- 0$]),
      ..ln(3, 0, [*while* $t_k < 1$ *do*]),
      ..ln(4, 1, [$v_k <- v(x_k, t_k)$, then $hat(m)_k, hat(epsilon)_k$ from $x_k, v_k$ by @eq:estimates]),
      ..ln(5, 1, [*if* $k >= 1$ *then*]),
      ..ln(6, 2, [$hat(dot(m))_k <- (hat(m)_k - hat(m)_(k - 1)) \/ h_(k - 1)$]),
      ..ln(7, 2, [$h_k <- sqrt(2 "tol" \/ (sigma_(t_k) dot(gamma)_(t_k) ||hat(dot(m))_k||))$]),
      ..ln(8, 2, [clip $h_k$ to $[h_min, h_max]$]),
      ..ln(9, 1, [*end if*]),
      ..ln(10, 1, [$h_k <- min(h_k, 1 - t_k)$, #h(1em) $t_(k + 1) <- t_k + h_k$]),
      ..ln(11, 1, [$x_(k + 1) <- alpha_(t_(k + 1)) hat(m)_k + sigma_(t_(k + 1)) hat(epsilon)_k$, #h(1em) $k <- k + 1$]),
      ..ln(12, 0, [*end while*]),
      ..ln(13, 0, [*return* $x_k$]),
    )
  },
) <alg:bendflow>
]

= Experiments <sec:experiments>

All experiments use the training splits of MNIST and Fashion-MNIST ($N = "60,000"$, $d = 784$) and CIFAR-10 ($N = "50,000"$, $d = 3072$), scaled to $[-1, 1]$. Every number and figure below was produced by a script in `experiments/` of the repository, and the outputs are in `results/`. The main text shows CIFAR-10; the remaining figures and tables are in @app:figures.

== Exact field on image data <sec:exp-exact>
*Setup.* The exact field @eq:u is evaluated in double precision over all training images, along $8$ trajectories per dataset (uniform Euler, $100$ steps). A point is _undecided_ while the largest posterior weight is below $0.99$; afterwards $C$ and $kappa$ are nearly $0$ and relative errors say nothing.\
*Questions.* (i) Do @eq:jac, @eq:div and @eq:mdot hold numerically against finite differences? (ii) How loose are the bounds #ref(<eq:lam>)--#ref(<eq:local>) along real trajectories, and how large is the active set over time? (iii) Does $||dot(m)||$ peak when a trajectory chooses between nearby images?

*Identities.* At $5$ random undecided points per dataset, @eq:jac and @eq:mdot agree with automatic differentiation to a relative error below $10^(-12)$ and with central finite differences to below $10^(-6)$; the divergence @eq:div agrees with the trace of the autodiff Jacobian to $3 dot.op 10^(-15)$ (@tab:identities in @app:details).

*Bounds and active set.* With $delta = 0.01$, on $4$ trajectories per dataset and every $0.05$ in $t$, no bound is violated at any of the $240$ points. At the undecided points (@tab:bounds), the bound on $lambda_max (C)$ is within a factor $11$ to $19$ in the median, while the bounds on $||kappa||$ and $||dot(m)||$ are two to three orders of magnitude too large and get looser as $d$ and $D$ grow. Late in the trajectory the bound on $||dot(m)||$ is useless (@fig:exact): once the active set is a single image, $D_A = 0$ and only the $delta$ terms remain, which grow like $(1 - t)^(-3)$. The active set contains every image at $t <= 0.05$ and shrinks to a single image by $t = 0.25$ to $0.35$ on CIFAR-10, and more slowly on MNIST and Fashion-MNIST (@app:figures).

#figure(
  placement: top,
  caption: [Bound over true value at the undecided points of $4$ trajectories per dataset: median, with minimum and maximum in brackets. $D$ is the diameter of the training set.],
  table(
    columns: 6,
    align: (left, right, right, left, left, left),
    toprule,
    [Dataset], [$D$], [Points], [$lambda_max (C)$, @eq:lam], [$||kappa||$, @eq:kap], [$||dot(m)||$, @eq:local],
    midrule,
    [MNIST], [$33.8$], [$29$], [$14.5$ ($3.5$-$81$)], [$180$ ($51$-$439$)], [$350$ ($123$-$1113$)],
    [Fashion-MNIST], [$44.9$], [$32$], [$11.0$ ($4.7$-$81$)], [$289$ ($48$-$750$)], [$541$ ($98$-$2036$)],
    [CIFAR-10], [$106.8$], [$19$], [$19.2$ ($5.5$-$36$)], [$910$ ($86$-$2127$)], [$1387$ ($280$-$5189$)],
    bottomrule,
  ),
) <tab:bounds>

#figure(
  placement: top,
  image(results + "exact_field_cifar10/mdot_and_bound.pdf", width: 100%),
  caption: [Exact field on CIFAR-10: $||dot(m)||$ from @eq:mdot along four trajectories and the bound @eq:local. The active set along the same trajectories is in @fig:active-cifar.],
) <fig:exact>

*Where $||dot(m)||$ peaks.* The time of the largest $||dot(m)||$ is within $0.02$ of the first time at which one image carries more than half of the posterior weight, which we take as the time of commitment, on $7$ of $8$ trajectories on MNIST, $6$ of $8$ on CIFAR-10 and $4$ of $8$ on Fashion-MNIST. The other trajectories peak at $t <= 0.08$, while thousands of images still share the weight. This early peak is not a choice between images: at $t = 0$, @eq:mdot reduces to $dot(m) = C x$ with $C$ the covariance of the whole training set.

== Trained models: U-Net and DiT <sec:exp-networks>
*Setup.* The U-Net and DiT of Appendices #nref(<app:unet>) and #nref(<app:dit>), trained with conditional flow matching (@app:details), each follow $8$ trajectories from the same noise as in @sec:exp-exact. At every point we also evaluate the exact field at the same input. The top eigenvalue of the symmetric part of $nabla_x v_theta$ is computed by power iteration on $4$ trajectories.\
*Questions.* (i) Where and when does the denoised prediction of each network deviate from the exact $m$, i.e. where does the network smooth the data instead of memorising it? (ii) Does $||hat(dot(m))||$ of the trained networks peak where the exact field predicts? (iii) Do the two architectures differ in this respect?

*Denoiser.* The median RMS of $hat(x)_1 - m$ per pixel is at most $0.035$ at $t <= 0.1$ for all six networks, rises over the range in which the exact posterior commits, and reaches $0.23$ to $0.31$ at $t = 0.9$ (@fig:networks in @app:figures). No network sample copies the image on which the exact field ends from the same noise: on MNIST that image is never even the nearest one, and where it is (3 of 8 samples per network on Fashion-MNIST, 2 of 8 on CIFAR-10) the RMS distance is $0.13$ to $0.56$. The networks follow the exact field until it commits and smooth afterwards.

*Curvature.* The networks do not reproduce the sharp peak of the exact $||dot(m)||$ (@fig:mdot-networks). At undecided points their $||dot(m)||$ is a median $0.27$ to $0.80$ times the exact value at the same point, lowest on CIFAR-10. It is largest near $t = 0$ and decays to about $10$ at $t = 1$, while the exact value drops to $0$ after commitment; only on MNIST does the estimate @eq:est still peak inside the commitment range ($t = 0.17$ to $0.33$). The estimate follows the autodiff value closely. The Jacobian tells the same story: before commitment the networks' top eigenvalue is close to the upper end of @eq:eigrange for the exact field, but afterwards it stays at about $2$ to $14$, where the exact field contracts every direction (its upper end tends to $-1 \/ (1 - t)$). It lies above the exact upper end at $26$ to $42$ of the $28$ to $44$ decided points per network.

*Architectures.* The U-Net and the DiT behave almost identically in all three respects, although the DiT has four times as many parameters; the DiT is slightly closer to the exact $||dot(m)||$.

#figure(
  placement: top,
  image(results + "trained_networks_cifar10/mdot_networks.pdf", width: 100%),
  caption: [$||dot(m)||$ on CIFAR-10 along four trajectories of each network: the exact field from @eq:mdot along its own trajectory from the same noise, the network by automatic differentiation, and the network by the estimate @eq:est used in @alg:bendflow.],
) <fig:mdot-networks>

== Curvature-adaptive sampling <sec:exp-sampler>
*Setup.* We compare four Euler samplers at matched mean numbers of function evaluations (NFE) of $5$, $10$, $20$ and $40$, on $128$ fresh noise samples per dataset: uniform Euler; @alg:bendflow with the default settings $h_0 = 0.05$, $h_min = 0.005$, $h_max = 0.25$ ("config"); @alg:bendflow with $h_0$, $h_min$ and $h_max$ chosen from a small grid by the mean error on $32$ separate calibration samples ("tuned"); and a fixed schedule, one grid for all samples placed by the rule @eq:h applied to the mean of @eq:est over the calibration samples. For both BendFlow variants, $"tol"$ is the smallest value whose mean NFE is at most the target. The fixed schedule separates placing the steps by curvature from adapting them to each sample. It follows the idea of #citet(<chen2024trajectory>), who calibrate one schedule on a few samples from the shape of sampling trajectories. The error is the RMS per pixel to uniform Euler with $1000$ steps from the same noise; for the exact field, where a sample either ends on the reference's training image or on another one, we report how often it ends nearest to the same image. @app:details gives the details.

#figure(
  placement: top,
  caption: [Trained networks: change of the mean RMS error to the reference relative to uniform Euler at the same mean NFE, in percent, for tuned BendFlow / the fixed schedule ($128$ samples). The better of the two is in bold. Absolute errors of all four samplers are in @tab:sampler.],
  {
    set text(size: 9pt)
    table(
      columns: 6,
      align: (left, left, center, center, center, center),
      toprule,
      [Dataset], [Network], [NFE 5], [NFE 10], [NFE 20], [NFE 40],
      midrule,
      [MNIST], [U-Net], [$+3$ / $bold(-7)$], [$-3$ / $bold(-8)$], [$-9$ / $bold(-10)$], [$-9$ / $bold(-16)$],
      [MNIST], [DiT], [$+3$ / $bold(-5)$], [$-3$ / $bold(-6)$], [$bold(-9)$ / $-6$], [$bold(-15)$ / $-2$],
      [Fashion-MNIST], [U-Net], [$bold(-25)$ / $-12$], [$bold(-16)$ / $-8$], [$bold(-13)$ / $-12$], [$bold(-11)$ / $-9$],
      [Fashion-MNIST], [DiT], [$bold(-25)$ / $-11$], [$bold(-20)$ / $-12$], [$bold(-18)$ / $-16$], [$bold(-19)$ / $-16$],
      [CIFAR-10], [U-Net], [$bold(-29)$ / $-8$], [$bold(-27)$ / $-12$], [$bold(-19)$ / $-12$], [$bold(-17)$ / $-13$],
      [CIFAR-10], [DiT], [$bold(-30)$ / $-8$], [$bold(-28)$ / $-13$], [$bold(-24)$ / $-14$], [$bold(-21)$ / $-14$],
      bottomrule,
    )
  },
) <tab:sampler-summary>

*Trained networks.* On Fashion-MNIST and CIFAR-10, tuned BendFlow has the lowest error in all $16$ combinations of network and NFE, $11%$ to $30%$ below uniform Euler and $1%$ to $24%$ below the fixed schedule, with the largest gains at low NFE (@tab:sampler-summary). With the default settings it can be worse than uniform Euler (CIFAR-10, NFE $40$); there the calibration chooses $h_0 = 0.01$ and $h_min = 0.001$, which removes the loss. On MNIST the fixed schedule is best in $6$ of $8$ combinations, and tuned BendFlow is $3%$ worse than uniform Euler at NFE $5$.

*Exact field.* Here the fixed schedule ends on the reference's image as often as the best alternative or more often in all $12$ cases, most clearly at NFE $5$ (CIFAR-10: $38%$, against $12%$ for tuned BendFlow and $9%$ for uniform Euler); at NFE $40$ tuned BendFlow catches up (@tab:sampler-exact). A possible reason, which we have not tested, is that @eq:est uses the previous step and so detects the short peak at commitment one step late. Plots of the error against NFE and sample grids are in @app:figures.

= Related work <sec:related>

The connection between the Jacobian of the denoiser and the posterior covariance goes back to higher-order Tweedie formulas @efron2011tweedie @manor2024posterior; #citet(<manor2024posterior>) also relate the third posterior moment to spatial derivatives of the second, whereas @thm:mdot relates it to the derivative along a trajectory. #citet(<wan2024elucidating>) state the Jacobian identity for general schedules and analyse how trajectories are attracted to and absorbed by local clusters of the data. #citet(<lv2026uncertainty>) derive the covariance and divergence identities for the linear interpolant and use them for uncertainty quantification.

#citet(<rao2026geometry>) prove the acceleration identity @eq:acc and use the accumulated acceleration as an uncertainty proxy for failure detection, not for choosing step sizes. #citet(<milanfar2026lagrangian>) argues qualitatively that the denoiser Jacobian is the source of trajectory curvature and of small solver steps. Global Lipschitz bounds on the velocity field are given by #citet(<gao2024convergence>) and #citet(<benton2024error>); our bounds are local and depend on the active set.

= Limitations and conclusion <sec:conclusion>

The theoretical results concern the exact marginal field of a finite training set under the Gaussian paths specified in @sec:setup. Sampling with that field reproduces training images, so it serves as a reference for memorisation and not as a generator. The bound @eq:local is an upper bound and can be loose, in particular through the factor $||hat(epsilon)||$ and the bound on $kappa$. @prop:est covers the exact field only, and the step-size rule controls the local error; we do not prove a bound on the global error. The novelty claims rest on a search of the closest literature and not on a proof that the results are absent elsewhere. The experiments confirm the identities; the bound on $lambda_max (C)$ is within one to two orders of magnitude, but those on $||kappa||$ and $||dot(m)||$ are too loose to use quantitatively. The bend peaks at commitment on most, but not all, exact trajectories, and trained networks follow the exact field only until commitment. The step-size rule helps on trained networks, most at low NFE, but it needs tuned settings, and a fixed schedule built from the mean curvature profile captures much of the gain; on MNIST and the exact field it is as good or better. We measure the error to a fine-step reference rather than sample quality, with $128$ samples, one draw of noise and small networks.

#heading(numbering: none)[LLM Usage]
Claude (Anthropic) was used to verify the derivations, to search for related work, and it introduced the notation $C$ and $kappa$; $m$ is also the standard notation for the denoiser in the literature. Claude was also used to check the author-written Python code and to write all the tests and the testing scripts, which the author reviewed manually. The handwritten math derivations are available in the repository, and this LaTeX document was constructed from them.

#bibliography("paper.bib", title: [References], style: "apa")

// ===========================================================================
// Appendix
// ===========================================================================
#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])

= The marginal field <app:marginal>

We need to show that $u_t$ in @eq:u is the marginal field. The marginalization theorem states that if $u_t (x | z)$ generates $p_t (x | z)$ for every $z$, then the posterior average $u_t (x) = EE_(z tilde.op p(dot.op | x_t = x)) [u_t (x | z)]$ and the marginal path $p_t (x) = integral p_t (x | z) q(z) dif z$ satisfy the continuity equation $partial_t p_t + nabla dot.op (p_t u_t) = 0$, and thus $u_t (x)$ generates $p_t (x)$ @lipman2023flow.

For the training set as data distribution, $q = 1/N sum_i delta_(x^((i)))$ and $p_t (x | x^((i))) = cal(N)(x; alpha_t x^((i)), sigma_t^2 I)$. By Bayes' rule the posterior probability of image $i$ is proportional to $1/N cal(N)(x; alpha_t x^((i)), sigma_t^2 I)$; the factor $1/N$ and the common normalising constant of the Gaussians cancel, which leaves exactly $w_i$ in @eq:w. Plugging $u_t (x | x_1) = dot(alpha)_t x_1 + dot(sigma)_t (x - alpha_t x_1)/sigma_t $ into the theorem,
$ u_t (x) = EE_(x_1 tilde.op p(dot.op | x_t = x)) [u_t (x | x_1)] = sum_i w_i u_t ( x | x^((i))) = dot(alpha)_t m + dot(sigma)_t (x - alpha_t m )/sigma_t = dot(alpha)_t m + dot(sigma)_t hat(epsilon)$

= Symmetry and positive semi-definiteness of $C$ <app:psd>

*Symmetric.* $(a a^top)^top = (a^top)^top a^top = a a^top$, so $(x^((i)) - m)(x^((i)) - m)^top$ is symmetric, and we only add such matrices with weights, so $C$ is symmetric.

*Positive semi-definite.* For all $v in RR^d$,
$ v^top C v = sum_i underbrace(w_i, >= 0) underbrace((v^top (x^((i)) - m))^2, >= 0) >= 0 . $
$C$ is not positive definite in general: if one image has all the weight then $C = 0$.

= Proof of @prop:jac <app:jac>

Recall $m = sum_i w_i x^((i))$. To shorten the exponentials, let
$ g_i (x) = -(||x - alpha_t x^((i))||^2)/(2sigma_t^2) = -1/(2sigma_t^2) (x^top x - 2alpha_t (x^((i)))^top x + alpha_t^2 (x^((i)))^top x^((i))), wide w_i = e^(g_i)/(sum_j e^(g_j)) . $
Here $alpha_t$, $sigma_t$ and $x^((i))$ are constants, and $nabla_x (x^top x) = 2x$, $nabla_x (a^top x) = a$, so
$ nabla g_i (x) = -1/(2sigma_t^2) (2x - 2alpha_t x^((i))) = -(x - alpha_t x^((i)))/sigma_t^2 . $
We also need $nabla w_i$. Since $w_i > 0$, $log w_i = g_i - log(sum_j e^(g_j))$ and $nabla log w_i = (nabla w_i)/w_i$, so $(nabla w_i)/w_i = nabla g_i - nabla log(sum_j e^(g_j))$. Now
$ nabla log(sum_j e^(g_j)) = (nabla (sum_j e^(g_j)))/(sum_k e^(g_k)) = (sum_j e^(g_j) nabla g_j)/(sum_k e^(g_k)) = sum_j ((e^(g_j))/(sum_k e^(g_k))) nabla g_j = sum_j w_j nabla g_j , $
which we plug back in to get $nabla w_i = w_i (nabla g_i - sum_j w_j nabla g_j)$. Since $sum_j w_j = 1$ and $m = sum_j w_j x^((j))$,
$ nabla g_i - sum_j w_j nabla g_j &= (alpha_t x^((i)) - x)/sigma_t^2 - sum_j w_j ((alpha_t x^((j)) - x)/sigma_t^2) = 1/sigma_t^2 (alpha_t x^((i)) - x - sum_j w_j (alpha_t x^((j)) - x)) \
&= 1/sigma_t^2 (alpha_t x^((i)) - x - alpha_t m + x) = alpha_t/sigma_t^2 (x^((i)) - m). $
So $nabla w_i = alpha_t/sigma_t^2 w_i (x^((i)) - m)$, which we substitute in $nabla_x m = sum_i x^((i)) (nabla w_i)^top$ to get
$ nabla_x m = sum_i x^((i)) (alpha_t/sigma_t^2 w_i (x^((i)) - m))^top = alpha_t/sigma_t^2 sum_i w_i x^((i)) (x^((i)) - m)^top . $
Notice that the weighted sum of deviations from the mean is zero:
$ sum_i w_i (x^((i)) - m) = sum_i w_i x^((i)) - m sum_i w_i = m - m = 0 . $
Therefore $sum_i w_i m (x^((i)) - m)^top = 0$, and subtracting this zero matrix does not change the value, so
$ nabla_x m = alpha_t/sigma_t^2 sum_i w_i (x^((i)) - m)(x^((i)) - m)^top = alpha_t/sigma_t^2 C . $
Differentiating @eq:u-x in space and using $sigma_t dot(gamma)_t dot.op alpha_t \/ sigma_t^2 = gamma_t dot(gamma)_t$,
$ nabla_x u_t = (dot(sigma)_t)/(sigma_t) I + sigma_t dot(gamma)_t nabla_x m = (dot(sigma)_t)/(sigma_t) I + sigma_t dot(gamma)_t alpha_t/sigma_t^2 C = gamma_t dot(gamma)_t C + (dot(sigma)_t)/(sigma_t) I, $
which is @eq:jac. #h(1fr) $square$

= Eigenvalues of the Jacobian <app:eig>

This appendix proves the statements that follow @eq:eigrange.

*Eigenvalues.* By @app:psd, $C$ is symmetric and positive semi-definite, so $C = sum_k lambda_k q_k q_k^top$ with orthonormal eigenvectors $q_k$ and eigenvalues $lambda_k in [0, lambda_max (C)]$. By @eq:jac,

$
  nabla_x u_t q_k = (gamma_t dot(gamma)_t C + (dot(sigma)_t)/(sigma_t) I) q_k = underbrace((gamma_t dot(gamma)_t lambda_k + (dot(sigma)_t)/(sigma_t)), = thin mu_k) q_k = mu_k q_k ,
$
so $nabla_x u_t$ has the eigenvectors of $C$ and the eigenvalues $mu_k$. For $t in (0, 1)$ the map $lambda |-> mu$ is increasing because $gamma_t dot(gamma)_t > 0$, so every $mu_k$ lies between its values at $lambda = 0$ and $lambda = lambda_max (C)$, which is @eq:eigrange. $nabla_x u_t$ is symmetric, since it is a linear combination of $C$ and $I$.

*Contraction and separation.* Let $x(t)$ and $y(t)$ be two trajectories and $Delta = y - x$ their separation. With $J = nabla_x u_t (x)$, Taylor's theorem gives $dot(Delta) = u_t (y) - u_t (x) = J Delta + O(||Delta||^2)$. Hence
$ (d)/(d t) 1/2 ||Delta||^2 = Delta^top J Delta + O(||Delta||^3), wide mu_min ||Delta||^2 <= Delta^top J Delta <= mu_max ||Delta||^2 , $
where the second statement holds because $J$ is symmetric. For a fixed eigenvector $q_k$ of $J$ at time $t$, symmetry gives $q_k^top J = mu_k q_k^top$, so
$ (d)/(d t) (q_k dot.op Delta) = mu_k (q_k dot.op Delta) + O(||Delta||^2). $
To first order in $Delta$, the component of the separation along $q_k$ therefore grows at relative rate $mu_k$ if $mu_k > 0$ and shrinks at relative rate $abs(mu_k)$ if $mu_k < 0$. If all $mu_k < 0$, nearby trajectories approach each other in every direction.

*Collapse onto one image.* For $t < 1$ every weight $w_i$ is positive, so $C = 0$ holds exactly only if all training images coincide. The statement "$C = 0$" describes the limit in which one image $x^((i^star))$ carries all the weight. In that limit $m = x^((i^star))$, $C = 0$ and $kappa = 0$, so $dot(m) = 0$ by @eq:mdot, the Jacobian is $J = (dot(sigma)_t \/ sigma_t) I$, and by @eq:u-x the field is $u_t (x) = (dot(sigma)_t \/ sigma_t) x + sigma_t dot(gamma)_t x^((i^star))$. Its trajectories are
$ x(t) - alpha_t x^((i^star)) = sigma_t/sigma_(t_0) (x(t_0) - alpha_(t_0) x^((i^star))), $
as differentiating both sides shows: since $sigma_t dot(gamma)_t - dot(alpha)_t = -alpha_t dot(sigma)_t \/ sigma_t$, the left side has derivative $u_t (x(t)) - dot(alpha)_t x^((i^star)) = (dot(sigma)_t \/ sigma_t) (x(t) - alpha_t x^((i^star)))$, and so does the right side. Every trajectory therefore follows a conditional path $alpha_t x^((i^star)) + sigma_t c$ with a fixed $c$ and reaches $x^((i^star))$ at $t = 1$; for the linear path this is a straight line. The distance between two trajectories is multiplied by $sigma_t \/ sigma_(t_0)$, so its instantaneous relative rate of change is $dot(sigma)_t \/ sigma_t$. On the linear path the distance shrinks at relative rate $1 \/ (1 - t)$.

*When trajectories separate.* For $t in (0, 1)$,
$ mu > 0 <==> gamma_t dot(gamma)_t lambda + (dot(sigma)_t)/(sigma_t) > 0 <==> lambda > -(dot(sigma)_t)/(sigma_t gamma_t dot(gamma)_t) = tau_t , $
so some direction separates if and only if $lambda_max (C) > tau_t$. Two upper bounds on $lambda_max (C)$ show what this requires. First, as in the proof of @eq:lam in @app:bounds, a weighted variance is at most the weighted mean squared deviation from any point; with the point $x^((i^star))$ of largest weight and a unit vector $v$,
$ v^top C v <= sum_i w_i (v dot.op (x^((i)) - x^((i^star))))^2 <= sum_(i != i^star) w_i D^2 = (1 - w_(i^star)) D^2 . $
Second, @eq:lam gives $lambda_max (C) <= D_A^2/4 + delta D^2$. For $D > 0$, separation in some direction therefore requires both
$ 1 - w_(i^star) > tau_t/D^2 wide "and" wide D_A^2/4 + delta D^2 > tau_t . $
When $tau_t > 0$, the first condition requires the posterior mass outside the most likely image to exceed a positive threshold. When $tau_t > delta D^2$, the second condition requires that $D_A > 2 sqrt(tau_t - delta D^2)$. On the linear path this threshold approaches $(2(1 - t))/sqrt(t)$ as $delta -> 0$.

= Proof of @thm:mdot <app:mdot>

Fix a trajectory $x = x_t$ with $dot(x) = u_t (x)$ and write a dot for the total derivative $(d)/(d t)$ along it. As before, $w_i = e^(g_i) \/ sum_j e^(g_j)$ and $g_i = -(||x - alpha_t x^((i))||^2)/(2 sigma_t^2)$. Since the training points are constant, $dot(m) = sum_i dot(w)_i x^((i))$, and differentiating $log w_i = g_i - log sum_j e^(g_j)$ in time gives the same softmax rule as for the spatial gradient,
$ dot(w)_i = w_i (dot(g)_i - sum_j w_j dot(g)_j). $
Denote $overline(g) = sum_j w_j dot(g)_j$; hence $dot(m) = sum_i w_i x^((i)) (dot(g)_i - overline(g))$. Since $sum_i w_i (dot(g)_i - overline(g)) = 0$, we may subtract $m$ times this zero:
$ dot(m) = sum_i w_i (x^((i)) - m)(dot(g)_i - overline(g)) = sum_i w_i (x^((i)) - m) dot(g)_i , $
where the last step uses $sum_i w_i (x^((i)) - m) = 0$. Any part of $dot(g)_i$ that does not depend on $i$ contributes nothing to $dot(m)$, so we only need to track the $i$-dependent part of $dot(g)_i$.

Recall $(d)/(d t) 1/(2 sigma_t^2) = -(dot(sigma)_t)/(sigma_t^3)$ and $(d)/(d t) (x - alpha_t x^((i))) = u - dot(alpha)_t x^((i))$. Then
$ dot(g)_i &= -((x - alpha_t x^((i))) dot.op (u - dot(alpha)_t x^((i))))/sigma_t^2 + (dot(sigma)_t ||x - alpha_t x^((i))||^2)/sigma_t^3 \
&= (dot(alpha)_t x dot.op x^((i)) + alpha_t u dot.op x^((i)) - alpha_t dot(alpha)_t ||x^((i))||^2)/sigma_t^2 + (dot(sigma)_t (alpha_t^2 ||x^((i))||^2 - 2 alpha_t x dot.op x^((i))))/sigma_t^3 + underbrace([-(x dot.op u)/sigma_t^2 + (dot(sigma)_t ||x||^2)/sigma_t^3], "does not depend on" i). $
Collecting the part linear in $x^((i))$ and the quadratic part,
$ dot(g)_i = x^((i)) dot.op underbrace([(dot(alpha)_t x + alpha_t u)/sigma_t^2 - (2 alpha_t dot(sigma)_t x)/sigma_t^3], =: thin b) + ||x^((i))||^2 underbrace([-(alpha_t dot(alpha)_t)/sigma_t^2 + (alpha_t^2 dot(sigma)_t)/sigma_t^3], =: thin a) + [-(x dot.op u)/sigma_t^2 + (dot(sigma)_t ||x||^2)/sigma_t^3]. $
Putting $a$ over the common denominator $sigma_t^3$ and factoring out $alpha_t \/ sigma_t$,
$ a = (alpha_t (alpha_t dot(sigma)_t - dot(alpha)_t sigma_t))/sigma_t^3 = -alpha_t/sigma_t dot.op (dot(alpha)_t sigma_t - alpha_t dot(sigma)_t)/sigma_t^2 = -gamma_t dot(gamma)_t . $
Substituting the above into $dot(m)$:
$ dot(m) = (sum_i w_i (x^((i)) - m)(x^((i)))^top) b + a sum_i w_i (x^((i)) - m) ||x^((i))||^2 . $
For the first sum, $(x^((i)))^top = (x^((i)) - m)^top + m^top$, where the $m^top$ part gives $(sum_i w_i (x^((i)) - m)) m^top = 0$, so the first sum equals $C$. For the second sum, let $z_i = x^((i)) - m$, so that $sum_i w_i z_i = 0$ and $||x^((i))||^2 = ||m||^2 + 2 z_i^top m + ||z_i||^2$. Then
$ sum_i w_i z_i ||x^((i))||^2 = underbrace((sum_i w_i z_i), = 0) ||m||^2 + 2 underbrace(sum_i w_i z_i z_i^top, = C) m + underbrace(sum_i w_i ||z_i||^2 z_i, = kappa) = 2 C m + kappa . $
Therefore
$ dot(m) = C b + a (2 C m + kappa) = C (b + 2 a m) + a kappa . $
It remains to simplify $b + 2 a m$. Inserting $x = alpha_t m + sigma_t hat(epsilon)$ and $u = dot(alpha)_t m + dot(sigma)_t hat(epsilon)$ from @eq:u into $b$,
$ b &= (2 alpha_t dot(alpha)_t m + (dot(alpha)_t sigma_t + alpha_t dot(sigma)_t) hat(epsilon))/sigma_t^2 - (2 alpha_t^2 dot(sigma)_t m + 2 alpha_t sigma_t dot(sigma)_t hat(epsilon))/sigma_t^3 \
&= (2 alpha_t (dot(alpha)_t sigma_t - alpha_t dot(sigma)_t))/sigma_t^3 m + (dot(alpha)_t sigma_t - alpha_t dot(sigma)_t)/sigma_t^2 hat(epsilon) = 2 gamma_t dot(gamma)_t m + dot(gamma)_t hat(epsilon). $
Since $2 a m = -2 gamma_t dot(gamma)_t m$, the terms in $m$ cancel and $b + 2 a m = dot(gamma)_t hat(epsilon)$. Hence
$ dot(m) = dot(gamma)_t C hat(epsilon) - gamma_t dot(gamma)_t kappa = dot(gamma)_t (C hat(epsilon) - gamma_t kappa), $
which is the first formula in @eq:mdot.

For the second formula, differentiate $x = alpha_t m + sigma_t hat(epsilon)$ along the trajectory and compare with @eq:u:
$ dot(x) &= dot(alpha)_t m + alpha_t dot(m) + dot(sigma)_t hat(epsilon) + sigma_t dot(hat(epsilon)) \
"and" quad dot(x) &= dot(alpha)_t m + dot(sigma)_t hat(epsilon). $
Subtracting leaves $alpha_t dot(m) + sigma_t dot(hat(epsilon)) = 0$, that is, $dot(hat(epsilon)) = -gamma_t dot(m)$, which is @eq:epsdot. Therefore
$ dot.double(x) &= (d)/(d t) (dot(alpha)_t m + dot(sigma)_t hat(epsilon)) = dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + (dot(alpha)_t - gamma_t dot(sigma)_t) dot(m) \
&= dot.double(alpha)_t m + dot.double(sigma)_t hat(epsilon) + sigma_t dot(gamma)_t dot(m), $
using $dot(alpha)_t - gamma_t dot(sigma)_t = (dot(alpha)_t sigma_t - alpha_t dot(sigma)_t) \/ sigma_t = sigma_t dot(gamma)_t$. This is @eq:acc-general, and inserting the first formula for $dot(m)$ gives the second formula in @eq:mdot. #h(1fr) $square$

= Geometric form of the active set <app:active>

For $t in (0, 1)$ we can write
$ g_i = -(||x - alpha_t x^((i))||^2)/(2sigma_t^2) = - gamma_t^2 / 2 ||x \/ alpha_t - x^((i))||^2 . $
So the largest $g_i$ belongs to the training image nearest to $x \/ alpha_t$, which is $x^((i^star))$. The condition $g_i >= g_(i^star) - log N/delta$ in @eq:active reads
$ - gamma_t^2 / 2 ||x \/ alpha_t - x^((i))||^2 >= - gamma_t^2 / 2 ||x \/ alpha_t - x^((i^star))||^2 - log N/delta . $
Multiplying by the negative number $-2 / gamma_t^2$ reverses the inequality and gives
$ ||x\/alpha_t - x^((i))||^2 <= ||x\/ alpha_t - x^((i^star))||^2 + 2 / gamma_t^2 log N/delta . $
Every step is an equivalence, so $i in A$ if and only if this holds. At the clean endpoint, $alpha_t -> 1$ and $sigma_t -> 0$, so the slack $2 / gamma_t^2 log N/delta$ tends to $0$ as $t -> 1$. At a pure-noise start with $alpha_0 = 0$, all logits $g_i$ are equal and every image is active.

= Proofs of the local bounds <app:bounds>

#proof(title: [Proof of @lem:tail])[
From $w_i = e^(g_i) \/ sum_j e^(g_j)$ we get $w_i/w_(i^star) = e^(g_i - g_(i^star))$. For $i in.not A$, $g_i - g_(i^star) < -log N/delta$, so $w_i < w_(i^star) dot.op delta/N <= delta/N$. There are at most $N$ such points, so their total weight is at most $N dot.op delta/N = delta$.
]

#proof(title: [Proof of @eq:lam])[
For a fixed unit vector $v in RR^d$, $v^top C v = sum_i w_i (v dot.op (x^((i)) - m))^2$, which is the weighted variance of the numbers $v dot.op x^((i))$. A variance is the smallest mean squared deviation, so for any point $c in RR^d$,
$ sum_i w_i (v dot.op (x^((i)) - m))^2 <= sum_i w_i (v dot.op (x^((i)) - c))^2, $
because the right-hand side equals the left-hand side plus $(v dot.op (m - c))^2$. Choice of $c$: among the active images, let $x^((a))$ have the smallest value of $v dot.op x^((i))$ and $x^((b))$ the largest, and take $c = 1/2 (x^((a)) + x^((b)))$, so every active image is within half the active spread of $c$.

_Case 1_ ($i in A$): $v dot.op x^((i)) in [v dot.op x^((a)), v dot.op x^((b))]$, so $abs(v dot.op (x^((i)) - c)) <= 1/2 abs(v dot.op (x^((b)) - x^((a)))) <= D_A/2$.

_Case 2_ ($i in.not A$): $abs(v dot.op (x^((i)) - c)) <= ||x^((i)) - c|| <= 1/2 ||x^((i)) - x^((a))|| + 1/2 ||x^((i)) - x^((b))|| <= D$.

Using $sum_(i in A) w_i <= 1$ and $sum_(i in.not A) w_i <= delta$,
$ v^top C v <= sum_(i in A) w_i D_A^2/4 + sum_(i in.not A) w_i D^2 <= D_A^2/4 + delta D^2 . $
This holds for every unit $v$, and $lambda_max (C) = max_(||v|| = 1) v^top C v$.
]

#proof(title: [Proof of @eq:kap])[
Write $z_i = x^((i)) - m$. By the triangle inequality, $||kappa|| = norm(sum_i w_i ||z_i||^2 z_i) <= sum_i w_i ||z_i||^3$, so we need to know how far each image is from $m$. Since $sum_j w_j = 1$, we can write $x^((i)) - m = sum_j w_j (x^((i)) - x^((j)))$. For $i in.not A$: $||z_i|| <= sum_j w_j D = D$. For $i in A$: $||z_i|| <= sum_(j in A) w_j D_A + sum_(j in.not A) w_j D <= D_A + delta D$. Splitting the sum over $i$ in the same way,
$ ||kappa|| <= sum_(i in A) w_i (D_A + delta D)^3 + sum_(i in.not A) w_i D^3 <= (D_A + delta D)^3 + delta D^3 . $
]

#proof(title: [Proof of @eq:local])[
Since $dot(gamma)_t > 0$, the triangle inequality and $||C hat(epsilon)|| <= lambda_max (C) ||hat(epsilon)||$ in @eq:mdot give
$
  ||dot(m)|| &= dot(gamma)_t ||C hat(epsilon) - gamma_t kappa|| \
  &<= dot(gamma)_t (lambda_max (C) ||hat(epsilon)|| + gamma_t ||kappa||),
$
and inserting the upper bounds @eq:lam and @eq:kap gives @eq:local.
]

= The DDIM step <app:ddim>

Let $x(s)$ be the exact trajectory with $x(t) = x$, and let $M(s) = m_s (x(s))$ and $E(s) = hat(epsilon)_s (x(s))$ be the two posterior means along it, so that $x(s) = alpha_s M(s) + sigma_s E(s)$ for all $s$. The DDIM step from $x$ is $x^"DDIM" = alpha_(t + h) M(t) + sigma_(t + h) E(t)$.

*Exact when $m$ is frozen.* Replacing $m$ by a fixed vector $hat(m)$ in @eq:u-x gives the linear equation $dot(x) = (dot(sigma)_s \/ sigma_s) x + sigma_s dot(gamma)_s hat(m)$. For every fixed $c$, the curve $x(s) = alpha_s hat(m) + sigma_s c$ solves it: its derivative is $dot(alpha)_s hat(m) + dot(sigma)_s c$, and the right-hand side is
$ (dot(sigma)_s)/(sigma_s) (alpha_s hat(m) + sigma_s c) + sigma_s dot(gamma)_s hat(m) = dot(sigma)_s c + ((alpha_s dot(sigma)_s)/(sigma_s) + sigma_s dot(gamma)_s) hat(m) = dot(sigma)_s c + dot(alpha)_s hat(m), $
because $sigma_s dot(gamma)_s = dot(alpha)_s - alpha_s dot(sigma)_s \/ sigma_s$. The curve passes through $x$ at $s = t$ exactly when $c = (x - alpha_t hat(m)) \/ sigma_t = hat(epsilon)$, and its value at $s = t + h$ is the DDIM step @eq:ddim.

*Local error.* Subtracting the DDIM step from $x(t + h) = alpha_(t + h) M(t + h) + sigma_(t + h) E(t + h)$ gives
$ x(t + h) - x^"DDIM" = alpha_(t + h) (M(t + h) - M(t)) + sigma_(t + h) (E(t + h) - E(t)). $
Expanding $alpha$, $sigma$, $M$ and $E$ around $t$ to second order, with all derivatives at $t$,
$ x(t + h) - x^"DDIM" = h (alpha_t dot(M) + sigma_t dot(E)) + h^2 (dot(alpha)_t dot(M) + dot(sigma)_t dot(E)) + h^2/2 (alpha_t dot.double(M) + sigma_t dot.double(E)) + o(h^2). $
By @eq:epsdot, $alpha_s dot(M) + sigma_s dot(E) = 0$ for every $s$, so the first bracket vanishes, and differentiating this identity gives $alpha_t dot.double(M) + sigma_t dot.double(E) = -(dot(alpha)_t dot(M) + dot(sigma)_t dot(E))$. Hence, using $dot(E) = -gamma_t dot(M)$ once more,
$ x(t + h) - x^"DDIM" = h^2/2 (dot(alpha)_t dot(M) + dot(sigma)_t dot(E)) + o(h^2) = h^2/2 (dot(alpha)_t - gamma_t dot(sigma)_t) dot(m) + o(h^2) = h^2/2 sigma_t dot(gamma)_t dot(m) + o(h^2), $
which is @eq:ddim-error. #h(1fr) $square$

= Proof of @prop:est <app:est>

Let $h = h_(k - 1)$, let $x(t)$ be the exact trajectory through $x_(k - 1)$, and let $M(t) = m_t (x(t))$, so that $dot(M) = dot(m)$ and $hat(m)_(k-1) = M(t_(k - 1))$. The estimate differs from $dot(M) (t_(k - 1))$ for two reasons.

_(1)_ $x_k$ was produced by a DDIM step, so it is not necessarily on the exact trajectory. By the local error formula @eq:ddim-error, $x_k = x(t_k) + e$ with $||e|| = O(h^2)$. Expanding $m$ in space around $x(t_k)$,
$ m_(t_k) (x_k) = M(t_k) + nabla_x m thick e + O(||e||^2), $
where $nabla_x m = alpha_t / sigma_t^2 C$ is bounded for $t <= 1 - c$, because $lambda_max (C) <= D^2$ for a finite training set. Hence $m_(t_k) (x_k) = M(t_k) + O(h^2)$.

_(2)_ A difference quotient is not a derivative, and Taylor in time gives $M(t_k) = M(t_(k - 1)) + h dot(M) (t_(k - 1)) + O(h^2)$.

Combining both,
$ hat(dot(m))_k = (M(t_k) + O(h^2) - M(t_(k - 1)))/h = (h dot(M) (t_(k - 1)) + O(h^2))/h = dot(m) (t_(k - 1)) + O(h). #h(2em) square $

= U-Net architecture <app:unet>

This appendix and the next specify the two networks of @sec:experiments completely. All numbers are the values in `configs/networks.toml` of the repository.

*Shared conventions.* Both networks are velocity models $v_theta (x, t)$: they receive a noisy image and a time and return a velocity of the same shape. The flat vector $x in RR^d$ of @sec:setup is reshaped to $c times S times S$ with $d = c S^2$, where $c = 1$, $S = 28$ for MNIST and Fashion-MNIST and $c = 3$, $S = 32$ for CIFAR-10. Pixel values are scaled to $[-1, 1]$.

Both networks work in pixel space, without an autoencoder or a latent space @rombach2022ldm, for two reasons. First, the images are small enough that pixel space is affordable: with patch size $p = 2$ a $28 times 28$ image gives $196$ tokens. Second, and more importantly, the exact field @eq:u is defined by the training images themselves, and its trajectories end on them. A network trained on latents $E(x^((i)))$ would have to be compared with the exact field of the encoded training set, and the decoder would add a second source of error that cannot be separated from the error of the flow.

Both networks are unconditional and take the time $t$ as their only side input, so there is no class or text embedding, no cross-attention and no in-context conditioning token. Neither of them uses dropout.

*Time embedding.* The scalar $t in [0, 1]$ is first expanded by a fixed sinusoidal embedding of even width $d_e$. With $s = 1000 t$ and $omega_j = 10000^(-2j \/ d_e)$ for $j = 0, dots, d_e/2 - 1$,
$ phi.alt(t) = [cos(omega_0 s), dots, cos(omega_(d_e \/ 2 - 1) s), sin(omega_0 s), dots, sin(omega_(d_e \/ 2 - 1) s)] in RR^(d_e). $
This is the usual embedding up to the order of the entries. A two-layer MLP then gives the time vector
$ e = W_2 "SiLU"(W_1 phi.alt(t) + b_1) + b_2 in RR^(d_e), wide "SiLU"(z) = z/(1 + e^(-z)), $
with both layers of width $d_e$. The U-Net uses $d_e = 128$, and the DiT of @app:dit uses the same construction with $d_e$ equal to its token width.

*Residual block.* The U-Net is built from residual blocks. A block with $c_"in"$ input and $c_"out"$ output channels maps a feature map $z$ and the time vector $e$ to
$ a &= "Conv"_(3 times 3) ("SiLU"("GN"(z))) + W_e e + b_e, \
z' &= "Conv"_(3 times 3) ("SiLU"("GN"(a))) + "skip"(z). $
Here $"GN"$ is group normalisation @wu2018groupnorm with $8$ groups, $epsilon.alt = 10^(-5)$ and a learned scale and shift per channel. Both convolutions have stride $1$ and zero padding $1$, so the resolution is kept. The output of the linear layer $W_e in RR^(c_"out" times d_e)$ is added to every spatial position, which is how the block is told the noise level, and $"skip"$ is the identity if $c_"in" = c_"out"$ and a $1 times 1$ convolution otherwise. SiLU is the only activation function in the U-Net.

*Self-attention block.* One self-attention block sits at the lowest resolution, where the feature map has $c_z = 64$ channels. After group normalisation the $S' times S'$ positions are read as $S'^2$ tokens $Z in RR^(S'^2 times c_z)$ ($49$ tokens for MNIST, $64$ for CIFAR-10), and
$ Q = Z W_Q, quad K = Z W_K, quad V = Z W_V, wide "Attn"(Z) = "softmax"((Q K^top)/sqrt(c_z)) V W_O, $
with the softmax taken over each row. This is scaled dot-product self-attention @vaswani2017attention with a single head, key dimension $c_z$, no bias terms and no mask, so every position attends to every other position. The result is reshaped to a feature map and added to the input of the block. The block has no MLP after the attention and no positional embedding, since the convolutions around it carry the position.

*Layout.* @tab:unet lists the network for $c = 1$, $S = 28$. The base width is $32$ channels and the three resolutions use $32$, $64$ and $64$ channels (multipliers $1, 2, 2$). The contracting path has one residual block per resolution and the expanding path has two. Downsampling is a $3 times 3$ convolution with stride $2$ and padding $1$, which halves the resolution ($28 -> 14 -> 7$, and $32 -> 16 -> 8$ for CIFAR-10). Upsampling is nearest-neighbour interpolation to the resolution of the next skip connection followed by a $3 times 3$ convolution. Skip connections concatenate along the channel axis: the output of the first convolution, of every residual block of the contracting path and of every downsampling convolution is stored, and each residual block of the expanding path receives its input concatenated with the most recently stored map that has not been used yet. This is why the expanding blocks in @tab:unet have $128$, $96$ or $64$ input channels.

#figure(
  caption: [The U-Net for a $1 times 28 times 28$ input. "Res" is the residual block and "Attn" the self-attention block defined above. For the expanding path the input channels include the concatenated skip connection.],
  table(
    columns: 4,
    align: left,
    toprule,
    [Stage], [Resolution], [Operation], [Channels],
    midrule,
    [Input], [$28$], [Conv $3 times 3$], [$1 -> 32$],
    [Down 1], [$28$], [Res], [$32 -> 32$],
    [], [$28 -> 14$], [Conv $3 times 3$, stride $2$], [$32 -> 32$],
    [Down 2], [$14$], [Res], [$32 -> 64$],
    [], [$14 -> 7$], [Conv $3 times 3$, stride $2$], [$64 -> 64$],
    [Down 3], [$7$], [Res], [$64 -> 64$],
    [Middle], [$7$], [Res, Attn, Res], [$64 -> 64$],
    [Up 3], [$7$], [Res, Res], [$128 -> 64$, $128 -> 64$],
    [], [$7 -> 14$], [nearest upsampling, Conv $3 times 3$], [$64 -> 64$],
    [Up 2], [$14$], [Res, Res], [$128 -> 64$, $96 -> 64$],
    [], [$14 -> 28$], [nearest upsampling, Conv $3 times 3$], [$64 -> 64$],
    [Up 1], [$28$], [Res, Res], [$96 -> 32$, $64 -> 32$],
    [Output], [$28$], [GN, SiLU, Conv $3 times 3$], [$32 -> 1$],
    bottomrule,
  ),
) <tab:unet>

*Initialisation and size.* The weights and biases of every linear and convolutional layer are drawn uniformly from $[-1/sqrt(n_"in"), 1/sqrt(n_"in")]$, where $n_"in"$ is the number of inputs of one output unit. The only exception is the last convolution, whose weight and bias start at zero, so the untrained network returns $v_theta = 0$. The network has $11$ residual blocks, one attention block and $"1,074,529"$ parameters for MNIST ($"1,075,683"$ for CIFAR-10).

*Relation to the original U-Net.* The contracting path, the expanding path and the skip connections that let detail bypass the bottleneck are those of #citet(<ronneberger2015unet>). The rest follows the U-Nets used for diffusion models @ho2020ddpm: residual blocks with group normalisation and SiLU in place of plain convolutions with ReLU, strided convolutions in place of max pooling, interpolation followed by a convolution in place of transposed convolutions, padded convolutions so that no cropping is needed, a time input in every block, and self-attention at the lowest resolution.

= Diffusion Transformer architecture <app:dit>

The second network is a Diffusion Transformer (DiT) @peebles2023dit. The conventions, the time embedding and the SiLU activation are those of @app:unet. @tab:dit collects the sizes.

*Patchify and token embedding.* The image $x in RR^(c times S times S)$ is cut into non-overlapping $p times p$ patches with $p = 2$. Each patch is flattened to a vector of $p^2 c$ numbers and mapped by one linear layer with bias to a token of width $d_"tok" = 192$. This gives $n = (S \/ p)^2$ tokens: $196$ for $S = 28$ and $256$ for $S = 32$. No autoencoder precedes this step, so the patches are patches of pixels.

Doubling the patch size to $p = 4$ would give four times fewer tokens ($49$ for $S = 28$) and sixteen times fewer attention pairs per layer, but every token would then have to produce $16$ pixels through the single linear layer at the output.

*Positional embedding.* Self-attention is permutation equivariant, so the position of a patch has to enter the network somewhere. We use a two-dimensional rotary positional embedding (RoPE) @su2021roformer. Nothing is added to the tokens,
$ Z^((0)) = "patchify"(x) W_"emb" + b_"emb" in RR^(n times d_"tok"), $
and the position enters inside every attention layer instead, through a rotation of the queries and keys that is defined in the paragraph on attention below. RoPE has no parameters, and we do not use a class token.

*Modulation vector.* The time embedding of @app:unet with $d_e = d_"tok" = 192$ gives the modulation vector $e in RR^192$. It is computed once per evaluation and read by every block.

*Multi-head self-attention.* All attention in the DiT is multi-head scaled dot-product self-attention @vaswani2017attention over the $n$ image tokens, with $n_h = 6$ heads of dimension $d_h = d_"tok" \/ n_h = 32$. For tokens $Z in RR^(n times d_"tok")$ and head $j$,
$ Q_j = Z W_Q^j + b_Q^j, wide K_j = Z W_K^j + b_K^j, wide V_j = Z W_V^j + b_V^j, $
with $W_Q^j, W_K^j, W_V^j in RR^(d_"tok" times d_h)$.

This is where the positions enter. Let token $i$ lie in row $r_i$ and column $c_i$ of the $(S \/ p) times (S \/ p)$ grid of patches, both counted from $0$. Row $i$ of $Q_j$ and row $i$ of $K_j$ are each split into $d_h \/ 2 = 16$ consecutive pairs of entries, and every pair $(a, b)$ is rotated in its plane,
$ (a, b) |-> (a cos theta - b sin theta, thick a sin theta + b cos theta). $
With $omega_k = 10000^(-k \/ (d_h \/ 4))$ for $k = 0, dots, d_h/4 - 1$, pair $k$ is rotated by $theta = r_i omega_k$ and pair $d_h/4 + k$ by $theta = c_i omega_k$, so one half of every head encodes the row and the other half the column. The values $V_j$ are not rotated. Writing $tilde(Q)_j$ and $tilde(K)_j$ for the rotated matrices,
$ A_j = "softmax"((tilde(Q)_j tilde(K)_j^top)/sqrt(d_h)) V_j in RR^(n times d_h), $
with the softmax over each row. Rotations of the same plane compose by adding their angles, so the inner product of the rotated query of token $i$ and the rotated key of token $i'$ depends on the positions only through $r_i - r_(i')$ and $c_i - c_(i')$. In other words, the attention weights see relative position. The heads are concatenated and mapped by an output layer,
$ "MHSA"(Z) = [A_1, dots, A_(n_h)] W_O + b_O in RR^(n times d_"tok"). $
There is no mask, so every token attends to all $n$ tokens, at a cost of $n^2$ pairs per head. Since the network is unconditional, there is no cross-attention.

*MLP.* The MLP of a block acts on each token separately and has two layers with one hidden layer of width $4 d_"tok" = 768$:
$ "MLP"(z) = W_b "GELU"(W_a z + b_a) + b_b, wide W_a in RR^(768 times 192), thick W_b in RR^(192 times 768). $
GELU @hendrycks2016gelu is used in its $tanh$ approximation, $"GELU"(z) approx z/2 (1 + tanh(sqrt(2\/pi) (z + 0.044715 z^3)))$.

*One block.* A standard pre-norm transformer block is
$ Z <- Z + "MHSA"("LN"(Z)), wide Z <- Z + "MLP"("LN"(Z)), $
with layer normalisation $"LN"$ @ba2016layernorm applied to each token. The DiT block keeps this structure and adds two things to each of the two residual branches, both driven by the modulation vector. One linear layer per block produces six vectors of width $d_"tok"$,
$ (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2) = W_"mod" "SiLU"(e) + b_"mod" in RR^(6 d_"tok"), $
and the block computes
$ Z &<- Z + alpha_1 dot.o "MHSA"((1 + gamma_1) dot.o "LN"(Z) + beta_1), \
Z &<- Z + alpha_2 dot.o "MLP"((1 + gamma_2) dot.o "LN"(Z) + beta_2), $
where $dot.o$ multiplies each token entrywise by the same vector.

_(1) Scale and shift (adaptive layer norm)._ $"LN"$ here only standardises each token to zero mean and unit variance over its $d_"tok"$ entries ($epsilon.alt = 10^(-5)$) and has no learned scale or shift of its own. The scale $1 + gamma$ and the shift $beta$ come from the time instead, so every block can rescale its features for the current noise level. This replaces feeding $t$ to the first layer only.

_(2) Gate._ The output of each residual branch is multiplied by $alpha$ before it is added back. $W_"mod"$ and $b_"mod"$ start at zero, so at initialisation $gamma = beta = alpha = 0$, both branches contribute nothing and every block is the identity. In our implementation, as in #citet(<peebles2023dit>), $alpha$ is a vector with one entry per channel and not a single scalar.

Together these two additions give the adaLN-Zero block of #citet(<peebles2023dit>), and the network stacks $6$ such blocks.

*Output layer.* After the last block, the tokens are normalised and modulated once more, without a gate, and mapped back to patches:
$ (beta_"out", gamma_"out") = W'_"mod" "SiLU"(e) + b'_"mod" in RR^(2 d_"tok"), $
$ Y = ((1 + gamma_"out") dot.o "LN"(Z) + beta_"out") W_"out" + b_"out" in RR^(n times p^2 c). $
Each row of $Y$ is reshaped to a $p times p$ patch with $c$ channels and the patches are put back in place, which gives the velocity in the shape of $x$. $W'_"mod"$, $b'_"mod"$, $W_"out"$ and $b_"out"$ start at zero, so the untrained network returns $v_theta = 0$, like the U-Net.

*Initialisation and size.* Apart from the zero-initialised layers named above, all linear layers are initialised as in @app:unet. The network has $"4,148,548"$ parameters for MNIST ($"4,151,628"$ for CIFAR-10).

#figure(
  caption: [Sizes of the DiT.],
  table(
    columns: 2,
    align: left,
    toprule,
    [Quantity], [Value],
    midrule,
    [Patch size $p$], [$2$],
    [Tokens $n$], [$196$ ($S = 28$), $256$ ($S = 32$)],
    [Token width $d_"tok"$], [$192$],
    [Blocks], [$6$],
    [Attention heads $n_h$, head dimension $d_h$], [$6$, $32$],
    [MLP layers, hidden width], [$2$, $768$],
    [Activation in the MLP], [GELU],
    [Activation in the time and modulation layers], [SiLU],
    [Positional embedding], [2D RoPE on queries and keys, no parameters],
    [Time embedding width $d_e$], [$192$],
    bottomrule,
  ),
) <tab:dit>

*Relation to the original DiT.* The block, the patchify step and the zero initialisation are those of #citet(<peebles2023dit>). Our network differs in four ways: it runs on pixels and not on the latent of an autoencoder; its positional embedding is a rotary embedding inside every attention layer, whereas the original adds a fixed sine-cosine embedding to the tokens once; it has no class label, so the modulation vector is the time embedding alone; and it outputs only the velocity, with no additional covariance output. It is also far smaller than the smallest model of that paper.

= Additional experimental details <app:details>

*Training.* Each of the six networks (U-Net and DiT on MNIST, Fashion-MNIST and CIFAR-10) is trained separately on the training split of its dataset with the conditional flow matching loss
$ cal(L)(theta) = EE_(t tilde.op cal(U) [0, 1], thin epsilon tilde.op cal(N)(0, I), thin x_1) 1/d norm(v_theta (t x_1 + (1 - t) epsilon, t) - (x_1 - epsilon))^2 . $
Its minimiser is the exact field @eq:u, but its minimum is not $0$, because the target $x_1 - epsilon$ is random given the noisy input. A random $5%$ of the training split is held out for validation. Both networks use the same settings: AdamW @loshchilov2019adamw with weight decay $0$, batch size $128$, peak learning rate $2 dot.op 10^(-4)$ reached by a linear warm-up over $500$ steps and followed by a cosine decay to $0$ over $400$ epochs, gradient clipping at global norm $1$, and an exponential moving average (EMA) of the weights with decay $0.999$. After every epoch the EMA weights are evaluated on the validation split with a fixed draw of $t$ and $epsilon$, so the validation losses of different epochs are comparable. Training stops when the validation loss has not improved for $5$ epochs, and the EMA weights of the best epoch are kept. All six runs stopped this way, long before the limit of $400$ epochs (@tab:training). Each run used one NVIDIA A100 (40 GB) with JAX `0.12.0.1`; the six runs and their per-epoch losses are in `results/training` of the repository.

#figure(
  caption: [Training runs. The training loss is the average over the best epoch with the current weights; the validation loss uses the EMA weights on the held-out $5%$. Time is the wall time of the run in minutes.],
  table(
    columns: 8,
    align: (left, left, right, right, right, center, center, right),
    toprule,
    [Dataset], [Network], [Parameters], [Epochs], [Best], [Train loss], [Val. loss], [Time],
    midrule,
    [MNIST], [U-Net], [$"1,074,529"$], [153], [148], [$0.1635$], [$0.1613$], [$25$],
    [MNIST], [DiT], [$"4,148,548"$], [168], [163], [$0.1633$], [$0.1607$], [$52$],
    [Fashion-MNIST], [U-Net], [$"1,074,529"$], [197], [192], [$0.2130$], [$0.2131$], [$33$],
    [Fashion-MNIST], [DiT], [$"4,148,548"$], [183], [178], [$0.2102$], [$0.2094$], [$58$],
    [CIFAR-10], [U-Net], [$"1,075,683"$], [276], [271], [$0.1798$], [$0.1810$], [$45$],
    [CIFAR-10], [DiT], [$"4,151,628"$], [237], [232], [$0.1762$], [$0.1765$], [$75$],
    bottomrule,
  ),
) <tab:training>

*Sampler comparison.* The evaluation noise ($128$ samples), the calibration noise ($32$ samples) and the noise of the sample grids are three separate draws, none of them used elsewhere. For both BendFlow variants, $"tol"$ is found by $14$ bisection steps on $log_10 "tol" in [-8, 3]$ as the smallest value whose mean NFE over the evaluation samples is at most the target; the NFE of individual samples varies around it (for example $26$ to $62$ at a target of $40$). The tuned variant chooses $h_0$ from ${0.01, 0.05, 0.1}$, $h_min$ from ${0.001, 0.005, 0.02}$ and $h_max$ from ${0.25, 0.5}$, separately for each field and target NFE; on the calibration samples $"tol"$ is matched to the NFE in the same way, and only $"tol"$ is matched again on the evaluation samples. With $h_max = 0.25$, @alg:bendflow needs at least $5$ steps, so at NFE $5$ the config variant has no room to adapt. The fixed schedule measures the mean of @eq:est over the calibration samples on a uniform grid of $200$ steps and places NFE steps with $h prop sqrt((1 - t) \/ overline(||hat(dot(m))||) (t))$. To check the references, each is compared with uniform Euler with $500$ steps: for the networks the median difference is at most $2.2 dot.op 10^(-3)$, but for a few exact CIFAR-10 references the two runs end on different images (largest difference $0.54$), so those references are unreliable.

*Numerical checks of the identities.* @tab:identities gives the largest relative error $||a - b|| \/ ||b||$ of the closed forms against automatic differentiation and against central finite differences with step $epsilon.alt^(1\/3) approx 6 dot.op 10^(-6)$, where $epsilon.alt$ is the machine precision of double precision. The Jacobian is checked along one random unit direction per point. The checks use $5$ random undecided points per dataset, and the divergence, which needs $d$ Jacobian-vector products, uses $2$ of them.

#figure(
  caption: [Largest relative error of the closed forms over the checked points.],
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    toprule,
    [], table.cell(colspan: 2)[Jacobian @eq:jac], table.cell(colspan: 2)[$dot(m)$, @eq:mdot], [Divergence @eq:div],
    [Dataset], [autodiff], [finite diff.], [autodiff], [finite diff.], [autodiff],
    midrule,
    [MNIST], [$6.2 dot.op 10^(-14)$], [$5.0 dot.op 10^(-7)$], [$6.2 dot.op 10^(-13)$], [$6.9 dot.op 10^(-9)$], [$6.4 dot.op 10^(-16)$],
    [Fashion-MNIST], [$3.0 dot.op 10^(-14)$], [$2.1 dot.op 10^(-7)$], [$2.8 dot.op 10^(-13)$], [$5.9 dot.op 10^(-9)$], [$1.4 dot.op 10^(-16)$],
    [CIFAR-10], [$1.4 dot.op 10^(-14)$], [$4.0 dot.op 10^(-7)$], [$4.8 dot.op 10^(-13)$], [$6.3 dot.op 10^(-8)$], [$2.5 dot.op 10^(-15)$],
    bottomrule,
  ),
) <tab:identities>

*Power iteration.* The largest relative residual $||C q - lambda q|| \/ abs(lambda)$ of the power iteration for $lambda_max (C)$ at the undecided points is $1.1 dot.op 10^(-4)$ on MNIST, $2.5 dot.op 10^(-5)$ on Fashion-MNIST and $3.4 dot.op 10^(-3)$ on CIFAR-10. For the top eigenvalue of the network Jacobians it is between $1.1 dot.op 10^(-2)$ and $6.3 dot.op 10^(-2)$.

= Additional results <app:figures>

The remaining CIFAR-10 figures and the full sampler tables of @sec:experiments, followed by the figures of Sections #nref(<sec:exp-exact>), #nref(<sec:exp-networks>) and #nref(<sec:exp-sampler>) for MNIST and Fashion-MNIST.

#figure(
  image(results + "exact_field_cifar10/active_set.pdf", width: 75%),
  caption: [Exact field on CIFAR-10: size of the active set @eq:active and its diameter $D_A$ along the trajectories of @fig:exact.],
) <fig:active-cifar>

#figure(
  grid(
    columns: (36%, 1fr, 62%),
    align: horizon,
    image(results + "trained_networks_cifar10/denoiser_error.pdf", width: 100%),
    [],
    image(results + "trained_networks_cifar10/jacobian_top_eig.pdf", width: 100%),
  ),
  caption: [Trained networks on CIFAR-10. Left: RMS per pixel of the difference between the network's denoised prediction $hat(x)_1$ and the exact $m$ at the same point, median over $8$ trajectories. Right: top eigenvalue of the symmetric part of the network Jacobian and the upper end of @eq:eigrange for the exact field at the same point, on $4$ trajectories (symmetric logarithmic axis).],
) <fig:networks>

#figure(
  caption: [Trained networks: mean RMS error per pixel to the $1000$-step reference over $128$ samples, at matched mean NFE. The lowest value in each column of each block is in bold.],
  {
    set text(size: 9pt)
    let gap = table.cell(colspan: 7, inset: 1pt)[]
    table(
      columns: 7,
      align: (left, left, left, center, center, center, center),
      toprule,
      [Dataset], [Network], [Sampler], [NFE 5], [NFE 10], [NFE 20], [NFE 40],
      midrule,
      [MNIST], [U-Net], [Uniform Euler], [$0.228$], [$0.147$], [$0.085$], [$0.050$],
      [], [], [BendFlow, config], [$0.260$], [$0.145$], [$0.078$], [$0.047$],
      [], [], [BendFlow, tuned], [$0.234$], [$0.143$], [$0.078$], [$0.046$],
      [], [], [Fixed schedule], [$bold(0.212)$], [$bold(0.135)$], [$bold(0.077)$], [$bold(0.042)$],
      gap,
      [MNIST], [DiT], [Uniform Euler], [$0.238$], [$0.146$], [$0.088$], [$0.043$],
      [], [], [BendFlow, config], [$0.269$], [$0.143$], [$bold(0.081)$], [$0.043$],
      [], [], [BendFlow, tuned], [$0.246$], [$0.141$], [$bold(0.081)$], [$bold(0.036)$],
      [], [], [Fixed schedule], [$bold(0.226)$], [$bold(0.137)$], [$0.082$], [$0.042$],
      gap,
      [Fashion-MNIST], [U-Net], [Uniform Euler], [$0.290$], [$0.159$], [$0.092$], [$0.049$],
      [], [], [BendFlow, config], [$0.286$], [$bold(0.134)$], [$bold(0.080)$], [$0.048$],
      [], [], [BendFlow, tuned], [$bold(0.218)$], [$bold(0.134)$], [$bold(0.080)$], [$bold(0.043)$],
      [], [], [Fixed schedule], [$0.256$], [$0.146$], [$0.081$], [$0.044$],
      gap,
      [Fashion-MNIST], [DiT], [Uniform Euler], [$0.309$], [$0.177$], [$0.096$], [$0.051$],
      [], [], [BendFlow, config], [$0.298$], [$bold(0.142)$], [$bold(0.079)$], [$0.051$],
      [], [], [BendFlow, tuned], [$bold(0.232)$], [$bold(0.142)$], [$bold(0.079)$], [$bold(0.041)$],
      [], [], [Fixed schedule], [$0.274$], [$0.156$], [$0.081$], [$0.043$],
      gap,
      [CIFAR-10], [U-Net], [Uniform Euler], [$0.282$], [$0.172$], [$0.093$], [$0.048$],
      [], [], [BendFlow, config], [$0.232$], [$bold(0.125)$], [$0.085$], [$0.064$],
      [], [], [BendFlow, tuned], [$bold(0.201)$], [$bold(0.125)$], [$bold(0.075)$], [$bold(0.040)$],
      [], [], [Fixed schedule], [$0.260$], [$0.151$], [$0.082$], [$0.042$],
      gap,
      [CIFAR-10], [DiT], [Uniform Euler], [$0.290$], [$0.178$], [$0.097$], [$0.050$],
      [], [], [BendFlow, config], [$0.235$], [$bold(0.128)$], [$0.089$], [$0.068$],
      [], [], [BendFlow, tuned], [$bold(0.203)$], [$bold(0.128)$], [$bold(0.074)$], [$bold(0.040)$],
      [], [], [Fixed schedule], [$0.267$], [$0.155$], [$0.083$], [$0.043$],
      bottomrule,
    )
  },
) <tab:sampler>

#figure(
  caption: [Exact field: fraction of the $128$ samples that end nearest to the same training image as the $1000$-step reference, at matched mean NFE. The highest value in each column of each block is in bold.],
  {
    set text(size: 9pt)
    let gap = table.cell(colspan: 6, inset: 1pt)[]
    table(
      columns: 6,
      align: (left, left, center, center, center, center),
      toprule,
      [Dataset], [Sampler], [NFE 5], [NFE 10], [NFE 20], [NFE 40],
      midrule,
      [MNIST], [Uniform Euler], [$0.34$], [$0.64$], [$0.79$], [$0.90$],
      [], [BendFlow, config], [$0.27$], [$0.72$], [$bold(0.91)$], [$bold(0.95)$],
      [], [BendFlow, tuned], [$0.27$], [$0.74$], [$bold(0.91)$], [$bold(0.95)$],
      [], [Fixed schedule], [$bold(0.59)$], [$bold(0.79)$], [$bold(0.91)$], [$bold(0.95)$],
      gap,
      [Fashion-MNIST], [Uniform Euler], [$0.16$], [$0.37$], [$0.62$], [$0.80$],
      [], [BendFlow, config], [$0.17$], [$0.56$], [$0.74$], [$0.82$],
      [], [BendFlow, tuned], [$0.26$], [$0.59$], [$0.80$], [$bold(0.91)$],
      [], [Fixed schedule], [$bold(0.39)$], [$bold(0.66)$], [$bold(0.82)$], [$bold(0.91)$],
      gap,
      [CIFAR-10], [Uniform Euler], [$0.09$], [$0.25$], [$0.51$], [$0.69$],
      [], [BendFlow, config], [$0.05$], [$0.52$], [$0.62$], [$0.66$],
      [], [BendFlow, tuned], [$0.12$], [$0.59$], [$0.76$], [$bold(0.90)$],
      [], [Fixed schedule], [$bold(0.38)$], [$bold(0.66)$], [$bold(0.83)$], [$bold(0.90)$],
      bottomrule,
    )
  },
) <tab:sampler-exact>

#figure(
  image(results + "schedule_baseline_cifar10_n128/error_vs_nfe.pdf", width: 100%),
  caption: [Mean RMS error to the reference against mean NFE on CIFAR-10, for the four samplers of @sec:exp-sampler. For the exact field the error mostly measures how often a sample ends on a different training image than the reference; @tab:sampler-exact reports that fraction directly.],
) <fig:sampler>

#figure(
  image(results + "sample_grids_cifar10/grid_dit.pdf", width: 100%),
  caption: [Samples of the CIFAR-10 DiT from the same $8$ noise vectors under each sampler, with the settings of @tab:sampler. The numbers above each row are the mean NFE and the mean RMS error to the reference over these $8$ images only.],
) <fig:grid>

#figure(
  {
    image(results + "exact_field_mnist/mdot_and_bound.pdf", width: 100%)
    v(2pt)
    image(results + "exact_field_mnist/active_set.pdf", width: 75%)
    v(6pt)
    image(results + "trained_networks_mnist/mdot_networks.pdf", width: 100%)
    v(2pt)
    grid(
      columns: (36%, 1fr, 62%),
      align: horizon,
      image(results + "trained_networks_mnist/denoiser_error.pdf", width: 100%),
      [],
      image(results + "trained_networks_mnist/jacobian_top_eig.pdf", width: 100%),
    )
  },
  caption: [MNIST. From top to bottom: the panels of @fig:exact, @fig:mdot-networks and @fig:networks.],
) <fig:mnist>

#figure(
  {
    image(results + "exact_field_fashion_mnist/mdot_and_bound.pdf", width: 100%)
    v(2pt)
    image(results + "exact_field_fashion_mnist/active_set.pdf", width: 75%)
    v(6pt)
    image(results + "trained_networks_fashion_mnist/mdot_networks.pdf", width: 100%)
    v(2pt)
    grid(
      columns: (36%, 1fr, 62%),
      align: horizon,
      image(results + "trained_networks_fashion_mnist/denoiser_error.pdf", width: 100%),
      [],
      image(results + "trained_networks_fashion_mnist/jacobian_top_eig.pdf", width: 100%),
    )
  },
  caption: [Fashion-MNIST. From top to bottom: the panels of @fig:exact, @fig:mdot-networks and @fig:networks.],
) <fig:fashion-mnist>

#figure(
  {
    image(results + "schedule_baseline_mnist_n128/error_vs_nfe.pdf", width: 100%)
    v(6pt)
    image(results + "schedule_baseline_fashion_mnist_n128/error_vs_nfe.pdf", width: 100%)
  },
  caption: [Mean RMS error to the reference against mean NFE, as in @fig:sampler, on MNIST (top) and Fashion-MNIST (bottom).],
) <fig:sampler-other>
