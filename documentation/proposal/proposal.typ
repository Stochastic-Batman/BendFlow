#import "@preview/touying:0.7.4": touying-slides, touying-slide, touying-slide-wrapper, config-page, config-common, pause, utils

// Compile with --input handout=true to show each slide fully revealed on one page.
#show: touying-slides.with(
  config-page(paper: "presentation-16-9", margin: 0pt, fill: white),
  config-common(handout: sys.inputs.at("handout", default: "false") == "true"),
)

#let coolpurple = rgb("#7f65a0")
#let lightpurple = rgb("#ede8f5")
#let darktext = rgb("#1a1a2e")
#let highlightpurple = rgb("#5a3e82")


#set text(
  font: "Libertinus Serif",
  size: 19pt,
  fill: darktext,
)

#show math.equation: set text(font: "New Computer Modern Math")

#set par(justify: false, leading: 0.6em)

#set list(
  marker: text(fill: coolpurple, size: 0.8em, baseline: -0.1em)[▶],
  spacing: 0.9em,
  body-indent: 0.6em,
)



// purple bold text, like \hl in the beamer deck
#let hl(body) = text(fill: coolpurple, weight: "bold", body)

// beamer-style block: purple title bar, light purple body
#let pblock(title, body, height: auto) = block(
  width: 100%,
  radius: 4pt,
  clip: true,
  stack(
    block(
      width: 100%,
      fill: coolpurple,
      inset: (x: 12pt, y: 7pt),
      text(fill: white, weight: "bold", title),
    ),
    block(
      width: 100%,
      height: height,
      fill: lightpurple,
      inset: 12pt,
      body,
    ),
  ),
)

// several blocks side by side; the grid gives all bodies the height of the tallest one
#let pblock-row(..blocks) = {
  let items = blocks.pos()

  grid(
    columns: items.len() * (1fr,),
    column-gutter: 0.5cm,

    ..items.map(((title, _)) => grid.cell(
      fill: coolpurple,
      inset: (x: 12pt, y: 7pt),
      text(fill: white, weight: "bold", title),
    )),

    ..items.map(((_, body)) => grid.cell(
      fill: lightpurple,
      inset: 12pt,
      body,
    )),
  )
}

// rounded box with a single message, used for the take-home line of a slide
#let callout(body) = align(center, block(
  fill: lightpurple,
  stroke: 1pt + coolpurple,
  radius: 8pt,
  inset: (x: 16pt, y: 11pt),
  align(center, body),
))

// Keep the slide furniture fixed as Touying reveals the body.
#let slide(title, body) = touying-slide-wrapper(self => {
  let layout(body) = {
    if title != none {
      block(
        width: 100%,
        fill: coolpurple,
        inset: (x: 1.1cm, y: 0.55cm),
        text(fill: white, size: 24pt, weight: "bold", title),
      )
      pad(x: 1.1cm, top: 0.5cm, body)
      place(
        bottom + right,
        dx: -0.6cm,
        dy: -0.4cm,
        text(size: 12pt, context [#utils.slide-counter.display() / #utils.last-slide-number]),
      )
    } else {
      align(center + horizon, body)
    }
  }
  touying-slide(self: self, setting: layout, body)
})

// straight arrow between two points given in cm
#let arrow(a, b, color: highlightpurple, thickness: 1.6pt, head: 0.25) = {
  let (ax, ay) = a
  let (bx, by) = b

  let len = calc.sqrt(calc.pow(bx - ax, 2) + calc.pow(by - ay, 2))
  let (ux, uy) = ((bx - ax) / len, (by - ay) / len)
  let (nx, ny) = (-uy, ux)
  let (cx, cy) = (bx - head * ux, by - head * uy)

  place(line(
    start: (ax * 1cm, ay * 1cm),
    end: (cx * 1cm, cy * 1cm),
    stroke: thickness + color,
  ))
  place(polygon(
    fill: color,
    (bx * 1cm, by * 1cm),
    ((cx + 0.45 * head * nx) * 1cm, (cy + 0.45 * head * ny) * 1cm),
    ((cx - 0.45 * head * nx) * 1cm, (cy - 0.45 * head * ny) * 1cm),
  ))
}

// filled dot at a point given in cm
#let marker(p, color: darktext, r: 0.09) = place(
  dx: (p.at(0) - r) * 1cm,
  dy: (p.at(1) - r) * 1cm,
  circle(radius: r * 1cm, fill: color),
)

// text label centred on a point given in cm
#let label-at(p, body, size: 14pt) = context {
  let content = text(size: size, body)
  let s = measure(content)

  place(
    dx: p.at(0) * 1cm - s.width / 2,
    dy: p.at(1) * 1cm - s.height / 2,
    content,
  )
}



// =================================================================
// 1. title
// =================================================================

#slide(none)[
  #block(
    fill: coolpurple,
    radius: 8pt,
    inset: (x: 28pt, y: 18pt),
    width: 80%,
  )[
    #text(fill: white, size: 30pt, weight: "bold")[BendFlow: Where Flow Matching Trajectories Bend]

    #v(0.2em)

    #text(fill: white, size: 18pt)[Project proposal, Visual Generative Models]
  ]

  #v(1.2em)

  #text(fill: coolpurple, size: 20pt)[
    Lado Turmanidze #h(2.5em) Geronimo Castaño
  ]

  #v(0.3em)

  #text(fill: coolpurple, size: 15pt)[Universität des Saarlandes]

  #v(0.8em)

  #text(fill: coolpurple, size: 15pt)[December 2, 2026]
]



// =================================================================
// 2. motivation
// =================================================================

#slide[The velocity tells us where to move][

  #grid(
    columns: (1.15fr, 1fr),
    column-gutter: 1.2cm,

    [
      #set text(size: 18pt)

      #v(1cm)

      - Start with noise at $t = 0$. The goal is an image at $t = 1$.

      #pause
      #v(2cm)
      - At state $x$, the model predicts a #hl[velocity] $v_theta (x,t)$. Euler takes a step $x_"next" = x + h v_theta (x,t)$.
    ],

    [#pause],

    [
      #let straight-ink = rgb("#007676")
      #let step-ink = rgb("#a34b07")
      #box(width: 10cm, height: 5.15cm, {
        // Curved ends share tangents with the straight middle.
        place(curve(
          stroke: 3.5pt + highlightpurple,
          curve.move((0.6cm, 4.4cm)),
          curve.cubic((2.30cm, 4.774cm), (2.13cm, 3.836cm), (2.55cm, 3.5cm)),
        ))
        place(line(start: (2.55cm, 3.5cm), end: (5.2cm, 1.38cm), stroke: 3.5pt + straight-ink))
        place(curve(
          stroke: 3.5pt + highlightpurple,
          curve.move((5.2cm, 1.38cm)),
          curve.cubic((5.96cm, 0.772cm), (7.75cm, -0.58cm), (8.58cm, 0.92cm)),
        ))
        arrow((0.6, 4.4), (3.42, 5), color: step-ink)
        arrow((3.2, 3), (5.54, 1.1), color: step-ink, thickness:  3.5pt)
        marker((0.6, 4.4))
        marker((8.58, 0.92))
        label-at((0.55, 4.96))[Noise]
        label-at((9.4, 0.96))[Image]
        label-at((4.2, 0.24), size: 13pt)[#text(fill: step-ink)[Large Euler step]]
      })
      #set text(size: 15pt)
      #grid(
        columns: (0.5cm, auto),
        column-gutter: 0.15cm,
        row-gutter: 0.2em,
        align: (left + horizon, left),
        rect(width: 0.5cm, height: 3pt, stroke: none, fill: highlightpurple),
        [#text(fill: highlightpurple, weight: "bold")[Curved] · Big steps overshoot],
        rect(width: 0.5cm, height: 3pt, stroke: none, fill: straight-ink),
        [#text(fill: straight-ink, weight: "bold")[Straight] · Big steps are fine],
      )
    ],
  )

  #pause

  #v(2cm)

  #align(center, block(
    fill: highlightpurple,
    radius: 8pt,
    inset: (x: 22pt, y: 12pt),
    text(fill: white)[Can we tell when the velocity will change enough to need a *smaller step*?],
  ))
]



// =================================================================
// 3. proposed method: the idea
// =================================================================

#slide[The velocity points toward a clean-image estimate][

  #grid(
    columns: (1.25fr, 1fr),
    column-gutter: 1cm,

    [
      #set text(size: 18pt)

      - Imagine we knew the final image. On the linear path, the arrow would point straight toward it.

      - Several images could explain the noisy state. The #hl[posterior weights] tell us how likely each one is.
    ],

    box(width: 9cm, height: 5.8cm, {
      let x = (0.6, 2.9)
      let imgs = ((5.6, 0.7), (5.6, 2.9), (5.6, 5.1))
      let widths = (3.6pt, 3.2pt, 1pt)

      for i in range(3) {
        arrow(x, (imgs.at(i).at(0) - 1.05, imgs.at(i).at(1)), color: coolpurple, thickness: widths.at(i), head: 0.3)
      }

      label-at((2.2, 1.45))[0.45]
      label-at((2.6, 2.5))[0.40]
      label-at((2.0, 4.3))[0.15]

      place(dx: 0.25cm, dy: 2.55cm, circle(radius: 0.35cm, fill: coolpurple, align(center + horizon, text(fill: white, size: 15pt)[$x$])))

      for (i, p) in imgs.enumerate() {
        place(
          dx: (p.at(0) - 1.05) * 1cm,
          dy: (p.at(1) - 0.35) * 1cm,
          box(
            fill: lightpurple,
            stroke: 1pt + coolpurple,
            radius: 4pt,
            inset: (x: 8pt, y: 5pt),
            text(size: 14pt)[image #(i + 1)],
          ),
        )
      }

      label-at((3.7, 5.62))[Posterior weights at one noisy point]
    }),
  )

  #pause

  #pblock[From velocity to a clean-image prediction][
    The ideal velocity averages these arrows and points toward the weighted mean image $m$.
    $ u_t (x) = (m-x)/(1-t) quad ==> quad m = x + (1-t)u_t (x) $
  ]

  #pause

  #text(size: 16pt)[
    The training set and Gaussian path $x_t = alpha_t x_1 + sigma_t epsilon$ give the exact weights. #hl[What happens to $m$ when they shift?]
  ]
]



// =================================================================
// 4. proposed method: theory
// =================================================================

#slide[Posterior uncertainty explains how the estimate moves][

  #set text(size: 17pt)

  The mean $m$ is the center of the posterior. The spread and asymmetry around it describe the uncertainty. Write $w_i$ for the weight of image $x^((i))$ and $z_i = x^((i)) - m$.

  #grid(
    columns: (1.3fr, 1.25fr, 1.8fr),
    column-gutter: 1em,
    row-gutter: 0.45em,
    [#hl[Mean]], [$m = sum_i w_i x^((i))$], [Current clean-image prediction],
    [#hl[Covariance]], [$C = sum_i w_i z_i z_i^top$], [Variance in different directions],
    [#hl[Contracted third moment]], [$kappa = sum_i w_i norm(z_i)^2 z_i$], [Skewness-like asymmetry],
  )

  #pause

  #hl[As we follow the velocity, the weights shift.] The exact rate at which the clean-image estimate moves is
  $ dot(m) = dot(gamma)_t (C hat(epsilon) - gamma_t kappa) $

  #text(size: 15pt)[$hat(epsilon)$ is the predicted noise and $gamma_t = alpha_t \/ sigma_t$ is the signal-to-noise amplitude ratio. The two terms can reinforce or cancel.]

  #pause

  #pblock[Back to the velocity arrow][
    On the linear path, a fast-changing image estimate means a fast-changing velocity. Along the trajectory, $dot(u) = dot(m)/(1-t)$.
  ]
]



// =================================================================
// 5. proposed method: sampler
// =================================================================

#slide[Use the changing velocity to choose a step][

  #set text(size: 17pt)

  #hl[What we need.] To choose an Euler step of size $h$ on the linear path, we need $dot(m) (t_k)$, the rate at which the clean-image prediction is changing now.
  $ norm("local error") approx h^2/(2(1-t_k)) thin norm(dot(m) (t_k)) $

  #pause

  #hl[What we have.] We reuse the predictions from the last two velocity evaluations, so the estimate takes #hl[no extra model calls].
  $ underbrace(hat(dot(m))_k, "average rate over the last step") = (hat(m)_k - hat(m)_(k - 1)) / (t_k - t_(k - 1)) $

  #pause

  #hl[Choose the next step.] Plug $hat(dot(m))_k$ into the error formula and choose $h = h_k$ for the target tolerance.

  #callout[
    If the prediction changes faster, take a #hl[smaller step]. \
    Each sample gets its own time grid.
  ]
]



// =================================================================
// 6. related work
// =================================================================

#slide[How this fits with existing samplers][

  #set text(size: 16pt)

  #pblock-row(
    (
      [Hard timesteps],
      [CSFlow and Stable Velocity (2026) identify #hl[difficult timesteps] for each dataset and concentrate training or sampling effort there.],
    ),

    (
      [Curvature and the denoiser],
      [Chen et al. (2023) and Rao et al. (2026) link bending to how fast the #hl[denoised prediction] moves.],
    ),

    (
      [Adaptive step sizes],
      [Jolicoeur-Martineau et al. (2021) adapt steps using error estimates with #hl[extra model calls].],
    ),
  )

  #pause
  #v(0.6em)

  #pblock[The question of this project][
    Can we use #hl[uncertainty about the clean image] to estimate the error of the next sampling step?
  ]
]



// =================================================================
// 7. datasets
// =================================================================

#slide[Experimental setup][

  #set text(size: 16pt)

  #pblock-row(
    (
      [Datasets],
      [
        - #hl[MNIST, Fashion-MNIST] · 60,000 each, #box[$28 times 28$]
        - #hl[CIFAR-10] · 50,000, $32 times 32 times 3$
        - Pixels scaled to $[-1, 1]$
      ],
    ),

    (
      [Velocity fields],
      [
        - #hl[Exact] · computed from the full training set
        - #hl[Learned] · one small U-Net and one DiT per dataset
      ],
    ),
  )

  #pause
  #v(0.5em)

  #pblock[Noise schedules][
    Each schedule sets the signal and noise coefficients in $x_t = alpha_t x_1 + sigma_t epsilon$.

    #v(0.35em)
    #grid(
      columns: (1fr, 1fr),
      column-gutter: 0.6em,
      [#hl[Linear] \ #text(size: 14pt)[$t x_1 + (1-t) epsilon$]],
      [#hl[Variance preserving (VP)] \ #text(size: 14pt)[$alpha_t x_1 + sqrt(1-alpha_t^2) epsilon$]],
    )
  ]

]



// =================================================================
// 8. evaluation
// =================================================================

#slide[Evaluation][

  #set text(size: 17pt)

  - #hl[Formula agreement.] Along exact-field trajectories, compute $dot(m)$ using the moment formula and automatic differentiation (AD).
    $ "Relative error" = frac(norm(dot(m)_"formula" - dot(m)_"AD"), norm(dot(m)_"AD")) $

  #v(1cm)

  #pause

  - #hl[Peak timing.] For each starting noise sample, compare the timesteps when $norm(dot(m))$ peaks along the exact and learned trajectories (e.g. $|t_"learned" - t_"exact"|$)

  #v(1cm)

  #pause

  - #hl[Sampler comparison.] Compare BendFlow with the standard sampler at matched average numbers of function evaluations, using pixel RMSE against a reference (1000 steps):

  $
    "RMSE"_"pixels" = sqrt(1/D sum_(j=1)^D (hat(x)_j - x_j^"ref")^2)
  $
]



// =================================================================
// 9. questions
// =================================================================

#slide(none)[#text(fill: coolpurple, size: 48pt, weight: "bold")[Questions?]]
