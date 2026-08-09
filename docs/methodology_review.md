# Adversarial Methodology Review: Can Behavioral Distances Between Domain-Pruned Models Detect Domains of Knowledge?

**Date:** 2026-07-18
**Question under review:** Is the pruning-metrics methodology a sound way to
measure whether distinct domains of knowledge (math / code / science QA) can be
detected and clustered inside an LLM — and how does it compare to alternatives
such as checking active parameters per token?

**How this review was produced.** Code-level claims were verified by direct
inspection of this repository. External claims went through an adversarial
deep-research pipeline: 5 parallel literature searches (25 unique sources
found), the top 15 sources fetched and read, 45 falsifiable claims extracted,
deduplicated to 15 load-bearing claims, each independently attacked by three
adversarial verifiers (source fidelity, contradiction search, methodological
validity; 2/3 refutations kill a claim). 8 claims survived unanimously, 3
survived contested (marked *(contested)*), 4 were refuted and are excluded
from the argument (Appendix A). Every external claim below carries a numbered
citation.

## The methodology under review

- Qwen2-72B is WANDA-pruned per-output-row
  (`infra/runners/_runner_common.py:apply_wanda_pruning`, score =
  |W[i,j]|·rms[j]) using calibration activation statistics collected from one
  of three task datasets — GSM8K, HumanEval+, ARC-Challenge
  (`infra/runners/run_pruning_calibration.py`) — at 20/40/60/80% sparsity:
  12 pruned variants + 1 baseline = 13 model points.
- Each variant is evaluated teacher-forced on all three benchmarks, recording
  top-5 next-token logprobs per answer position
  (`src/pruning_metrics/evals/coding/teacher_forcing.py`, `top_k=5`), plus
  free-form pass@1.
- Four distances are computed on the renormalized top-5 support
  (`src/pruning_metrics/metrics/distributions.py`): KLD, √JSD, 1-D
  Wasserstein over logprob values ("EMD"), and set-based Chamfer -- the
  founding set this review evaluates. *(Update 2026-08-09: sixteen
  distances now exist, one per module under
  `src/pruning_metrics/prob_measures/` and re-exported through the
  `metrics/distributions.py` facade; see the §6 addendum.)*
- **Analysis A** (`notebooks/experiment/04_metric_spaces.ipynb`): Pearson R²
  between mean per-sample distance and cross-task pass@1 drop.
- **Analysis B** (`notebooks/experiment/05_tsne.ipynb`): 13×13 pairwise model
  distance matrices embedded in 2-D via t-SNE / UMAP / PCA-on-MDS / Isomap /
  LLE; clustering by calibration domain vs. pruning level is judged visually.
- **Implicit claim:** if models cluster by calibration domain in these
  behavioral metric spaces, domains of knowledge are detectable and separable
  in the base model — a prototype for later deciding whether MoE experts are
  redundant/mergeable.

## 1. Verdict

**No — as specified, the methodology cannot support the claim that distinct
domains of knowledge are "detected and clustered" inside Qwen2-72B.** The
design chains three independently weak links, and the weaknesses compound
rather than cancel. First, the input knob (WANDA calibration domain) is the
*least* domain-sensitive pruning method in the sparsity band the study mostly
occupies, so the "signal" it is supposed to inject is expected to be faint or
absent [1]. Second, the readout (teacher-forced top-5 output-distribution
distances) is a behavioral/functional construct that the
representational-similarity literature treats as *non-diagnostic* of internal
organization, and the methodology relies on the unsupported converse inference
(behavioral divergence ⇒ internal domain separation) [4]. Third, the analysis
(visual clustering of 13 points in t-SNE/UMAP/Isomap/LLE space) is below any
sample size at which cluster presence can be assessed, using embedding methods
that manufacture apparent clusters from structureless data [5][6][7][8][9][10].
Even a clean "models cluster by calibration domain" picture would be consistent
with output-format confounds, WANDA-specific artifacts, and embedding
artifacts — none of which is internal knowledge modularity. The methodology is
a reasonable *behavioral* exploratory sketch, but it is mis-scoped for the
mechanistic conclusion it is being asked to license, especially as a prototype
for deciding whether MoE experts are redundant/mergeable.

*(Update 2026-08-01: the v2 experiment in §6 empirically **refutes the strong
form of C2** — behavioral distances do track parameter-level structure once
sparsity is stratified out — while confirming the design criticisms I1–I3 and
identifying level-pooling as the dominant failure mode. See §6.)*

## 2. What the methodology gets right

- **Activation-aware pruning is a defensible starting point.** WANDA is a
  sensible, cheap way to obtain domain-conditioned model variants without
  retraining, and using real task calibration data is more principled than
  random text.
- **Multiple distance metrics hedge against single-metric artifacts.**
  Computing KLD, √JSD, EMD, and Chamfer is good instinct; disagreement among
  them is itself informative.
- **A sparsity sweep (20/40/60/80%) is the right axis to expose
  dose-response** — and it is exactly the axis on which the calibration-domain
  effect is known to grow [1], so the design *could* be repurposed to test the
  real moderator (sparsity) if paired with quantitative validation.
- **Analysis A (brittleness regression) is a legitimate, falsifiable
  behavioral question** in its own right, independent of the modularity claim.
- **The raw material for a far stronger test already exists in-repo:**
  `wanda_stats.pt` stores per-layer input-channel RMS per calibration domain,
  from which the exact pruning mask at any level is deterministically
  re-derivable — so a direct parameter-level probe is nearly free (§4–5).

## 3. Where it is weakest, ordered by severity

### Construct-validity failures (most severe — more data cannot fix these)

**C1. The input knob barely moves for WANDA in this regime.** For WANDA
specifically, the calibration-domain effect is near-negligible below 50%
sparsity (<0.1% accuracy difference between calibration sources), rising only
to ~2.3% at 60% unstructured sparsity, and WANDA is the least
calibration-sensitive pruning method tested (3.2-point best-vs-worst spread on
Gemma 2B versus 7.5 for SparseGPT) [1]. At 20–40% sparsity — half of the
design's variants — any domain-driven behavioral separation should be faint at
best. A single pruning method therefore confounds every conclusion with
WANDA's particular (low) calibration sensitivity.

**C2. Behavioral output-similarity is the wrong construct for an
internal-structure claim.** Functional/behavioral similarity is sample-limited
and non-diagnostic of internal structure; it is unstable across tasks, and the
field pairs it with *representational* similarity precisely to catch models
that behave alike but "produce the same output differently" internally [4].
The established inference runs representational→functional; the methodology's
implicit claim is the unsupported converse (behavioral divergence → internal
domain separation) [4]. This is the single deepest flaw: the readout and the
conclusion live in different construct spaces.

**C3. What the four distances actually measure is weaker than intended**
(`src/pruning_metrics/metrics/distributions.py`). All four operate on a
renormalized top-5 support, discarding tail mass; tokens absent from one
model's top-5 are imputed logprob −50 (`_MISSING_LOGPROB`) before
renormalization, so KLD/JSD are dominated by *support disjointness* rather
than probability shape. The implemented "EMD" (`compute_emd`) is 1-D
Wasserstein where the atom *positions* are the logprob values themselves —
token identity never enters, so two models predicting entirely different
tokens with the same logprob shape score ≈ 0. This is much weaker than the
2-D token-embedding EMD the proposal specifies
(`geom_compute_final_proj_proposal.md`, step 6). Chamfer (`compute_chamfer`)
matches nearest neighbors across *all* positions, so unrelated positions can
match. These are truncated-distribution-shape probes, not knowledge probes.

**C4. Domain is confounded with output format.** GSM8K (free-form
chain-of-thought), HumanEval+ (code), and ARC-Challenge (single-letter MCQ)
differ in output *format* as much as in knowledge domain — ARC answer
sequences are ~1 token, so per-position distances measure something entirely
different there than on code or math chains. Apparent "domain clustering"
could be format clustering — a construct confound no embedding sophistication
removes.

**C5. Teacher forcing understates generation-phase differences**
*(contested)*. Components that look negligible under teacher-forced/prompt-only
evaluation can become highly consequential once free-running generation begins
[5] *(contested — the refuting lens noted the underlying figure concerns the
network's last two layers, and extending it to top-5 recording is an
extrapolation; treat as directional caution)*. The better-established related
point: a domain knob's capability boost is entangled with generic degradation —
math-style calibration produced the *largest* general perplexity increase of
any source tested (+3.52) — so behavioral distances conflate targeted
specialization with non-specific damage [3].

### Internal-validity failures (independently fatal to Analysis B)

**I1. n=13 is far below the floor for any cluster claim.** Cluster analysis
needs roughly 20–30 points *per expected subgroup* for adequate power even at
large effect sizes [6]. With ~4 points per hypothesized domain cluster,
Analysis B cannot reliably assess cluster presence or absence.

**I2. The embedding methods fabricate clusters.** t-SNE provably exaggerates
inter-cluster separation, shows crisp clusters where data cluster weakly or
not at all, changes shape with perplexity with no "correct" setting, and
requires perplexity < n — a near-unusable constraint at n=13 [7][8][9]. UMAP
manufactures visually distinct clusters from a single Gaussian and is
non-deterministic run-to-run [10]. Neighbor-graph and hierarchical embeddings
*always* yield apparent groupings even with no true structure [11][8][7].

**I3. No quantitative validation.** `05_tsne.ipynb` judges clustering
visually — no silhouette, no ARI against calibration/level labels, no
permutation test. Valid cluster claims require exactly these; at n=13 the
permutation test is exactly enumerable [11]. Without them, Analysis B is
uninterpretable.

**I4. HumanEval+ has N=33 prompts** (flagged `small_sample` in
`04_metric_spaces.ipynb`), giving the code "domain" high variance and making
any code-vs-rest separation especially unreliable.

## 4. Comparison against active-parameters-per-token and other mechanistic alternatives

The premise the methodology needs — a domain occupies a *fixed, separable
parameter subset* — is exactly what mechanistic methods test directly, and the
evidence is genuinely mixed, which is why a direct probe matters.

| Approach | What it shows | Construct validity for "which params serve a domain" | Cost | Key limitation |
|---|---|---|---|---|
| **Behavioral top-5 distances (this repo)** | Output-distribution divergence under teacher forcing | Low — functional, not representational [4] | Low | Format/damage confounds; converse-inference gap [3][4] |
| **Pruning-mask / importance overlap** (Jaccard of retained WANDA masks per domain, per layer) | Whether domain calibrations retain overlapping vs. disjoint weights | Moderate–high; parameter-level, directly on-topic | **Near-zero — masks re-derivable from existing `wanda_stats.pt` + base weights** | Still WANDA-specific; overlap ≠ causal necessity |
| **Per-token contextual sparsity** (DejaVu-style active subnetworks) | Which params activate per token; >95% MLP / >80% heads silenceable while matching dense output | High *where dense-output fidelity holds* *(contested)* | Moderate | Fidelity breaks on reasoning/generation (GSM8K 75.5→38.6/58.7; HumanEval 56.0→20.7/45.7 at ~50% sparsity) [14]; active sets are per-input, not domain-static [12] *(contested)* |
| **Knowledge / task / instruction neurons** | Whether domains map to identifiable neuron subsets | Moderate–high; causal if intervened on | Moderate | Same-type inputs overlap; cross-type minimal [15][16][17] |
| **MoE expert-routing specialization** | Whether experts specialize by domain | Moderate; needs causal controls | Low if MoE available | Routing is linear in hidden states, so raw overlap is noisy without controls [5] |
| **SAEs / probing classifiers** | Feature- or direction-level domain signal | Potentially high, but not evidenced in this review's verified set | High (SAE) / low (probes) | Forward-looking, not validated here |

The "active parameters per token" alternative is directly measurable and
mechanistically closer to the target construct: contextual sparsity silences
>95% of MLP parameters and >80% of attention heads per token while
reproducing the dense output [12] *(contested)*. Two contested qualifications
matter. (a) The dense-output equivalence breaks precisely on the
reasoning/generation/code domains this study cares about, where sparse models
introduce propagating token errors [14]. (b) Active-parameter sets are a
*per-input* phenomenon, so a domain is not obviously a fixed subset that a
single domain-calibrated prune can capture [12] *(contested)*. The refutations
here actually cut *toward* the methodology's premise: same-domain inputs
recruit largely overlapping subnetworks whose union forms a stable, separable
subset — instruction-specific neurons overlap highly within a type and
minimally across types [15], task-specific neurons are identifiable with
overlap tracking generalization [16], and keystone neurons (<0.2%) are stable
under prompt resampling [17]. The mechanistic literature does not settle
whether domains are modular — which is exactly why a *direct* parameter test
beats a behavioral proxy.

**Whether knowledge is modular at all.** The strongest caution: causal tracing
localizes facts to specific MLP layers, yet editing those layers is no more
effective than editing others [13]. "Where something is represented" need not
tell you where to change it — a direct warning against inferring separable
domain subsets from *any* localization-style signal, and a caveat that
transfers to the MoE expert-merging end goal.

## 5. Recommendations, ranked by cost-to-benefit

1. **Add the pruning-mask-overlap analysis — highest benefit, near-zero
   cost.** Masks for every (domain, level) are exactly reconstructable from
   `wanda_stats.pt` + base weights (per-row |W|·rms thresholding in
   `_runner_common.py`). Since |W| is shared across domains and only the
   per-domain RMS vector differs, per-layer Jaccard overlap of retained-weight
   sets across the three calibrations, per sparsity level, *is* the direct
   parameter-level answer to "do domains occupy different parameters" — it
   sidesteps the format confound (C4) and the truncated-distribution issues
   (C3) entirely [12][15][16]. Expect high overlap at low sparsity, diverging
   with level, mirroring the known WANDA sensitivity curve [1].
2. **Replace visual clustering with quantitative validation — essential and
   cheap.** On the existing 13×13 matrices: silhouette and ARI against both
   calibration and level labels, plus exact permutation tests (enumerable at
   n=13) [11]; compare against the sorted distance matrix directly. If
   clusters do not survive permutation, do not claim them. This makes
   Analysis B falsifiable, though it cannot rescue the n=13 power problem [6].
3. **Reframe the primary axis as sparsity, not domain.** The detectable
   dose-response is sparsity-driven; the WANDA domain effect is expected to be
   faint below 50% and ~2.3% at 60% [1]. Present domain as a weak secondary
   moderator, explicitly hedged.
4. **Break the format confound.** Add tasks sharing format across domains
   (e.g., MCQ math alongside ARC) and varying format within a domain, so
   "domain" separation cannot be explained by CoT-vs-code-vs-single-letter
   output shape.
5. **Add a second pruning method and a representational readout.** SparseGPT
   is far more calibration-sensitive than WANDA [1]; if domain structure is
   real it should appear more strongly there. Pairing behavioral distances
   with a representational-similarity measure (e.g., CKA on hidden states)
   runs the inference in the supported direction [4].
6. **Fix or re-scope the mis-specified distances.** Implement the proposal's
   token-embedding EMD, retain tail mass instead of the −50 imputation, and
   constrain Chamfer to aligned positions — or explicitly document the current
   metrics as truncated-distribution-shape probes, not knowledge probes.
7. **Lowest priority / highest cost: a genuine mechanistic probe.** For the
   MoE expert-mergeability goal, a per-token activated-subnetwork or
   expert-routing analysis with causal controls is the construct-valid tool —
   budgeting for its failure modes on reasoning/generation [14] *(contested)*,
   routing-geometry noise [5], and the localization-≠-editability caveat [13].

## 6. Addendum: v2 empirical results (2026-08-01)

After this review, we ran the experiment it recommends (§5, items 1, 2, 4, 5)
to test finding **C2** empirically: is a behavioral distance matrix diagnostic
of parameter-level structure, or not?

**Design (v2).** Qwen2-7B; two pruners (WANDA and SparseGPT); five calibration
domains chosen to break the format confound (GSM8K free-form math, MathQA
multiple-choice math, HumanEval+ and MBPP+ code, ARC-Challenge multiple-choice
science); three calibration seeds; eight sparsity levels (10–80%); plus the
unpruned baseline. 232 of the 241 planned variants completed (missing: 9
MBPP-calibrated tail variants). Readout: teacher-forced top-10 per-token
distances (KLD, √JSD, EMD, Chamfer) on all five benchmarks, 200 samples each →
twenty 232×232 behavioral distance matrices. Ground truth: pairwise Jaccard
distance between pruning masks (1/32 pseudorandom digest of every `.layers.`
weight matrix; digest sampling error SE ≈ 3.5e-5, negligible).

**Analysis correction.** The pre-registered notebook analysis (Mantel +
partial Mantel controlling |Δlevel|; pooled domain silhouette/ARI; a
domain-vs-format mean contrast) was itself adversarially verified before
reporting, and two of its three statistics failed review. (1) |Δlevel| is a
*suppressor*, not a confound control: partialling it raised r in 20/20 combos
(e.g., 0.884 → 0.923) because the true confound is the joint sparsity pair
(level_i, level_j), which explains R² ≈ 0.94 of behavioral and ≈ 0.97 of
mask-distance ranks. (2) The pooled domain tests and the domain-vs-format
contrast have no power by construction: both contrast groups share identical
level marginals, so a level-only model predicts their means to three decimals.
Sparsity level is a nuisance axis so dominant that any statistic pooled across
it is uninterpretable. The corrected analysis below stratifies by level (and
by pruner where noted); it reproduces the notebook's raw numbers exactly and
was re-run on all 20 (benchmark × metric) combinations.

**Result 1 — C2's strong form is refuted: behavioral distances track mask
structure at matched sparsity.** Restricting to same-sparsity pairs and
removing per-level means, behavioral distance correlates with mask-Jaccard at
**r = +0.745 to +0.832 across all 20 combos**, each at the floor of a
*restricted* permutation test that shuffles variants only within level strata
(p = 0.001, null centred at 0.000 ± 0.037). The result is robust to
additionally controlling pruner and domain identity (r = +0.700 to +0.836).
This cannot be attributed to the sparsity dose-response — the permutation null
holds sparsity fixed. Behavioral output-distribution distances, even the
truncated top-k probes criticized in C3, carry substantial parameter-level
signal. (For contrast: the *pooled* Mantel r ≈ 0.86–0.91 is ~97%
sparsity-driven and collapses to 0.09–0.36 under the correct level-pair
control — the headline number lives at matched sparsity, not in the pooled
matrix.)

**Result 2 — domains do separate, in both behavioral and mask space, once
sparsity is stratified.** Within (pruner, level) strata, calibration-domain
labels separate in behavioral space on every benchmark: for **WANDA**,
silhouette is positive in 160/160 strata (mean ≈ +0.40), ARI = 1.000 in
146/160 (never below +0.49), and the label-permutation test is at its floor
(p = 0.0005) in 160/160. For **SparseGPT** the signal is real but much weaker
(silhouette ≈ +0.08, ARI +0.1 to +0.8, p < 0.01 in 155/160); the only
failures are at 80% sparsity, four of five on MBPP+, the stratum thinned by
the 9 missing variants. The parameter-level ground truth shows the same
structure directly: same-domain WANDA masks are ~4× closer in Jaccard than
cross-domain masks at every level (e.g., 0.064 vs 0.248 at L=80), while
SparseGPT's mask-level domain differentiation is far smaller (Δ ≈ 0.005–0.046).
Notably this *inverts* the accuracy-based expectation from §5 item 5:
SparseGPT is the more calibration-sensitive pruner by benchmark accuracy [1],
but WANDA's masks — pure |W|·rms thresholds — are far more domain-differentiated
than SparseGPT's Hessian-corrected ones. The domain-vs-format question (C4)
remains genuinely open: within strata, the same-domain/different-format vs
same-format/different-domain gaps are small and flip sign between pruners.

**Verdict on C2: refuted.** Behavioral distance matrices *are* diagnostic of
parameter-level structure — they recover mask-overlap geometry at matched
sparsity (r ≈ 0.8) and recover calibration domain with up to perfect ARI once
the sparsity axis is stratified out. The v1 review's practical criticisms of
the original pipeline stand in full — 13 points, visual embeddings, no
stratification, format confounds (I1–I3, C4) — and v2 adds a new one: the
dominant failure mode is not the readout but *pooling across sparsity*, which
buries both signals (the executed notebook's own pre-registered verdict cell
printed "MIXED" for exactly this reason and is superseded by the stratified
analysis). But C2's central assertion, that behavioral output divergence is
non-diagnostic of internal organization, is empirically wrong for this system.

**Caveats.** (i) The 20 combos are ~one effective test, not twenty: the four
metrics correlate at ρ = 0.945–0.999 within a bench and ρ = 0.83–0.96 across
benches. (ii) WANDA and SparseGPT must be reported separately — averaging
them mixes a near-perfect domain signal with a weak one. (iii) The
SparseGPT/L=80/MBPP+ corner (ARI −0.07, p = 0.20) is the documented boundary
condition, confounded with the 9 missing variants (232/241 completed). (iv)
"Domain-differentiated masks" means the calibration data's activation
statistics select measurably different weights — a necessary condition for
domain modularity, not proof that the selected weights *causally serve* the
domain (the localization ≠ editability caveat [13] applies unchanged).

**Addendum (2026-08-09).** Notebook `08_distribution_metrics.ipynb`
broadened the four founding distances to all sixteen and stress-tested C3's
criticism of what the four distances actually measure. It found tight
agreement across all sixteen (Spearman ρ ≥ 0.838 over 360 pairwise
comparisons, median 0.968), so the choice among them is not load-bearing for
any v1 or v2 conclusion above. Notebook `09_reducer_sweep.ipynb` then
crossed every distance against every dimensionality reduction and found the
reducer choice, not the distance choice, dominates the resulting picture.

## Appendix A: claims killed by adversarial verification

Four claims were refuted by ≥2 of 3 independent verifiers and are **excluded**
from the argument above. Notably, the strongest *pro-methodology* claim was
among them — evidence the review did not stack the deck:

1. *"Domain-matched calibration measurably boosts corresponding-domain
   performance after compression (MathQA +4.35 pruning / +5.92 quantization;
   CodeQA up to +29.7% relative for SparseGPT)…"* — refuted on source
   fidelity/overgeneralization grounds.
2. *"No calibration source is consistently best; the advantage tracks
   distributional similarity to pretraining data rather than topical domain
   content; self-generated calibration text matches or beats external domain
   corpora in 17 of 20 cases."* — refuted.
3. *"Larger models are less sensitive to calibration-data choice than smaller
   ones (13B vs 7B), so 72B effects may be attenuated."* — refuted.
4. *"MoE router logits are linear in hidden states, so ~60% cross-model expert
   overlap is statistically indistinguishable from noise, making
   domain-overlap clustering weak evidence of shared substructure."* — refuted
   as stated (a weaker form survives in [5]).

That claims 1 and 2 both died cuts both ways: the best evidence that the
calibration knob *works* and the sharpest evidence that it is *not a domain
knob at all* each failed verification. The honest reading is that
calibration-domain sensitivity is real but small for WANDA [1][3], and nothing
verified establishes that it selects *domain knowledge* rather than
distributional statistics.

## Sources

1. https://arxiv.org/abs/2410.17711 — Beware of Calibration Data for Pruning Large Language Models
2. https://arxiv.org/html/2410.17170v1 — calibration-data analysis for compression
3. https://arxiv.org/html/2510.10618 — calibration-domain effects on capability vs. general degradation
4. https://arxiv.org/html/2312.02730v1 — representational vs. functional model similarity
5. https://arxiv.org/html/2604.09780v1 — MoE routing overlap analysis
6. https://arxiv.org/abs/2003.00381 — sample-size requirements for cluster analysis
7. https://arxiv.org/abs/2510.07746 — t-SNE distortion analysis
8. https://arxiv.org/abs/2110.02573 — reliability of neighbor-embedding structure
9. https://distill.pub/2016/misread-tsne — How to Use t-SNE Effectively
10. https://simplystatistics.org/posts/2024-12-23-biologists-stop-including-umap-plots-in-your-papers — UMAP cluster fabrication
11. https://pmc.ncbi.nlm.nih.gov/articles/PMC3023458 — cluster validation practice
12. https://arxiv.org/abs/2310.17157 — Deja Vu: contextual sparsity
13. https://arxiv.org/abs/2301.04213 — Does Localization Inform Editing? (Hase et al.)
14. https://arxiv.org/html/2409.03856v1 — Sirius: contextual-sparsity failure on reasoning/generation
15. https://arxiv.org/html/2505.21191v1 — instruction-specific neurons
16. https://arxiv.org/html/2407.06488 — task-specific neurons and generalization
17. https://arxiv.org/html/2605.24846v2 — keystone neurons
