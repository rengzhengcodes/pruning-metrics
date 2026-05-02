1. **Background:** Machine learning models are often evaluated by how closely their output distributions match a target distribution. *Kullback–Leibler divergence* (KLD) is commonly used to quantify deviation from the target distribution. This is employed in tasks such as quantization, pruning, and distillation, where one starts from a large model and deliberately perturbs it in exchange for better *computational efficiency* (i.e., wall-clock compute time/CPU or GPU cycles). In such settings, there is a natural tension between efficiency and preservation of behavior, and KLD and related divergence-based losses are commonly used as proxies for preservation of output behavior. However, KLD is asymmetric and does not define a metric space, so it is limited as a tool for comparing many related models to one another.

**Motivation and Proposal:** A longer-term research goal of mine is to find a measure that determines whether *n* experts inside a mixture-of-experts (MoE) model are functionally similar enough to be merged, swapped, or otherwise treated as redundant, which can help reduce MoE parameter counts. For this project, I propose to study a simpler prototype: families of task-specific *pruned models* derived from a common base model. The intuition is that pruning on one task induces specialization, and the resulting family of specialized models can serve as a stand-in for the later MoE setting, where the “objects being compared” would be experts rather than whole pruned models.

I will use measures such as square root Jensen-Shannon Divergence (JSD), Chamfer Distance (CD), and Earth Mover’s Distance (EMD). The main technical difficulty is that each measure is usually used in somewhat different settings, so one must define a common denominator before the comparison is meaningful. I therefore propose the following shared representation. For each next-token prediction site in the answer, I will represent the next-token output of a model as a probability measure on a *shared discrete support*[^1] of tokens, together with a geometric embedding of that support into R2. I will compute a single global embedding of the vocabulary into R2 by applying a fixed projection, such as PCA, to the base model’s token embedding or unembedding matrix. For each prompt-position pair, the shared support is then treated as a subset of these globally embedded tokens. This produces a shared representation on which all four distances can be computed: KLD and JSD compare the renormalized probability masses on the shared token support, while EMD and CD additionally use the geometric structure induced by the 2D embedding of that support.[^2]

**Hypothesis:** Symmetric or geometry-aware distances computed on a shared representation of output distributions will be more predictive of *cross-task brittleness* than KLD. In particular, I expect EMD to be more informative than KLD as pruning moves probability mass among outputs that are nearby under the chosen token embedding geometry. 

I expect JSD​ to improve over KLD by removing asymmetry, but to have difficulty separating very brittle models because it is bounded. I expect Chamfer distance to provide a cheaper geometric baseline, but to underperform EMD in some cases because it is a nearest-point measure and does not fully account for transported probability mass.

**Experiment:**

1. Select a base language model M.  
2. Choose a set T of verifiable text-only tasks. Suitable examples include coding tasks with unit-test evaluation, math tasks with exact numeric answers, and multiple-choice or exact-match reasoning tasks.  
3. For each task t in T, construct a family of task-specialized pruned models Mt, a​ at multiple pruning severities a.  
   1. I expect to use 3 task families and 5 pruning severities per task, yielding a manageable but nontrivial family of models.  
4. For each model Mt, a​, and for each evaluation task u in T, run the model under teacher forcing[^3] on a fixed benchmark set for u and record next-token distributions at each answer position. Teacher forcing allows for us to evaluate each model for the same conditionalized distribution.  
5. For each next token distribution k, define a shared support Si​ as the union of the smallest token sets carrying most of the probability mass under the base and pruned distributions (e.g., 99%). Renormalize both distributions onto Si.  
6. Using a single global R2 embedding of the vocabulary computed in advance, restrict attention to the points corresponding to the tokens in Si. This yields a local point set with masses on which all four distances can be computed.  
7. Compute, for each k, the measures KLD(k), JSD(k), EMD(k), CD(k), where we compare M against Mt,a.  Then aggregate them over all k to obtain per-task drift scores.  
8. Evaluate every Mt, a​ on all tasks in T. For each cross-task u \!= t, define the cross degradation Yt,a,u\= Perf(M ,u) − Perf(Mt,a, u). Define the cross-task brittleness of Mt, a​ as the average of Yt,a,u​ over all u \!= t. Separately record the lost performance Lt,a\=Perf(M, t) − Perf(Mt,, t).  
9. Compare how well each distance predicts brittleness. Concretely, I will measure rank correlation and regression fit between each metric and cross-task performance degradation, while controlling for pruning severity and lost performance.

   Concretely, for each cross-task evaluation row (t, a, u) with u \!= t, I will compute a specialization brittleness score and a distance score. I will then evaluate each metric using:  
   1. Spearman rank correlation between each distance score Dt, a, u​ and the cross-task degradation Yt, a, u​ per u.   
   2. For each metric, regress cross-task degradation Yt, a, u on the distance score Dt,a,u while controlling for pruning severity a, lost performance Lt, a and source/evaluation task effects. In other words, Yt,a,u​\=b0​+b1​Dt,a,u​+b2a+b3​Lt,a\+gt​+du​+et,a,u and we want the correlation with Dt,a,u to remain stronger even when information about pruning severity and lost performance is available  
   3. Fixed-task and pruning level analyses across cross tasks u, so that pruning severity is held constant and the analysis isolates whether the metric predicts which cross tasks break more severely.  
10. Compare wall-clock runtime for each metric as a function of support size and pruning severity to measure the tradeoff between predictive power and computational cost.

**Evaluation Criteria:** The project will be considered successful if it answers the following questions clearly:

* Which of KLD, JSD, EMD, and CD is most predictive of cross-task brittleness under task-specific pruning?  
* How computationally expensive are the geometry-aware distances relative to the information they provide?

[^1]:  A shared discrete support is a common finite set of tokens on which both the base-model and pruned-model output distributions are represented. For each token, we form this support by taking the union of the high-probability next tokens under both models and then renormalizing the two distributions on that union.

[^2]:  We could use the unembedded space, but that computation time may exceed the time left in the semester.

[^3]:  If a next-token prediction is wrong, the teacher forces the correct token in its place for the rest of the experiment.