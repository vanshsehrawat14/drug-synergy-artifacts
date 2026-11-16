# Authors

Author order follows the accompanying manuscript.

All authors are affiliated with the University of Nevada, Las Vegas.

| Author | Department | Institutional email |
|---|---|---|
| Vanshdeep Sehrawat | Computer Science | [sehrawat@unlv.nevada.edu](mailto:sehrawat@unlv.nevada.edu) |
| Kingsley Ukandu | Mathematical Sciences | [Kingsley.ukandu@unlv.edu](mailto:Kingsley.ukandu@unlv.edu) |
| Sharlene Kim | Brain Health | [sharlene.kim@unlv.edu](mailto:sharlene.kim@unlv.edu) |

| Author | Contribution | Implementation |
|---|---|---|
| [Vanshdeep Sehrawat](https://github.com/vanshsehrawat14) | Neural models, original-label DeepSynergy port and dose-aware adaptation; shared data, splits, runner and metrics; GP and CF extensions; interaction, replicate, Loewe and screening analyses; manuscript and figures | `synergy_artifacts/`, `experiments/`, `manuscript/` |
| [Kingsley Ukandu](https://github.com/kingsleyukandu) | Original Gaussian-process framework, pair kernel, drug-dose representation, kernel selection, fitting and evaluation | `gaussian_process/src/project_utils.py` |
| [Sharlene Kim](https://github.com/sharkim-source) | Original neighbourhood collaborative filtering, response-target construction, tuning and five-fold evaluation | `experiments/neighborhood_cf/` |

The [GP report](reports/gp.md) and [CF report](reports/cf.md) distinguish the original frameworks from their extensions. All three authors are listed in [CITATION.cff](CITATION.cff) and the [MIT notice](LICENSE).
