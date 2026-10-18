# LACE: Latent Alignment via Counterfactual Embeddings

**Harsh Udai, Konda Reddy Mopuri**
Department of Artificial Intelligence, IIT Hyderabad

*NeurIPS 2026*

[Paper (coming soon)] · [OpenReview (coming soon)] · [Code: coming soon]

> 🚧 **Code release in progress.** This repository currently hosts the project overview. Training, evaluation, and dictionary-construction code will be released here shortly. Watch or star the repo to be notified.

---

## Overview

CLIP-style vision–language models align images and text well at a coarse level, but often fail on **compositional** distinctions, such as telling *"a red suitcase next to a black backpack"* apart from *"a black suitcase next to a red backpack."*

The common fix is to add **text-side hard negatives** (caption swaps, negations, perturbations). This puts all the fine-grained pressure on the text branch. The image branch gets no comparable near-miss signal, which can distort the shared embedding space: hubness goes up and neighborhood reciprocity goes down, even when Recall@K improves.

**LACE** generates **counterfactual hard negatives for both modalities directly in embedding space**. It needs no pixel editing, diffusion models, or retrieval mining.

1. **Sparse concept decomposition.** Each image or text embedding is expressed as a sparse mixture of atoms from a fixed dictionary of object, attribute, and relation concepts.
2. **Latent counterfactual synthesis.** A small active support is edited by attenuation, swap, or transplant to a same-type concept. The embedding is then reconstructed, giving a targeted near-miss that preserves the surrounding context.
3. **Symmetric training.** Text-side counterfactuals are negatives for image→text retrieval, and image-side counterfactuals are negatives for text→image retrieval.

LACE changes **only training**. At inference the model is a standard dual encoder, with no dictionary lookup, sparse solver, or extra module.

## Highlights

- **Image- and text-side hard negatives** from a single latent editing operator, generated on the fly
- **Better compositionality** on COLA, What'sUp, Winoground, SugarCrepe/SugarCrepe++, and MMVP-VLM
- **Healthier embedding geometry**: higher mutual reciprocity, lower hubness, fewer brittle local edits
- **Works across objectives**: softmax InfoNCE (OpenCLIP) and pairwise sigmoid (SigLIP)
- **No inference overhead**

## Results

### Controlled comparison (OpenCLIP ViT-B/32, MS-COCO fine-tuning)

Same initialization, data, LoRA configuration, optimizer, and schedule. The only difference is LACE's counterfactual supervision.

| Model | What'sUp | COLA | Winoground | SC | SC++ | MMVP-VLM | Avg. |
|---|---|---|---|---|---|---|---|
| Matched fine-tuning | 42.12 | 35.56 | 15.83 | 82.80 | 64.30 | 29.62 | 45.04 |
| **+ LACE** | **43.56** | **39.05** | **17.42** | **83.52** | **65.22** | **35.56** | **47.39** |

### Cross-objective: SigLIP ViT-B/16

| Model | What'sUp | COLA | Winoground | SC | SC++ | MMVP-VLM | Avg. |
|---|---|---|---|---|---|---|---|
| SigLIP fine-tuning | 45.00 | 35.00 | 18.00 | 80.00 | 65.00 | 32.00 | 45.83 |
| **+ LACE** | **47.00** | **41.00** | **19.00** | **84.00** | **68.00** | **37.78** | **49.46** |

### Embedding geometry (CC3M-50k, K = 10)

| Model | Mutual@10 ↑ | Edge recip.@10 ↑ | I2T Gini ↓ | Relation-swap brittle ↓ | Attribute-swap brittle ↓ |
|---|---|---|---|---|---|
| Matched fine-tuning | 46.57 | 52.46 | 0.480 | 3.27% | 4.65% |
| **+ LACE** | **51.41** | **53.79** | **0.455** | **1.10%** | **1.60%** |

See the paper for full comparisons with DAC, CLoVe, CE-CLIP, FSC-CLIP, CLIC, NegCLIP, and TripletCLIP, plus ablations and per-split results.

## Planned code release

```
LACE/
├── configs/              # OpenCLIP and SigLIP training configs
├── dictionary/           # concept extraction (spaCy + VLM), filtering, embedding
├── lace/
│   ├── sparse_coding.py  # batched FISTA / LASSO solver
│   ├── counterfactual.py # editable-set filtering, attenuation / swap / transplant
│   └── losses.py         # symmetric InfoNCE and SigLIP counterfactual losses
├── train.py              # LoRA fine-tuning (Accelerate)
├── eval/                 # compositional benchmarks, retrieval, zero-shot
└── geometry/             # hubness, reciprocity, local-edit diagnostics
```

- [ ] Concept dictionary construction scripts and released dictionaries
- [ ] Sparse coding and counterfactual generation
- [ ] OpenCLIP and SigLIP training code
- [ ] Evaluation on COLA, What'sUp, Winoground, SugarCrepe/SugarCrepe++, MMVP-VLM, retrieval, ImageNetV2
- [ ] Geometry diagnostics (hubness, reciprocity, local edit stability)
- [ ] Trained LoRA checkpoints

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{udai2026lace,
  title     = {{LACE}: Latent Alignment via Counterfactual Embeddings},
  author    = {Udai, Harsh and Mopuri, Konda Reddy},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

This work builds on [OpenCLIP](https://github.com/mlfoundations/open_clip), [SigLIP](https://arxiv.org/abs/2303.15343), and the public compositionality benchmarks listed above. We thank their authors for releasing code and data.

## License

This project is released under the [MIT License](LICENSE). Datasets, benchmarks, and pretrained checkpoints used with this code remain under their own licenses.

## Contact

Harsh Udai: ai23resch11003@iith.ac.in
Konda Reddy Mopuri: krmopuri@ai.iith.ac.in
