# Constructing Per-Shot Bitrate Ladders using Visual Information Fidelity

The following is the official implementation of **Constructing Per-Shot Bitrate Ladders using Visual Information Fidelity**. This is built upon our prior work titled **Bitrate Ladder Construction using Visual Information Fidelity**. The code utilized in the experiments for the previous work bears a strong resemblance to the one used here.

## Introduction and Abstracts
Video service providers need their delivery systems to be able to adapt to network conditions, user preferences, display settings, and other factors. HTTP Adaptive Streaming (HAS) provides numerous options that allow dynamic switching between different video representations to simultaneously enhance bandwidth consumption and users’ streaming experiences. Fixed bitrate ladders as employed in the past are limited in their ability to deliver high-quality visual experiences while minimizing bitrate budgets. Adaptive video streaming allows for the construction of bitrate ladders that deliver perceptually optimized visual quality to viewers under bandwidth constraints. Two common approaches to adaptation are per-title encoding and per-shot encoding. The former involves encoding each program, movie, or other content in a manner that is perceptually- and bandwidth-optimized for that content but is otherwise fixed. The latter is a more granular approach that optimizes the encoding parameters for each scene or shot (however defined) of a video content. Per-shot video encoding, as pioneered by Netflix, encodes on a per-shot basis using the Dynamic Optimizer (DO). Under the control of the VMAF perceptual video quality prediction engine, the DO delivers high-quality videos to millions of viewers at considerably reduced bitrates than per-title or fixed bitrate ladder encoding. A variety of per-title and per-shot encoding techniques have been recently proposed that seek to reduce computational overhead and to construct optimal bitrate ladders more efficiently using low-level features extracted from source videos.

We deploy features drawn from Visual Information Fidelity (VIF) (VIF features) extracted from uncompressed videos to predict the visual quality (VMAF) of compressed videos. We present multiple VIF feature sets extracted from different scales and subbands of a video to tackle the problem of bitrate ladder construction. We also develop a perceptually optimized method of constructing optimal per-shot bitrate and quality ladders, using an ensemble of low-level features and Visual Information Fidelity (VIF) features extracted from different scales and subbands. We compare the performance of our models, against other content-adaptive bitrate ladder prediction methods, counterparts of them that we designed to construct quality ladders, a fixed bitrate ladder, and bitrate ladders constructed via exhaustive encoding using Bjontegaard delta metrics.

## Papers and Citations
Constructing Per-Shot Bitrate Ladders using Visual Information Fidelity: [https://arxiv.org/pdf/2408.01932](https://arxiv.org/pdf/2408.01932)
```
@misc{durbha2023bitrate,
	title          = {Constructing Per-Shot Bitrate Ladders using Visual Information Fidelity},
	author         = {Krishna Srikar Durbha and Alan C. Bovik},
	year           = {2024},
	eprint         = {2408.01932},
	archiveprefix  = {arXiv},
	primaryclass   = {eess.IV}
}
```

Bitrate Ladder Construction using Visual Information Fidelity: [https://arxiv.org/pdf/2312.07780.pdf](https://arxiv.org/pdf/2312.07780.pdf)
```
@inproceedings{10566405,
	title          = {Bitrate Ladder Construction Using Visual Information Fidelity},
	author         = {Durbha, Krishna Srikar and Tmar, Hassene and Stejerean, Cosmin and Katsavounidis, Ioannis and Bovik, Alan C.},
	year           = 2024,
	booktitle      = {2024 Picture Coding Symposium (PCS)},
	volume         = {},
	number         = {},
	pages          = {1--4},
	doi            = {10.1109/PCS60826.2024.10566405}
}
```