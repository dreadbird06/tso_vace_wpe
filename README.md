# tso_vace_wpe
Official implementation of the deep neural network (DNN)-based weighted prediction error (WPE) algorithm and virtual acoustic channel expansion (VACE)-WPE algorithm variants presented in [[1]](#1).

<a id="1">[1]</a> 
J.-Y. Yang and J.-H. Chang, "Task-specific Optimization of Virtual Channel Linear Prediction-based Speech Dereverberation Front-End for Far-Field Speaker Verification," *arXiv:2112.13569*, 2021. ([link](https://arxiv.org/abs/2112.13569))


## Background
* DNN-WPE (or neural WPE): Exploits a DNN to estimate the power spectra of the desired (dereverberated) signal.
* VACE-WPE: A neural WPE variant designed to exploit dual-channel neural WPE algorithm in a single-microphone setup. Employs another DNN to generate a virtual signal, and the pair of actual (observed) signal and virtual signals are directly introduced to the dual-channel neural WPE.


## Run
```
python run.py
```
* Neural WPE: Neural WPE algorithm for speech dereverberation
* Drv-VACE-WPE: VACE-WPE trained to output noisy early-arriving signals.
* Dns-VACE-WPE: VACE-WPE trained to output noise-free early-arriving signals.
* TSO_\mathcal{N}-VACE-WPE

## Acknowledgments

Inspiration, code snippets, etc.
* [dbader](https://github.com/dbader/readme-template)
