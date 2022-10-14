# tso_vace_wpe
Official implementation of the deep neural network (DNN)-based weighted prediction error (WPE) algorithm and virtual acoustic channel expansion (VACE)-WPE algorithm variants presented in [[1]](#1).

<a id="1">[1]</a> 
J.-Y. Yang and J.-H. Chang, "Task-specific Optimization of Virtual Channel Linear Prediction-based Speech Dereverberation Front-End for Far-Field Speaker Verification," *IEEE/ACM Trans. Audio, Speech, Lang. Process.*, vol. 30, pp. 3144--3159, 2022. ([link](https://ieeexplore.ieee.org/document/9889165))


## DNN-WPE and VACE-WPE
* DNN-WPE (or neural WPE): Exploits a DNN to estimate the power spectra of the desired (dereverberated) signal.
* VACE-WPE: A neural WPE variant designed to exploit dual-channel neural WPE algorithm in a single-microphone setup. Employs another DNN(=VACENet) to generate a virtual signal, and the pair of actual (observed) signal and virtual signal are directly introduced to the dual-channel neural WPE. The VACENet is trained end-to-end such that a desired target signal obtained as the output of the dual-channel neural WPE.


## Run
```
python run.py
```
* Neural WPE: Neural WPE algorithm for speech dereverberation.
* Drv-VACE-WPE: VACE-WPE trained to produce noisy early-arriving signals.
* Dns-VACE-WPE: VACE-WPE trained to produce noise-free early-arriving signals.
* TSO_N-VACE-WPE: Additionially fine-tuned Drv-VACE-WPE to produce noisy early-arriving signals, but within the task-specific optimization (TSO) framework by using a pretrained deep speaker embedding (DSE) model.
* TSO_C-VACE-WPE: Additionially fine-tuned Drv-VACE-WPE to produce noise-free early-arriving signals, but within the TSO framework by using a pretrained DSE model.
* DR-TSO_C-VACE-WPE: Additionially fine-tuned Drv-VACE-WPE to produce noise-free early-arriving signals, but within the distortion-regularized (DR) TSO framework by using a pretrained DSE model.

## Reference codes for WPE implementation
* [NARA-WPE](https://github.com/fgnt/nara_wpe)
