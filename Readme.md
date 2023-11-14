# **ECG-Healthcare-Biometrics**
---

1. Experimentation documentation consists of details about the datasets used, network and experimentation protocols (as well as the obtained results). Link to the [Doc](https://docs.google.com/document/d/1OG9AVy24xzk5IjuLsir8rr-9SLH3W2ByKJwE6MGqhNw/edit?usp=sharing)
2. We have used 3 Datasets (MIT-BIH, ECG-1D, PTB), all of these can be found on [PhysioNet](https://physionet.org/about/database/). There are multiple other datasets which can be checked.
3. I have shared most of the codes in the Notebooks folder. It comprises code to Arc-Loss, Transformer Network, SCNR-Net model as well as the preprocessing chains.
4. Dataset and Models are here on [Google Drive]()
5. Key Challenges (which I remember):
   *  There was no coherent experimentation protocol across the literature, therefore we had experimented with a lot many protocols. Open-Set with different time-interval based splitting of ECG Signals was one such example, especially for ECG-1D.
   *  Interpretability of some of the results on PTB was not there, I remember us doing some tSNE based analysis to seek separability across the subjects in the dataset.
   *  Model did recognize users when most of the signals were masked: What factors actually influenced high performance was exactly not clear? I am pointing towards interpretability of models, specifically for such signals.
   *  There was challenges in pre-processing (again very weakly defined in the literature)
