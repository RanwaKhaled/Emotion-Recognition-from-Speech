# Emotion Detection and Speaker Recognition Application
### Folder Structure:
```
main
|__ models
    |__ contains: scaler parameters, PCA model, Logistic regression model, emotion detection model
|__ notebooks
    |__ contains: notebooks used to train sentiment and speaker ID models
|__ assets
    |__ pictures and icons in the user interface
|__ python files
    |__ user interface code
|__ RAVDESS dataset zip file
    |__ sentiment dataset
```
---
## Setup Instructions

Follow the steps below to run the Emotion Detection and Speaker Recognition application:

1. **Clone the repository**  

   ```bash
   git clone https://github.com/RanwaKhaled/Emotion-Recognition-from-Speech.git
   cd Emotion-Recognition-from-Speech
   ```

2. **Install the required dependencies**  
   Make sure you have Python 3.9+ installed, then install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**  
   Launch the user interface:

   ```bash
   flet run app.py
   ```

That’s it — the interface should open and the models will load automatically.

---

##  RAVDESS Dataset Accreditation
```
@dataset{livingstone_2018_1188976,
  author       = {Livingstone, Steven R. and
                  Russo, Frank A.},
  title        = {The Ryerson Audio-Visual Database of Emotional
                   Speech and Song (RAVDESS)},
  month        = apr,
  year         = 2018,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.1188976},
  url          = {https://doi.org/10.5281/zenodo.1188976},
}
```

---
