# Controllable Symbolic Music Generation

## Project Overview

This project explores controllable symbolic music generation using a Transformer-based architecture.
Our goal is to generate expressive piano solos that follow user-specified chord progressions, enabling interactive and harmonically guided music creation.

## Dataset

We collected a dataset of pop piano performances and converted them into event-based sequences using the REMI tokenization method.
To support chord control, we extended REMI by inserting two chord tokens per bar — representing the harmony in the first and second halves — which we call ChordREMI.

## Code Structure

| File | Description |
| --- | --- |
| `hello_remi.py` | Implements an extended version of REMI tokenization based on the miditok library. Adds chord tokens to support harmonic conditioning, forming the basis of our ChordREMI method. |
| `infer.py` | Used for generating piano music. Loads the trained model and performs inference based on user-defined chord inputs. |
| `tokenizer_exp_02.json` | Vocabulary file that defines the token set used during training and generation, including pitch, duration, bar, position, and chord tokens. |
| `train.ipynb` | Jupyter Notebook used to train the model. Handles data loading, model initialization, training loop, and evaluation. |

## Usage

### Generating Music with Chord Progressions

The basic usage is:

```bash
python infer.py --chords C_maj G_maj A_min F_maj --output output.mid
```

Where:
- `--chords`: List of chords in the format `root_type` (e.g., C_maj, D_min, G_maj)
- `--output`: Output MIDI filename (default: "output.mid")
- `--temperature`: Temperature for generation (0.1-1.0, default: 0.9)
- `--max_tokens`: Maximum number of tokens to generate (default: 1024)

- 
### Model Download

Download our pre-trained model from [this Google Drive link](https://drive.google.com/file/d/1P92lUBfaDp65xHGGl88UNoz9gvmjDddT/view?usp=sharing) and extract it to the `Models/` directory:

```bash
mkdir -p Models/
# After downloading, move the files to the Models/checkpoint-6200 directory
```


## Results and Key Findings

- Our model achieves 95% accuracy in matching generated melodies to given chord progressions.
- Listening tests show that users generally prefer our ChordREMI-based outputs over baseline models, indicating stronger harmonic alignment and musicality.

## Team Contributions

- **Kaiqi Chen**: Data collection, organization, and model evaluation
- **Linhan Wang**: Data preprocessing
- **Yuke Tian**: Chord detection
- **Juncheng Yu**: Model architecture design, training, and MIDI tokenization implementation


