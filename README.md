# Homo-GE2PE: Persian Grapheme-to-Phoneme Conversion with Homograph Disambiguation


**Homo-GE2PE** is a Persian grapheme-to-phoneme (G2P) model specialized in homograph disambiguation—words with identical spellings but context-dependent pronunciations (e.g., *مرد* pronounced as *mard* "man" or *mord* "died"). Introduced in *[Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models](link)*, the model extends **GE2PE** by fine-tuning it on the **HomoRich** dataset, explicitly designed for such pronunciation challenges.  

---

## Repository Structure

```
model-weights/
│   ├── homo-ge2pe.zip       # Homo-GE2PE model checkpoint
│   └── homo-t5.zip          # Homo-T5 model checkpoint (T5-based G2P model)

training-scripts/
│   ├── finetune-ge2pe.py    # Fine-tuning script for GE2PE
│   └── finetune-t5.py       # Fine-tuning script for T5

testing-scripts/
│   └── test.ipynb           # Benchmarking the models with SentenceBench Persian G2P Benchmark

assets/
│   └── (files required for inference, e.g., Parsivar, GE2PE.py)

```

---

### Model Performance

Below are the performance metrics for each model variant on the SentenceBench dataset:

| Model        | PER (%) | Homograph Acc. (%) | Avg. Inf. Time (s) |
| ------------ | ------- | ------------------ | ------------------ |
| GE2PE (Base) | 4.81    | 47.17              | 0.4464             |
| Homo-T5      | 4.12    | 76.32              | 0.4141             |
| Homo-GE2PE   | 3.98    | 76.89              | 0.4473             |

---

## Inference

For inference, use the provided `inference.ipynb` notebook or the [Colab link](https://colab.research.google.com/drive/1Osue8HOgTGMZXIhpvCuiRyfuxpte1v0p?usp=sharing). The notebook demonstrates how to load the checkpoints and perform grapheme-to-phoneme conversion using Homo-GE2PE and Homo-T5.

---

## Dataset: HomoRich G2P Persian

The models in this repository were fine-tuned on HomoRich, the first large-scale public Persian homograph dataset for grapheme-to-phoneme (G2P) tasks, resolving pronunciation/meaning ambiguities in identically spelled words. Introduced in "Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models", the dataset is available [here](https://huggingface.co/datasets/MahtaFetrat/HomoRich).

---

## Citation

If you use this project in your work, please cite the corresponding paper:

> TODO

---

## Contributions

Contributions and pull requests are welcome. Please open an issue to discuss the changes you intend to make.

---

### Additional Links

* [Paper PDF](#) (TODO: link to paper)
* [Base GE2PE Paper](https://aclanthology.org/2024.findings-emnlp.196/)
* [Base GE2PE Model](https://github.com/Sharif-SLPL/GE2PE)
* [HomoRich Dataset](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian)
* [SentenceBench Persian G2P Benchmark](https://huggingface.co/datasets/MahtaFetrat/SentenceBench)
