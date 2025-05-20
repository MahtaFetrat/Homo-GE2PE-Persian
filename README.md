# Homo-GE2PE: Persian Grapheme-to-Phoneme Conversion with Homograph Disambiguation


**Homo-GE2PE** is a Persian grapheme-to-phoneme (G2P) model specialized in homograph disambiguation—words with identical spellings but context-dependent pronunciations (e.g., *مرد* pronounced as *mard* "man" or *mord* "died"). Introduced in *[Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models](https://arxiv.org/abs/2505.12973)*, the model extends **GE2PE** by fine-tuning it on the **HomoRich** dataset, explicitly designed for such pronunciation challenges.  

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

Here’s a merged version that combines your existing intro with the condensed usage instructions:

---

## Usage  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Osue8HOgTGMZXIhpvCuiRyfuxpte1v0p?usp=sharing)  

For inference, run the provided [`inference.ipynb`](https://github.com/MahtaFetrat/Homo-GE2PE-Persian/blob/main/Inference.ipynb) notebook either locally or via the [Colab link](https://colab.research.google.com/drive/1Osue8HOgTGMZXIhpvCuiRyfuxpte1v0p?usp=sharing) (recommended for easy setup).

### Quick Setup  
1. **Install dependencies**:  
   ```bash
   pip install unidecode
   ```  

2. **Download models**:  
   ```bash
   git clone https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian/
   unzip -q Homo-GE2PE-Persian/assets/Parsivar.zip
   unzip -q Homo-GE2PE-Persian/model-weights/homo-ge2pe.zip -d homo-ge2pe
   unzip -q Homo-GE2PE-Persian/model-weights/homo-t5.zip -d homo-t5
   mv Homo-GE2PE-Persian/assets/GE2PE.py ./
   ```  

3. **Fix compatibility** (if needed):  
   ```bash
   sed -i 's/from collections import Iterable/from collections.abc import Iterable/g' Parsivar/token_merger.py
   ```  

### Example Usage  
```python
from GE2PE import GE2PE

g2p = GE2PE(model_path='/content/homo-ge2pe') # or homo-t5
g2p.generate(['تست مدل تبدیل نویسه به واج', 'این کتابِ علی است'], use_rules=True)

# Output: ['teste model t/bdil nevise be vaj', '@in ketabe @ali @/st']
```  
---

## Dataset: HomoRich G2P Persian

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-dataset-orange)](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian)

The models in this repository were fine-tuned on HomoRich, the first large-scale public Persian homograph dataset for grapheme-to-phoneme (G2P) tasks, resolving pronunciation/meaning ambiguities in identically spelled words. Introduced in *[Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models](https://arxiv.org/abs/2505.12973)*, the dataset is available [here](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian).

---

## Citation

If you use this project in your work, please cite the corresponding paper:

```bibtex
@misc{qharabagh2025fastfancyrethinkingg2p,
      title={Fast, Not Fancy: Rethinking G2P with Rich Data and Rule-Based Models}, 
      author={Mahta Fetrat Qharabagh and Zahra Dehghanian and Hamid R. Rabiee},
      year={2025},
      eprint={2505.12973},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.12973}, 
}
```
---

## Contributions

Contributions and pull requests are welcome. Please open an issue to discuss the changes you intend to make.

---

### Additional Links

* [Link to Paper](https://arxiv.org/abs/2505.12973)
* [Homo-GE2PE (Huggingface)](https://huggingface.co/MahtaFetrat/Homo-GE2PE-Persian)
* [Base GE2PE Paper](https://aclanthology.org/2024.findings-emnlp.196/)
* [Base GE2PE Model](https://github.com/Sharif-SLPL/GE2PE)
* [HomoRich Dataset (Huggingface)](https://huggingface.co/datasets/MahtaFetrat/HomoRich-G2P-Persian)
* [HomoRich Dataset (Github)](https://github.com/MahtaFetrat/HomoRich-G2P-Persian)
* [SentenceBench Persian G2P Benchmark](https://huggingface.co/datasets/MahtaFetrat/SentenceBench)
