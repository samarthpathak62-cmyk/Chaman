# Neuron ~2B Model

**Target size: ~2.02 billion total parameters**

### Config
- hidden_size: 2048
- layers: 32
- experts: 8 (top-2)
- MLA + MoE architecture

### Dataset
- **HuggingFaceFW/fineweb-edu** (streaming=True)
- buffer_size: 20000
- max_steps: 50000 (adjust as needed)

### Kaggle Warning
2B model free GPU pe bahut tight hai.  
Agar OOM aaye to `num_hidden_layers` 28 pe le aa dena.

### Run
```bash
!pip install transformers datasets accelerate -q
!python train_neuron.py
```
