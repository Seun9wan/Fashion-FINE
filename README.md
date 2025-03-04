# FashionFINE: Integration of global and local representations for fine-grained cross-modal alignment

FashionFINE is a framework for fine-grained cross-modal retrieval in fashion-related datasets. This paper is accepted by The European Conference on Computer Vision (ECCV 2024) [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10886.pdf).

---

## 📌 Requirements
To install dependencies, run:
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation
FashionFINE follows the dataset preparation process of [FashionSAP](https://github.com/hssip/FashionSAP).

### 🛍️ FashionGen
1. Download and extract the raw dataset to `data_root`.
2. Update `data_root` and `split` in `prepare_dataset.py`.
3. Run the script to generate the assistance file:
   ```bash
   python prepare_dataset.py
   ```
4. Ensure the original FashionGen images are stored in:
   ```
   data_root/images
   ```

### 🏷️ FashionIQ
1. Download and extract the raw dataset to `data_root`.
2. Move the `captions` and `images` directories into `data_root`.
3. Merge all train files into `cap.train.json` within `captions`, and do the same for validation files.
4. Ensure FashionIQ images are stored in:
   ```
   data_root/tgir_images
   ```

---

## 🔥 Pre-training
### 1️⃣ Download Pre-trained Model
Download the pre-trained model from the link below and place it in `checkpoint_pretrain`:
[Download Model](https://drive.google.com/file/d/16kxbK7u86jVUfkwM7_4q2lJhCYeutgRv/view?usp=sharing)

### 2️⃣ Start Pre-training
Run the following commands:
```bash
cd checkpoint_pretrain
cd ../
bash pt.sh
```

---

## 🎯 Fine-tuning & Evaluation
### 🚀 Fine-tuning Cross-modal Retrieval
```bash
bash retrieval.sh
```

### 📊 Evaluate Cross-modal Retrieval
1. Download the fine-tuned checkpoint from:
   [Download Model](https://drive.google.com/file/d/1IRAs-UG8cwtogEWPYLFetyG8jJ-7mJuz/view?usp=sharing)
2. Run evaluation:
   ```bash
   bash eval_cmr.sh
   ```

### 🔍 Fine-tuning Text-guided Image Retrieval
```bash
bash tgir.sh
```

### 📈 Evaluate Text-guided Image Retrieval
1. Download the fine-tuned checkpoint from:
   [Download Model](https://drive.google.com/file/d/1e5tF-QWM2RZa5W4My7SdiOJF2jl3sydN/view?usp=sharing)
2. Run evaluation:
   ```bash
   bash eval_TGIR.sh
   ```

---

## 📌 FashionFINE on BLIP
### 1️⃣ Download Pre-trained BLIP Model
```bash
cd FashionFINE_BLIP
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth
```

### 2️⃣ Fine-tuning FashionFINE on BLIP
```bash
bash fashionfine_blip.sh
```

### 3️⃣ Evaluate Retrieval with Fine-tuned Checkpoint
1. Download the checkpoint from:
   [Download Model](https://drive.google.com/file/d/1rjQXvixkCYwOgC2QcjrQMRIhFxLR0IA4/view?usp=sharing)
2. Run evaluation:
   ```bash
   cd ../checkpoint
   cd ../FashionFINE_BLIP

## Citations

If you find this code useful for your research, please cite:

```
@inproceedings{jin2024integration,
  title={Integration of global and local representations for fine-grained cross-modal alignment},
  author={Jin, Seungwan and Choi, Hoyoung and Noh, Taehyung and Han, Kyungsik},
  booktitle={European Conference on Computer Vision},
  pages={53--70},
  year={2024},
  organization={Springer}
}
```

   bash eval_fashionfine_blip.sh
   ```
