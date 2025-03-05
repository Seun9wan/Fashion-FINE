# Fashion-FINE: Integration of global and local representations for fine-grained cross-modal alignment

Fashion-FINE is a framework for fine-grained cross-modal retrieval in fashion-related datasets. This paper is accepted by The European Conference on Computer Vision (ECCV) 2024 [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10886.pdf).

---

## 📌 Requirements
To install dependencies, run:
```
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation
Fashion-FINE follows the dataset preparation process of [FashionSAP](https://github.com/hssip/FashionSAP).

### 🛍️ FashionGen
1. Download and extract the raw dataset to `data_root`.
2. Update `data_root` and `split` in `prepare_dataset.py`.
3. Run the script to generate the assistance file:
   ```
   python prepare_dataset.py
   ```
4. Ensure the original FashionGen images are stored in:
   ```
   data_root/images
   ```

### 🏷️ FashionIQ
1. Download and extract the raw dataset to `data_root`.
2. Move the `captions` and `images` directories into `data_root`.
3. Ensure FashionIQ images are stored in:
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
```
bash pt.sh
```

---

## 🎯 Fine-tuning & Evaluation
### 🚀 Fine-tuning Cross-modal Retrieval
```
bash retrieval.sh
```

### 📊 Evaluate Cross-modal Retrieval
1. Download the fine-tuned checkpoint from:
   [Download Model](https://drive.google.com/file/d/1IRAs-UG8cwtogEWPYLFetyG8jJ-7mJuz/view?usp=sharing)
2. Run evaluation:
   ```
   bash eval_cmr.sh
   ```

### 🔍 Fine-tuning Text-guided Image Retrieval
```
bash tgir.sh
```

### 📈 Evaluate Text-guided Image Retrieval
1. Download the fine-tuned checkpoint from:
   [Download Model](https://drive.google.com/file/d/1e5tF-QWM2RZa5W4My7SdiOJF2jl3sydN/view?usp=sharing)
2. Run evaluation:
   ```
   bash eval_TGIR.sh
   ```

---

## 📌 Fashion-FINE on BLIP
### 1️⃣ Download Pre-trained BLIP Model
```bash
cd Fashion-FINE_BLIP
wget https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth
```

### 2️⃣ Fine-tuning Fashion-FINE on BLIP
```
bash fashionfine_blip.sh
```

### 3️⃣ Evaluate Retrieval with Fine-tuned Checkpoint
1. Download the checkpoint from:
   [Download Model](https://drive.google.com/file/d/1rjQXvixkCYwOgC2QcjrQMRIhFxLR0IA4/view?usp=sharing)
2. Run evaluation:
   ```
   cd ../FashionFINE_BLIP
   bash eval_fashionfine_blip.sh
   ```

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
