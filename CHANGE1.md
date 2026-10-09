# MTKD4SER – Phase 1 (Language-Aware fixes)

Code đã sửa để chạy trên **Google Colab Free (T4 15 GB)**.

## Những gì đã thay đổi so với paper gốc

| Vấn đề gốc | Cách sửa trong Phase 1 |
|------------|------------------------|
| Cosine trên logits 4-dim không language-aware | Thêm 4 routing mode: `language` (oracle), `uniform`, `cosine`, `confidence` |
| Teacher `.train()` + có grad | Luôn `.eval()` + `torch.no_grad()` |
| 2 optimizer, chỉ step 1 | Chỉ còn 1 AdamW trên student |
| Cosine weight không detach | `.detach()` bắt buộc |
| `dev = test` (IEMOCAP/CaFE) | Tạo dev thật bằng stratified split từ train |
| Không có `language_id` | Mọi sample mang `language_id` (0=EN, 1=FI, 2=FR) |
| Teacher path hard-code cluster | Path truyền qua argparse, lỗi rõ nếu thiếu |
| Ghi đè checkpoint mỗi epoch | Chỉ lưu **best** theo val UR + early stopping |
| Backbone cố định wav2vec2-base | `--BACKBONE` linh hoạt (xls-r-300m, wavlm, …) |

## Cấu trúc

```
mtkd4ser_phase1/
├── main.py          # entry point
├── data.py          # datasets + language_id + real dev split
├── models.py        # FT / KD / MTKD model factories
├── pipeline.py      # training loops, best-ckpt, early stop
├── utils.py         # losses, routing, metrics, collate
└── README_PHASE1.md
```

## Quy trình chạy trên Colab Free

### 0. Setup

```python
# Cell 1
!pip install -q transformers datasets torchaudio librosa soundfile scikit-learn tqdm tensorboard
!git clone <your-fork-or-upload-the-folder>
%cd mtkd4ser_phase1
```

Mount Drive nếu muốn lưu checkpoint lâu dài:

```python
from google.colab import drive
drive.mount('/content/drive')
# sau đó dùng --SAVE_DIR /content/drive/MyDrive/mtkd_ckpts
```

### 1. Chuẩn bị dữ liệu

Xem file Word “Giai đoạn 0”. Tóm tắt nhanh:

- **CaFE**: Zenodo `CaFE_48k.zip` → giải nén vào `/content/CaFE`
- **IEMOCAP**: Kaggle `dejolilandry/iemocapfullrelease` + json splits
- FESC: bỏ qua nếu không có (`--INCLUDE_FI 0`)

### 2. Train monolingual teachers (bắt buộc trước MTKD)

```bash
# English teacher (session 1)
python main.py --PARADIGM FT --LANGUAGE EN --LINGUALITY Monolingual \
  --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 \
  --TRAINING 1 --SAVE_DIR /content/checkpoints

# French teacher
python main.py --PARADIGM FT --LANGUAGE FR --LINGUALITY Monolingual \
  --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 8 --LEARNING_RATE 3e-5 \
  --TRAINING 1 --SAVE_DIR /content/checkpoints
```

### 3. Chạy MTKD ablation (quan trọng nhất)

```bash
# Oracle language routing (kỳ vọng tốt nhất)
python main.py --PARADIGM MTKD --LANGUAGE EN --LINGUALITY Multilingual \
  --SESSION 1 --N_EPOCHS 12 --BATCH_SIZE 4 --LEARNING_RATE 3e-5 \
  --TRAINING 1 --ROUTING language \
  --TEACHER_EN_CKPT /content/checkpoints/FT_Monolingual_EN_S1.pth \
  --TEACHER_FR_CKPT /content/checkpoints/FT_Monolingual_FR_S1.pth \
  --INCLUDE_FI 0 --SAVE_DIR /content/checkpoints

# Baseline uniform
python main.py ... --ROUTING uniform ...

# Original paper method
python main.py ... --ROUTING cosine ...
```

### 4. Gợi ý batch size (T4 15 GB)

| Backbone | FT / KD batch | MTKD (2 teachers: EN+FR) | MTKD (3 teachers) |
|----------|---------------|--------------------------|-------------------|
| wav2vec2-base | 8–16 | 6–12 | 4–8 |
| xls-r-300m | 4–8 | 4–6 | 2–4 |

Dùng `torch.cuda.empty_cache()` giữa các run nếu cần.

## Checklist Phase 1

- [x] `language_id` trên mọi sample
- [x] Dev set riêng (không còn dev=test)
- [x] Teacher eval + no_grad
- [x] Single optimizer, weight detach
- [x] 4 routing modes
- [x] Best-checkpoint + early stopping
- [x] Test + confusion matrix cuối
- [x] Chạy được trên Colab Free

## Bước tiếp theo (Giai đoạn 2+)

- Đổi student sang `facebook/wav2vec2-xls-r-300m`
- Language / class balanced sampler
- Multi-seed + significance test
- Phân tích phân phối weight theo language thật (đã print sẵn trong log)
