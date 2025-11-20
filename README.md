# Hướng dẫn chạy code Transfer Learning cho KL Grade Classification

## Mô tả
Notebook này thực hiện transfer learning sử dụng Vision Transformer (ViT) để phân loại mức độ thoái hóa khớp gối (KL Grade) từ dữ liệu MRI 3D.

## Yêu cầu hệ thống
- Python 3.8+
- CUDA 12.1+ (để chạy trên GPU)
- RAM: tối thiểu 16GB
- GPU: tối thiểu 8GB VRAM (khuyến nghị 16GB+)
- Dung lượng ổ cứng: ~50GB (cho dữ liệu và model)

## Bước 1: Chuẩn bị môi trường

### 1.1. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv mrivenv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 1.2. Cài đặt các thư viện cần thiết

**Cài đặt PyTorch với CUDA 12.1:**
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

**Cài đặt các thư viện phụ trợ:**
```bash
pip install -r requirements.txt
```

**Cài đặt ipykernel để chạy notebook:**
```bash
python -m ipykernel install --user --name=venv --display-name "Python (knee mri)"
```

## Bước 2: Tải và giải nén dữ liệu

### 2.1. Tải dữ liệu và nhãn từ Google Drive
```bash
gdown 1LB3OIrZOE7BkS96BfpskFgScAlso0DHy
gdown 15LP1xFpwZlYIjT93SmG_yW85zMnkZ3xl
```

### 2.2. Giải nén dữ liệu
```bash
unzip data.zip -d ./data
```

**Cấu trúc thư mục dữ liệu:**
```
./data/
├── SAG_3D_DESS_v2_full/
│   └── MRI_Numpy/
│       └── *.npz
└── unified_xray_mri_label.csv
```

## Bước 3: Cấu trúc thư mục dự án

```
project/
├── transfer.ipynb
├── requirements.txt
├── data/
│   ├── SAG_3D_DESS_v2_full/
│   └── unified_xray_mri_label.csv
├── pretrained/
│   └── vit_base_patch16_224_in21k.pth
└── output/
```

## Bước 4: Chạy notebook

### 4.1. Khởi động Jupyter và chọn Kernel đã tạo ở bước 2
```bash
jupyter notebook transfer.ipynb
```

### 4.2. Chỉnh sửa đường dẫn

**⚠️ QUAN TRỌNG:** Thay đổi đường dẫn dữ liệu trong notebook:

```python
# Từ:
mri_file = '/workspace/data/SAG_3D_DESS_v2_full/MRI_Numpy/' + path_object
df = pd.read_csv('/workspace/data/unified_xray_mri_label.csv')

# Thành:
mri_file = './data/SAG_3D_DESS_v2_full/MRI_Numpy/' + path_object
df = pd.read_csv('./data/unified_xray_mri_label.csv')
```

## Xử lý lỗi thường gặp

### CUDA Out of Memory
- Giảm `batch_size = 4` hoặc `2`
- Giảm `num_workers = 4` hoặc `2`

### File not found
- Kiểm tra đường dẫn dữ liệu
- Đảm bảo đã giải nén đúng thư mục

### Kernel died
- RAM không đủ → giảm batch size
- Restart kernel và chạy lại

## Kết quả

```
./output/
├── best.pt
├── training_log.txt
├── training_curves.png
└── test_evaluation/
```

---
