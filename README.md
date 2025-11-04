# Deep Learning (01204466) Project  
## Brain Tumor Detection  

**นาย ชยกร ศรุตยาพร 6610505331**

---

## 🧠 หัวข้อโครงงาน
**Brain Tumor Detection from Brain MRI using Convolutional Neural Networks**

---

## 💡 ทำไมหัวข้อนี้น่าสนใจ และเหตุผลในการเลือก

### ความสำคัญทางการแพทย์  
เนื้องอกในสมองเป็นภาวะรุนแรง การตรวจพบเร็วช่วยเพิ่มโอกาสรักษาได้ทันท่วงที และการมีโมเดลที่สามารถทำนายได้อย่างแม่นยำสามารถลดภาระของแพทย์ในการตรวจและเพิ่มประสิทธิภาพของการวินิจฉัยโดยรวมได้

### โอกาสทางเทคโนโลยี  
ภาพ MRI มีโครงสร้างเชิงพื้นที่และลวดลายที่ซับซ้อน เหมาะกับการประยุกต์ **Deep Learning** โดยเฉพาะ **CNN** ที่เด่นด้านการจับ pattern ของรูปภาพ

---

## 🤖 ต้องใช้ Deep Learning?

### Classical Machine Learning
**Pipeline ทั่วไป:** Feature engineering + ML Classifier (SVM, Random Forest)  
**ข้อดี:** ต้องการข้อมูลน้อยกว่า, อธิบายง่ายกว่า, เทรนเร็ว  
**ข้อด้อย:** คุณภาพขึ้นกับการทำ feature engineering  

### Deep Learning (CNN)
**ข้อดี:** เรียนรู้ฟีเจอร์จากข้อมูลดิบโดยอัตโนมัติ, รองรับข้อมูลขนาดใหญ่  
**ข้อเสีย:** ต้องการข้อมูลมาก, ใช้ทรัพยากรสูง  

➡️ **สรุป:** สำหรับโจทย์ภาพ MRI การใช้ CNN ให้ผลลัพธ์ที่ดีกว่าแน่นอน เทียบกับข้อเสียแล้วถือว่าคุ้มค่าที่จะลองทำ  

---

## 🧱 สถาปัตยกรรม Deep Learning ที่ใช้

เลือกใช้ **CNN** ทำ Binary Classification (Tumor / Not Tumor)

### โครงสร้างเลเยอร์
**Input:** แปลงภาพให้เป็นขนาด 224×224 pixels

#### Feature Extractor (Convolution Layers)
```python
nn.Conv2d(3, 16, kernel_size=3, padding=1)
nn.Conv2d(16, 32, kernel_size=3, padding=1)
nn.Conv2d(32, 64, kernel_size=3, padding=1)
```
- หลังแต่ละ Convolution มี Batch Normalization, ReLU, MaxPooling  
- Filter: 16 → 32 → 64  
  - ชั้นแรก: จับ pattern พื้นฐาน (เส้นตรง, โค้ง)
  - ชั้นลึก: จับ pattern ซับซ้อน (รอยหยักสมอง, โครงสร้างเฉพาะ)

#### Classifier
```python
Flatten(28×28×64)
Linear(28×28×64 → 256) → ReLU → Dropout(0.25)
Linear(256 → 2)
```

---

### แผนภาพการเชื่อมต่อ (ย่อ)
```
Input (224×224×3)
→ Conv(16) → BN → ReLU → Pool
→ Conv(32) → BN → ReLU → Pool
→ Conv(64) → BN → ReLU → Pool
→ Flatten → FC(256) → ReLU → Dropout(0.25)
→ FC(2) → Softmax
```

---

### จำนวนพารามิเตอร์ (โดยประมาณ)

| Layer | พารามิเตอร์ |
|--------|---------------|
| Conv1 (3→16) | 448 |
| Conv2 (16→32) | 13,856 |
| Conv3 (32→64) | 55,360 |
| BatchNorm (16+32+64) | 112 |
| FC (50176→256) | 12,845,312 |
| FC (256→2) | 514 |
| **รวมทั้งหมด** | ≈ 12,869,643 |

**Activation Function:** ReLU หลัง Conv และ FC เพื่อเพิ่ม nonlinearity และลดปัญหา vanishing gradient  

---

## ⚙️ อธิบายโค้ด PyTorch

### การจัดการข้อมูล
- แบ่ง **train (80%) / test (20%)**  
- ใช้ `DataLoader` (batch_size, shuffle, pin_memory)  
- ใช้ `transform` แยกฝั่ง train/test เพื่อป้องกัน data leakage  
- ฝั่ง train มี `RandomHorizontalFlip` และ `RandomRotation` เพิ่มความหลากหลาย (data augmentation)

### นิยามโมเดล
- สร้างคลาส `CNN` ตามโครงสร้างด้านบน  
- เขียนใน `__init__` และ `forward`  

### การเทรนและปรับพารามิเตอร์
- **Loss:** CrossEntropyLoss  
- **Optimizer:** Adam (learning rate = 1e-5)  
- **Scheduler:** ReduceLROnPlateau (ลด LR อัตโนมัติเมื่อ loss ไม่ดีขึ้น 2 epoch)  

**Hyperparameters:**
```
Learning rate = 1e-5
Epochs = 10
Batch size = 16
```

---

## 🧮 Training Method
1. เตรียม loss = 0, progress bar, clear gradient  
2. วนลูปแต่ละ batch  
   - Forward → คำนวณ loss  
   - Backward → optimizer step  
   - เก็บสถิติ  
3. สรุปค่าเฉลี่ยต่อ epoch และแสดงผล  

---

## 📊 Model Evaluation

- ใช้โมเดลที่เทรนแล้วกับ **testing set**  
- Metrics:
  - **Confusion Matrix**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

---

### Dataset
> [Brain MRI Images for Brain Tumor Detection (Kaggle)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

- แบ่ง training set 80%, testing set 20%  
- ใช้ Optimizer Adam + Scheduler ปรับ LR อัตโนมัติ  
- ทดสอบกับ test set เพียงครั้งเดียวหลังเทรนเสร็จ  

---

### Training Results

| Metric | Value |
|---------|--------|
| **Training Loss** | 0.0784 |
| **Training Accuracy** | 99.51% |
| **Test Accuracy** | 82% |
| **Recall (Tumor)** | 85.29% |

> Recall สูงในคลาส “Tumor” แสดงว่าโมเดลตรวจจับผู้ป่วยได้ดี แม้เป็นโมเดลที่สร้างเองจากศูนย์  
> หากนำแนวทางนี้ไปพัฒนาต่อด้วย **pretrained model** เช่น ResNet18 คาดว่าผลลัพธ์จะดีกว่านี้มาก  

---

## 📈 Confusion Matrix
False Negative & False Positive มีค่าต่ำ ถือว่าเป็นผลลัพธ์ที่ดีและน่าพอใจ

---

## 🎥 DEMO
*(เพิ่มภาพผลการทดสอบหรือกราฟเทรนได้ที่นี่)*

---

## 💾 Public Repository
🔗 **GitHub Repo:** [https://github.com/kkeenchs/01204466-Deep-learning-final-project.git](https://github.com/kkeenchs/01204466-Deep-learning-final-project.git)

---

## 📚 References
- Batch Normalization — [GeeksforGeeks](https://www.geeksforgeeks.org/deep-learning/what-is-batch-normalization-in-deep-learning/)  
- Brain MRI Dataset — [Kaggle Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- MIT Computer Vision Lab — [MIT Deep Learning Labs (GitHub)](https://github.com/MITDeepLearning/introtodeeplearning/blob/master/lab2/PT_Part1_MNIST.ipynb)
