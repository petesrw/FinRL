# ⚙️ Setup Guides - คู่มือการติดตั้ง

เอกสารในหมวดนี้เหมาะสำหรับการติดตั้งและตั้งค่าระบบ FinRL

## 📚 เอกสารในหมวดนี้

### 🔧 การติดตั้งพื้นฐาน

#### **[SETUP_GUIDE_ENV.md](SETUP_GUIDE_ENV.md)**
- **เหมาะสำหรับ**: ผู้ที่ต้องการติดตั้ง Environment ครบถ้วน
- **เนื้อหา**: 
  - การติดตั้ง Python และ Dependencies
  - การตั้งค่า Virtual Environment
  - การติดตั้ง TA-Lib และ Libraries ที่จำเป็น
  - การตั้งค่า Environment Variables
- **ระบบที่รองรับ**: Windows, macOS, Linux

### 📊 การติดตั้ง Trading Platform

#### **[MT5_CROSS_PLATFORM_GUIDE.md](MT5_CROSS_PLATFORM_GUIDE.md)**
- **เหมาะสำหรับ**: ผู้ที่ต้องการเชื่อมต่อกับ MetaTrader 5
- **เนื้อหา**:
  - การติดตั้ง MT5 บนระบบต่างๆ
  - การตั้งค่า API และ Expert Advisors
  - การเชื่อมต่อ Python กับ MT5
  - การแก้ไขปัญหาการเชื่อมต่อ
- **ระบบที่รองรับ**: Windows, macOS (ผ่าน Wine), Linux (ผ่าน Wine)

## 🗺️ ลำดับการติดตั้ง

### สำหรับผู้เริ่มต้น
```
1. SETUP_GUIDE_ENV.md
   ├── ติดตั้ง Python
   ├── สร้าง Virtual Environment
   ├── ติดตั้ง Dependencies
   └── ทดสอบการติดตั้ง
   
2. MT5_CROSS_PLATFORM_GUIDE.md (ถ้าต้องการเทรดจริง)
   ├── ติดตั้ง MT5
   ├── ตั้งค่า API
   └── ทดสอบการเชื่อมต่อ
```

## 🎯 เป้าหมายของแต่ละเอกสาร

| เอกสาร | เป้าหมาย | เวลาติดตั้ง | ความยาก |
|--------|----------|-------------|----------|
| SETUP_GUIDE_ENV | ติดตั้ง Environment สำเร็จ | 30-60 นาที | ⭐⭐ |
| MT5_CROSS_PLATFORM_GUIDE | เชื่อมต่อ MT5 ได้ | 45-90 นาที | ⭐⭐⭐ |

## 🔍 ข้อกำหนดระบบ

### ข้อกำหนดขั้นต่ำ
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **RAM**: 8GB (แนะนำ 16GB+)
- **Storage**: 10GB ว่าง
- **Python**: 3.8+ (แนะนำ 3.11)
- **Internet**: สำหรับดาวน์โหลด Dependencies

### ข้อกำหนดแนะนำ
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **RAM**: 16GB+ (32GB สำหรับ GPU Training)
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA RTX series (สำหรับ GPU Training)
- **Python**: 3.11

## 🛠️ เครื่องมือที่ต้องใช้

### เครื่องมือพื้นฐาน
- **Python 3.8+**: ภาษาหลักของระบบ
- **pip**: Package Manager
- **Git**: Version Control
- **Visual Studio Code**: Code Editor (แนะนำ)

### เครื่องมือเฉพาะทาง
- **TA-Lib**: Technical Analysis Library
- **MetaTrader 5**: Trading Platform
- **Wine**: สำหรับรัน MT5 บน macOS/Linux

## 📋 Checklist การติดตั้ง

### ก่อนเริ่มติดตั้ง
- [ ] ตรวจสอบข้อกำหนดระบบ
- [ ] เตรียม Internet Connection ที่เสถียร
- [ ] สำรองข้อมูลสำคัญ
- [ ] ปิด Antivirus ชั่วคราว (ถ้าจำเป็น)

### หลังติดตั้งเสร็จ
- [ ] ทดสอบ Python Import
- [ ] ทดสอบ TA-Lib
- [ ] ทดสอบ GPU (ถ้ามี)
- [ ] ทดสอบ MT5 Connection (ถ้าติดตั้ง)
- [ ] รัน Quick Test Script

## 🚨 ปัญหาที่พบบ่อย

### 1. Python Installation Issues
**อาการ**: Python ไม่ทำงาน หรือ pip ไม่พบ
**แก้ไข**: 
- ตรวจสอบ PATH Environment
- ติดตั้งใหม่ด้วย "Add to PATH" option

### 2. TA-Lib Installation Failed
**อาการ**: Error ขณะติดตั้ง TA-Lib
**แก้ไข**:
- Windows: ใช้ .whl file
- macOS: ใช้ Homebrew
- Linux: ติดตั้ง build tools

### 3. GPU Not Detected
**อาการ**: PyTorch ไม่เห็น GPU
**แก้ไข**:
- ติดตั้ง CUDA Toolkit
- ติดตั้ง PyTorch GPU version
- ตรวจสอบ GPU Driver

### 4. MT5 Connection Failed
**อาการ**: Python เชื่อมต่อ MT5 ไม่ได้
**แก้ไข**:
- เปิด "Allow DLL imports" ใน MT5
- ตรวจสอบ Firewall
- ใช้ Demo Account ทดสอบ

## 🔗 เอกสารที่เกี่ยวข้อง

### ก่อนติดตั้ง
- **[COMPLETE_BEGINNER_GUIDE_TH.md](../beginner-guides/COMPLETE_BEGINNER_GUIDE_TH.md)** - ทำความเข้าใจระบบก่อน

### หลังติดตั้ง
- **[QUICK_START_CHECKLIST_TH.md](../beginner-guides/QUICK_START_CHECKLIST_TH.md)** - ทดสอบการติดตั้ง
- **[TROUBLESHOOTING_GUIDE.md](../technical-guides/TROUBLESHOOTING_GUIDE.md)** - แก้ไขปัญหา

## 💡 เทิปการติดตั้ง

### 1. เตรียมความพร้อม
- อ่านเอกสารทั้งหมดก่อนเริ่ม
- เตรียม Internet ที่เร็วและเสถียร
- จัดเวลาให้เพียงพอ (2-3 ชั่วโมง)

### 2. ติดตั้งทีละขั้นตอน
- ไม่ต้องรีบ ทำทีละขั้นตอน
- ทดสอบหลังติดตั้งแต่ละส่วน
- บันทึกข้อผิดพลาดที่พบ

### 3. ใช้ Virtual Environment
- แยก Environment สำหรับ FinRL
- ป้องกันความขัดแย้งของ Package
- ง่ายต่อการจัดการและลบ

### 4. สำรองการตั้งค่า
- Export requirements.txt
- บันทึก Environment Variables
- สำรอง Configuration Files

## 📈 หลังติดตั้งเสร็จ

เมื่อติดตั้งตามเอกสารในหมวดนี้เสร็จแล้ว คุณจะสามารถ:

- ✅ รัน Python Scripts ได้
- ✅ ใช้ TA-Lib คำนวณ Technical Indicators ได้
- ✅ เทรน AI Models ได้
- ✅ เชื่อมต่อ MT5 ได้ (ถ้าติดตั้ง)
- ✅ พร้อมสำหรับการเทรด AI

**พร้อมเริ่มเทรน AI แล้ว! 🚀**

## 🆘 ต้องการความช่วยเหลือ?

หากมีปัญหาขณะติดตั้ง:

1. **ตรวจสอบ Checklist**: ดูว่าพลาดขั้นตอนไหน
2. **ดู Troubleshooting**: [TROUBLESHOOTING_GUIDE.md](../technical-guides/TROUBLESHOOTING_GUIDE.md)
3. **ลองติดตั้งใหม่**: ใช้ Clean Environment
4. **ขอความช่วยเหลือ**: สร้าง Issue ใน Repository