# 🚀 TASK 2: QUICK START GUIDE

## Medical QLoRA Fine-tuning with Unsloth in Colab

**Total Setup Time: ~1 minute | Training Time: ~10-20 minutes | Total: ~15-25 minutes**

---

## ⚡ FASTEST PATH (Google Colab)

### Step 1: Open Google Colab
```
1. Go to https://colab.research.google.com
2. Click "File" > "Open notebook"
3. Click "Upload" tab
4. Select: Medical_QLoRA_Finetuning.ipynb
```

### Step 2: Enable GPU
```
1. Click: Runtime > Change runtime type
2. Select: GPU (T4 or better preferred)
3. Click: Save
```

### Step 3: Run Notebook
```
1. Cell 1 (Section 1): Click ▶ (Run all in section)
   → Checks GPU/CUDA setup
   
2. Cell 2 (Section 2): Click ▶
   → Installs Unsloth and dependencies (~3 min)
   
3. Cell 3 (Section 3): Click ▶
   → Loads pre-quantized model (~2 min)
   
4. Cell 4 (Section 4): Click ▶
   → Creates medical dataset
   
5. Cell 5 (Section 5): Click ▶
   → Tokenizes data
   
6. Cell 6 (Section 6): Click ▶
   → Configures LoRA adapter
   
7. Cell 7 (Section 7): Click ▶
   → Sets training parameters
   
8. Cell 8 (Section 8): Click ▶
   → TRAINS MODEL (~5-10 min on T4)
   → Watch loss decrease!
   
9. Cell 9 (Section 9): Click ▶
   → Shows memory usage and efficiency
   
10. Cell 10 (Section 10): Click ▶
    → Saves fine-tuned adapter
    
11. Cell 11 (Section 11): Click ▶
    → Tests on medical queries
    → See medical responses!
```

**✅ DONE! You've successfully fine-tuned a medical LLM!**

---

## 📊 What Happens During Training

```
Epoch 1/3:
  Step 1/7: Loss=2.45
  Step 2/7: Loss=1.85
  ...
  Epoch completed: Loss=1.20

Epoch 2/3:
  Step 1/7: Loss=1.18
  ...
  Epoch completed: Loss=0.95

Epoch 3/3:
  Step 1/7: Loss=0.92
  ...
  Epoch completed: Loss=0.88

✓ Training Complete!
✓ Model Saved!
✓ Ready for Inference!
```

---

## 💾 What Gets Created

After training, you'll have:
```
medical_lora_adapter_TIMESTAMP/
├── lora_adapter/           ← Fine-tuned weights (~50 MB)
├── adapter_config.json     ← Configuration
├── inference_results.json  ← Test results
└── logs/                   ← TensorBoard logs
```

---

## 🧪 Test Results You'll See

**Section 11 Output Example:**

```
Query 1: What are symptoms of type 2 diabetes?
Response: Type 2 diabetes symptoms include increased thirst, 
frequent urination, fatigue, and slow wound healing...

Query 2: How is hypertension treated?
Response: Hypertension is treated through lifestyle changes 
including reducing salt intake, exercise, and medications...

✓ Domain Adaptation Assessment: SUCCESSFUL
✓ Medical terminology usage detected
✓ Evidence-based information provided
✓ Training Project Complete!
```

---

## 🎯 Key Metrics to Watch

| Metric | Expected Value | What It Means |
|--------|----------------|---------------|
| Training Loss | 2.5 → 0.8 | Model is learning ✓ |
| GPU Memory | 8-10 GB | Efficient use ✓ |
| Training Time | 5-15 min | Fast training ✓ |
| Trainable Params | 5M (0.07%) | Memory efficient ✓ |

---

## 🐛 Common Issues & Quick Fixes

| Problem | Solution |
|---------|----------|
| **GPU not found** | Runtime > Change runtime type > GPU |
| **Out of memory** | Section 7: Change `batch_size = 2` |
| **Installation fails** | Re-run Section 2, or skip to Section 3 |
| **Model loading fails** | Use alternative model in Section 3 |

---

## 📁 All Files Explained

| File | Purpose |
|------|---------|
| `Medical_QLoRA_Finetuning.ipynb` | **START HERE** - Main notebook |
| `medical_dataset_sample.csv` | Sample medical Q&A data |
| `requirements.txt` | Dependencies to install |
| `README.md` | Full documentation |
| `config.json` | Configuration presets |
| `inference_script.py` | Use after training |
| `dataset_utilities.py` | Data tools |
| `PROJECT_SUMMARY.txt` | Detailed summary |

---

## 🎓 What You're Learning

✅ **QLoRA Fine-tuning** - 4-bit quantized parameter-efficient adaptation  
✅ **Memory Optimization** - 90% memory reduction vs full training  
✅ **Medical Domain Adaptation** - Specializing models for clinical tasks  
✅ **Production ML** - Training, evaluation, deployment  
✅ **Unsloth Acceleration** - 2-3x faster training  

---

## 🚀 After Training (Optional)

### Use Your Fine-tuned Model

```bash
# Test locally
python inference_script.py \
  --adapter_path medical_lora_adapter_* \
  --query "What are symptoms of myocardial infarction?"

# With custom data
python dataset_utilities.py validate --input your_data.csv
```

---

## 📚 Learn More

- **Full Guide**: See `README.md` for comprehensive documentation
- **Configurations**: Check `config.json` for different hardware setups
- **Sample Data**: Edit `medical_dataset_sample.csv` with your data
- **Advanced**: Study `inference_script.py` and `dataset_utilities.py`

---

## ✅ Expected Runtime

| Task | Time |
|------|------|
| Setup (Section 1) | ~1 min |
| Install Dependencies (Section 2) | ~3 min |
| Load Model (Section 3) | ~2 min |
| Prepare Data (Sections 4-7) | ~2 min |
| **TRAIN MODEL (Section 8)** | **~5-10 min** ⏱️ |
| Evaluate (Sections 9-11) | ~2 min |
| **TOTAL** | **~15-25 minutes** |

---

## 🎉 You're Ready!

1. ✅ Check you have GPU enabled
2. ✅ Click on notebook cell 1
3. ✅ Press ▶ (Run)
4. ✅ Follow the cells in order
5. ✅ Watch your medical LLM train!

---

## 💡 Pro Tips

- **Monitor GPU**: Watch VRAM usage in Colab sidebar
- **Save Progress**: Colab auto-saves, but good to checkpoint
- **Adjust Training**: For better results, modify Section 7 parameters
- **Use Custom Data**: Replace Section 4 CSV with your medical Q&A
- **Deploy Adapter**: Save and use with `inference_script.py`

---

## 🆘 Need Help?

1. Check `README.md` - Troubleshooting section
2. Review `PROJECT_SUMMARY.txt` - Detailed explanations
3. Read inline code comments in notebook
4. See `config.json` - Preset configurations

---

## 🚀 START HERE

**→ Open `Medical_QLoRA_Finetuning.ipynb` in Google Colab and run!**

**Total time from start to fine-tuned model: ~20 minutes**

---

**Happy Learning! 🎓**

*You're about to train a medical language model using cutting-edge techniques!*
