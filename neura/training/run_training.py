"""
Neura 11.0 Training Runner - Production
"""
import os
import torch
from neura.tokenizer import TOKENIZER
from neura.training.dataset import CausalLMDataset
from neura.training.dataloader import CausalLMDataLoader
from neura.training.loss import CrossEntropyLoss
from neura.training.optimizer import AdamW
from neura.training.trainer import Trainer
from neura.model.causal_lm import CausalLM

def main():
    print("============================================================")
    print("            NEURA LLM TRAINING ENGINE STARTING             ")
    print("============================================================")
    
    # 1. تجهيز ملف البيانات بنص كافٍ للتدريب التجريبي
    data_path = "data/train.txt"
    os.makedirs("data", exist_ok=True)
    
    sample_text = (
        "مرحبا بكم في مشروع نيورا للذكاء الاصطناعي. هذا النموذج يتعلم اللغة العربية بكفاءة عالية. "
        "نيورا هو مساعد ذكاء اصطناعي متطور مبني من الصفر. نصوص ويكيبيديا العربية تساعد في التدريب. "
        "الذكاء الاصطناعي يتطور بسرعة كبيرة، ونحن نقوم ببناء البنية التحتية البرمجية لنموذج نيورا الآن. "
        "تدريب النماذج اللغوية الكبيرة يتطلب معالجة دقيقة للبيانات العربية وتنسيقها بشكل سليم دائماً."
    )
    
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 2. قراءة البيانات وترميزها
    print("[+] Loading and tokenizing Arabic corpus...")
    with open(data_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    
    token_ids = TOKENIZER.encode(text_data)
    print(f"[+] Total tokens in training set: {len(token_ids)}")

    # 3. بناء كلاس البيانات والداتا لودر
    sequence_length = 16  
    dataset = CausalLMDataset(token_ids=token_ids, sequence_length=sequence_length)
    
    batch_size = 2
    dataloader = CausalLMDataLoader(dataset, batch_size=batch_size)
    
    try:
        steps_per_epoch = len(dataloader)
        print(f"[+] Batches/Steps per epoch: {steps_per_epoch}")
    except TypeError:
        steps_per_epoch = 1

    # 4. إعداد موديل CausalLM المتكامل
    print("[+] Initializing Neura CausalLM architectures...")
    if hasattr(TOKENIZER, 'vocab_size') and TOKENIZER.vocab_size is not None:
        v_size = TOKENIZER.vocab_size
    else:
        v_size = 55  # الحجم المكتشف في التشغيل السابق لـ Vocab Size
        
    print(f"[+] Using Vocab Size: {v_size}")

    # تمرير المتغيرات بالأسماء الدقيقة المطابقة لكلاس CausalLM
    model = CausalLM(
        vocab_size=int(v_size),
        hidden_size=256,
        heads=4,
        num_layers=4,
        context_length=64
    )
    
    # 5. إعداد المحسن
    optimizer = AdamW(learning_rate=5e-4)

    # 6. بدء التدريب الفعلي عبر الـ Trainer
    print("[+] Launching Trainer loop...")
    
    # فحص بارامترات الـ Trainer لتمرير total_steps ديناميكياً إن لزم الأمر
    import inspect
    trainer_kwargs = {
        "model": model,
        "dataloader": dataloader,
        "optimizer": optimizer,
        "epochs": 1,
        "max_grad_norm": 1.0
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "total_steps" in trainer_params:
        trainer_kwargs["total_steps"] = steps_per_epoch * 1

    trainer = Trainer(**trainer_kwargs)
    
    if hasattr(trainer, 'train'):
        trainer.train()
    elif hasattr(trainer, 'run'):
        trainer.run()
    else:
        print("[-] Warning: Could not find train() method. Checking trainer methods...")
        print(dir(trainer))

    print("============================================================")
    print("             TRAINING COMPLETED SUCCESSFULLY                ")
    print("============================================================")

if __name__ == "__main__":
    main()
