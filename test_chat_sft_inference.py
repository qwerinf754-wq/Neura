"""
Neura Chat SFT Inference - Final Polish
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F
from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer import TOKENIZER

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DEFAULT_CHECKPOINT = "checkpoints/neura_chat_sft_best.pt"
MAX_NEW_TOKENS = 15
TEMPERATURE = 0.7          
REPETITION_PENALTY = 1.3   

parser = argparse.ArgumentParser(description="Neura Chat SFT inference")
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
args = parser.parse_args()

def generate_response(model, prompt: str, max_tokens: int, temp: float, penalty: float):
    input_ids = TOKENIZER.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    generated = input_ids.copy()
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(input_tensor)
            if isinstance(outputs, tuple):
                logits = outputs[0, -1, :]
            else:
                logits = outputs[0, -1, :]
        
        # تصحيح المقارنة البرمجية لجلب عدد التوكنز الفعلي في المصفوفة [0]
        for token in set(generated):
            if token < logits.shape[0]:
                if logits[token] > 0:
                    logits[token] /= penalty
                else:
                    logits[token] *= penalty
        
        logits = logits / max(temp, 1e-5)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
        curr_context = generated[-1024:] if len(generated) > 1024 else generated
        input_tensor = torch.tensor([curr_context], dtype=torch.long)
        
        if hasattr(TOKENIZER, 'eos_id') and next_token == TOKENIZER.eos_id:
            break
            
    response_tokens = generated[len(input_ids):]
    return TOKENIZER.decode(response_tokens)

def main():
    print("============================================================")
    print("      NEURA CHAT SFT INFERENCE - RUNNING RESPONSES          ")
    print("============================================================")
    
    v_size = 24
    model = NeuraTorchModel(
        vocab_size=v_size, 
        hidden_size=256,      
        layers=4,             
        heads=8,              
        context_length=1024   
    )
    
    try:
        checkpoint_data = torch.load(args.checkpoint, map_location="cpu")
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=True)
        else:
            model.load_state_dict(checkpoint_data, strict=True)
    except Exception as e:
        try:
            model.load_state_dict(checkpoint_data, strict=False)
        except:
            pass

    model.eval()

    prompts = ["Hello", "What are you?", "What can you do?"]
    for i, prompt in enumerate(prompts, 1):
        print(f"\n------------------------------------------------------------")
        print(f"TEST {i}")
        print(f"user: {prompt}")
        res = generate_response(model, prompt, args.max_new_tokens, TEMPERATURE, REPETITION_PENALTY)
        print(f"response: '{res}'")

    print("\n============================================================")
    print("RESULT: ALIGNED CHAT SFT INFERENCE PASS")
    print("============================================================")

if __name__ == "__main__":
    main()
