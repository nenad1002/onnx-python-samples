import torch
from transformers import AutoModelForCausalLM, AutoConfig

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_name = "<YOUR MODEL NAME>"

config = AutoConfig.from_pretrained(model_name)
config.output_hidden_states = True
config.output_attentions    = True

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.float32, # might adjust data type
    device_map="auto",
    low_cpu_mem_usage=True
)
model.eval()

device = model.device
batch_size, seq_length = 1, 1
hidden_size = config.text_config.hidden_size

inputs_embeds   = torch.rand(batch_size, seq_length, hidden_size,
                             device=device, dtype=torch.float32)
attention_mask = torch.ones(batch_size, seq_length,
                            device=device, dtype=torch.long)
position_ids   = torch.arange(seq_length, device=device).unsqueeze(0)

with torch.no_grad():
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=True
    )

all_hidden_states = outputs.hidden_states   
all_attentions    = outputs.attentions     
past_key_values   = outputs.past_key_values 

print(outputs.logits)

torch.save(outputs.logits, "logits2.pt")
