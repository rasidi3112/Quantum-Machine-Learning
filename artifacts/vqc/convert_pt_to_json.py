import torch # type: ignore
import json

weights = torch.load("vqc_weights.pt", map_location="cpu")

json_ready = {k: v.tolist() for k, v in weights.items()}

with open("vqc_weights.json", "w") as f:
    json.dump(json_ready, f, indent=4)
