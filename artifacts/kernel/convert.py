import pickle
import json

with open("kernel_model_params.pkl", "rb") as f:
    data = pickle.load(f)

with open("kernel_model_params.json", "w") as f:
    json.dump(data, f, indent=4)

print("Selesai. File json sudah dibuat.")
