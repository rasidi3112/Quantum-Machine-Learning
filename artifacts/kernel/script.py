import joblib # type: ignore
import numpy as np # type: ignore

joblib_file = "kernel_svc.joblib"
try:
    model = joblib.load(joblib_file)
    print("Isi file kernel_svc.joblib (model):")
    print(model)
except Exception as e:
    print(f"Gagal membuka kernel_svc.joblib: {e}")

print("\n" + "-"*40 + "\n")

npy_file = "train_features.npy"
try:
    features = np.load(npy_file)
    print("Isi file train_features.npy (array fitur):")
    print(features)
    print("Bentuk array:", features.shape)
except Exception as e:
    print(f"Gagal membuka train_features.npy: {e}")
