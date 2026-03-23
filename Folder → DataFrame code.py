import os
import pandas as pd

DATASET_PATH = "Downloads/Audio_Dataset"

data = []

for label_name in ["Normal", "Depression"]:
    class_path = os.path.join(DATASET_PATH, label_name)

    for root, _, files in os.walk(class_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)

                mfcc = extract_mfcc(file_path)
                if mfcc is not None:
                    data.append({
                        "mfcc": mfcc,
                        "label": label_name
                    })

df_audio = pd.DataFrame(data)
print(df_audio.head())
