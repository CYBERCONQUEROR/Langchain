import librosa
import numpy as np
import os
import pandas as pd


SAMPLE_RATE = 16000
N_MFCC = 40
MAX_LEN = 300   # time steps

def extract_mfcc(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

        # Pad or truncate
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc
    except Exception as e:
        print("Error:", file_path, e)
        return None

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

    