import numpy as np
import os

DATA_PATH = './gesture_data'
ACTIONS = ['blanc', 'hello', 'goodbye', 'yes', 'no', 'thank you', 'i love you']
NUM_SEQUENCES = 30

for action in ACTIONS:
    zeroed_count = 0
    total_frames = 0
    action_path = os.path.join(DATA_PATH, action)
    for sequence in range(NUM_SEQUENCES):
        seq_path = os.path.join(action_path, f'seq_{sequence}.npy')
        if os.path.exists(seq_path):
            seq = np.load(seq_path)  # Shape: (15, 229)
            hand_keypoints = seq[:, 132:258]  # Left+right hands
            zeroed_frames = np.sum(np.all(hand_keypoints == 0, axis=1))
            zeroed_count += zeroed_frames
            total_frames += len(seq)
        else:
            print(f"Missing: {seq_path}")
    print(f"{action}: {zeroed_count}/{total_frames} frames with no hands ({zeroed_count/total_frames*100:.1f}%)")