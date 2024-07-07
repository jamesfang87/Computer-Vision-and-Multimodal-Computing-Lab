import pandas as pd
import torch
import os
import numpy as np

ts_first_appearance = 1
first_appearance = 0

a = np.zeros((150, 1, 1))
unaligned = np.zeros((50, 1, 1))

gap = int((ts_first_appearance - first_appearance) / 0.04)
print(gap)
unaligned = unaligned[gap:]
print(unaligned.shape)



"""
data = pd.read_csv('AVDIAR_ASD/csv/train_orig.csv')
print(data)
videos = set(data['video_id'])

for video in videos:
    print(f"Video {video}")
    video_data = data[data['video_id'] == video]

    entities = set(video_data['entity_id'])
    print(f"\tnumber of entities: {len(entities)}\n")

    #entities.sort(key = lambda entity: video_data[video_data['entity_id'] == entity].iloc[0]['frame_timestamp'])

    for entity in entities:
        pass
        #print("\t" + entity)
        #print("\tfirst appearance: " + str(video_data[video_data['entity_id'] == entity].iloc[0]['frame_timestamp']), end='\t')
        #print("\tlast appearance: " + str(video_data[video_data['entity_id'] == entity].iloc[-1]['frame_timestamp']))
"""