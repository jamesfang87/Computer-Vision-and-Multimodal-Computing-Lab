import os, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile


def generate_audio_set(data_path: str, batch: list) -> dict:
    audio_set = {}
    for sample in batch:
        data = sample.split('\t')
        video_name = data[0][:13] # 13 for AVDIAR
        data_name = data[0]
        _, audio = wavfile.read(os.path.join(data_path, video_name, data_name + '.wav'))
        audio_set[data_name] = audio
    return audio_set


def overlap(data_name, audio, audio_set):
    noise_name = random.sample(list(set(list(audio_set.keys())) - {data_name}), 1)[0]
    noise_audio = audio_set[noise_name]
    snr = [random.uniform(-5, 5)]
    if len(noise_audio) < len(audio):
        shortage = len(audio) - len(noise_audio)
        noise_audio = numpy.pad(noise_audio, (0, shortage), 'wrap')
    else:
        noise_audio = noise_audio[:len(audio)]
    noise_DB = 10 * numpy.log10(numpy.mean(abs(noise_audio ** 2)) + 1e-4)
    clean_DB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noise_audio = numpy.sqrt(10 ** ((clean_DB - noise_DB - snr) / 10)) * noise_audio
    audio = audio + noise_audio    
    return audio.astype(numpy.int16)


def load_audio(data: list[str], data_path: str, num_frames: int, augment_audio: bool, batch: list):
    data_name: str = data[0]
    fps: float = float(data[2])

    audio_set = generate_audio_set(data_path, batch)
    audio = audio_set[data_name]

    if augment_audio:
        augType = random.randint(0, 1)
        if augType == 1:
            audio = overlap(data_name, audio, audio_set)
        else:
            audio = audio
    
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    max_audio = int(num_frames * 4)
    if audio.shape[0] < max_audio:
        shortage = max_audio - audio.shape[0]
        audio = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(num_frames * 4)),:]  
    return audio


def load_visual(data: list[str], data_path: str, num_frames: int, augment_visual: bool): 
    data_name = data[0]
    video_name = data[0][:13] # 13 for AVDIAR
    face_folder_path = os.path.join(data_path, video_name, data_name)
    face_files = glob.glob("%s/*.jpg"%face_folder_path)
    sorted_face_files = sorted(face_files, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    
    faces = []
    img_size: int = 112
    if augment_visual:
        new = int(img_size * random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, img_size - new), numpy.random.randint(0, img_size - new)
        M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'

    for face_file in sorted_face_files[:num_frames]:
        face = cv2.imread(face_file)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (img_size, img_size))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y + new, x:x + new] , (img_size, img_size)))
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (img_size, img_size)))
    faces = numpy.array(faces)
    return faces


def load_label(data: str, num_frames: int):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:num_frames])
    return res

class TrainLoader():
    def __init__(self, trial_file_name: str, audio_path: str, visual_path: str, batch_size: int):
        self.audio_path  = audio_path
        self.visual_path = visual_path
        self.batches = []

        samples = open(trial_file_name).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sorted_samples = sorted(samples, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         
        
        # assign samples to batches
        start = 0
        while True:
            length = int(sorted_samples[start].split('\t')[1])
            end = min(len(sorted_samples), start + max(int(batch_size / length), 1))
            self.batches.append(sorted_samples[start:end])
            if end == len(sorted_samples):
                break
            start = end

    def __getitem__(self, index):
        """
        returns audio features, visual features, and labels as tuple of Tensors for a "batch".
        """
        batch = self.batches[index]
        num_frames = int(batch[-1].split('\t')[1])
        audio_features, visual_features, labels = [], [], []
        # audio_set = generate_audio_set(self.audio_path, batch) # load the audios in this batch to do augmentation
        
        for sample in batch:
            data = sample.split('\t')
            audio_features.append(load_audio(data, self.audio_path, num_frames, True, batch))
            visual_features.append(load_visual(data, self.visual_path, num_frames, True))
            labels.append(load_label(data, num_frames))

        return (torch.FloatTensor(numpy.array(audio_features)),
                torch.FloatTensor(numpy.array(visual_features)),
                torch.LongTensor(numpy.array(labels)))    

    def __len__(self):
        """
        returns the number of batches
        """
        return len(self.batches)


class ValLoader():
    def __init__(self, trial_file_name, audio_path, visual_path):
        self.audio_path = audio_path
        self.visual_path = visual_path
        self.batches = open(trial_file_name).read().splitlines()

    def __getitem__(self, index):
        """
        returns audio features, visual features, and labels as tuple of Tensors for a "batch".

        batch size for validation set is 1.
        variable names are in terms of batches to be consistent with TrainLoader, 
        other functions written for it, and the potential change to a batch size != 1,
        although the latter change would require changing the code so it loops.
        """
        batch = [self.batches[index]]
        num_frames = int(batch[0].split('\t')[1])
        data: list[str] = batch[0].split('\t')

        audio_features = [load_audio(data, self.audio_path, num_frames, False, batch)]
        visual_features = [load_visual(data, self.visual_path, num_frames, False)]
        labels = [load_label(data, num_frames)]

        return (torch.FloatTensor(numpy.array(audio_features)),
                torch.FloatTensor(numpy.array(visual_features)),
                torch.LongTensor(numpy.array(labels)))

    def __len__(self):
        """
        the batch size for validation set is 1 
        so this function just returns the number of total samples
        """
        return len(self.batches)
