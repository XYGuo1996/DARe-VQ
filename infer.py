# --coding:utf-8--
import os
import argparse

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer
from tqdm import tqdm
import time

import logging

def main():
    parser = argparse.ArgumentParser(description="Audio reconstruction using WavTokenizer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model config yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_path", type=str, required=True, help="File containing list of audio paths")
    parser.add_argument("--out_folder", type=str, required=True, help="Output folder")
    parser.add_argument("--label", type=str, required=True, help="Experiment label for output subfolder")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    device1 = torch.device(args.device)

    tmptmp = os.path.join(args.out_folder, args.label)
    os.system("rm -r %s" % (tmptmp))
    os.system("mkdir -p %s" % (tmptmp))

    wavtokenizer = WavTokenizer.from_pretrained0802(args.config_path, args.model_path)
    wavtokenizer = wavtokenizer.to(device1)

    with open(args.input_path, 'r') as fin:
        x = fin.readlines()

    x = [i.strip() for i in x]

    features_all = []

    for i in tqdm(range(len(x)), desc="encode_infer"):
        wav, sr = torchaudio.load(x[i])
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)
            sr = 24000
        bandwidth_id = torch.tensor([0])
        wav = wav.to(device1)
        # print(i)

        features, discrete_code = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
        features_all.append(features)

    for i in tqdm(range(len(x)), desc="decode"):
        bandwidth_id = torch.tensor([0])
        bandwidth_id = bandwidth_id.to(device1)

        # print(i)
        audio_out = wavtokenizer.decode(features_all[i], bandwidth_id=bandwidth_id)
        audio_path = os.path.join(args.out_folder, args.label, x[i].split('/')[-1])
        torchaudio.save(audio_path, audio_out.cpu(), sample_rate=24000, encoding='PCM_S', bits_per_sample=16)


if __name__ == "__main__":
    main()
