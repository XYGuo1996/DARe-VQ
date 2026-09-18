import os
import argparse
import glob
from UTMOS import UTMOSScore
from periodicity import calculate_periodicity_metrics
import torchaudio
from pesq import pesq
import numpy as np
import torch
import math
from pystoi import stoi
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Audio quality metrics (UTMOS, PESQ, F1, STOI)")
    parser.add_argument("--prepath", type=str, required=True, help="Path to reconstructed audio directory")
    parser.add_argument("--rawlist_file", type=str, required=True, help="File containing list of raw audio paths")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    device = torch.device(args.device)
    prepath = args.prepath
    rawlist_file = args.rawlist_file

    # 递归扫描重构目录下的音频文件 (支持平铺或嵌套目录结构)
    audio_exts = (".wav", ".flac", ".mp3")
    pre_map = {}
    for root, _, files in os.walk(prepath):
        for fname in files:
            if fname.lower().endswith(audio_exts):
                pre_map[fname] = os.path.join(root, fname)
    preaudio = sorted(pre_map.keys())

    # 从文本文件读取原始音频路径，建立文件名到绝对路径的映射
    with open(rawlist_file, 'r') as f:
        raw_lines = [line.strip() for line in f if line.strip()]
    raw_map = {os.path.basename(p): p for p in raw_lines}

    # 按推理文件名匹配原始音频路径
    rawaudio = []
    valid_preaudio = []
    for name in preaudio:
        if name in raw_map:
            rawaudio.append(raw_map[name])
            valid_preaudio.append(pre_map[name])
        else:
            print(f"Warning: no matching raw audio for {name}, skipping")
    preaudio = valid_preaudio

    UTMOS = UTMOSScore(device=args.device)

    utmos_sumgt = 0
    utmos_sumencodec = 0
    pesq_sumpre = 0
    f1score_sumpre = 0
    stoi_sumpre = []
    f1score_filt = 0

    for i in tqdm(range(len(preaudio)), desc="metrics"):
        # print(i)
        rawwav, rawwav_sr = torchaudio.load(rawaudio[i])
        prewav, prewav_sr = torchaudio.load(preaudio[i])
        rawwav = rawwav.to(device)
        prewav = prewav.to(device)
        rawwav_16k = torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=16000)
        prewav_16k = torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=16000)
        rawwav_24k = torchaudio.functional.resample(rawwav, orig_freq=rawwav_sr, new_freq=24000)
        prewav_24k = torchaudio.functional.resample(prewav, orig_freq=prewav_sr, new_freq=24000)

        # 1.UTMOS
        # print("****UTMOS_raw", i, UTMOS.score(rawwav_16k.unsqueeze(1))[0].item())
        # print("****UTMOS_encodec", i, UTMOS.score(prewav_16k.unsqueeze(1))[0].item())
        utmos_sumgt += UTMOS.score(rawwav_16k.unsqueeze(1))[0].item()
        utmos_sumencodec += UTMOS.score(prewav_16k.unsqueeze(1))[0].item()

        ## 2.PESQ
        min_len = min(rawwav_16k.size()[1], prewav_16k.size()[1])
        rawwav_16k_pesq = rawwav_16k[:, :min_len].squeeze(0)
        prewav_16k_pesq = prewav_16k[:, :min_len].squeeze(0)
        pesq_score = pesq(16000, rawwav_16k_pesq.cpu().numpy(), prewav_16k_pesq.cpu().numpy(), "wb", on_error=1)
        # print("****PESQ", i, pesq_score)
        pesq_sumpre += pesq_score

        ## 3.F1-score
        min_len = min(rawwav_16k.size()[1], prewav_16k.size()[1])
        rawwav_16k_f1score = rawwav_16k[:, :min_len]
        prewav_16k_f1score = prewav_16k[:, :min_len]
        periodicity_loss, pitch_loss, f1_score = calculate_periodicity_metrics(rawwav_16k_f1score, prewav_16k_f1score)
        # print("****f1", periodicity_loss, pitch_loss, f1_score, f1score_sumpre)
        if (math.isnan(f1_score)):
            f1score_filt += 1
            # print("*****", f1score_filt)
        else:
            f1score_sumpre += f1_score

        ## 4.STOI
        min_len = min(rawwav_24k.size()[1], prewav_24k.size()[1])
        rawwav_stoi = rawwav_24k[:, :min_len].squeeze(0)
        prewav_stoi = prewav_24k[:, :min_len].squeeze(0)
        tmp_stoi = stoi(rawwav_stoi.cpu().numpy(), prewav_stoi.cpu().numpy(), 24000, extended=False)
        # print("****stoi", tmp_stoi)
        stoi_sumpre.append(tmp_stoi)

    print("*************prepath", prepath)
    print("*************UTMOS_raw", utmos_sumgt, utmos_sumgt / len(preaudio))
    print("*************UTMOS_encodec", utmos_sumgt, utmos_sumencodec / len(preaudio))
    print("*************PESQ:", pesq_sumpre, pesq_sumpre / len(preaudio))
    print("*************F1_score:", f1score_sumpre, f1score_sumpre / (len(preaudio) - f1score_filt), f1score_filt)
    print("*************STOI:", np.mean(stoi_sumpre))


if __name__ == "__main__":
    main()
