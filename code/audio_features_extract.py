import numpy as np
import librosa
import librosa.display

from scipy.signal import lfilter, get_window
import math
from python_speech_features import *
import pandas as pd
from tqdm import tqdm


class RhythmFeatures:
    """韵律学特征"""

    def __init__(self, input_file, sr=22050, frame_len=1024, n_fft=None, win_step=2 / 3, window="hamming"):
        """
        初始化
        :param input_file: 输入音频文件
        :param sr: 所输入音频文件的采样率，默认为None
        :param frame_len: 帧长，默认512个采样点(此处1024个点，48kHz，对应约21.3ms),与窗长相同
        :param n_fft: FFT窗口的长度，默认与窗长相同
        :param win_step: 窗移，默认移动2/3，512*2/3个采样点
        :param window: 窗类型，默认汉明窗
        """
        self.input_file = input_file
        self.frame_len = frame_len  # 帧长，单位采样点数
        self.wave_data, self.sr = librosa.load(self.input_file, sr=22050, mono=True, offset=0.0)
        self.wave_data = self.wave_data[0:45 * sr, ]
        self.window_len = frame_len
        if n_fft is None:
            self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
        else:
            self.fft_num = n_fft
        self.win_step = win_step
        self.hop_length = round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        self.window = window
        self.numc = 13

    def get_tempos(self, file_ID):
        # sr = 22050
        #   chromagram = librosa.feature.chroma_stft(self.wave_data, self.sr)
        #   np.save("./data/chromagram/"+file_ID+'.npy', chromagram)

        y_harmonic = librosa.effects.harmonic(self.wave_data)
        onset_env_harmonic = librosa.onset.onset_strength(y=y_harmonic, sr=self.sr)
        tempo_harmonic = librosa.beat.tempo(onset_envelope=onset_env_harmonic, sr=self.sr)

        y_percussive = librosa.effects.percussive(self.wave_data)
        onset_env_percussive = librosa.onset.onset_strength(y=y_percussive, sr=self.sr)
        tempo_percussive = librosa.beat.tempo(onset_envelope=onset_env_percussive, sr=self.sr)

        return tempo_harmonic, tempo_percussive

    def short_time_energy(self):
        """
        计算语音短时能量：每一帧中所有语音信号的平方和
        :return: 语音短时能量列表(值范围0-每帧归一化后能量平方和)，
        np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=float64]
        """
        energy = []  # 语音短时能量列表
        energy_sum_per_frame = 0  # 每一帧短时能量累加和
        for i in range(len(self.wave_data)):  # 遍历每一个采样点数据
            energy_sum_per_frame += self.wave_data[i] ** 2  # 求语音信号能量的平方和
            if (i + 1) % self.frame_len == 0:  # 一帧所有采样点遍历结束
                energy.append(energy_sum_per_frame)  # 加入短时能量列表
                energy_sum_per_frame = 0  # 清空和
            elif i == len(self.wave_data) - 1:  # 不满一帧，最后一个采样点
                energy.append(energy_sum_per_frame)  # 将最后一帧短时能量加入列表
        energy = np.array(energy)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
        return energy

    def zero_crossing_rate(self):
        """
        计算语音短时过零率：单位时间(每帧)穿过横轴（过零）的次数
        :return: 每帧过零率次数列表，np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=uint32]
        """
        zcr = []  # 语音短时过零率列表
        counting_sum_per_frame = 0  # 每一帧过零次数累加和，即过零率
        for i in range(len(self.wave_data)):  # 遍历每一个采样点数据
            if i % self.frame_len == 0:  # 开头采样点无过零，因此每一帧的第一个采样点跳过
                continue
            if self.wave_data[i] * self.wave_data[i - 1] < 0:  # 相邻两个采样点乘积小于0，则说明穿过横轴
                counting_sum_per_frame += 1  # 过零次数加一
            if (i + 1) % self.frame_len == 0:  # 一帧所有采样点遍历结束
                zcr.append(counting_sum_per_frame)  # 加入短时过零率列表
                counting_sum_per_frame = 0  # 清空和
            elif i == len(self.wave_data) - 1:  # 不满一帧，最后一个采样点
                zcr.append(counting_sum_per_frame)  # 将最后一帧短时过零率加入列表
        return np.array(zcr, dtype=np.uint32)

    def energy(self):
        """
        每帧内所有采样点的幅值平方和作为能量值
        :return: 每帧能量值，np.ndarray[shape=(1，n_frames), dtype=float64]
        """
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.frame_len, hop_length=self.hop_length,
                                       win_length=self.frame_len, window=self.window))
        pow_spec = np.square(mag_spec)
        energy = np.sum(pow_spec, axis=0)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
        return energy

    def intensity(self):
        """
        计算声音强度，用声压级表示：每帧语音在空气中的声压级Sound Pressure Level(SPL)，单位dB
        公式：20*lg(P/Pref)，P为声压（Pa），Pref为参考压力(听力阈值压力)，一般为2.0*10-5 Pa
        这里P认定为声音的幅值：求得每帧所有幅值平方和均值，除以Pref平方，再取10倍lg
        :return: 每帧声压级，dB，np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=float64]
        """
        p0 = 2.0e-5  # 听觉阈限压力auditory threshold pressure: 2.0*10-5 Pa
        e = self.short_time_energy()
        spl = 10 * np.log10(1 / (np.power(p0, 2) * self.frame_len) * e)
        return spl

    def duration(self, **kwargs):
        """
        持续时间：浊音、轻音段持续时间，有效语音段持续时间,一段有效语音段由浊音段+浊音段两边的轻音段组成
        :param kwargs: activity_detect参数
        :return: np.ndarray[dtype=uint32],浊音shape=(1，n)、轻音段shape=(1，2*n)、有效语音段持续时间列表shape=(1，n)，单位ms
        """
        wav_dat_split_f, wav_dat_split, voiced_f, unvoiced_f = self.activity_detect(**kwargs)  # 端点检测
        duration_voiced = []  # 浊音段持续时间
        duration_unvoiced = []  # 轻音段持续时间
        duration_all = []  # 有效语音段持续时间
        if np.array(voiced_f).size > 1:  # 避免语音过短，只有一帧浊音段
            for voiced in voiced_f:  # 根据帧分割计算浊音段持续时间，两端闭区间
                duration_voiced.append(round((voiced[1] - voiced[0] + 1) * self.frame_len / self.sr * 1000))
        else:  # 只有一帧时
            duration_voiced.append(round(self.frame_len / self.sr * 1000))
        for unvoiced in unvoiced_f:  # 根据帧分割计算清音段持续时间，浊音段左侧左闭右开，浊音段右侧左开右闭
            duration_unvoiced.append(round((unvoiced[1] - unvoiced[0]) * self.frame_len / self.sr * 1000))
        if len(duration_unvoiced) <= 1:  # 避免语音过短，只有一帧浊音段
            duration_unvoiced.append(0)
        for i in range(len(duration_voiced)):  # 浊音段+浊音段两边的轻音段组成一段有效语音段
            duration_all.append(duration_unvoiced[i * 2] + duration_voiced[i] + duration_unvoiced[i * 2 + 1])

        audio_total_time = int(len(self.wave_data) / self.sr * 1000)  # 音频总时间ms
        return (np.array(duration_voiced, dtype=np.uint32), np.array(duration_unvoiced, dtype=np.uint32),
                np.array(duration_all, dtype=np.uint32), audio_total_time)

    def pitch(self, ts_mag=0.25):
        """
        获取每帧音高，即基频，这里应该包括基频和各次谐波，最小的为基频（一次谐波），其他的依次为二次、三次...谐波
        各次谐波等于基频的对应倍数，因此基频也等于各次谐波除以对应的次数，精确些等于所有谐波之和除以谐波次数之和
        :param ts_mag: 幅值倍乘因子阈值，>0，大于np.average(np.nonzero(magnitudes)) * ts_mag则认为对应的音高有效,默认0.25
        :return: 每帧基频及其对应峰的幅值(>0)，
                 np.ndarray[shape=(1 + n_fft/2，n_frames), dtype=float32]
        """
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
                                       win_length=self.frame_len, window=self.window))
        # pitches:shape=(d,t)  magnitudes:shape=(d.t), Where d is the subset of FFT bins within fmin and fmax.
        # pitches[f,t] contains instantaneous frequency at bin f, time t
        # magnitudes[f,t] contains the corresponding magnitudes.
        # pitches和magnitudes大于maximal magnitude时认为是一个pitch，否则取0，maximal默认取threshold*ref(S)=1*mean(S, axis=0)
        pitches, magnitudes = librosa.piptrack(S=mag_spec, sr=self.sr, threshold=1.0, ref=np.mean,
                                               fmin=50, fmax=500)  # 人类正常说话基频最大可能范围50-500Hz
        ts = np.average(magnitudes[np.nonzero(magnitudes)]) * ts_mag
        pit_likely = pitches
        mag_likely = magnitudes
        pit_likely[magnitudes < ts] = 0
        # mag_likely[magnitudes < ts] = 0

        f0_likely = []  # 可能的基频F0
        for i in range(pit_likely.shape[1]):  # 按列遍历非0最小值，作为每帧可能的F0
            try:
                f0_likely.append(np.min(pit_likely[np.nonzero(pit_likely[:, i]), i]))
            except ValueError:
                f0_likely.append(np.nan)  # 当一列，即一帧全为0时，赋值最小值为nan
        f0 = np.array(f0_likely)
        return f0

    def activity_detect(self, min_interval=15, e_low_multifactor=1.0, zcr_multifactor=1.0, pt=False):
        """
        利用短时能量，短时过零率，使用双门限法进行端点检测
        :param min_interval: 最小浊音间隔，默认15帧
        :param e_low_multifactor: 能量低阈值倍乘因子，默认1.0
        :param zcr_multifactor: 过零率阈值倍乘因子，默认1.0
        :param pt: 输出打印标志位，默认为False
        :return: 全部有效语音段:按帧分割后(list,n*2)、按全部采样点的幅值分割(np.ndarray[shape=(n, 采样值数), dtype=float32])、
                浊音段(list,n*2)、轻音段(list,n*2)
        """
        ste = self.short_time_energy()
        zcr = self.zero_crossing_rate()
        energy_average = sum(ste) / len(ste)  # 求全部帧的短时能量均值
        energy_high = energy_average / 4  # 能量均值的4分之一作为能量高阈值
        energy_low = (sum(ste[:5]) / 5 + energy_high / 5) * e_low_multifactor  # 前5帧能量均值+能量高阈值的5分之一作为能量低阈值
        zcr_threshold = sum(zcr) / len(zcr) * zcr_multifactor  # 过零率均值*zcr_multfactor作为过零率阈值
        voiced_sound = []  # 语音段的浊音部分
        voiced_sound_added = []  # 浊音扩充后列表
        wave_detected = []  # 轻音扩充后的最终列表
        # 首先利用能量高阈值energy_high进行初步检测，得到语音段的浊音部分
        add_flag = True  # 加入voiced_sound列表标志位
        for i in range(len(ste)):  # 遍历短时能量数据
            if len(voiced_sound) == 0 and add_flag and ste[i] >= energy_high:  # 第一次达到阈值
                voiced_sound.append(i)  # 加入列表
                add_flag = False  # 接下来禁止加入
            if (not add_flag) and ste[i] < energy_high:  # 直到未达到阈值，此时该阶段为一段浊音语音
                if i - voiced_sound[-1] <= 2:  # 检测帧索引间隔，去掉间隔小于2的索引，判断该段为噪音
                    voiced_sound = voiced_sound[:-1]  # 该段不加入列表
                else:  # 否则加入列表
                    voiced_sound.append(i)
                add_flag = True  # 继续寻找下一段浊音（下一个阈值）
            # 再次达到阈值，判断两个浊音间隔是否大于最小浊音间隔
            elif add_flag and ste[i] >= energy_high and i - voiced_sound[-1] > min_interval:
                voiced_sound.append(i)  # 大于，则分段，加入列表
                add_flag = False  # 接下来禁止加入
            elif add_flag and ste[i] >= energy_high and i - voiced_sound[-1] <= min_interval:
                voiced_sound = voiced_sound[:-1]  # 小于，则不分段，该段不加入列表
                add_flag = False  # 接下来禁止加入
            if (i == len(ste) - 1) and (len(voiced_sound) % 2 == 1):  # 当到达最后一帧，发现浊音段为奇数，则此时到最后一帧为浊音段
                if i - voiced_sound[-1] <= 2:  # 检测帧索引间隔，去掉间隔小于2的索引，判断该段为噪音
                    voiced_sound = voiced_sound[:-1]  # 该段不加入列表
                else:  # 否则加入列表
                    voiced_sound.append(i)
        _print(pt, "能量高阈值:{}，浊音段:{}".format(energy_high, voiced_sound))
        # 再通过能量低阈值energy_low在浊音段向两端进行搜索，超过energy_low便视为有效语音
        for j in range(len(voiced_sound)):  # 遍历浊音列表
            i_minus_flag = False  # i值减一标志位
            i = voiced_sound[j]  # 浊音部分帧索引
            if j % 2 == 1:  # 每段浊音部分的右边帧索引
                while i < len(ste) and ste[i] >= energy_low:  # 搜索超过能量低阈值的帧索引
                    i += 1  # 向右搜索
                voiced_sound_added.append(i)  # 搜索到则加入扩充列表，右闭
            else:  # 每段浊音部分的左边帧索引
                while i > 0 and ste[i] >= energy_low:  # 搜索超过能量低阈值的帧索引
                    i -= 1  # 向左搜索
                    i_minus_flag = True  # i值减一标志位置位
                if i_minus_flag:  # 搜索到则加入扩充列表，左闭
                    voiced_sound_added.append(i + 1)
                else:
                    voiced_sound_added.append(i)
        _print(pt, "能量低阈值:{}，浊音再次扩展后:{}".format(energy_low, voiced_sound_added))
        # 最后通过过零率对浊音扩充后列表向两端再次进行搜索，获取轻音部分
        for j in range(len(voiced_sound_added)):  # 遍历浊音扩充后列表
            i_minus_flag = False  # i值减一标志位
            i = voiced_sound_added[j]  # 浊音扩充后部分帧索引
            if j % 2 == 1:  # 每段浊音扩充部分的右边帧索引
                while i < len(zcr) and zcr[i] >= zcr_threshold:  # 搜索超过过零率阈值的帧索引
                    i += 1  # 向右搜索
                wave_detected.append(i)  # 搜索到则加入扩充列表，右开
            else:  # 每段浊音扩充部分的左边帧索引
                while i > 0 and zcr[i] >= zcr_threshold:  # 搜索超过过零率阈值的帧索引
                    i -= 1  # 向左搜索
                    i_minus_flag = True  # i值减一标志位置位
                if i_minus_flag:  # 搜索到则加入扩充列表，左闭
                    wave_detected.append(i + 1)
                else:
                    wave_detected.append(i)
        _print(pt, "过零率阈值:{}，轻音段增加后:{}".format(zcr_threshold, wave_detected))
        wave_data_detected_frame = []  # 端点检测后，以帧为单位的有效语音列表
        for index in range(len(wave_detected)):
            if index % 2 == 0:  # 按段分割成列表
                wave_data_detected_frame.append(wave_detected[index:index + 2])
            else:
                continue
        _print(pt, "分割后共{}段语音，按帧分割为{}".format(len(wave_data_detected_frame), wave_data_detected_frame))
        wave_data_detected = []  # 端点检测后，对应全部采样点的幅值列表，其中列表代表每个有效语音段
        for index in wave_data_detected_frame:
            try:
                wave_data_detected.append(self.wave_data[index[0] * int(self.frame_len):
                                                         index[1] * int(self.frame_len)])
            except IndexError:
                wave_data_detected.append(self.wave_data[index[0] * int(self.frame_len):-1])
        _print(pt, "分割后共{}段语音，按全部采样点的幅值分割为{}".format(len(wave_data_detected), wave_data_detected))
        if np.array(voiced_sound_added).size > 1:  # 避免语音过短，只有一帧浊音段
            voiced_frame = np.array(voiced_sound_added).reshape((-1, 2)).tolist()  # 按帧分割的浊音段
        else:  # 只有一帧时
            voiced_frame = np.array(voiced_sound_added).tolist()
        unvoiced_frame = []  # 按帧分割的轻音段
        for i in range(len(wave_detected)):  # 根据最终的扩充后列表和浊音段列表求得轻音段
            if wave_detected[i] < voiced_sound_added[i]:
                unvoiced_frame.append([wave_detected[i], voiced_sound_added[i]])
            elif wave_detected[i] > voiced_sound_added[i]:
                unvoiced_frame.append([voiced_sound_added[i], wave_detected[i]])
            else:
                unvoiced_frame.append([0, 0])
        return wave_data_detected_frame, wave_data_detected, voiced_frame, unvoiced_frame

    def formant(self, ts_e=0.01, ts_f_d=200, ts_b_u=2000):
        """
        LPC求根法估计每帧前三个共振峰的中心频率及其带宽
        :param ts_e: 能量阈值：默认当能量超过0.01时认为可能会出现共振峰
        :param ts_f_d: 共振峰中心频率下阈值：默认当中心频率超过200，小于采样频率一半时认为可能会出现共振峰
        :param ts_b_u: 共振峰带宽上阈值：默认低于2000时认为可能会出现共振峰
        :return: F1/F2/F3、B1/B2/B3,每一列为一帧 F1/F2/F3或 B1/B2/B3，np.ndarray[shape=(3, n_frames), dtype=float64]
        """
        _data = lfilter([1., 0.83], [1], self.wave_data)  # 预加重0.83：高通滤波器
        inc_frame = self.hop_length  # 窗移
        n_frame = int(np.ceil(len(_data) / inc_frame))  # 分帧数
        n_pad = n_frame * self.window_len - len(_data)  # 末端补零数
        _data = np.append(_data, np.zeros(n_pad))  # 无法整除则末端补零
        win = get_window(self.window, self.window_len, fftbins=False)  # 获取窗函数
        formant_frq = []  # 所有帧组成的第1/2/3共振峰中心频率
        formant_bw = []  # 所有帧组成的第1/2/3共振峰带宽
        rym = RhythmFeatures(self.input_file, self.sr, self.frame_len, self.fft_num, self.win_step, self.window)
        e = rym.energy()  # 获取每帧能量值
        e = e / np.max(e)  # 归一化
        for i in range(n_frame):
            f_i = _data[i * inc_frame:i * inc_frame + self.window_len]  # 分帧
            if np.all(f_i == 0):  # 避免上面的末端补零导致值全为0，防止后续求LPC线性预测误差系数出错(eps是取非负的最小值)
                f_i[0] = np.finfo(np.float64).eps
            f_i_win = f_i * win  # 加窗
            a = librosa.lpc(f_i_win, 8)  # 获取LPC线性预测误差系数，即滤波器分母多项式，阶数为 预期共振峰数3 *2+2，即想要得到F1-3
            rts = np.roots(a)  # 求LPC返回的预测多项式的根,为共轭复数对
            rts = np.array([r for r in rts if np.imag(r) >= 0])  # 只保留共轭复数对一半，即虚数部分为+或-的根
            rts = np.where(rts == 0, np.finfo(np.float64).eps, rts)  # 避免值为0，防止后续取log出错(eps是取非负的最小值)
            ang = np.arctan2(np.imag(rts), np.real(rts))  # 确定根对应的角(相位）
            # F(i) = ang(i)/(2*pi*T) = ang(i)*f/(2*pi)
            frq = ang * (self.sr / (2 * np.pi))  # 将以角度表示的rad/sample中的角频率转换为赫兹sample/s
            indices = np.argsort(frq)  # 获取frq从小到大排序索引
            frequencies = frq[indices]  # frq从小到大排序
            # 共振峰的带宽由预测多项式零点到单位圆的距离表示: B(i) = -ln(r(i))/(pi*T) = -ln(abs(rts[i]))*f/pi
            bandwidths = -(self.sr / np.pi) * np.log(np.abs(rts[indices]))
            formant_f = []  # F1/F2/F3
            formant_b = []  # B1/B2/B3
            if e[i] > ts_e:  # 当能量超过ts_e时认为可能会出现共振峰
                # 采用共振峰频率大于ts_f_d小于self.sr/2赫兹，带宽小于ts_b_u赫兹的标准来确定共振峰
                for j in range(len(frequencies)):
                    if (ts_f_d < frequencies[j] < self.sr / 2) and (bandwidths[j] < ts_b_u):
                        formant_f.append(frequencies[j])
                        formant_b.append(bandwidths[j])
                # 只取前三个共振峰
                if len(formant_f) < 3:  # 小于3个，则补nan
                    formant_f += ([np.nan] * (3 - len(formant_f)))
                else:  # 否则只取前三个
                    formant_f = formant_f[0:3]
                formant_frq.append(np.array(formant_f))  # 加入帧列表
                if len(formant_b) < 3:
                    formant_b += ([np.nan] * (3 - len(formant_b)))
                else:
                    formant_b = formant_b[0:3]
                formant_bw.append(np.array(formant_b))
            else:  # 能量过小，认为不会出现共振峰，此时赋值为nan
                formant_frq.append(np.array([np.nan, np.nan, np.nan]))
                formant_bw.append(np.array([np.nan, np.nan, np.nan]))
        formant_frq = np.array(formant_frq).T
        formant_bw = np.array(formant_bw).T
        return formant_frq, formant_bw

    def get_mfcc(self):
        # 提取MFCC特征
        wav_feature = mfcc(self.wave_data, self.sr, numcep=self.numc, winlen=0.02, winstep=0.01, nfilt=26, nfft=1024,
                           lowfreq=0, highfreq=None, preemph=0.97)
        '''
        signal - 需要用来计算特征的音频信号，应该是一个N*1的数组
        samplerate - 我们用来工作的信号的采样率
        winlen - 分析窗口的长度，按秒计，默认0.025s(25ms)
        winstep - 连续窗口之间的步长，按秒计，默认0.01s（10ms）
        numcep - 倒频谱返回的数量，默认13
        nfilt - 滤波器组的滤波器数量，默认26
        nfft - FFT的大小，默认512
        lowfreq - 梅尔滤波器的最低边缘，单位赫兹，默认为0
        highfreq - 梅尔滤波器的最高边缘，单位赫兹，默认为采样率/2
        preemph - 应用预加重过滤器和预加重过滤器的系数，0表示没有过滤器，默认0.97
        ceplifter - 将升降器应用于最终的倒谱系数。 0没有升降机。默认值为22。
        appendEnergy - 如果是true，则将第0个倒谱系数替换为总帧能量的对数。 
        '''
        d_mfcc_feat = delta(wav_feature, 1)
        d_mfcc_feat2 = delta(wav_feature, 2)
        mfcc_feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
        return mfcc_feature


def analyze_MFCC(file_ID):
    file_path = "./data/audio/audio_long/" + file_ID + ".mp3"
    rhythm_f = RhythmFeatures(file_path)
    tempo_harmonic, tempo_percussive = rhythm_f.get_tempos(file_ID)
    ss = rhythm_f.get_mfcc()
    mfcc_res = []
    for ii in range(ss.shape[1]):
        # print(ii)
        mfcc_res.append(stati(ss[:, ii]))

    return rhythm_f, mfcc_res, tempo_harmonic, tempo_percussive


def get_rhythm(rhythm_f, mfcc_res, tempo_harmonic, tempo_percussive):
    Normalized_duration_voiced = []
    for ii in range(len(rhythm_f.duration()[0])):
        rhythm_sum = rhythm_f.duration()[0][ii] + rhythm_f.duration()[1][(ii * 2)] + rhythm_f.duration()[1][
            (ii * 2 + 1)]
        Normalized_duration_voiced.append(rhythm_f.duration()[0][ii] / rhythm_sum)

    Total_Duty_Ratio = sum(rhythm_f.duration()[0]) / rhythm_f.duration()[3]
    Voice_or_not = []
    if Total_Duty_Ratio >= 0.01:
        Voice_or_not = 1
    else:
        Voice_or_not = 0
    data = {
        'Voice_or_not': Voice_or_not,
        'Total_Duty_Ratio': Total_Duty_Ratio,
        'Normalized_number_of_interrupts': len(rhythm_f.duration()[2]) / rhythm_f.duration()[3],
        'Normalized_duration_voiced': stati(Normalized_duration_voiced),
        'short_time_energy': stati(rhythm_f.short_time_energy().tolist()),
        'intensity': stati(rhythm_f.intensity().tolist()),
        'zero_crossing_rate': stati(rhythm_f.zero_crossing_rate().tolist()),
        'f0': stati(rhythm_f.pitch().tolist()),
        'formant_frq_1': stati(rhythm_f.formant()[0][0, :].tolist()),
        'formant_frq_2': stati(rhythm_f.formant()[0][1, :].tolist()),
        'formant_frq_3': stati(rhythm_f.formant()[0][2, :].tolist()),
        'formant_bw_1': stati(rhythm_f.formant()[1][0, :].tolist()),
        'formant_bw_2': stati(rhythm_f.formant()[1][1, :].tolist()),
        'formant_bw_3': stati(rhythm_f.formant()[1][2, :].tolist()),
        'mfcc': mfcc_res,
        "tempo_harmonic": tempo_harmonic,
        "tempo_percussive": tempo_percussive
    }

    return data


def stati(values):
    # 计算各个时间序列的统计值，包括平均值，标准差，最大值，最小值，分布曲线对应的峰值
    # 删除nan值
    values = [value for value in values if math.isnan(value) == 0]

    arr_mean = np.mean(values)
    arr_std = np.std(values, ddof=1)
    arr_max = max(values)
    arr_min = min(values)
    # 用直方图统计进行核密度估计，分成50个区间, max_KDE为密度最大值所在区间的中间值
    #     xn, bin, patches = plt.hist(values, bins=50)
    #     # plt.show()
    #     max_KDE = (bin[np.argmax(xn)] + bin[np.argmax(xn) + 1]) / 2

    return arr_mean, arr_std, arr_max, arr_min  # , max_KDE


# 加载程序功能模块
def _print(bl=True, s=None):
    if bl:
        print(s)
    else:
        pass

def func_format(x, pos):
    return "%d" % (1000 * x)


def get_features(file_ID):
    file_path = "./data/audio/audio_long/" + file_ID + ".mp3"
    wave_data, sr = librosa.load(file_path, sr=22050, mono=True, offset=0.0)

    y_harmonic = librosa.effects.harmonic(wave_data)
    onset_env_harmonic = librosa.onset.onset_strength(y=y_harmonic, sr=sr)
    tempo_harmonic = librosa.beat.tempo(onset_envelope=onset_env_harmonic, sr=sr)

    y_percussive = librosa.effects.percussive(wave_data)
    onset_env_percussive = librosa.onset.onset_strength(y=y_percussive, sr=sr)
    tempo_percussive = librosa.beat.tempo(onset_envelope=onset_env_percussive, sr=sr)

    threshold = 0.4
    chromagram = librosa.feature.chroma_stft(wave_data, sr)
    R = librosa.segment.recurrence_matrix(chromagram, k=100, metric='cosine', mode='affinity')
    self_similarity = np.sqrt(np.count_nonzero((R > threshold).astype('int64') == 1) / R.size)

    return [tempo_harmonic[0], tempo_percussive[0], self_similarity]
