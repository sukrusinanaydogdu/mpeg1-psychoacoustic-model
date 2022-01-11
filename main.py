from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np

def decimateMasker(masker_thresholded_index_bark, P_masker_thresholded):
    P_masker_decim = []
    masker_decim_index = []

    for k in range(len(masker_thresholded_index_bark) - 1):
        # If they occur within a distance of 0.5
        if np.absolute(masker_thresholded_index_bark[k + 1] - masker_thresholded_index_bark[k]) <= 0.5:
            if P_masker_thresholded[k + 1] > P_masker_thresholded[k]:
                P_masker_decim.append(P_tonal_masker_thresholded[k + 1])
                masker_decim_index.append(masker_thresholded_index_bark[k + 1])
            else:
                P_masker_decim.append(P_tonal_masker_thresholded[k])
                masker_decim_index.append(masker_thresholded_index_bark[k])
        else:
            P_masker_decim.append(P_tonal_masker_thresholded[k])
            masker_decim_index.append(masker_thresholded_index_bark[k])
    P_masker_decim.append(P_tonal_masker_thresholded[len(masker_thresholded_index_bark) - 1])
    masker_decim_index.append(masker_thresholded_index_bark[len(masker_thresholded_index_bark) - 1])

    return [masker_decim_index, P_masker_decim]

def bark2f(bark):
    # this function is used to convert bark to frequency
    return 600 * np.sinh(bark/6)

def f2bark(f):
    # This function is used to convert frequency to barks
    return 6 * np.arcsinh(f/600)

def geo_mean(num1, num2):
    # This function is used to calculate the geometric mean of two numbers
    return np.sqrt(num1*num2)

 # finding spread of masking functions
def SpreadingFunction(i, j, p):
    delta_z = i - j
    sf = 0
    if -3 <= delta_z < -1:
        sf = 17 * delta_z - 0.4 * p + 11
    elif -1 <= delta_z < 0:
        sf = (0.4 * p + 6) * delta_z
    elif 0 <= delta_z < 1:
        sf = -17 * delta_z
    elif 1 <= delta_z < 8:
        sf = (0.15 * p - 17) * delta_z - 0.15 * p

    return sf

def index2frequency(input):
    return np.multiply(input, 22050 / 256)

def index2bark(input):
    return (f2bark(np.multiply(input, 22050 / 256)))

# Read monophonic PCM audio file
rate, audio_original = read("sin4k.wav")
# audio_original = audio_original[:,0]

bits = 16  # number of bits per sample
N = 512  # fft length

frequency_Hz = np.arange(1, (N / 2 + 1)) * (rate / N)  # 256 bins, each representing 86.13 Hz
bark_frequencies = f2bark(frequency_Hz)  # Convert frequency scale to bark scale
frequency = np.arange(N / 2)

# Absolute threshold of hearing or just noticable difference
threshold = 3.64 * ((frequency_Hz / 1000) ** -0.8) - 6.5 * np.exp(-0.6 * ((frequency_Hz / 1000 - 3.3) ** 2)) + (10 ** -3) * ((frequency_Hz / 1000) ** 4)
# threshold[187:256] = 69.13  # Last part of the curve is flat

# Center frequencies for 25 critical filter bank
# Adapted from Table 5.1
center_frequencies = np.array([50, 150, 250, 350, 450, 570, 700, 840, 1000, 1175, 1370, 1600, 1850, 2150, 2500, 2900, 3400, 4000, 4800, 5800, 7000, 8500, 10500, 13500, 19500])

center_frequencies_bark = f2bark(center_frequencies)  # bark conversion of center frequencies
center_frequencies_256 = np.divide(np.multiply(center_frequencies, N // 2), (rate / 2))  # converting center freqs. to 0-256 range
center_frequencies_256 = center_frequencies_256.astype(int)  # making it integer

band_boundaries = np.array([0, 1, 2, 3, 5, 6, 8, 9, 11, 13, 15, 17, 20, 23, 27, 32, 37, 45, 52, 62, 74, 88, 108, 132, 180, 232])  # critical band boundaries
bark = np.arange(25)

bandwidth_hz = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 22050])

k_bar = np.zeros(25)

# Calculation of k_bar according to Eq. 5.25
for i in range(0, 25):
    k_bar[i] = geo_mean(bandwidth_hz[i], bandwidth_hz[i + 1])

k_bark = f2bark(k_bar)
k_bark256 = np.divide(np.multiply(k_bar, N // 2), (rate / 2))
k_bark256 = k_bark256.astype(int)

center_frequencies_bark = k_bark
center_frequencies_256 = k_bark256

""" --------------------------Step 1: Spectral Analysis and SPL Normalization--------------------------------"""

for i in range(0, 512, 512):
    # Get a new frame of 512 samples in each loops
    audio = audio_original[i:(i + N)]

    # Normalization is done with number of bits and FFT Length (Eq. 5.18)
    # Normalization is done beacuse playback levels ara unknown
    audio = audio / (N * (2 ** (bits - 1)))

    # Hanning (Hann) Window is constructed (Eq.5.20)
    # 1.63 is the window correction factor
    # https: // community.sw.siemens.com / s / article / window - correction - factors
    h = np.hanning(M=512) * 1.63

    # FFT is calculated with N points but PSD estimete is calculated with N/2 points. (Eq. 5.19)
    # Before calculating FFT, window function is applied
    X = np.fft.fft(h * audio, 512) / 512

    fft = abs(X)
    fft = fft[0:(N // 2)]

    # In the book, 10log(fft^2) but this is equal to 20log(fft)
    PSD = 20 * np.log10(fft)

    # This finalizes Eq. 5.19
    # In the book, delta = 90.302 dB
    delta = 90.302 - np.amax(PSD)
    PSD += delta

    P_tonal_masker = []
    tonal_masker_index = []

    """ ------------------------Step 2: Identification of Tonal and Noise Maskers ------------------------------"""

    # Finding "P_tonal_masker" from spectral peaks

    for k in np.arange(2, 250):
        # Definition of tonal set in Eq. 5.21
        if (PSD[k - 1] < PSD[k] and PSD[k] > PSD[k + 1]):

            delta_k = []

            # Exact implementation of Eq. 5.22
            if k > 2 and k < 63:
                delta_k = np.array([-2, +2])
            elif k >= 63 and k < 127:
                delta_k = np.array([-3, -2, 2, 3])
            elif k >= 127 and k <= 256:
                delta_k = np.array([-6, -5, -4, -3, -2, +2, +3, +4, +5, +6])
            else:
                delta_k = 0

            # delta part can be altered by only using positive delta values and changind that if
            # Exact implementation of Eq.5.23
            # Tonal maskers are computed and appended to the empty vector for further usage
            if all(PSD[k] > PSD[k + delta_k] + 7):
                P_tonal_masker.append(10 * np.log10(10 ** (0.1 * PSD[k - 1]) + 10 ** (0.1 * PSD[k]) + 10 ** (0.1 * PSD[k + 1])))
                tonal_masker_index.append(k)

                tonal_masker_freq = index2frequency(tonal_masker_index)
                tonal_masker_freq_bark = f2bark(tonal_masker_freq)


    """ ------------------- Calculating nontonal noise maskers in critical bands  -------------------"""

    # We need to exclude already found spectral lines
    noise_masker_index = []  # noise masker indices
    exclude_index = []  # indices to be excluded,
    noise_masker_arr = []  # noise masker array

    for x in range(0, len(band_boundaries) - 1):

        for idx in range(band_boundaries[x], band_boundaries[x + 1]):
            delta_k = []

            if idx > 2 and idx < 63:
                delta_k = np.array([-2, -1, 0, 1, 2])
            elif idx >= 63 and idx < 127:
                delta_k = np.array([-3, -2, -1, 0, 1, 2, 3])
            elif idx >= 127 and idx <= 256:
                delta_k = np.array([-6, -5, -4, -3, -2, -1, 0, 1, 2, +3, +4, +5, +6])

            for j in range(0, len(tonal_masker_index)):
                for f in range(0, len(delta_k)):
                    if (idx == (tonal_masker_index[j] + delta_k[f])):
                        exclude_index.append(idx)

    bnd_k = np.arange(0, 232)

    # indices without previously found spectral lines
    # Tonal maskers are excluded and noise maskers will be computed with the remaining spectral lines
    noise_masker_index = list(set(bnd_k) ^ set(exclude_index))

    # P_noise_maskers are calculated by Eq. 5.29
    for x in range(0, len(band_boundaries) - 1):
        total = 0
        for a in range(len(noise_masker_index)):

            if (band_boundaries[x] <= noise_masker_index[a] and noise_masker_index[a] < band_boundaries[x + 1]):
                total += 10 ** (0.1 * PSD[noise_masker_index[a]])

        noise_masker_arr.append(10 * np.log10(total))

    """-----------------------------------Step 3: Decimation and Reorganization of Maskers.-----------------------"""

    noise_masker_thresholded_index = []
    tonal_masker_thresholded_index = []
    P_tonal_masker_thresholded = []
    P_noise_masker_thresholded = []

    # Any tonal or noise maskers below the absolute threshold are discarded
    # According to Eq 5.26
    for k in range(len(tonal_masker_index)):
        if (P_tonal_masker[k] >= threshold[tonal_masker_index[k]]):
            P_tonal_masker_thresholded.append(P_tonal_masker[k])
            tonal_masker_thresholded_index.append(tonal_masker_index[k])

    for l in range(len(center_frequencies_256)):
        if (noise_masker_arr[l] >= threshold[center_frequencies_256[l]]):
            P_noise_masker_thresholded.append(noise_masker_arr[l])
            noise_masker_thresholded_index.append(center_frequencies_256[l])

    """----------------------------------a sliding 0.5 Bark-wide Window|-----------------------------------------------"""
    # After that point tonal and noisemaskers are thresholded
    # Convert the thresholded indexes to bark scale
    tonal_masker_thresholded_index_bark = f2bark(index2frequency(tonal_masker_thresholded_index))
    noise_masker_thresholded_index_bark = f2bark(index2frequency(noise_masker_thresholded_index))

    P_tonal_masker_decim_1 = []
    tonal_masker_decim_1_index = []
    P_noise_masker_decim_1 = []
    noise_masker_decim_1_index = []
    P_tonal_masker_decim_2 = []
    tonal_masker_decim_2_index = []
    P_noise_masker_decim_2 = []
    noise_masker_decim_2_index = []
    #
    # [tonal_masker_decim_1_index, P_tonal_masker_decim_1 ] = decimateMasker(tonal_masker_thresholded_index_bark, P_tonal_masker_thresholded)
    #
    # [noise_masker_decim_1_index, P_noise_masker_decim_1] = decimateMasker(noise_masker_thresholded_index_bark, P_noise_masker_thresholded)

    # [tonal_masker_decim_2_index, P_tonal_masker_decim_2] = decimateMasker(tonal_masker_decim_1_index,
    #                                                                       P_tonal_masker_thresholded)
    #
    # [tonal_masker_decim_1_index, P_tonal_masker_decim_1] = decimateMasker(tonal_masker_thresholded_index_bark,
    #                                                                       P_tonal_masker_thresholded)

    # Tonal and noise maskers are decimated by 2 in each stage.
    # In the normal case, decimation should not be equal in all bins
    # Decimate tonal maskers

    # if (len(tonal_masker_thresholded_index_bark) == 1):
    #     tonal_masker_decim_1_index.append(tonal_masker_freq_bark[0])

    for k in range(len(tonal_masker_thresholded_index_bark) - 1):
        # If they occur within a distance of 0.5
        if np.absolute(tonal_masker_thresholded_index_bark[k + 1] - tonal_masker_thresholded_index_bark[k]) <= 0.5:
            if P_tonal_masker_thresholded[k + 1] > P_tonal_masker_thresholded[k]:
                P_tonal_masker_decim_1.append(P_tonal_masker_thresholded[k + 1])
                tonal_masker_decim_1_index.append(tonal_masker_thresholded_index_bark[k + 1])
            else:
                P_tonal_masker_decim_1.append(P_tonal_masker_thresholded[k])
                tonal_masker_decim_1_index.append(tonal_masker_thresholded_index_bark[k])
        else:
            P_tonal_masker_decim_1.append(P_tonal_masker_thresholded[k])
            tonal_masker_decim_1_index.append(tonal_masker_thresholded_index_bark[k])
    # P_tonal_masker_decim_1.append(P_tonal_masker_thresholded[len(tonal_masker_thresholded_index_bark) - 1])
    # tonal_masker_decim_1_index.append(tonal_masker_thresholded_index_bark[len(tonal_masker_thresholded_index_bark) - 1])

    # # Decimate noise maskers
    for k in range(len(noise_masker_thresholded_index_bark) - 1):
        if np.absolute(noise_masker_thresholded_index_bark[k + 1] - noise_masker_thresholded_index_bark[k]) <= 0.5:
            if P_noise_masker_thresholded[k + 1] > P_noise_masker_thresholded[k]:
                P_noise_masker_decim_1.append(P_noise_masker_thresholded[k + 1])
                noise_masker_decim_1_index.append(noise_masker_thresholded_index_bark[k + 1])
            else:
                P_noise_masker_decim_1.append(P_noise_masker_thresholded[k])
                noise_masker_decim_1_index.append(noise_masker_thresholded_index_bark[k])
        else:
            P_noise_masker_decim_1.append(P_noise_masker_thresholded[k])
            noise_masker_decim_1_index.append(noise_masker_thresholded_index_bark[k])

    # P_noise_masker_decim_1.append(P_noise_masker_thresholded[len(noise_masker_thresholded_index_bark) - 1])
    # noise_masker_decim_1_index.append(noise_masker_thresholded_index_bark[len(noise_masker_thresholded_index_bark) - 1])

    # This block cross-decimates the noise and tonal maskers
    # Compare the tonal masker with all the noise maskers
    # If the tonal masker is larger, append this value to new tonal masker array
    # If noise masker is larger, append this value to new noise masker array
    #
    # if (len(tonal_masker_decim_1_index) == 1):
    #     tonal_masker_decim_2_index.append(tonal_masker_decim_1_index[0])

    for k in range(len(tonal_masker_decim_1_index)):
        oneandonlyone = True
        for j in range(len(noise_masker_decim_1_index)):
            if np.absolute(tonal_masker_decim_1_index[k] - noise_masker_decim_1_index[j]) <= 0.5:
                oneandonlyone = False
                if P_tonal_masker_decim_1[k] < P_noise_masker_decim_1[j]:
                    P_noise_masker_decim_2.append(P_noise_masker_decim_1[j])
                    noise_masker_decim_2_index.append(noise_masker_decim_1_index[j])
                if P_tonal_masker_decim_1[k] >= P_noise_masker_decim_1[j]:
                    P_tonal_masker_decim_2.append(P_tonal_masker_decim_1[k])
                    tonal_masker_decim_2_index.append(tonal_masker_decim_1_index[k])
        if oneandonlyone:
            P_tonal_masker_decim_2.append(P_tonal_masker_decim_1[k])
            tonal_masker_decim_2_index.append(tonal_masker_decim_1_index[k])

    # Cross-decimation for noise maskers
    for k in range(len(noise_masker_decim_1_index)):
        oneandonlyone = True
        for j in range(len(tonal_masker_decim_2_index)):
            if np.absolute(tonal_masker_decim_2_index[j] - noise_masker_decim_1_index[k]) <= 0.5:
                oneandonlyone = False

        if oneandonlyone:
            P_noise_masker_decim_2.append(P_noise_masker_decim_1[k])
            noise_masker_decim_2_index.append(noise_masker_decim_1_index[k])

    """----------------------------Step 4: Calculation of Individual Masking Thresholds.---------------------------"""
    noise_masker_threshold = []
    noise_masker_threshold_index = []
    tonal_masker_threshold = []
    tonal_masker_threshold_index = []

    for k in range(len(tonal_masker_decim_2_index)):
        j = tonal_masker_decim_2_index[k]
        masker_threshold_temp = []
        masker_index_temp = []
        for i in np.arange(j - 3, j + 8, 0.1):

            # According to Eq. 5.30
            masker_threshold_temp.append(P_tonal_masker_decim_2[k] - 0.175 * j - 2.025 + SpreadingFunction(i, j, P_tonal_masker_decim_2[k]))
            masker_index_temp.append(i)

        tonal_masker_threshold.append(masker_threshold_temp)
        tonal_masker_threshold_index.append(masker_index_temp)

    for k in range(len(noise_masker_decim_2_index)):
        j = noise_masker_decim_2_index[k]
        masker_threshold_temp = []
        masker_index_temp = []
        for i in np.arange(j - 3, j + 8, 0.1):
            # According to Eq. 5.31
            masker_threshold_temp.append(P_noise_masker_decim_2[k] - 0.175 * j - 2.025 + SpreadingFunction(i, j, P_noise_masker_decim_2[k]))
            masker_index_temp.append(i)

        noise_masker_threshold.append(masker_threshold_temp)
        noise_masker_threshold_index.append(masker_index_temp)

    """--------------------------Step 5: Calculation of Global Masking Thresholds.------------------------------"""
    global_masking_threshold = []
    global_masking_threshold_index = []
    for i in np.arange(0, 25, 0.1):

        total = 0
        temp_storage = []

        # First term
        for j in range(len(bark_frequencies)):
            if i <= bark_frequencies[j] < i + 0.1:
                temp_storage.append(threshold[j])

        if len(temp_storage) > 0:
            total += 10 ** (0.1 * np.mean(temp_storage))

        temp_storage = []

        # first sigma
        for n in range(len(noise_masker_threshold_index)):
            for k in range(len(noise_masker_threshold_index[n])):
                if i <= noise_masker_threshold_index[n][k] < i + 0.1:
                    temp_storage.append(noise_masker_threshold[n][k])
                    if len(temp_storage) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, temp_storage)))
        temp_storage = []

        # second sigma
        for t in range(len(tonal_masker_threshold_index)):
            for k in range(len(tonal_masker_threshold_index[t])):
                if i <= tonal_masker_threshold_index[t][k] < i + 0.1:
                    temp_storage.append(tonal_masker_threshold[t][k])
                    if len(temp_storage) > 0:
                        total += np.sum(np.power(10, np.multiply(0.1, temp_storage)))

        # last calculation
        if total:
            global_masking_threshold.append(10 * np.log10(total))
            global_masking_threshold_index.append(i)

    ''' Compare Global Masking Threshold with Absolute Threshold of Hearing'''
    ath_for_gmt = 3.64 * ((bark2f(np.asarray(global_masking_threshold_index)) / 1000) ** -0.8) - 6.5 * np.exp(
        -0.6 * ((bark2f(np.asarray(global_masking_threshold_index)) / 1000 - 3.3) ** 2)) + (
                          10 ** -3) * ((bark2f(np.asarray(global_masking_threshold_index)) / 1000) ** 4)
    # ath_for_gmt[index2frequency(187): index2frequency(256)] = 69.13  # Last part of the curve is flat

    bark_index_for_gmt = f2bark(bark2f(np.asarray(global_masking_threshold_index)))
    global_masking_threshold = np.maximum(ath_for_gmt, global_masking_threshold)

    ''' Plotting'''
    plt.figure()
    plt.title('Result after Step 1')
    plt.plot(bark_frequencies, PSD, label='PSD')
    plt.ylabel("SPL (dB)")
    plt.xlabel("Bark")
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('Result after Step 2')
    plt.plot(bark_frequencies, PSD, label='PSD')
    plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'bo', label="Tonal maskers")
    plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise maskers")
    plt.ylabel("SPL (dB)")
    plt.xlabel("Bark")
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('PSD, Threshold and Maskers before thresholding')
    plt.plot(bark_frequencies, PSD, label="PSD")
    plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'bo', label="Tonal maskers")
    plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise maskers")
    plt.plot(bark_frequencies, threshold, label="Threshold")
    plt.ylabel("SPL (dB)")
    plt.xlabel("Bark")
    plt.legend(loc='best')
    plt.xticks(np.arange(1, max(bark_frequencies) + 1, 1))
    plt.show()

    # plt.figure()
    # plt.title('asd')
    # plt.plot(frequency, global_masking_threshold)
    # plt.show()

    plt.figure()
    plt.title('Step3 : Thresholding only')
    plt.plot(bark_frequencies, PSD, label="PSD")
    plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'ro', label="Tonal maskers")
    plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise Maskers")
    plt.plot(f2bark(index2frequency(tonal_masker_thresholded_index)), P_tonal_masker_thresholded, 'go', label="Thresholded Tonal Maskers")
    plt.plot(f2bark(index2frequency(noise_masker_thresholded_index)), P_noise_masker_thresholded, 'gx', label="Thresholded Noise Maskers")
    plt.plot(bark_frequencies, threshold, label="Threshold")
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('PSD, Threshold and Maskers after thresholding')
    plt.plot(frequency, PSD, label="PSD")
    plt.plot(tonal_masker_index, P_tonal_masker, 'ro', label="Tonal maskers")
    plt.plot(center_frequencies_256, noise_masker_arr, 'rx', label="Noise Maskers")
    plt.plot(noise_masker_thresholded_index, P_noise_masker_thresholded, 'gx', label="Thresholded Noise Maskers")
    plt.plot(tonal_masker_thresholded_index, P_tonal_masker_thresholded, 'go', label="Thresholded Tonal Maskers")
    plt.plot(frequency, threshold, label="Threshold")
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('Step 4: Individual Masking Thresholds')
    plt.plot(bark_frequencies, PSD, label="PSD")
    plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'ro', label="Tonal Maskers")
    plt.plot(tonal_masker_decim_2_index, P_tonal_masker_decim_2, 'bo', label="Tonal Maskers Decimated")
    plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise Maskers")
    plt.plot(noise_masker_decim_2_index, P_noise_masker_decim_2, 'bx', label="Noise Maskers Decimated")
    plt.plot(bark_frequencies, threshold, label="Threshold")
    plt.xticks(np.arange(1, max(bark_frequencies) + 1, 1))

    # Plot individual noisemasker functions
    for i in range(len(noise_masker_threshold)):
        plt.plot(noise_masker_threshold_index[i], noise_masker_threshold[i])

    # plot individual tonal masker functions
    for i in range(len(tonal_masker_threshold)):
        plt.plot(tonal_masker_threshold_index[i], tonal_masker_threshold[i])

    # for i in global_masking_threshold_index:
    #     global_masking_threshold = np.maximum(global_masking_threshold, threshold[i])

    # plt.plot(global_masking_threshold_index, global_masking_threshold, label="Global Masking Threshold")
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('After thresholding and decimating')
    plt.plot(bark_frequencies, PSD, label="PSD")
    plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'ro', label="Tonal Maskers")
    plt.plot(tonal_masker_decim_2_index, P_tonal_masker_decim_2, 'bo', label="Tonal Maskers Decimated")
    plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise Maskers")
    plt.plot(noise_masker_decim_2_index, P_noise_masker_decim_2, 'bx', label="Noise Maskers Decimated")
    plt.plot(bark_frequencies, threshold, label="Threshold")
    plt.xticks(np.arange(1, max(bark_frequencies) + 1, 1))
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.title('Step 5: Calculation of Global Masking Threshold')
    plt.plot(bark_frequencies, PSD, label="PSD")
    # plt.plot(tonal_masker_freq_bark, P_tonal_masker, 'ro', label="Tonal Maskers")
    plt.plot(tonal_masker_decim_2_index, P_tonal_masker_decim_2, 'bo', label="Tonal Maskers Decimated")
    # plt.plot(center_frequencies_bark, noise_masker_arr, 'rx', label="Noise Maskers")
    plt.plot(noise_masker_decim_2_index, P_noise_masker_decim_2, 'bx', label="Noise Maskers Decimated")
    plt.plot(bark_frequencies, threshold, label="Threshold")
    plt.plot(global_masking_threshold_index, global_masking_threshold, label="Global Masking Threshold")
    plt.xticks(np.arange(1, max(bark_frequencies) + 1, 1))

    # Plot individual noisemasker functions
    for i in range(len(noise_masker_threshold)):
        plt.plot(noise_masker_threshold_index[i], noise_masker_threshold[i])

    # plot individual tonal masker functions
    for i in range(len(tonal_masker_threshold)):
        plt.plot(tonal_masker_threshold_index[i], tonal_masker_threshold[i])

    # for i in global_masking_threshold_index:
    #     global_masking_threshold = np.maximum(global_masking_threshold, threshold[i])

    # plt.plot(global_masking_threshold_index, global_masking_threshold, label="Global Masking Threshold")
    plt.legend(loc='best')
    plt.show()



    print('Bark2F = {}'.format(bark2f(13)))
    print('index2bark = {}'.format(index2bark(187)))
    print('index2bark = {}'.format(index2bark(256)))
    print(bark_index_for_gmt)


    plt.figure()
    plt.title('Global Masking Threshold and ATH')
    plt.plot(global_masking_threshold_index,  global_masking_threshold, label="Global Masking Threshold")
    # plt.plot(bark_frequencies, PSD, label="PSD")
    plt.plot(bark_index_for_gmt, ath_for_gmt, label='Absolute Threshold of Hearing')
    # plt.plot(bark_frequencies, threshold, 'bo', label="Threshold")
    # plt.plot(tonal_masker_freq_bark, P_tonal_masker,'bo')
    plt.legend(loc='best')
    plt.show()



