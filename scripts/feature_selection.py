import pandas as pd
import numpy as np
import sys

ann_path = '../emomusic/annotations/'
clips_path = '../emomusic/clips/'
ext = '.mp3'
dsr = 44100

#musical dimension, musical element, musical feature, characteristic
stat_feat_dic = {
	'harmony': {
        'intervals': {
            'chromagram':{
                #'hpcp': ('tonal.hpcp.mean', 36),
                'hpcp_crest': ('tonal.hpcp_crest.mean', 1),
                'hpcp_entropy': ('tonal.hpcp_entropy.mean', 1)
            },
            'chord_sequence':{
                'changes_rate': ('tonal.chords_changes_rate', 1),
                #'histogram': ('tonal.chords_histogram', 24),
                'number_rate': ('tonal.chords_number_rate', 1),
                'strength': ('tonal.chords_strength.mean', 1)
            }
        },
        'tonality': {
            'tuning_frequency': {
                'tuning_frequency': ('tonal.tuning_frequency', 1)
            },
            'key': {
                'edma_str': ('tonal.key_edma.key', 1)
            },
            'key_strength': {
                'edma_strength': ('tonal.key_edma.strength', 1)
            }
        },
        'mode': {
            'scale': {
                'edma_str': ('tonal.key_edma.scale', 1)
            }
        }
    },
    'rhythm': {
        'tempo': {
            'event_density': {
                'onset_rate': ('rhythm.onset_rate', 1)
            },
            'tempo': {
                'bpm': ('rhythm.bpm', 1)
            },
            'hwps': {
                'first_peak_bpm': ('rhythm.bpm_histogram_first_peak_bpm', 1),
                'first_peak_weight': ('rhythm.bpm_histogram_first_peak_weight', 1),
                'second_peak_bpm': ('rhythm.bpm_histogram_second_peak_bpm', 1),
                'second_peak_weight': ('rhythm.bpm_histogram_second_peak_weight', 1),
                'second_peak_spread': ('rhythm.bpm_histogram_second_peak_spread', 1)
            }
        },
        'types': {
            'clarity': {
                'beats_loudness': ('rhythm.beats_loudness.mean', 1),
                'beats_loudness_band_ratio': ('rhythm.beats_loudness_band_ratio.mean', 6)
            },
            'danceability': {
                'danceability': ('rhythm.danceability', 1)
            }
        },
    },
}

'''cont_feat_dic = {
    'timbre': {
        'spectral_features': {
            'mel_bands': {
                'mel_bands': ('lowlevel.melbands.mean', 40)
            },
            'mfcc': {
                'mfcc': ('lowlevel.mfcc.mean', 13)
            },
        }
    },
}'''

cont_feat_dic = {
    'melody': {
        'pitch':{
            'pitch_salience': {
                'pitch_salience': ('lowlevel.pitch_salience.mean', 1)
            }
        }
    },
    'dynamics': {
        'levels': {
            'rms_energy': {
                'spectral_energy': ('lowlevel.spectral_rms.mean', 1)
            },
            'loudness': {
                'loudness': ('lowlevel.average_loudness', 1),
                'loudness_ebu128_integrated': ('lowlevel.loudness_ebu128.integrated', 1),
                'loudness_ebu128_momentary': ('lowlevel.loudness_ebu128.momentary.mean', 1),
                'loudness_ebu128_short_term': ('lowlevel.loudness_ebu128.short_term.mean', 1)
            }
        }
    },
    'timbre': {
        'amplitude_envelope': {
            'zero_crossing_rate': {
                'zero_crossing_rate': ('lowlevel.zerocrossingrate.mean', 1)
            }
        },
        'spectral_features': {
            'characteristics': {
                'entropy': ('lowlevel.spectral_entropy.mean', 1),
                'flux': ('lowlevel.spectral_flux.mean', 1),
                'rolloff': ('lowlevel.spectral_rolloff.mean', 1),
                #'contrast_coefs': ('lowlevel.spectral_contrast_coeffs.mean', 6),
                #'valleys': ('lowlevel.spectral_contrast_valleys.mean', 6),
                'complexity': ('lowlevel.spectral_complexity.mean', 1),
                'decrease': ('lowlevel.spectral_decrease.mean', 1),
                'strongpeak': ('lowlevel.spectral_strongpeak.mean', 1)
            },
            'high_frequency_energy': {
                'high_frequency_components': ('lowlevel.hfc.mean', 1),
                'energy': ('lowlevel.spectral_energy.mean', 1),
                'energyband_low': ('lowlevel.spectral_energyband_low.mean', 1),
                'energyband_low_middle': ('lowlevel.spectral_energyband_middle_low.mean', 1),
                'energyband_middle_high': ('lowlevel.spectral_energyband_middle_high.mean', 1),
                'energyband_high': ('lowlevel.spectral_energyband_high.mean', 1)
            },
            'moments': {
                'centroid': ('lowlevel.spectral_centroid.mean', 1),
                'spread': ('lowlevel.spectral_spread.mean', 1),
                'skewness': ('lowlevel.spectral_skewness.mean', 1),
                'kurtosis': ('lowlevel.spectral_kurtosis.mean', 1)
            },
            'mel_bands': {
                #'mel_bands': ('lowlevel.melbands.mean', 40),
                'crest': ('lowlevel.melbands_crest.mean', 1),
                'flatness_db': ('lowlevel.melbands_flatness_db.mean', 1),
                'kurtosis': ('lowlevel.melbands_kurtosis.mean', 1),
                'skewness': ('lowlevel.melbands_skewness.mean', 1),
                'spread': ('lowlevel.melbands_spread.mean', 1)
            },
            #'mfcc': {
            #    'mfcc': ('lowlevel.mfcc.mean', 13)
            #},
            'sensory_dissonance': {
                'sensory_dissonance': ('lowlevel.dissonance.mean', 1)
            }
        }
    },
    'expresivity': {
        'articulation': {
            'silence_ratio':{
                'silence_20db': ('lowlevel.silence_rate_20dB.mean', 1),
                'silence_30db': ('lowlevel.silence_rate_30dB.mean', 1),
                'silence_60db': ('lowlevel.silence_rate_60dB.mean', 1)
            }
        }
    }
}

#####################
## STATIC FEATURES ##
#####################
'''stat_column_list = []
for mus_dim, mus_dim_val in stat_feat_dic.items():
    for mus_elem, mus_elem_val in mus_dim_val.items():
        for mus_feat, mus_feat_val in mus_elem_val.items():
            for char, char_val in mus_feat_val.items():
                if stat_feat_dic[mus_dim][mus_elem][mus_feat][char][1] > 1:
                    for i in range(stat_feat_dic[mus_dim][mus_elem][mus_feat][char][1]):
                        stat_column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char])+'_'+str(i))
                else:
                    stat_column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char]))
stat_column_list.append('arousal_mean')
stat_column_list.append('arousal_std')
stat_column_list.append('valence_mean')
stat_column_list.append('valence_std')

df = pd.read_parquet(ann_path+'static_features.pqt')
df = df[stat_column_list]

key_to_int = {
	'A': 0,
	'Bb': 1,
	'B': 2,
	'C': 3,
	'C#': 4,
	'D': 5,
	'Eb': 6,
	'E': 7,
	'F': 8,
	'F#': 9,
	'G': 10,
	'Ab': 11,
}
df['harmony.tonality.key.edma_key_sin'] = df['harmony.tonality.key.edma_str'].apply(lambda x: np.sin(2 * np.pi * key_to_int[x]/11.0))
df['harmony.tonality.key.edma_key_cos'] = df['harmony.tonality.key.edma_str'].apply(lambda x: np.cos(2 * np.pi * key_to_int[x]/11.0))
del df['harmony.tonality.key.edma_str']

scale_to_int = {
	'minor': 1,
	'major': 2
}
df['harmony.mode.scale.edma_scale'] = df['harmony.mode.scale.edma_str'].apply(lambda x: scale_to_int[x])
del df['harmony.mode.scale.edma_str']

#Normalization of static features
def normalizer_static(x):
    return 2*((x-1)/8)-1

df['arousal_mean'] = df['arousal_mean'].apply(lambda x: normalizer_static(x))
df['arousal_std'] = df['arousal_std'].apply(lambda x: x/4)
df['valence_mean'] = df['valence_mean'].apply(lambda x: normalizer_static(x))
df['valence_std'] = df['valence_std'].apply(lambda x: x/4)

print(df.shape)

df.to_parquet(ann_path+'static_selected_features.pqt')'''

#########################
## CONTINUOUS FEATURES ##
#########################

cont_column_list = []
for mus_dim, mus_dim_val in cont_feat_dic.items():
    for mus_elem, mus_elem_val in mus_dim_val.items():
        for mus_feat, mus_feat_val in mus_elem_val.items():
            for char, char_val in mus_feat_val.items():
                if cont_feat_dic[mus_dim][mus_elem][mus_feat][char][1] > 1:
                    for i in range(cont_feat_dic[mus_dim][mus_elem][mus_feat][char][1]):
                        cont_column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char])+'_'+str(i))
                else:
                    cont_column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char]))
'''cont_column_list.append('arousal_mean')
cont_column_list.append('arousal_std')
cont_column_list.append('valence_mean')
cont_column_list.append('valence_std')'''

stat = pd.read_parquet(ann_path+'static_features.pqt')

df = pd.read_parquet(ann_path+'cont_features_5000.pqt')
df = df[cont_column_list]

print(df.shape)
df.to_parquet(ann_path+'cont_selected_features_5000.pqt')