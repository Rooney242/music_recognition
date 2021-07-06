#tutorials
# https://essentia.upf.edu/essentia_python_tutorial.html
import matplotlib.pyplot as plt
import essentia 
import essentia.standard as es
import numpy as np
import pandas as pd
import sys

'''pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)'''

#definitions
ann_path = '../emomusic/annotations/'
clips_path = '../emomusic/clips/'
ext = '.mp3'
dsr = 44100

#musical dimension, musical element, musical feature, characteristic
feat_dic = {
    'melody': {
        'pitch':{
            'pitch_salience': {
                'pitch_salience': ('lowlevel.pitch_salience.mean', 1)
            }
        }
    },
    'harmony': {
        'intervals': {
            'chromagram':{
                'hpcp': ('tonal.hpcp.mean', 36),
                'hpcp_crest': ('tonal.hpcp_crest.mean', 1),
                'hpcp_entropy': ('tonal.hpcp_entropy.mean', 1)
            },
            'chord_sequence':{
                'changes_rate': ('tonal.chords_changes_rate', 1),
                'histogram': ('tonal.chords_histogram', 24),
                'chords_key_str': ('tonal.chords_key', 1),
                'number_rate': ('tonal.chords_number_rate', 1),
                'scale_str': ('tonal.chords_scale', 1),
                'strength': ('tonal.chords_strength.mean', 1)
            }
        },
        'tonality': {
            'tuning_frequency': {
                'tuning_frequency': ('tonal.tuning_frequency', 1)
            },
            'key': {
                'edma_str': ('tonal.key_edma.key', 1),
                'krumhansl_str': ('tonal.key_krumhansl.key', 1),
                'temperley_str': ('tonal.key_temperley.key', 1)
            },
            'key_strength': {
                'tuning_diatonic_strength': ('tonal.tuning_diatonic_strength', 1),
                'edma_strength': ('tonal.key_edma.strength', 1),
                'krumhansl_strength': ('tonal.key_krumhansl.strength', 1),
                'temperley_str': ('tonal.key_temperley.strength', 1)
            },
        },
        'mode': {
            'scale': {
                'edma_str': ('tonal.key_edma.scale', 1),
                'krumhansl_str': ('tonal.key_krumhansl.scale', 1),
                'temperley_str': ('tonal.key_temperley.scale', 1)
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
                'contrast_coefs': ('lowlevel.spectral_contrast_coeffs.mean', 6),
                'valleys': ('lowlevel.spectral_contrast_valleys.mean', 6),
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
                'mel_bands': ('lowlevel.melbands.mean', 40),
                'crest': ('lowlevel.melbands_crest.mean', 1),
                'flatness_db': ('lowlevel.melbands_flatness_db.mean', 1),
                'kurtosis': ('lowlevel.melbands_kurtosis.mean', 1),
                'skewness': ('lowlevel.melbands_skewness.mean', 1),
                'spread': ('lowlevel.melbands_spread.mean', 1)
            },
            'bark_bands': {
                'bark_bands': ('lowlevel.barkbands.mean', 27),
                'crest': ('lowlevel.barkbands_crest.mean', 1),
                'flatness_db': ('lowlevel.barkbands_flatness_db.mean', 1),
                'kurtosis': ('lowlevel.barkbands_kurtosis.mean', 1),
                'skewness': ('lowlevel.barkbands_skewness.mean', 1),
                'spread': ('lowlevel.barkbands_spread.mean', 1)
            },
            'erb_bands': {
                'erb_bands': ('lowlevel.erbbands.mean', 40),
                'crest': ('lowlevel.erbbands_crest.mean', 1),
                'flatness_db': ('lowlevel.erbbands_flatness_db.mean', 1),
                'kurtosis': ('lowlevel.erbbands_kurtosis.mean', 1),
                'skewness': ('lowlevel.erbbands_skewness.mean', 1),
                'spread': ('lowlevel.erbbands_spread.mean', 1)
            },
            'mfcc': {
                'mfcc': ('lowlevel.mfcc.mean', 13)
            },
            'gfcc': {
                'gfcc': ('lowlevel.gfcc.mean', 13)
            },
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


#loading all of the annitations
stat_lab = pd.read_csv(ann_path+'static_annotations.csv', index_col=0)
stat_clip_list = stat_lab.index

cont_lab = pd.DataFrame(index=stat_clip_list)

aro_mean = pd.read_csv(ann_path+'arousal_cont_average.csv', index_col=0)
aro_std = pd.read_csv(ann_path+'arousal_cont_std.csv', index_col=0)
val_mean = pd.read_csv(ann_path+'valence_cont_average.csv', index_col=0)
val_std = pd.read_csv(ann_path+'valence_cont_std.csv', index_col=0)

for sam in range(15000, 45001, 500):
    cont_lab[str(sam)+'_arousal_mean'] = aro_mean['sample_'+str(sam)+'ms']
    cont_lab[str(sam)+'_arousal_std'] = aro_std['sample_'+str(sam)+'ms']
    cont_lab[str(sam)+'_valence_mean'] = val_mean['sample_'+str(sam)+'ms']
    cont_lab[str(sam)+'_valence_std'] = val_std['sample_'+str(sam)+'ms']

#loading all the songs and generating the features stat_lab.index

column_list = []
for mus_dim, mus_dim_val in feat_dic.items():
    for mus_elem, mus_elem_val in mus_dim_val.items():
        for mus_feat, mus_feat_val in mus_elem_val.items():
            for char, char_val in mus_feat_val.items():
                if feat_dic[mus_dim][mus_elem][mus_feat][char][1] > 1:
                    for i in range(feat_dic[mus_dim][mus_elem][mus_feat][char][1]):
                        column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char])+'_'+str(i))
                else:
                    column_list.append('.'.join([mus_dim, mus_elem, mus_feat, char]))
column_list.append('arousal_mean')
column_list.append('arousal_std')
column_list.append('valence_mean')
column_list.append('valence_std')

def extract_feature(df, clip_id, feats, mus_dim, mus_elem, mus_feat, char):
    col_name = '.'.join([mus_dim, mus_elem, mus_feat, char])
    ess_rel = feat_dic[mus_dim][mus_elem][mus_feat][char]
    if ess_rel[1] <= 1:
        #print(col_name, feats[ess_rel[0]])
        df.loc[clip_id][col_name] = feats[ess_rel[0]]
    else:
        for i in range(ess_rel[1]):
            #print(col_name+'_'+str(i), feats[ess_rel[0]][i])
            df.loc[clip_id][col_name+'_'+str(i)] = feats[ess_rel[0]][i]

    return df

#####################
## STATIC FEATURES ##
#####################
'''stat_feat = pd.DataFrame(index=stat_clip_list, columns=column_list)
for i, clip_id in enumerate(stat_clip_list):
    print('Extracting features from song {} of {}'.format(i, len(stat_clip_list)))
    clip_path = clips_path+str(clip_id)+ext
    #loader = essentia.standard.MonoLoader(filename=clip_path)
    #audio = loader()
    feats, feat_frames = es.MusicExtractor(
        startTime=0,
        endTime=45,
        lowlevelStats=['mean'],
        lowlevelSilentFrames='keep',
        rhythmStats=['mean'],
        tonalStats=['mean'],
        tonalSilentFrames='keep',
        gfccStats=['mean'],
        mfccStats=['mean'])(clip_path)

    for mus_dim, mus_dim_val in feat_dic.items():
        for mus_elem, mus_elem_val in mus_dim_val.items():
            for mus_feat, mus_feat_val in mus_elem_val.items():
                for char, char_val in mus_feat_val.items():
                    stat_feat = extract_feature(stat_feat, clip_id, feats, mus_dim, mus_elem, mus_feat, char)
                    stat_feat.loc[clip_id]['arousal_mean'] = stat_lab.loc[clip_id]['mean_arousal']
                    stat_feat.loc[clip_id]['arousal_std'] = stat_lab.loc[clip_id]['std_arousal']
                    stat_feat.loc[clip_id]['valence_mean'] = stat_lab.loc[clip_id]['mean_valence']
                    stat_feat.loc[clip_id]['valence_std'] = stat_lab.loc[clip_id]['std_valence']

stat_feat.to_parquet(ann_path+'static_features.pqt')'''


#########################
## CONTINUOUS FEATURES ##
#########################

'''cont_clip_list = []
for clip_id in stat_clip_list:
    for sam in range(15000, 45001, 500):
        cont_clip_list.append(str(clip_id)+'_'+str(sam))

cont_feat = pd.DataFrame(index=cont_clip_list, columns=column_list)
for i, clip in enumerate(cont_clip_list):
    print('Extracting features from song {} of {}'.format(i, len(cont_clip_list)))

    clip_id = clip.split('_')[0]
    clip_end_ms = int(clip.split('_')[1])
    clip_end = float(clip_end_ms)/1000
    clip_start = clip_end - 15

    clip_path = clips_path+clip_id+ext

    #loader = essentia.standard.MonoLoader(filename=clip_path)
    #audio = loader()
    feats, feat_frames = es.MusicExtractor(
        startTime=clip_start,
        endTime=clip_end,
        lowlevelStats=['mean'],
        lowlevelSilentFrames='keep',
        rhythmStats=['mean'],
        tonalStats=['mean'],
        tonalSilentFrames='keep',
        gfccStats=['mean'],
        mfccStats=['mean'])(clip_path)

    for mus_dim, mus_dim_val in feat_dic.items():
        for mus_elem, mus_elem_val in mus_dim_val.items():
            for mus_feat, mus_feat_val in mus_elem_val.items():
                for char, char_val in mus_feat_val.items():
                    cont_feat = extract_feature(cont_feat, clip, feats, mus_dim, mus_elem, mus_feat, char)
                    cont_feat.loc[clip]['arousal_mean'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_arousal_mean']
                    cont_feat.loc[clip]['arousal_std'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_arousal_std']
                    cont_feat.loc[clip]['valence_mean'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_valence_mean']
                    cont_feat.loc[clip]['valence_std'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_valence_std']

cont_feat.to_parquet(ann_path+'cont_features.pqt')'''

###########################
## CONTINUOUS FEATURES 2 ##
###########################

#This time, we get intervals of 5 seconds and with an overlap of 2
window_size_ms = 5000
window_shift = 2500
cont_clip_list = []
for clip_id in stat_clip_list[:3]:
    for sam in range(window_size_ms, 45001, window_shift):
        cont_clip_list.append(str(clip_id)+'_'+str(sam))

cont_feat = pd.DataFrame(index=cont_clip_list, columns=column_list)

for i, clip in enumerate(cont_clip_list):
    print('Extracting features from song {} of {}'.format(i, len(cont_clip_list)))

    clip_id = clip.split('_')[0]
    clip_end_ms = int(clip.split('_')[1])
    clip_start_ms = clip_end_ms - window_size_ms
    clip_start = clip_start_ms/1000
    clip_end = clip_end_ms/1000

    clip_path = clips_path+clip_id+ext

    #loader = essentia.standard.MonoLoader(filename=clip_path)
    #audio = loader()
    try:
        feats, feat_frames = es.MusicExtractor(
            startTime=clip_start,
            endTime=clip_end,
            lowlevelStats=['mean'],
            lowlevelSilentFrames='keep',
            rhythmStats=['mean'],
            tonalStats=['mean'],
            tonalSilentFrames='keep',
            gfccStats=['mean'],
            mfccStats=['mean'])(clip_path)
    except:
        feats = last_feats
    last_feats = feats 

    for mus_dim, mus_dim_val in feat_dic.items():
        for mus_elem, mus_elem_val in mus_dim_val.items():
            for mus_feat, mus_feat_val in mus_elem_val.items():
                for char, char_val in mus_feat_val.items():
                    cont_feat = extract_feature(cont_feat, clip, feats, mus_dim, mus_elem, mus_feat, char)
                    if clip_end_ms >= 15000:
                        cont_feat.loc[clip]['arousal_mean'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_arousal_mean']
                        cont_feat.loc[clip]['arousal_std'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_arousal_std']
                        cont_feat.loc[clip]['valence_mean'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_valence_mean']
                        cont_feat.loc[clip]['valence_std'] = cont_lab.loc[int(clip_id)][str(clip_end_ms)+'_valence_std']
                    else:
                        cont_feat.loc[clip]['arousal_mean'] = cont_feat.loc[clip]['arousal_std'] = cont_feat.loc[clip]['valence_mean'] = cont_feat.loc[clip]['valence_std'] = 0

##Estimation of empty values
weigths = [round(i, 3) for i in np.arange(0, 1, 1/((15000-window_size_ms)/window_shift))]

for clip_id in stat_clip_list:
    for i, end_ms in enumerate(range(window_size_ms, 15000, window_shift)):
        cont_feat.loc[str(clip_id)+'_'+str(end_ms)]['arousal_mean'] = cont_feat.loc[str(clip_id)+'_15000']['arousal_mean']*weigths[i]
        cont_feat.loc[str(clip_id)+'_'+str(end_ms)]['arousal_std'] = cont_feat.loc[str(clip_id)+'_15000']['arousal_std']*weigths[i]
        cont_feat.loc[str(clip_id)+'_'+str(end_ms)]['valence_mean'] = cont_feat.loc[str(clip_id)+'_15000']['valence_mean']*weigths[i]
        cont_feat.loc[str(clip_id)+'_'+str(end_ms)]['valence_std'] = cont_feat.loc[str(clip_id)+'_15000']['valence_std']*weigths[i]

#cont_feat.to_parquet(ann_path+'cont_features_'+str(window_size_ms)+'_new.pqt')