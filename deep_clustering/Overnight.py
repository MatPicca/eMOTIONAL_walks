'''
This script runs the full data processing and deep clustering pipeline for a given list of participants.
it suppresses output during execution for cleaner logs.
the results are saved and can be analyzed later.
'''


import os
import sys

class Silent:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._null = open(os.devnull, "w")
        sys.stdout = self._null
        sys.stderr = self._null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._null.close()


def run_full_pipeline(run_value):
    """
    run_value: η τιμή που αλλάζει (π.χ. subj_name, minutes, threshold set κ.λπ.)
    """

    print(f"\n==============================")
    print(f" START RUN: {run_value}")
    print(f"==============================\n")

    # %%
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    from io import StringIO
    import json
    import pickle
    import matplotlib.pyplot as plt
    import os
    import torch
    import sys
    import gc
    import matplotlib
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.metrics import silhouette_score

    # %%
    FORCE_DEVICE = "cuda"   

    if FORCE_DEVICE.startswith("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""    
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


    # %%
    #Check whether we have GPU to use

    def gpu_report():
        if not torch.cuda.is_available():
            print("CUDA not available -> running on CPU")
            return
        n = torch.cuda.device_count()
        print(f"{n} CUDA device(s) available")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / (1024**3)
            print(f"[{i}] {props.name} | {gb:.1f} GB | CC {props.major}.{props.minor}")

    gpu_report()

    # %%
    subj_name = run_value
    # problems: 489, 473, 457, 105, 416, 105 one, 114 FAIL, 117 NUMERIC,
    minutes = 5
    personal_ID = 1036

    # %%
    # Append the **parent** of the deepemotions folder
    sys.path.append(f"/run/user/{personal_ID}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/Code/Marta")
    from deepemotions.preprocess import preprocessing
    from deepemotions.questionnaire_utils import read_quest
    from deepemotions.utils import resampling_labels, accuracy, init_membership
    from deepemotions.FINAL_deepcluster_NOTSEQ_GPU import DeepCMeams

    # %%
    #Load the data
    plt.rcParams.update({'font.size': 18})

    global_path = f'/run/user/{personal_ID}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/'

    # --- option A: os.path.join ---
    ema_join   = os.path.join(global_path, 'Xing', 'FINAL',
                            'eMotionalCities - DTU_ema_responses_2024-08-14T10_30_10+00_00.csv')
    trip_join  = os.path.join(global_path, 'Xing', 'FINAL',
                            'eMotionalCities - DTU_trip_diary_data_2024-08-14T10_30_08+00_00.csv')
    dict_join  = os.path.join(global_path, 'Xing', 'FINAL',
                            'dictionary_for_categorical_fields_Eng.csv')

    # Load EMA
    ema=pd.read_csv(ema_join)
    trip_diary_data=pd.read_csv(trip_join, encoding='utf8')
    dict_categorical=pd.read_csv(dict_join, sep=";", encoding='cp1252')
    ema.head()

    # %%
    ### Process Dictionary ###
    dict_categorical_trip=dict_categorical.iloc[12:25,:]
    dict_categorical_trip=dict_categorical_trip[['Value','Meaning']]
    dict_categorical_act=dict_categorical.iloc[151:168,:]
    dict_categorical_act=dict_categorical_act[['Value','Meaning']]
    dict_categorical_trip.reset_index(drop=True,inplace=True)
    dict_categorical_act.reset_index(drop=True,inplace=True)
    print(dict_categorical_trip)
    print(dict_categorical_act)

    # %%
    cols_to_check = [
        'morningFeelings', 'afternoonFeelings', 'eveningFeelings',
        'dayStress',
        'Updated at (TZ Aware)', 'Updated at (TZ Aware).1', 'Updated at (TZ Aware).2'
    ]

    for c in cols_to_check:
        if c in ema.columns:
            print(f"\n{c} — first 5 values:")
            print(ema[c].head(3).to_string(index=False))

    # %%
    ### Process EMA ###

    # Split how you feel into seperate columns for monring afternoon evening.

    # Monring 
    ema_nanmorning=ema.loc[ema.morningFeelings.isna(),:]
    ema_NOTnanmorning=ema.loc[~ema.morningFeelings.isna(),:]

    scale_EMA=ema_NOTnanmorning['morningFeelings'].apply(lambda x: x.split(':'))
    scale_EMA=scale_EMA.values[0]
    scale_EMA=scale_EMA[:-1]
    scale_EMA=[j.split(',') for j in scale_EMA]
    scale_EMA=[j[-1] for j in scale_EMA]
    scale_EMA_morning=scale_EMA
    scale_EMA_afternoon=scale_EMA
    scale_EMA_evening=scale_EMA

    scale_EMA_morning=['emaMorning_+'+j for j in scale_EMA_morning]
    scale_EMA_afternoon=['emaAfternoon_'+j for j in scale_EMA_afternoon]
    scale_EMA_evening=['emaEvening_'+j for j in scale_EMA_evening]

    s=ema_NOTnanmorning['morningFeelings'].apply(lambda x: x.split(': ') )
    for i in range(len(s)):
        s.values[i]=s.values[i][1:]
        s.values[i]=[j[0] for j in s.values[i]]
    ema_NOTnanmorning=ema_NOTnanmorning.drop(['morningFeelings'],axis=1)
    ema_nanmorning=ema_nanmorning.drop(['morningFeelings'],axis=1)
    morning=pd.DataFrame(s.tolist(),index=ema_NOTnanmorning.index,columns=scale_EMA_morning)
    morning=morning.astype(int)
    ema_NOTnanmorning=pd.concat([ema_NOTnanmorning,morning],axis=1)
    ema_nanmorning=pd.concat([ema_nanmorning,pd.DataFrame(np.nan,index=ema_nanmorning.index,columns=scale_EMA_morning)],axis=1)

    ema=pd.concat([ema_NOTnanmorning,ema_nanmorning],axis=0)

    # Afternoon
    ema_nanafternoon=ema.loc[ema.afternoonFeelings.isna(),:]
    ema_NOTnanafternoon=ema.loc[~ema.afternoonFeelings.isna(),:]

    s=ema_NOTnanafternoon['afternoonFeelings'].apply(lambda x: x.split(': ') )
    for i in range(len(s)):
        s.values[i]=s.values[i][1:]
        s.values[i]=[j[0] for j in s.values[i]]
    ema_NOTnanafternoon=ema_NOTnanafternoon.drop(['afternoonFeelings'],axis=1)
    ema_nanafternoon=ema_nanafternoon.drop(['afternoonFeelings'],axis=1)
    afternoon=pd.DataFrame(s.tolist(),index=ema_NOTnanafternoon.index,columns=scale_EMA_afternoon)
    afternoon=afternoon.loc[afternoon['emaAfternoon_Unwell - Well']!='O']
    afternoon=afternoon.astype(int)
    ema_NOTnanafternoon=pd.concat([ema_NOTnanafternoon,afternoon],axis=1)
    ema_nanafternoon=pd.concat([ema_nanafternoon,pd.DataFrame(np.nan,index=ema_nanafternoon.index,columns=scale_EMA_afternoon)],axis=1)

    ema=pd.concat([ema_NOTnanafternoon,ema_nanafternoon],axis=0)

    # Evening
    ema_nanevening=ema.loc[ema.eveningFeelings.isna(),:]
    ema_NOTnanevening=ema.loc[~ema.eveningFeelings.isna(),:]

    s=ema_NOTnanevening['eveningFeelings'].apply(lambda x: x.split(': ') )
    for i in range(len(s)):
        s.values[i]=s.values[i][1:]
        s.values[i]=[j[0] for j in s.values[i]]
    ema_NOTnanevening=ema_NOTnanevening.drop(['eveningFeelings'],axis=1)
    ema_nanevening=ema_nanevening.drop(['eveningFeelings'],axis=1)
    evening=pd.DataFrame(s.tolist(),index=ema_NOTnanevening.index,columns=scale_EMA_evening)
    evening=evening.astype(int)
    ema_NOTnanevening=pd.concat([ema_NOTnanevening,evening],axis=1)
    ema_nanevening=pd.concat([ema_nanevening,pd.DataFrame(np.nan,index=ema_nanevening.index,columns=scale_EMA_evening)],axis=1)

    ema=pd.concat([ema_NOTnanevening,ema_nanevening],axis=0)
    ema.sort_index(inplace=True)
    ema.head()

    # %%
    #Last process of EMA

    #Rename PSQI and convert sleep quality to numbers
    ema.rename({'lastNightSleep':'PSQI'},axis=1,inplace=True)
    ema.loc[ema.PSQI=='Very good','PSQI']=4
    ema.loc[ema.PSQI=='Fairly good','PSQI']=3
    ema.loc[ema.PSQI=='Fairly bad','PSQI']=2
    ema.loc[ema.PSQI=='Very bad','PSQI']=1

    #Make text into number for Morning, Afternoon, Evening
    ema.loc[ema.morningFeelingsEnvironment=='To a very large extent','morningFeelingsEnvironment']=5
    ema.loc[ema.morningFeelingsEnvironment=='To a large extent','morningFeelingsEnvironment']=4
    ema.loc[ema.morningFeelingsEnvironment=='To some extent','morningFeelingsEnvironment']=3
    ema.loc[ema.morningFeelingsEnvironment=='To a small extent','morningFeelingsEnvironment']=2
    ema.loc[ema.morningFeelingsEnvironment=='Not at all','morningFeelingsEnvironment']=1

    ema.loc[ema.afternoonFeelingsEnvironment=='To a very large extent','afternoonFeelingsEnvironment']=5
    ema.loc[ema.afternoonFeelingsEnvironment=='To a large extent','afternoonFeelingsEnvironment']=4
    ema.loc[ema.afternoonFeelingsEnvironment=='To some extent','afternoonFeelingsEnvironment']=3
    ema.loc[ema.afternoonFeelingsEnvironment=='To a small extent','afternoonFeelingsEnvironment']=2
    ema.loc[ema.afternoonFeelingsEnvironment=='Not at all','afternoonFeelingsEnvironment']=1

    ema.loc[ema.eveningFeelingsEnvironment=='To a very large extent','eveningFeelingsEnvironment']=5
    ema.loc[ema.eveningFeelingsEnvironment=='To a large extent','eveningFeelingsEnvironment']=4
    ema.loc[ema.eveningFeelingsEnvironment=='To some extent','eveningFeelingsEnvironment']=3
    ema.loc[ema.eveningFeelingsEnvironment=='To a small extent','eveningFeelingsEnvironment']=2
    ema.loc[ema.eveningFeelingsEnvironment=='Not at all','eveningFeelingsEnvironment']=1

    ema_nanstressful     = ema.loc[ema.dayStress.isna(), :].copy()
    ema_NOTnanstressful  = ema.loc[~ema.dayStress.isna(), :].copy()

    #Parse dayStress to extrac only the numeric value
    s = ema_NOTnanstressful['dayStress'].apply(lambda x: x.split(': '))
    for i in range(len(s)):
        s.values[i] = s.values[i][1:][0]
    ema_NOTnanstressful['dayStress'] = s.values

    ema_nanstressful = ema_nanstressful.drop(['dayStress'], axis=1)
    ema = pd.concat([ema_NOTnanstressful, ema_nanstressful], axis=0).sort_index()
    ema.sort_index(inplace=True)

    #Update columns AND concat columns
    list_common=['HHID','INDIVID','Date','PSQI','dayStress']
    ema_morning=ema.loc[:,list_common+['Updated at (TZ Aware)','morningFeelingsReason', 'morningFeelingsEnvironment','morningInteractions']+scale_EMA_morning]
    ema_morning.rename({'Updated at (TZ Aware)':'DateTime_Tz'},axis=1,inplace=True)
    ema_morning.rename({'morningFeelingsReason':'FeelingsReason'},axis=1,inplace=True)
    ema_morning.rename({'morningFeelingsEnvironment':'FeelingsEnvironment'},axis=1,inplace=True)
    ema_morning.rename({'morningInteractions':'FeelingsInteractions'},axis=1,inplace=True)
    ema_morning.columns=list(ema_morning.columns[:-6])+scale_EMA
    ema_morning['Activity']='ema_Morning'

    ema_afternoon=ema.loc[:,list_common+['Updated at (TZ Aware).1','afternoonFeelingsReason', 'afternoonFeelingsEnvironment','afternoonInteractions']+scale_EMA_afternoon]
    ema_afternoon.rename({'Updated at (TZ Aware).1':'DateTime_Tz'},axis=1,inplace=True)
    ema_afternoon.rename({'afternoonFeelingsReason':'FeelingsReason'},axis=1,inplace=True)
    ema_afternoon.rename({'afternoonFeelingsEnvironment':'FeelingsEnvironment'},axis=1,inplace=True)
    ema_afternoon.rename({'afternoonInteractions':'FeelingsInteractions'},axis=1,inplace=True)
    ema_afternoon.columns=list(ema_afternoon.columns[:-6])+scale_EMA
    ema_afternoon['Activity']='ema_Afternoon'

    ema_evening=ema.loc[:,list_common+['Updated at (TZ Aware).2','eveningFeelingsReason', 'eveningFeelingsEnvironment','eveningInteractions']+scale_EMA_evening]
    ema_evening.rename({'Updated at (TZ Aware).2':'DateTime_Tz'},axis=1,inplace=True)
    ema_evening.rename({'eveningFeelingsReason':'FeelingsReason'},axis=1,inplace=True)
    ema_evening.rename({'eveningFeelingsEnvironment':'FeelingsEnvironment'},axis=1,inplace=True)
    ema_evening.rename({'eveningInteractions':'FeelingsInteractions'},axis=1,inplace=True)
    ema_evening.columns=list(ema_evening.columns[:-6])+scale_EMA
    ema_evening['Activity']='ema_Evening'

    ema=pd.concat([ema_morning,ema_afternoon,ema_evening],axis=0)
    ema=ema.reset_index(drop=True)

    #Prase datetime to the format we want.
    ema_nandatetimetz     = ema.loc[ema.DateTime_Tz.isna(), :].copy()
    ema_NOTnandatetimetz  = ema.loc[~ema.DateTime_Tz.isna(), :].copy()

    s = ema_NOTnandatetimetz['DateTime_Tz'].apply(lambda x: x.split(' '))
    for i in range(len(s)):
        s.values[i] = s.values[i][0] + ' ' + s.values[i][1]

    ema_NOTnandatetimetz['DateTime_Tz'] = s.values
    format_string_with_tz = '%Y-%m-%d %H:%M:%S'
    ema_NOTnandatetimetz['DateTime_Tz'] = ema_NOTnandatetimetz['DateTime_Tz'].apply(
        lambda x: datetime.strptime(x, format_string_with_tz)
    )
    ema=ema_NOTnandatetimetz
    ema.sort_index(inplace=True)

    ema['Valence']=ema['Unwell - Well'] + ema['Discontent - Content']
    ema['Arousal']=ema['Tired - Awake'] + ema['Without energy - Full of energy']
    ema['Calmness']=ema['Agitated - Calm'] + ema['Tense - Relaxed']
    ema.head()

    # %%
    # Load raw data
    with open(f"/run/user/{personal_ID}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/E4/completed/{subj_name}/Data/ALL_DAYS_PROCESSED_BIG.pickle", "rb") as f:
        alldata = pickle.load(f)

    alldata.keys()

    # %%
    # Preview pickle
    def show_cols(obj, prefix=""):
        if isinstance(obj, pd.DataFrame):
            print(prefix, "→", list(obj.columns))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                show_cols(v, f"{prefix}{k}/")
        else:
            print(prefix, f"→ {type(obj)}")

    top = next(iter(alldata))       # first top-level key
    print("top key:", top)

    v = alldata[top]                # value under that key
    print("subkeys:", list(v)[:10]) # preview first few subkeys
    show_cols(v)                    # recurse and print columns

    # %%
    # Load raw data
    EDA_subj=pd.DataFrame()
    TEMP_subj=pd.DataFrame()
    BVP_subj=pd.DataFrame()
    RETROSP_subj=pd.DataFrame()

    for key in alldata.keys():
        BVP_subj=pd.concat([BVP_subj,alldata[key]['bvp_subj']],axis=0)
        EDA_subj=pd.concat([EDA_subj,alldata[key]['eda_subj']],axis=0)
        TEMP_subj=pd.concat([TEMP_subj,alldata[key]['temp_subj']],axis=0)
        # Divide by 2 the VAC because the user put it 2 times per day
        df = alldata[key]['retrosp_subj']
        if df is not None and len(df) > 0:
            retrosp = df.dropna(subset=['Valence']).copy()          
            cols = ['Valence', 'Arousal', 'Calmness']
            retrosp.loc[:, cols] = (
                retrosp.loc[:, cols]
                .apply(pd.to_numeric, errors='coerce')               # ensure numeric
                .div(2)                                              # divide by 2
            )
            RETROSP_subj = pd.concat([RETROSP_subj, retrosp], axis=0)
        else:
            print('No Rectrospective on day ', key)

    # %%
    RETROSP_subj=RETROSP_subj.drop_duplicates()
    RETROSP_subj['phase']='x'
    RETROSP_subj['stress_axis']='NO'
    RETROSP_subj.loc[(RETROSP_subj['Calmness']>=5),'stress_axis']='No'
    RETROSP_subj.loc[(RETROSP_subj['Calmness']<5),'stress_axis']='Yes'

    INDIVID_ema=RETROSP_subj.INDIVID.unique()[0]
    ema=ema.loc[ema.INDIVID==INDIVID_ema,:]

    # Get all possible DateTimes.
    t0_bvp=BVP_subj.Timestamp.min()
    tf_bvp=BVP_subj.Timestamp.max()
    BVP_subj.Timestamp.min()==EDA_subj.Timestamp.min()

    fs_bvp=64
    fs_others=4

    df_all_possibilities_bvp=pd.DataFrame(columns=['Timestamp','DateTime'])
    df_all_possibilities_bvp['Timestamp']=np.arange(BVP_subj.Timestamp.min(), BVP_subj.Timestamp.max()+1/fs_bvp, 1/fs_bvp)
    df_all_possibilities_bvp['DateTime']=pd.to_datetime(df_all_possibilities_bvp['Timestamp'],unit='s')
    df_all_possibilities_bvp['DateTime']=df_all_possibilities_bvp['DateTime']+timedelta(hours=1)
    BVP_subj=BVP_subj.sort_values('Timestamp')
    bvp_subj_complete=pd.merge_asof(BVP_subj,df_all_possibilities_bvp,on="Timestamp",tolerance=1/65,direction="nearest")
    df_all_possibilities_bvp=pd.merge(df_all_possibilities_bvp,bvp_subj_complete,left_on="DateTime",right_on="DateTime_y",how="left")

    df_all_possibilities_eda=pd.DataFrame(columns=['Timestamp','DateTime'])
    df_all_possibilities_eda['Timestamp']=np.arange(EDA_subj.Timestamp.min(), EDA_subj.Timestamp.max()+1/fs_others, 1/fs_others)
    df_all_possibilities_eda['DateTime']=pd.to_datetime(df_all_possibilities_eda['Timestamp'],unit='s')
    df_all_possibilities_eda['DateTime']=df_all_possibilities_eda['DateTime']+timedelta(hours=1)
    EDA_subj=EDA_subj.sort_values('Timestamp')
    eda_subj_complete=pd.merge_asof(EDA_subj,df_all_possibilities_eda,on="Timestamp",tolerance=1/5,direction="nearest")
    df_all_possibilities_eda=pd.merge(df_all_possibilities_eda,eda_subj_complete,left_on="DateTime",right_on="DateTime_y",how="left")

    df_all_possibilities_temp=pd.DataFrame(columns=['Timestamp','DateTime'])
    df_all_possibilities_temp['Timestamp']=np.arange(TEMP_subj.Timestamp.min(), TEMP_subj.Timestamp.max()+1/fs_others, 1/fs_others)
    df_all_possibilities_temp['DateTime']=pd.to_datetime(df_all_possibilities_temp['Timestamp'],unit='s')
    df_all_possibilities_temp['DateTime']=df_all_possibilities_temp['DateTime']+timedelta(hours=1)
    TEMP_subj=TEMP_subj.sort_values('Timestamp')
    temp_subj_complete=pd.merge_asof(TEMP_subj,df_all_possibilities_temp,on="Timestamp",tolerance=1/5,direction="nearest")
    df_all_possibilities_temp=pd.merge(df_all_possibilities_temp,temp_subj_complete,left_on="DateTime",right_on="DateTime_y",how="left")

    # Get activities
    Activities=RETROSP_subj

    # dicio_colors=pd.DataFrame(Activities.Activity.unique(),columns=['Activity'])
    # dicio_colors['Color']='White'
    dicio_colors={
        'Bicycle': 'limegreen','Foot': 'yellow','Other': 'gray','DataGap': 'gray','No information provided': 'gray','Unknown': 'gray','Default': 'gray','Bus': 'darkorange','Train/Rail': 'darkgreen','Train': 'darkgreen',
        'Metro': 'forestgreen','Vehicle': 'red','Car/Truck': 'red','Taxi/Car Service': 'orange','Taxi': 'orange','Ferry/Boat': 'blue','Motorcycle': 'coral','Work': 'beige','Education': 'khaki','Home': 'lavender',
        'Change Travel Mode/Transfer': 'olive','Work Related': 'beige','Exercise': 'purple','Exercise/Play Sports': 'purple','Socialize': 'turquoise','Entertainment': 'royalblue','Eat Out/Takeout': 'deepskyblue',
        'Medical': 'pink','Healthcare': 'pink','Shopping': 'black','Accompany/Dropoff/Pickup': 'slateblue','Personal Errands/Tasks': 'brown','Summer cottage, allotment garden, holiday': 'burlywood'
    }
    dicio_colors=pd.DataFrame(list(dicio_colors.items()), columns=['Activity', 'Color'])
    Activities=Activities.merge(dicio_colors,how='left',left_on='Activity',right_on='Activity')
    Activities['Color'] = Activities['Color'].fillna('gray') # MATTEO 

    Activities=Activities.loc[Activities.EndTime>=EDA_subj.DateTime.min()]
    Activities=Activities.loc[Activities.StartTime<=EDA_subj.DateTime.max()]

    ema=ema.loc[ema.DateTime_Tz>=EDA_subj.DateTime.min()]
    ema=ema.loc[ema.DateTime_Tz<=EDA_subj.DateTime.max()]
    ema.sort_values(by='DateTime_Tz',inplace=True)
    ema.reset_index(drop=True,inplace=True)

    Activities_ema=ema
    Activities_ema['EndTime']= Activities_ema['DateTime_Tz'] 
    Activities_ema['StartTime']= Activities_ema['DateTime_Tz'] - pd.Timedelta(minutes=minutes) # 2 minutes before the questionnaire

    Activities_ema_good=pd.DataFrame(columns=Activities_ema.columns)
    for irow in range(len(Activities_ema)):
        EDA_cur=EDA_subj.loc[(EDA_subj.DateTime>=Activities_ema.loc[irow,'StartTime']) & (EDA_subj.DateTime<=Activities_ema.loc[irow,'EndTime']),:]
        if (len(EDA_cur))>0:
            Activities_ema_good=pd.concat([Activities_ema_good,Activities_ema.loc[[irow],:]],axis=0)
    Activities_ema=Activities_ema_good
    Activities_ema.reset_index(drop=True,inplace=True)

    INIT_TIME = 0
    NB_CLASSES = 2 # Calmness high or low. 
    CARDIO_SAMPLING_RATE = 64 #hz

    results_dfcm = {}
    for i in range(NB_CLASSES):
        results_dfcm["class " + str(i)] = []
    results_dfcm["accuracy"] = []
    results_dfcm_train = {}
    for i in range(NB_CLASSES):
        results_dfcm_train["class " + str(i)] = []
    results_dfcm_train["accuracy"] = []
    results_dfcm_val = {}
    for i in range(NB_CLASSES):
        results_dfcm_val["class " + str(i)] = []
    results_dfcm_val["accuracy"] = []
    results_dfcm_test = {}
    for i in range(NB_CLASSES):
        results_dfcm_test["class " + str(i)] = []
    results_dfcm_test["accuracy"] = []

    subjects=[subj_name]
    s=subjects[0]

    score_s=Activities_ema

    df_all_possibilities_eda.loc[df_all_possibilities_eda.Values.isna(),'Values']=0
    df_all_possibilities_bvp.loc[df_all_possibilities_bvp.Values.isna(),'Values']=0
    df_all_possibilities_temp.loc[df_all_possibilities_temp.Values.isna(),'Values']=0

    df_all_possibilities_eda=df_all_possibilities_eda.loc[(df_all_possibilities_eda.DateTime<=score_s.EndTime.max()),:]
    df_all_possibilities_bvp=df_all_possibilities_bvp.loc[(df_all_possibilities_bvp.DateTime<=score_s.EndTime.max()),:]
    df_all_possibilities_temp=df_all_possibilities_temp.loc[(df_all_possibilities_temp.DateTime<=score_s.EndTime.max()),:]

    eda = df_all_possibilities_eda.Values.values 
    ecg = df_all_possibilities_bvp.Values.values
    temp = df_all_possibilities_temp.Values.values

    '''list_Activities=[]
    fig, ax = plt.subplots()
    fig.set_size_inches(30.,15.)
    for i in range(len(Activities)):
        if Activities.iloc[i].Activity in list_Activities:
            ax.axvspan(Activities.iloc[i].StartTime, Activities.iloc[i].EndTime, alpha=0.5, color=Activities.iloc[i].Color)
        else:
            ax.axvspan(Activities.iloc[i].StartTime, Activities.iloc[i].EndTime, alpha=0.5, color=Activities.iloc[i].Color, label=Activities.iloc[i].Activity)
            list_Activities.append(Activities.iloc[i].Activity)

    plt.legend()
    plt.plot(EDA_subj.DateTime,EDA_subj.Values,label='eda')
    plt.legend()'''

    # %%
    # Run Physiological Metrics Code
    # Create signal and time in order to put them later into our clustering model. We also normalize the data

    signal, time = preprocessing(eda, ecg, temp, cardio_sampling_rate=CARDIO_SAMPLING_RATE, bvp=True, init_time=INIT_TIME)
    '''plt.plot(signal)'''

    # ---------------------------
    # Static precomputations
    # ---------------------------

    # Constant label mapping
    phases_list = ['D','C']
    dicio_labels_ids = {'D': 1, 'C': 2}
    time_shift = 0
    LABEL_RATE = 1

    # Precompute time_labels (threshold-independent)
    time_labels = np.arange(0, time.max() * LABEL_RATE, 1) + INIT_TIME

    # Prepare BVP validity table (threshold-independent)
    bvp_valid = df_all_possibilities_bvp.loc[
        ~df_all_possibilities_bvp.Timestamp_x.isna(),
        ['DateTime', 'Timestamp_x']
    ].sort_values('DateTime')

    t0_bvp = df_all_possibilities_bvp.Timestamp_x.min()

    # Keep original for repeated processing
    Activities_ema_base = Activities_ema.copy()


    # %%
    def smart_thresholds(calmness_values, verbose=True):

        uniq = np.sort(np.unique(calmness_values))
        counts = calmness_values.value_counts().to_dict()

        def midpoint(a, b):
            return round((a + b) / 2, 3)

        # ---------------------------------------------------------
        # DEBUG HEADER
        # ---------------------------------------------------------
        if verbose:
            print("\n=============================================")
            print(" SMART EMA THRESHOLD SELECTION - MANOS LOGIC")
            print("=============================================")
            print(f"Unique Calmness values: {list(uniq)}")
            print(f"Counts per value: {counts}")
            print("Total unique:", len(uniq))
            print("---------------------------------------------")

        # ---------------------------------------------------------
        # CASE 1 — Only 1 unique value
        # ---------------------------------------------------------
        if len(uniq) == 1:
            raise ValueError("❌ Cannot split: Only one unique Calmness value.")

        # ---------------------------------------------------------
        # CASE 2 — Two unique values → 1 midpoint
        # ---------------------------------------------------------
        if len(uniq) == 2:
            thr = midpoint(uniq[0], uniq[1])
            if verbose:
                print(f"✔ Case 2: Two unique values → using threshold {thr}")
            return [thr]

        # ---------------------------------------------------------
        # CASE 3 — Three unique → 2 thresholds
        # ---------------------------------------------------------
        if len(uniq) == 3:
            thr1 = midpoint(uniq[0], uniq[1])
            thr2 = midpoint(uniq[1], uniq[2])
            if verbose:
                print(f"✔ Case 3: Three unique → thresholds {thr1}, {thr2}")
            return [thr1, thr2]

        # ---------------------------------------------------------
        # CASE 4 — Four unique → 3 thresholds
        # ---------------------------------------------------------
        if len(uniq) == 4:
            thr = [
                midpoint(uniq[0], uniq[1]),
                midpoint(uniq[1], uniq[2]),
                midpoint(uniq[2], uniq[3])
            ]
            if verbose:
                print(f"✔ Case 4: Four unique → thresholds {thr}")
            return thr

        # ---------------------------------------------------------
        # CASE 5 — 5+ unique values → smart logic
        # ---------------------------------------------------------
        if verbose:
            print("✔ Case 5: 5+ unique → applying advanced Manos logic")
            print("---------------------------------------------")

        # =================================================
        #               LOW THRESHOLD
        # =================================================
        u1, u2 = uniq[0], uniq[1]

        if counts[u1] >= 4:
            thr_low = midpoint(u1, u2)
            if verbose:
                print(f"LOW: u1={u1} count {counts[u1]} ≥ 4 → thr_low = midpoint({u1},{u2}) = {thr_low}")
        else:
            if verbose:
                print(f"LOW: u1={u1} count {counts[u1]} < 4 → rare low cluster")

            if (u2 - u1) <= 1:
                thr_low = midpoint(uniq[1], uniq[2])
                if verbose:
                    print(f"LOW: (u2 - u1) <= 1 → thr_low = midpoint({uniq[1]}, {uniq[2]}) = {thr_low}")
            else:
                thr_low = midpoint(u1, u2)
                if verbose:
                    print(f"LOW: gap >1 → thr_low = midpoint({u1},{u2}) = {thr_low}")

        # =================================================
        #               MID THRESHOLD
        # =================================================
        mid_idx = len(uniq) // 2
        thr_mid = midpoint(uniq[mid_idx - 1], uniq[mid_idx])

        if verbose:
            print(f"MID: middle pair ({uniq[mid_idx - 1]}, {uniq[mid_idx]}) → thr_mid = {thr_mid}")

        # =================================================
        #               HIGH THRESHOLD
        # =================================================
        uk, uk_1 = uniq[-1], uniq[-2]

        if counts[uk] >= 4:
            thr_high = midpoint(uk_1, uk)
            if verbose:
                print(f"HIGH: uk={uk} count {counts[uk]} ≥ 4 → thr_high = midpoint({uk_1},{uk}) = {thr_high}")
        else:
            if verbose:
                print(f"HIGH: uk={uk} count {counts[uk]} < 4 → rare high cluster")

            if (uk - uk_1) <= 1:
                thr_high = midpoint(uniq[-3], uniq[-2])
                if verbose:
                    print(f"HIGH: (uk - uk_1) <= 1 → thr_high = midpoint({uniq[-3]}, {uniq[-2]}) = {thr_high}")
            else:
                thr_high = midpoint(uk_1, uk)
                if verbose:
                    print(f"HIGH: gap >1 → thr_high = midpoint({uk_1},{uk}) = {thr_high}")

        # =========================================================
        # FIX DUPLICATES — Ensure all thresholds are unique
        # =========================================================

        # Fix 1 — mid == low
        if thr_mid == thr_low:
            if verbose:
                print("⚠ FIX: thr_mid equals thr_low → adjusting upward...")

            for i in range(1, len(uniq) - 1):
                cand = midpoint(uniq[i], uniq[i + 1])
                if cand not in [thr_low, thr_high]:
                    thr_mid = cand
                    break

        # Fix 2 — mid == high
        if thr_mid == thr_high:
            if verbose:
                print("⚠ FIX: thr_mid equals thr_high → adjusting downward...")

            for i in range(len(uniq) - 3, -1, -1):
                cand = midpoint(uniq[i], uniq[i + 1])
                if cand not in [thr_low, thr_high]:
                    thr_mid = cand
                    break

        thresholds = sorted([thr_low, thr_mid, thr_high])

        if verbose:
            print("---------------------------------------------")
            print("FINAL SELECTED THRESHOLDS:", thresholds)
            print("=============================================\n")

        return thresholds
    thresholds = smart_thresholds(Activities_ema_base["Calmness"])

    # %%
    Activities_ema.Calmness.hist()

    # %%
    kfolds = 10
    ntrials = 10

    # %%
    import numpy as np
    import pandas as pd
    import random
    import torch
    import optuna

    # =============================================================================
    #  UTILS
    # =============================================================================

    def reset_seeds(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def build_fixed_splits(u_init, k_folds=5):
        idx = np.arange(len(u_init))
        np.random.seed(42)
        np.random.shuffle(idx)
        return np.array_split(idx, k_folds)


    def stress_accuracy(y_true, y_pred, stress_label):
        mask = (y_true == stress_label)
        if np.sum(mask) == 0:
            return np.nan
        return np.mean(y_pred[mask] == y_true[mask])


    # =============================================================================
    #  MAIN OPTIMIZATION PIPELINE
    # =============================================================================

    results_per_threshold = {}

    for ema_threshold in thresholds:

        print("\n========================================================")
        print(f" RUNNING THRESHOLD = {ema_threshold}")
        print("========================================================\n")

        # -------------------------------------------------------------------------
        # (1) Compute stress_axis → score_s
        # -------------------------------------------------------------------------
        Activities_ema = Activities_ema_base.copy()
        Activities_ema["stress_axis"] = np.where(
            Activities_ema["Calmness"] >= ema_threshold, "No", "Yes"
        )

        score_s = Activities_ema.copy()
        score_s["phase"] = np.where(score_s["stress_axis"] == "Yes", "D", "C")

        score_s = score_s.loc[
            score_s.StartTime >= df_all_possibilities_bvp.DateTime.min()
        ].reset_index(drop=True)

        start_idx = pd.merge_asof(
            score_s, bvp_valid, left_on="StartTime", right_on="DateTime"
        ).Timestamp_x
        end_idx = pd.merge_asof(
            score_s, bvp_valid, left_on="EndTime", right_on="DateTime"
        ).Timestamp_x

        score_s["start"] = np.round(start_idx - t0_bvp).astype(int)
        score_s["end"]   = np.round(end_idx   - t0_bvp).astype(int)

        # -------------------------------------------------------------------------
        # (2) Build labels_true
        # -------------------------------------------------------------------------
        labels_true = np.zeros_like(time_labels, dtype=int)
        for i in range(len(score_s)):
            labels_true[
                score_s.loc[i, "start"]: score_s.loc[i, "end"]
            ] = dicio_labels_ids[score_s.loc[i, "phase"]]

        # -------------------------------------------------------------------------
        # (3) Build u_init
        # -------------------------------------------------------------------------
        u_init = init_membership(
            score_s=score_s,
            time=time,
            nb_classes=NB_CLASSES,
            time_shift=time_shift,
            one_hot=True
        )

        # -------------------------------------------------------------------------
        # (4) DEFINE TRAIN-ONE-FOLD
        # -------------------------------------------------------------------------
        def train_one_fold(config, fold_idx):

            reset_seeds(12345 + fold_idx)

            dfcm = DeepCMeams(signal, time)

            try:
                labels_hat, U, time_hat, best_model_wts, losses, losses_AE, \
                losses_silh, centroids, indices_train, indices_test, \
                Z_train, train_Data, test_Data = dfcm.run(
                    nb_classes=NB_CLASSES,
                    seq_len=config["SEQ_LEN"],
                    u_init=u_init,
                    epochs_pre=config["EPOCHS_PRE"],
                    epochs_train=config["EPOCHS_TRAIN"],
                    batch_size=config["BATCH_SIZE"],
                    embedding=config["EMBEDDING"],
                    lr_pretrain=config["LR_PRE"],
                    lr_train=config["LR_TRAIN"],
                    sigma=config["SIGMA"],
                    gamma=config["GAMMA"],
                    pre_train_cluster=True,
                    range_Test=fold_idx
                )
            except:
                return None, None, None, None, None, None, None

            y_train = labels_hat[indices_train]
            if len(np.unique(y_train)) < 2:
                sil = 0                 
                acc = 0
                stress_acc = 0
                calm_acc = 0
                return sil, acc, stress_acc, calm_acc, best_model_wts, centroids, best_model_wts


            # --- SILHOUETTE ---
            Z_full = Z_train.detach().cpu().numpy()
            sil = silhouette_score(Z_full, y_train)

            # --- DOWNSAMPLED TRAIN LABELS ---
            tl = pd.DataFrame([
                np.round(time[[k[1] for k in train_Data]]).astype(int),
                np.array(labels_hat[indices_train])
            ]).T

            tl_un = tl.groupby([0]).agg(lambda x: x.value_counts().index[0])
            labels_hat_down = tl_un[1].values
            good_idx = list(tl_un.index)
            labels_true_train = labels_true[good_idx]

            # --- GENERAL ACCURACY (KEPT) ---
            acc = np.mean(labels_hat_down == labels_true_train)

            # --- 🔴 STRESS ACCURACY (NEW, BUT ONLY USED FOR OPTIMIZATION) ---
            stress_acc = stress_accuracy(
                labels_true_train,
                labels_hat_down,
                stress_label=dicio_labels_ids["D"]
            )

            calm_acc = stress_accuracy(
                labels_true_train,
                labels_hat_down,
                stress_label=dicio_labels_ids["C"]
            )


            return sil, acc, stress_acc, calm_acc, best_model_wts, centroids, best_model_wts


        # -------------------------------------------------------------------------
        # (5) OPTUNA OBJECTIVE
        # -------------------------------------------------------------------------
        def objective(trial):

            reset_seeds(1000 + trial.number)

            config = {
                "SEQ_LEN": 128,
                "BATCH_SIZE": 512,
                "EPOCHS_PRE": 1,
                "EPOCHS_TRAIN": 30,
                "EMBEDDING": trial.suggest_categorical("EMBEDDING", [16, 32, 48, 64]),
                "LR_PRE": trial.suggest_float("LR_PRE", 3e-5, 3e-3, log=True),
                "LR_TRAIN": trial.suggest_float("LR_TRAIN", 1e-5, 1e-3, log=True),
                "SIGMA": trial.suggest_categorical("SIGMA", [0.3, 0.5, 1, 2, 3]),
                "GAMMA": trial.suggest_float("GAMMA", 0.03, 1, log=True),
            }

            sil_list, acc_list, stress_acc_list, calm_acc_list = [], [], [], []
            wts_per_fold, cents_per_fold = [], []

            for fold_idx in range(kfolds):

                out = train_one_fold(config, fold_idx)
                if out[0] is None:
                    sil_list.append(np.nan)
                    acc_list.append(np.nan)
                    stress_acc_list.append(np.nan)
                    wts_per_fold.append(None)
                    cents_per_fold.append(None)
                    calm_acc_list.append(np.nan)
                    continue

                sil, acc, stress_acc, calm_acc, wts, cents, wts_sd = out


                calm_acc_list.append(calm_acc)
                sil_list.append(sil)
                acc_list.append(acc)
                stress_acc_list.append(stress_acc)
                wts_per_fold.append(wts)
                cents_per_fold.append(cents)

            valid_sil     = [s for s in sil_list if not np.isnan(s)]
            valid_stress  = [s for s in stress_acc_list if not np.isnan(s)]
            valid_calm    = [c for c in calm_acc_list if not np.isnan(c)]

            # --- basic validity checks ---
            if len(valid_sil) < max(2, int(0.6 * kfolds)):
                return -999
            if len(valid_stress) == 0 or len(valid_calm) == 0:
                return -999

            mean_sil        = np.mean(valid_sil)
            mean_acc        = np.nanmean(acc_list)
            mean_stress_acc = np.mean(valid_stress)
            mean_calm_acc   = np.mean(valid_calm)

            std_stress_acc = np.std(valid_stress)
            std_calm_acc   = np.std(valid_calm)
            std_sil        = np.std(valid_sil)

            if np.isnan(mean_sil) or mean_sil < 0.10:
                return -999

            mean_stress_acc = np.mean(valid_stress)
            mean_calm_acc   = np.mean(valid_calm)

            balanced_acc = 0.5 * (mean_stress_acc + mean_calm_acc)

            final_score = (
                (mean_sil ** 2) * balanced_acc
                - 0.2 * np.nanstd(valid_stress)
                - 0.2 * np.nanstd(valid_calm)
            )

            # --- SAVE EVERYTHING AS BEFORE ---
            trial.set_user_attr("mean_sil", mean_sil)
            trial.set_user_attr("mean_acc", mean_acc)
            trial.set_user_attr("mean_stress_acc", mean_stress_acc)
            trial.set_user_attr("std_stress_acc", std_stress_acc)
            trial.set_user_attr("all_wts", wts_per_fold)
            trial.set_user_attr("all_centroids", cents_per_fold)
            best_fold = np.nanargmax(stress_acc_list)
            trial.set_user_attr("best_fold", int(best_fold))

            return final_score


        # -------------------------------------------------------------------------
        # (6) RUN OPTUNA
        # -------------------------------------------------------------------------
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=ntrials)

        best_trial = study.best_trial

        ua = best_trial.user_attrs
        results_per_threshold[ema_threshold] = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": best_trial,
            "all_folds_wts": ua.get("all_wts", None),
            "all_folds_centroids": ua.get("all_centroids", None),
            "best_fold": ua.get("best_fold", None),
            "study": study
        }

    # %%
    # =============================================================================
    #  BUILD GLOBAL LEADERBOARD (STRESS-OPTIMIZED, CORRECT)
    # =============================================================================

    rows = []

    for thr, res in results_per_threshold.items():

        study = res["study"]

        for t in study.trials:
            if t.value is None or t.value == -999:
                continue

            row = {
                "threshold": thr,
                "score": t.value,                          # optimization objective
                "mean_stress_acc": t.user_attrs.get("mean_stress_acc", np.nan),
                "mean_sil": t.user_attrs.get("mean_sil", np.nan),
                "std_stress_acc": t.user_attrs.get("std_stress_acc", np.nan),
                "mean_acc": t.user_attrs.get("mean_acc", np.nan),
                "best_fold": t.user_attrs.get("best_fold", np.nan),
            }

            # add hyperparameters
            for k, v in t.params.items():
                row[k] = v

            rows.append(row)

    all_results = pd.DataFrame(rows)

    # Sort by optimization objective
    all_results = all_results.sort_values("score", ascending=False).reset_index(drop=True)

    print("\n============== GLOBAL LEADERBOARD (Stress-Optimized) ==============\n")
    '''display(all_results.head(10))'''


    # %%
    '''plt.figure(figsize=(8,5))
    (
        all_results.groupby("threshold")["score"].mean()
                .plot(marker="o", linewidth=2)
    )
    plt.title("Mean Score vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Mean Score")
    plt.grid(True)
    plt.show()'''

    # %%
    threshold_rank = all_results.groupby("threshold")["score"].max().sort_values(ascending=False)

    print("\n====================== THRESHOLD RANKING ======================\n")
    print(threshold_rank)

    # %%
    dist_data = []

    for thr in results_per_threshold.keys():
        df = Activities_ema_base.copy()
        df["stress_axis"] = np.where(df["Calmness"] >= thr, "No", "Yes")

        dist_data.append({
            "threshold": thr,
            "Stress_count": (df["stress_axis"]=="Yes").sum(),
            "Calm_count":   (df["stress_axis"]=="No").sum()
        })

    dist_df = pd.DataFrame(dist_data).sort_values("threshold")

    print("\n====================== DISTRIBUTION PER THRESHOLD ======================\n")
    '''display(dist_df)'''

    # %%
    '''matplotlib.use('module://matplotlib_inline.backend_inline') # Show GUI'''

    # %%
    matplotlib.use('Agg')  # backend with no GUI

    # %%
    #####################################################################
    # ========== BUILD EMA INTERVALS, LABEL TIMELINE & COMPRESSION ======
    #####################################################################

    print("Building EMA timeline start/end indices...")

    # Create local copy
    Activities_ema_base = Activities_ema_base.copy()

    # ---- Compute start indices ----
    start_idx = pd.merge_asof(
        Activities_ema_base.sort_values("StartTime"),
        bvp_valid[["DateTime", "Timestamp_x"]].sort_values("DateTime"),
        left_on="StartTime", right_on="DateTime"
    )["Timestamp_x"]

    # ---- Compute end indices ----
    end_idx = pd.merge_asof(
        Activities_ema_base.sort_values("EndTime"),
        bvp_valid[["DateTime", "Timestamp_x"]].sort_values("DateTime"),
        left_on="EndTime", right_on="DateTime"
    )["Timestamp_x"]

    Activities_ema_base["start"] = np.round(start_idx - t0_bvp).astype("Int64")
    Activities_ema_base["end"]   = np.round(end_idx   - t0_bvp).astype("Int64")

    # Remove rows where mapping failed
    Activities_ema_base = Activities_ema_base.dropna(subset=["start", "end"]).reset_index(drop=True)

    # ---- Build true calmness timeline (per second) ----
    calmness_timeline = np.zeros_like(time_labels, dtype=float)

    for i, row in Activities_ema_base.iterrows():
        s = int(row["start"])
        e = int(row["end"])
        calmness_timeline[s:e] = row["Calmness"]

    print("Done: EMA start/end indices + calmness timeline created.")



    #####################################################################
    # =============== FUNCTION TO BUILD SCORE_S + LABELS_TRUE ===========
    #####################################################################

    def build_score_and_uinit(thr):
        """
        Για δοσμένο calmness threshold φτιάχνει:
        - score_s with start/end
        - labels_true timeline (0,1,2)
        - u_init (membership tensor for DeepCMeans)
        """

        Activities_ema = Activities_ema_base.copy()
        Activities_ema["stress_axis"] = np.where(
            Activities_ema["Calmness"] >= thr, "No", "Yes"
        )
        Activities_ema["phase"] = np.where(
            Activities_ema["stress_axis"] == "Yes", "D", "C"
        )

        score_s = Activities_ema.copy()

        # Keep only intervals inside BVP timestamp range
        score_s = score_s.loc[
            score_s.StartTime >= df_all_possibilities_bvp.DateTime.min()
        ].reset_index(drop=True)

        # Sort for merge_asof alignment
        score_s_sorted_start = score_s.sort_values("StartTime")
        score_s_sorted_end   = score_s.sort_values("EndTime")

        start_idx = pd.merge_asof(
            score_s_sorted_start,
            bvp_valid.sort_values("DateTime"),
            left_on="StartTime",
            right_on="DateTime"
        )["Timestamp_x"].values

        end_idx = pd.merge_asof(
            score_s_sorted_end,
            bvp_valid.sort_values("DateTime"),
            left_on="EndTime",
            right_on="DateTime"
        )["Timestamp_x"].values

        # Assign start/end back in original score_s order
        score_s = score_s.reset_index(drop=True)
        score_s["start"] = np.round(start_idx - t0_bvp).astype(int)
        score_s["end"]   = np.round(end_idx   - t0_bvp).astype(int)

        # Build labels_true (0 outside EMA, 1 stress, 2 calm)
        labels_true = np.zeros_like(time_labels, dtype=int)
        for i in range(len(score_s)):
            s = score_s.loc[i, "start"]
            e = score_s.loc[i, "end"]
            phase = score_s.loc[i, "phase"]
            labels_true[s:e] = dicio_labels_ids[phase]

        # Compute initial membership
        u_init = init_membership(
            score_s=score_s,
            time=time,
            nb_classes=NB_CLASSES,
            time_shift=time_shift,
            one_hot=True
        )

        return score_s, labels_true, u_init



    #####################################################################
    # ================== COMPRESS TIMELINE TO EMA-ONLY =================
    #####################################################################

    print("\nBuilding EMA-only compressed timeline...")

    # ── 1. Mask where inside EMA
    # (labels_true will be overwritten inside test loop, so build mask later)
    # But create placeholders here:
    # We will rebuild compressed mapping INSIDE each threshold loop

    print("Compression will be re-built inside test loop for each threshold.")


    # %%
    # ============================================================
    # IMPORTS
    # ============================================================

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import gc

    from sklearn.metrics import accuracy_score, silhouette_score, confusion_matrix


    # ============================================================
    # CONSTANTS
    # ============================================================

    STRESS_IDX = dicio_labels_ids["D"]   # = 1
    CALM_IDX   = dicio_labels_ids["C"]   # = 2 (ή 0 αν έτσι το έχεις)


    # ============================================================
    # UTILS
    # ============================================================

    def get_fallback_model(all_wts, all_centroids, fold_idx):
        """
        Επιστρέφει (weights, centroids) με fallback:
        1) ίδιο fold
        2) προηγούμενα folds
        3) επόμενα folds
        Αν δεν βρεθεί τίποτα, επιστρέφει (None, None)
        """

        # --- 1. ίδιο fold ---
        if all_wts[fold_idx] is not None and all_centroids[fold_idx] is not None:
            return all_wts[fold_idx], all_centroids[fold_idx]

        # --- 2. προς τα πίσω ---
        for j in range(fold_idx - 1, -1, -1):
            if all_wts[j] is not None and all_centroids[j] is not None:
                print(f"[Fold {fold_idx}] fallback → previous fold {j}")
                return all_wts[j], all_centroids[j]

        # --- 3. προς τα μπροστά ---
        for j in range(fold_idx + 1, len(all_wts)):
            if all_wts[j] is not None and all_centroids[j] is not None:
                print(f"[Fold {fold_idx}] fallback → next fold {j}")
                return all_wts[j], all_centroids[j]

        # --- 4. τίποτα ---
        print(f"[Fold {fold_idx}] ❌ no valid fallback model found")
        return None, None


    def stress_accuracy(y_true, y_pred, stress_label):
        mask = (y_true == stress_label)
        if np.sum(mask) == 0:
            return np.nan
        return accuracy_score(y_true[mask], y_pred[mask])


    def plot_ema_timeline(prob_stress, ema_idx, labels_true_fold, title):
        """
        - prob_stress: P(stress) ανά EMA index
        - axvspan: EMA blocks (orange=stress, blue=calm)
        - black line: model prediction P(stress)
        """

        prob_stress = np.asarray(prob_stress)
        ema_idx = np.asarray(ema_idx)
        labels_true_fold = np.asarray(labels_true_fold)

        order = np.argsort(ema_idx)
        prob_stress = prob_stress[order]
        ema_idx = ema_idx[order]
        labels_true_fold = labels_true_fold[order]

        plt.figure(figsize=(18, 5))

        # ---- EMA background ----
        for i in range(len(ema_idx)):
            if labels_true_fold[i] == STRESS_IDX:
                plt.axvspan(
                    ema_idx[i] - 0.5,
                    ema_idx[i] + 0.5,
                    color="orange",
                    alpha=0.25,
                    label="Stress EMA" if i == 0 else None
                )
            else:
                plt.axvspan(
                    ema_idx[i] - 0.5,
                    ema_idx[i] + 0.5,
                    color="steelblue",
                    alpha=0.25,
                    label="Calm EMA" if i == 0 else None
                )

        # ---- Model prediction ----
        plt.plot(
            ema_idx,
            prob_stress,
            color="black",
            linewidth=2,
            label="Model prediction: P(stress)"
        )

        plt.xlabel("EMA index")
        plt.ylabel("Stress probability")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


    # ============================================================
    # FINAL TEST BLOCK
    # ============================================================

    test_results = {}

    for thr in thresholds:

        print("\n===================================================")
        print(f" TESTING THRESHOLD = {thr}")
        print("===================================================\n")

        best_params = results_per_threshold[thr]["best_params"]
        best_trial  = results_per_threshold[thr]["best_trial"]

        all_wts       = best_trial.user_attrs["all_wts"]
        all_centroids = best_trial.user_attrs["all_centroids"]

        score_s, labels_true, u_init_thr = build_score_and_uinit(thr)

        # EMA compression
        ema_mask = labels_true > 0
        full_to_ema = np.full_like(labels_true, -1)
        full_to_ema[ema_mask] = np.arange(np.sum(ema_mask))

        fold_results = []

        for fold in range(kfolds):

            gc.collect()
            torch.cuda.empty_cache()

            print(f"\n→ Fold {fold+1}/{kfolds}")
            print("Hyperparameters:", best_params)

            if all_wts[fold] is None:
                print("⚠ Missing weights")
            if all_centroids[fold] is None:
                print("⚠ Missing centroids")

            best_model_wts, centroids = get_fallback_model(
                all_wts,
                all_centroids,
                fold
            )
            dfcm = DeepCMeams(signal, time)

            labels_hat, U, _, _, _, _, _, _, \
            indices_train, indices_test, Z, train_Data, test_Data = dfcm.run(
                nb_classes=NB_CLASSES,
                seq_len=128,
                u_init=u_init_thr,
                embedding=best_params["EMBEDDING"],
                lr_pretrain=best_params["LR_PRE"],
                lr_train=best_params["LR_TRAIN"],
                sigma=best_params["SIGMA"],
                gamma=best_params["GAMMA"],
                batch_size=512,
                epochs_pre=0,
                epochs_train=0,
                pre_train_cluster=False,
                best_model_wts=best_model_wts,
                centroids=centroids,
                range_Test=fold
            )

            # ------------------------------------------------
            # Downsample to timeline
            # ------------------------------------------------
            match_test = np.array([x[1] for x in test_Data])
            time_bins = np.round(time[match_test]).astype(int)

            tl = pd.DataFrame({
                "time_bin": time_bins,
                "label": labels_hat[indices_test]
            })

            tl_un = tl.groupby("time_bin")["label"].agg(
                lambda x: x.value_counts().index[0]
            )

            full_idx = tl_un.index.values
            labels_hat_down = tl_un.values

            ema_idx = full_to_ema[full_idx]
            valid = ema_idx >= 0

            ema_idx = ema_idx[valid]
            labels_hat_down = labels_hat_down[valid]
            labels_true_fold = labels_true[full_idx][valid]

            # ---- Fold-level counts / metadata ----
            n_train = len(indices_train)
            n_test  = len(indices_test)

            n_ema_total  = len(labels_true_fold)
            n_ema_stress = np.sum(labels_true_fold == STRESS_IDX)
            n_ema_calm   = np.sum(labels_true_fold == CALM_IDX)

            # ---- Explicit EMA labels (for future pickle) ----
            labels_true_ema = labels_true_fold
            labels_pred_ema = labels_hat_down

            mask_valid = labels_true_fold > 0

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------
            acc = accuracy_score(
                labels_true_fold[mask_valid],
                labels_hat_down[mask_valid]
            )

            acc_stress = stress_accuracy(
                labels_true_fold[mask_valid],
                labels_hat_down[mask_valid],
                STRESS_IDX
            )

            # ------------------------------------------------
            # Silhouette (TEST latent space)
            # ------------------------------------------------
            sil_test = np.nan

            Z_test = Z.detach().cpu().numpy()
            labels_test = labels_hat[indices_test]

            # κρατάμε μόνο Stress vs Calm
            mask_sc = np.isin(labels_test, [STRESS_IDX, CALM_IDX])

            if np.sum(mask_sc) > 10 and len(np.unique(labels_test[mask_sc])) >= 2:
                sil_test = silhouette_score(
                    Z_test[mask_sc],
                    labels_test[mask_sc]
                )

            
            print(f"Accuracy        : {acc:.4f}")
            print(f"Stress Accuracy : {acc_stress:.4f}")
            print(f"Silhouette      : {sil_test:.4f}")

            # ------------------------------------------------
            # Confusion Matrix (RAW)
            # ------------------------------------------------
            '''cm_raw = confusion_matrix(
                labels_true_fold[mask_valid],
                labels_hat_down[mask_valid]
            )

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Thr {thr} — Fold {fold} — Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()'''
            
            # ------------------------------------------------
            # Plot 1: U_test probabilities vs time (per class)
            # ------------------------------------------------

            match_indices_test_outside = [k[1] for k in test_Data]

            # ---- make U numpy-safe ----
            if hasattr(U, "detach"):
                U_np = U.detach().cpu().numpy()
            else:
                U_np = U

            # ---- define U_test EXPLICITLY ----
            U_test = U_np[indices_test]

            '''plt.figure(figsize=(20, 6))
            for i in range(NB_CLASSES):
                plt.plot(
                    time[match_indices_test_outside],
                    U_test[:, i],
                    label=f"Class {i}"
                )

            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title(f"U_test probabilities — Thr {thr}, Fold {fold}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()'''

            # ------------------------------------------------
            # PROBABILITY = P(STRESS) ONLY
            # ------------------------------------------------
            # Continuous probability Manos
            #prob_stress_test = U_np[:, STRESS_IDX][indices_test]
            stress_col = STRESS_IDX - 1
            prob_stress_test = U_np[indices_test, stress_col]


            df_prob = pd.DataFrame({
                "time_bin": time_bins,
                "prob_stress": prob_stress_test
            })

            prob_stress_down = (
                df_prob
                .groupby("time_bin")["prob_stress"]
                .mean()
                .loc[full_idx][valid]
                .values
            )

            # ------------------------------------------------
            # Plot EMA timeline
            # ------------------------------------------------
            '''plot_ema_timeline(
                prob_stress_down,
                ema_idx,
                labels_true_fold,
                title=f"EMA Timeline — Thr {thr}, Fold {fold}"
            )'''

            fold_results.append({
                "fold": fold,

                # ---- metrics ----
                "acc": acc,
                "acc_stress": acc_stress,
                "sil": sil_test,

                # ===== TIMELINE KEYS (NEW – ΑΠΑΡΑΙΤΗΤΑ) =====
                "time_sec": full_idx[valid],   # seconds since t0_bvp
                "ema_idx": ema_idx,
                "t0_bvp": t0_bvp,
                "t0_datetime": bvp_valid["DateTime"].min(),
                
                # ---- counts / bookkeeping ----
                "n_train": n_train,
                "n_test": n_test,
                "n_ema_total": n_ema_total,
                "n_ema_stress": n_ema_stress,
                "n_ema_calm": n_ema_calm,

                # ---- EMA-level labels ----
                "labels_true_ema": labels_true_fold,
                "labels_pred_ema": labels_hat_down,
                "prob_stress_ema": prob_stress_down,
                
                # ---- model outputs (test only, safe) ----
                "U_test": U_np[indices_test],
                
                # ---- clustering ----
                "centroids": all_centroids[fold],

                # ---- model weights ----
                "ae_state_dict": best_model_wts,
            })



        # ------------------------------------------------
        # Aggregate results
        # ------------------------------------------------
        test_results[thr] = {
            "threshold": thr,
            "mean_acc": np.nanmean([f["acc"] for f in fold_results]),
            "mean_acc_stress": np.nanmean([f["acc_stress"] for f in fold_results]),
            "mean_sil": np.nanmean([f["sil"] for f in fold_results]),
            "std_sil": np.nanstd([f["sil"] for f in fold_results]),
            "fold_results": fold_results
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    summary_df = pd.DataFrame([
        {
            "threshold": thr,
            "mean_acc": test_results[thr]["mean_acc"],
            "mean_acc_stress": test_results[thr]["mean_acc_stress"],
            "mean_sil": test_results[thr]["mean_sil"],
            "std_sil": test_results[thr]["std_sil"],
        }
        for thr in test_results
    ]).sort_values("threshold")


    print("\n==================== FINAL SUMMARY ====================")
    print(summary_df)

    '''plt.figure(figsize=(12, 6))
    plt.plot(summary_df["threshold"], summary_df["mean_acc_stress"], "-o",
            label="Stress Accuracy", linewidth=3)
    plt.plot(summary_df["threshold"], summary_df["mean_acc"], "-o",
            label="Overall Accuracy", linewidth=2)

    plt.xlabel("Threshold")
    plt.ylabel("Metric value")
    plt.title("Test Performance Across Thresholds")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()'''

    # %%
    import pickle
    import os

    # -------------------------------------------------------
    # 0. Make sure directory exists
    # -------------------------------------------------------
    save_path = (
        f"/run/user/{personal_ID}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/"
        f"Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/E4/completed/{subj_name}/Data/"
    )
    os.makedirs(save_path, exist_ok=True)

    print(f"SAVE DIRECTORY: {save_path}")

    # -------------------------------------------------------
    # 1. Save results per threshold
    # -------------------------------------------------------
    for thr, thr_data in test_results.items():

        thr_str = str(thr).replace(".", "_")

        # --------------------------
        # (A) Save full threshold dict (RECOMMENDED)
        # --------------------------
        full_file = os.path.join(
            save_path,
            f"test_results_thr{thr_str}.pkl"
        )
        with open(full_file, "wb") as f:
            pickle.dump(thr_data, f)

        print(f"[OK] Saved pickles for threshold {thr} → {thr_str}")


    # %%
    # ============================================================
    # SANITY CHECK: PRINT test_results STRUCTURE
    # ============================================================

    print("\n================= TEST_RESULTS STRUCTURE =================\n")

    for thr, thr_data in test_results.items():

        print(f"\n--- Threshold: {thr} ---")

        # ---- Top-level keys ----
        print("Top-level keys:")
        for k in thr_data.keys():
            print(f"  - {k}")

        # ---- Mean metrics ----
        print("\nMean metrics:")
        print(f"  mean_acc        : {thr_data.get('mean_acc')}")
        print(f"  mean_acc_stress : {thr_data.get('mean_acc_stress')}")
        print(f"  mean_sil        : {thr_data.get('mean_sil')}")

        # ---- Fold results ----
        fold_results = thr_data.get("fold_results", [])
        print(f"\nNumber of folds stored: {len(fold_results)}")

        if len(fold_results) == 0:
            print("⚠ WARNING: fold_results is empty!")
            continue

        # ---- Inspect first fold only (avoid spam) ----
        fr0 = fold_results[0]
        print("\nFold[0] keys:")
        for k in fr0.keys():
            v = fr0[k]
            if hasattr(v, "__len__") and not isinstance(v, str):
                try:
                    print(f"  - {k}: type={type(v).__name__}, len={len(v)}")
                except:
                    print(f"  - {k}: type={type(v).__name__}")
            else:
                print(f"  - {k}: type={type(v).__name__}")

        # ---- Extra consistency checks ----
        print("\nQuick consistency checks (Fold 0):")
        if "labels_true_ema" in fr0 and "labels_pred_ema" in fr0:
            print(
                "  labels_true_ema vs labels_pred_ema:",
                len(fr0["labels_true_ema"]),
                len(fr0["labels_pred_ema"])
            )

        if "prob_stress_ema" in fr0:
            print(
                "  prob_stress_ema length:",
                len(fr0["prob_stress_ema"])
            )

        if "U_test" in fr0:
            print(
                "  U_test shape:",
                fr0["U_test"].shape
            )

        if "centroids" in fr0 and fr0["centroids"] is not None:
            if isinstance(fr0["centroids"], torch.Tensor):
                print(
                    "  centroids shape:",
                    fr0["centroids"].detach().cpu().numpy().shape
                )
            else:
                print(
                    "  centroids shape:",
                    np.array(fr0["centroids"]).shape
                )
        else:
            print("  centroids: None (train failed in this fold)")


    print("\n================= END STRUCTURE CHECK =================\n")

    # %%
    import numpy as np
    import pandas as pd
    import pickle
    import json
    import os
    import gc
    import torch
    from sklearn.metrics import accuracy_score, silhouette_score

    # --------------------------
    # MARTA PARAMETERS
    # --------------------------
    MARTA_PARAMS = {
        "seq_len": 128,
        "epochs_pre": 1,
        "epochs_train": 30,
        "batch_size": 512,
        "embedding": 30,
        "lr_pretrain": 1e-6,
        "lr_train": 5e-5,
        "sigma": 2,
        "gamma": 0.1
    }

    def stress_accuracy(y_true, y_pred, stress_label):
        mask = (y_true == stress_label)
        if np.sum(mask) == 0:
            return np.nan
        return accuracy_score(y_true[mask], y_pred[mask])

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)

    # ======================================================
    # TRAIN FOLD
    # ======================================================
    def run_marta_train_fold(signal, time, u_init, NB_CLASSES, fold, params):
        
        dfcm = DeepCMeams(signal, time)

        try:
            out = dfcm.run(
                nb_classes=NB_CLASSES,
                seq_len=params["seq_len"],
                u_init=u_init,
                epochs_pre=params["epochs_pre"],
                epochs_train=params["epochs_train"],
                batch_size=params["batch_size"],
                embedding=params["embedding"],
                lr_pretrain=params["lr_pretrain"],
                lr_train=params["lr_train"],
                sigma=params["sigma"],
                gamma=params["gamma"],
                pre_train_cluster=True,
                range_Test=fold
            )

        except Exception as e:
            print(f"[TRAIN FAIL] Fold {fold}: {e}")
            return {
                "train_failed": True
            }
        
        (
                labels_hat, U, time_hat, best_model_wts, losses,
                losses_AE, losses_silh, centroids,
                indices_train, indices_test,
                Z_train, train_Data, test_Data
            ) = out

        return {
            "labels_hat": labels_hat,
            "U": U,
            "time_hat": time_hat,
            "best_model_wts": best_model_wts,
            "centroids": centroids,
            "indices_train": indices_train,
            "indices_test": indices_test,
            "Z_train": Z_train,
            "train_Data": train_Data,
            "test_Data": test_Data
        }


    # ======================================================
    # TEST FOLD (using weights from train)
    # ======================================================
    def run_marta_test_fold(signal, time, u_init, NB_CLASSES, fold, params, best_model_wts, centroids):

        dfcm = DeepCMeams(signal, time)

        out = dfcm.run(
            nb_classes=NB_CLASSES,
            seq_len=params["seq_len"],
            u_init=u_init,
            epochs_pre=0,
            epochs_train=0,
            batch_size=params["batch_size"],
            embedding=params["embedding"],
            lr_pretrain=params["lr_pretrain"],
            lr_train=params["lr_train"],
            sigma=params["sigma"],
            gamma=params["gamma"],
            pre_train_cluster=False,
            best_model_wts=best_model_wts,
            centroids=centroids,
            range_Test=fold
        )

        (
            labels_hat, U, time_hat, _,
            losses, losses_AE, losses_silh, _,
            indices_train, indices_test,
            Z_test, train_Data, test_Data
        ) = out

        return {
            "labels_hat": labels_hat,
            "U": U,
            "time_hat": time_hat,
            "indices_train": indices_train,
            "indices_test": indices_test,
            "Z_test": Z_test,
            "train_Data": train_Data,
            "test_Data": test_Data
        }


    # ======================================================
    # MAIN FUNCTION: SAME OUTPUT FORMAT AS OPTUNA TEST BLOCK
    # ======================================================
    def run_marta_for_all_thresholds(thresholds, signal, time, NB_CLASSES,
                                    build_score_and_uinit):

        marta_test_results = {}   # same structure as before

        for thr in thresholds:

            print("\n================================")
            print(f"   PROCESSING THRESHOLD {thr}")
            print("================================")

            # Rebuild labels + u_init for this threshold
            score_s, labels_true_thr, u_init_thr = build_score_and_uinit(thr)

            # For metrics summary per threshold
            fold_results = []   # REQUIRED – Optuna-compatible

            for fold in range(10):

                gc.collect()
                torch.cuda.empty_cache()

                print(f"\n--- TRAIN FOLD {fold} ---")
                tr = run_marta_train_fold(signal, time, u_init_thr, NB_CLASSES, fold, MARTA_PARAMS)

                # 🔴 EARLY EXIT ON TRAIN FAILURE
                if tr.get("train_failed", False):

                    fold_results.append({
                        "fold": fold,
                        "train_failed": True,

                        # metrics
                        "acc": np.nan,
                        "acc_stress": np.nan,
                        "sil": np.nan,

                        # counts
                        "n_train": 0,
                        "n_test": 0,
                        "n_ema_total": 0,
                        "n_ema_stress": 0,
                        "n_ema_calm": 0,

                        # EMA outputs
                        "labels_true_ema": np.array([]),
                        "labels_pred_ema": np.array([]),
                        "prob_stress_ema": np.array([]),
                        "ema_idx": np.array([]),

                        # model outputs
                        "U_test": np.empty((0, NB_CLASSES)),
                        "centroids": np.empty((0,)),
                    })
                    print(f"[SKIPPED TEST] Fold {fold} due to train failure")
                    continue   # ⬅️ THIS IS THE KEY LINE

                print(f"--- TEST FOLD {fold} ---")
                try:
                    te = run_marta_test_fold(signal, time, u_init_thr, NB_CLASSES, fold, MARTA_PARAMS,
                                            best_model_wts=tr["best_model_wts"],
                                            centroids=tr["centroids"])
                except Exception as e:
                    print(f"[TEST FAIL] Fold {fold}: {e}")

                    fold_results.append({
                        "fold": fold,
                        "test_failed": True,

                        # ---- metrics ----
                        "acc": np.nan,
                        "acc_stress": np.nan,
                        "sil": np.nan,

                        # ---- counts ----
                        "n_train": len(tr["indices_train"]),
                        "n_test": 0,
                        "n_ema_total": 0,
                        "n_ema_stress": 0,
                        "n_ema_calm": 0,

                        # ---- EMA outputs ----
                        "labels_true_ema": np.array([]),
                        "labels_pred_ema": np.array([]),
                        "prob_stress_ema": np.array([]),
                        "ema_idx": np.array([]),

                        # ---- model outputs ----
                        "U_test": np.empty((0, NB_CLASSES)),
                        "centroids": np.empty((0,)),
                    })
                    continue
                
                # ----------------------
                # DOWNSAMPLING TRAIN
                # ----------------------
                match_train = np.array([x[1] for x in tr["train_Data"]])
                df_tr = pd.DataFrame({
                    "tb": np.round(time[match_train]).astype(int),
                    "label": tr["labels_hat"][tr["indices_train"]]
                })
                down_tr = df_tr.groupby("tb")["label"].agg(lambda x: x.value_counts().index[0])
                labels_hat_down_train = down_tr.values
                labels_true_train = labels_true_thr[down_tr.index]

                # ----------------------
                # DOWNSAMPLING TEST
                # ----------------------
                match_test = np.array([x[1] for x in te["test_Data"]])
                df_te = pd.DataFrame({
                    "tb": np.round(time[match_test]).astype(int),
                    "label": te["labels_hat"][te["indices_test"]]
                })
                down_te = df_te.groupby("tb")["label"].agg(lambda x: x.value_counts().index[0])
                labels_hat_down_test = down_te.values
                labels_true_test = labels_true_thr[down_te.index]

                labels_true_ema = labels_true_test
                labels_pred_ema = labels_hat_down_test
                ema_idx = down_te.index.values

                # P(stress)
                U_np = to_numpy(te["U"])
                prob_stress_ema = (
                    pd.DataFrame({
                        "tb": np.round(time[match_test]).astype(int),
                        "p": U_np[te["indices_test"], dicio_labels_ids["D"]]
                    })
                    .groupby("tb")["p"]
                    .mean()
                    .loc[ema_idx]
                    .values
                )


                # ----------------------
                # METRICS
                # ----------------------
            
                mask_valid = labels_true_test > 0

                # ----------------------
                # ACCURACY
                # ----------------------
                acc = accuracy_score(
                    labels_true_test[mask_valid],
                    labels_hat_down_test[mask_valid]
                )

                # ----------------------
                # STRESS ACCURACY
                # ----------------------
                acc_stress = stress_accuracy(
                    labels_true_test[mask_valid],
                    labels_hat_down_test[mask_valid],
                    stress_label=dicio_labels_ids["D"]
                )

                # ----------------------
                # SILHOUETTE (TEST, INDICATIVE)
                # ----------------------
                sil = np.nan
                Z_test = te["Z_test"].detach().cpu().numpy()
                labels_test = te["labels_hat"][te["indices_test"]]

                mask_sc = np.isin(labels_test, [dicio_labels_ids["D"], dicio_labels_ids["C"]])

                if np.sum(mask_sc) > 10 and len(np.unique(labels_test[mask_sc])) >= 2:
                    sil = silhouette_score(Z_test[mask_sc], labels_test[mask_sc])

                print(f"Accuracy        : {acc:.4f}")
                print(f"Stress Accuracy : {acc_stress:.4f}")
                print(f"Silhouette      : {sil:.4f}")

                fold_results.append({
                    "fold": fold,

                    # ---- metrics ----
                    "acc": acc,
                    "acc_stress": acc_stress,
                    "sil": sil,

                    # ===== TIMELINE KEYS (NEW – ΑΠΑΡΑΙΤΗΤΑ) =====
                    "time_sec": full_idx[valid],   # seconds since t0_bvp
                    "ema_idx": ema_idx,
                    "t0_bvp": t0_bvp,
                    "t0_datetime": bvp_valid["DateTime"].min(),

                    # ---- counts ----
                    "n_train": len(tr["indices_train"]),
                    "n_test": len(te["indices_test"]),
                    "n_ema_total": len(labels_true_ema),
                    "n_ema_stress": np.sum(labels_true_ema == dicio_labels_ids["D"]),
                    "n_ema_calm": np.sum(labels_true_ema == dicio_labels_ids["C"]),

                    # ---- EMA outputs ----
                    "labels_true_ema": labels_true_ema,
                    "labels_pred_ema": labels_pred_ema,
                    "prob_stress_ema": prob_stress_ema,
                    
                    # ---- model outputs ----
                    "U_test": to_numpy(U_np[te["indices_test"]]),
                    "centroids": to_numpy(tr["centroids"]),

                    # ---- model weights ----
                    "ae_state_dict": tr["best_model_wts"],
                })

            marta_test_results[thr] = {
                "threshold": thr,
                "mean_acc": np.nanmean([f["acc"] for f in fold_results]),
                "mean_acc_stress": np.nanmean([f["acc_stress"] for f in fold_results]),
                "mean_sil": np.nanmean([f["sil"] for f in fold_results]),
                "std_sil": np.nanstd([f["sil"] for f in fold_results]),
                "fold_results": fold_results
            }

        return marta_test_results

    # %%
    results_all = run_marta_for_all_thresholds(
        thresholds=thresholds,
        signal=signal,
        time=time,
        NB_CLASSES=NB_CLASSES,
        build_score_and_uinit=build_score_and_uinit,
    )

    summary_df = pd.DataFrame([
        {
            "threshold": thr,
            "mean_acc": res["mean_acc"],
            "mean_acc_stress": res["mean_acc_stress"],
            "mean_sil": res["mean_sil"],
            "std_sil": res["std_sil"],
        }
        for thr, res in results_all.items()
    ]).sort_values("threshold")

    print("\n==================== MARTA TEST SUMMARY ====================")
    print(summary_df)


    # %%
    import pickle
    import os

    # -------------------------------------------------------
    # 0. Make sure directory exists
    # -------------------------------------------------------
    save_path = (
        f"/run/user/{personal_ID}/gvfs/smb-share:server=ait-pdfs.win.dtu.dk,share=department/Man/"
        f"Public/4233-81647-eMOTIONAL-Cities/5 Data/ECDTU/E4/completed/{subj_name}/Data/"
    )
    os.makedirs(save_path, exist_ok=True)

    print(f"SAVE DIRECTORY: {save_path}")

    # -------------------------------------------------------
    # 1. Save MARTA results per threshold
    # -------------------------------------------------------
    for thr, thr_data in results_all.items():

        thr_str = str(thr).replace(".", "_")

        # --------------------------
        # (A) Save full threshold dict (MARTA)
        # --------------------------
        full_file = os.path.join(
            save_path,
            f"marta_test_results_thr{thr_str}.pkl"
        )

        with open(full_file, "wb") as f:
            pickle.dump(thr_data, f)

        print(f"[OK] Saved MARTA pickles for threshold {thr} → {thr_str}")


    # %%
    print("\n================= MARTA TEST_RESULTS STRUCTURE =================\n")

    for thr, thr_data in results_all.items():

        print(f"\n--- Threshold: {thr} ---")

        # ---- Top-level keys ----
        print("Top-level keys:")
        for k in thr_data.keys():
            print(f"  - {k}")

        # ---- Mean metrics ----
        print("\nMean metrics:")
        print(f"  mean_acc        : {thr_data.get('mean_acc')}")
        print(f"  mean_acc_stress : {thr_data.get('mean_acc_stress')}")
        print(f"  mean_sil        : {thr_data.get('mean_sil')}")
        print(f"  std_sil         : {thr_data.get('std_sil')}")

        # ---- Fold results ----
        fold_results = thr_data.get("fold_results", [])
        print(f"\nNumber of folds stored: {len(fold_results)}")

        if len(fold_results) == 0:
            print("⚠ WARNING: fold_results is empty!")
            continue

        # ---- Inspect first fold only ----
        fr0 = fold_results[0]
        print("\nFold[0] keys:")
        for k in fr0.keys():
            v = fr0[k]
            if hasattr(v, "__len__") and not isinstance(v, str):
                try:
                    print(f"  - {k}: type={type(v).__name__}, len={len(v)}")
                except:
                    print(f"  - {k}: type={type(v).__name__}")
            else:
                print(f"  - {k}: type={type(v).__name__}")

        # ---- Extra consistency checks ----
        print("\nQuick consistency checks (Fold 0):")

        if "labels_true_ema" in fr0 and "labels_pred_ema" in fr0:
            print(
                "  labels_true_ema vs labels_pred_ema:",
                len(fr0["labels_true_ema"]),
                len(fr0["labels_pred_ema"])
            )

        if "prob_stress_ema" in fr0:
            print("  prob_stress_ema length:", len(fr0["prob_stress_ema"]))

        if "U_test" in fr0:
            print("  U_test shape:", fr0["U_test"].shape)

        if "centroids" in fr0 and fr0["centroids"] is not None:
            if isinstance(fr0["centroids"], torch.Tensor):
                print(
                    "  centroids shape:",
                    fr0["centroids"].detach().cpu().numpy().shape
                )
            else:
                print(
                    "  centroids shape:",
                    np.array(fr0["centroids"]).shape
                )
        else:
            print("  centroids: None (train failed in this fold)")

    print("\n================= END STRUCTURE CHECK =================\n")

    # %%

    print(f"\n✔ FINISHED RUN: {run_value}\n")


# change here the list to run different subjects. 
run_values = ['105', '473', '489', '136', '389', '139', 
              '303', '305', '271', '114', '384', 
              '238', '159', '172', '117', '123', '457', 
              '164', '313', '176', '151', '161', '188', '165', '217', 
              '255', '416', '154']
# note: problems 271, 303, 313, 231, 255, 238, 217
import traceback

failed_runs = []

for run in run_values:

    print(f"\n▶ START RUN: {run}")
    
    try:
        with Silent():
            run_full_pipeline(run)

    except Exception:
        print(f"❌ Run {run} failed")
        traceback.print_exc()

        with open("failed_runs.log", "a") as f:
            f.write(f"\n=== RUN {run} FAILED ===\n")
            traceback.print_exc(file=f)

        failed_runs.append(run)
        continue

print("\nFAILED RUNS:", failed_runs)