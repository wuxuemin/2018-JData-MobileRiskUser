import pandas as pd
import numpy as np
uid_train = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_train/uid_train.txt',sep='\t',header=None,names=('uid','label'))
voice_train = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_train/voice_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_train = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_train/sms_train.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_train = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_train/wa_train.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})
voice_test = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_Test-B/voice_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out'),dtype={'start_time':str,'end_time':str})
sms_test = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_Test-B/sms_test_b.txt',sep='\t',header=None,names=('uid','opp_num','opp_head','opp_len','start_time','in_out'),dtype={'start_time':str})
wa_test = pd.read_csv('/home/wxm/Downloads/data_mining/JDATA_Test-B/wa_test_b.txt',sep='\t',header=None,names=('uid','wa_name','visit_cnt','visit_dura','up_flow','down_flow','wa_type','date'),dtype={'date':str})

uid_test = pd.DataFrame({'uid':pd.unique(wa_test['uid'])})
uid_test.to_csv('/home/wxm/Downloads/data_mining/JDATA_Test-B/uid_test_b.txt',index=None)

voice = pd.concat([voice_train,voice_test],axis=0)
sms = pd.concat([sms_train,sms_test],axis=0)
wa = pd.concat([wa_train,wa_test],axis=0)

voice_opp_num = voice.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('voice_opp_num_').reset_index()
voice_opp_head=voice.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('voice_opp_head_').reset_index()
voice_opp_len=voice.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('voice_opp_len_').reset_index().fillna(0)
voice_call_type = voice.groupby(['uid','call_type'])['uid'].count().unstack().add_prefix('voice_call_type_').reset_index().fillna(0)
voice_in_out = voice.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('voice_in_out_').reset_index().fillna(0)


sms_opp_num = sms.groupby(['uid'])['opp_num'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('sms_opp_num_').reset_index()
sms_opp_head=sms.groupby(['uid'])['opp_head'].agg({'unique_count': lambda x: len(pd.unique(x))}).add_prefix('sms_opp_head_').reset_index()
sms_opp_len=sms.groupby(['uid','opp_len'])['uid'].count().unstack().add_prefix('sms_opp_len_').reset_index().fillna(0)
sms_in_out = sms.groupby(['uid','in_out'])['uid'].count().unstack().add_prefix('sms_in_out_').reset_index().fillna(0)

wa_name = wa.groupby(['uid'])['wa_name'].agg({'unique_count': lambda x: len(pd.unique(x)),'count':'count'}).add_prefix('wa_name_').reset_index()
visit_cnt = wa.groupby(['uid'])['visit_cnt'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_cnt_').reset_index()
visit_dura = wa.groupby(['uid'])['visit_dura'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_visit_dura_').reset_index()
up_flow = wa.groupby(['uid'])['up_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_up_flow_').reset_index()
down_flow = wa.groupby(['uid'])['down_flow'].agg(['std','max','min','median','mean','sum']).add_prefix('wa_down_flow_').reset_index()


def translate_time(date):
    #day = int(date[:1])
    hour = int(date[2:3])
    minute = int(date[4:5])
    seconds = int(date[6:7])
    return ((((hour)*60+minute)*60)+seconds)
voice['start_time'] = translate_time(voice['start_time'])
voice['end_time'] = translate_time(voice['end_time'])
voice["voice_time_long"] =voice['end_time']-voice['start_time']
voice['opp_len'] = voice['opp_len'].astype('category')

time_sum_by_uid = voice.groupby('uid')['voice_time_long'].sum()
time_sum_by_uid_opp = voice.groupby(['uid','opp_len'])['voice_time_long'].sum().unstack('opp_len')
time_sum_by_uid_opp.fillna(0,inplace=True)
voice = pd.concat([time_sum_by_uid,time_sum_by_uid_opp],axis =1).reset_index()


feature = [voice_opp_num,voice_opp_head,voice_opp_len,voice_call_type,voice_in_out,sms_opp_num,sms_opp_head,sms_opp_len,sms_in_out,wa_name,visit_cnt,visit_dura,up_flow,
           down_flow]

train_feature = uid_train
for feat in feature:
    train_feature=pd.merge(train_feature,feat,how='left',on='uid')

test_feature = uid_test
for feat in feature:
    test_feature=pd.merge(test_feature,feat,how='left',on='uid')

#train_feature.to_csv('/home/wxm/Downloads/data_mining/train_featureV1.csv',index=None)
#test_feature.to_csv('/home/wxm/Downloads/data_mining/test_featureV1.csv',index=None)

train=train_feature.fillna(0)
test=test_feature.fillna(0)

def get_interaction_feature(table, feature1, feature2):
    p = 0
    feature1_array = sorted(table[feature1].unique())
    feature2_array = sorted(table[feature2].unique())
    newfeat = {}
    for i in feature1_array:
        newfeat[int(i)] = {}
        for j in feature2_array:
            newfeat[int(i)][int(j)] = p
            p += 1
    return table.apply(lambda x: newfeat[int(x[feature1])][int(x[feature2])], axis=1)


def get_cross_feature(table):
    new_feature = table.copy()
    new_feature["1"] = get_interaction_feature(new_feature, "voice_opp_num_unique_count", "sms_opp_len_11")
    new_feature['2'] = get_interaction_feature(new_feature, 'voice_opp_num_unique_count', 'sms_opp_len_13')
    new_feature['3'] = get_interaction_feature(new_feature, 'voice_opp_num_unique_count', 'sms_in_out_0')
    new_feature['4'] = get_interaction_feature(new_feature, 'voice_opp_num_unique_count', 'sms_in_out_1')
    new_feature['5'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_opp_num_unique_count')
    new_feature['1_1'] = get_interaction_feature(new_feature, 'voice_opp_len_12', 'sms_opp_len_11')
    new_feature['2_1'] = get_interaction_feature(new_feature, 'voice_opp_len_12', 'sms_opp_len_13')
    new_feature['3_1'] = get_interaction_feature(new_feature, 'voice_opp_len_12', 'sms_in_out_0')
    new_feature['4_1'] = get_interaction_feature(new_feature, 'voice_opp_len_12', 'sms_in_out_1')
    new_feature['5_1'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_opp_len_12')
    new_feature['1_2'] = get_interaction_feature(new_feature, 'voice_call_type_1', 'sms_opp_len_11')
    new_feature['2_2'] = get_interaction_feature(new_feature, 'voice_call_type_1', 'sms_opp_len_13')
    new_feature['3_2'] = get_interaction_feature(new_feature, 'voice_call_type_1', 'sms_in_out_0')
    new_feature['4_2'] = get_interaction_feature(new_feature, 'voice_call_type_1', 'sms_in_out_1')
    new_feature['5_2'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_call_type_1')
    new_feature['1_3'] = get_interaction_feature(new_feature, 'voice_call_type_3', 'sms_opp_len_11')
    new_feature['2_3'] = get_interaction_feature(new_feature, 'voice_call_type_3', 'sms_opp_len_13')
    new_feature['3_3'] = get_interaction_feature(new_feature, 'voice_call_type_3', 'sms_in_out_0')
    new_feature['4_3'] = get_interaction_feature(new_feature, 'voice_call_type_3', 'sms_in_out_1')
    new_feature['5_3'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_call_type_3')
    new_feature['1_4'] = get_interaction_feature(new_feature, 'voice_in_out_0', 'sms_opp_len_11')
    new_feature['2_4'] = get_interaction_feature(new_feature, 'voice_in_out_0', 'sms_opp_len_13')
    new_feature['3_4'] = get_interaction_feature(new_feature, 'voice_in_out_0', 'sms_in_out_0')
    new_feature['4_4'] = get_interaction_feature(new_feature, 'voice_in_out_0', 'sms_in_out_1')
    new_feature['5_4'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_in_out_0')
    new_feature['1_5'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'sms_opp_len_11')
    new_feature['2_5'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'sms_opp_len_13')
    new_feature['3_5'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'sms_in_out_0')
    new_feature['4_5'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'sms_in_out_1')
    new_feature['5_5'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'voice_in_out_1')
    new_feature['1_6'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'voice_in_out_1')
    new_feature['2_6'] = get_interaction_feature(new_feature, 'sms_opp_len_13', 'sms_opp_len_13')
    new_feature['3_6'] = get_interaction_feature(new_feature, 'sms_in_out_0', 'sms_in_out_0')
    new_feature['4_6'] = get_interaction_feature(new_feature, 'sms_in_out_1', 'sms_in_out_1')
    new_feature['5_6'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'wa_down_flow_std')
    new_feature['1_7'] = get_interaction_feature(new_feature, 'voice_call_type_1', 'voice_call_type_1')
    new_feature['2_7'] = get_interaction_feature(new_feature, 'voice_call_type_3', 'voice_call_type_3')
    new_feature['3_7'] = get_interaction_feature(new_feature, 'voice_in_out_0', 'voice_in_out_0')
    new_feature['4_7'] = get_interaction_feature(new_feature, 'sms_in_out_1', 'sms_in_out_1')
    new_feature['5_7'] = get_interaction_feature(new_feature, 'wa_down_flow_std', 'wa_down_flow_std') 
    new_feature['1_8'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'voice_in_out_1')
    new_feature['2_8'] = get_interaction_feature(new_feature, 'voice_in_out_1', 'voice_in_out_1')
    return new_feature

train_feature1 = get_cross_feature(train)
test_feature1 = get_cross_feature(test)

train_feature1.to_csv('/home/wxm/Downloads/data_mining/feature_engineer/train_feature.csv')
test_feature1.to_csv('/home/wxm/Downloads/data_mining/feature_engineer/test_feature.csv')


