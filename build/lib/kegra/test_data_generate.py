
from utils import *
from tqdm import tqdm
import labels
import pandas as pd
import time
import os
import Levenshtein
def str2int(str):
    return int(str[:2])*60 + int(str[3:])

def get_encode(road_id, start_time, end_time):
    new_timing_restday = {}
    for key in list(timing_restday[road_id]):
        new_timing_restday[str2int(key)] = timing_restday[road_id][key]
    times = [key for key in new_timing_restday]
    encode = []
    start_time = str2int(start_time)
    end_time = str2int(end_time)
    for index in range(len(times)):  
        if (times[index] <= start_time and times[index + 1] > start_time):
            start_index = index
            break
    while (start_time < end_time):
        while(start_time < times[start_index + 1]):
            
            timings = new_timing_restday[times[start_index]]
            for i in range(len(timings)):
                if (road_id == '6410' and len(timings) == 4):
                    phase = phases_restday['6410-4'][i].replace(' ', '')
                elif (road_id == '6410' and len(timings) == 5):
                    phase = phases_restday['6410-5'][i].replace(' ', '')
                else:
                    phase = phases_restday[road_id][i].replace(' ', '')
                encode.extend([phase for i in range(int(timings[i]))])
            start_time += 5
            if (start_time > end_time):
              break
        start_index += 1
    return encode
import difflib

#判断相似度的方法，用到了difflib库
def get_equal_rate_1(str1, str2):
   return difflib.SequenceMatcher(None, str1, str2).quick_ratio()


def similarity(road_id, start_time, end_time, road_id2, start_time2, end_time2):
    id_transfer = {'26':'6513', '63':'6512', '62':'6511', '59':'6408', '60':'6409', '61':'6410'}
    code1 = get_encode(id_transfer[road_id],start_time,end_time)
    code2 = get_encode(id_transfer[road_id2],start_time2,end_time2)
    str1 = "".join(code1)
    str2 = "".join(code2)
    # zip() 函数接受两个字符串并将它们聚合在一个元组中。ord() 函数返回表示字节字符串中字符的整数。
    cnt = 0
    for a,b in zip(str1,str2):
        if (ord(a) ^ ord(b)):
            cnt = cnt + 1
    # return Levenshtein.ratio(str1, str2)
    # return get_equal_rate_1(str1, str2)
    return cnt/len(str1)

#     return difflib.SequenceMatcher(None, str1.join(code1), str2.join(code2)).ratio()
timing_restday = labels.timing_restday
phases_restday = labels.phases_restday

def getCrossFlow(df_cross, start_time, end_time):
    start = df_cross[df_cross.time==start_time].index[0]
    end = df_cross[df_cross.time==end_time].index[0]
    data_N = np.array(df_cross[start:end]['North'])
    data_W = np.array(df_cross[start:end]['West'])
    data_S = np.array(df_cross[start:end]['South'])
    data_E = np.array(df_cross[start:end]['East'])
    return np.stack([data_N, data_W, data_S, data_E],axis=1)
def getSample(roads, start_time, end_time):
    sample = []
    data_path = '/home/user1/GCN_ori/smartTraffic/traffic_data/cross/{}.csv'
    for road in roads:
        df = pd.read_csv(data_path.format(road))
        df.fillna(0,inplace = True)
        df = df.reset_index(drop=True)
        data = getCrossFlow(df,start_time, end_time)
        sample.append(data)
    return np.array(sample)



def getSample_for_road(road, start_time, end_time):
    sample = []
    data_path = '/home/user1/GCN_ori/smartTraffic/traffic_data/cross/{}.csv'
 
    df = pd.read_csv(data_path.format(road))
    df.fillna(0,inplace = True)
    df = df.reset_index(drop=True)
    data = getCrossFlow(df,start_time, end_time)
    sample.append(data)
    return np.array(sample)


def main():
    days = ['2020-10-30 ', '2020-10-31 ', '2020-11-01 ', '2020-11-02 ']
    start_time = '2020-10-30 08:00:00'
    end_time = '2020-10-30 10:00:00'
    roads = ['26','63','62','59','60','61']

    #4重循环
    car_flow_data = []
    labels = []
    for hour in tqdm(range(21)):
        for day1 in tqdm(days):
            for day2 in tqdm(days):
                for day3 in days:
                    for day4 in days:
                        for day5 in days:
                            for day6 in days:
                                start_hour = hour
                                end_hour = hour + 1
                                if(start_hour < 10):
                                    start_hour = '0'+str(start_hour)
                                else:
                                    start_hour = str(start_hour)
                                if(end_hour < 10):
                                    end_hour = '0'+str(end_hour)
                                else:
                                    end_hour = str(end_hour)
                                start_time = start_hour +':00:00'
                                end_time = end_hour +':00:00'
                                labels.append([[similarity(road1,start_hour +':00',end_hour +':00',road2,start_hour +':00',end_hour +':00') for road2 in roads] for road1 in roads])
                                car_flow1 = getSample_for_road('26', day1 + start_time, day1 + end_time)[0]
                                car_flow2 = getSample_for_road('63', day2 + start_time, day2 + end_time)[0]
                                car_flow3 = getSample_for_road('62', day3 + start_time, day3 + end_time)[0]
                                car_flow4 = getSample_for_road('59', day4 + start_time, day4 + end_time)[0]
                                car_flow5 = getSample_for_road('60', day5 + start_time, day5 + end_time)[0]
                                car_flow6 = getSample_for_road('61', day6 + start_time, day6 + end_time)[0]
                                car_flow_data.append([car_flow1,car_flow2,car_flow3,car_flow4,car_flow5,car_flow6])

    car_flow_data=np.array(car_flow_data)
    labels = np.array(labels)
    np.save('car_flow_data.npy',car_flow_data)
    np.save('labels.npy',labels)


# @profile
def main2():
    days = ['2020-10-30 ', '2020-10-31 ', '2020-11-01 ', '2020-11-02 ']
    hours = ['00', '01', '02','03','04','05','06', '07', '08', '09','10', '11','12', '13','14','15', '16','17', '18', '19','20', '21','22','23', '24']
    start_time = '2020-10-30 08:00:00'
    end_time = '2020-10-30 10:00:00'
    roads = ['26','63','62','59','60','61']

    #4重循环
    car_flow_data = []
    labels = []
    day1 = days[0]
    day2 = days[0]
    day3 = days[0]
    day4 = days[0]
    day5 = days[0]
    day6 = days[0]
    hour = 0
    # for hour in tqdm(range(23)):
    #     for day1 in tqdm(days):
    #         for day2 in tqdm(days):
    #             for day3 in tqdm(days):
    #                 for day4 in tqdm(days):
    #                     for day5 in tqdm(days):
    #                         for day6 in tqdm(days):
    start_hour = hour
    end_hour = hour + 1
    if(start_hour < 10):
        start_hour = '0'+str(start_hour)
    else:
        start_hour = str(start_hour)
    if(end_hour < 10):
        end_hour = '0'+str(end_hour)
    else:
        end_hour = str(end_hour)
    start_time = start_hour +':00:00'
    end_time = end_hour +':00:00'
    labels.append([[similarity(road1,start_hour +':00',end_hour +':00',road2,start_hour +':00',end_hour +':00') for road2 in roads] for road1 in roads])
    car_flow1 = getSample_for_road('26', day1 + start_time, day1 + end_time)[0]
    car_flow2 = getSample_for_road('63', day2 + start_time, day2 + end_time)[0]
    car_flow3 = getSample_for_road('62', day3 + start_time, day3 + end_time)[0]
    car_flow4 = getSample_for_road('59', day4 + start_time, day4 + end_time)[0]
    car_flow5 = getSample_for_road('60', day5 + start_time, day5 + end_time)[0]
    car_flow6 = getSample_for_road('61', day6 + start_time, day6 + end_time)[0]
    car_flow_data.append([car_flow1,car_flow2,car_flow3,car_flow4,car_flow5,car_flow6])

    car_flow_data=np.array(car_flow_data)
    labels = np.array(labels)


if __name__=="__main__":
    main()