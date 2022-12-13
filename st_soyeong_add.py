"""
지금까지 쓴 코드 1차 수합 및 스트림릿으로..
""" 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px #일단 이거 이용해 기본 그래프 그림
import nltk 

st.set_page_config(page_icon="🗽", page_title="데이터저널리즘 2조")

st.markdown("""
# 아시아인 혐오 범죄, 확산과 심화: 코로나19 전후 미국의 혐오 범죄를 조명하다
* 데이터저널리즘 2조: 오소영, 이혜정, 최다연
* 저희 팀은 코로나19 팬데믹과 트럼프의 등장이 미국의 아시아인 혐오 범죄의 심각성에 어떤 영향을 미쳤는지 그 상관성에 주목해보고자 했습니다. 미국 FBI 범죄 통계와 대통령 연설문을 주 자료로 활용해 (1)인구 10만 명 당 코로나19 확진자·사망자 수가 가장 많은 10개 주와 가장 적은 10개 주에서의 아시아인 대상 혐오 범죄 발생 빈도수와 범죄 성격을 분석하고 (2)트럼프의 등장 전후 아시아인 혐오 범죄 양상을 파악함으로써 “트럼프 등장 이후 혐오 범죄 수가 증가했으며, 코로나19의 피해가 심각했을수록 아시아인 혐오 범죄의 심각성이 크다”는 결론을 도출했습니다.
----
""")

fbi =  pd.read_csv("fbi_data_edited.csv", low_memory=False)
df = pd.DataFrame(fbi, columns=['DATA_YEAR', 'BIAS_DESC'])

#소영언니 파트다른 인종에 대한 범죄도 늘었는가, 아시아에 대한 범죄만 늘었는가(범죄 성격)

st.markdown("""
## 문제의식
""")

#Anti Asian 증가만 보여주는 그래프
st.subheader("1. 2015년에서 2020년까지, 미국 전역 혐오 범죄 건수 변화")
options_2 = st.radio('옵션을 선택하세요', ['전체 혐오 범죄', '아시아인 대상 혐오 범죄', '전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율'])
df_2015 = df[(df['DATA_YEAR'] == 2015)]
df_2016 = df[(df['DATA_YEAR'] == 2016)]
df_2017 = df[(df['DATA_YEAR'] == 2017)]
df_2018 = df[(df['DATA_YEAR'] == 2018)]
df_2019 = df[(df['DATA_YEAR'] == 2019)]
df_2020 = df[(df['DATA_YEAR'] == 2020)]

#새로운 df: 연도별 전체 / 아시아인 대상 혐오범죄 count. index: 연도, columns: 값
#전체
all_hates = [len(df_2015['BIAS_DESC']), len(df_2016['BIAS_DESC']), len(df_2017['BIAS_DESC']), len(df_2018['BIAS_DESC']),
            len(df_2019['BIAS_DESC']), len(df_2020['BIAS_DESC'])]
df_all = pd.DataFrame(all_hates, index = ['2015', '2016', '2017', '2018', '2019', '2020'])

all_fig = px.line(df_all, title = "2015-2020 All Hate Crime Cases in the US", markers=True)
all_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Counted Cases"}))

#아시아인 대상 혐오범죄

asian_hates = [len(df_2015.loc[df_2015['BIAS_DESC'] == 'Anti-Asian']), len(df_2016.loc[df_2016['BIAS_DESC'] == 'Anti-Asian']), 
               len(df_2017.loc[df_2017['BIAS_DESC'] == 'Anti-Asian']), len(df_2018.loc[df_2018['BIAS_DESC'] == 'Anti-Asian']),
               len(df_2019.loc[df_2019['BIAS_DESC'] == 'Anti-Asian']), len(df_2020.loc[df_2020['BIAS_DESC'] == 'Anti-Asian'])]
df_asian_edit = pd.DataFrame(asian_hates, index = ['2015', '2016', '2017', '2018', '2019', '2020'])
asian_fig = px.line(df_asian_edit, title = "2015-2020 Asian Hate Crime Cases in the US", markers=True)
asian_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Counted Cases"}))

#전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율
all_asian_ratio = []
for i in range(len(asian_hates)):
    all_asian_ratio.append((int(asian_hates[i]) / int(all_hates[i])) * 100) #아시아인 혐오범죄 건수 / 전체 혐오범죄 건수 * 100 : 비율 (맞나?)

#print(all_asian_ratio)

df_compare = pd.DataFrame(all_asian_ratio, index = ['2015', '2016', '2017', '2018', '2019', '2020'])
compare_fig = px.bar(df_compare, title = "2015-2020 Asian Hate Crime Cases Ratio Out of All Hate Crimes in the US")
compare_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "%"}))

if options_2 == '전체 혐오 범죄':
    st.plotly_chart(all_fig)
elif options_2 == '아시아인 대상 혐오 범죄':
    st.plotly_chart(asian_fig)
elif options_2 == '전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율':
    st.plotly_chart(compare_fig)


st.markdown("""
##### 분석
* FBI에서 미국 전역의 혐오 범죄 건수를 피해대상(victims)별로 분류해 센 자료를 시각화한 결과, 2015년부터 2020년까지 미국 전역의 혐오 범죄 건수는 증가 양상을 보여 왔습니다.
* 특히 트럼프 전 대통령의 당선이 확정된 2016년 대선을 기점으로, 그가 임기를 시작한 2017년까지 혐오 범죄의 증가율은 더욱 가팔라졌다는 것이 주목할 만합니다.
* 팬데믹이 시작된 2020년의 혐오 범죄 건수는 전년도 대비 30% 이상 증가해 전례를 찾아볼 수 없는 변화를 보였으며, 그중에서도 아시아인 대상 혐오 범죄는 55% 증가했습니다. 5년 전인 2015년에 비교했을 때는 283% 가량 증가한 수치입니다. 전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율 또한 2020년 최고치인 3.13%을 기록했습니다.에서 미국 전역의 혐오 범죄 건수를 피해대상(victims)별로 분류해 센 자료를 시각화한 결과, 2015년부터 2020년까지 미국 전역의 혐오 범죄 건수는 증가 양상을 보여 왔습니다.
""")

#혐오 범죄 대상별 비율
st.subheader("2. 2015년에서 2020년, 미국 전역 혐오 범죄의 대상별 비율")
options_1 = st.radio('연도를 선택하세요', ['2015', '2016', '2017', '2018', '2019', '2020'])

@st.cache #함수가 있어야 캐시설정이 되는 듯해 아래 다 함수로 바꿈.
def crime_pie_chart(filename):
    crime_data_count = pd.read_csv(filename)
    crime_data_count['number'].astype(int)
    cdc_2015 = crime_data_count['number'].tolist() #대상 인종별 범죄수 담은 리스트.
    idx_2015 = list(crime_data_count['race'])

    return crime_data_count
    


if options_1 == "2015":
    fig_2015 = px.pie(crime_pie_chart("hate_crime_race.csv"), values = 'number', names = 'race', title = '2015')
    st.plotly_chart(fig_2015)

elif options_1 == "2016":
    fig_2016 = px.pie(crime_pie_chart("hate_crime_race_2016.csv"), values = 'number', names = 'race', title = '2016')
    st.plotly_chart(fig_2016)


elif options_1 == "2017":
    fig_2017 = px.pie(crime_pie_chart("hate_crime_race_2017.csv"), values = 'number', names = 'race', title = '2017')
    st.plotly_chart(fig_2017)

elif options_1 == "2018":
    fig_2018 = px.pie(crime_pie_chart("hate_crime_race_2018.csv"), values = 'number', names = 'race', title = '2018')
    st.plotly_chart(fig_2018)


elif options_1 == "2019":
    fig_2019 = px.pie(crime_pie_chart("hate_crime_race_2019.csv"), values = 'number', names = 'race', title = '2019')
    st.plotly_chart(fig_2019)

elif options_1 == "2020":
    fig_2020 = px.pie(crime_pie_chart("hate_crime_race_2020.csv"), values = 'number', names = 'race', title = '2020')
    st.plotly_chart(fig_2020)

st.markdown("""
##### 분석
* FBI에서 미국 전역의 혐오 범죄 건수를 피해대상(victims)별로 분류한 자료를 바탕으로, 아시아인을 대상으로 하는 혐오 범죄가 전체 혐오 범죄 중 얼마를 차지하는지 원 그래프로 나타냈습니다.
* 특정 대상을 향한 혐오 범죄가 50건 이하 발생한 경우는 기타로 통합했습니다.
* 아시아인 대상 혐오 범죄가 전체 혐오 범죄에서 차지하는 비율은 꾸준히 상승했으며, 코로나19가 발생한 해 눈에 띄게 증가했습니다.
* 이처럼 혐오 범죄는 트럼프 출현 이후로 계속적으로 증가해왔으나, 그중에서도 아시아인 혐오 범죄 건수는 코로나19 이후로 눈에 띄게 증가했다고 볼 수 있겠습니다. 
----
""")

#가능하면 scatterplot으로 코로나19 case 증가 - 아시아인 혐오 범죄 증가 사이 상관관계 그리면 좋을 듯.

#혜정 파트: Asian hate crime 
# 범죄 유형이 과격해지지 않았는가 
# 장소의 공개성이 증가했는가

st.subheader("아시아인 대상 혐오 범죄 양상 변화")
st.markdown("""
##### 1. 2016년-2020년: 월별 아시아인 대상 혐오 범죄 건수 시각화
""")

df1 = pd.DataFrame(fbi, columns=['DATA_YEAR', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME']) #혜정이가 쓴 데이터프레임 변수명 안겹치게 df1으로.
df1 = df1[df1['BIAS_DESC'].str.contains('Anti-Asian', na = False)]     ## Anti-Asian
df1 = df1[(df1['DATA_YEAR'] >= 2016)]      ## 연도별로 보기 위해 2016(트럼프 당선 연도부터: 전후비교 느낌)

df1['INCIDENT_DATE'] = pd.to_datetime(df1['INCIDENT_DATE'])
#df1['INCIDENT_DATE'].dtype()

#연도-달 컬러맵
#각 연도 기준으로, 달별로 범죄 건수 카운트해서 저장. 이후 컬러맵 그리기
df1['INCIDENT_DATE'] = pd.to_datetime(df1['INCIDENT_DATE'])

@st.cache
def count_monthly_crime(year_num):
    cases = []
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-01-01', f'{year_num}-01-31')])) #1월 - 순서랑 날짜 맞추려고 일일히 했음
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-02-01', f'{year_num}-02-28')])) #2월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-03-01', f'{year_num}-03-31')])) #3월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-04-01', f'{year_num}-04-30')])) #4월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-05-01', f'{year_num}-05-31')])) #5월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-06-01', f'{year_num}-06-30')])) #6월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-07-01', f'{year_num}-07-31')])) #7월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-08-01', f'{year_num}-08-30')])) #8월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-09-01', f'{year_num}-09-30')])) #9월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-10-01', f'{year_num}-10-31')])) #10월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-11-01', f'{year_num}-11-30')])) #11월
    cases.append(len(df1[df1['INCIDENT_DATE'].between(f'{year_num}-12-01', f'{year_num}-12-31')])) #12월
    return cases

monthly_crime = [count_monthly_crime(2016), count_monthly_crime(2017), count_monthly_crime(2018), count_monthly_crime(2019), count_monthly_crime(2020)]

month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
year_month_colormap = px.imshow(monthly_crime,
                labels=dict(x="Months", y="Year", color="Cases", width=500, height=700), 
                x = month_name,
                y = ['2016', '2017', '2018', '2019', '2020'],
                title = "2016-2020 Asian Hate Crimes",
                color_continuous_scale='Reds',
                width = 800, height = 600)
# year_month_colormap.update_layout() #color="Cases" : 못생겨서 뺌
st.plotly_chart(year_month_colormap)

st.markdown("""
##### 분석
* 2016년 대선이 있었던 11월 아시아인 혐오 범죄가 소폭 증가한 양상을 보이며, 이후 간헐적으로 발생 빈도가 높아지던 아시아인 혐오 범죄 건수는 2020년 코로나19가 창궐하며 정점을 찍었습니다. 미국에 코로나19가 상륙해 가장 기승을 부리던 3월과 4월의 아시아인 혐오 범죄 건수는 각각 52건으로, 이전 최다발생 월인 2019년 12월의 22건보다도 2.5배가량 높은 수치입니다. 같은 해 10월까지 아시아인 대상 혐오 범죄 건수는 매월 20건 이상으로 유례없는 발생 현황을 보였습니다. 
""")

# # Colormap of crime at 2019
# df_2019 = df1[(df1['DATA_YEAR'] == 2019)] 
# df_2019['INCIDENT_DATE'] = pd.to_datetime(df_2019['INCIDENT_DATE'])
# df_2019['INCIDENT_day'] = df_2019['INCIDENT_DATE'].dt.day
# df_2019['INCIDENT_month_name'] = df_2019['INCIDENT_DATE'].dt.month_name()
# df_2019['INCIDENT_month_name'].sort_values()
# df_2019_c = pd.crosstab(df_2019.INCIDENT_day, df_2019.INCIDENT_month_name)
# print(df_2019_c)


# df_2019_fig = px.imshow(df_2019_c, color_continuous_scale=px.colors.sequential.Reds, title="Asian Hate crimes at 2019")
# df_2019_fig.update_layout(xaxis = dict({"title" : "Month"}), yaxis = dict({"title" : "Day"}))
# st.plotly_chart(df_2019_fig)

# # Colormap of crime at 2020
# df_2020 = df1[(df1['DATA_YEAR'] == 2020)] 
# df_2020['INCIDENT_DATE'] = pd.to_datetime(df_2020['INCIDENT_DATE'])
# df_2020['INCIDENT_day'] = df_2020['INCIDENT_DATE'].dt.day
# df_2020['INCIDENT_month_name'] = df_2020['INCIDENT_DATE'].dt.month_name()

# df_2020_c = pd.crosstab(df_2020.INCIDENT_day, df_2020.INCIDENT_month_name)


# df_2020_fig = px.imshow(df_2020_c, color_continuous_scale=px.colors.sequential.Reds, title="Asian Hate crimes at 2020")
# df_2020_fig.update_layout(xaxis = dict({"title" : "Month"}), yaxis = dict({"title" : "Day"}))

###########################

st.markdown("""
##### 2. 2019년~2020년: 범죄 유형의 과격성 변화\n표 설명: 과격성은 신체에 직접적으로 가해지는 위해 정도를 따져, 범죄 종류의 위해성이 높을수록 4에 가깝게 분류했습니다. 미국 보통법 기준 경범죄와 중범죄 분류를 참고했을 때, 대체로 1~2에 해당하는 범죄는 경범죄, 3~4에 해당하는 범죄는 중범죄로 분류됩니다.
* 5는 기타 범죄 유형입니다.
1 = 협박(intimidation), 문서 위조(counterfeiting/forgery), 약물 소지 및 학교에 들고 감(Drug violations), 무기 소지 및 학교에 들고 감(Weapon Law Violation)\n
2 = 절도(theft), 재물손괴(destruction/damage/vandalism of property), 기타 절도(all other larceny)\n
3 = 폭행(assault), 강도(robbery), 주거침입강도(burglary/breaking & entering), 방화(arson)\n
4 = 강간(rape), 유괴(Kidnapping), 살인·과실치사·모살(murder and nonnegligent manslaughter)\n
5 = 범죄 형태 특정되지 않음(not specified), 해킹·컴퓨터 범죄(hacking/computer invasion)
""")

df1 = pd.DataFrame(fbi, columns=['DATA_YEAR', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME']) #혜정이가 쓴 데이터프레임 변수명 안겹치게 df1으로.
df1 = df1[df1['BIAS_DESC'].str.contains('Anti-Asian', na = False)]     ## Anti-Asian
df1 = df1[(df1['DATA_YEAR'] >= 2019)]      ## 2019, 2020



#################################################

# crime type table
offense_type_tab = pd.crosstab(df1.DATA_YEAR, df1.OFFENSE_NAME)
# offense_type_tab

# crime type cleaning
df1.loc[(df1['OFFENSE_NAME'].str.startswith('Aggravated Assault')), 'OFFENSE_NAME'] = 'Aggravated Assault'
df1.loc[(df1['OFFENSE_NAME'].str.startswith('Destruction/Damage/Vandalism of Property')), 'OFFENSE_NAME'] = 'Destruction/Damage/Vandalism of Property'
df1.loc[(df1['OFFENSE_NAME'].str.startswith('Intimidation')), 'OFFENSE_NAME'] = 'Intimidation'
df1.loc[(df1['OFFENSE_NAME'].str.contains('Drug')), 'OFFENSE_NAME'] = 'Drug Violations'
df1.loc[(df1['OFFENSE_NAME'].str.contains('Theft')), 'OFFENSE_NAME'] = 'Theft'

# crime type cleaning (2nd)
df1.loc[df1['OFFENSE_NAME'] == 'Intimidation', 'OFFENSE_NAME'] = 1
df1.loc[df1['OFFENSE_NAME'] == 'Counterfeiting/Forgery', 'OFFENSE_NAME'] = 1
df1.loc[df1['OFFENSE_NAME'] == 'Drug Violations', 'OFFENSE_NAME'] = 1
df1.loc[df1['OFFENSE_NAME'] == 'Weapon Law Violations', 'OFFENSE_NAME'] = 1

df1.loc[df1['OFFENSE_NAME'] == 'All Other Larceny', 'OFFENSE_NAME'] = 2
df1.loc[df1['OFFENSE_NAME'] == 'Theft', 'OFFENSE_NAME'] = 2
df1.loc[df1['OFFENSE_NAME'] == 'Destruction/Damage/Vandalism of Property', 'OFFENSE_NAME'] = 2

#방화, 강도, 주거침입강도 모두 2에서 3으로 옮김
df1.loc[df1['OFFENSE_NAME'] == 'Simple Assault', 'OFFENSE_NAME'] = 3
df1.loc[df1['OFFENSE_NAME'] == 'Aggravated Assault', 'OFFENSE_NAME'] = 3
df1.loc[df1['OFFENSE_NAME'] == 'Robbery', 'OFFENSE_NAME'] = 3
df1.loc[df1['OFFENSE_NAME'] == 'Burglary/Breaking & Entering', 'OFFENSE_NAME'] = 3
df1.loc[df1['OFFENSE_NAME'] == 'Arson', 'OFFENSE_NAME'] = 3

df1.loc[df1['OFFENSE_NAME'] == 'Rape', 'OFFENSE_NAME'] = 4
df1.loc[df1['OFFENSE_NAME'] == 'Murder and Nonnegligent Manslaughter', 'OFFENSE_NAME'] = 4

df1.loc[df1['OFFENSE_NAME'] == 'Not Specified', 'OFFENSE_NAME'] = 5
df1.loc[df1['OFFENSE_NAME'] == 'Hacking/Computer Invasion', 'OFFENSE_NAME'] = 5

# crime type table (numeric order)
offense_type_tab_c = pd.crosstab(df1.DATA_YEAR, df1.OFFENSE_NAME)

# Line and Bar plot for crime type
offense_type_line = pd.crosstab(df1.OFFENSE_NAME, columns= df1.DATA_YEAR)
offense_type_bar = pd.crosstab(df1.OFFENSE_NAME, columns= df1.DATA_YEAR, normalize=True)
offense_type_line_plt = px.line(offense_type_line, title = "범죄 종류별 분류 선그래프", markers=True)
offense_type_bar_plt = px.bar(offense_type_bar, title = "범죄 종류별 분류 막대그래프")

st.plotly_chart(offense_type_line_plt)
st.plotly_chart(offense_type_bar_plt)


st.markdown("""
**분석**
* 범죄의 과격성은 범죄 발생 빈도수와 마찬가지로 2020년 대폭 증가했습니다. 살인과 같은 과격성 4 정도의 중범죄를 제외하고, 2019년 대비 2020년 아시아인을 대상으로 한 과격성 1~3의 혐오 범죄가 크게 증가했습니다. 경범죄와 중범죄가 고루 증가했다고 볼 수 있으며, 범죄 종류별 분류 막대그래프를 통해 증가율을 한눈에 볼 수 있습니다.
""")

# =====================================

st.markdown("""
##### 3. 2019년~2020년: 범죄 장소의 공개성 변화\n범죄 장소의 공개성은 장소의 개방성과 공공성, 유동 인구를 고려해 1~5의 수치로 분류했으며, 숫자가 커질수록 공개성이 높음을 의미합니다. 공개성이 1, 2이면 폐쇄적인 장소, 공개성이 3이면 공간 개방성은 높으나 유동인구는 적은 장소, 공개성이 4,5이면 유동 인구와 공공성이 모두 높은 장소입니다. 
* 6은 공개성을 임의로 측정하기 어려운 기타 범죄 장소입니다.
1 = 집(Residence/Home), 숙박업소(Hotel/Motel/Etc.)\n
2 = 전문상점(Specialty store), 주류 판매점(Liquor Store), 차 딜러샵(Auto Dealership New/Used)\n
3 = 공장 부지(Industrial Site), 들판·숲(Field/Woods), 강가·바닷가(Lake/Waterway/Beach), 주차장·차고(Parking/Drop Lot/Garage), 주유소·서비스센터(Service/Gas Station)\n
4 = 술집·나이트클럽(Bar/Nightclub), 식당(Restaurant), 공원·놀이터(Park/Playground), 도로·보도(Highway/Road/Alley/Street/Sidewalk), 휴게소(rest area), convenience store(편의점), 백화점·할인판매점(Department/Discount Store), 쇼핑몰(Shopping Mall), 식료품점·슈퍼마켓(Grocery/Supermarket), 교통 시설(Air/Bus/Train Terminal), 상업용 건물(Commercial/Office Building)\n
5 = 초·중등·대학교(School-College/Elementary/Secondary), 주민자치센터(Community Center), 홈리스 쉼터(Shelter-Mission/Homeless), 은행(Bank/Savings and Loan), 종교 시설(Church/Synagogue/Temple/Mosque), 정부·공공기관 건물(Church/Synagogue/Temple/Mosque), 의료기관(Drug Store/Doctor's Office/Hospital)\n
6 = 알려지지 않음(Other/Unknown), 온라인 공간(Cyber space)
""")

# crime place cleaning
df1.loc[df1['LOCATION_NAME'] == 'Residence/Home', 'LOCATION_NAME'] = 1
df1.loc[df1['LOCATION_NAME'] == 'Hotel/Motel/Etc.', 'LOCATION_NAME'] = 1

df1.loc[df1['LOCATION_NAME'] == 'Auto Dealership New/Used', 'LOCATION_NAME'] = 2
df1.loc[df1['LOCATION_NAME'] == 'Specialty Store', 'LOCATION_NAME'] = 2
df1.loc[df1['LOCATION_NAME'] == 'Liquor Store', 'LOCATION_NAME'] = 2

df1.loc[df1['LOCATION_NAME'] == 'Field/Woods', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Lake/Waterway/Beach', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Parking/Drop Lot/Garage', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Service/Gas Station', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Industrial Site', 'LOCATION_NAME'] = 3

df1.loc[df1['LOCATION_NAME'] == 'Convenience Store', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Department/Discount Store', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Shopping Mall', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Grocery/Supermarket', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Air/Bus/Train Terminal', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Commercial/Office Building', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Park/Playground', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Rest Area', 'LOCATION_NAME'] = 4.
df1.loc[df1['LOCATION_NAME'] == 'Bar/Nightclub', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Restaurant', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk;Residence/Home', 'LOCATION_NAME'] = 4


#감옥 애매해서 뺌
df1.loc[df1['LOCATION_NAME'] == 'School-College/University', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'School-Elementary/Secondary', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'School/College', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Community Center', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Shelter-Mission/Homeless', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Bank/Savings and Loan', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Church/Synagogue/Temple/Mosque', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Government/Public Building', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == "Drug Store/Doctor's Office/Hospital", 'LOCATION_NAME'] = 5


df1.loc[df1['LOCATION_NAME'] == 'Cyberspace', 'LOCATION_NAME'] = 6
df1.loc[df1['LOCATION_NAME'] == 'Other/Unknown', 'LOCATION_NAME'] = 6

# Line and Bar plot for publicity of place
offense_place_tab_c = pd.crosstab(df1.LOCATION_NAME, df1.DATA_YEAR)
offense_place_line = pd.crosstab(df1.LOCATION_NAME, columns = df1.DATA_YEAR)
offense_place_bar = pd.crosstab(df1.LOCATION_NAME, columns = df1.DATA_YEAR)

offense_place_line_plt = px.line(offense_place_line, title = "범죄 장소별 분류 선그래프", markers=True)
offense_place_bar_plt = px.bar(offense_place_bar, title = "범죄 장소별 분류 막대그래프")

st.plotly_chart(offense_place_line_plt)
st.plotly_chart(offense_place_bar_plt)

#################################################

st.markdown("""##### 분석
* 공개된 장소에서 일어난 범죄도 유의미하게 증가했습니다.
* 전반적으로 혐오 범죄 건수가 늘어난 가운데, 유동 인구가 많고 개방성이 높은 공개성 4의 장소에서 발생한 범죄 건수는 전년도 대비 2020년 두 배 가량 증가했다는 사실을 막대그래프를 통해 포착할 수 있습니다. 이처럼 더욱 공개적으로 변화한 범죄 장소는 아시아인 대상 혐오 범죄가 더욱 공공연하게 발생하고 있음을 보여줍니다.
* 소결론: 코로나19가 발생한 2020년 초를 기점으로 아시아인 대상 혐오 범죄 건수가 급증했고, 강도와 폭행 등 과격성이 높은 혐오 범죄의 빈도수가 함께 증가했습니다. 범죄 장소의 공공성과 개방성을 척도로 매긴 공개성도 마찬가지로 높아졌습니다. 요컨대 코로나19의 발생과 아시아인 혐오 범죄의 심각성이 높은 상관관계를 지니고 있음을 추론할 수 있습니다.
-----
""")

#다연 파트: 코로나의 심각성에 따른 비교- 주별 비교

st.subheader("2020년 기준 주별 코로나19 피해와 혐오 범죄 발생 양상")
st.markdown("""
* 이전까지는 미국 전체를 대상으로 코로나19 발생 전후 아시아인 혐오 범죄의 심각성이 어떻게 변화했는지를 봤다면, 이어서 2020년 코로나19로 인한 피해가 심했던 지역과 그렇지 않았던 지역의 아시안 혐오 범죄 양상을 시각화했습니다./n
#### 1. 혐오 범죄 빈도수 변화\n붉은 색 지도는 2020년 10만 명당 코로나19 확진자수·사망자수, 푸른 색 지도는 2019년과 2020년의 혐오 범죄 건수를 주별로 시각화한 것입니다.
""")

#파일 임포트 
covid_state = pd.read_csv("us_county_covid_2020.csv") #2020년 12월 31일 기준, 미국 각 주 county별 누적 확진 수를 담은 파일.

#주별 인구 10만명 당 코로나19 확진자 수 표 그리기
covid_state_df = pd.DataFrame(covid_state)

#주별 누적 확진 건수 추출하는 함수
@st.cache #캐시 사용하도록 설정

def state_cases(state_name):
    state = covid_state_df.loc[covid_state_df['state'] == state_name]
    return state['cases'].sum()

@st.cache #캐시 사용하도록 설정
#주별 누적 사망 건수 추출 함수
def state_deaths(state_name):
    state = covid_state_df.loc[covid_state_df['state'] == state_name]
    return state['deaths'].sum()
#state_deaths('Alabama')

state_name_list = list(covid_state_df['state'].unique())

cases_list = []
for item in state_name_list:
    cases_list.append(int(state_cases(item)))
#cases_list

state_case = {'State':state_name_list, 'Cases':cases_list}
state_case_df = pd.DataFrame(state_case)
#state_case_df

deaths_list = []
for item in state_name_list:
    deaths_list.append(int(state_deaths(item)))

#deaths_list

state_death = {'State':state_name_list, 'Deaths':deaths_list}
state_death_df = pd.DataFrame(state_death)

#10만 명당: (전체 건수 / 2020 기준 해당 주 인구수) * 100,000 - 해당 주 인구수는 us census.gov 사이트에서 가져온 파일, 2020 7월 기준
us_population = pd.read_csv("us_population_2020.csv")
us_population['2020'] = us_population['2020'].apply(lambda x: int(x.replace(',', '')))

us_case_population = pd.merge(state_case_df, us_population)
#us_case_population

#10만 명당: (전체 건수 / 2020 기준 해당 주 인구수) * 100,000 - 해당 주 인구수는 us census.gov 사이트에서 가져온 파일, 2020 7월 기준
us_case_population['cases_per_population'] = us_case_population.apply(lambda x: ((x['Cases'] / x['2020']) * 100000), axis=1)
us_case_population_final = us_case_population[['State', 'cases_per_population']]
#us_case_population_final

us_death_population = pd.merge(state_death_df, us_population)
#1천 명당 사망자 수 : (전체 건수 / 2020 기준 해당 주 인구수) * 1000
us_death_population['deaths_per_1k'] = us_death_population.apply(lambda x: ((x['Deaths'] / x['2020']) * 1000), axis=1)
us_death_population_final = us_death_population[['State', 'deaths_per_1k']]
#us_death_population_final

from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/ChoiDayeun/datajournalism_streamlit/main/us-states_json_edit.json') as response:
    states = json.load(response)

#지도 데이터 불러와서 시각화.
fig_case_population = px.choropleth(us_case_population_final, geojson= states, locations='State', 
                    color = 'cases_per_population',
                    color_continuous_scale="Reds",
                    range_color=(0, us_case_population_final.cases_per_population.max()),
                    featureidkey='properties.State',
                    scope="usa",
                    labels={'cases_per_population':'cases per 100k'},
                    title = 'US COVID-19 Cases per 100k by States', 
                    )
fig_case_population.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
#fig_case_population.show()

#지도 데이터 불러와서 시각화.
fig_death_population = px.choropleth(us_death_population_final, geojson= states, locations='State', 
                    color = 'deaths_per_1k',
                    color_continuous_scale="Reds",
                    range_color=(0, us_death_population_final.deaths_per_1k.max()),
                    featureidkey='properties.State',
                    scope="usa",
                    labels={'deaths_per_1k':'deaths per 1k'},
                    title = 'US COVID-19 Deaths per 1k by States', 
                    )
fig_death_population.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
#fig_death_population.show()

select_type = st.selectbox('원하는 항목을 선택하세요', ['확진자수 기준', '사망자수 기준'])
if select_type == '확진자수 기준':
    st.plotly_chart(fig_case_population)
elif select_type == '사망자수 기준':
    st.plotly_chart(fig_death_population)

#주별 혐오범죄 건수 표 그리기: 2019, 2020
fbi_state_2019 = fbi.loc[fbi['DATA_YEAR'] == 2019][['STATE_NAME']]

#각 주별 범죄 총 건수 세는 함수
@st.cache #캐시 사용하도록 설정
def state_crimes(state_name):
    state = fbi_state_2019.loc[fbi_state_2019['STATE_NAME'] == state_name]
    return len(state)

#state_crimes('New York')

crime_list_2019 = []
for item in state_name_list:
    #print(item)
    crime_list_2019.append(int(state_crimes(item)))

state_crime_2019 = {'State':state_name_list, 'Crimes':crime_list_2019}
state_crime_df_2019 = pd.DataFrame(state_crime_2019)
#state_crime_df_2019

###############2020년##################
fbi_state_2020 = fbi.loc[fbi['DATA_YEAR'] == 2020][['STATE_NAME']]
#각 주별 범죄 총 건수 세는 함수
@st.cache #캐시 사용하도록 설정
def state_crimes(state_name):
    state = fbi_state_2020.loc[fbi_state_2020['STATE_NAME'] == state_name]
    return len(state)

#state_crimes('New York')

crime_list_2020 = []
for item in state_name_list:
    #print(item)
    crime_list_2020.append(int(state_crimes(item)))

state_crime_2020 = {'State':state_name_list, 'Crimes':crime_list_2020}
state_crime_df_2020 = pd.DataFrame(state_crime_2020)
#state_crime_df_2019

######표######

fig_crime_2019 = px.choropleth(state_crime_df_2019, geojson= states, locations='State', 
                    color = 'Crimes',
                    color_continuous_scale="Blues",
                    range_color=(0, state_crime_df_2020.Crimes.max()),
                    featureidkey='properties.State',
                    scope="usa",
                    labels={'case':'Crimes'},
                    title = 'Crime Cases per State 2019', 
                    )
fig_crime_2019.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
#fig_crime_2019.show()


fig_crime_2020 = px.choropleth(state_crime_df_2020, geojson= states, locations='State', 
                    color = 'Crimes',
                    color_continuous_scale="Blues",
                    range_color=(0, state_crime_df_2020.Crimes.max()),
                    featureidkey='properties.State',
                    scope="usa",
                    labels={'case':'Crimes'},
                    title = 'Crime Cases per State 2020', 
                    )
fig_crime_2020.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
#fig_crime_2020.show()

select_yr = st.selectbox('확인할 년도를 선택하세요', ['2019', '2020'])
if select_yr == '2019':
    st.plotly_chart(fig_crime_2019)
elif select_yr == '2020':
    st.plotly_chart(fig_crime_2020)


st.markdown("""
##### 분석
* 혐오범죄 건수는 코로나19의 주별 심각성과는 무관하게 모든 주에서 증가했습니다.
""")

###주별 정리 다시: 5개로 하니 차이가 너무 안 보여서 10개로 했는데.. 결과가 애매꾸리. 표 형식을 바꾸라면 바꾸겠음.###

st.markdown("""
### 2. 혐오 범죄 유형의 과격성과 심각성 - 상위, 하위 10개주별 비교\n요컨대 혐오 범죄 발생 빈도수에서 유의미한 주별 차이는 없었습니다. 그렇다면 범죄의 과격성과 공개성은 코로나10 전후로 어떤 변화가 있었을까요?\n
인구 10만 명당 코로나19 확진자수, 인구 1000명당 코로나19 사망자수 상위 10개 주와 하위 10개 주를 나눠 과격성과 공개성을 비교했습니다.
""")


######여기부터는 표 만드는 노가다- 코드 .. 
#혜정코드 가져옴, crime type 2019-2020 정리 (전체 혐오범죄 대상으로 해서 수정함)

df_new = pd.DataFrame(fbi, columns=['DATA_YEAR', 'STATE_NAME', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME'])
df_new = df_new[(df_new['DATA_YEAR'] >= 2019)]

# crime type cleaning 1st
df_new.loc[(df_new['OFFENSE_NAME'].str.startswith('Aggravated Assault')), 'OFFENSE_NAME'] = 'Aggravated Assault'
df_new.loc[(df_new['OFFENSE_NAME'].str.startswith('Murder and Nonnegligent Manslaughter')), 'OFFENSE_NAME'] = 'Murder and Nonnegligent Manslaughter'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Destruction/Damage/Vandalism of Property')), 'OFFENSE_NAME'] = 'Destruction/Damage/Vandalism of Property'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Intimidation')), 'OFFENSE_NAME'] = 'Intimidation'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Drug')), 'OFFENSE_NAME'] = 'Drug Violations'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Theft')), 'OFFENSE_NAME'] = 'Theft'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Robbery')), 'OFFENSE_NAME'] = 'Robbery'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Burglary')), 'OFFENSE_NAME'] = 'Burglary'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Fraud')), 'OFFENSE_NAME'] = 'Fraud'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Rape')), 'OFFENSE_NAME'] = 'Rape'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Prostitution')), 'OFFENSE_NAME'] = 'Prostitution'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Arson')), 'OFFENSE_NAME'] = 'Arson'
df_new.loc[(df_new['OFFENSE_NAME'].str.startswith('Kidnapping')), 'OFFENSE_NAME'] = 'Kidnapping'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Extortion')), 'OFFENSE_NAME'] = 'Extortion'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Shoplifting')), 'OFFENSE_NAME'] = 'Shoplifting'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Purse-snatching')), 'OFFENSE_NAME'] = 'Shoplifting'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Pocket-picking')), 'OFFENSE_NAME'] = 'Shoplifting' #약간 좀도둑 느낌이면 다 shoplifting 
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Fondling')), 'OFFENSE_NAME'] = 'Fondling'
df_new.loc[(df_new['OFFENSE_NAME'].str.contains('Simple Assault')), 'OFFENSE_NAME'] = 'Simple Assault'


df_new.loc[(df_new['OFFENSE_NAME'].str.contains('All Other Larceny')), 'OFFENSE_NAME'] = 'All Other Larceny' #다 정리하고도 남으면 기타로 빼기.

# crime type cleaning (2nd)
df_new.loc[df_new['OFFENSE_NAME'] == 'Intimidation', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Counterfeiting/Forgery', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Drug Violations', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Weapon Law Violations', 'OFFENSE_NAME'] = 1 
df_new.loc[df_new['OFFENSE_NAME'] == 'Embezzlement', 'OFFENSE_NAME'] = 1 
df_new.loc[df_new['OFFENSE_NAME'] == 'Sodomy', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Extortion', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Fondling', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Pornography/Obscene Material', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Shoplifting', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Impersonation', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Fraud', 'OFFENSE_NAME'] = 1 #사기까지
df_new.loc[df_new['OFFENSE_NAME'] == 'False Pretenses/Swindle/Confidence Game', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Stolen Property Offenses', 'OFFENSE_NAME'] = 1
df_new.loc[df_new['OFFENSE_NAME'] == 'Animal Cruelty', 'OFFENSE_NAME'] = 1

df_new.loc[df_new['OFFENSE_NAME'] == 'All Other Larceny', 'OFFENSE_NAME'] = 2
df_new.loc[df_new['OFFENSE_NAME'] == 'Theft', 'OFFENSE_NAME'] = 2 #여기서부터는 절도 등등
df_new.loc[df_new['OFFENSE_NAME'] == 'Destruction/Damage/Vandalism of Property', 'OFFENSE_NAME'] = 2
df_new.loc[df_new['OFFENSE_NAME'] == 'Prostitution', 'OFFENSE_NAME'] = 2 
df_new.loc[df_new['OFFENSE_NAME'] == 'Human Trafficking, Commercial Sex Acts', 'OFFENSE_NAME'] = 2

df_new.loc[df_new['OFFENSE_NAME'] == 'Robbery', 'OFFENSE_NAME'] = 3
df_new.loc[df_new['OFFENSE_NAME'] == 'Burglary', 'OFFENSE_NAME'] = 3
df_new.loc[df_new['OFFENSE_NAME'] == 'Arson', 'OFFENSE_NAME'] = 3 
df_new.loc[df_new['OFFENSE_NAME'] == 'Simple Assault', 'OFFENSE_NAME'] = 3
df_new.loc[df_new['OFFENSE_NAME'] == 'Aggravated Assault', 'OFFENSE_NAME'] = 3 #가중 폭행
df_new.loc[df_new['OFFENSE_NAME'] == 'Sexual Assault With An Object', 'OFFENSE_NAME'] = 3
df_new.loc[df_new['OFFENSE_NAME'] == 'Negligent Manslaughter', 'OFFENSE_NAME'] = 3

df_new.loc[df_new['OFFENSE_NAME'] == 'Kidnapping', 'OFFENSE_NAME'] = 4 #유괴 3? 4?
df_new.loc[df_new['OFFENSE_NAME'] == 'Rape', 'OFFENSE_NAME'] = 4
df_new.loc[df_new['OFFENSE_NAME'] == 'Murder and Nonnegligent Manslaughter', 'OFFENSE_NAME'] = 4

df_new.loc[df_new['OFFENSE_NAME'] == 'Not Specified', 'OFFENSE_NAME'] = 5
df_new.loc[df_new['OFFENSE_NAME'] == 'Hacking/Computer Invasion', 'OFFENSE_NAME'] = 5 #1로 보내는 것은 어떨지? 


# crime place cleaning 1st

df_new.loc[(df_new['LOCATION_NAME'].str.contains('Highway/Road/Alley/Street/Sidewalk')), 'LOCATION_NAME'] = 'Highway/Road/Alley/Street/Sidewalk'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('Store')), 'LOCATION_NAME'] = 'Store'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('Facility')), 'LOCATION_NAME'] = 'Facility'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('School-Elementary/Secondary')), 'LOCATION_NAME'] = 'School-Elementary/Secondary'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('Auto Dealership New/Used')), 'LOCATION_NAME'] = 'Auto Dealership New/Used'
df_new.loc[(df_new['LOCATION_NAME'].str.startswith('Hotel/Motel/Etc')), 'LOCATION_NAME'] = 'Hotel/Motel/Etc'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('Amusement Park')), 'LOCATION_NAME'] = 'Amusement Park' 
df_new.loc[(df_new['LOCATION_NAME'].str.contains('Commercial/Office Building')), 'LOCATION_NAME'] = 'Commercial/Office Building'
df_new.loc[(df_new['LOCATION_NAME'].str.contains('ATM Separate from Bank')), 'LOCATION_NAME'] = 'ATM Separate from Bank'
df_new.loc[(df_new['LOCATION_NAME'].str.startswith('Grocery/Supermarket')), 'LOCATION_NAME'] = 'Grocery/Supermarket'  

df_new.loc[(df_new['LOCATION_NAME'].str.startswith('Other/Unknown')), 'LOCATION_NAME'] = 'Other/Unknown' #기타 

# crime place cleaning 2nd
df_new.loc[df_new['LOCATION_NAME'] == 'Residence/Home', 'LOCATION_NAME'] = 1
df_new.loc[df_new['LOCATION_NAME'] == 'Hotel/Motel/Etc', 'LOCATION_NAME'] = 1



df_new.loc[df_new['LOCATION_NAME'] == 'ATM Separate from Bank', 'LOCATION_NAME'] = 2 
df_new.loc[df_new['LOCATION_NAME'] == 'Auto Dealership New/Used', 'LOCATION_NAME'] = 2
df_new.loc[df_new['LOCATION_NAME'] == 'Abandoned/Condemned Structure', 'LOCATION_NAME'] = 2
df_new.loc[df_new['LOCATION_NAME'] == 'Military Installation', 'LOCATION_NAME'] = 2

df_new.loc[df_new['LOCATION_NAME'] == 'Construction Site', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Industrial Site', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Field/Woods', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Camp/Campground', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Arena/Stadium/Fairgrounds/Coliseum', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Tribal Lands', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Parking/Drop Lot/Garage', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Lake/Waterway/Beach', 'LOCATION_NAME'] = 3
df_new.loc[df_new['LOCATION_NAME'] == 'Service/Gas Station', 'LOCATION_NAME'] = 3

df_new.loc[df_new['LOCATION_NAME'] == 'Store', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Facility', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Shopping Mall', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Grocery/Supermarket', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Air/Bus/Train Terminal', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Dock/Wharf/Freight/Modal Terminal', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Commercial/Office Building', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Bar/Nightclub', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Restaurant', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Amusement Park', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Park/Playground', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk', 'LOCATION_NAME'] = 4
df_new.loc[df_new['LOCATION_NAME'] == 'Rest Area', 'LOCATION_NAME'] = 4


df_new.loc[df_new['LOCATION_NAME'] == 'School-College/University', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'School-Elementary/Secondary', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'School/College', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'Community Center', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'Shelter-Mission/Homeless', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'Bank/Savings and Loan', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'Church/Synagogue/Temple/Mosque', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == 'Government/Public Building', 'LOCATION_NAME'] = 5
df_new.loc[df_new['LOCATION_NAME'] == "Drug Store/Doctor's Office/Hospital", 'LOCATION_NAME'] = 5


df_new.loc[df_new['LOCATION_NAME'] == 'Cyberspace', 'LOCATION_NAME'] = 6
df_new.loc[df_new['LOCATION_NAME'] == 'Other/Unknown', 'LOCATION_NAME'] = 6

#####상위 10개주 case 추출, 하위 10개주 case 추출: 10만 명당 코로나19 확진수 기준 

#10만 명당 확진 건수별로 정렬: 상위 10개, 하위 10개 주 추출
#상위 10개, 하위 10개
case_top10_df = us_case_population_final.sort_values('cases_per_population', ascending = False)[:10]
case_bottom10_df = us_case_population_final.sort_values('cases_per_population')[:10]

#case_top10_df['State']

#해당 주 이름들 추출
case_top10_names = []
case_bottom10_names = []
for item in case_top10_df['State']:
    case_top10_names.append(item)

for item in case_bottom10_df['State']:
    case_bottom10_names.append(item)


#1000명당 사망 건수별로 정렬: 상위 10개, 하위 10개 주 추출
#상위 10개, 하위 10개
death_top10_df = us_death_population_final.sort_values('deaths_per_1k', ascending = False)[:10]
death_bottom10_df = us_death_population_final.sort_values('deaths_per_1k')[:10]

#death_top10_df['State']

#해당 주 이름들 추출
death_top10_names = []
death_bottom10_names = []
for item in death_top10_df['State']:
    death_top10_names.append(item)


for item in death_bottom10_df['State']:
    death_bottom10_names.append(item)

#case_top10_names,  case_bottom10_names에 있는 주들의 값을 바꾼 후, df에 해당 컬럼들만 남김: 상위 10개 : TOP_10_COVID_CASES, BOTTOM_10_COVID_CASES
for item in case_top10_names:
    df_new.loc[df_new['STATE_NAME'] == item, 'STATE_NAME'] = 'TOP_10_COVID_CASES'
for item in case_bottom10_names:
    df_new.loc[df_new['STATE_NAME'] == item, 'STATE_NAME'] = 'BOTTOM_10_COVID_CASES'

df_cases_state = df_new.loc[(df_new['STATE_NAME'] == 'TOP_10_COVID_CASES') | (df_new['STATE_NAME'] == 'BOTTOM_10_COVID_CASES')]
#print(df_cases_state)

####cases top10 추출
top10_df = df_cases_state[df_cases_state['STATE_NAME'] == 'TOP_10_COVID_CASES']
top10_2019 = top10_df[top10_df['DATA_YEAR'] == 2019]
top10_2020 = top10_df[top10_df['DATA_YEAR'] == 2020]

bottom10_df = df_cases_state[df_cases_state['STATE_NAME'] == 'BOTTOM_10_COVID_CASES']
bottom10_2019 = bottom10_df[bottom10_df['DATA_YEAR'] == 2019]
bottom10_2020 = bottom10_2020 = bottom10_df[bottom10_df['DATA_YEAR'] == 2020]


offense_list = list(df_cases_state['OFFENSE_NAME'].unique())
offense_list.sort()
#print(offense_list)

offense_case_list_top_2019 = []
offense_case_list_bottom_2019 = []
offense_case_list_top_2020 = []
offense_case_list_bottom_2020 = []

for item in offense_list:
    offense_case_list_top_2019.append(len(top10_2019[top10_2019['OFFENSE_NAME'] == item]))
    offense_case_list_top_2020.append(len(top10_2020[top10_2020['OFFENSE_NAME'] == item]))
    offense_case_list_bottom_2019.append(len(bottom10_2019[bottom10_2019['OFFENSE_NAME'] == item]))
    offense_case_list_bottom_2020.append(len(bottom10_2020[bottom10_2020['OFFENSE_NAME'] == item]))

# print(offense_case_list_top_2019)
# print(offense_case_list_top_2020)
# print(offense_case_list_bottom_2019)
# print(offense_case_list_bottom_2020) 

location_list = list(df_cases_state['LOCATION_NAME'].unique())
location_list.sort()
# print(location_list)

location_case_list_top_2019 = []
location_case_list_bottom_2019 = []
location_case_list_top_2020 = []
location_case_list_bottom_2020 = []

for item in location_list:
    location_case_list_top_2019.append(len(top10_2019[top10_2019['LOCATION_NAME'] == item]))
    location_case_list_top_2020.append(len(top10_2020[top10_2020['LOCATION_NAME'] == item]))
    location_case_list_bottom_2019.append(len(bottom10_2019[bottom10_2019['LOCATION_NAME'] == item]))
    location_case_list_bottom_2020.append(len(bottom10_2020[bottom10_2020['LOCATION_NAME'] == item]))

# print(location_case_list_top_2019)
# print(location_case_list_top_2020)
# print(location_case_list_bottom_2019)
# print(location_case_list_bottom_2020) 


################사망자수 기준으로 위에랑 똑같이##################

df_new2 = pd.DataFrame(fbi, columns=['DATA_YEAR', 'STATE_NAME', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME'])
df_new2 = df_new2[(df_new2['DATA_YEAR'] >= 2019)]

df_new2.loc[(df_new2['OFFENSE_NAME'].str.startswith('Aggravated Assault')), 'OFFENSE_NAME'] = 'Aggravated Assault'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.startswith('Murder and Nonnegligent Manslaughter')), 'OFFENSE_NAME'] = 'Murder and Nonnegligent Manslaughter'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Destruction/Damage/Vandalism of Property')), 'OFFENSE_NAME'] = 'Destruction/Damage/Vandalism of Property'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Intimidation')), 'OFFENSE_NAME'] = 'Intimidation'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Drug')), 'OFFENSE_NAME'] = 'Drug Violations'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Theft')), 'OFFENSE_NAME'] = 'Theft'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Robbery')), 'OFFENSE_NAME'] = 'Robbery'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Burglary')), 'OFFENSE_NAME'] = 'Burglary'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Fraud')), 'OFFENSE_NAME'] = 'Fraud'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Rape')), 'OFFENSE_NAME'] = 'Rape'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Prostitution')), 'OFFENSE_NAME'] = 'Prostitution'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Arson')), 'OFFENSE_NAME'] = 'Arson'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.startswith('Kidnapping')), 'OFFENSE_NAME'] = 'Kidnapping'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Extortion')), 'OFFENSE_NAME'] = 'Extortion'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Shoplifting')), 'OFFENSE_NAME'] = 'Shoplifting'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Purse-snatching')), 'OFFENSE_NAME'] = 'Shoplifting'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Pocket-picking')), 'OFFENSE_NAME'] = 'Shoplifting' #약간 좀도둑 느낌이면 다 shoplifting 
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Fondling')), 'OFFENSE_NAME'] = 'Fondling'
df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('Simple Assault')), 'OFFENSE_NAME'] = 'Simple Assault'


df_new2.loc[(df_new2['OFFENSE_NAME'].str.contains('All Other Larceny')), 'OFFENSE_NAME'] = 'All Other Larceny' #다 정리하고도 남으면 기타로 빼기.

# crime type cleaning (2nd)
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Intimidation', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Counterfeiting/Forgery', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Drug Violations', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Weapon Law Violations', 'OFFENSE_NAME'] = 1 
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Embezzlement', 'OFFENSE_NAME'] = 1 
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Sodomy', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Extortion', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Fondling', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Pornography/Obscene Material', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Shoplifting', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Impersonation', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Fraud', 'OFFENSE_NAME'] = 1 #사기까지
df_new2.loc[df_new2['OFFENSE_NAME'] == 'False Pretenses/Swindle/Confidence Game', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Stolen Property Offenses', 'OFFENSE_NAME'] = 1
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Animal Cruelty', 'OFFENSE_NAME'] = 1

df_new2.loc[df_new2['OFFENSE_NAME'] == 'All Other Larceny', 'OFFENSE_NAME'] = 2
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Theft', 'OFFENSE_NAME'] = 2 #여기서부터는 절도 등등
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Destruction/Damage/Vandalism of Property', 'OFFENSE_NAME'] = 2
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Prostitution', 'OFFENSE_NAME'] = 2 
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Human Trafficking, Commercial Sex Acts', 'OFFENSE_NAME'] = 2

df_new2.loc[df_new2['OFFENSE_NAME'] == 'Arson', 'OFFENSE_NAME'] = 3 #방화는 3으로 빼는 것은 어떨지? 
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Robbery', 'OFFENSE_NAME'] = 3
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Burglary', 'OFFENSE_NAME'] = 3
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Simple Assault', 'OFFENSE_NAME'] = 3
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Aggravated Assault', 'OFFENSE_NAME'] = 3 #가중 폭행
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Sexual Assault With An Object', 'OFFENSE_NAME'] = 3
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Negligent Manslaughter', 'OFFENSE_NAME'] = 3

df_new2.loc[df_new2['OFFENSE_NAME'] == 'Kidnapping', 'OFFENSE_NAME'] = 4 #유괴 3? 4?
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Rape', 'OFFENSE_NAME'] = 4
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Murder and Nonnegligent Manslaughter', 'OFFENSE_NAME'] = 4

df_new2.loc[df_new2['OFFENSE_NAME'] == 'Not Specified', 'OFFENSE_NAME'] = 5
df_new2.loc[df_new2['OFFENSE_NAME'] == 'Hacking/Computer Invasion', 'OFFENSE_NAME'] = 5 #1로 보내는 것은 어떨지? 


# crime place cleaning 1st

df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Highway/Road/Alley/Street/Sidewalk')), 'LOCATION_NAME'] = 'Highway/Road/Alley/Street/Sidewalk'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Store')), 'LOCATION_NAME'] = 'Store'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Facility')), 'LOCATION_NAME'] = 'Facility'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('School-Elementary/Secondary')), 'LOCATION_NAME'] = 'School-Elementary/Secondary'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Auto Dealership New/Used')), 'LOCATION_NAME'] = 'Auto Dealership New/Used'
df_new2.loc[(df_new2['LOCATION_NAME'].str.startswith('Hotel/Motel/Etc')), 'LOCATION_NAME'] = 'Hotel/Motel/Etc'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Amusement Park')), 'LOCATION_NAME'] = 'Amusement Park' 
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('Commercial/Office Building')), 'LOCATION_NAME'] = 'Commercial/Office Building'
df_new2.loc[(df_new2['LOCATION_NAME'].str.contains('ATM Separate from Bank')), 'LOCATION_NAME'] = 'ATM Separate from Bank'
df_new2.loc[(df_new2['LOCATION_NAME'].str.startswith('Grocery/Supermarket')), 'LOCATION_NAME'] = 'Grocery/Supermarket'  

df_new2.loc[(df_new2['LOCATION_NAME'].str.startswith('Other/Unknown')), 'LOCATION_NAME'] = 'Other/Unknown' #기타 

# crime place cleaning 2nd
df_new2.loc[df_new2['LOCATION_NAME'] == 'Residence/Home', 'LOCATION_NAME'] = 1
df_new2.loc[df_new2['LOCATION_NAME'] == 'Hotel/Motel/Etc', 'LOCATION_NAME'] = 1


df_new2.loc[df_new2['LOCATION_NAME'] == 'ATM Separate from Bank', 'LOCATION_NAME'] = 2 
df_new2.loc[df_new2['LOCATION_NAME'] == 'Auto Dealership New/Used', 'LOCATION_NAME'] = 2
df_new2.loc[df_new2['LOCATION_NAME'] == 'Abandoned/Condemned Structure', 'LOCATION_NAME'] = 2
df_new2.loc[df_new2['LOCATION_NAME'] == 'Military Installation', 'LOCATION_NAME'] = 2 #얘도 3으로 빼는 것은 어떨지?


df_new2.loc[df_new2['LOCATION_NAME'] == 'Construction Site', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Industrial Site', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Field/Woods', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Camp/Campground', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Arena/Stadium/Fairgrounds/Coliseum', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Tribal Lands', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Service/Gas Station', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Lake/Waterway/Beach', 'LOCATION_NAME'] = 3
df_new2.loc[df_new2['LOCATION_NAME'] == 'Parking/Drop Lot/Garage', 'LOCATION_NAME'] = 3

df_new2.loc[df_new2['LOCATION_NAME'] == 'Store', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Facility', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Shopping Mall', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Grocery/Supermarket', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Air/Bus/Train Terminal', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Dock/Wharf/Freight/Modal Terminal', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Commercial/Office Building', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Bar/Nightclub', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Restaurant', 'LOCATION_NAME'] = 4

df_new2.loc[df_new2['LOCATION_NAME'] == 'School-College/University', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'School-Elementary/Secondary', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'School/College', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'Community Center', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'Shelter-Mission/Homeless', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'Bank/Savings and Loan', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'Church/Synagogue/Temple/Mosque', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == 'Government/Public Building', 'LOCATION_NAME'] = 5
df_new2.loc[df_new2['LOCATION_NAME'] == "Drug Store/Doctor's Office/Hospital", 'LOCATION_NAME'] = 5 


df_new2.loc[df_new2['LOCATION_NAME'] == 'Amusement Park', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Park/Playground', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk', 'LOCATION_NAME'] = 4
df_new2.loc[df_new2['LOCATION_NAME'] == 'Rest Area', 'LOCATION_NAME'] = 4


df_new2.loc[df_new2['LOCATION_NAME'] == 'Cyberspace', 'LOCATION_NAME'] = 6
df_new2.loc[df_new2['LOCATION_NAME'] == 'Other/Unknown', 'LOCATION_NAME'] = 6


#############death로 다시#############
#death_top10_names,  death_bottom10_names에 있는 주들의 값을 바꾼 후, df에 해당 컬럼들만 남김: 상위 10개 : TOP_10_COVID_deathS, BOTTOM_10_COVID_deathS
for item in death_top10_names:
    df_new2.loc[df_new2['STATE_NAME'] == item, 'STATE_NAME'] = 'TOP_10_COVID_DEATHS'
for item in death_bottom10_names:
    df_new2.loc[df_new2['STATE_NAME'] == item, 'STATE_NAME'] = 'BOTTOM_10_COVID_DEATHS'
df_deaths_state = df_new2.loc[(df_new2['STATE_NAME'] == 'TOP_10_COVID_DEATHS') | (df_new2['STATE_NAME'] == 'BOTTOM_10_COVID_DEATHS')]
#df_deaths_state

offense_death_list = list(df_deaths_state['OFFENSE_NAME'].unique())
offense_death_list.sort()
#print(offense_list)

#####death 기준 top10 추출
top10_death_df = df_deaths_state[df_deaths_state['STATE_NAME'] == 'TOP_10_COVID_DEATHS']
top10_death_2019 = top10_df[top10_df['DATA_YEAR'] == 2019]
top10_death_2020 = top10_df[top10_df['DATA_YEAR'] == 2020]

bottom10_df = df_deaths_state[df_deaths_state['STATE_NAME'] == 'BOTTOM_10_COVID_DEATHS']
bottom10_2019 = bottom10_df[bottom10_df['DATA_YEAR'] == 2019]
bottom10_2020 = bottom10_2020 = bottom10_df[bottom10_df['DATA_YEAR'] == 2020]


offense_death_list_top_2019 = []
offense_death_list_bottom_2019 = []
offense_death_list_top_2020 = []
offense_death_list_bottom_2020 = []

for item in offense_list:
    offense_death_list_top_2019.append(len(top10_2019[top10_2019['OFFENSE_NAME'] == item]))
    offense_death_list_top_2020.append(len(top10_2020[top10_2020['OFFENSE_NAME'] == item]))
    offense_death_list_bottom_2019.append(len(bottom10_2019[bottom10_2019['OFFENSE_NAME'] == item]))
    offense_death_list_bottom_2020.append(len(bottom10_2020[bottom10_2020['OFFENSE_NAME'] == item]))




location_death_list = list(df_deaths_state['LOCATION_NAME'].unique())
location_death_list.sort()

location_death_list_top_2019 = []
location_death_list_bottom_2019 = []
location_death_list_top_2020 = []
location_death_list_bottom_2020 = []

for item in location_list:
    location_death_list_top_2019.append(len(top10_2019[top10_2019['LOCATION_NAME'] == item]))
    location_death_list_top_2020.append(len(top10_2020[top10_2020['LOCATION_NAME'] == item]))
    location_death_list_bottom_2019.append(len(bottom10_2019[bottom10_2019['LOCATION_NAME'] == item]))
    location_death_list_bottom_2020.append(len(bottom10_2020[bottom10_2020['LOCATION_NAME'] == item]))

# print(location_death_list_top_2019)
# print(location_death_list_top_2020)
# print(location_death_list_bottom_2019)
# print(location_death_list_bottom_2020) 


#####################표 그리기####################

import plotly.graph_objects as go
#아래 표 제작 참고 링크: https://plotly.com/python/categorical-axes/

@st.cache(allow_output_mutation=True)
def fig_states(off_or_loc, list_2019, list_2020):
    fig_states = go.Figure()
    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = list_2019,
    name = "2019",
    ))

    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = list_2020,
    name = "2020",
    ))

    return fig_states

#표 세트 1: 확진자 수 상위/하위 10개 주 기준 범죄 유형 과격성/장소 공개성 비교

#유형 과격성 - 상위10개
fig_offensetype_top10 = fig_states(offense_list, offense_case_list_top_2019, offense_case_list_top_2020)
fig_offensetype_top10.update_layout(title_text="코로나19 확진자 수가 가장 많은 10개 주: 범죄 유형 과격성 변화")


#유형 과격성 - 하위10개
fig_offensetype_bottom10 = fig_states(offense_list, offense_case_list_bottom_2019, offense_case_list_bottom_2020)
fig_offensetype_bottom10.update_layout(title_text="코로나19 확진자 수가 가장 적은 10개 주: 범죄 유형 과격성 변화")

#장소 공개성 - 상위10개
fig_locationtype_top10 = fig_states(location_list, location_case_list_top_2019, location_case_list_top_2020)
fig_locationtype_top10.update_layout(title_text="코로나19 확진자 수가 가장 많은 10개 주: 범죄 장소의 공개성 변화")


#장소 공개성 - 하위10개
fig_locationtype_bottom10 = fig_states(location_list, location_case_list_bottom_2019, location_case_list_bottom_2020)
fig_locationtype_bottom10.update_layout(title_text="코로나19 확진자 수가 가장 적은 10개 주: 범죄 장소의 공개성 변화")

#fig_locationtype_bottom10.show()



#표 세트 2: 사망자 수 상위/하위 10개 주 기준 범죄 유형 과격성/장소 공개성 비교
@st.cache(allow_output_mutation=True)
def fig_states_death(off_or_loc, death_list_2019, death_list_2020):
    fig_states = go.Figure()
    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2019,
    name = "2019",
    ))

    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2020,
    name = "2020",
    ))

    return fig_states

#유형 과격성 - 상위10개
fig_offensetype_death_top10 = fig_states_death(offense_death_list, offense_death_list_top_2019, offense_death_list_top_2020)
fig_offensetype_death_top10.update_layout(title_text="코로나19 사망자 수가 가장 많은 10개 주: 범죄 유형 과격성 변화")


#유형 과격성 - 하위10개
fig_offensetype_death_bottom10 = fig_states_death(offense_death_list, offense_death_list_bottom_2019, offense_death_list_bottom_2020)
fig_offensetype_death_bottom10.update_layout(title_text="코로나19 사망자 수가 가장 적은 10개 주: 범죄 유형 과격성 변화")


#장소 공개성 - 상위10개
fig_locationtype_death_top10 = fig_states_death(location_death_list, location_death_list_top_2019, location_death_list_top_2020)
fig_locationtype_death_top10.update_layout(title_text="코로나19 사망자 수가 가장 많은 10개 주: 범죄 장소의 공개성 변화")


#장소 공개성 - 하위10개
fig_locationtype_death_bottom10 = fig_states_death(location_death_list, location_death_list_bottom_2019, location_death_list_bottom_2020)
fig_locationtype_death_bottom10.update_layout(title_text="코로나19 사망자 수가 가장 적은 10개 주: 범죄 장소의 공개성 변화")



#표 보이기
my_order = ['코로나19 확진자 수 기준', '코로나19 사망자 수 기준']
status = st.radio('무엇을 기준으로 확인할지 선택하세요', my_order)

if status == '코로나19 확진자 수 기준':
    st.markdown("""
    #### 1. 범죄 유형의 과격성 변화
    # """)
    st.plotly_chart(fig_offensetype_top10)
    st.plotly_chart(fig_offensetype_bottom10)

    st.markdown("""
##### 분석
* 확진자수가 가장 많은 10개 주의 경우, 2019년 대비 2020년 과격성 지수가 높은 범죄 유형과 낮은 범죄 유형이 고루 증가했습니다.
반면 확진자수가 가장 적은 10개 주에서는 과격성 지수가 3 이상인 범죄는 오히려 줄어든 양상을 보였습니다.
""")

    st.markdown("""
    #### 2. 범죄 장소의 공개성 변화
    # """)
    st.plotly_chart(fig_locationtype_top10)
    st.plotly_chart(fig_locationtype_bottom10)

    
elif status== '코로나19 사망자 수 기준':
    ("""
    #### 범죄 유형의 과격성
    # """)
    st.plotly_chart(fig_offensetype_death_top10)
    st.plotly_chart(fig_offensetype_death_bottom10)

    st.markdown("""
##### 분석
*  사망자수가 가장 많은 10개 주의 경우, 2019년 대비 2020년 과격성 지수가 높은 범죄 유형과 낮은 범죄 유형이 고루 증가했습니다.
반면 사망자수가 가장 적은 10개 주에서는 과격성 지수가 3 이상인 범죄는 오히려 줄어든 양상을 보였습니다.
""")

    
    st.markdown("""
    #### 범죄 장소의 공개성
    # """)
    st.plotly_chart(fig_locationtype_death_top10)
    st.plotly_chart(fig_locationtype_death_bottom10)

    st.markdown("""
##### 분석
* 공개성이 4인 장소에서 발생한 혐오 범죄 건수의 경우, 사망자수가 가장 많은 10개주에서의 증가율이 사망자 수가 가장 적은 10개주에서의 증가율보다 높게 나타났습니다.
* 공개성이 5인 장소에서 발생한 혐오 범죄는 사망자수가 가장 많은 10개주에서 증가한 반면, 사망자수가 가장 적은 10개주에서는 감소했습니다.
""")

st.markdown("""##### 소결론 TOP_10_COVID_DEATHS
* 코로나19 확진자수와 사망자 수 중 어느 것을 기준으로 잡아도, 상위 10개주와 하위 10개주에서 발생한 혐오 범죄 유형의 과격성과 장소의 공개성에는 유의미한 차이가 발생했습니다.\n
* 요컨대 코로나19로 인한 피해가 심했던 주에서 발생한 혐오 범죄가 더욱 과격하고 공개적으로 이뤄지는 경향이 있다고 할 수 있습니다.
----
""")



################아시아인 대상 혐오범죄만, 확진 건수 기준##################
df_cases_asian = df_cases_state[df_cases_state['BIAS_DESC'].str.contains('Anti-Asian', na = False)]
offense_asian = list(df_cases_asian['OFFENSE_NAME'].unique())
offense_asian.sort()
#print(offense_asian)

location_asian = list(df_cases_asian['LOCATION_NAME'].unique())
location_asian.sort()

# 건수 top10 bottom10 추출
top10_asian_cases_df = df_cases_asian[df_cases_asian['STATE_NAME'] == 'TOP_10_COVID_CASES']
top10_asian_cases_2019 = top10_asian_cases_df[top10_asian_cases_df['DATA_YEAR'] == 2019]
top10_asian_cases_2020 = top10_asian_cases_df[top10_asian_cases_df['DATA_YEAR'] == 2020]

bottom10_asian_cases_df = df_cases_asian[df_cases_asian['STATE_NAME'] == 'BOTTOM_10_COVID_CASES']
bottom10_asian_cases_2019 = bottom10_asian_cases_df[bottom10_asian_cases_df['DATA_YEAR'] == 2019]
bottom10_asian_cases_2020 = bottom10_asian_cases_df[bottom10_asian_cases_df['DATA_YEAR'] == 2020]


offense_list_asian_top_2019 = []
offense_list_asian_bottom_2019 = []
offense_list_asian_top_2020 = []
offense_list_asian_bottom_2020 = []

for item in offense_asian:
    offense_list_asian_top_2019.append(len(top10_asian_cases_2019[top10_asian_cases_2019['OFFENSE_NAME'] == item]))
    offense_list_asian_top_2020.append(len(top10_asian_cases_2020[top10_asian_cases_2020['OFFENSE_NAME'] == item]))
    offense_list_asian_bottom_2019.append(len(bottom10_asian_cases_2019[bottom10_asian_cases_2019['OFFENSE_NAME'] == item]))
    offense_list_asian_bottom_2020.append(len(bottom10_asian_cases_2020[bottom10_asian_cases_2020['OFFENSE_NAME'] == item]))

print(offense_list_asian_top_2019)


location_list_asian_top_2019 = []
location_list_asian_bottom_2019 = []
location_list_asian_top_2020 = []
location_list_asian_bottom_2020 = []

for item in location_asian:
    location_list_asian_top_2019.append(len(top10_asian_cases_2019[top10_asian_cases_2019['LOCATION_NAME'] == item]))
    location_list_asian_top_2020.append(len(top10_asian_cases_2020[top10_asian_cases_2020['LOCATION_NAME'] == item]))
    location_list_asian_bottom_2019.append(len(bottom10_asian_cases_2019[bottom10_asian_cases_2019['LOCATION_NAME'] == item]))
    location_list_asian_bottom_2020.append(len(bottom10_asian_cases_2020[bottom10_asian_cases_2020['LOCATION_NAME'] == item]))

@st.cache(allow_output_mutation=True)
def fig_states_asian(off_or_loc, death_list_2019, death_list_2020):
    fig_states = go.Figure()
    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2019,
    name = "2019",
    ))

    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2020,
    name = "2020",
    ))

    return fig_states

#아시아인 혐오 범죄, 확진 건수 기준
#유형 과격성 - 상위10개
fig_asian_offensetype_top10 = fig_states_asian(offense_asian, offense_list_asian_top_2019, offense_list_asian_top_2020)
fig_asian_offensetype_top10.update_layout(title_text="코로나19 확진자 수가 가장 많은 10개 주: 아시아인 대상 혐오 범죄 유형 과격성 변화")
#fig_asian_offensetype_death_top10.show()

#유형 과격성 - 하위10개
fig_asian_offensetype_bottom10 = fig_states_asian(offense_asian, offense_list_asian_bottom_2019, offense_list_asian_bottom_2020)
fig_asian_offensetype_bottom10.update_layout(title_text="코로나19 확진자 수가 가장 적은 10개 주: 아시아인 대상 혐오 범죄 유형 과격성 변화")
#fig_asian_offensetype_death_bottom10.show()

#장소 공개성 - 상위10개
fig_asian_locationtype_top10 = fig_states_asian(location_asian, location_list_asian_top_2019, location_list_asian_top_2020)
fig_asian_locationtype_top10.update_layout(title_text="코로나19 확진자 수가 가장 많은 10개 주: 아시아인 대상 혐오 범죄 장소의 공개성 변화")
#fig_asian_locationtype_death_top10.show()

#장소 공개성 - 하위10개
fig_asian_locationtype_bottom10 = fig_states_asian(location_asian, location_list_asian_bottom_2019, location_list_asian_bottom_2020)
fig_asian_locationtype_bottom10.update_layout(title_text="코로나19 확진자 수가 가장 적은 10개 주: 아시아인 대상 혐오 범죄 장소의 공개성 변화")
#fig_asian_locationtype_death_bottom10.show()


################아시아인 대상 혐오범죄만, 사망자수 기준으로 다시.##################
df_state_asian = df_deaths_state[df_deaths_state['BIAS_DESC'].str.contains('Anti-Asian', na = False)] 
offense_list_asian = list(df_state_asian['OFFENSE_NAME'].unique())
offense_list_asian.sort()
#print(offense_list)

top10_asian_df = df_state_asian[df_state_asian['STATE_NAME'] == 'TOP_10_COVID_DEATHS']
top10_asian_2019 = top10_asian_df[top10_asian_df['DATA_YEAR'] == 2019]
top10_asian_2020 = top10_asian_df[top10_asian_df['DATA_YEAR'] == 2020]

bottom10_asiandf = df_state_asian[df_state_asian['STATE_NAME'] == 'BOTTOM_10_COVID_DEATHS']
bottom10_asian2019 = bottom10_asiandf[bottom10_asiandf['DATA_YEAR'] == 2019]
bottom10_asian2020 = bottom10_asian2020 = bottom10_asiandf[bottom10_asiandf['DATA_YEAR'] == 2020]


offense_death_list_asian_top_2019 = []
offense_death_list_asian_bottom_2019 = []
offense_death_list_asian_top_2020 = []
offense_death_list_asian_bottom_2020 = []

for item in offense_list_asian:
    offense_death_list_asian_top_2019.append(len(top10_asian_2019[top10_asian_2019['OFFENSE_NAME'] == item]))
    offense_death_list_asian_top_2020.append(len(top10_asian_2020[top10_asian_2020['OFFENSE_NAME'] == item]))
    offense_death_list_asian_bottom_2019.append(len(bottom10_asian2019[bottom10_asian2019['OFFENSE_NAME'] == item]))
    offense_death_list_asian_bottom_2020.append(len(bottom10_asian2020[bottom10_asian2020['OFFENSE_NAME'] == item]))


location_list_asian = list(df_state_asian['LOCATION_NAME'].unique())
location_list_asian.sort()
# print(location_list_asian)

location_death_list_asian_top_2019 = []
location_death_list_asian_bottom_2019 = []
location_death_list_asian_top_2020 = []
location_death_list_asian_bottom_2020 = []

for item in location_list_asian:
    location_death_list_asian_top_2019.append(len(top10_asian_2019[top10_asian_2019['LOCATION_NAME'] == item]))
    location_death_list_asian_top_2020.append(len(top10_asian_2020[top10_asian_2020['LOCATION_NAME'] == item]))
    location_death_list_asian_bottom_2019.append(len(bottom10_asian2019[bottom10_asian2019['LOCATION_NAME'] == item]))
    location_death_list_asian_bottom_2020.append(len(bottom10_asian2020[bottom10_asian2020['LOCATION_NAME'] == item]))

#표 세트 4: 사망자 수 상위/하위 10개 주 기준 아시아인 대상 혐오 범죄 유형 과격성/장소 공개성 비교
@st.cache(allow_output_mutation=True)
def fig_states_asian_death(off_or_loc, death_list_2019, death_list_2020):
    fig_states = go.Figure()
    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2019,
    name = "2019",
    ))

    fig_states.add_trace(go.Bar(
    x = off_or_loc,
    y = death_list_2020,
    name = "2020",
    ))

    return fig_states


fig_asian_offensetype_death_top10 = fig_states_asian_death(offense_list_asian, offense_death_list_asian_top_2019, offense_death_list_asian_top_2020)
fig_asian_offensetype_death_top10.update_layout(title_text="코로나19 사망자 수가 가장 많은 10개 주: 아시아인 대상 혐오 범죄 유형 과격성 변화")
#fig_asian_offensetype_death_top10.show()

#유형 과격성 - 하위10개
fig_asian_offensetype_death_bottom10 = fig_states_asian_death(offense_list_asian, offense_death_list_asian_bottom_2019, offense_death_list_asian_bottom_2020)
fig_asian_offensetype_death_bottom10.update_layout(title_text="코로나19 사망자 수가 가장 적은 10개 주: 아시아인 대상 혐오 범죄 유형 과격성 변화")
#fig_asian_offensetype_death_bottom10.show()

#장소 공개성 - 상위10개
fig_asian_locationtype_death_top10 = fig_states_asian_death(location_list_asian, location_death_list_asian_top_2019, location_death_list_asian_top_2020)
fig_asian_locationtype_death_top10.update_layout(title_text="코로나19 사망자 수가 가장 많은 10개 주: 아시아인 대상 혐오 범죄 장소의 공개성 변화")
#fig_asian_locationtype_death_top10.show()

#장소 공개성 - 하위10개
fig_asian_locationtype_death_bottom10 = fig_states_asian_death(location_list_asian, location_death_list_asian_bottom_2019, location_death_list_asian_bottom_2020)
fig_asian_locationtype_death_bottom10.update_layout(title_text="코로나19 사망자 수가 가장 적은 10개 주: 아시아인 대상 혐오 범죄 장소의 공개성 변화")
#fig_asian_locationtype_death_bottom10.show()

st.markdown("""
### 아시아인 혐오 범죄 유형의 과격성과 심각성 - 상위, 하위 10개주별 비교
* 그렇다면 아시아인 대상 혐오범죄만 상위 10개 주, 하위 10개 주별로 비교한 결과는 어떨까요?
""")

order2 = ['확진자 수 기준', '사망자 수 기준']
status2 = st.radio('확인 기준을 선택하세요', order2)

if status2 == '확진자 수 기준':
    st.markdown("""
    #### 1. 범죄 유형의 과격성
    """)
    st.plotly_chart(fig_asian_offensetype_top10)
    st.plotly_chart(fig_asian_offensetype_bottom10)

    st.markdown("""
    ##### 분석
    * 분석분석
    """)
        
    st.markdown("""
    #### 2. 범죄 장소의 공개성
    """)
    st.plotly_chart(fig_asian_locationtype_top10)
    st.plotly_chart(fig_asian_locationtype_bottom10)

    st.markdown("""
    ##### 분석
    * 분석분석
    ----
    """)
elif status2 == '사망자 수 기준':
    st.markdown("""
    #### 1. 범죄 유형의 과격성
    """)
    st.plotly_chart(fig_asian_offensetype_death_top10)
    st.plotly_chart(fig_asian_offensetype_death_bottom10)

    st.markdown("""
    ##### 분석
    * 코로나19 사망자 기준 상위 10개 주에서는 아시아인을 대상으로 한 과격성 1~2의 경범죄, 과격성 3의 중범죄가 모두 증가했습니다.
    * 반면 코로나 19 사망자가 적었던 10개 주에서는 과격성 3인 혐오 범죄가 줄어들었고, 1~2의 경범죄에서만 유의미한 증가세가 보였습니다. 
    """)
        
    st.markdown("""
    #### 2. 범죄 장소의 공개성
    """)
    st.plotly_chart(fig_asian_locationtype_death_top10)
    st.plotly_chart(fig_asian_locationtype_death_bottom10)

    st.markdown("""
    ##### 분석
    * 코로나19 사망자 수가 가장 적은 10개주에서는 공개성 6에 해당하는 혐오 범죄가 아시아인 대상으로 전혀 일어나지 않았습니다.
    * 공개성 4 혐오 범죄 증가율은 코로나19 사망자 수가 가장 적은 10개주보다 코로나19 사망자 수가 가장 많은 10개주에서 더 유의미하게 높았습니다.
    ----
    """)


st.markdown("""
##### 소결론
* 아시아인 혐오 범죄로 국한해 봤을 때도, 상위 10개주와 하위 10개주에서 발생한 혐오 범죄의 특성을 비교해봤을 때 코로나19 피해가 심했던 지역에서 범죄의 과격성이 더욱 높게 나타났습니다.
* 전체 혐오 범죄를 대상으로 했을 때만큼은 아니지만, 장소의 공개성 또한 상위 10개주에서 하위 10개주보다 높은 증가율을 보였습니다.
----
""")

#연설문 워드클라우드
st.markdown("""
### 트럼프 변수가 아시아인 혐오 범죄에 준 영향은?
* 소수 인종을 향한 혐오를 직접적으로 드러내온 트럼프 전 대통령의 발언을 분석함으로써 일명 ‘트럼프 변수’와 아시안 혐오 범죄의 상관성을 알아보고자 했습니다.
이에 트럼프 대통령의 연설문에서 혐오 감정이 드러나는 키워드가 얼마나 자주 등장하는지와 혐오 범죄 실제 발생 건수를 연결지어 시각화했습니다. 
* 다음은 트럼프 전 대통령의 연설문을 바이든 현 대통령의 연설문과 대조해 봤을 때 혐오를 조장하는 단어가 얼마나 자주 등장했는지를 빈도수 분석을 통해 나타낸 결과입니다.
""")

from bs4 import BeautifulSoup
import random
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import urllib.request

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

#words filtered
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!',"''",'``', ':', ';', '’', '(', ')','`','[', ']','--','–', '{', '}','_', "'\\n", "\n", "—", '#', '###']) #계속 업데이트하며 필터링


@st.cache(suppress_st_warning=True)
def speech_keywords_counter(url):
    doc = ""
    with urllib.request.urlopen(url) as url:
        doc = url.read()

    soup = BeautifulSoup(doc, "html.parser")
    speech = soup.find_all("p", class_="paragraph inline-placeholder")


    sentences = []
    for i in range(len(speech)):
        sentences.append(speech[i].text.strip())

    lemma = nltk.wordnet.WordNetLemmatizer()
    speech_tokens = []
    for sentence in sentences:
        speech_tokens.append(nltk.word_tokenize(sentence))

    speech_lemma = []

    for token in speech_tokens :
        for tok in token:
            speech_lemma.append(lemma.lemmatize(tok))

    #stopwords 위에 해둠       
    words_filtered = [word.lower() for word in speech_lemma if word.lower() not in stop_words]
    speech_cnt = Counter(words_filtered)

    return speech_cnt



############### 2017 #################
url_2017 = "https://edition.cnn.com/2017/02/28/politics/donald-trump-speech-transcript-full-text/index.html"
word_2017 = pd.DataFrame(speech_keywords_counter(url_2017 ).most_common())

##############2018##############
url_2018 = "https://edition.cnn.com/2018/01/30/politics/2018-state-of-the-union-transcript/index.html"
word_2018 = pd.DataFrame(speech_keywords_counter(url_2018).most_common())

##############2019##############
url_2019 = "https://edition.cnn.com/2019/02/05/politics/donald-trump-state-of-the-union-2019-transcript/index.html"
word_2019 = pd.DataFrame(speech_keywords_counter(url_2019).most_common())

##############2020##############
url_2020 = "https://edition.cnn.com/2020/02/04/politics/trump-2020-state-of-the-union-address/index.html"
word_2020 = pd.DataFrame(speech_keywords_counter(url_2020).most_common())

###################2021################# 얘만 링크가 달라서 다르게 함..
url_2021 = "https://www.whitehouse.gov/briefing-room/speeches-remarks/2022/03/01/remarks-of-president-joe-biden-state-of-the-union-address-as-delivered/"
doc_2021 = ""
with urllib.request.urlopen(url_2021) as url:
    doc_2021 = url.read()

soup = BeautifulSoup(doc_2021, "html.parser")

s_2021 = soup.find_all("p")

sentences_2021 = []
for i in range(len(s_2021)):
    sentences_2021.append(s_2021[i].text.strip().replace('\xa0', ''))

#sentences_2021

lemma = nltk.wordnet.WordNetLemmatizer()
s_2021_tokens = []
for sentence in sentences_2021:
    s_2021_tokens.append(nltk.word_tokenize(sentence))

s_2021_lemma = []

for token in s_2021_tokens :
    for tok in token:
        s_2021_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2021 = [word.lower() for word in s_2021_lemma if word.lower() not in stop_words]

s_2021_cnt = Counter(words_filtered_2021)
word_2021 = pd.DataFrame(s_2021_cnt.most_common())

##############2022##############

url_2022 = "https://edition.cnn.com/2022/03/01/politics/biden-state-of-the-union-2022-transcript/index.html"
word_2022 = pd.DataFrame(speech_keywords_counter(url_2022).most_common())

select_year = st.selectbox('확인할 연도를 선택하세요', ('2017', '2018', '2019', '2020', '2021', '2022'))

if select_year == '2017':
    st.write(word_2017)
elif select_year == '2018':
    st.write(word_2018)
elif select_year == '2019':
    st.write(word_2018)
elif select_year == '2020':
    st.write(word_2018)
elif select_year == '2021':
    st.write(word_2018)
elif select_year == '2022':
    st.write(word_2018)


#추가 필요한 표: 주요 워드 빈도수 분석 표 등장
#혐오범죄 아시안 혐오 범죄 관련 단어 막대그래프 모두 있어야 할 것 같음
#연설문 나온 기간의 범죄 수 선그래프와 막대그래프가 같은 면에 있다면 너무 좋겠지만..어떻게든 둘다 있기만 하면 될 것 같음.

st.markdown("""
##### 예상 분석
* 트럼프 기점으로 혐오 범죄 관련 단어 수/범죄 수 모두 증가했을 것
* 아시안 범죄까지 유의미하게 증가하지는 않았을 것 같음. 오히려 이슬람 관련 차별이 훨 많이 등장했을 것 같고(terrorist, isis 등) 그래서 맨 처음 등장했던 원그래프의 정보를 여기다 응용하는 게 오히려 좋지 않을까 싶은 생각도(선택 사항: 없어도 됨).
* 바이든의 경우, 혐오를 직접적으로 드러내는 단어는 없어도 아시안 언급은 좀 있을 것 같음. 
* 트럼프 쪽이 더 많다면 상관없지만, 빈도수를 봤을 때 만약 바이든 연설문에서 아시아인 언급이 더 많이 나왔다면, \n
* 혐오 감정 및 혐오 범죄 조장 맥락에서 단어가 쓰이지 않았고, 코로나19와 같이 등장했다는 걸 강조해서 아시아인 혐오에 유감을 표시하는 용으로만 등장했다고 일축할 수 있을 듯.
----
""")


st.markdown("""
##### 예상 최종 결론
* 트럼프 당선 기점으로 혐오 범죄 수가 전반적으로 증가했고, 아시안 혐오 범죄는 코로나19를 기점으로 급증했다. 전반적으로 코로나19 발생 전후, 첫 발생지가 중국이었다는 점에서 아시아인 대상 혐오 범죄가 급증했던 것으로 예상된다.
* 이는 코로나19로 인한 피해가 심했던 상위 10개 주와 덜했던 하위 10개 주와의 비교에서 더욱 극명하게 드러났다. 
우선 전체 혐오 범죄의 건수는 상위 10개 주보다 하위 10개 주가 더 많았던 반면, 아시아인 대상 혐오 범죄의 건수는 상위 10개주에서 훨씬 높았다.
아시아인 혐오 범죄의 과격성 또한 상위 10개주에서 하위 10개주보다 높았다. 범죄 장소의 공개성 또한 코로나19 피해가 심했던 상위 10개주에서 더욱 유의미하게 증가했다. 
* 마무리말(어떻게 키워드 결과가 나오든): 트럼프 기점으로 ‘전체 혐오 범죄’ 증가, 코로나19 기점으로 ‘아시아인 대상 혐오 범죄’ 유의미한 증가. 
""")
