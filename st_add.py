"""
지금까지 쓴 코드 1차 수합 및 스트림릿으로..
""" 

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px #일단 이거 이용해 기본 그래프 그림

st.set_page_config(page_icon="🗽", page_title="데이터저널리즘 2조")

st.markdown("""
# 아시아인 혐오 범죄, 확산과 심화: 코로나19 전후 미국의 혐오 범죄를 조명하다
* 데이터저널리즘 2조: 오소영, 이혜정, 최다연
#### 코로나19 팬데믹과 트럼프의 등장은 아시아인 혐오 범죄에 어떤 영향을 미쳤는지, 그 상관성을 중심으로 범죄의 심각성을 조명했다💡
----
""")

fbi =  pd.read_csv("fbi_data_edited.csv", low_memory=False)
    
df = pd.DataFrame(fbi, columns=['DATA_YEAR', 'BIAS_DESC'])

#소영언니 파트: 다른 인종에 대한 범죄도 늘었는가, 아시아에 대한 범죄만 늘었는가(범죄 성격)
st.subheader("2015-2020 미국 전역 혐오 범죄 통계 : 대상 인종별 비율")
options_1 = st.radio('연도를 선택하세요', ['2015', '2016', '2017', '2018', '2019', '2020'])
if options_1 == "2015":
    crime_data_count = pd.read_csv("hate_crime_race.csv")
    crime_data_count['number'].astype(int)
    cdc_2015 = crime_data_count['number'].tolist() #대상 인종별 범죄수 담은 리스트.
    idx_2015 = list(crime_data_count['race'])

    #df_2015_edit = pd.DataFrame(cdc_2015, index = idx_2015) 확인용 데이터프레임: 대상 인종 index, 인종별 범죄수 columns
    #st.DataFrame(df_2015_edit)

    fig_2015 = px.pie(crime_data_count, values = 'number', names = 'race', title = '2015')
    st.plotly_chart(fig_2015)

elif options_1 == "2016":
    crime_data_count_16 = pd.read_csv("hate_crime_race_2016.csv")
    crime_data_count_16['number'].astype(int)
    cdc_2016 = list(crime_data_count_16['number']) #대상 인종별 범죄수 담은 리스트.
    idx_2016 = list(crime_data_count_16['race'])

    # df_2016_edit = pd.DataFrame(cdc_2016, index = idx_2016)
    # st.DataFrame(df_2016_edit)

    fig_2016 = px.pie(crime_data_count_16, values = 'number', names = 'race', title = '2016')
    st.plotly_chart(fig_2016)


elif options_1 == "2017":
    crime_data_count_17 = pd.read_csv("hate_crime_race_2017.csv")
    crime_data_count_17['number'].astype(int)
    cdc_2017 = list(crime_data_count_17['number']) #대상 인종별 범죄수 담은 리스트.
    idx_2017 = list(crime_data_count_17['race'])

    #df_2017_edit = pd.DataFrame(cdc_2017, index = idx_2017)
    #st.DataFrame(df_2017_edit)

    fig_2017 = px.pie(crime_data_count_17, values = 'number', names = 'race', title = '2017')
    st.plotly_chart(fig_2017)

elif options_1 == "2018":
    crime_data_count_18 = pd.read_csv("hate_crime_race_2018.csv")
    crime_data_count_18['number'].astype(int)
    cdc_2018 = list(crime_data_count_18['number']) #대상 인종별 범죄수 담은 리스트.
    idx_2018 = list(crime_data_count_18['race'])

    #df_2018_edit = pd.DataFrame(cdc_2018, index = idx_2018)
    #st.DataFrame(df_2018_edit)

    fig_2018 = px.pie(crime_data_count_18, values = 'number', names = 'race', title = '2018')
    st.plotly_chart(fig_2018)


elif options_1 == "2019":
    crime_data_count_19 = pd.read_csv("hate_crime_race_2019.csv")
    crime_data_count_19['number'].astype(int)
    cdc_2019 = list(crime_data_count_19['number']) #대상 인종별 범죄수 담은 리스트.
    idx_2019 = list(crime_data_count_19['race'])

    #df_2019_edit = pd.DataFrame(cdc_2019, index = idx_2019)
    #st.DataFrame(df_2019_edit)

    fig_2019 = px.pie(crime_data_count_19, values = 'number', names = 'race', title = '2019')
    st.plotly_chart(fig_2019)

elif options_1 == "2020":
    crime_data_count_20 = pd.read_csv("hate_crime_race_2020.csv")
    crime_data_count_20['number'].astype(int)
    cdc_2020 = list(crime_data_count_20['number']) #대상 인종별 범죄수 담은 리스트.
    idx_2020 = list(crime_data_count_20['race'])

    #df_2020_edit = pd.DataFrame(cdc_2020, index = idx_2020)
    #st.DataFrame(df_2020_edit)

    fig_2020 = px.pie(crime_data_count_20, values = 'number', names = 'race', title = '2020')
    st.plotly_chart(fig_2020)


#Anti Asian 증가만 보여주는 그래프
st.subheader("2015년에서 2020년까지, 미국 전역 혐오 범죄 건수 변화")
options_2 = st.radio('옵션을 선택하세요', ['전체 혐오 범죄', '아시아인 대상 혐오 범죄', '전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율'])
df_2015 = df[(df['DATA_YEAR'] == 2015)]
df_2016 = df[(df['DATA_YEAR'] == 2016)]
df_2017 = df[(df['DATA_YEAR'] == 2017)]
df_2018 = df[(df['DATA_YEAR'] == 2018)]
df_2019 = df[(df['DATA_YEAR'] == 2019)]
df_2020 = df[(df['DATA_YEAR'] == 2020)]

#새로운 df: 연도별 전체 / 아시아인 대상 혐오범죄 count. index: 연도, columns: 값
all_hates = [len(df_2015['BIAS_DESC']), len(df_2016['BIAS_DESC']), len(df_2017['BIAS_DESC']), len(df_2018['BIAS_DESC']),
            len(df_2019['BIAS_DESC']), len(df_2020['BIAS_DESC'])]
df_all = pd.DataFrame(all_hates, index = ['2015', '2016', '2017', '2018', '2019', '2020'])

all_fig = px.line(df_all, title = "2015-2020 All Hate Crime Cases in the US", markers=True)
all_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Counted Cases"}))

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

print(all_asian_ratio)

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
**결론 정리~**
* 혐오 범죄 자체도 늘어났고 
* 아시아인 대상 혐오 범죄 수와 비율 모두 2020년에 확 늘어남~
-----
""")

#가능하면 scatterplot으로 코로나19 case 증가 - 아시아인 혐오 범죄 증가 사이 상관관계 그리면 좋을 듯.

#혜정 파트: Asian hate crime 
# 범죄 유형이 과격해지지 않았는가 
# 장소의 공개성이 증가했는가
st.subheader("아시아인 대상 혐오 범죄 양상 변화")
st.markdown("""
##### 1. 범죄 유형의 과격성 변화\n2019년(코로나19 발생 이전)과 2020년(코로나19 발생 이후) 비교
* 1-4로 갈수록 과격성 심화, 5는 기타
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
df1.loc[df1['OFFENSE_NAME'] == 'Robbery', 'OFFENSE_NAME'] = 2
df1.loc[df1['OFFENSE_NAME'] == 'Burglary/Breaking & Entering', 'OFFENSE_NAME'] = 2
df1.loc[df1['OFFENSE_NAME'] == 'Destruction/Damage/Vandalism of Property', 'OFFENSE_NAME'] = 2
df1.loc[df1['OFFENSE_NAME'] == 'Arson', 'OFFENSE_NAME'] = 2

df1.loc[df1['OFFENSE_NAME'] == 'Simple Assault', 'OFFENSE_NAME'] = 3
df1.loc[df1['OFFENSE_NAME'] == 'Aggravated Assault', 'OFFENSE_NAME'] = 3

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

# =====================================

st.markdown("""
##### 2. 범죄 장소의 공개성 변화\n2019년(코로나19 발생 이전)과 2020년(코로나19 발생 이후) 비교
* 1-5로 갈수록 공개성 높아짐, 6은 기타
""")

# crime place cleaning
df1.loc[df1['LOCATION_NAME'] == 'Residence/Home', 'LOCATION_NAME'] = 1
df1.loc[df1['LOCATION_NAME'] == 'Hotel/Motel/Etc.', 'LOCATION_NAME'] = 1

df1.loc[df1['LOCATION_NAME'] == 'Bar/Nightclub', 'LOCATION_NAME'] = 2
df1.loc[df1['LOCATION_NAME'] == 'Auto Dealership New/Used', 'LOCATION_NAME'] = 2
df1.loc[df1['LOCATION_NAME'] == 'Restaurant', 'LOCATION_NAME'] = 2

df1.loc[df1['LOCATION_NAME'] == 'Liquor Store', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Convenience Store', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Specialty Store', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Department/Discount Store', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Shopping Mall', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Grocery/Supermarket', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Air/Bus/Train Terminal', 'LOCATION_NAME'] = 3
df1.loc[df1['LOCATION_NAME'] == 'Commercial/Office Building', 'LOCATION_NAME'] = 3

df1.loc[df1['LOCATION_NAME'] == 'School-College/University', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'School-Elementary/Secondary', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'School/College', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Community Center', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Shelter-Mission/Homeless', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Bank/Savings and Loan', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Church/Synagogue/Temple/Mosque', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Government/Public Building', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == 'Jail/Prison/Penitentiary/Corrections Facility', 'LOCATION_NAME'] = 4
df1.loc[df1['LOCATION_NAME'] == "Drug Store/Doctor's Office/Hospital", 'LOCATION_NAME'] = 4

df1.loc[df1['LOCATION_NAME'] == 'Field/Woods', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Lake/Waterway/Beach', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Park/Playground', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk;Residence/Home', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Parking/Drop Lot/Garage', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Rest Area', 'LOCATION_NAME'] = 5
df1.loc[df1['LOCATION_NAME'] == 'Service/Gas Station', 'LOCATION_NAME'] = 5

df1.loc[df1['LOCATION_NAME'] == 'Cyberspace', 'LOCATION_NAME'] = 6
df1.loc[df1['LOCATION_NAME'] == 'Industrial Site', 'LOCATION_NAME'] = 6
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

st.markdown("""
##### 2019와 2020 아시아인 대상 혐오 범죄 컬러맵
순서가 이게 맞나?
""")

# Colormap of crime at 2019
df_2019 = df1[(df1['DATA_YEAR'] == 2019)] 
df_2019['INCIDENT_DATE'] = pd.to_datetime(df_2019['INCIDENT_DATE'])
df_2019['INCIDENT_day'] = df_2019['INCIDENT_DATE'].dt.day
df_2019['INCIDENT_month_name'] = df_2019['INCIDENT_DATE'].dt.month_name()

df_2019_c = pd.crosstab(df_2019.INCIDENT_day, df_2019.INCIDENT_month_name)


df_2019_fig = px.imshow(df_2019_c, color_continuous_scale=px.colors.sequential.Reds, title="Asian Hate crimes at 2019")
df_2019_fig.update_layout(xaxis = dict({"title" : "Month"}), yaxis = dict({"title" : "Day"}))
st.plotly_chart(df_2019_fig)

# Colormap of crime at 2020
df_2020 = df1[(df1['DATA_YEAR'] == 2020)] 
df_2020['INCIDENT_DATE'] = pd.to_datetime(df_2020['INCIDENT_DATE'])
df_2020['INCIDENT_day'] = df_2020['INCIDENT_DATE'].dt.day
df_2020['INCIDENT_month_name'] = df_2020['INCIDENT_DATE'].dt.month_name()

df_2020_c = pd.crosstab(df_2020.INCIDENT_day, df_2020.INCIDENT_month_name)


df_2020_fig = px.imshow(df_2020_c, color_continuous_scale=px.colors.sequential.Reds, title="Asian Hate crimes at 2020")
df_2020_fig.update_layout(xaxis = dict({"title" : "Month"}), yaxis = dict({"title" : "Day"}))
st.plotly_chart(df_2020_fig)

st.markdown("""##### 결론 정리 어쩌구저쩌구
* 근데 이거 2019 2020 비슷해보이는 것은 나만의 착각일까?
-----
""")

#다연 파트: 코로나의 심각성에 따른 비교- 주별 비교

st.markdown("""
## 2020년 기준 코로나19 심각성 보여주는 자료
""")

#파일 임포트 
covid_state = pd.read_csv("us_county_covid_2020.csv") #2020년 12월 31일 기준, 미국 각 주 county별 누적 확진 수를 담은 파일.

#주별 인구 10만명 당 코로나19 확진자 수 표 그리기
covid_state_df = pd.DataFrame(covid_state)

#주별 누적 확진 건수 추출하는 함수
def state_cases(state_name):
    state = covid_state_df.loc[covid_state_df['state'] == state_name]
    return state['cases'].sum()

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

#지도 데이터 불러와서 시각화 레츠고.
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

#지도 데이터 불러와서 시각화 레츠고.
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

#주별 혐오범죄 유형 과격성 표: 2019, 2020

#주별 혐오범죄 장소 공개성 표: 2019, 2020


st.markdown("""
## 코로나19 심각성에 따른 혐오 범죄 발생 양상, 주별 비교
* 2020년 말까지, 인구 10만 명 당 코로나19 사망자 수가 평균 이상인 2개 주(캘리포니아, 워싱턴)와 펑균 이하인 2개 주(뉴욕, 뉴저지) ← 확인 필요 아직도 헷갈림;
##### 1. 범죄 유형의 과격성 변화\n2019년(코로나19 발생 이전)과 2020년(코로나19 발생 이후) 비교
* 1-4로 갈수록 공개성 높아짐, 5는 기타
""")

# 범죄 유형이 과격해지지 않았는가 
df_dy = pd.DataFrame(fbi, columns=['STATE_NAME', 'DATA_YEAR', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME']) #raw fbi 변수는 fbi로, 내가 쓴 df 컬럼 달라서 df_dy로

#2020
california_2020 = df_dy.loc[(fbi["STATE_NAME"] == "California") & (fbi["DATA_YEAR"] == 2020)] #캘리포니아
newjersey_2020 = df_dy.loc[(fbi["STATE_NAME"] == "New Jersey") & (fbi["DATA_YEAR"] == 2020)] #뉴저지
newyork_2020 = df_dy.loc[(fbi["STATE_NAME"] == "New York") & (fbi["DATA_YEAR"] == 2020)] #뉴욕
washington_2020 = df_dy.loc[(fbi["STATE_NAME"] == "Washington") & (fbi["DATA_YEAR"] == 2020)] #워싱턴

#2020 df of 4 states
cases_2020 = [newyork_2020, newjersey_2020,  california_2020, washington_2020] #적은 주 2 많은 주 2
cases_2020_pd = pd.concat(cases_2020)
#cases_2020_pd

cases_2020_pd['OFFENSE_NAME'].unique()

#범죄 유형 분류 cleanup 1: 컬럼 전체적으로 합치기
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.startswith('Aggravated Assault')), 'OFFENSE_NAME'] = 'Aggravated Assault'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.startswith('Destruction/Damage/Vandalism of Property')), 'OFFENSE_NAME'] = 'Destruction/Damage/Vandalism of Property'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.startswith('Intimidation')), 'OFFENSE_NAME'] = 'Intimidation'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.startswith('Burglary/Breaking & Entering')), 'OFFENSE_NAME'] = 'Burglary/Breaking & Entering'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.contains('Drug')), 'OFFENSE_NAME'] = 'Drug Violations'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.contains('Theft')), 'OFFENSE_NAME'] = 'Theft'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.contains('Robbery')), 'OFFENSE_NAME'] = 'Robbery'
cases_2020_pd.loc[(cases_2020_pd['OFFENSE_NAME'].str.contains('Sexual Assault')), 'OFFENSE_NAME'] = 'Sexual Assault'


#cases_2020_pd['OFFENSE_NAME'].unique()

#범죄 유형별 발생 수 표: 주요 4개 주
off_2020 = pd.crosstab(index = cases_2020_pd.STATE_NAME, columns = cases_2020_pd.OFFENSE_NAME)
# off_2020

#범죄 유형 분류 cleanup 2: 컬럼 합치기 
off_2020["misdemeanor_1"] = off_2020['Intimidation'] + off_2020['Drug Violations'] + off_2020['False Pretenses/Swindle/Confidence Game'] + off_2020['Weapon Law Violations'] + off_2020['Animal Cruelty'] + off_2020['Shoplifting'] + off_2020['Impersonation']
off_2020["larceny_2"] = off_2020['All Other Larceny'] + off_2020['Theft'] + off_2020['Robbery'] + off_2020['Burglary/Breaking & Entering'] + off_2020['Destruction/Damage/Vandalism of Property']
off_2020["assault_3"] = off_2020['Fondling'] + off_2020['Sexual Assault'] + off_2020['Arson'] + off_2020['Simple Assault'] + off_2020['Aggravated Assault']
off_2020["felony_4"] = off_2020['Murder and Nonnegligent Manslaughter']
off_2020["etc_5"] = off_2020['Not Specified']

off_2020_edited = off_2020.loc[:, ['misdemeanor_1', 'larceny_2' , 'assault_3', 'felony_4', 'etc_5']]
# off_2020_edited

#2019년도 같은 방식으로


california_2019 = df_dy.loc[(fbi["STATE_NAME"] == "California") & (fbi["DATA_YEAR"] == 2019)] #캘리포니아
newjersey_2019 = df_dy.loc[(fbi["STATE_NAME"] == "New Jersey") & (fbi["DATA_YEAR"] == 2019)] #뉴저지
newyork_2019 = df_dy.loc[(fbi["STATE_NAME"] == "New York") & (fbi["DATA_YEAR"] == 2019)] #뉴욕
washington_2019 = df_dy.loc[(fbi["STATE_NAME"] == "Washington") & (fbi["DATA_YEAR"] == 2019)] #워싱턴

#2019 df of 4 states
cases_2019 = [newyork_2019, newjersey_2019,  california_2019, washington_2019] #적은 주 2 많은 주 2
cases_2019_pd = pd.concat(cases_2019)
# cases_2019_pd
# cases_2019_pd['OFFENSE_NAME'].unique()

#범죄 유형 분류 cleanup 1: 컬럼 전체적으로 합치기
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.startswith('Aggravated Assault')), 'OFFENSE_NAME'] = 'Aggravated Assault'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.startswith('Destruction/Damage/Vandalism of Property')), 'OFFENSE_NAME'] = 'Destruction/Damage/Vandalism of Property'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.startswith('Intimidation')), 'OFFENSE_NAME'] = 'Intimidation'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.startswith('Burglary/Breaking & Entering')), 'OFFENSE_NAME'] = 'Burglary/Breaking & Entering'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.contains('Drug')), 'OFFENSE_NAME'] = 'Drug Violations'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.contains('Theft')), 'OFFENSE_NAME'] = 'Theft'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.contains('Robbery')), 'OFFENSE_NAME'] = 'Robbery'
cases_2019_pd.loc[(cases_2019_pd['OFFENSE_NAME'].str.contains('Sexual Assault')), 'OFFENSE_NAME'] = 'Sexual Assault'


#cases_2019_pd['OFFENSE_NAME'].unique() #일부 종류 차이 있음, 명시 필요

#범죄 유형별 발생 수 표: 주요 4개 주
off_2019 = pd.crosstab(index = cases_2019_pd.STATE_NAME, columns = cases_2019_pd.OFFENSE_NAME)
# off_2019
# off_2019.columns.unique()

#범죄 유형 분류 cleanup 2: 컬럼 합치기 
off_2019["misdemeanor_1"] = off_2019['Intimidation'] + off_2019['Drug Violations'] + off_2019['Weapon Law Violations'] + off_2019['Shoplifting']
off_2019["larceny_2"] = off_2019['All Other Larceny'] + off_2019['Theft'] + off_2019['Robbery'] + off_2019['Burglary/Breaking & Entering'] + off_2019['Destruction/Damage/Vandalism of Property']
off_2019["assault_3"] = off_2019['Fondling'] + off_2019['Sexual Assault'] + off_2019['Arson'] + off_2019['Simple Assault'] + off_2019['Aggravated Assault'] + off_2019['Kidnapping/Abduction;Simple Assault']
off_2019["felony_4"] = off_2019['Rape'] + off_2019['Murder and Nonnegligent Manslaughter']
off_2019["etc_5"] = off_2019['Not Specified']

off_2019_edited = off_2019.loc[:, ['misdemeanor_1', 'larceny_2' , 'assault_3', 'felony_4', 'etc_5']]
# off_2019_edited
# off_2019_edited['misdemeanor_1']
# off_2020_edited['misdemeanor_1']


#위의 이상한 정리 바탕으로 그린 표! 주별로

import plotly.graph_objects as go #그룹 표 만들기 위해 임포트

crime_type_idx = list(off_2019_edited.columns)


#리스트의 인덱스로 주 구별한거다.... 캘리포니아
yr_2019_ca = [off_2019_edited['misdemeanor_1'].California, off_2019_edited['larceny_2'].California, off_2019_edited['assault_3'].California, off_2019_edited['felony_4'].California, off_2019_edited['etc_5'].California]
yr_2020_ca = [off_2020_edited['misdemeanor_1'].California, off_2020_edited['larceny_2'].California, off_2020_edited['assault_3'].California, off_2020_edited['felony_4'].California, off_2020_edited['etc_5'].California]

fig_ca = go.Figure(data = [go.Bar(name='2019', x = crime_type_idx, y = yr_2019_ca),go.Bar(name='2020', x = crime_type_idx, y = yr_2020_ca)])
fig_ca.update_layout(barmode='group', title='2019-2020 Offense Type Comparison: California')

#워싱턴
yr_2019_wa = [off_2019_edited['misdemeanor_1'][3], off_2019_edited['larceny_2'][3], off_2019_edited['assault_3'][3], off_2019_edited['felony_4'][3], off_2019_edited['etc_5'][3]]
yr_2020_wa = [off_2020_edited['misdemeanor_1'][3], off_2020_edited['larceny_2'][3], off_2020_edited['assault_3'][3], off_2020_edited['felony_4'][3], off_2020_edited['etc_5'][3]]

fig_wa = go.Figure(data = [go.Bar(name='2019', x = crime_type_idx, y = yr_2019_wa),go.Bar(name='2020', x = crime_type_idx, y = yr_2020_wa)])
fig_wa.update_layout(barmode='group', title='2019-2020 Offense Type Comparison: Washington')


#얘는 뉴욕
yr_2019_ny = [off_2019_edited['misdemeanor_1'][2], off_2019_edited['larceny_2'][2], off_2019_edited['assault_3'][2], off_2019_edited['felony_4'][2], off_2019_edited['etc_5'][2]]
yr_2020_ny = [off_2020_edited['misdemeanor_1'][2], off_2020_edited['larceny_2'][2], off_2020_edited['assault_3'][2], off_2020_edited['felony_4'][2], off_2020_edited['etc_5'][2]]

fig_ny = go.Figure(data = [go.Bar(name='2019', x = crime_type_idx, y = yr_2019_ny),go.Bar(name='2020', x = crime_type_idx, y = yr_2020_ny)])
fig_ny.update_layout(barmode='group', title='2019-2020 Offense Type Comparison: New York')

#뉴저지
yr_2019_nj = [off_2019_edited['misdemeanor_1'][1], off_2019_edited['larceny_2'][1], off_2019_edited['assault_3'][1], off_2019_edited['felony_4'][1], off_2019_edited['etc_5'][1]]
yr_2020_nj = [off_2020_edited['misdemeanor_1'][1], off_2020_edited['larceny_2'][1], off_2020_edited['assault_3'][1], off_2020_edited['felony_4'][1], off_2020_edited['etc_5'][1]]
fig_nj = go.Figure(data = [go.Bar(name='2019', x = crime_type_idx, y = yr_2019_nj),go.Bar(name='2020', x = crime_type_idx, y = yr_2020_nj)])
fig_nj.update_layout(barmode='group', title='2019-2020 Offense Type Comparison: New Jersey')

#표 보이기
options_state = st.radio('주를 선택하세요', ['California', 'Washington', 'New York', 'New Jersey'])
if options_state == 'California':
    st.plotly_chart(fig_ca)
elif options_state == 'Washington':
    st.plotly_chart(fig_wa)
elif options_state == 'New York':
    st.plotly_chart(fig_ny)
elif options_state == 'New Jersey':    
    st.plotly_chart(fig_nj)

# 장소의 공개성이 증가했는가
st.markdown("""
##### 2. 범죄 장소의 공개성 비교\n2019년(코로나19 발생 이전)과 2020년(코로나19 발생 이후) 비교
* 1-5로 갈수록 공개성 높아짐, 6은 기타
""")

df2 = pd.DataFrame(fbi, columns=['STATE_NAME', 'DATA_YEAR', 'INCIDENT_DATE', 'LOCATION_NAME', 'BIAS_DESC'])
df2 = df2[(df2['DATA_YEAR'] >= 2019)]
# st.write(df2)   

df2.loc[df2['LOCATION_NAME'] == 'Residence/Home', 'LOCATION_NAME'] = 1
df2.loc[df2['LOCATION_NAME'] == 'Hotel/Motel/Etc.', 'LOCATION_NAME'] = 1

df2.loc[df2['LOCATION_NAME'] == 'Bar/Nightclub', 'LOCATION_NAME'] = 2
df2.loc[df2['LOCATION_NAME'] == 'Auto Dealership New/Used', 'LOCATION_NAME'] = 2
df2.loc[df2['LOCATION_NAME'] == 'Restaurant', 'LOCATION_NAME'] = 2

df2.loc[df2['LOCATION_NAME'] == 'Liquor Store', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Convenience Store', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Specialty Store', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Department/Discount Store', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Shopping Mall', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Grocery/Supermarket', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Air/Bus/Train Terminal', 'LOCATION_NAME'] = 3
df2.loc[df2['LOCATION_NAME'] == 'Commercial/Office Building', 'LOCATION_NAME'] = 3

df2.loc[df2['LOCATION_NAME'] == 'School-College/University', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'School-Elementary/Secondary', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'School/College', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Community Center', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Shelter-Mission/Homeless', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Bank/Savings and Loan', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Church/Synagogue/Temple/Mosque', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Government/Public Building', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == 'Jail/Prison/Penitentiary/Corrections Facility', 'LOCATION_NAME'] = 4
df2.loc[df2['LOCATION_NAME'] == "Drug Store/Doctor's Office/Hospital", 'LOCATION_NAME'] = 4

df2.loc[df2['LOCATION_NAME'] == 'Field/Woods', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Lake/Waterway/Beach', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Park/Playground', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Highway/Road/Alley/Street/Sidewalk;Residence/Home', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Parking/Drop Lot/Garage', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Rest Area', 'LOCATION_NAME'] = 5
df2.loc[df2['LOCATION_NAME'] == 'Service/Gas Station', 'LOCATION_NAME'] = 5

df2.loc[df2['LOCATION_NAME'] == 'Cyberspace', 'LOCATION_NAME'] = 6
df2.loc[df2['LOCATION_NAME'] == 'Industrial Site', 'LOCATION_NAME'] = 6
df2.loc[df2['LOCATION_NAME'] == 'Other/Unknown', 'LOCATION_NAME'] = 6

#캘리포니아
df2_c = df2.loc[(df2["STATE_NAME"] == "California")]
df2_cross = pd.crosstab(df2_c.DATA_YEAR, df2_c.LOCATION_NAME)

df2_c_edit = df2_cross.loc[:,[1, 2, 3, 4, 5, 6]]

fig2_ca = px.bar(df2_c_edit, x = df2_c_edit.columns, y = ['2019', '2020'], title = 'Openness of Crime Location: California', orientation='h')
fig2_ca.update_layout(xaxis = {'title':'Crime Counts'}, yaxis = {'title': 'Year'})

#워싱턴
df2_w = df2.loc[(df2["STATE_NAME"] == "Washington")]
df2_w_cross = pd.crosstab(df2_w.DATA_YEAR, df2_w.LOCATION_NAME)

df2_w_edit = df2_w_cross.loc[:,[1, 2, 3, 4, 5, 6]]

fig2_wa = px.bar(df2_w_edit, x = df2_w_edit.columns, y = ['2019', '2020'], title = 'Openness of Crime Location: Washington', orientation='h')
fig2_wa.update_layout(xaxis = {'title':'Crime Counts'}, yaxis = {'title': 'Year'})

#뉴욕
df2_ny = df2.loc[(df2["STATE_NAME"] == "New York")]
df2_ny_cross = pd.crosstab(df2_ny.DATA_YEAR, df2_ny.LOCATION_NAME)

df2_ny_edit = df2_ny_cross.loc[:,[1, 3, 4, 5, 6]]

fig2_ny = px.bar(df2_ny_edit, x = df2_ny_edit.columns, y = ['2019', '2020'], title = 'Openness of Crime Location: New York', orientation='h')
fig2_ny.update_layout(xaxis = {'title':'Crime Counts'}, yaxis = {'title': 'Year'})

#뉴저지
df2_nj = df2.loc[(df2["STATE_NAME"] == "New Jersey")]
df2_nj_cross = pd.crosstab(df2_nj.DATA_YEAR, df2_nj.LOCATION_NAME)

df2_nj_edit = df2_nj_cross.loc[:,[1, 2, 3, 4, 5, 6]]

fig2_nj = px.bar(df2_nj_edit, x = df2_nj_edit.columns, y = ['2019', '2020'], title = 'Openness of Crime Location: New Jersey', orientation='h')
fig2_nj.update_layout(xaxis = {'title':'Crime Counts'}, yaxis = {'title': 'Year'})



#표 보이기
options_state_2 = st.radio('확인할 주를 선택하세요', ['California', 'Washington', 'New York', 'New Jersey'])
if options_state_2 == 'California':
    st.plotly_chart(fig2_ca)
elif options_state_2 == 'Washington':
    st.plotly_chart(fig2_wa)
elif options_state_2 == 'New York':
    st.plotly_chart(fig2_ny)
elif options_state_2 == 'New Jersey':    
    st.plotly_chart(fig2_nj)

st.markdown("""##### 결론 정리 어쩌구저쩌구
* 점점 산으로 가고 있는 느낌은 뭘까
* 내 파트 보완 필yoyoyo...
-----
""")

#연설문 워드클라우드
st.markdown("""
## 트럼프 2017-2020, 바이든 2021-2022 연설문 키워드
##### 아까 쓴 코드로 대신 넣어둠 !
""")


from bs4 import BeautifulSoup
import random
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

#words filtered
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!',"''",'``', ':', ';', '’', '(', ')','`','[', ']','--','–', '{', '}','_', "'\\n", "\n", "—", '#', '###']) #계속 업데이트하며 필터링

import urllib.request

url = "https://edition.cnn.com/2017/02/28/politics/donald-trump-speech-transcript-full-text/index.html"
doc = ""
with urllib.request.urlopen(url) as url:
    doc = url.read()

soup = BeautifulSoup(doc, "html.parser")

s_2017 = soup.find_all("p", class_="paragraph inline-placeholder")


sentences_2017 = []
for i in range(len(s_2017)):
    sentences_2017.append(s_2017[i].text.strip())


lemma = nltk.wordnet.WordNetLemmatizer()
s_2017_tokens = []
for sentence in sentences_2017:
    s_2017_tokens.append(nltk.word_tokenize(sentence))

s_2017_lemma = []

for token in s_2017_tokens :
    for tok in token:
        s_2017_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2017 = [word.lower() for word in s_2017_lemma if word.lower() not in stop_words]

s_2017_cnt = Counter(words_filtered_2017)
word_2017 = pd.DataFrame(s_2017_cnt.most_common())

##############2018##############

url_2018 = "https://edition.cnn.com/2018/01/30/politics/2018-state-of-the-union-transcript/index.html"
doc_2018 = ""
with urllib.request.urlopen(url_2018) as url:
    doc_2018 = url.read()

soup = BeautifulSoup(doc_2018, "html.parser")

s_2018 = soup.find_all("p", class_="paragraph inline-placeholder")


sentences_2018 = []
for i in range(len(s_2018)):
    sentences_2018.append(s_2018[i].text.strip())


lemma = nltk.wordnet.WordNetLemmatizer()
s_2018_tokens = []
for sentence in sentences_2018:
    s_2018_tokens.append(nltk.word_tokenize(sentence))

s_2018_lemma = []

for token in s_2018_tokens :
    for tok in token:
        s_2018_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2018 = [word.lower() for word in s_2018_lemma if word.lower() not in stop_words]

s_2018_cnt = Counter(words_filtered_2018)
word_2018 = pd.DataFrame(s_2018_cnt.most_common())

##############2019##############

url_2019 = "https://edition.cnn.com/2019/02/05/politics/donald-trump-state-of-the-union-2019-transcript/index.html"
doc_2019 = ""
with urllib.request.urlopen(url_2019) as url:
    doc_2019 = url.read()

soup = BeautifulSoup(doc_2019, "html.parser")

s_2019 = soup.find_all("p", class_="paragraph inline-placeholder")


sentences_2019 = []
for i in range(len(s_2019)):
    sentences_2019.append(s_2019[i].text.strip())


lemma = nltk.wordnet.WordNetLemmatizer()
s_2019_tokens = []
for sentence in sentences_2019:
    s_2019_tokens.append(nltk.word_tokenize(sentence))

s_2019_lemma = []

for token in s_2019_tokens :
    for tok in token:
        s_2019_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2019 = [word.lower() for word in s_2019_lemma if word.lower() not in stop_words]

s_2019_cnt = Counter(words_filtered_2019)
word_2019 = pd.DataFrame(s_2019_cnt.most_common())


##############2020##############

url_2020 = "https://edition.cnn.com/2020/02/04/politics/trump-2020-state-of-the-union-address/index.html"
doc_2020 = ""
with urllib.request.urlopen(url_2020) as url:
    doc_2020 = url.read()

soup = BeautifulSoup(doc_2020, "html.parser")

s_2020 = soup.find_all("p", class_="paragraph inline-placeholder")


sentences_2020 = []
for i in range(len(s_2020)):
    sentences_2020.append(s_2020[i].text.strip())


lemma = nltk.wordnet.WordNetLemmatizer()
s_2020_tokens = []
for sentence in sentences_2020:
    s_2020_tokens.append(nltk.word_tokenize(sentence))

s_2020_lemma = []

for token in s_2020_tokens :
    for tok in token:
        s_2020_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2020 = [word.lower() for word in s_2020_lemma if word.lower() not in stop_words]

s_2020_cnt = Counter(words_filtered_2020)
word_2020 = pd.DataFrame(s_2020_cnt.most_common())

###################2021#################

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
doc_2022 = ""
with urllib.request.urlopen(url_2022) as url:
    doc_2022 = url.read()

soup = BeautifulSoup(doc_2022, "html.parser")

s_2022 = soup.find_all("p", class_="paragraph inline-placeholder")


sentences_2022 = []
for i in range(len(s_2022)):
    sentences_2022.append(s_2022[i].text.strip())


lemma = nltk.wordnet.WordNetLemmatizer()
s_2022_tokens = []
for sentence in sentences_2022:
    s_2022_tokens.append(nltk.word_tokenize(sentence))

s_2022_lemma = []

for token in s_2022_tokens :
    for tok in token:
        s_2022_lemma.append(lemma.lemmatize(tok))

#stopwords 위에 해둠       
words_filtered_2022 = [word.lower() for word in s_2022_lemma if word.lower() not in stop_words]

s_2022_cnt = Counter(words_filtered_2022)
word_2022 = pd.DataFrame(s_2022_cnt.most_common())

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

st.markdown("""
### 결론 정리
* 연설문 분석 워드클라우드 말고 다른 방법은 없을지, 어떻게 보충?
* 코드 정리 필요할듯 지금 단순 수합이라 그런지 너무 길어서 로딩이 느림..
""")
