# module/package
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px #일단 이거 이용해 기본 그래프 그림
import nltk 
import random
import plotly.graph_objects as go 
import urllib.request

from PIL import Image
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


# set-page
st.set_page_config(page_icon="🗽",
                   page_title="데이터저널리즘-2조")
add_radio = st.sidebar.radio("Table of Contents", ("📅 미국의 혐오 범죄, 2015년부터 2020년까지", "🚫아시아인 혐오 범죄, 좀 더 자세히 알아볼까?", "🔍아시아인 혐오 범죄, 지역으로 좁혀 보자!", "📈 트럼프 등장! 혐오 범죄도 상승?"))

# header
st.markdown("""
# 아시아인 혐오 범죄, 확산과 심화: 코로나19 전후 미국의 혐오 범죄를 조명하다💡
* 데이터저널리즘: 오소영, 이혜정, 최다연
""") 

# introduce
st.error('''
❓코로나19는 미국에서 일어나는 아시아인 혐오 범죄에 어떤 영향을 미쳤을까?

   ✅2019년과 2020년의 혐오 범죄 건수, 과격성, 공개성 비교하기
    
❓코로나19 피해가 심했던 주에서는 혐오 범죄도 심했을까?

   ✅코로나19 확진자/사망자 수 상위 10개 주와 하위 10개 주 범죄 양상 들여다 보기

❓트럼프의 등장이 혐오 범죄 양상에 가져온 변화는?

   ✅미국 국정연설 키워드 분석하기
           ''')   
 
st.text_area('data', '''Raw data: 
CNN Politics State of the Union: 2017 ~ 2021,
The Hate Crime Statistics Program of the FBI Uniform Crime Reporting (UCR) Program
             ''', label_visibility = "hidden") 

st.markdown("""
----------------
""")

fbi =  pd.read_csv("fbi_data_edited.csv", low_memory=False)
df = pd.DataFrame(fbi, columns=['DATA_YEAR', 'BIAS_DESC'])


if add_radio == "📅 미국의 혐오 범죄, 2015년부터 2020년까지":
  # section1
  st.markdown("""
  ## 📅 미국의 혐오 범죄, 2015년부터 2020년까지
  """)

  st.markdown('''
              #
              ''')

  st.subheader("점점 증가하는 혐오 범죄 발생 건수")



  # section1-1: 전체혐오/아시아인 대상 혐오 범죄

  #Anti Asian 증가만 보여주는 그래프
  options_2 = st.radio('옵션을 선택하세요', ['전체 혐오 범죄', '아시아인 대상 혐오 범죄'])
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

  all_fig = px.line(df_all, title = "All Hate Crime Cases in the US", markers=True)
  all_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Counted Cases"}), showlegend = False)

  #아시아인 대상 혐오범죄

  asian_hates = [len(df_2015.loc[df_2015['BIAS_DESC'] == 'Anti-Asian']), len(df_2016.loc[df_2016['BIAS_DESC'] == 'Anti-Asian']), 
              len(df_2017.loc[df_2017['BIAS_DESC'] == 'Anti-Asian']), len(df_2018.loc[df_2018['BIAS_DESC'] == 'Anti-Asian']),
              len(df_2019.loc[df_2019['BIAS_DESC'] == 'Anti-Asian']), len(df_2020.loc[df_2020['BIAS_DESC'] == 'Anti-Asian'])]
  df_asian_edit = pd.DataFrame(asian_hates, index = ['2015', '2016', '2017', '2018', '2019', '2020'])
  asian_fig = px.line(df_asian_edit, title = "Asian Hate Crime Cases in the US", markers=True)
  asian_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Counted Cases"}), showlegend = False)

  if options_2 == '전체 혐오 범죄':
      st.plotly_chart(all_fig)
  elif options_2 == '아시아인 대상 혐오 범죄':
      st.plotly_chart(asian_fig)

  st.info('''
  * 전체 혐오 범죄 꾸준히 증가: 미국 전역의 혐오 범죄 빈도수를 시각화한 결과, 2015년부터 2020년까지 증가하고 있는 모습을 볼 수 있어요. 
  * 무엇이 기점?: 트럼프 전 대통령의 당선이 확정된 2016년, 코로나19가 발생한 2020년을 기점으로 혐오 범죄 건수가 두드러지게 증가했어요! 
               ''')

  st.markdown('''
              #
              ''')
  # section1-2: 전체 혐오 범죄 대비 아시아인 대상 혐오 범죄 비율 + 매트릭스

  all_asian_ratio = []
  for i in range(len(asian_hates)):
      all_asian_ratio.append((int(asian_hates[i]) / int(all_hates[i])) * 100) #아시아인 혐오범죄 건수 / 전체 혐오범죄 건수 * 100 : 비율 (맞나?)

  df_compare = pd.DataFrame(all_asian_ratio, index = ['2015', '2016', '2017', '2018', '2019', '2020'])
  compare_fig = px.bar(df_compare, title = "Asian Hate Crime Cases Ratio Out of All Hate Crimes")
  compare_fig.update_layout(xaxis = dict({"title" : "Year"}), yaxis = dict({"title" : "Relatvie proportion(%)"}), showlegend = False)

  cols = st.columns((1, 1, 3))


  cols[0].metric("","", "")
  cols[0].write("Crime Cases")
  cols[0].metric("2019", "188", "11.5%")
  cols[0].metric("","", "")
  cols[0].write("Crime Proportion")
  cols[0].metric("2019", "2.38%", "0.3%p")

  cols[1].metric("","", "")
  cols[1].metric("","", "")
  cols[1].metric("2020", "323", "26.4%")
  cols[1].metric("","", "")
  cols[1].metric("","", "")
  cols[1].metric("2020", "3.14%", "0.76%p")

  cols[2].plotly_chart(compare_fig, use_container_width = True)


  st.info('''
  * ‘코시국’의 도래: 아시아인 혐오 범죄 빈도수는 한번도 꺾이지 않고 상승세에 있었는데요📈
  특히 팬데믹이 시작된 2020년의 아시아인 혐오 범죄 건수는 전년도 대비 55% 증가했습니다. 5년 전인 2015년에 비교했을 때는 283% 가량 증가한 수치에요.
   동일한 기간 전체 혐오 범죄는 30% 증가한 것과 비교해도 엄청난 변화죠? 
          ''')    

  st.markdown('''
              #
              #
              ''')

  # section1-3: 원그래프
  st.subheader("아시아인 혐오 범죄의 비중은 어떻게 변화했을까?")

  tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["2015", "2016", "2017", "2018", "2019", "2020"])

  @st.cache
  def crime_pie_chart(filename):
      crime_data_count = pd.read_csv(filename)
      crime_data_count['number'].astype(int)
      cdc_2015 = crime_data_count['number'].tolist() #대상 인종별 범죄수 담은 리스트.
      idx_2015 = list(crime_data_count['race'])

      return crime_data_count


  with tab1:
      st.header("2015")
      fig_2015 = px.pie(crime_pie_chart("hate_crime_race.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2015.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2015)

  with tab2:
      st.header("2016")
      fig_2016 = px.pie(crime_pie_chart("hate_crime_race_2016.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2016.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2016)


  with tab3:
      st.header("2017")
      fig_2017 = px.pie(crime_pie_chart("hate_crime_race_2017.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2017.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2017)

  with tab4:
      st.header("2018")
      fig_2018 = px.pie(crime_pie_chart("hate_crime_race_2018.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2018.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2018)


  with tab5:
      st.header("2019")
      fig_2019 = px.pie(crime_pie_chart("hate_crime_race_2019.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2019.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2019)

  with tab6:
      st.header("2020")
      fig_2020 = px.pie(crime_pie_chart("hate_crime_race_2020.csv"), values = 'number', names = 'race', color_discrete_sequence=px.colors.sequential.RdBu)
      fig_2020.update_traces(pull=[0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      st.plotly_chart(fig_2020)

  st.caption('*특정 대상을 향한 혐오 범죄가 50건 이하 발생한 경우는 기타로 통합')

  st.info("""
  * 매년 100건 이상 발생: 미국 전역의 혐오 범죄 건수를 피해대상(victims)별로 분류해봤더니,아시아인을 대상으로 하는 혐오 범죄는 2015년부터 2020년까지 매넌 100건 넘게 발생했어요. 갈수록 건수, 비중 모두 늘어나고 있는 모습을 보이네요.
  * 2020년 역대 최고치: 특히 코로나19가 발생한 2020년 아시아인 혐오 범죄는 전체 혐오 범죄 중 3.13%를 차지하며 최고치를 기록했어요. 
  """)

  st.markdown('''
              #
  --------
  #
              ''')



elif add_radio == "🚫아시아인 혐오 범죄, 좀 더 자세히 알아볼까?":
  # section2
  st.markdown("""
  ##  🚫아시아인 혐오 범죄, 좀 더 자세히 알아볼까?
  """)

  st.subheader("우리는 범죄의 심각성을 아래의 세 척도로 확인했어요💡")

  st.success('''
  📌 범죄 빈도수

  📌 (범죄 유형에 따라)과격성

  📌 (범죄 장소)공개성
             ''')



  # section2-1 : 히트맵
  st.subheader("월별 아시아인 혐오 범죄 건수")

  df1 = pd.DataFrame(fbi, columns=['DATA_YEAR', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME']) 
  df1 = df1[df1['BIAS_DESC'].str.contains('Anti-Asian', na = False)]     ## Anti-Asian
  df1 = df1[(df1['DATA_YEAR'] >= 2015)]      ## 연도별로 보기 위해 2016(트럼프 당선 연도부터: 전후비교 느낌)

  df1['INCIDENT_DATE'] = pd.to_datetime(df1['INCIDENT_DATE'])

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

  monthly_crime = [count_monthly_crime(2015), count_monthly_crime(2016), count_monthly_crime(2017), count_monthly_crime(2018), count_monthly_crime(2019), count_monthly_crime(2020)]

  month_name = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
  year_month_colormap = px.imshow(monthly_crime,
                  labels=dict(x="Months", y="Year", color="Cases", width=500, height=700), 
                  x = month_name,
                  y = ['2015', '2016', '2017', '2018', '2019', '2020'],
                  color_continuous_scale='Reds',
                  width = 800, height = 600)


  st.markdown("##### Asian Hate Crimes Cases (2015-2020)")
  st.plotly_chart(year_month_colormap, use_container_width = True)


  st.info('''
  * 2020년 3월, 저게 뭐야?: 2016년 대선이 있었던 11월 아시아인 혐오 범죄가 소폭 증가한 양상을 보이며, 이후 간헐적으로 발생 빈도가 높아지던 아시아인 혐오 범죄 건수는 2020년 코로나19가 창궐하던 시기 정점을 찍었어요.
  미국에 코로나19가 막 상륙해 기승을 부리던 3월과 4월의 아시아인 혐오 범죄 건수는 각각 52건으로,
  이전 최다발생 월인 2019년 12월의 22건보다도 2.5배가량 높은 수치에요!
  같은 해 10월까지 아시아인 대상 혐오 범죄 건수는 매월 20건 이상으로 유례없는 발생 현황을 보였습니다🚨 
          ''')

  st.markdown('''
              #
  #
              ''')



  # section2-2 : 범죄의 과격성: 라인, 막대
  st.subheader("혐오 범죄의 과격성, 코로나19 전후를 볼까?")

  st.success('''
  과격성은 신체에 직접적으로 가해지는 위해 정도를 따져, 범죄 종류의 위해성이 높을수록 4에 가깝게 분류했습니다. 미국 보통법 기준 경범죄와 중범죄 분류를 참고했을 때, 대체로 1 ~ 2에 해당하는 범죄는 경범죄, 3 ~ 4에 해당하는 범죄는 중범죄로 분류됩니다.

  * 1 = 협박(intimidation), 문서 위조(counterfeiting/forgery), 약물 소지 및 학교에 들고 감(Drug violations), 무기 소지 및 학교에 들고 감(Weapon Law Violation)\n
  * 2 = 절도(theft), 재물손괴(destruction/damage/vandalism of property), 기타 절도(all other larceny)\n
  * 3 = 폭행(assault), 강도(robbery), 주거침입강도(burglary/breaking & entering), 방화(arson)\n
  * 4 = 강간(rape), 유괴(Kidnapping), 살인·과실치사·모살(murder and nonnegligent manslaughter)\n
             ''')

  df1 = pd.DataFrame(fbi, columns=['DATA_YEAR', 'INCIDENT_DATE', 'OFFENSE_NAME', 'BIAS_DESC', 'LOCATION_NAME']) #혜정이가 쓴 데이터프레임 변수명 안겹치게 df1으로.
  df1 = df1[df1['BIAS_DESC'].str.contains('Anti-Asian', na = False)]     ## Anti-Asian
  df1 = df1[(df1['DATA_YEAR'] >= 2019)]      ## 2019, 2020

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
  offense_type_line_plt = px.line(offense_type_line, title = "Graph by type of crimes", markers=True, )
  offense_type_line_plt.update_layout(xaxis = dict({"title" : "Crime Type"}), yaxis = dict({"title" : "Counted Cases"}))
  offense_type_bar_plt = px.bar(offense_type_bar, title = " ", barmode = "group")
  offense_type_bar_plt.update_layout(xaxis = dict({"title" : "Crime Type"}), yaxis = dict({"title" : "Counted Cases"}))

  col1, col2 = st.columns(2)

  with col1:
      st.plotly_chart(offense_type_line_plt, use_container_width = True)

  with col2:
      st.plotly_chart(offense_type_bar_plt, use_container_width = True)

  st.info('''
  * 혐오 범죄, ‘많이’만 일어난 게 아니다: 범죄의 과격성도 대폭 증가했어요. 살인과 같은 과격성 4 정도의 중범죄를 제외하고, 2019년 대비 2020년 아시아인을 대상으로 한 과격성 1~3의 혐오 범죄 비중이 높아진 것 보이시나요?
  * 경범죄, 중범죄 모두 증가: 1~2로 분류되는 경범죄는 물론, 폭행과 강도 등의 중범죄도 아시아인을 대상으로 적지 않게 행해졌네요.
          ''')

  st.markdown('''
              #
  #
              ''')


  # section2-3 : 범죄의 공개성: 라인, 막대
  st.subheader("혐오 범죄의 공개성, 코로나19 전후를 볼까?")

  st.success('''
  2019년 ~ 2020년: 범죄 장소의 공개성 변화\n범죄 장소의 공개성은 장소의 개방성과 공공성, 유동 인구를 고려해 1~5의 수치로 분류했으며, 숫자가 커질수록 공개성이 높음을 의미합니다. 공개성이 1, 2이면 폐쇄적인 장소, 공개성이 3이면 공간 개방성은 높으나 유동인구는 적은 장소, 공개성이 4,5이면 유동 인구와 공공성이 모두 높은 장소입니다. 

  *6은 공개성을 임의로 측정하기 어려운 기타 범죄 장소입니다.

  1 = 집(Residence/Home), 숙박업소(Hotel/Motel/Etc.)\n
  2 = 전문상점(Specialty store), 주류 판매점(Liquor Store), 차 딜러샵(Auto Dealership New/Used)\n
  3 = 공장 부지(Industrial Site), 들판·숲(Field/Woods), 강가·바닷가(Lake/Waterway/Beach), 주차장·차고(Parking/Drop Lot/Garage), 주유소·서비스센터(Service/Gas Station)\n
  4 = 술집·나이트클럽(Bar/Nightclub), 식당(Restaurant), 공원·놀이터(Park/Playground), 도로·보도(Highway/Road/Alley/Street/Sidewalk), 휴게소(rest area), convenience store(편의점), 백화점·할인판매점(Department/Discount Store), 쇼핑몰(Shopping Mall), 식료품점·슈퍼마켓(Grocery/Supermarket), 교통 시설(Air/Bus/Train Terminal), 상업용 건물(Commercial/Office Building)\n
  5 = 초·중등·대학교(School-College/Elementary/Secondary), 주민자치센터(Community Center), 홈리스 쉼터(Shelter-Mission/Homeless), 은행(Bank/Savings and Loan), 종교 시설(Church/Synagogue/Temple/Mosque), 정부·공공기관 건물(Church/Synagogue/Temple/Mosque), 의료기관(Drug Store/Doctor's Office/Hospital)\n
  6 = 알려지지 않음(Other/Unknown), 온라인 공간(Cyber space)
             ''')

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

  offense_place_line_plt = px.line(offense_place_line, title = "Graph by type of places")
  offense_place_bar_plt = px.bar(offense_place_bar, title = " ", barmode = "group")

  offense_place_line_plt.update_layout(xaxis = dict({"title" : "Crime Location"}), yaxis = dict({"title" : "Counted Cases"}))
  offense_place_bar_plt.update_layout(xaxis = dict({"title" : "Crime Location"}), yaxis = dict({"title" : "Counted Cases"}))

  col1, col2 = st.columns(2)

  with col1:
      st.plotly_chart(offense_place_line_plt, use_container_width = True)
  with col2:
      st.plotly_chart(offense_place_bar_plt, use_container_width = True)

  st.info('''
  * 일상적 공간에 스며든 혐오 범죄: 유동 인구가 많고 개방성이 높은 공개성 4의 장소에서 발생한 혐오 범죄 빈도수가 전년도 대비 2020년 두 배 가량 증가했어요. 공개성 1~3의 장소에서 발생한 범죄도 증가했지만, 공개성 4가 유독 증가했네요. 범죄 장소의 증가한 공개성은 아시아인 대상 혐오 범죄가 더욱 공공연하게 발생하고 있음을 보여줘요.
          ''')


  st.markdown("""
  ##### ✋잠깐! 여기까지 정리 ✋
  * 미국 전역 기준으로 봤을 때, 코로나19가 발생한 2020년 초를 기점으로 아시아인 대상 혐오 범죄 건수가 급증했어요. 강도, 폭행 등 범죄의 과격성도 증가했고, 범죄 장소의 공공성과 개방성, 유동인구를 척도로 매긴 공개성도 함께 증가했고요. 
  * 요컨대 코로나19의 발생과 미국 내 아시아인 혐오 범죄의 심각성은, 일종의 상관관계를 지니고 있음을 추론할 수 있어요.
  """)

elif add_radio == "🔍아시아인 혐오 범죄, 지역으로 좁혀 보자!":
    # section 3: 주별 비교

    st.markdown('''
    ## 🔍아시아인 혐오 범죄, 지역으로 좁혀 보자!
                ''')
    st.subheader("코로나19 피해가 심한 주일수록 혐오 범죄도 많았을까?")
    st.success("""
    📌 미국 주별 아시아인 인구\n
    📌 2020년 기준, 미국 주별 인구 천 명당 코로나19로 인한 사망자 수\n
    📌 2019년과 2020년, 미국 주별 아시아인 혐오 범죄 건수\n
    세 가지를 각각 시각화한 지도들 살펴보아요! 
    """)



    # section 3-1: 아시아인 인구

    image = Image.open('population_map.png')
    st.image(image, caption = "Population of Asian in US")

    st.markdown('''
    #
                ''')



    # section 3-2: 사망자수

    #파일 임포트 
    covid_state = pd.read_csv("us_county_covid_2020.csv") #2020년 12월 31일 기준, 미국 각 주 county별 누적 확진 수를 담은 파일.

    #주별 인구 10만명 당 코로나19 확진자 수 표 그리기
    covid_state_df = pd.DataFrame(covid_state)

    @st.cache #캐시 사용하도록 설정
    #주별 누적 사망 건수 추출 함수
    def state_deaths(state_name):
        state = covid_state_df.loc[covid_state_df['state'] == state_name]
        return state['deaths'].sum()

    state_name_list = list(covid_state_df['state'].unique())

    deaths_list = []
    for item in state_name_list:
        deaths_list.append(int(state_deaths(item)))

    #deaths_list

    state_death = {'State':state_name_list, 'Deaths':deaths_list}
    state_death_df = pd.DataFrame(state_death)

    #10만 명당: (전체 건수 / 2020 기준 해당 주 인구수) * 100,000 - 해당 주 인구수는 us census.gov 사이트에서 가져온 파일, 2020 7월 기준
    us_population = pd.read_csv("us_population_2020.csv")
    us_population['2020'] = us_population['2020'].apply(lambda x: int(x.replace(',', '')))
    us_death_population = pd.merge(state_death_df, us_population)

    #1천 명당 사망자 수 : (전체 건수 / 2020 기준 해당 주 인구수) * 1000
    us_death_population['deaths_per_1k'] = us_death_population.apply(lambda x: ((x['Deaths'] / x['2020']) * 1000), axis=1)
    us_death_population_final = us_death_population[['State', 'deaths_per_1k']]

    from urllib.request import urlopen
    import json
    with urlopen('https://raw.githubusercontent.com/ChoiDayeun/datajournalism_streamlit/main/us-states_json_edit.json') as response:
        states = json.load(response)

    #지도 데이터 불러와서 시각화.
    fig_death_population = px.choropleth(us_death_population_final, geojson= states, locations='State', 
                        color = 'deaths_per_1k',
                        color_continuous_scale="Reds",
                        range_color=(0, us_death_population_final.deaths_per_1k.max()),
                        featureidkey='properties.State',
                        scope="usa",
                        labels={'deaths_per_1k':'deaths per 1k'},
                        title = 'US COVID-19 Deaths per 1k by States'
                        )
    fig_death_population.update_layout(margin={"r":0,"t":60,"l":0,"b":0})
    #fig_death_population.show()

    st.plotly_chart(fig_death_population, use_container_width = True)




    # section 3-3 : 2019/2020 Asian hate crime
    #주별 아시아인 혐오범죄 건수 표 그리기: 2019, 2020
    fbi_2019 = fbi.loc[fbi['DATA_YEAR'] == 2019]
    fbi_2019 = fbi_2019[fbi_2019['BIAS_DESC'].str.contains('Anti-Asian', na = False)] 
    fbi_state_2019 = fbi_2019[['STATE_NAME']]

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
    fbi_2020 = fbi.loc[fbi['DATA_YEAR'] == 2020]
    fbi_2020 = fbi_2020[fbi_2020['BIAS_DESC'].str.contains('Anti-Asian', na = False)] 
    fbi_state_2020 = fbi_2020[['STATE_NAME']]

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
                        title = 'Asian Hate Crime Cases per State 2019', 
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
                        title = 'Asian Hate Crime Cases per State 2020', 
                        )
    fig_crime_2020.update_layout(margin={"r":0,"t":60,"l":0,"b":0})


    st.markdown('''
    #
                ''')
    select_yr = st.selectbox('확인할 해를 선택하세요', ['2019', '2020'])
    if select_yr == '2019':
        st.plotly_chart(fig_crime_2019)
    elif select_yr == '2020':
        st.plotly_chart(fig_crime_2020)

    st.info('''
    * 코로나19 피해 심했던 시기, 아시아인 혐오 범죄도 많았다: 인구 천 명당 코로나19로 인한 사망자 수와 아시아인 혐오 범죄 발생 건수를 주별로 그린 지도랑, 아시아인 인구 분포도를 보니 사망자가 많은 주에서 혐오 범죄 수도 많이 발생하는 경향이 보이네요.
    * 위 표에서 보이듯이 미국 내 주별로 아시아인 인구 분포의 차이가 크다 보니, 아시아인 혐오 범죄 발생 건수가 많은 주들이 인구 천 명당 코로나19로 인한 사망자 수가 많은 주들과 직접적으로 일치하지 않는 경우도 있어요.
    ''')
    
    st.markdown('''
              #
  #
              ''')

    
    st.subheader("코로나19 피해 정도와 아시아인 혐오 범죄의 심각성 간에 관계가 있을까?")
    st.success('''
   2020년에 미국 내 코로나19로 인해 인구 천 명당 사망자 순으로 정렬한 결과는 다음과 같아요📇
   * 가장 사망자가 많았던 상위 10개 주는😢\n: New Jersey, New York, Massachusetts, South Dakota, North Dakota, Connecticut, Rhode Island, Mississippi, Louisiana, Illinois
   * 비교적 사망자가 적었던 하위 10개 주는🥲\n: Hawaii, Vermont, Maine, Alaska, Oregon, Utah, Washington, Puerto Rico, New Hampshire, Virginia
   ''')
    st.markdown(
              #
  #
              )

    ####주별 정리 다시: 10개###
    #1000명당 사망자수별로 정렬: 상위 10개, 하위 10개 주 추출
    #상위 10개, 하위 10개
    death_top10_df = us_death_population_final.sort_values('deaths_per_1k', ascending = False)[:10]
    death_bottom10_df = us_death_population_final.sort_values('deaths_per_1k')[:10]

    #해당 주 이름들 추출
    death_top10_names = []
    death_bottom10_names = []
    for item in death_top10_df['State']:
        death_top10_names.append(item)

    for item in death_bottom10_df['State']:
        death_bottom10_names.append(item)


    #################사망자수 기준 ##################

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


    #death_top10_names,  death_bottom10_names에 있는 주들의 값을 바꾼 후, df에 해당 컬럼들만 남김: 상위 10개 : TOP_10_COVID_deathS, BOTTOM_10_COVID_deathS
    for item in death_top10_names:
        df_new2.loc[df_new2['STATE_NAME'] == item, 'STATE_NAME'] = 'TOP_10_COVID_DEATHS'
    for item in death_bottom10_names:
        df_new2.loc[df_new2['STATE_NAME'] == item, 'STATE_NAME'] = 'BOTTOM_10_COVID_DEATHS'
    df_deaths_state = df_new2.loc[(df_new2['STATE_NAME'] == 'TOP_10_COVID_DEATHS') | (df_new2['STATE_NAME'] == 'BOTTOM_10_COVID_DEATHS')]


    ##아시아인 대상 혐오범죄만, 사망자수 기준으로 추출
    df_state_asian = df_deaths_state[df_deaths_state['BIAS_DESC'].str.contains('Anti-Asian', na = False)] 
    df_state_asian = df_deaths_state[df_deaths_state['BIAS_DESC'].str.contains('Anti-Asian', na = False)] 
    
    #사망자 수 상위 10개/하위 10개 주 데이터프레임
    top10_asian_df = df_state_asian[df_state_asian['STATE_NAME'] == 'TOP_10_COVID_DEATHS']
    bottom10_asian_df = df_state_asian[df_state_asian['STATE_NAME'] == 'BOTTOM_10_COVID_DEATHS']
    
    #상위 10개 주 종류별 정리
    pd_top10 = pd.crosstab(top10_asian_df['OFFENSE_NAME'], top10_asian_df['DATA_YEAR'])
    pd_top10_loc = pd.crosstab(top10_asian_df['LOCATION_NAME'],top10_asian_df['DATA_YEAR'])
    
    #하위 10개 주 종류별 정리
    pd_bottom10 = pd.crosstab(bottom10_asian_df['OFFENSE_NAME'], bottom10_asian_df['DATA_YEAR'])
    pd_bottom10_loc = pd.crosstab(bottom10_asian_df['LOCATION_NAME'], bottom10_asian_df['DATA_YEAR'])
    
    
    
    #사망자수 가장 많은 상위 10개 주 기준
    
    #범죄의 과격성
    raw_data = {'year': [2019, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020],
                'offense_name': list(pd_top10.index) * 2,
                'offense_number': list(pd_top10[2019]) + list(pd_top10[2020])}

    offense_name = pd.DataFrame(raw_data)


    fig_off_asian = px.scatter(offense_name, x="offense_number", y="offense_name",  color="year", color_continuous_scale='Bluered_r',  title = '범죄 양상의 과격성-코로나19 사망자 상위 10개 주 기준')
    # iterate on each region
    for i in offense_name["offense_name"].unique():
        # filter by region
        df_sub = offense_name[offense_name["offense_name"] == i]

        fig_off_asian.add_shape(
            type="line",
            layer="below",
            # connect the two markers
            y0=df_sub.offense_name.values[0], x0=df_sub.offense_number.values[0],
            y1=df_sub.offense_name.values[1], x1=df_sub.offense_number.values[1], 
        )
    

    #범죄 장소의 공개성
    raw_data2 = {'year': [2019, 2019, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020, 2020],
                'location_name': list(pd_top10_loc.index) * 2,
                'location_number': list(pd_top10_loc[2019]) + list(pd_top10_loc[2020])}

    location_names = pd.DataFrame(raw_data2)

    fig_loc_asian = px.scatter(location_names, x="location_number", y="location_name", color="year", color_continuous_scale='Bluered_r', title = '범죄 장소의 공개성-코로나19 사망자 상위 10개 주 기준')
    # iterate on each region
    for i in location_names["location_name"].unique():
        df_sub = location_names[location_names["location_name"] == i]

        fig_loc_asian.add_shape(
            type="line",
            layer="below",
            # connect the two markers
            y0=df_sub.location_name.values[0], x0=df_sub.location_number.values[0],
            y1=df_sub.location_name.values[1], x1=df_sub.location_number.values[1], 
        )
    
    
   #사망자 수 하위 10개 주의 경우
    raw_data_3 = {'year': [2019, 2019, 2019, 2020, 2020, 2020],
                'offense_name': list(pd_bottom10.index) * 2,
                'offense_number': list(pd_bottom10[2019]) + list(pd_bottom10[2020])}

    offense_name3 = pd.DataFrame(raw_data_3)

    fig_off_asian_bottom = px.scatter(offense_name3, x="offense_number", y="offense_name",    color="year", color_continuous_scale='Bluered_r',  title = '범죄 양상의 과격성-코로나19 사망자 하위 10개 주 기준')
    # iterate on each region
    for i in offense_name3["offense_name"].unique():
        # filter by region
        df_sub = offense_name3[offense_name3["offense_name"] == i]

        fig_off_asian_bottom.add_shape(
            type="line",
            layer="below",
            y0=df_sub.offense_name.values[0], x0=df_sub.offense_number.values[0],
            y1=df_sub.offense_name.values[1], x1=df_sub.offense_number.values[1], 
        )


    raw_data4 = {'year': [2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020],
                'location_name': list(pd_bottom10_loc.index) * 2,
                'location_number': list(pd_bottom10_loc[2019]) + list(pd_bottom10_loc[2020])}

    location_names4 = pd.DataFrame(raw_data4)

    fig_loc_asian_bottom = px.scatter(location_names4, x="location_number", y="location_name", color="year", color_continuous_scale='Bluered_r',  title = '범죄 장소의 공개성-코로나19 사망자 하위 10개 주 기준')
    # iterate on each region
    for i in location_names4["location_name"].unique():
        # filter by region
        df_sub = location_names4[location_names4["location_name"] == i]

        fig_loc_asian_bottom.add_shape(
            type="line",
            layer="below",
            # connect the two markers
            ## e.g., y0='Robredo', x0=43.53
            y0=df_sub.location_name.values[0], x0=df_sub.location_number.values[0],
            ## e.g., y1='Marcos', x1=26.60
            y1=df_sub.location_name.values[1], x1=df_sub.location_number.values[1], 
        )
                                           
                          
    st.subheader("코로나19 사망자 상위-하위 10개 주, 아시아인 혐오 범죄 과격성 비교하기👊")
    st.plotly_chart(fig_off_asian)
    st.plotly_chart(fig_off_asian_bottom)
    
    #범죄 과격성 비교
    st.info('''
    * 중범죄, 상위 10개주에서만 증가: 과격성 3, 4인 중범죄는 사망자 수가 많았던 상위 10개 주에서는 증가한 반면 하위 10개 주에서는 감소했어요. 하위 10개주에서 과격성 4의 범죄는 아예 일어나지 않았다는 것도 주목해주세요!
    * 경범죄도 더 높은 비율로 증가: 과격성 2인 범죄의 경우 하위 10개 주에서도 두 배 증가했지만, 상위 10개 주에서는 전년도 대비 171% 증가했네요. 코로나19가 심했던 지역에서는 과격성이 낮은 범죄와 높은 범죄가 고루 아시아인을 대상으로 행해졌다는 것을 알 수 있어요.
    ''')
    st.markdown(
              #
              )
      
    st.subheader("그렇다면 아시아인 혐오 범죄가 일어난 장소의 공개성은?🏙️")
                                           
    st.plotly_chart(fig_loc_asian)
    st.write(fig_loc_asian_bottom)
    
    #범죄 장소 공개성 비교
    st.info('''
    * 코로나19 심한 지역이 범죄 장소의 공개성 더 높아: 우선 공개성이 5인 장소에서 발생한 아시아인 혐오 범죄는 사망자수가 가장 많은 10개주에서 증가한 반면, 사망자수가 가장 적은 10개주에서는 오히려 감소했어요. 공개성 4인 장소에서 일어난 혐오 범죄의 증가율은 상위 10개 주에서는 전년도 대비 80%, 하위 10개 주에서는 16%였고요. 코로나19 사망자가 많은 주에서 더 공개성 높은 혐오 범죄가 이전에 비해 많이 발생했음을 알 수 있어요🧑‍⚖️
    ''')


    st.markdown("""
    ##### ✋잠깐! 여기까지 정리 ✋
    * 미국 전역에 이어 코로나19 사망자가 많았던 10개 주와 그렇지 않은 10개 주를 비교해봤는데요, 상위 10개 주에서 발생한 범죄의 과격성이 하위 10개 주보다 더욱 높게 나타났습니다.
    * 범죄 장소의 공개성 또한, 코로나19를 전후로 그 피해가 심했던 지역에서 더욱 증가한 것을 볼 수 있었어요. 
    * 결론적으로 코로나19가 심했던 주의 아시아인 혐오 범죄 양상이, 2019년과 2020년 사이 더욱 유의미하게 변화했음을 알 수 있어요. 
    이를 통해, 코로나 19의 발생과 아시아인 혐오 범죄의 심각성 사이 강한 상관관계가 있음을 보일 수 있습니다📊🖊️
    """)

    st.markdown('''
    -----------
                ''')


elif add_radio == "📈 트럼프 등장! 혐오 범죄도 상승?":

  # section 4: 연설문 트리맵
  st.markdown('''
  ## 📈 트럼프 등장! 혐오 범죄도 상승?
              ''')


  st.warning('''
  * 트럼프 집권 시기, 바이든 집권 시기 국정 연설(State of the Union Speech) 키워드 분석을 해봤어요. 
  * 1년에 한 번 있는 국정 연설은 대통령이 직접 국가 상황과 정책 기조를 설명하는 담화인 만큼, 해당 담화에서 나타난 키워드들은 당시의 사회적 상황을 설명하는 데 있어 매우 중요해요.
  * 보라색 트리맵은 자주 등장한 키워드를 빈도수대로 그린 것이고, 아래의 트리맵은 그중에서도 혐오 범죄 관련 단어들을 모아놓은 것이에요! 
  ''')

                                           
  #*stopwords 업데이트
  #words filtered
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  stop_words.update(['.', ',', '"', "'", '?', '!',"''",'``', '', '', ':', ';', '’', '(', ')','`','[', ']',
  '--','–', '“', '”', '{', '}','_', "'\\n", "\n", "—", '%', '#', '###', 'u', 'wa', '$', 'america', 'american', 
  'americans', 'people', 'year', 'ha', 'also', 'tonight']) #매년 반복되는 의례적 단어들은 제외.


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

  # 전체 리스트 + 혐오 범죄 관련 단어들 빈도수 세는 리스트 만드는 함수 추가
  @st.cache(allow_output_mutation=True)
  def count_words_total(counter):
      hate_words_list = ['isis', 'islamic', 'terrorist', 'asian', 'border', 'race', 'racism', 'african-american', 'immigration', 'china', 'threat']
      hate_words_dict = {}

      counter_df = pd.DataFrame(counter.most_common(), columns = ['word', 'count'])
      for w in counter_df['word']:
          if w in hate_words_list:
              hate_words_dict[w] = len(counter_df[counter_df['word'] == w])

      hate_words = list(hate_words_dict.keys())
      hate_words_count = list(hate_words_dict.values())

      counter_df_hate = pd.DataFrame({'word': hate_words, 'count': hate_words_count, 'type': hate_words_count})
      counter_df_hate['type'] = 'hate crime word'

      return counter_df_hate

  ############### 2017 #################
  url_2017 = "https://edition.cnn.com/2017/02/28/politics/donald-trump-speech-transcript-full-text/index.html"
  word_2017 = pd.DataFrame(speech_keywords_counter(url_2017).most_common(70), columns = ['word', 'count'])
  hate_2017 = count_words_total(speech_keywords_counter(url_2017))

  ##############2018##############
  url_2018 = "https://edition.cnn.com/2018/01/30/politics/2018-state-of-the-union-transcript/index.html"
  word_2018 = pd.DataFrame(speech_keywords_counter(url_2018).most_common(70), columns = ['word', 'count'])
  hate_2018 = count_words_total(speech_keywords_counter(url_2018))

  ##############2019##############
  url_2019 = "https://edition.cnn.com/2019/02/05/politics/donald-trump-state-of-the-union-2019-transcript/index.html"
  word_2019 = pd.DataFrame(speech_keywords_counter(url_2019).most_common(70), columns = ['word', 'count'])
  hate_2019= count_words_total(speech_keywords_counter(url_2019))

  ##############2020##############
  url_2020 = "https://edition.cnn.com/2020/02/04/politics/trump-2020-state-of-the-union-address/index.html"
  word_2020 = pd.DataFrame(speech_keywords_counter(url_2020).most_common(70), columns = ['word', 'count'])
  hate_2020 = count_words_total(speech_keywords_counter(url_2020))

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
  word_2021 = pd.DataFrame(s_2021_cnt.most_common(70), columns = ['word', 'count'])
  hate_2021 = count_words_total(s_2021_cnt)

  ##############2022##############
  url_2022 = "https://edition.cnn.com/2022/03/01/politics/biden-state-of-the-union-2022-transcript/index.html"
  word_2022 = pd.DataFrame(speech_keywords_counter(url_2022).most_common(70), columns = ['word', 'count'])
  hate_2022 = count_words_total(speech_keywords_counter(url_2022))


  select_year = st.selectbox('확인할 연도를 선택하세요', ('2017', '2018', '2019', '2020', '2021', '2022'))

  ## *트리맵 그리는 함수!
  #from plotly.subplots import make_subplots

  @st.cache
  def draw_treemap(word_df):
      fig = px.treemap(
          word_df, path=[px.Constant("total"), 'word'],
          values = 'count',
          color = 'count',
          color_continuous_scale='Purples',
          color_continuous_midpoint = 0, title='State of the Union Speech Keywords'
      )
      fig.update_traces(root_color="lightgrey")
      fig.update_layout(font_size=15, margin = dict(t=50, l=25, r=25, b=25))

      return fig

  @st.cache
  def draw_treemap_hate(word_df):
      fig = px.treemap(
          word_df, path=[px.Constant("total"),  'word'],
          values = 'count',
          color_discrete_map= {'word' : 'red'},
          color_continuous_midpoint = 0, title='Hate Crime Keywords'
      )
      fig.update_traces(root_color="lightgrey")
      fig.update_layout(font_size=20, margin = dict(t=50, l=25, r=25, b=25))

      return fig

  #color_continuous_midpoint

  if select_year == '2017':
      st.plotly_chart(draw_treemap(word_2017), use_container_width=True)
      st.plotly_chart(draw_treemap_hate(hate_2017), use_container_width=True)
  elif select_year == '2018':
      st.plotly_chart(draw_treemap(word_2018))
      st.plotly_chart(draw_treemap_hate(hate_2018))
  elif select_year == '2019':
      st.plotly_chart(draw_treemap(word_2019))
      st.plotly_chart(draw_treemap_hate(hate_2019))
  elif select_year == '2020':
      st.plotly_chart(draw_treemap(word_2020))
      st.plotly_chart(draw_treemap_hate(hate_2020))
  elif select_year == '2021':
      st.plotly_chart(draw_treemap(word_2021))
      st.plotly_chart(draw_treemap_hate(hate_2021))
  elif select_year == '2022':
      st.plotly_chart(draw_treemap(word_2022))
      st.plotly_chart(draw_treemap_hate(hate_2022))


  st.info('''
  * 2016년부터 2022년, 국정 연설 비교해보니: 빈도수를 비교해보니, 트럼프 시기에는 race, islamic, isis, African-American과 같은 직접적인 인종 및 민족 언급이 많았던 반면 바이든 시기에는 그런 키워드가 등장하지 않고 immigration, threat 등의 단어만 공통적으로 연설에 사용됐어요. 
  * 이슬람 관련 언급이 특히 많아: 트럼프 시기 국정연설에는 다른 것보다도 terrorist라는 단어와 이슬람 관련 단어인 ISIS, Islamic이 자주 함께 등장했어요. 2020년까지 무슬림 혐오 범죄가 꾸준히 비중을 차지했던 것과 무관하지 않아보입니다. 
  * 바이든 시기에도 terrorist라는 단어는 국정 연설에 사용됐지만, 특정 인종과 민족을 콕 집어 언급하지는 않았다는 차이점이 있어요.
  ''')


  st.markdown('''
  ##### 💬마무리하는 말💬
  📍 트럼프 당선 시기를 기점으로 전체 혐오 범죄가 증가했음은 물론,
  트럼프 집권 시기에는 대통령 국정연설에서도 인종 관련 문제가 더욱 자주 등장했어요.
  또한 특정 인종에 관한 부정적인 감정을 유발할 수 있는 단어도 바이든 시기에 비해 훨씬 자주 등장했답니다.
  즉 트럼프의 등장과 전반적인 혐오 범죄의 증가가 나름의 상관관계를 지니고 있다고 정리해볼 수 있겠어요!

  📍 코로나19가 창궐한 이후, 미국 내 아시아인 혐오 범죄의 심각성은 그전과 비교해 심화됐습니다. 
  특히 코로나19가 상륙한 시기 아시아인 혐오 범죄 건수는 극에 달했고, 과격성은 물론 공개적인 장소에서 벌어지는 범죄도 유의미하게 증가했어요⬆️

  📍 코로나19와 아시아인 혐오 범죄의 관계는 코로나19 사망자가 많았던 10개 주와 사망자가 적었던 10개 주의 비교를 통해 가장 분명히 볼 수 있었어요. 
  사망자 수가 많은 지역일수록 아시아인 혐오 범죄 수, 범죄의 과격성, 공격성이 2020년 더 높은 비율로 증가했고,
   낮은 지역일수록 코로나19의 등장 여부와 상관없이 범죄 양상이 유지되거나 오히려 하락하는 모습을 보였답니다.
   요컨대 코로나19의 등장과 아시아인 혐오 범죄의 심각성은 정의 관계(➕)에 있다고 말할 수 있습니다!
          ''')

