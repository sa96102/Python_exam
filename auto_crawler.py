# -*- coding: utf-8 -*-
"""Auto_Crawler.ipynb

"""

from selenium import webdriver # 웹페이지를 자동으로 테스트할 때 자주 사용.
from bs4 import BeautifulSoup
import time, os
from datetime import datetime
import pandas as pd

# review link link
link = 'https://play.google.com/store/apps/details?id=com.miso&hl=ko&showAllReviews=true'

# how many scrolls we need
scroll_cnt = 10 # 스크롤 횟수

# download chrome driver https://sites.google.com/a/chromium.org/chromedriver/home
driver = webdriver.Chrome('./chromedriver')
driver.get(link) # 링크를 접속하라고 명령. 자동으로 브라우저가 실행됨.

os.makedirs('result', exist_ok=True) # os.makedirs()_새로운 폴더 생성.

for i in range(scroll_cnt): # 스크롤을 하는 for문
  # scroll to bottom
  driver.execute_script('window.scrollTo(0, document.body.scrollHeight);') # driver.execute_script()_자바스크립트 실행.
  time.sleep(3)

  # click 'Load more' button, if exists
  try:
    load_more = driver.find_element_by_xpath('//*[contains(@class,"U26fgb O0WRkf oG5Srb C0oVfc n9lfJ")]').click() # driver.find_element_by_xpath()_XML 문서에서 노드의 위치를 찾을 때 사용.
  except:
    print('Cannot find load more button...')

# get review containers
reviews = driver.find_elements_by_xpath('//*[@jsname="fk8dgd"]//div[@class="d15Mdf bAhLNe"]')

print('There are %d reviews avaliable!' % len(reviews))
print('Writing the data...')

# create empty dataframe to store data
df = pd.DataFrame(columns=['name', 'ratings', 'date', 'helpful', 'comment', 'developer_comment'])

# get review data
for review in reviews:
  # parse string to html using bs4
  soup = BeautifulSoup(review.get_attribute('innerHTML'), 'html.parser') # review.get_attribute('innerHTML')_HTML 요소를 텍스트 형태로 가져옴.

  # reviewer
  name = soup.find(class_='X43Kjb').text # soup.find()_파싱된 HTML에서 요소 찾기.

  # rating
  ratings = int(soup.find('div', role='img').get('aria-label').replace('별표 5개 만점에', '').replace('개를 받았습니다.', '').strip())

  # review date
  date = soup.find(class_='p2TkOb').text
  date = datetime.strptime(date, '%Y년 %m월 %d일')
  date = date.strftime('%Y-%m-%d')

  # helpful
  helpful = soup.find(class_='jUL89d y92BAb').text
  if not helpful:
    helpful = 0 # 추천이 없는 경우도 있기 때문에 0으로 저장하도록.
  
  # review text
  comment = soup.find('span', jsname='fbQN7e').text
  if not comment:
    comment = soup.find('span', jsname='bN97Pc').text
  
  # developer comment
  developer_comment = None
  dc_div = soup.find('div', class_='LVQB0b')
  if dc_div:
    developer_comment = dc_div.text.replace('\n', ' ')
  
  # append to dataframe
  df = df.append({
    'name': name,
    'ratings': ratings,
    'date': date,
    'helpful': helpful,
    'comment': comment,
    'developer_comment': developer_comment
  }, ignore_index=True)

# finally save the dataframe into csv file
filename = datetime.now().strftime('result/%Y-%m-%d_%H-%M-%S.csv')
df.to_csv(filename, encoding='utf-8-sig', index=False)
driver.stop_client()
driver.close()

print('Done!')