import time
import pickle
import os
from selenium import webdriver


driver = webdriver.Chrome('./chromedriver')
driver.get("https://nextdoor.com/news_feed/")

username = driver.find_element_by_id('id_email')
password = driver.find_element_by_id('id_password')

username.send_keys(os.environ.get("NEXTDOOR_EMAIL"))
password.send_keys(os.environ.get("NEXTDOOR_PASSWORD"))

driver.find_element_by_id('signin_button').click()

for i in range(1, 5):
	driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
	time.sleep(1.5)

bag_of_text = []

elements = driver.find_element_by_id('main_content').find_elements_by_xpath('//div/article[@tabindex="0"]')
for tag in elements:
	if len(tag.text.split(' ')) > 6:
		btn = tag.find_elements_by_class_name("truncate-view-more-link")
		if len(btn) == 0:
			bag_of_text.append(tag.text)
		else:
			driver.execute_script("arguments[0].scrollIntoView();", tag)
			btn[0].click()
			bag_of_text.append(tag.text)

with open('bag-of-text-new.pkl', 'wb') as f:
	pickle.dump(bag_of_text, f)