import bs4 as bs
import urllib
import pandas as pd
post_params = {
              'DateFrom' : '03/09/2023',
              'DateTo' : '03/10/2023'
              }
post_args = urllib.parse.urlencode(post_params)
post_args = post_args.encode("utf-8")
url = 'https://www.nba.com/stats/teams/traditional?'
source = urllib.request.urlopen(url, data=post_args)
soup = bs.BeautifulSoup(source,'lxml')

table = soup.find('table', attrs={'class':''})
table_rows = table.find_all('tr')
table = soup.find_all('table')
df = pd.read_html(str(table))[0]
print('pare')