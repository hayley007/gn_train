import json
import requests
import datetime
import iso8601
import pandas as pd

class GlassnodeClient:

  def __init__(self):
    self._api_key = ''

  @property
  def api_key(self):
    return self._api_key

  def set_api_key(self, value):
    self._api_key = value

  def get(self, url, a='BTC', i='24h', c='native', s=None, u=None, e=None):
    p = dict()
    p['a'] = a
    p['i'] = i
    p['c'] = c

    if s is not None:
      try:
        p['s'] = iso8601.parse_date(s).strftime('%s')
      except ParseError:
        p['s'] = s

    if u is not None:
      try:
        p['u'] = iso8601.parse_date(u).strftime('%s')
      except ParseError:
        p['u'] = s

    if e is not None:
        p['e'] = e

    p['api_key'] = self.api_key

    r = requests.get(url, params=p)

    try:
       r.raise_for_status()
    except Exception as e:
        print(e)
        print(r.text)

    try:
        # print (r.text)
        df = pd.DataFrame(json.loads(r.text))
        # print(df.head(5))
        df = df.set_index('t')
        df.index = pd.to_datetime(df.index, unit='s')
        df = df.sort_index()
        if 'v' in df.columns:
            s = df.v
        elif 'o' in df.columns:
            s = df.o
        else:
            s = df
        # s.name = '_'.join(url.split('/')[-2:])
        return s
    except Exception as e:
        print(e)

  def get_URPD(self, url, a='BTC', i='24h', c='native', s=None, u=None):
      p = dict()
      p['a'] = a
      p['i'] = i
      p['c'] = c

      if s is not None:
          try:
              p['s'] = iso8601.parse_date(s).strftime('%s')
          except ParseError:
              p['s'] = s

      if u is not None:
          try:
              p['u'] = iso8601.parse_date(u).strftime('%s')
          except ParseError:
              p['u'] = s

      p['api_key'] = self.api_key

      r = requests.get(url, params=p)

      try:
          r.raise_for_status()
      except Exception as e:
          print(e)
          print(r.text)

      try:
          df = pd.DataFrame(json.loads(r.text))
          print('df: ', df)
          print(df.shape)
          df = df.set_index('current_price')
          # df.index = pd.to_datetime(df.index, unit='s')
          df = df.sort_index()
          s = df.total_supply
          # s.name = '_'.join(url.split('/')[-2:])
          return s
      except Exception as e:
          print(e)

  def get_and_store(self, url, a='BTC', i='24h', c='native', s=None, u=None, e=None, indicator = None):
      p = dict()
      p['a'] = a
      p['i'] = i
      p['c'] = c

      if s is not None:
          try:
              p['s'] = iso8601.parse_date(s).strftime('%s')
          except ParseError:
              p['s'] = s

      if u is not None:
          try:
              p['u'] = iso8601.parse_date(u).strftime('%s')
          except ParseError:
              p['u'] = s

      if e is not None:
          p['e'] = e

      p['api_key'] = self.api_key

      r = requests.get(url, params=p)

      try:
          r.raise_for_status()
      except Exception as e:
          print(e)
          print(r.text)

      try:
          fn = './data_glassnode/'  + '-'.join(indicator.split(' ')) + '.csv'
          print('fn: ',fn)

          df = pd.DataFrame(json.loads(r.text))
          df.t = pd.to_datetime(df.t, unit='s')
          print('df: ',df)

          df.to_csv(fn,index=False)


      except Exception as e:
          print(e)