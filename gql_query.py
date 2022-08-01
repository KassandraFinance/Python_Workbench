import requests as re
import json
import pandas as pd

def get_supply():
    #Supply
    query = """query {
        poolSupplies (where:{pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000) { 
        timestamp
        value
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['poolSupplies']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['value'] = df['value'].astype(float)
    df = df.sort_index().resample("1D").last()
    df.rename(columns={'value':'LP_token_supply'},inplace=True)
    return df.sort_index()


def get_swap_volumes():

    query = """query {
      volumes (where:{
      type:"swap" period:86400 pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"
      } 
        first:1000 orderBy:timestamp) {
        timestamp
        volume_usd
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['volumes']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
#     print(df.columns)
    df['volume_usd'] = df['volume_usd'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'volume_usd':'swap_volumes'},inplace=True)
    return df

def get_swap_fees():

    query = """query {
      fees (where:{type:"swap" period:86400 pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000 orderBy:timestamp) {
        timestamp
        volume_usd
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['fees']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['volume_usd'] = df['volume_usd'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'volume_usd':'swap_fees'},inplace=True)
    return df

def get_withdraw_volumes():

    query = """query {
      volumes (where:{type:"exit" period:86400 pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000 orderBy:timestamp) {
        timestamp
        volume_usd
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['volumes']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['volume_usd'] = df['volume_usd'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'volume_usd':'withdraw_volumes'},inplace=True)
    return df

def get_withdraw_fees():

    query = """query {
      fees (where:{type:"exit" period:86400 pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000 orderBy:timestamp) {
        timestamp
        volume_usd
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['fees']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['volume_usd'] = df['volume_usd'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'volume_usd':'withdraw_fees'},inplace=True)
    return df

def get_join_volumes():

    query = """query {
      volumes (where:{type:"join" period:86400 pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000 orderBy:timestamp) {
        timestamp
        volume_usd
        }
    }"""

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['volumes']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['volume_usd'] = df['volume_usd'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'volume_usd':'join_volumes'},inplace=True)
    return df

def get_ahype():
    query = """query {
      candles (where:{base:"usd" period:86400} orderBy:timestamp first:1000) {
        timestamp
        close
        period
      }
    }
    """

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['candles']
    df = pd.DataFrame(df_data)
    df.drop(labels=['period'], axis=1, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['close'] = df['close'].astype(float)
    df = df.sort_index().resample("1D").sum()
    df.rename(columns={'close':'aHYPE_price'},inplace=True)
    return df

def get_TVL():

    query = """query 
    {
      totalValueLockeds (where:{base:"usd" pool:"0x38918142779e2CD1189cBd9e932723C968363D1E"} first:1000) {
        timestamp
        close
      }
    }
    """

    url = 'https://graph.kassandra.finance/subgraphs/name/KassandraAvalanche'
    r = re.post(url, json={'query': query})
    json_data = json.loads(r.text)
    # json_data
    df_data = json_data['data']['totalValueLockeds']
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace = True)
    df['close'] = df['close'].astype(float)
    df = df.sort_index().resample("1D").last()
    df.rename(columns={'close':'Total_Value_locked'},inplace=True)
    return df