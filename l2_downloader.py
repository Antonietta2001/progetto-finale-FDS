# l2_downloader.py (Script di esempio per CoinAPI)
import requests
import pandas as pd
from datetime import datetime, timedelta

# Inserisci qui la tua chiave API
YOUR_API_KEY = "e76a9336-871d-4362-b362-2ce05358775f"
# Utilizziamo un ID comune per Binance BTC/USDT spot
SYMBOL_ID = "BINANCE_SPOT_BTC_USDT" 

def download_order_book_data(start_date: str) -> pd.DataFrame:
    """Scarica i dati di Order Book Snapshot per una data specifica."""
    
    # La risoluzione di CoinAPI è limitata nel piano gratuito. 
    # Proviamo a scaricare gli Snapshot che sono meno pesanti dei tick completi.
    url = f"https://rest.coinapi.io/v1/orderbooks/{SYMBOL_ID}/snapshots"
    
    # Per una data specifica, usiamo 'time_start' e 'time_end'
    # Esempio: 2021-05-18T00:00:00.0000000Z
    time_start = f"{start_date}T00:00:00.0000000Z"
    time_end = f"{(datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00.0000000Z"
    
    headers = {'X-CoinAPI-Key': YOUR_API_KEY}
    params = {
        'time_start': time_start,
        'time_end': time_end,
        'limit': 100000 # Limite massimo di oggetti per chiamata
    }
    
    print(f"🌍 Richiesta dati L2 per {start_date}...")
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status() # Lancia un errore se lo stato non è 200
        
        data = response.json()
        if not data:
            print("⚠️ Nessun dato Order Book trovato per questa data. Controlla il tuo piano API.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        print(f"✅ Scaricati {len(df):,} snapshot L2 per {start_date}.")
        return df
        
    except requests.exceptions.HTTPError as err:
        print(f"❌ Errore HTTP CoinAPI (Controlla la chiave e i limiti): {err}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Errore generico: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Scarica i dati per il 18 e 19 Maggio 2021
    df_18 = download_order_book_data('2021-05-18')
    df_19 = download_order_book_data('2021-05-19')
    
    combined_l2 = pd.concat([df_18, df_19], ignore_index=True)
    if not combined_l2.empty:
        # Salva i dati L2 grezzi. Useremo la funzione di merge nel data_processor.py
        combined_l2.to_csv('L2_snapshots_raw.csv', index=False)
        print("\n✅ Dati Order Book L2 salvati con successo come L2_snapshots_raw.csv")