import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, BarChart, Bar } from 'recharts';
import { Activity, AlertTriangle, TrendingUp, TrendingDown, Heart, Wifi, WifiOff, Database, Cpu, Zap, Twitter } from 'lucide-react';

const MARKET_ML_API = 'http://localhost:8000/predict_risk'; 
const LSTM_SEQUENCE_LENGTH = 10; 

const MarketHeartbeat = () => {
  const [ofiData, setOfiData] = useState([]);
  const [currentMetrics, setCurrentMetrics] = useState({
    ofi: 0,
    spread: 0,
    spreadBps: 0,
    price: 0, // Inviato al backend (CORREZIONE 1: Aggiunto al modello di stato)
    hawkesIntensity: 0,
    riskLevel: 'Initializing',
    bidDepth: 0,
    askDepth: 0,
    imbalanceRatio: 0,
    mlProbability: 0,
    fearGreedIndex: 50,
    twitterSentiment: 0.0
  });
  const [isConnected, setIsConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('Initializing...');

  // Riferimenti
  const wsTradeRef = useRef(null);
  const wsDepthRef = useRef(null);
  const ofiHistoryRef = useRef([]); 
  const tradesRef = useRef([]); 
  const orderBookRef = useRef({ bids: [], asks: [] });
  const lastUpdateTime = useRef(Date.now());
  const lstmSequenceRef = useRef([]); 

  // 1. Connessione a Binance WebSocket - DATI REALI
  useEffect(() => {
    connectToBinance();
        
    return () => {
      if (wsTradeRef.current) wsTradeRef.current.close();
      if (wsDepthRef.current) wsTradeRef.current.close();
    };
  }, []);

  const connectToBinance = () => {
    const symbol = 'btcusdt';
    const tradeWsUrl = `wss://stream.binance.com:9443/ws/${symbol}@trade`;
    const depthWsUrl = `wss://stream.binance.com:9443/ws/${symbol}@depth10@100ms`;

    // Trade Stream
    wsTradeRef.current = new WebSocket(tradeWsUrl);
    wsTradeRef.current.onopen = () => { setConnectionStatus('Trade Stream: Connected ✓'); };
    wsTradeRef.current.onmessage = (event) => { processTrade(JSON.parse(event.data)); };
    wsTradeRef.current.onerror = (error) => { setConnectionStatus('Trade Stream: Error ✗'); console.error('Trade WS Error:', error); };
    wsTradeRef.current.onclose = () => { setConnectionStatus('Trade Stream: Disconnected'); setIsConnected(false); setTimeout(() => connectToBinance(), 5000); };

    // Depth Stream
    wsDepthRef.current = new WebSocket(depthWsUrl);
    wsDepthRef.current.onopen = () => { console.log('✅ Connected to Binance Depth Stream'); setIsConnected(true); };
    wsDepthRef.current.onmessage = (event) => { processOrderBook(JSON.parse(event.data)); };
    wsDepthRef.current.onerror = (error) => { console.error('Depth WS Error:', error); };
  };

  const processTrade = (trade) => {
    const tradeData = {
      timestamp: trade.T, 
      price: parseFloat(trade.p),
      quantity: parseFloat(trade.q),
      side: trade.m ? 'SELL' : 'BUY',
      value: parseFloat(trade.p) * parseFloat(trade.q)
    };
    tradesRef.current.push(tradeData);
    if (tradesRef.current.length > 100) { tradesRef.current.shift(); }

    const now = Date.now();
    if (now - lastUpdateTime.current >= 1000) {
      calculateMetrics();
      lastUpdateTime.current = now;
    }
  };

  const processOrderBook = (depth) => {
    orderBookRef.current = {
      bids: depth.bids.map(([price, qty]) => ({ price: parseFloat(price), quantity: parseFloat(qty) })),
      asks: depth.asks.map(([price, qty]) => ({ price: parseFloat(price), quantity: parseFloat(qty) }))
    };
  };

  // 2. CALCOLO DELLE FEATURE MICROSTRUTTURALI
  const calculateMetrics = () => {
    if (tradesRef.current.length < 2) return;

    const timestamp = Date.now();
    const recentTrades = tradesRef.current.slice(-30); 

    // OFI
    const buyVolume = recentTrades.filter(t => t.side === 'BUY').reduce((sum, t) => sum + t.quantity, 0);
    const sellVolume = recentTrades.filter(t => t.side === 'SELL').reduce((sum, t) => sum + t.quantity, 0);
    const ofi = buyVolume - sellVolume;
    const totalVolume = buyVolume + sellVolume;
    const imbalanceRatio = totalVolume > 0 ? (ofi / totalVolume) * 100 : 0;

    // Spread & Depth
    const book = orderBookRef.current;
    let spread = 0;
    let spreadBps = 0;
    let bidDepth = 0;
    let askDepth = 0;
    if (book.bids.length > 0 && book.asks.length > 0) {
      const bestBid = book.bids[0].price;
      const bestAsk = book.asks[0].price;
      spread = bestAsk - bestBid;
      spreadBps = (spread / bestBid) * 10000;
      bidDepth = book.bids.slice(0, 5).reduce((sum, l) => sum + l.quantity, 0);
      askDepth = book.asks.slice(0, 5).reduce((sum, l) => sum + l.quantity, 0);
    }

    // Calcola il Prezzo (WVAP) - ESSENZIALE
    const price = recentTrades.reduce((sum, t) => sum + t.price * t.quantity, 0) / 
                   recentTrades.reduce((sum, t) => sum + t.quantity, 0);

    // Hawkes Process Intensity
    const hawkesIntensity = calculateHawkesIntensity(tradesRef.current, timestamp);

    // ⭐ CORREZIONE 2: Creazione del Punto Dati - Tutti i campi devono essere inclusi!
    const newDataPoint = {
      time: new Date(timestamp).toLocaleTimeString(),
      ofi: parseFloat(ofi.toFixed(4)),
      spread_bps: parseFloat(spreadBps.toFixed(2)),
      price: parseFloat(price.toFixed(2)), // CAMPO MANCANTE - ORA INCLUSO
      hawkes_intensity: parseFloat(hawkesIntensity.toFixed(3)),
      imbalance_ratio: parseFloat(imbalanceRatio.toFixed(2)),
      bid_depth: parseFloat(bidDepth.toFixed(2)),
      ask_depth: parseFloat(askDepth.toFixed(2))
    };

    // Aggiorna History
    ofiHistoryRef.current.push(newDataPoint);
    if (ofiHistoryRef.current.length > 60) { ofiHistoryRef.current.shift(); }
    setOfiData([...ofiHistoryRef.current]);

    // Gestione della sequenza LSTM
    lstmSequenceRef.current.push(newDataPoint);
    if (lstmSequenceRef.current.length > LSTM_SEQUENCE_LENGTH) { 
        lstmSequenceRef.current.shift(); 
    }

    // CHIAMA IL MODELLO ML
    if (lstmSequenceRef.current.length >= LSTM_SEQUENCE_LENGTH) {
        predictWithML(lstmSequenceRef.current)
            .then(mlResult => {
                setCurrentMetrics(prev => ({
                    ...prev,
                    riskLevel: mlResult.risk_level_predicted,
                    mlProbability: mlResult.probability_of_crash,
                    fearGreedIndex: mlResult.fear_greed_index,
                    twitterSentiment: mlResult.twitter_sentiment,
                    
                    // Microstruttura (per display)
                    price: newDataPoint.price.toFixed(2), 
                    ofi: newDataPoint.ofi.toFixed(4),
                    spread: spread.toFixed(4), 
                    spreadBps: newDataPoint.spread_bps.toFixed(2),
                    hawkesIntensity: newDataPoint.hawkes_intensity.toFixed(3),
                    bidDepth: newDataPoint.bid_depth.toFixed(2),
                    askDepth: newDataPoint.ask_depth.toFixed(2),
                    imbalanceRatio: newDataPoint.imbalance_ratio.toFixed(2)
                }));
            })
            .catch(() => {
                // Il catch cattura l'errore 500 del backend e lo mostra come API Error
                setCurrentMetrics(prev => ({ ...prev, riskLevel: 'API Error', mlProbability: 0 })); 
            });
    } else {
        // Aggiornamento normale delle metriche durante l'inizializzazione
        setCurrentMetrics(prev => ({
            ...prev,
            riskLevel: `Collecting ${lstmSequenceRef.current.length}/${LSTM_SEQUENCE_LENGTH}`, 
            price: newDataPoint.price.toFixed(2), 
            ofi: newDataPoint.ofi.toFixed(4),
            spread: spread.toFixed(4), 
            spreadBps: newDataPoint.spread_bps.toFixed(2),
            hawkesIntensity: newDataPoint.hawkes_intensity.toFixed(3),
            bidDepth: newDataPoint.bid_depth.toFixed(2),
            askDepth: newDataPoint.ask_depth.toFixed(2),
            imbalanceRatio: newDataPoint.imbalance_ratio.toFixed(2)
        }));
    }
  };
  
  // FUNZIONE CHIAVE: Chiama l'API ML
  const predictWithML = async (sequence) => {
    try {
        const payload = { sequence: sequence.slice(-LSTM_SEQUENCE_LENGTH) }; 
        
        const response = await fetch(MARKET_ML_API, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
            console.error("Backend Error Details:", data); 
            throw new Error(`ML API Error: ${data.detail || response.statusText}`);
        }
        
        if (data.risk_level_predicted === 'Critical' || data.risk_level_predicted === 'High') {
            addAlert(
                data.risk_level_predicted, 
                data.probability_of_crash
            );
        }
        
        return data; 

    } catch (error) {
        console.error("Errore durante la predizione ML:", error);
        return { 
            risk_level_predicted: 'API Error', 
            probability_of_crash: 0,
            fear_greed_index: currentMetrics.fearGreedIndex, 
            twitter_sentiment: currentMetrics.twitterSentiment 
        }; 
    }
  };


  // Hawkes Process
  const calculateHawkesIntensity = (trades, currentTime) => {
    const lambda0 = 0.5;
    const alpha = 1.2;
    const beta = 2.0;
    let intensity = lambda0;
    
    trades.forEach(trade => {
      const timeDiff = (currentTime - trade.timestamp) / 1000;
      if (timeDiff < 30) {
        const contribution = alpha * Math.exp(-beta * timeDiff);
        intensity += contribution;
      }
    });
    return intensity;
  };
  
  // Modifica Alert
  const addAlert = (level, mlProbability) => {
    const lastAlert = alerts[0];
    const now = Date.now();
    if (lastAlert && now - lastAlert.timestamp < 10000) return;

    const newAlert = {
      id: now,
      timestamp: now,
      time: new Date().toLocaleTimeString(),
      level,
      message: level === 'Critical' 
         ? `🚨 ARITMIA CRITICA ML: Probabilità Crash: ${(mlProbability * 100).toFixed(1)}%`
         : `⚠️ Allerta ML Rilevata: Probabilità: ${(mlProbability * 100).toFixed(1)}%`,
      prediction: level === 'Critical' 
         ? '🎯 CRASH POTENZIALE RILEVATO: Prepararsi a movimenti rapidi (30-60s)' 
         : 'Monitoraggio intensificato'
    };
    setAlerts(prev => [newAlert, ...prev].slice(0, 5));
  };
  
  const getRiskColor = (level) => {
    const colors = {
      Normal: 'text-green-400', Medium: 'text-yellow-400', High: 'text-orange-400', 
      Critical: 'text-red-500', 'API Error': 'text-gray-500', 
      'Collecting': 'text-cyan-400', 'Initializing': 'text-gray-400'
    };
    return colors[level] || 'text-gray-400';
  };
  
  const getRiskBg = (level) => {
    const colors = {
      Normal: 'bg-green-500/10 border-green-500/30', Medium: 'bg-yellow-500/10 border-yellow-500/30', 
      High: 'bg-orange-500/10 border-orange-500/30', Critical: 'bg-red-500/20 border-red-500/50',
      'API Error': 'bg-gray-500/10 border-gray-500/30', 'Collecting': 'bg-cyan-500/10 border-cyan-500/30', 
      'Initializing': 'bg-gray-500/10 border-gray-500/30'
    };
    return colors[level] || 'bg-gray-500/10 border-gray-500/30';
  };

  const getFNGColor = (index) => {
    if (index < 20) return 'text-red-500';
    if (index < 40) return 'text-orange-400';
    if (index > 75) return 'text-green-500';
    if (index > 55) return 'text-lime-400';
    return 'text-yellow-400';
  };

  
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Heart className={`w-10 h-10 text-red-400 ${currentMetrics.riskLevel === 'Critical' ? 'animate-pulse' : ''}`} />
            <h1 className="text-4xl font-bold text-white">MARKET HEARTBEAT</h1>
          </div>
          <p className="text-gray-300 text-sm mb-2">Microstructure & Sentiment Analysis • LSTM HFT Prediction</p>
          <div className="flex items-center justify-center gap-4 text-xs">
            {/* Status indicators... */}
            <div className="flex items-center gap-2"> {isConnected ? <Wifi className="w-4 h-4 text-green-400" /> : <WifiOff className="w-4 h-4 text-red-400" />}
              <span className={isConnected ? 'text-green-400' : 'text-red-400'}>{connectionStatus}</span>
            </div>
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span className="text-purple-400">ML Sequence: {lstmSequenceRef.current.length}/{LSTM_SEQUENCE_LENGTH}</span>
            </div>
          </div>
        </div>

        {/* Real-Time Metrics (HFT & SENTIMENT) */}
        <div className="grid grid-cols-2 md:grid-cols-7 gap-3 mb-4">
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700 col-span-2 md:col-span-1">
            <div className="text-gray-400 text-xs mb-1">BTC Price</div>
            <div className="text-2xl font-bold text-white">${currentMetrics.price}</div>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <div className="text-gray-400 text-xs mb-1">OFI</div>
            <div className={`text-2xl font-bold ${parseFloat(currentMetrics.ofi) > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {currentMetrics.ofi > 0 ? '+' : ''}{currentMetrics.ofi}
            </div>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <div className="text-gray-400 text-xs mb-1">Hawkes λ(t)</div>
            <div className="text-2xl font-bold text-purple-400">{currentMetrics.hawkesIntensity}</div>
          </div>
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <div className="text-gray-400 text-xs mb-1">Spread (bps)</div>
            <div className="text-2xl font-bold text-blue-400">{currentMetrics.spreadBps}</div>
          </div>
          {/* F&G Index */}
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
                <Zap className='w-3 h-3 text-yellow-400'/> F&G Index
            </div>
            <div className={`text-2xl font-bold ${getFNGColor(currentMetrics.fearGreedIndex)}`}>
                {currentMetrics.fearGreedIndex}
            </div>
            <div className="text-xs text-gray-500">{(currentMetrics.fearGreedIndex < 50 ? 'Fear' : 'Greed')}</div>
          </div>
          {/* Twitter Sentiment */}
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
                <Twitter className='w-3 h-3 text-cyan-400'/> Twitter Score
            </div>
            <div className={`text-2xl font-bold ${currentMetrics.twitterSentiment > 0.1 ? 'text-green-400' : currentMetrics.twitterSentiment < -0.1 ? 'text-red-400' : 'text-gray-400'}`}>
                {currentMetrics.twitterSentiment.toFixed(2)}
            </div>
            <div className="text-xs text-gray-500">(-1 Panico / +1 FOMO)</div>
          </div>
          {/* Risk (ML) */}
          <div className={`rounded-lg p-4 border-2 ${getRiskBg(currentMetrics.riskLevel)}`}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-gray-300 text-xs font-semibold">Risk (LSTM)</span>
              <span className={`text-xs font-bold ${getRiskColor(currentMetrics.riskLevel)}`}>{(currentMetrics.mlProbability * 100).toFixed(1)}%</span>
            </div>
            <div className={`text-xl font-bold ${getRiskColor(currentMetrics.riskLevel)}`}>
              {currentMetrics.riskLevel}
            </div>
          </div>
        </div>

        {/* Charts Grid... */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
           {/* OFI Heartbeat */}
           <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <Heart className="w-4 h-4 text-red-400" />
              Order Flow Imbalance (Battito)
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={ofiData}>
                <defs>
                  <linearGradient id="colorOfi" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.6}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#6b7280" tick={{fontSize: 10}} />
                <YAxis stroke="#6b7280" tick={{fontSize: 10}} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', fontSize: 12 }} />
                <Area type="monotone" dataKey="ofi" stroke="#10b981" fill="url(#colorOfi)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          {/* Hawkes Intensity */}
          <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
            <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4 text-purple-400" />
              Hawkes Intensity (Aritmia)
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={ofiData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#6b7280" tick={{fontSize: 10}} />
                <YAxis stroke="#6b7280" tick={{fontSize: 10}} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', fontSize: 12 }} />
                <Line type="monotone" dataKey="hawkes_intensity" stroke="#a855f7" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Alert System */}
        <div className="bg-slate-800/80 rounded-lg p-4 border border-slate-700">
          <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-yellow-400" />
            Sistema di Allerta (Predizione LSTM - Microstructure + Sentiment)
          </h3>
          <div className="space-y-2">
            {alerts.length === 0 ? (
              <div className="text-gray-400 text-sm text-center py-3">Sistema in monitoraggio continuo...</div>
            ) : (
              alerts.map(alert => (
                <div key={alert.id} className={`p-3 rounded-lg border-l-4 text-sm ${
                  alert.level === 'Critical' 
                     ? 'bg-red-500/10 border-red-500' 
                     : 'bg-orange-500/10 border-orange-500'
                }`}>
                  <div className="flex justify-between items-start">
                    <div className="flex-1">
                      <div className="font-semibold text-white">{alert.message}</div>
                      <div className="text-xs text-gray-400 mt-1">{alert.prediction}</div>
                    </div>
                    <div className="text-xs text-gray-400 ml-2">{alert.time}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
        
      </div>
    </div>
  );
};

export default MarketHeartbeat;