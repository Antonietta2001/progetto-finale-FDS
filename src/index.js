import React from 'react';
import ReactDOM from 'react-dom/client';
import MarketHeartbeat from './MarketHeartbeat';

// Importa il CSS (se ne hai uno, altrimenti puoi omettere questa riga)
// import './index.css'; 

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MarketHeartbeat />
  </React.StrictMode>
);