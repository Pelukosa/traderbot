#!/usr/bin/env python3
"""
Analisis detallado de las 6 estrategias principales.
Muestra por cada una: % por operacion, tiempo medio, ROI diario/mensual, etc.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).parent.parent / "historical_1h.csv"
SIM_DIR = Path(__file__).parent.parent / "simulations"
INITIAL_EUR = 120.0
INVEST_PCT = 95.0
TRAIL_RETAIN = 0.6
MAX_CANDLES = 12


def ema(data, period):
    r = np.empty_like(data); r[:] = np.nan
    first = np.where(~np.isnan(data))[0]
    if len(first)==0: return r
    s=first[0]+period-1
    if s>=len(data): return r
    r[s]=np.mean(data[s-period+1:s+1])
    m=2.0/(period+1)
    for i in range(s+1,len(data)): r[i]=(data[i]-r[i-1])*m+r[i-1]
    return r

def sma(data, period):
    r=np.empty_like(data); r[:]=np.nan
    for i in range(period-1,len(data)): r[i]=np.mean(data[i-period+1:i+1])
    return r

def std(data, period):
    r=np.empty_like(data); r[:]=np.nan
    for i in range(period-1,len(data)): r[i]=np.std(data[i-period+1:i+1])
    return r

def rsi_from_closes(closes, period=14):
    deltas=np.diff(closes); rsi=np.full_like(closes,np.nan)
    if len(deltas)<period: return rsi
    seed=deltas[:period]
    up=seed[seed>=0].sum()/period
    down=-seed[seed<0].sum()/period
    rs=up/down if down!=0 else 0
    rsi[period]=100-100/(1+rs)
    for i in range(period+1,len(closes)):
        d=deltas[i-1]
        upval=d if d>0 else 0; downval=-d if d<0 else 0
        up=(up*(period-1)+upval)/period; down=(down*(period-1)+downval)/period
        rs=up/down if down!=0 else 0
        rsi[i]=100-100/(1+rs)
    return rsi

def atr(ohlcv, period=14):
    r=np.full(len(ohlcv),np.nan)
    for i in range(1,len(ohlcv)):
        h,l,pc=ohlcv[i][2],ohlcv[i][3],ohlcv[i-1][4]
        r[i]=max(h-l,abs(h-pc),abs(l-pc))
    return ema(r,period)


def load_data():
    rows=[]
    with open(DATA_FILE) as f:
        for r in csv.DictReader(f):
            rows.append([int(r['timestamp']),float(r['open']),float(r['high']),float(r['low']),float(r['close']),float(r['volume'])])
    return rows


def simulate(ohlcv, gen_fn, fee_pct, sl_pct, trail_min, **extra_params):
    """Devuelve lista de trades con detalle: (entrada, salida, %ganancia, velas_en_posicion, tipo_salida)"""
    fee_rate = fee_pct / 100.0
    sl_rate = sl_pct / 100.0
    
    closes = np.array([c[4] for c in ohlcv])
    vol = np.array([c[5] for c in ohlcv])
    highs = np.array([c[2] for c in ohlcv])
    lows = np.array([c[3] for c in ohlcv])
    
    # Precomputar indicadores comunes
    ema20 = ema(closes, 20)
    rsi_arr = rsi_from_closes(closes)
    bb_mid = sma(closes, 20)
    bb_std = std(closes, 20)
    bb_lower = bb_mid - bb_std * 2.0
    atr_arr = atr(ohlcv)
    vol_sma20 = sma(vol, 20)
    ema_f = ema(closes, 12); ema_s = ema(closes, 26)
    macd_line = ema_f - ema_s; sig_line = ema(macd_line, 9)
    hist_arr = macd_line - sig_line
    
    trades_detail = []
    eur = INITIAL_EUR
    btc = 0.0
    in_pos = False
    entry_p = 0.0; entry_i = 0
    highest = 0.0; stop_loss = None
    total_fees = 0.0
    
    for i in range(len(ohlcv)):
        close = float(closes[i])
        
        # Precomputar buffer de histograma para MACD
        h = []
        for j in range(max(0,i-4), i+1):
            if not np.isnan(hist_arr[j]): h.append(float(hist_arr[j]))
            else: h = []
            if len(h)>5: h = h[-5:]
        
        ind = {
            'in_pos': in_pos, 'ema20': ema20, 'vol_avg': vol_sma20,
            'volumes': vol, 'highs': highs, 'lows': lows,
            'rsi_arr': rsi_arr, 'atr_arr': atr_arr, 'hist_buf': h,
        }
        
        if in_pos:
            # SL
            if stop_loss is not None and close <= stop_loss:
                pnl_pct = (close - entry_p) / entry_p * 100
                fee = btc * close * fee_rate
                total_fees += fee
                eur += btc * close - fee
                trades_detail.append((entry_p, close, pnl_pct, i-entry_i, 'SL'))
                btc=0.0; in_pos=False; continue
            
            # Trailing
            if close > highest:
                highest = close
                gain = (close-entry_p)/entry_p*100
                if gain > trail_min:
                    trail_pct = gain * TRAIL_RETAIN
                    new_sl = entry_p * (1+trail_pct/100.0)
                    if stop_loss is None or new_sl > stop_loss: stop_loss = new_sl
            
            # Max time
            if i - entry_i >= MAX_CANDLES:
                pnl_pct = (close - entry_p)/entry_p*100
                fee = btc*close*fee_rate
                total_fees += fee
                eur += btc*close-fee
                trades_detail.append((entry_p,close,pnl_pct,i-entry_i,'MAX_TIME'))
                btc=0.0; in_pos=False; continue
        
        # Generar senal
        cfg = {**extra_params, 'sl_percent': sl_pct, 'trailing_min_gain': trail_min, 'fee_percent': fee_pct}
        sig = gen_fn(closes, ind, i, cfg)
        
        if sig and sig['action']=='buy' and not in_pos:
            invest = eur * INVEST_PCT / 100.0
            if invest >= 5.0:
                buy_fee = invest * fee_rate
                total_fees += buy_fee
                btc = invest / close
                eur -= invest + buy_fee
                entry_p = close; entry_i = i
                highest = close
                stop_loss = entry_p * (1 - sl_rate)
                in_pos = True
    
    # Cerrar ultima si quedo abierta
    if in_pos:
        last_c = float(closes[-1])
        pnl_pct = (last_c - entry_p)/entry_p*100
        fee = btc*last_c*fee_rate
        total_fees += fee
        trades_detail.append((entry_p,last_c,pnl_pct,len(ohlcv)-entry_i,'END'))
    
    return trades_detail, total_fees


def gen_macd(closes, ind, i, cfg):
    h = ind.get('hist_buf',[])
    if len(h)<4: return None
    min_h = cfg.get('min_histogram_abs',60)
    if h[-4]>=h[-3]>h[-2]<h[-1] and h[-2]<0 and h[-1]<0 and abs(h[-2])>=min_h:
        return {'action':'buy'}
    return None

def gen_macd_rsi(closes, ind, i, cfg):
    h = ind.get('hist_buf',[])
    rsi = ind.get('rsi_arr',[])
    if len(h)<4: return None
    min_h=cfg.get('min_histogram_abs',60); rsi_max=cfg.get('rsi_max',40)
    rsi_val=float(rsi[i]) if i<len(rsi) and not np.isnan(rsi[i]) else 999
    if h[-4]>=h[-3]>h[-2]<h[-1] and h[-2]<0 and h[-1]<0 and abs(h[-2])>=min_h and rsi_val<rsi_max:
        return {'action':'buy'}
    # Venta: pico de MACD
    if h[-4]<=h[-3]<h[-2]>h[-1] and h[-2]>0 and h[-1]>0 and abs(h[-2])>=min_h:
        return {'action':'sell'}
    return None

def gen_vwap(closes,ind,i,cfg):
    return None  # No implementado aqui

def gen_rsi(closes,ind,i,cfg):
    return None

def gen_bb(closes,ind,i,cfg):
    return None


# Estrategias definidas
ESTRATEGIAS = {
    'MACD+RSI<30': {
        'gen': gen_macd_rsi,
        'config': {'min_histogram_abs':60, 'sl_percent':4.95, 'trailing_min_gain':1.05, 'fee_percent':0.0, 'rsi_max':40},
    },
    'MACD puro': {
        'gen': gen_macd,
        'config': {'min_histogram_abs':60, 'sl_percent':4.95, 'trailing_min_gain':1.05, 'fee_percent':0.0},
    },
}


def main():
    ohlcv = load_data()
    closes = np.array([c[4] for c in ohlcv])
    print(f'Datos: {len(ohlcv)} velas 1h ({len(ohlcv)//24:.1f} dias)')
    print(f'Capital: {INITIAL_EUR}€')
    print()
    
    for nombre, e in sorted(ESTRATEGIAS.items()):
        cfg = e['config']
        trades, fees = simulate(ohlcv, e['gen'], fee_pct=cfg.get('fee_percent',0),
                                sl_pct=cfg['sl_percent'], trail_min=cfg['trailing_min_gain'],
                                **{k:v for k,v in cfg.items() if k not in ('sl_percent','trailing_min_gain','fee_percent')})
        
        num = len(trades)
        if num==0:
            print(f'{nombre}: 0 operaciones')
            continue
        
        pct_ganancias = [t[2] for t in trades]
        duraciones = [t[3] for t in trades]
        ganancias = [t[2] for t in trades]
        wins = [g for g in ganancias if g>0]
        losses = [g for g in ganancias if g<=0]
        wr = len(wins)/num*100
        
        pnl_bruto = sum(ganancias) / 100.0 * INITIAL_EUR
        pnl_neto = pnl_bruto - fees
        
        dias = len(ohlcv)/24
        ops_mes = num / dias * 30
        roi_mensual_bruto = (pnl_bruto/INITIAL_EUR)*100 / dias * 30
        roi_mensual_neto = (pnl_neto/INITIAL_EUR)*100 / dias * 30
        roi_diario_neto = roi_mensual_neto / 30
        
        print(f'=== {nombre} ===')
        print(f'  Operaciones: {num} ({ops_mes:.1f}/mes)')
        print(f'  Win rate: {wr:.1f}%')
        print(f'  Ganancia media por operacion: {np.mean(ganancias):+.2f}%')
        print(f'  Ganancia media de las ganadoras: {np.mean(wins):+.2f}%' if wins else '  Sin ganadoras')
        print(f'  Perdida media de las perdedoras: {np.mean(losses):+.2f}%' if losses else '  Sin perdedoras')
        print(f'  Mayor ganancia: {max(ganancias):+.2f}%')
        print(f'  Mayor perdida: {min(ganancias):+.2f}%')
        print(f'  Tiempo medio en posicion: {np.mean(duraciones):.1f} velas ({np.mean(duraciones)*1:.1f}h)')
        print(f'  Tiempo maximo: {max(duraciones)}h')
        print(f'  Comisiones totales: {fees:.2f}€')
        print(f'  ROI bruto mensual: {roi_mensual_bruto:+.2f}%')
        print(f'  ROI neto mensual: {roi_mensual_neto:+.2f}%')
        print(f'  ROI diario neto: {roi_diario_neto:+.2f}%')
        print(f'  PnL neto mensual estimado: {pnl_neto/dias*30:+.2f}€')
        print(f'  PnL por operacion: {pnl_neto/num:+.2f}€')
        print()
        
        # Detalle de las ultimas 10 operaciones como ejemplo
        print(f'  Ultimas operaciones:')
        print(f'  #  Entrada   Salida    %      Dur')
        for idx, t in enumerate(trades[-10:], max(1,num-9)):
            print(f'  {idx:<3} {t[0]:.0f}   {t[1]:.0f}   {t[2]:+6.2f}%  {t[3]:.0f}h {t[4]}')


if __name__=='__main__':
    main()
