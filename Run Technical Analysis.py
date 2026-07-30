import os
import pandas as pd
from dash import Dash, html, dcc, Output, Input
from plotly.subplots import make_subplots
from SmartApi import SmartConnect

# ==================== CREDENTIALS ====================
API_KEY = os.getenv("API_KEY", "o2b7s4Oo")
CLIENT_CODE = os.getenv("CLIENT_CODE", "AACK311190")
PASSWORD = os.getenv("PASSWORD", "8547")
TOTP_SECRET = os.getenv("TOTP_SECRET", "YCRQCDQ7NPUHKYH7RS73NXQ5VE")
SCRIP_MASTER_FILE = "scrip_master.json"

selected_symbol = "NIFTY 50"
STD_PARAMS = {'LOT_SIZE': 25, 'SL_PCT': 0.002, 'RR_RATIO': 1.5}
# ==========================================================

def fetch_today_or_previous():
    # Placeholder or actual fetch function
    return pd.DataFrame(), False


# ==================== OPTIONS & P&L CALCULATOR MODULE ====================

def generate_options_dashboard_cards(df):
    if df.empty:
        return html.Div("No Data Available", style={'color': 'white'})

    curr_spot = df['Close'].iloc[-1] if 'Close' in df.columns else 22000.0
    curr_regime = "SIDEWAYS"
    last_signal = 0
    trade_type = "WAITING"
    opt_symbol = "NIFTY ATM"

    call_short_strike = round((curr_spot + 500) / 50) * 50
    call_hedge_strike = call_short_strike + 200
    put_short_strike = round((curr_spot - 500) / 50) * 50
    put_hedge_strike = put_short_strike - 200

    # Estimated premiums & Stop-Losses
    est_call_short_prem = 3.50
    est_call_hedge_prem = 1.70
    est_put_short_prem = 19.00
    est_put_hedge_prem = 9.25

    net_credit_pts = (est_call_short_prem - est_call_hedge_prem) + (est_put_short_prem - est_put_hedge_prem)
    net_credit_rupees = net_credit_pts * STD_PARAMS['LOT_SIZE']
    ic_sl_pts = net_credit_pts * 1.5

    # Estimate Live P&L for Iron Condor
    if put_short_strike <= curr_spot <= call_short_strike:
        ic_pnl_pts = net_credit_pts * 0.85
        ic_status_color = "#00e676"
    else:
        breach_dist = max(curr_spot - call_short_strike, put_short_strike - curr_spot)
        ic_pnl_pts = net_credit_pts - (breach_dist * 0.5)
        ic_status_color = "#ff1744" if ic_pnl_pts < 0 else "#00e676"

    # --- ATM DIRECTIONAL SIGNAL & OPTION SL/TP ---
    spot_sl_dist = curr_spot * STD_PARAMS['SL_PCT']
    spot_tp_dist = spot_sl_dist * STD_PARAMS['RR_RATIO']

    option_delta = 0.50
    entry_prem = 120.00
    option_sl_dist = spot_sl_dist * option_delta
    option_tp_dist = spot_tp_dist * option_delta

    sl_prem = round(entry_prem - option_sl_dist, 2)
    tp_prem = round(entry_prem + option_tp_dist, 2)

    entry_spot = df[df['Combined_Sig'] == 1].iloc[-1]['Close'] if ('Combined_Sig' in df.columns and (df['Combined_Sig'] == 1).any()) else curr_spot
    spot_change = curr_spot - entry_spot
    opt_pnl_pts = spot_change * option_delta if last_signal != 0 else 0.0
    opt_pnl_rupees = opt_pnl_pts * STD_PARAMS['LOT_SIZE']

    is_active = (last_signal != 0)

    return html.Div(style={'backgroundColor': '#1e1e1e', 'border': '1px solid #ffd700', 'borderRadius': '8px', 'padding': '15px', 'marginBottom': '20px'}, children=[
        html.H4("💡 LIVE OPTIONS TRADING SUGGESTIONS & P&L TRACKER", style={'color': '#ffd700', 'marginTop': '0', 'textAlign': 'center'}),
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}, children=[

            # Left Card: Iron Condor Position
            html.Div(style={'flex': '1', 'minWidth': '320px', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '6px', 'margin': '5px'}, children=[
                html.H5("🗓️ 1-Day Expiry Strategy (Iron Condor)", style={'color': '#00e676', 'marginTop': '0'}),
                html.Div(f"Current NIFTY Spot Price: {curr_spot:.2f}", style={'color': '#fff', 'fontWeight': 'bold'}),
                html.Hr(style={'borderColor': '#444'}),
                html.Ul(style={'color': '#ccc', 'fontSize': '12px', 'paddingLeft': '18px'}, children=[
                    html.Li([f"SELL Call Leg: ", html.B(f"{call_short_strike} CE"), f" | Premium: ~₹{est_call_short_prem:.2f} | ", html.Span(f"Leg SL: ₹{est_call_short_prem*2:.2f}", style={'color': '#ff1744'})]),
                    html.Li([f"BUY Call Hedge: ", html.B(f"{call_hedge_strike} CE"), f" | Premium: ~₹{est_call_hedge_prem:.2f}"]),
                    html.Li([f"SELL Put Leg: ", html.B(f"{put_short_strike} PE"), f" | Premium: ~₹{est_put_short_prem:.2f} | ", html.Span(f"Leg SL: ₹{est_put_short_prem*2:.2f}", style={'color': '#ff1744'})]),
                    html.Li([f"BUY Put Hedge: ", html.B(f"{put_hedge_strike} PE"), f" | Premium: ~₹{est_put_hedge_prem:.2f}"]),
                ]),
                html.Div(style={'backgroundColor': '#121212', 'padding': '10px', 'borderRadius': '4px', 'marginTop': '10px'}, children=[
                    html.Div(f"Est. Net Credit: {net_credit_pts:.2f} pts (₹{net_credit_rupees:.2f}/lot)", style={'color': '#00e676', 'fontWeight': 'bold'}),
                    html.Div(f"Iron Condor SL: {ic_sl_pts:.2f} pts loss", style={'color': '#ff1744', 'fontSize': '11px'}),
                    html.Div(f"ESTIMATED IC P&L: {ic_pnl_pts:+.2f} pts", style={'color': ic_status_color, 'fontSize': '13px', 'fontWeight': 'bold', 'marginTop': '5px'})
                ])
            ]),

            # Right Card: Directional Signal
            html.Div(style={'flex': '1', 'minWidth': '320px', 'backgroundColor': '#2a2a2a', 'padding': '15px', 'borderRadius': '6px', 'margin': '5px'}, children=[
                html.H5("⚡ Live Directional Signal (ATM Strategy)", style={'color': '#29b6f6', 'marginTop': '0'}),
                html.Div([f"Market Regime: ", html.B(curr_regime, style={'color': '#ffea00' if curr_regime == 'SIDEWAYS' else '#00e676'})]),
                html.Div([f"Signal Action: ", html.B(trade_type, style={'color': '#00e676' if last_signal == 1 else '#ff1744' if last_signal == -1 else '#aaa'})]),
                html.Hr(style={'borderColor': '#444'}),
                html.Div(children=[
                    html.Div([f"Target ATM Option: ", html.B(opt_symbol, style={'color': '#29b6f6'})]),
                    html.Div([f"Est. Entry Premium: ", html.B(f"₹{entry_prem:.2f}")]),
                    html.Div([f"Option SL Target: ", html.B(f"₹{sl_prem:.2f}", style={'color': '#ff1744'}), f" | Option TP Target: ", html.B(f"₹{tp_prem:.2f}", style={'color': '#00e676'})]),
                    html.Div(f"Spot SL Distance: {spot_sl_dist:.2f} pts | Spot TP Distance: {spot_tp_dist:.2f} pts", style={'color': '#888', 'fontSize': '10px'}),
                    html.Hr(style={'borderColor': '#333', 'margin': '5px 0'}),
                    html.Div(
                        f"ESTIMATED TRADE P&L: {opt_pnl_pts:+.2f} pts ({opt_pnl_rupees:+.2f} ₹/lot)" if is_active else "ESTIMATED TRADE P&L: ₹0.00 (WAITING FOR SIGNAL)",
                        style={'color': '#00e676' if opt_pnl_pts >= 0 else '#ff1744', 'fontSize': '14px', 'fontWeight': 'bold'}
                    )
                ])
            ])
        ])
    ])


# Dash Web UI Layout
app = Dash(__name__)
server = app.server  # Required for Cloud Hosting (Gunicorn)

app.layout = html.Div(style={'backgroundColor': '#121212', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    html.H2(f"Dynamic Regime Strategy Engine: {selected_symbol}", style={'color': '#ffffff', 'textAlign': 'center'}),
    html.Div(id='options-trading-banner'),
    html.Div(id='strategy-performance-cards', style={'display': 'flex', 'justify': 'space-around', 'flexWrap': 'wrap', 'marginBottom': '20px'}),
    dcc.Graph(id='multi-indicator-graph'),
    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)
])


@app.callback(
    [Output('multi-indicator-graph', 'figure'),
     Output('strategy-performance-cards', 'children'),
     Output('options-trading-banner', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    df, _ = fetch_today_or_previous()
    options_banner = generate_options_dashboard_cards(df)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.22, 0.23]
    )

    cards = [html.Div("System Active", style={'color': '#00e676'})]

    return fig, cards, options_banner


if __name__ == '__main__':
    # Dynamically bind host and port for Cloud Deployment
    port = int(os.environ.get("PORT", 8050))
    app.run(host='0.0.0.0', port=port, debug=False)
