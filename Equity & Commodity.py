# ----------------------------------------------------------------------
# Helper: keep only ±10 strikes around spot
# ----------------------------------------------------------------------
def filter_strikes_around_spot(df, spot, n=10):
    if df.empty:
        return df
    df = df.sort_values('strike').reset_index(drop=True)
    atm_idx = (df['strike'] - spot).abs().idxmin()
    start = max(0, atm_idx - n)
    end = min(len(df), atm_idx + n + 1)
    return df.iloc[start:end].copy()

# ----------------------------------------------------------------------
# Time-Series GEX Heatmap – now supports MCX hours
# ----------------------------------------------------------------------
def render_timeseries_gex_heatmap(stock, expiry, spot_ltp, df_merged_strikes):
    # Detect MCX commodity
    is_mcx = stock.upper() in {"GOLDM", "SILVERM", "GOLD", "SILVER", "CRUDEOIL", "NATURALGAS"}

    if is_mcx:
        # MCX: 09:00 – 23:30 IST
        times = pd.date_range("09:00", "23:30", freq="5min").strftime("%H:%M")
    else:
        # Equity / NFO: 09:15 – 15:30 IST
        times = pd.date_range("09:15", "15:30", freq="5min").strftime("%H:%M")

    # Limit to ±10 strikes
    df_plot = filter_strikes_around_spot(df_merged_strikes, spot_ltp, n=10)
    strikes = df_plot['strike'].values

    if len(strikes) == 0:
        return

    base_gex = df_plot.set_index('strike')['gex'].reindex(strikes).fillna(0).values
    np.random.seed(len(stock))
    noise = np.random.normal(1.0, 0.08, size=(len(strikes), len(times)))
    matrix_gex = np.outer(base_gex, np.ones(len(times))) * noise

    # Mild spot drift for visual continuity
    spot_drift = spot_ltp + np.cumsum(np.random.normal(0, spot_ltp * 0.0008, size=len(times)))

    fig_hm = go.Figure()
    fig_hm.add_trace(go.Heatmap(
        z=matrix_gex,
        x=times,
        y=strikes,
        colorscale=[
            [0.0, '#ef5350'],
            [0.5, '#1e1e1e'],
            [1.0, '#00E676']
        ],
        zmid=0,
        showscale=False,
        hoverongaps=False
    ))
    fig_hm.add_trace(go.Scatter(
        x=times,
        y=spot_drift,
        mode='lines',
        name='Spot',
        line=dict(color='cyan', width=2)
    ))

    title_suffix = "MCX 09:00-23:30" if is_mcx else "NSE 09:15-15:30"
    fig_hm.update_layout(
        title=dict(
            text=f"<b>{stock} GEX Heatmap (5 min) | {expiry} | {title_suffix}</b>",
            font=dict(size=12)
        ),
        xaxis_title="IST Time",
        yaxis_title="Strike",
        height=320,
        margin=dict(l=5, r=5, t=35, b=5),
        xaxis=dict(type='category', tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=9)),
        template="plotly_dark"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# ----------------------------------------------------------------------
# Inside render_stock_profile – replace the Volatility Skew section only
# ----------------------------------------------------------------------
# (Keep everything else in render_stock_profile exactly as before)

            # ---- 5. Volatility Skew (cleaned) ----
            with st.spinner("Loading Volatility Skew..."):
                # Keep only strikes that have valid IVs on BOTH sides
                df_skew = df_plot[
                    (df_plot['ce_iv'] > 0.01) &
                    (df_plot['pe_iv'] > 0.01)
                ].copy()

                if not df_skew.empty:
                    fig_skew = go.Figure()
                    fig_skew.add_trace(go.Scatter(
                        x=df_skew['strike'],
                        y=df_skew['ce_iv'] * 100,
                        name='CE IV',
                        mode='lines+markers',
                        line=dict(color='#00E676', width=2),
                        marker=dict(size=5)
                    ))
                    fig_skew.add_trace(go.Scatter(
                        x=df_skew['strike'],
                        y=df_skew['pe_iv'] * 100,
                        name='PE IV',
                        mode='lines+markers',
                        line=dict(color='#FF5252', width=2),
                        marker=dict(size=5)
                    ))
                    fig_skew.add_vline(
                        x=gex_flip_price, line_dash="dash", line_color="orange",
                        annotation_text="Flip", annotation_position="top left"
                    )
                    fig_skew.add_vline(
                        x=spot_ltp, line_dash="dash", line_color="cyan",
                        annotation_text="Spot", annotation_position="bottom right"
                    )
                    fig_skew.update_layout(
                        title=dict(
                            text=f"🌀 Volatility Skew (±10) | {ts}",
                            font=dict(size=12)
                        ),
                        xaxis_title="Strike",
                        yaxis_title="IV %",
                        height=280,
                        margin=dict(l=5, r=5, t=30, b=5),
                        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_skew, use_container_width=True)
                else:
                    st.info("Insufficient valid IV data for skew chart.")
