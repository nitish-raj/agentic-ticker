import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any


def create_price_chart(price_data: List[Dict[str, Any]], indicator_data: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated price chart using Plotly."""
    price_df = pd.DataFrame(price_data)
    ind_df = pd.DataFrame(indicator_data)
    
    if not price_df.empty and 'date' in price_df.columns:
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_df = price_df.sort_values('date')
    if not ind_df.empty and 'date' in ind_df.columns:
        ind_df['date'] = pd.to_datetime(ind_df['date'])
        ind_df = ind_df.sort_values('date')
    
    # Create animated figure
    fig = go.Figure()
    
    # Add price line
    if not price_df.empty:
        fig.add_trace(go.Scatter(
            x=price_df['date'],
            y=price_df['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add moving averages
    if not ind_df.empty:
        fig.add_trace(go.Scatter(
            x=ind_df['date'],
            y=ind_df['ma5'],
            mode='lines',
            name='MA5',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>MA5: $%{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=ind_df['date'],
            y=ind_df['ma10'],
            mode='lines',
            name='MA10',
            line=dict(color='#2ca02c', width=2, dash='dot'),
            hovertemplate='Date: %{x}<br>MA10: $%{y:.2f}<extra></extra>'
        ))
    
    # Create animation frames that match the initial chart structure
    frames = []
    if not price_df.empty:
        for i in range(len(price_df)):
            frame_data = []
            
            # Add price data up to current point (start from first frame)
            frame_data.append(go.Scatter(
                x=price_df['date'][:i+1],
                y=price_df['close'][:i+1],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=3),
                hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            
            # Add indicator data up to current point (synchronized with price data)
            if not ind_df.empty:
                # Use the same number of points as price data for synchronization
                ind_points = min(i+1, len(ind_df))
                if ind_points > 0:
                    frame_data.append(go.Scatter(
                        x=ind_df['date'][:ind_points],
                        y=ind_df['ma5'][:ind_points],
                        mode='lines',
                        name='MA5',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>MA5: $%{y:.2f}<extra></extra>'
                    ))
                    
                    frame_data.append(go.Scatter(
                        x=ind_df['date'][:ind_points],
                        y=ind_df['ma10'][:ind_points],
                        mode='lines',
                        name='MA10',
                        line=dict(color='#2ca02c', width=2, dash='dot'),
                        hovertemplate='Date: %{x}<br>MA10: $%{y:.2f}<extra></extra>'
                    ))
            
            frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Update layout with animation controls
    fig.update_layout(
        title='📈 Price & Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[None, {"frame": {"duration": 500, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 300}}],
                        label="▶️ Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
                        label="⏸️ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": "Frame: "},
                transition={"duration": 300},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 300}}],
                        label=f.name,
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ] if frames else []
    )
    
    if frames:
        fig.frames = frames
    
    return fig


def create_forecast_chart(forecasts: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated forecast chart using Plotly."""
    if not forecasts:
        # Return empty figure if no forecasts
        fig = go.Figure()
        fig.update_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        )
        return fig
    
    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        # Return empty figure if no valid forecast data
        fig = go.Figure()
        fig.update_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        )
        return fig
    
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    forecast_df = forecast_df.sort_values('date')
    
    # Handle missing or invalid values
    forecast_df['confidence'] = forecast_df['confidence'].apply(lambda x: 0.5 if pd.isna(x) or x is None else x)
    forecast_df['forecast_price'] = forecast_df['forecast_price'].apply(lambda x: 0.0 if pd.isna(x) or x is None else x)
    
    # Drop any rows with invalid data
    forecast_df = forecast_df.dropna()
    
    # Create animated forecast chart
    fig = go.Figure()
    
    # Create a continuous line chart with confidence-based coloring
    # Determine overall trend for line color
    overall_trend = forecast_df['trend'].iloc[-1] if not forecast_df.empty else 'NEUTRAL'
    avg_confidence = forecast_df['confidence'].mean() if not forecast_df.empty else 0.5
    
    # Color based on overall trend and average confidence
    if overall_trend == 'UP':
        line_color = '#2ca02c' if avg_confidence > 0.7 else '#98df8a'
    elif overall_trend == 'DOWN':
        line_color = '#d62728' if avg_confidence > 0.7 else '#ff9999'
    else:
        line_color = '#ff7f0e'
    
    # Add main forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast_price'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color=line_color, width=3),
        marker=dict(
            size=8,
            color=line_color,
            symbol='circle'
        ),
        hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
    ))
    
    # Add confidence band as filled area
    if len(forecast_df) > 1:
        # Create upper and lower confidence bounds using list comprehension to handle None values
        confidence_values = [0.5 if pd.isna(x) or x is None else x for x in forecast_df['confidence']]
        forecast_prices = [0.0 if pd.isna(x) or x is None else x for x in forecast_df['forecast_price']]
        
        upper_bound = [fp * (1 + (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(forecast_prices, confidence_values)]
        lower_bound = [fp * (1 - (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(forecast_prices, confidence_values)]
        
        # Convert hex color to rgba for transparency
        if line_color == '#2ca02c':  # Green
            fill_color = 'rgba(44, 160, 44, 0.2)'
        elif line_color == '#d62728':  # Red
            fill_color = 'rgba(214, 39, 40, 0.2)'
        elif line_color == '#98df8a':  # Light green
            fill_color = 'rgba(152, 223, 138, 0.2)'
        elif line_color == '#ff9999':  # Light red
            fill_color = 'rgba(255, 153, 153, 0.2)'
        else:  # Orange
            fill_color = 'rgba(255, 127, 14, 0.2)'
        
        # Add confidence band
        fig.add_trace(go.Scatter(
            x=list(forecast_df['date']) + list(forecast_df['date'])[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band',
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Individual points are now included in the main line chart with markers
    
# Create animation frames that match the initial chart style
    frames = []
    for i in range(len(forecast_df)):
        frame_data = []
        
        # Get data up to current frame
        current_df = forecast_df.iloc[:i+1]
        
        if len(current_df) > 0:
            # Calculate trend and confidence for current segment
            current_trend = current_df['trend'].iloc[-1] if not current_df.empty else 'NEUTRAL'
            current_avg_confidence = current_df['confidence'].mean() if not current_df.empty else 0.5
            
            # Color based on current trend and confidence
            if current_trend == 'UP':
                current_line_color = '#2ca02c' if current_avg_confidence > 0.7 else '#98df8a'
            elif current_trend == 'DOWN':
                current_line_color = '#d62728' if current_avg_confidence > 0.7 else '#ff9999'
            else:
                current_line_color = '#ff7f0e'
            
            # Add main forecast line for current segment
            frame_data.append(go.Scatter(
                x=current_df['date'],
                y=current_df['forecast_price'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=current_line_color, width=3),
                marker=dict(
                    size=8,
                    color=current_line_color,
                    symbol='circle'
                ),
                hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
            ))
            
            # Add confidence band for current segment if we have more than 1 point
            if len(current_df) > 1:
                # Create upper and lower confidence bounds
                current_confidence_values = [0.5 if pd.isna(x) or x is None else x for x in current_df['confidence']]
                current_forecast_prices = [0.0 if pd.isna(x) or x is None else x for x in current_df['forecast_price']]
                
                current_upper_bound = [fp * (1 + (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(current_forecast_prices, current_confidence_values)]
                current_lower_bound = [fp * (1 - (1 - cv) * 0.1) if fp is not None and cv is not None else 0 for fp, cv in zip(current_forecast_prices, current_confidence_values)]
                
                # Convert hex color to rgba for transparency
                if current_line_color == '#2ca02c':  # Green
                    current_fill_color = 'rgba(44, 160, 44, 0.2)'
                elif current_line_color == '#d62728':  # Red
                    current_fill_color = 'rgba(214, 39, 40, 0.2)'
                elif current_line_color == '#98df8a':  # Light green
                    current_fill_color = 'rgba(152, 223, 138, 0.2)'
                elif current_line_color == '#ff9999':  # Light red
                    current_fill_color = 'rgba(255, 153, 153, 0.2)'
                else:  # Orange
                    current_fill_color = 'rgba(255, 127, 14, 0.2)'
                
                # Add confidence band
                frame_data.append(go.Scatter(
                    x=list(current_df['date']) + list(current_df['date'])[::-1],
                    y=current_upper_bound + current_lower_bound[::-1],
                    fill='toself',
                    fillcolor=current_fill_color,
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Confidence Band',
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        frames.append(go.Frame(data=frame_data, name=str(i)))
    
    # Update layout with animation controls
    fig.update_layout(
        title='🔮 Price Forecasts',
        xaxis_title='Date',
        yaxis_title='Forecast Price ($)',
        hovermode='x unified',
        showlegend=True,
        height=500,
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[None, {"frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True, "transition": {"duration": 400}}],
                        label="▶️ Play",
                        method="animate"
                    ),
                    dict(
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}],
                        label="⏸️ Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={"prefix": "Day: "},
                transition={"duration": 400},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[[f.name], {"frame": {"duration": 400, "redraw": True},
                                        "mode": "immediate", "transition": {"duration": 400}}],
                        label=f"Day {int(f.name)+1}",
                        method="animate"
                    )
                    for f in frames
                ]
            )
        ] if frames else []
    )
    
    if frames:
        fig.frames = frames
    
    return fig




