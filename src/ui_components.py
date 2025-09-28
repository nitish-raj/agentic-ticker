import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional

from .chart_utils import (
    preprocess_dataframe, get_trend_color, hex_to_rgba, 
    create_animation_controls, create_chart_layout, 
    create_price_traces, create_forecast_traces, create_animation_frames
)
from .decorators import handle_errors, log_execution, time_execution, validate_inputs
from .date_utils import safe_to_datetime, sort_by_date
from .validation_utils import clean_numeric_data, validate_dataframe, sanitize_forecast_data


@handle_errors(default_return=go.Figure())
@log_execution(include_args=False)
@validate_inputs(price_data='list_of_dicts', indicator_data='list_of_dicts')
def create_price_chart(price_data: List[Dict[str, Any]], indicator_data: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated price chart using Plotly."""
    price_df = pd.DataFrame(price_data)
    ind_df = pd.DataFrame(indicator_data)
    
    # Use utility functions for preprocessing
    price_df = preprocess_dataframe(price_df)
    ind_df = preprocess_dataframe(ind_df)
    
    # Create animated figure
    fig = go.Figure()
    
    # Use utility function to create traces
    traces = create_price_traces(price_df, ind_df)
    for trace in traces:
        fig.add_trace(trace)
    
    # Create animation frames using utility function
    frames = []
    if not price_df.empty:
        frames = create_animation_frames(
            price_df, 
            create_price_traces, 
            ind_df
        )
    
    # Use utility functions for layout and animation controls
    layout_config = create_chart_layout(
        title='📈 Price & Moving Averages',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500
    )
    
    animation_config = create_animation_controls(
        duration=500,
        transition_duration=300,
        button_prefix="Frame: "
    )
    
    # Update slider steps with frames
    if frames:
        animation_config['sliders'][0]['steps'] = [
            dict(
                args=[[f.name], {"frame": {"duration": 300, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 300}}],
                label=f.name,
                method="animate"
            )
            for f in frames
        ]
    
    # Merge layout and animation configs
    layout_config.update(animation_config)
    fig.update_layout(**layout_config)
    
    if frames:
        fig.frames = frames
    
    return fig


@handle_errors(default_return=go.Figure())
@log_execution(include_args=False)
@validate_inputs(forecasts='list_of_dicts')
def create_forecast_chart(forecasts: List[Dict[str, Any]]) -> go.Figure:
    """Create an animated forecast chart using Plotly."""
    if not forecasts:
        # Return empty figure if no forecasts
        fig = go.Figure()
        fig.update_layout(**create_chart_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        ))
        return fig
    
    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        # Return empty figure if no valid forecast data
        fig = go.Figure()
        fig.update_layout(**create_chart_layout(
            title="🔮 Price Forecasts",
            xaxis_title="Date",
            yaxis_title="Forecast Price ($)",
            height=400
        ))
        return fig
    
    # Use utility functions for preprocessing and validation
    forecast_df = preprocess_dataframe(forecast_df)
    forecast_df = sanitize_forecast_data(forecast_df)
    
    # Create animated forecast chart
    fig = go.Figure()
    
    # Determine overall trend for line color
    overall_trend = forecast_df['trend'].iloc[-1] if not forecast_df.empty else 'NEUTRAL'
    avg_confidence = forecast_df['confidence'].mean() if not forecast_df.empty else 0.5
    
    # Use utility function to get trend color
    line_color = get_trend_color(overall_trend, avg_confidence)
    
    # Use utility function to create forecast traces
    traces = create_forecast_traces(forecast_df, line_color)
    for trace in traces:
        fig.add_trace(trace)
    
    # Individual points are now included in the main line chart with markers
    
# Create animation frames using utility function
    def create_forecast_frame_traces(current_df, current_ind_df=None):
        if current_df.empty:
            return []
        
        # Calculate trend and confidence for current segment
        current_trend = current_df['trend'].iloc[-1] if not current_df.empty else 'NEUTRAL'
        current_avg_confidence = current_df['confidence'].mean() if not current_df.empty else 0.5
        
        # Use utility function to get trend color
        current_line_color = get_trend_color(current_trend, current_avg_confidence)
        
        # Use utility function to create forecast traces
        return create_forecast_traces(current_df, current_line_color)
    
    frames = create_animation_frames(
        forecast_df, 
        create_forecast_frame_traces
    )
    
    # Use utility functions for layout and animation controls
    layout_config = create_chart_layout(
        title='🔮 Price Forecasts',
        xaxis_title='Date',
        yaxis_title='Forecast Price ($)',
        height=500
    )
    
    animation_config = create_animation_controls(
        duration=800,
        transition_duration=400,
        button_prefix="Day: "
    )
    
    # Update slider steps with frames
    if frames:
        animation_config['sliders'][0]['steps'] = [
            dict(
                args=[[f.name], {"frame": {"duration": 400, "redraw": True},
                                "mode": "immediate", "transition": {"duration": 400}}],
                label=f"Day {int(f.name)+1}",
                method="animate"
            )
            for f in frames
        ]
    
    # Merge layout and animation configs
    layout_config.update(animation_config)
    fig.update_layout(**layout_config)
    
    if frames:
        fig.frames = frames
    
    return fig




